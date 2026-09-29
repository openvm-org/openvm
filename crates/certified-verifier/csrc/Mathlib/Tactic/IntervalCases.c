// Lean compiler output
// Module: Mathlib.Tactic.IntervalCases
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Attr public import Mathlib.Tactic.NormNum public meta import Mathlib.Tactic.Simps
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
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_mkDecideProofQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkArrow(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_repr(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_appendTag(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Subarray_get___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* l_Lean_mkNot(lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_exfalso(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_derive(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_Result_toRawIntEq(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Int_instInhabited;
lean_object* lp_Qq_Qq_instInhabitedQuoted(lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_Meta_substCore(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Meta_FVarSubst_get(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_binderIdent;
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Meta_FVarSubst_apply(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_getFVarIdsAt(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_generalizeHyp(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_lt_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_lt_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_le_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_le_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__3_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__5_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__6_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "IntervalCases"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "of_not_lt_left"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__3_value),LEAN_SCALAR_PTR_LITERAL(163, 255, 167, 41, 66, 177, 143, 33)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "of_not_lt_right"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__5_value),LEAN_SCALAR_PTR_LITERAL(127, 63, 248, 208, 88, 200, 72, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "of_le_right"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__7_value),LEAN_SCALAR_PTR_LITERAL(46, 88, 24, 21, 138, 93, 156, 220)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "of_le_left"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__9_value),LEAN_SCALAR_PTR_LITERAL(190, 87, 6, 44, 158, 75, 190, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "of_not_le_left"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__11_value),LEAN_SCALAR_PTR_LITERAL(170, 195, 20, 204, 43, 46, 245, 169)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "of_not_le_right"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__13_value),LEAN_SCALAR_PTR_LITERAL(196, 94, 186, 137, 98, 156, 46, 247)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "of_lt_right"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__15_value),LEAN_SCALAR_PTR_LITERAL(58, 247, 91, 72, 33, 173, 214, 171)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "of_lt_left"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__17_value),LEAN_SCALAR_PTR_LITERAL(233, 248, 25, 9, 56, 139, 194, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_trans"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 164, 114, 182, 61, 254, 17, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "le_of_not_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__2_value),LEAN_SCALAR_PTR_LITERAL(106, 23, 193, 173, 181, 13, 88, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__2_value),LEAN_SCALAR_PTR_LITERAL(125, 57, 202, 239, 254, 91, 104, 33)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "no goals"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "Mathlib.Tactic.IntervalCases.Methods.bisect"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Mathlib.Tactic.IntervalCases"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "le_antisymm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(60, 197, 251, 149, 196, 83, 222, 132)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "dite"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(137, 166, 197, 161, 68, 218, 116, 116)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ge_of_not_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "gt_of_not_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instLENat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(211, 47, 64, 46, 87, 101, 57, 105)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__1_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__2_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instOfNatNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__7_value),LEAN_SCALAR_PTR_LITERAL(217, 8, 172, 44, 179, 254, 147, 95)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "zero_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__5_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__4___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__5___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 0, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__8_value;
static const lean_string_object lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instOfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 253, 199, 38, 151, 242, 146)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instNegInt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "le_sub_one_of_not_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "add_one_le_of_not_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "instLEInt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__4___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__6___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 0, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "interval_cases failed: provided bound '"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "' cannot be evaluated"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "interval_cases failed: could not find bounds on "};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "interval_cases failed: could not find lower bound on "};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "interval_cases failed: could not find upper bound on "};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__5;
static const lean_array_object lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "interval_cases failed: unsupported type "};
static const lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "intervalCases"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 18, 124, 109, 77, 34, 39, 207)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "interval_cases"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__8_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__11_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__15_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__23_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__27;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__28;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__30_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__31_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__33_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__34_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_intervalCases___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_intervalCases___closed__36_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__37;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_intervalCases___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_intervalCases___closed__38;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_intervalCases;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Failed"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7_spec__9(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__6(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "expected a term of the form "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = " < _ or "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 10, .m_data = " ≤ _, got "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "expected a term of the form _ < "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = " or _ ≤ "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = ", got "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorIdx(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorIdx___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorIdx(v_x_4_);
lean_dec_ref(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___redArg(lean_object* v_t_6_, lean_object* v_k_7_){
_start:
{
lean_object* v_n_8_; lean_object* v___x_9_; 
v_n_8_ = lean_ctor_get(v_t_6_, 0);
lean_inc(v_n_8_);
lean_dec_ref(v_t_6_);
v___x_9_ = lean_apply_1(v_k_7_, v_n_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim(lean_object* v_motive_10_, lean_object* v_ctorIdx_11_, lean_object* v_t_12_, lean_object* v_h_13_, lean_object* v_k_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___redArg(v_t_12_, v_k_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_18_, v_h_19_, v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_lt_elim___redArg(lean_object* v_t_22_, lean_object* v_lt_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___redArg(v_t_22_, v_lt_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_lt_elim(lean_object* v_motive_25_, lean_object* v_t_26_, lean_object* v_h_27_, lean_object* v_lt_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___redArg(v_t_26_, v_lt_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_le_elim___redArg(lean_object* v_t_30_, lean_object* v_le_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___redArg(v_t_30_, v_le_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_le_elim(lean_object* v_motive_33_, lean_object* v_t_34_, lean_object* v_h_35_, lean_object* v_le_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_ctorElim___redArg(v_t_34_, v_le_36_);
return v___x_37_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = lean_unsigned_to_nat(1u);
v___x_39_ = lean_nat_to_int(v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower(lean_object* v_x_40_){
_start:
{
if (lean_obj_tag(v_x_40_) == 0)
{
lean_object* v_n_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v_n_41_ = lean_ctor_get(v_x_40_, 0);
v___x_42_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0);
v___x_43_ = lean_int_add(v_n_41_, v___x_42_);
return v___x_43_;
}
else
{
lean_object* v_n_44_; 
v_n_44_ = lean_ctor_get(v_x_40_, 0);
lean_inc(v_n_44_);
return v_n_44_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___boxed(lean_object* v_x_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower(v_x_45_);
lean_dec_ref(v_x_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper(lean_object* v_x_47_){
_start:
{
if (lean_obj_tag(v_x_47_) == 0)
{
lean_object* v_n_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v_n_48_ = lean_ctor_get(v_x_47_, 0);
v___x_49_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0);
v___x_50_ = lean_int_sub(v_n_48_, v___x_49_);
return v___x_50_;
}
else
{
lean_object* v_n_51_; 
v_n_51_ = lean_ctor_get(v_x_47_, 0);
lean_inc(v_n_51_);
return v_n_51_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper___boxed(lean_object* v_x_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper(v_x_52_);
lean_dec_ref(v_x_52_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0_spec__0(lean_object* v_msgData_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_){
_start:
{
lean_object* v___x_60_; lean_object* v_env_61_; lean_object* v___x_62_; lean_object* v_mctx_63_; lean_object* v_lctx_64_; lean_object* v_options_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_60_ = lean_st_ref_get(v___y_58_);
v_env_61_ = lean_ctor_get(v___x_60_, 0);
lean_inc_ref(v_env_61_);
lean_dec(v___x_60_);
v___x_62_ = lean_st_ref_get(v___y_56_);
v_mctx_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc_ref(v_mctx_63_);
lean_dec(v___x_62_);
v_lctx_64_ = lean_ctor_get(v___y_55_, 2);
v_options_65_ = lean_ctor_get(v___y_57_, 2);
lean_inc_ref(v_options_65_);
lean_inc_ref(v_lctx_64_);
v___x_66_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_66_, 0, v_env_61_);
lean_ctor_set(v___x_66_, 1, v_mctx_63_);
lean_ctor_set(v___x_66_, 2, v_lctx_64_);
lean_ctor_set(v___x_66_, 3, v_options_65_);
v___x_67_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v_msgData_54_);
v___x_68_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0_spec__0___boxed(lean_object* v_msgData_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0_spec__0(v_msgData_69_, v___y_70_, v___y_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(lean_object* v_msg_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v_ref_82_; lean_object* v___x_83_; lean_object* v_a_84_; lean_object* v___x_86_; uint8_t v_isShared_87_; uint8_t v_isSharedCheck_92_; 
v_ref_82_ = lean_ctor_get(v___y_79_, 5);
v___x_83_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0_spec__0(v_msg_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
v_a_84_ = lean_ctor_get(v___x_83_, 0);
v_isSharedCheck_92_ = !lean_is_exclusive(v___x_83_);
if (v_isSharedCheck_92_ == 0)
{
v___x_86_ = v___x_83_;
v_isShared_87_ = v_isSharedCheck_92_;
goto v_resetjp_85_;
}
else
{
lean_inc(v_a_84_);
lean_dec(v___x_83_);
v___x_86_ = lean_box(0);
v_isShared_87_ = v_isSharedCheck_92_;
goto v_resetjp_85_;
}
v_resetjp_85_:
{
lean_object* v___x_88_; lean_object* v___x_90_; 
lean_inc(v_ref_82_);
v___x_88_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_88_, 0, v_ref_82_);
lean_ctor_set(v___x_88_, 1, v_a_84_);
if (v_isShared_87_ == 0)
{
lean_ctor_set_tag(v___x_86_, 1);
lean_ctor_set(v___x_86_, 0, v___x_88_);
v___x_90_ = v___x_86_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_91_; 
v_reuseFailAlloc_91_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_91_, 0, v___x_88_);
v___x_90_ = v_reuseFailAlloc_91_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
return v___x_90_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg___boxed(lean_object* v_msg_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v_msg_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
return v_res_99_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_114_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__8));
v___x_115_ = l_Lean_stringToMessageData(v___x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(lean_object* v_ty_124_, lean_object* v_a_125_, lean_object* v_a_126_, lean_object* v_a_127_, lean_object* v_a_128_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = l_Lean_Meta_whnfR(v_ty_124_, v_a_125_, v_a_126_, v_a_127_, v_a_128_);
if (lean_obj_tag(v___x_130_) == 0)
{
lean_object* v_a_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_208_; 
v_a_131_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_208_ == 0)
{
v___x_133_ = v___x_130_;
v_isShared_134_ = v_isSharedCheck_208_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_a_131_);
lean_dec(v___x_130_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_208_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_135_; lean_object* v___x_136_; uint8_t v___x_137_; uint8_t v___x_138_; 
v___x_135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__1));
v___x_136_ = lean_unsigned_to_nat(1u);
v___x_137_ = l_Lean_Expr_isAppOfArity(v_a_131_, v___x_135_, v___x_136_);
v___x_138_ = 1;
if (v___x_137_ == 0)
{
lean_object* v___x_139_; lean_object* v___x_140_; uint8_t v___x_141_; 
v___x_139_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__4));
v___x_140_ = lean_unsigned_to_nat(4u);
v___x_141_ = l_Lean_Expr_isAppOfArity(v_a_131_, v___x_139_, v___x_140_);
if (v___x_141_ == 0)
{
lean_object* v___x_142_; uint8_t v___x_143_; 
v___x_142_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7));
v___x_143_ = l_Lean_Expr_isAppOfArity(v_a_131_, v___x_142_, v___x_140_);
if (v___x_143_ == 0)
{
lean_object* v___x_144_; lean_object* v___x_145_; 
lean_del_object(v___x_133_);
lean_dec(v_a_131_);
v___x_144_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9, &lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9);
v___x_145_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_144_, v_a_125_, v_a_126_, v_a_127_, v_a_128_);
return v___x_145_;
}
else
{
lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_155_; 
v___x_146_ = l_Lean_Expr_appFn_x21(v_a_131_);
v___x_147_ = l_Lean_Expr_appArg_x21(v___x_146_);
lean_dec_ref(v___x_146_);
v___x_148_ = l_Lean_Expr_appArg_x21(v_a_131_);
lean_dec(v_a_131_);
v___x_149_ = lean_box(v___x_141_);
v___x_150_ = lean_box(v___x_138_);
v___x_151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_149_);
lean_ctor_set(v___x_151_, 1, v___x_150_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_148_);
lean_ctor_set(v___x_152_, 1, v___x_151_);
v___x_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_147_);
lean_ctor_set(v___x_153_, 1, v___x_152_);
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 0, v___x_153_);
v___x_155_ = v___x_133_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v___x_153_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
}
else
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_164_; 
v___x_157_ = l_Lean_Expr_appFn_x21(v_a_131_);
v___x_158_ = l_Lean_Expr_appArg_x21(v___x_157_);
lean_dec_ref(v___x_157_);
v___x_159_ = l_Lean_Expr_appArg_x21(v_a_131_);
lean_dec(v_a_131_);
v___x_160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__10));
v___x_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_159_);
lean_ctor_set(v___x_161_, 1, v___x_160_);
v___x_162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_158_);
lean_ctor_set(v___x_162_, 1, v___x_161_);
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 0, v___x_162_);
v___x_164_ = v___x_133_;
goto v_reusejp_163_;
}
else
{
lean_object* v_reuseFailAlloc_165_; 
v_reuseFailAlloc_165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_165_, 0, v___x_162_);
v___x_164_ = v_reuseFailAlloc_165_;
goto v_reusejp_163_;
}
v_reusejp_163_:
{
return v___x_164_;
}
}
}
else
{
lean_object* v___x_166_; lean_object* v___x_167_; 
lean_del_object(v___x_133_);
v___x_166_ = l_Lean_Expr_appArg_x21(v_a_131_);
lean_dec(v_a_131_);
v___x_167_ = l_Lean_Meta_whnfR(v___x_166_, v_a_125_, v_a_126_, v_a_127_, v_a_128_);
if (lean_obj_tag(v___x_167_) == 0)
{
lean_object* v_a_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_199_; 
v_a_168_ = lean_ctor_get(v___x_167_, 0);
v_isSharedCheck_199_ = !lean_is_exclusive(v___x_167_);
if (v_isSharedCheck_199_ == 0)
{
v___x_170_ = v___x_167_;
v_isShared_171_ = v_isSharedCheck_199_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_a_168_);
lean_dec(v___x_167_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_199_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_172_; lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_172_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__4));
v___x_173_ = lean_unsigned_to_nat(4u);
v___x_174_ = l_Lean_Expr_isAppOfArity(v_a_168_, v___x_172_, v___x_173_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; uint8_t v___x_176_; 
v___x_175_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7));
v___x_176_ = l_Lean_Expr_isAppOfArity(v_a_168_, v___x_175_, v___x_173_);
if (v___x_176_ == 0)
{
lean_object* v___x_177_; lean_object* v___x_178_; 
lean_del_object(v___x_170_);
lean_dec(v_a_168_);
v___x_177_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9, &lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9);
v___x_178_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_177_, v_a_125_, v_a_126_, v_a_127_, v_a_128_);
return v___x_178_;
}
else
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_188_; 
v___x_179_ = l_Lean_Expr_appArg_x21(v_a_168_);
v___x_180_ = l_Lean_Expr_appFn_x21(v_a_168_);
lean_dec(v_a_168_);
v___x_181_ = l_Lean_Expr_appArg_x21(v___x_180_);
lean_dec_ref(v___x_180_);
v___x_182_ = lean_box(v___x_138_);
v___x_183_ = lean_box(v___x_174_);
v___x_184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_182_);
lean_ctor_set(v___x_184_, 1, v___x_183_);
v___x_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_181_);
lean_ctor_set(v___x_185_, 1, v___x_184_);
v___x_186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_179_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
if (v_isShared_171_ == 0)
{
lean_ctor_set(v___x_170_, 0, v___x_186_);
v___x_188_ = v___x_170_;
goto v_reusejp_187_;
}
else
{
lean_object* v_reuseFailAlloc_189_; 
v_reuseFailAlloc_189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_189_, 0, v___x_186_);
v___x_188_ = v_reuseFailAlloc_189_;
goto v_reusejp_187_;
}
v_reusejp_187_:
{
return v___x_188_;
}
}
}
else
{
lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_197_; 
v___x_190_ = l_Lean_Expr_appArg_x21(v_a_168_);
v___x_191_ = l_Lean_Expr_appFn_x21(v_a_168_);
lean_dec(v_a_168_);
v___x_192_ = l_Lean_Expr_appArg_x21(v___x_191_);
lean_dec_ref(v___x_191_);
v___x_193_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__11));
v___x_194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_192_);
lean_ctor_set(v___x_194_, 1, v___x_193_);
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_190_);
lean_ctor_set(v___x_195_, 1, v___x_194_);
if (v_isShared_171_ == 0)
{
lean_ctor_set(v___x_170_, 0, v___x_195_);
v___x_197_ = v___x_170_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_198_, 0, v___x_195_);
v___x_197_ = v_reuseFailAlloc_198_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
return v___x_197_;
}
}
}
}
else
{
lean_object* v_a_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_207_; 
v_a_200_ = lean_ctor_get(v___x_167_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_167_);
if (v_isSharedCheck_207_ == 0)
{
v___x_202_ = v___x_167_;
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_a_200_);
lean_dec(v___x_167_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_205_; 
if (v_isShared_203_ == 0)
{
v___x_205_ = v___x_202_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_a_200_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
}
}
else
{
lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
v_a_209_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_216_ == 0)
{
v___x_211_ = v___x_130_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_130_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_a_209_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___boxed(lean_object* v_ty_217_, lean_object* v_a_218_, lean_object* v_a_219_, lean_object* v_a_220_, lean_object* v_a_221_, lean_object* v_a_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v_ty_217_, v_a_218_, v_a_219_, v_a_220_, v_a_221_);
lean_dec(v_a_221_);
lean_dec_ref(v_a_220_);
lean_dec(v_a_219_);
lean_dec_ref(v_a_218_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0(lean_object* v_00_u03b1_224_, lean_object* v_msg_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v_msg_225_, v___y_226_, v___y_227_, v___y_228_, v___y_229_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___boxed(lean_object* v_00_u03b1_232_, lean_object* v_msg_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0(v_00_u03b1_232_, v_msg_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___redArg(lean_object* v_k_240_, uint8_t v_allowLevelAssignments_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_241_, v_k_240_, v___y_242_, v___y_243_, v___y_244_, v___y_245_);
if (lean_obj_tag(v___x_247_) == 0)
{
lean_object* v_a_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_255_; 
v_a_248_ = lean_ctor_get(v___x_247_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v___x_247_);
if (v_isSharedCheck_255_ == 0)
{
v___x_250_ = v___x_247_;
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_a_248_);
lean_dec(v___x_247_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_253_; 
if (v_isShared_251_ == 0)
{
v___x_253_ = v___x_250_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_a_248_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
else
{
lean_object* v_a_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_263_; 
v_a_256_ = lean_ctor_get(v___x_247_, 0);
v_isSharedCheck_263_ = !lean_is_exclusive(v___x_247_);
if (v_isSharedCheck_263_ == 0)
{
v___x_258_ = v___x_247_;
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_a_256_);
lean_dec(v___x_247_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___x_261_; 
if (v_isShared_259_ == 0)
{
v___x_261_ = v___x_258_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v_a_256_);
v___x_261_ = v_reuseFailAlloc_262_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
return v___x_261_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___redArg___boxed(lean_object* v_k_264_, lean_object* v_allowLevelAssignments_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_271_; lean_object* v_res_272_; 
v_allowLevelAssignments_boxed_271_ = lean_unbox(v_allowLevelAssignments_265_);
v_res_272_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___redArg(v_k_264_, v_allowLevelAssignments_boxed_271_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0(lean_object* v_00_u03b1_273_, lean_object* v_k_274_, uint8_t v_allowLevelAssignments_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___redArg(v_k_274_, v_allowLevelAssignments_275_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___boxed(lean_object* v_00_u03b1_282_, lean_object* v_k_283_, lean_object* v_allowLevelAssignments_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_290_; lean_object* v_res_291_; 
v_allowLevelAssignments_boxed_290_ = lean_unbox(v_allowLevelAssignments_284_);
v_res_291_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0(v_00_u03b1_282_, v_k_283_, v_allowLevelAssignments_boxed_290_, v___y_285_, v___y_286_, v___y_287_, v___y_288_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___lam__0(uint8_t v___x_292_, lean_object* v_e_293_, lean_object* v_fst_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
lean_object* v_keyedConfig_300_; uint8_t v_trackZetaDelta_301_; lean_object* v_zetaDeltaSet_302_; lean_object* v_lctx_303_; lean_object* v_localInstances_304_; lean_object* v_defEqCtx_x3f_305_; lean_object* v_synthPendingDepth_306_; lean_object* v_customCanUnfoldPredicate_x3f_307_; uint8_t v_univApprox_308_; uint8_t v_inTypeClassResolution_309_; uint8_t v_cacheInferType_310_; lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_319_; 
v_keyedConfig_300_ = lean_ctor_get(v___y_295_, 0);
v_trackZetaDelta_301_ = lean_ctor_get_uint8(v___y_295_, sizeof(void*)*7);
v_zetaDeltaSet_302_ = lean_ctor_get(v___y_295_, 1);
v_lctx_303_ = lean_ctor_get(v___y_295_, 2);
v_localInstances_304_ = lean_ctor_get(v___y_295_, 3);
v_defEqCtx_x3f_305_ = lean_ctor_get(v___y_295_, 4);
v_synthPendingDepth_306_ = lean_ctor_get(v___y_295_, 5);
v_customCanUnfoldPredicate_x3f_307_ = lean_ctor_get(v___y_295_, 6);
v_univApprox_308_ = lean_ctor_get_uint8(v___y_295_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_309_ = lean_ctor_get_uint8(v___y_295_, sizeof(void*)*7 + 2);
v_cacheInferType_310_ = lean_ctor_get_uint8(v___y_295_, sizeof(void*)*7 + 3);
v_isSharedCheck_319_ = !lean_is_exclusive(v___y_295_);
if (v_isSharedCheck_319_ == 0)
{
v___x_312_ = v___y_295_;
v_isShared_313_ = v_isSharedCheck_319_;
goto v_resetjp_311_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_307_);
lean_inc(v_synthPendingDepth_306_);
lean_inc(v_defEqCtx_x3f_305_);
lean_inc(v_localInstances_304_);
lean_inc(v_lctx_303_);
lean_inc(v_zetaDeltaSet_302_);
lean_inc(v_keyedConfig_300_);
lean_dec(v___y_295_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_319_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v___x_314_; lean_object* v___x_316_; 
v___x_314_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_292_, v_keyedConfig_300_);
if (v_isShared_313_ == 0)
{
lean_ctor_set(v___x_312_, 0, v___x_314_);
v___x_316_ = v___x_312_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_318_, 0, v___x_314_);
lean_ctor_set(v_reuseFailAlloc_318_, 1, v_zetaDeltaSet_302_);
lean_ctor_set(v_reuseFailAlloc_318_, 2, v_lctx_303_);
lean_ctor_set(v_reuseFailAlloc_318_, 3, v_localInstances_304_);
lean_ctor_set(v_reuseFailAlloc_318_, 4, v_defEqCtx_x3f_305_);
lean_ctor_set(v_reuseFailAlloc_318_, 5, v_synthPendingDepth_306_);
lean_ctor_set(v_reuseFailAlloc_318_, 6, v_customCanUnfoldPredicate_x3f_307_);
lean_ctor_set_uint8(v_reuseFailAlloc_318_, sizeof(void*)*7, v_trackZetaDelta_301_);
lean_ctor_set_uint8(v_reuseFailAlloc_318_, sizeof(void*)*7 + 1, v_univApprox_308_);
lean_ctor_set_uint8(v_reuseFailAlloc_318_, sizeof(void*)*7 + 2, v_inTypeClassResolution_309_);
lean_ctor_set_uint8(v_reuseFailAlloc_318_, sizeof(void*)*7 + 3, v_cacheInferType_310_);
v___x_316_ = v_reuseFailAlloc_318_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
lean_object* v___x_317_; 
v___x_317_ = l_Lean_Meta_isExprDefEq(v_e_293_, v_fst_294_, v___x_316_, v___y_296_, v___y_297_, v___y_298_);
lean_dec_ref(v___x_316_);
return v___x_317_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___lam__0___boxed(lean_object* v___x_320_, lean_object* v_e_321_, lean_object* v_fst_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
uint8_t v___x_5448__boxed_328_; lean_object* v_res_329_; 
v___x_5448__boxed_328_ = lean_unbox(v___x_320_);
v_res_329_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___lam__0(v___x_5448__boxed_328_, v_e_321_, v_fst_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound(lean_object* v_m_381_, lean_object* v_e_382_, lean_object* v_pf_383_, uint8_t v_lb_384_, lean_object* v_a_385_, lean_object* v_a_386_, lean_object* v_a_387_, lean_object* v_a_388_){
_start:
{
lean_object* v_fst_391_; lean_object* v_snd_392_; lean_object* v___y_393_; lean_object* v___y_394_; lean_object* v___y_395_; lean_object* v___y_396_; lean_object* v___x_421_; 
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
lean_inc_ref(v_pf_383_);
v___x_421_ = lean_infer_type(v_pf_383_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_421_) == 0)
{
lean_object* v_a_422_; lean_object* v___x_423_; 
v_a_422_ = lean_ctor_get(v___x_421_, 0);
lean_inc(v_a_422_);
lean_dec_ref_known(v___x_421_, 1);
v___x_423_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v_a_422_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_423_) == 0)
{
lean_object* v_a_424_; lean_object* v_snd_425_; lean_object* v_snd_426_; lean_object* v_fst_427_; uint8_t v___x_428_; 
v_a_424_ = lean_ctor_get(v___x_423_, 0);
lean_inc(v_a_424_);
lean_dec_ref_known(v___x_423_, 1);
v_snd_425_ = lean_ctor_get(v_a_424_, 1);
lean_inc(v_snd_425_);
v_snd_426_ = lean_ctor_get(v_snd_425_, 1);
v_fst_427_ = lean_ctor_get(v_snd_426_, 0);
v___x_428_ = lean_unbox(v_fst_427_);
if (v___x_428_ == 0)
{
lean_object* v_snd_429_; uint8_t v___x_430_; 
v_snd_429_ = lean_ctor_get(v_snd_426_, 1);
v___x_430_ = lean_unbox(v_snd_429_);
if (v___x_430_ == 0)
{
if (v_lb_384_ == 0)
{
lean_object* v_fst_431_; lean_object* v_fst_432_; lean_object* v_eval_433_; lean_object* v___x_434_; 
v_fst_431_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_431_);
lean_dec(v_a_424_);
v_fst_432_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_432_);
lean_dec(v_snd_425_);
v_eval_433_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_433_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_434_ = lean_apply_6(v_eval_433_, v_fst_432_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_434_) == 0)
{
lean_object* v_a_435_; lean_object* v_snd_436_; lean_object* v_fst_437_; lean_object* v___x_439_; uint8_t v_isShared_440_; uint8_t v_isSharedCheck_469_; 
v_a_435_ = lean_ctor_get(v___x_434_, 0);
lean_inc(v_a_435_);
lean_dec_ref_known(v___x_434_, 1);
v_snd_436_ = lean_ctor_get(v_a_435_, 1);
v_fst_437_ = lean_ctor_get(v_a_435_, 0);
v_isSharedCheck_469_ = !lean_is_exclusive(v_a_435_);
if (v_isSharedCheck_469_ == 0)
{
v___x_439_ = v_a_435_;
v_isShared_440_ = v_isSharedCheck_469_;
goto v_resetjp_438_;
}
else
{
lean_inc(v_snd_436_);
lean_inc(v_fst_437_);
lean_dec(v_a_435_);
v___x_439_ = lean_box(0);
v_isShared_440_ = v_isSharedCheck_469_;
goto v_resetjp_438_;
}
v_resetjp_438_:
{
lean_object* v_fst_441_; lean_object* v_snd_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_468_; 
v_fst_441_ = lean_ctor_get(v_snd_436_, 0);
v_snd_442_ = lean_ctor_get(v_snd_436_, 1);
v_isSharedCheck_468_ = !lean_is_exclusive(v_snd_436_);
if (v_isSharedCheck_468_ == 0)
{
v___x_444_ = v_snd_436_;
v_isShared_445_ = v_isSharedCheck_468_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_snd_442_);
lean_inc(v_fst_441_);
lean_dec(v_snd_436_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_468_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_446_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__4));
v___x_447_ = lean_unsigned_to_nat(2u);
v___x_448_ = lean_mk_empty_array_with_capacity(v___x_447_);
v___x_449_ = lean_array_push(v___x_448_, v_pf_383_);
v___x_450_ = lean_array_push(v___x_449_, v_snd_442_);
v___x_451_ = l_Lean_Meta_mkAppM(v___x_446_, v___x_450_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_object* v_a_452_; lean_object* v___x_453_; lean_object* v___x_455_; 
v_a_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_a_452_);
lean_dec_ref_known(v___x_451_, 1);
v___x_453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_453_, 0, v_fst_437_);
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v_a_452_);
v___x_455_ = v___x_444_;
goto v_reusejp_454_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_fst_441_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v_a_452_);
v___x_455_ = v_reuseFailAlloc_459_;
goto v_reusejp_454_;
}
v_reusejp_454_:
{
lean_object* v___x_457_; 
if (v_isShared_440_ == 0)
{
lean_ctor_set(v___x_439_, 1, v___x_455_);
lean_ctor_set(v___x_439_, 0, v___x_453_);
v___x_457_ = v___x_439_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_453_);
lean_ctor_set(v_reuseFailAlloc_458_, 1, v___x_455_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
v_fst_391_ = v_fst_431_;
v_snd_392_ = v___x_457_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_460_; lean_object* v___x_462_; uint8_t v_isShared_463_; uint8_t v_isSharedCheck_467_; 
lean_del_object(v___x_444_);
lean_dec(v_fst_441_);
lean_del_object(v___x_439_);
lean_dec(v_fst_437_);
lean_dec(v_fst_431_);
lean_dec_ref(v_e_382_);
v_a_460_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_467_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_467_ == 0)
{
v___x_462_ = v___x_451_;
v_isShared_463_ = v_isSharedCheck_467_;
goto v_resetjp_461_;
}
else
{
lean_inc(v_a_460_);
lean_dec(v___x_451_);
v___x_462_ = lean_box(0);
v_isShared_463_ = v_isSharedCheck_467_;
goto v_resetjp_461_;
}
v_resetjp_461_:
{
lean_object* v___x_465_; 
if (v_isShared_463_ == 0)
{
v___x_465_ = v___x_462_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_466_; 
v_reuseFailAlloc_466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_466_, 0, v_a_460_);
v___x_465_ = v_reuseFailAlloc_466_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
return v___x_465_;
}
}
}
}
}
}
else
{
lean_object* v_a_470_; lean_object* v___x_472_; uint8_t v_isShared_473_; uint8_t v_isSharedCheck_477_; 
lean_dec(v_fst_431_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_470_ = lean_ctor_get(v___x_434_, 0);
v_isSharedCheck_477_ = !lean_is_exclusive(v___x_434_);
if (v_isSharedCheck_477_ == 0)
{
v___x_472_ = v___x_434_;
v_isShared_473_ = v_isSharedCheck_477_;
goto v_resetjp_471_;
}
else
{
lean_inc(v_a_470_);
lean_dec(v___x_434_);
v___x_472_ = lean_box(0);
v_isShared_473_ = v_isSharedCheck_477_;
goto v_resetjp_471_;
}
v_resetjp_471_:
{
lean_object* v___x_475_; 
if (v_isShared_473_ == 0)
{
v___x_475_ = v___x_472_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_476_; 
v_reuseFailAlloc_476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_476_, 0, v_a_470_);
v___x_475_ = v_reuseFailAlloc_476_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
return v___x_475_;
}
}
}
}
else
{
lean_object* v_fst_478_; lean_object* v_fst_479_; lean_object* v_eval_480_; lean_object* v___x_481_; 
v_fst_478_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_478_);
lean_dec(v_a_424_);
v_fst_479_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_479_);
lean_dec(v_snd_425_);
v_eval_480_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_480_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_481_ = lean_apply_6(v_eval_480_, v_fst_478_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_481_) == 0)
{
lean_object* v_a_482_; lean_object* v_snd_483_; lean_object* v_fst_484_; lean_object* v___x_486_; uint8_t v_isShared_487_; uint8_t v_isSharedCheck_516_; 
v_a_482_ = lean_ctor_get(v___x_481_, 0);
lean_inc(v_a_482_);
lean_dec_ref_known(v___x_481_, 1);
v_snd_483_ = lean_ctor_get(v_a_482_, 1);
v_fst_484_ = lean_ctor_get(v_a_482_, 0);
v_isSharedCheck_516_ = !lean_is_exclusive(v_a_482_);
if (v_isSharedCheck_516_ == 0)
{
v___x_486_ = v_a_482_;
v_isShared_487_ = v_isSharedCheck_516_;
goto v_resetjp_485_;
}
else
{
lean_inc(v_snd_483_);
lean_inc(v_fst_484_);
lean_dec(v_a_482_);
v___x_486_ = lean_box(0);
v_isShared_487_ = v_isSharedCheck_516_;
goto v_resetjp_485_;
}
v_resetjp_485_:
{
lean_object* v_fst_488_; lean_object* v_snd_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_515_; 
v_fst_488_ = lean_ctor_get(v_snd_483_, 0);
v_snd_489_ = lean_ctor_get(v_snd_483_, 1);
v_isSharedCheck_515_ = !lean_is_exclusive(v_snd_483_);
if (v_isSharedCheck_515_ == 0)
{
v___x_491_ = v_snd_483_;
v_isShared_492_ = v_isSharedCheck_515_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_snd_489_);
lean_inc(v_fst_488_);
lean_dec(v_snd_483_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_515_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; 
v___x_493_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__6));
v___x_494_ = lean_unsigned_to_nat(2u);
v___x_495_ = lean_mk_empty_array_with_capacity(v___x_494_);
v___x_496_ = lean_array_push(v___x_495_, v_pf_383_);
v___x_497_ = lean_array_push(v___x_496_, v_snd_489_);
v___x_498_ = l_Lean_Meta_mkAppM(v___x_493_, v___x_497_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_499_; lean_object* v___x_500_; lean_object* v___x_502_; 
v_a_499_ = lean_ctor_get(v___x_498_, 0);
lean_inc(v_a_499_);
lean_dec_ref_known(v___x_498_, 1);
v___x_500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_500_, 0, v_fst_484_);
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 1, v_a_499_);
v___x_502_ = v___x_491_;
goto v_reusejp_501_;
}
else
{
lean_object* v_reuseFailAlloc_506_; 
v_reuseFailAlloc_506_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_506_, 0, v_fst_488_);
lean_ctor_set(v_reuseFailAlloc_506_, 1, v_a_499_);
v___x_502_ = v_reuseFailAlloc_506_;
goto v_reusejp_501_;
}
v_reusejp_501_:
{
lean_object* v___x_504_; 
if (v_isShared_487_ == 0)
{
lean_ctor_set(v___x_486_, 1, v___x_502_);
lean_ctor_set(v___x_486_, 0, v___x_500_);
v___x_504_ = v___x_486_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v___x_500_);
lean_ctor_set(v_reuseFailAlloc_505_, 1, v___x_502_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
v_fst_391_ = v_fst_479_;
v_snd_392_ = v___x_504_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
lean_del_object(v___x_491_);
lean_dec(v_fst_488_);
lean_del_object(v___x_486_);
lean_dec(v_fst_484_);
lean_dec(v_fst_479_);
lean_dec_ref(v_e_382_);
v_a_507_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_498_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_498_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_512_; 
if (v_isShared_510_ == 0)
{
v___x_512_ = v___x_509_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_a_507_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
}
}
}
else
{
lean_object* v_a_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_524_; 
lean_dec(v_fst_479_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_517_ = lean_ctor_get(v___x_481_, 0);
v_isSharedCheck_524_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_524_ == 0)
{
v___x_519_ = v___x_481_;
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_a_517_);
lean_dec(v___x_481_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_524_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_522_; 
if (v_isShared_520_ == 0)
{
v___x_522_ = v___x_519_;
goto v_reusejp_521_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v_a_517_);
v___x_522_ = v_reuseFailAlloc_523_;
goto v_reusejp_521_;
}
v_reusejp_521_:
{
return v___x_522_;
}
}
}
}
}
else
{
if (v_lb_384_ == 0)
{
lean_object* v_fst_525_; lean_object* v_fst_526_; lean_object* v_eval_527_; lean_object* v___x_528_; 
v_fst_525_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_525_);
lean_dec(v_a_424_);
v_fst_526_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_526_);
lean_dec(v_snd_425_);
v_eval_527_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_527_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_528_ = lean_apply_6(v_eval_527_, v_fst_526_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_528_) == 0)
{
lean_object* v_a_529_; lean_object* v_snd_530_; lean_object* v_fst_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_563_; 
v_a_529_ = lean_ctor_get(v___x_528_, 0);
lean_inc(v_a_529_);
lean_dec_ref_known(v___x_528_, 1);
v_snd_530_ = lean_ctor_get(v_a_529_, 1);
v_fst_531_ = lean_ctor_get(v_a_529_, 0);
v_isSharedCheck_563_ = !lean_is_exclusive(v_a_529_);
if (v_isSharedCheck_563_ == 0)
{
v___x_533_ = v_a_529_;
v_isShared_534_ = v_isSharedCheck_563_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_snd_530_);
lean_inc(v_fst_531_);
lean_dec(v_a_529_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_563_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v_fst_535_; lean_object* v_snd_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_562_; 
v_fst_535_ = lean_ctor_get(v_snd_530_, 0);
v_snd_536_ = lean_ctor_get(v_snd_530_, 1);
v_isSharedCheck_562_ = !lean_is_exclusive(v_snd_530_);
if (v_isSharedCheck_562_ == 0)
{
v___x_538_ = v_snd_530_;
v_isShared_539_ = v_isSharedCheck_562_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_snd_536_);
lean_inc(v_fst_535_);
lean_dec(v_snd_530_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_562_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v___x_540_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__8));
v___x_541_ = lean_unsigned_to_nat(2u);
v___x_542_ = lean_mk_empty_array_with_capacity(v___x_541_);
v___x_543_ = lean_array_push(v___x_542_, v_pf_383_);
v___x_544_ = lean_array_push(v___x_543_, v_snd_536_);
v___x_545_ = l_Lean_Meta_mkAppM(v___x_540_, v___x_544_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_545_) == 0)
{
lean_object* v_a_546_; lean_object* v___x_547_; lean_object* v___x_549_; 
v_a_546_ = lean_ctor_get(v___x_545_, 0);
lean_inc(v_a_546_);
lean_dec_ref_known(v___x_545_, 1);
v___x_547_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_547_, 0, v_fst_531_);
if (v_isShared_539_ == 0)
{
lean_ctor_set(v___x_538_, 1, v_a_546_);
v___x_549_ = v___x_538_;
goto v_reusejp_548_;
}
else
{
lean_object* v_reuseFailAlloc_553_; 
v_reuseFailAlloc_553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_553_, 0, v_fst_535_);
lean_ctor_set(v_reuseFailAlloc_553_, 1, v_a_546_);
v___x_549_ = v_reuseFailAlloc_553_;
goto v_reusejp_548_;
}
v_reusejp_548_:
{
lean_object* v___x_551_; 
if (v_isShared_534_ == 0)
{
lean_ctor_set(v___x_533_, 1, v___x_549_);
lean_ctor_set(v___x_533_, 0, v___x_547_);
v___x_551_ = v___x_533_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v___x_547_);
lean_ctor_set(v_reuseFailAlloc_552_, 1, v___x_549_);
v___x_551_ = v_reuseFailAlloc_552_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
v_fst_391_ = v_fst_525_;
v_snd_392_ = v___x_551_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_561_; 
lean_del_object(v___x_538_);
lean_dec(v_fst_535_);
lean_del_object(v___x_533_);
lean_dec(v_fst_531_);
lean_dec(v_fst_525_);
lean_dec_ref(v_e_382_);
v_a_554_ = lean_ctor_get(v___x_545_, 0);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_545_);
if (v_isSharedCheck_561_ == 0)
{
v___x_556_ = v___x_545_;
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_545_);
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
}
}
}
else
{
lean_object* v_a_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_571_; 
lean_dec(v_fst_525_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_564_ = lean_ctor_get(v___x_528_, 0);
v_isSharedCheck_571_ = !lean_is_exclusive(v___x_528_);
if (v_isSharedCheck_571_ == 0)
{
v___x_566_ = v___x_528_;
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_a_564_);
lean_dec(v___x_528_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_571_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
lean_object* v___x_569_; 
if (v_isShared_567_ == 0)
{
v___x_569_ = v___x_566_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_a_564_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
}
else
{
lean_object* v_fst_572_; lean_object* v_fst_573_; lean_object* v_eval_574_; lean_object* v___x_575_; 
v_fst_572_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_572_);
lean_dec(v_a_424_);
v_fst_573_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_573_);
lean_dec(v_snd_425_);
v_eval_574_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_574_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_575_ = lean_apply_6(v_eval_574_, v_fst_572_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_575_) == 0)
{
lean_object* v_a_576_; lean_object* v_snd_577_; lean_object* v_fst_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_610_; 
v_a_576_ = lean_ctor_get(v___x_575_, 0);
lean_inc(v_a_576_);
lean_dec_ref_known(v___x_575_, 1);
v_snd_577_ = lean_ctor_get(v_a_576_, 1);
v_fst_578_ = lean_ctor_get(v_a_576_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v_a_576_);
if (v_isSharedCheck_610_ == 0)
{
v___x_580_ = v_a_576_;
v_isShared_581_ = v_isSharedCheck_610_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_snd_577_);
lean_inc(v_fst_578_);
lean_dec(v_a_576_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_610_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
lean_object* v_fst_582_; lean_object* v_snd_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_609_; 
v_fst_582_ = lean_ctor_get(v_snd_577_, 0);
v_snd_583_ = lean_ctor_get(v_snd_577_, 1);
v_isSharedCheck_609_ = !lean_is_exclusive(v_snd_577_);
if (v_isSharedCheck_609_ == 0)
{
v___x_585_ = v_snd_577_;
v_isShared_586_ = v_isSharedCheck_609_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_snd_583_);
lean_inc(v_fst_582_);
lean_dec(v_snd_577_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_609_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; 
v___x_587_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__10));
v___x_588_ = lean_unsigned_to_nat(2u);
v___x_589_ = lean_mk_empty_array_with_capacity(v___x_588_);
v___x_590_ = lean_array_push(v___x_589_, v_pf_383_);
v___x_591_ = lean_array_push(v___x_590_, v_snd_583_);
v___x_592_ = l_Lean_Meta_mkAppM(v___x_587_, v___x_591_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_592_) == 0)
{
lean_object* v_a_593_; lean_object* v___x_594_; lean_object* v___x_596_; 
v_a_593_ = lean_ctor_get(v___x_592_, 0);
lean_inc(v_a_593_);
lean_dec_ref_known(v___x_592_, 1);
v___x_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_594_, 0, v_fst_578_);
if (v_isShared_586_ == 0)
{
lean_ctor_set(v___x_585_, 1, v_a_593_);
v___x_596_ = v___x_585_;
goto v_reusejp_595_;
}
else
{
lean_object* v_reuseFailAlloc_600_; 
v_reuseFailAlloc_600_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_600_, 0, v_fst_582_);
lean_ctor_set(v_reuseFailAlloc_600_, 1, v_a_593_);
v___x_596_ = v_reuseFailAlloc_600_;
goto v_reusejp_595_;
}
v_reusejp_595_:
{
lean_object* v___x_598_; 
if (v_isShared_581_ == 0)
{
lean_ctor_set(v___x_580_, 1, v___x_596_);
lean_ctor_set(v___x_580_, 0, v___x_594_);
v___x_598_ = v___x_580_;
goto v_reusejp_597_;
}
else
{
lean_object* v_reuseFailAlloc_599_; 
v_reuseFailAlloc_599_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_599_, 0, v___x_594_);
lean_ctor_set(v_reuseFailAlloc_599_, 1, v___x_596_);
v___x_598_ = v_reuseFailAlloc_599_;
goto v_reusejp_597_;
}
v_reusejp_597_:
{
v_fst_391_ = v_fst_573_;
v_snd_392_ = v___x_598_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_601_; lean_object* v___x_603_; uint8_t v_isShared_604_; uint8_t v_isSharedCheck_608_; 
lean_del_object(v___x_585_);
lean_dec(v_fst_582_);
lean_del_object(v___x_580_);
lean_dec(v_fst_578_);
lean_dec(v_fst_573_);
lean_dec_ref(v_e_382_);
v_a_601_ = lean_ctor_get(v___x_592_, 0);
v_isSharedCheck_608_ = !lean_is_exclusive(v___x_592_);
if (v_isSharedCheck_608_ == 0)
{
v___x_603_ = v___x_592_;
v_isShared_604_ = v_isSharedCheck_608_;
goto v_resetjp_602_;
}
else
{
lean_inc(v_a_601_);
lean_dec(v___x_592_);
v___x_603_ = lean_box(0);
v_isShared_604_ = v_isSharedCheck_608_;
goto v_resetjp_602_;
}
v_resetjp_602_:
{
lean_object* v___x_606_; 
if (v_isShared_604_ == 0)
{
v___x_606_ = v___x_603_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v_a_601_);
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
lean_object* v_a_611_; lean_object* v___x_613_; uint8_t v_isShared_614_; uint8_t v_isSharedCheck_618_; 
lean_dec(v_fst_573_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_611_ = lean_ctor_get(v___x_575_, 0);
v_isSharedCheck_618_ = !lean_is_exclusive(v___x_575_);
if (v_isSharedCheck_618_ == 0)
{
v___x_613_ = v___x_575_;
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
else
{
lean_inc(v_a_611_);
lean_dec(v___x_575_);
v___x_613_ = lean_box(0);
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
v_resetjp_612_:
{
lean_object* v___x_616_; 
if (v_isShared_614_ == 0)
{
v___x_616_ = v___x_613_;
goto v_reusejp_615_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v_a_611_);
v___x_616_ = v_reuseFailAlloc_617_;
goto v_reusejp_615_;
}
v_reusejp_615_:
{
return v___x_616_;
}
}
}
}
}
}
else
{
lean_object* v_snd_619_; uint8_t v___x_620_; 
v_snd_619_ = lean_ctor_get(v_snd_426_, 1);
v___x_620_ = lean_unbox(v_snd_619_);
if (v___x_620_ == 0)
{
if (v_lb_384_ == 0)
{
lean_object* v_fst_621_; lean_object* v_fst_622_; lean_object* v_eval_623_; lean_object* v___x_624_; 
v_fst_621_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_621_);
lean_dec(v_a_424_);
v_fst_622_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_622_);
lean_dec(v_snd_425_);
v_eval_623_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_623_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_624_ = lean_apply_6(v_eval_623_, v_fst_622_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_624_) == 0)
{
lean_object* v_a_625_; lean_object* v_snd_626_; lean_object* v_fst_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_659_; 
v_a_625_ = lean_ctor_get(v___x_624_, 0);
lean_inc(v_a_625_);
lean_dec_ref_known(v___x_624_, 1);
v_snd_626_ = lean_ctor_get(v_a_625_, 1);
v_fst_627_ = lean_ctor_get(v_a_625_, 0);
v_isSharedCheck_659_ = !lean_is_exclusive(v_a_625_);
if (v_isSharedCheck_659_ == 0)
{
v___x_629_ = v_a_625_;
v_isShared_630_ = v_isSharedCheck_659_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_snd_626_);
lean_inc(v_fst_627_);
lean_dec(v_a_625_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_659_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v_fst_631_; lean_object* v_snd_632_; lean_object* v___x_634_; uint8_t v_isShared_635_; uint8_t v_isSharedCheck_658_; 
v_fst_631_ = lean_ctor_get(v_snd_626_, 0);
v_snd_632_ = lean_ctor_get(v_snd_626_, 1);
v_isSharedCheck_658_ = !lean_is_exclusive(v_snd_626_);
if (v_isSharedCheck_658_ == 0)
{
v___x_634_ = v_snd_626_;
v_isShared_635_ = v_isSharedCheck_658_;
goto v_resetjp_633_;
}
else
{
lean_inc(v_snd_632_);
lean_inc(v_fst_631_);
lean_dec(v_snd_626_);
v___x_634_ = lean_box(0);
v_isShared_635_ = v_isSharedCheck_658_;
goto v_resetjp_633_;
}
v_resetjp_633_:
{
lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_636_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__12));
v___x_637_ = lean_unsigned_to_nat(2u);
v___x_638_ = lean_mk_empty_array_with_capacity(v___x_637_);
v___x_639_ = lean_array_push(v___x_638_, v_pf_383_);
v___x_640_ = lean_array_push(v___x_639_, v_snd_632_);
v___x_641_ = l_Lean_Meta_mkAppM(v___x_636_, v___x_640_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_641_) == 0)
{
lean_object* v_a_642_; lean_object* v___x_643_; lean_object* v___x_645_; 
v_a_642_ = lean_ctor_get(v___x_641_, 0);
lean_inc(v_a_642_);
lean_dec_ref_known(v___x_641_, 1);
v___x_643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_643_, 0, v_fst_627_);
if (v_isShared_635_ == 0)
{
lean_ctor_set(v___x_634_, 1, v_a_642_);
v___x_645_ = v___x_634_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_649_; 
v_reuseFailAlloc_649_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_649_, 0, v_fst_631_);
lean_ctor_set(v_reuseFailAlloc_649_, 1, v_a_642_);
v___x_645_ = v_reuseFailAlloc_649_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
lean_object* v___x_647_; 
if (v_isShared_630_ == 0)
{
lean_ctor_set(v___x_629_, 1, v___x_645_);
lean_ctor_set(v___x_629_, 0, v___x_643_);
v___x_647_ = v___x_629_;
goto v_reusejp_646_;
}
else
{
lean_object* v_reuseFailAlloc_648_; 
v_reuseFailAlloc_648_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_648_, 0, v___x_643_);
lean_ctor_set(v_reuseFailAlloc_648_, 1, v___x_645_);
v___x_647_ = v_reuseFailAlloc_648_;
goto v_reusejp_646_;
}
v_reusejp_646_:
{
v_fst_391_ = v_fst_621_;
v_snd_392_ = v___x_647_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_650_; lean_object* v___x_652_; uint8_t v_isShared_653_; uint8_t v_isSharedCheck_657_; 
lean_del_object(v___x_634_);
lean_dec(v_fst_631_);
lean_del_object(v___x_629_);
lean_dec(v_fst_627_);
lean_dec(v_fst_621_);
lean_dec_ref(v_e_382_);
v_a_650_ = lean_ctor_get(v___x_641_, 0);
v_isSharedCheck_657_ = !lean_is_exclusive(v___x_641_);
if (v_isSharedCheck_657_ == 0)
{
v___x_652_ = v___x_641_;
v_isShared_653_ = v_isSharedCheck_657_;
goto v_resetjp_651_;
}
else
{
lean_inc(v_a_650_);
lean_dec(v___x_641_);
v___x_652_ = lean_box(0);
v_isShared_653_ = v_isSharedCheck_657_;
goto v_resetjp_651_;
}
v_resetjp_651_:
{
lean_object* v___x_655_; 
if (v_isShared_653_ == 0)
{
v___x_655_ = v___x_652_;
goto v_reusejp_654_;
}
else
{
lean_object* v_reuseFailAlloc_656_; 
v_reuseFailAlloc_656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_656_, 0, v_a_650_);
v___x_655_ = v_reuseFailAlloc_656_;
goto v_reusejp_654_;
}
v_reusejp_654_:
{
return v___x_655_;
}
}
}
}
}
}
else
{
lean_object* v_a_660_; lean_object* v___x_662_; uint8_t v_isShared_663_; uint8_t v_isSharedCheck_667_; 
lean_dec(v_fst_621_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_660_ = lean_ctor_get(v___x_624_, 0);
v_isSharedCheck_667_ = !lean_is_exclusive(v___x_624_);
if (v_isSharedCheck_667_ == 0)
{
v___x_662_ = v___x_624_;
v_isShared_663_ = v_isSharedCheck_667_;
goto v_resetjp_661_;
}
else
{
lean_inc(v_a_660_);
lean_dec(v___x_624_);
v___x_662_ = lean_box(0);
v_isShared_663_ = v_isSharedCheck_667_;
goto v_resetjp_661_;
}
v_resetjp_661_:
{
lean_object* v___x_665_; 
if (v_isShared_663_ == 0)
{
v___x_665_ = v___x_662_;
goto v_reusejp_664_;
}
else
{
lean_object* v_reuseFailAlloc_666_; 
v_reuseFailAlloc_666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_666_, 0, v_a_660_);
v___x_665_ = v_reuseFailAlloc_666_;
goto v_reusejp_664_;
}
v_reusejp_664_:
{
return v___x_665_;
}
}
}
}
else
{
lean_object* v_fst_668_; lean_object* v_fst_669_; lean_object* v_eval_670_; lean_object* v___x_671_; 
v_fst_668_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_668_);
lean_dec(v_a_424_);
v_fst_669_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_669_);
lean_dec(v_snd_425_);
v_eval_670_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_670_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_671_ = lean_apply_6(v_eval_670_, v_fst_668_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_671_) == 0)
{
lean_object* v_a_672_; lean_object* v_snd_673_; lean_object* v_fst_674_; lean_object* v___x_676_; uint8_t v_isShared_677_; uint8_t v_isSharedCheck_706_; 
v_a_672_ = lean_ctor_get(v___x_671_, 0);
lean_inc(v_a_672_);
lean_dec_ref_known(v___x_671_, 1);
v_snd_673_ = lean_ctor_get(v_a_672_, 1);
v_fst_674_ = lean_ctor_get(v_a_672_, 0);
v_isSharedCheck_706_ = !lean_is_exclusive(v_a_672_);
if (v_isSharedCheck_706_ == 0)
{
v___x_676_ = v_a_672_;
v_isShared_677_ = v_isSharedCheck_706_;
goto v_resetjp_675_;
}
else
{
lean_inc(v_snd_673_);
lean_inc(v_fst_674_);
lean_dec(v_a_672_);
v___x_676_ = lean_box(0);
v_isShared_677_ = v_isSharedCheck_706_;
goto v_resetjp_675_;
}
v_resetjp_675_:
{
lean_object* v_fst_678_; lean_object* v_snd_679_; lean_object* v___x_681_; uint8_t v_isShared_682_; uint8_t v_isSharedCheck_705_; 
v_fst_678_ = lean_ctor_get(v_snd_673_, 0);
v_snd_679_ = lean_ctor_get(v_snd_673_, 1);
v_isSharedCheck_705_ = !lean_is_exclusive(v_snd_673_);
if (v_isSharedCheck_705_ == 0)
{
v___x_681_ = v_snd_673_;
v_isShared_682_ = v_isSharedCheck_705_;
goto v_resetjp_680_;
}
else
{
lean_inc(v_snd_679_);
lean_inc(v_fst_678_);
lean_dec(v_snd_673_);
v___x_681_ = lean_box(0);
v_isShared_682_ = v_isSharedCheck_705_;
goto v_resetjp_680_;
}
v_resetjp_680_:
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; 
v___x_683_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__14));
v___x_684_ = lean_unsigned_to_nat(2u);
v___x_685_ = lean_mk_empty_array_with_capacity(v___x_684_);
v___x_686_ = lean_array_push(v___x_685_, v_pf_383_);
v___x_687_ = lean_array_push(v___x_686_, v_snd_679_);
v___x_688_ = l_Lean_Meta_mkAppM(v___x_683_, v___x_687_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_688_) == 0)
{
lean_object* v_a_689_; lean_object* v___x_690_; lean_object* v___x_692_; 
v_a_689_ = lean_ctor_get(v___x_688_, 0);
lean_inc(v_a_689_);
lean_dec_ref_known(v___x_688_, 1);
v___x_690_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_690_, 0, v_fst_674_);
if (v_isShared_682_ == 0)
{
lean_ctor_set(v___x_681_, 1, v_a_689_);
v___x_692_ = v___x_681_;
goto v_reusejp_691_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_fst_678_);
lean_ctor_set(v_reuseFailAlloc_696_, 1, v_a_689_);
v___x_692_ = v_reuseFailAlloc_696_;
goto v_reusejp_691_;
}
v_reusejp_691_:
{
lean_object* v___x_694_; 
if (v_isShared_677_ == 0)
{
lean_ctor_set(v___x_676_, 1, v___x_692_);
lean_ctor_set(v___x_676_, 0, v___x_690_);
v___x_694_ = v___x_676_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v___x_690_);
lean_ctor_set(v_reuseFailAlloc_695_, 1, v___x_692_);
v___x_694_ = v_reuseFailAlloc_695_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
v_fst_391_ = v_fst_669_;
v_snd_392_ = v___x_694_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_704_; 
lean_del_object(v___x_681_);
lean_dec(v_fst_678_);
lean_del_object(v___x_676_);
lean_dec(v_fst_674_);
lean_dec(v_fst_669_);
lean_dec_ref(v_e_382_);
v_a_697_ = lean_ctor_get(v___x_688_, 0);
v_isSharedCheck_704_ = !lean_is_exclusive(v___x_688_);
if (v_isSharedCheck_704_ == 0)
{
v___x_699_ = v___x_688_;
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_a_697_);
lean_dec(v___x_688_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v___x_702_; 
if (v_isShared_700_ == 0)
{
v___x_702_ = v___x_699_;
goto v_reusejp_701_;
}
else
{
lean_object* v_reuseFailAlloc_703_; 
v_reuseFailAlloc_703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_703_, 0, v_a_697_);
v___x_702_ = v_reuseFailAlloc_703_;
goto v_reusejp_701_;
}
v_reusejp_701_:
{
return v___x_702_;
}
}
}
}
}
}
else
{
lean_object* v_a_707_; lean_object* v___x_709_; uint8_t v_isShared_710_; uint8_t v_isSharedCheck_714_; 
lean_dec(v_fst_669_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_707_ = lean_ctor_get(v___x_671_, 0);
v_isSharedCheck_714_ = !lean_is_exclusive(v___x_671_);
if (v_isSharedCheck_714_ == 0)
{
v___x_709_ = v___x_671_;
v_isShared_710_ = v_isSharedCheck_714_;
goto v_resetjp_708_;
}
else
{
lean_inc(v_a_707_);
lean_dec(v___x_671_);
v___x_709_ = lean_box(0);
v_isShared_710_ = v_isSharedCheck_714_;
goto v_resetjp_708_;
}
v_resetjp_708_:
{
lean_object* v___x_712_; 
if (v_isShared_710_ == 0)
{
v___x_712_ = v___x_709_;
goto v_reusejp_711_;
}
else
{
lean_object* v_reuseFailAlloc_713_; 
v_reuseFailAlloc_713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_713_, 0, v_a_707_);
v___x_712_ = v_reuseFailAlloc_713_;
goto v_reusejp_711_;
}
v_reusejp_711_:
{
return v___x_712_;
}
}
}
}
}
else
{
if (v_lb_384_ == 0)
{
lean_object* v_fst_715_; lean_object* v_fst_716_; lean_object* v_eval_717_; lean_object* v___x_718_; 
v_fst_715_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_715_);
lean_dec(v_a_424_);
v_fst_716_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_716_);
lean_dec(v_snd_425_);
v_eval_717_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_717_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_718_ = lean_apply_6(v_eval_717_, v_fst_716_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_718_) == 0)
{
lean_object* v_a_719_; lean_object* v_snd_720_; lean_object* v_fst_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_753_; 
v_a_719_ = lean_ctor_get(v___x_718_, 0);
lean_inc(v_a_719_);
lean_dec_ref_known(v___x_718_, 1);
v_snd_720_ = lean_ctor_get(v_a_719_, 1);
v_fst_721_ = lean_ctor_get(v_a_719_, 0);
v_isSharedCheck_753_ = !lean_is_exclusive(v_a_719_);
if (v_isSharedCheck_753_ == 0)
{
v___x_723_ = v_a_719_;
v_isShared_724_ = v_isSharedCheck_753_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_snd_720_);
lean_inc(v_fst_721_);
lean_dec(v_a_719_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_753_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v_fst_725_; lean_object* v_snd_726_; lean_object* v___x_728_; uint8_t v_isShared_729_; uint8_t v_isSharedCheck_752_; 
v_fst_725_ = lean_ctor_get(v_snd_720_, 0);
v_snd_726_ = lean_ctor_get(v_snd_720_, 1);
v_isSharedCheck_752_ = !lean_is_exclusive(v_snd_720_);
if (v_isSharedCheck_752_ == 0)
{
v___x_728_ = v_snd_720_;
v_isShared_729_ = v_isSharedCheck_752_;
goto v_resetjp_727_;
}
else
{
lean_inc(v_snd_726_);
lean_inc(v_fst_725_);
lean_dec(v_snd_720_);
v___x_728_ = lean_box(0);
v_isShared_729_ = v_isSharedCheck_752_;
goto v_resetjp_727_;
}
v_resetjp_727_:
{
lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v___x_730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__16));
v___x_731_ = lean_unsigned_to_nat(2u);
v___x_732_ = lean_mk_empty_array_with_capacity(v___x_731_);
v___x_733_ = lean_array_push(v___x_732_, v_pf_383_);
v___x_734_ = lean_array_push(v___x_733_, v_snd_726_);
v___x_735_ = l_Lean_Meta_mkAppM(v___x_730_, v___x_734_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_735_) == 0)
{
lean_object* v_a_736_; lean_object* v___x_737_; lean_object* v___x_739_; 
v_a_736_ = lean_ctor_get(v___x_735_, 0);
lean_inc(v_a_736_);
lean_dec_ref_known(v___x_735_, 1);
v___x_737_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_737_, 0, v_fst_721_);
if (v_isShared_729_ == 0)
{
lean_ctor_set(v___x_728_, 1, v_a_736_);
v___x_739_ = v___x_728_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_743_; 
v_reuseFailAlloc_743_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_743_, 0, v_fst_725_);
lean_ctor_set(v_reuseFailAlloc_743_, 1, v_a_736_);
v___x_739_ = v_reuseFailAlloc_743_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
lean_object* v___x_741_; 
if (v_isShared_724_ == 0)
{
lean_ctor_set(v___x_723_, 1, v___x_739_);
lean_ctor_set(v___x_723_, 0, v___x_737_);
v___x_741_ = v___x_723_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v___x_737_);
lean_ctor_set(v_reuseFailAlloc_742_, 1, v___x_739_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
v_fst_391_ = v_fst_715_;
v_snd_392_ = v___x_741_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_751_; 
lean_del_object(v___x_728_);
lean_dec(v_fst_725_);
lean_del_object(v___x_723_);
lean_dec(v_fst_721_);
lean_dec(v_fst_715_);
lean_dec_ref(v_e_382_);
v_a_744_ = lean_ctor_get(v___x_735_, 0);
v_isSharedCheck_751_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_751_ == 0)
{
v___x_746_ = v___x_735_;
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_a_744_);
lean_dec(v___x_735_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_751_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
lean_object* v___x_749_; 
if (v_isShared_747_ == 0)
{
v___x_749_ = v___x_746_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v_a_744_);
v___x_749_ = v_reuseFailAlloc_750_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
return v___x_749_;
}
}
}
}
}
}
else
{
lean_object* v_a_754_; lean_object* v___x_756_; uint8_t v_isShared_757_; uint8_t v_isSharedCheck_761_; 
lean_dec(v_fst_715_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_754_ = lean_ctor_get(v___x_718_, 0);
v_isSharedCheck_761_ = !lean_is_exclusive(v___x_718_);
if (v_isSharedCheck_761_ == 0)
{
v___x_756_ = v___x_718_;
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
else
{
lean_inc(v_a_754_);
lean_dec(v___x_718_);
v___x_756_ = lean_box(0);
v_isShared_757_ = v_isSharedCheck_761_;
goto v_resetjp_755_;
}
v_resetjp_755_:
{
lean_object* v___x_759_; 
if (v_isShared_757_ == 0)
{
v___x_759_ = v___x_756_;
goto v_reusejp_758_;
}
else
{
lean_object* v_reuseFailAlloc_760_; 
v_reuseFailAlloc_760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_760_, 0, v_a_754_);
v___x_759_ = v_reuseFailAlloc_760_;
goto v_reusejp_758_;
}
v_reusejp_758_:
{
return v___x_759_;
}
}
}
}
else
{
lean_object* v_fst_762_; lean_object* v_fst_763_; lean_object* v_eval_764_; lean_object* v___x_765_; 
v_fst_762_ = lean_ctor_get(v_a_424_, 0);
lean_inc(v_fst_762_);
lean_dec(v_a_424_);
v_fst_763_ = lean_ctor_get(v_snd_425_, 0);
lean_inc(v_fst_763_);
lean_dec(v_snd_425_);
v_eval_764_ = lean_ctor_get(v_m_381_, 6);
lean_inc_ref(v_eval_764_);
lean_dec_ref(v_m_381_);
lean_inc(v_a_388_);
lean_inc_ref(v_a_387_);
lean_inc(v_a_386_);
lean_inc_ref(v_a_385_);
v___x_765_ = lean_apply_6(v_eval_764_, v_fst_762_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, lean_box(0));
if (lean_obj_tag(v___x_765_) == 0)
{
lean_object* v_a_766_; lean_object* v_snd_767_; lean_object* v_fst_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_800_; 
v_a_766_ = lean_ctor_get(v___x_765_, 0);
lean_inc(v_a_766_);
lean_dec_ref_known(v___x_765_, 1);
v_snd_767_ = lean_ctor_get(v_a_766_, 1);
v_fst_768_ = lean_ctor_get(v_a_766_, 0);
v_isSharedCheck_800_ = !lean_is_exclusive(v_a_766_);
if (v_isSharedCheck_800_ == 0)
{
v___x_770_ = v_a_766_;
v_isShared_771_ = v_isSharedCheck_800_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_snd_767_);
lean_inc(v_fst_768_);
lean_dec(v_a_766_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_800_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v_fst_772_; lean_object* v_snd_773_; lean_object* v___x_775_; uint8_t v_isShared_776_; uint8_t v_isSharedCheck_799_; 
v_fst_772_ = lean_ctor_get(v_snd_767_, 0);
v_snd_773_ = lean_ctor_get(v_snd_767_, 1);
v_isSharedCheck_799_ = !lean_is_exclusive(v_snd_767_);
if (v_isSharedCheck_799_ == 0)
{
v___x_775_ = v_snd_767_;
v_isShared_776_ = v_isSharedCheck_799_;
goto v_resetjp_774_;
}
else
{
lean_inc(v_snd_773_);
lean_inc(v_fst_772_);
lean_dec(v_snd_767_);
v___x_775_ = lean_box(0);
v_isShared_776_ = v_isSharedCheck_799_;
goto v_resetjp_774_;
}
v_resetjp_774_:
{
lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; 
v___x_777_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___closed__18));
v___x_778_ = lean_unsigned_to_nat(2u);
v___x_779_ = lean_mk_empty_array_with_capacity(v___x_778_);
v___x_780_ = lean_array_push(v___x_779_, v_pf_383_);
v___x_781_ = lean_array_push(v___x_780_, v_snd_773_);
v___x_782_ = l_Lean_Meta_mkAppM(v___x_777_, v___x_781_, v_a_385_, v_a_386_, v_a_387_, v_a_388_);
if (lean_obj_tag(v___x_782_) == 0)
{
lean_object* v_a_783_; lean_object* v___x_784_; lean_object* v___x_786_; 
v_a_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc(v_a_783_);
lean_dec_ref_known(v___x_782_, 1);
v___x_784_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_784_, 0, v_fst_768_);
if (v_isShared_776_ == 0)
{
lean_ctor_set(v___x_775_, 1, v_a_783_);
v___x_786_ = v___x_775_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v_fst_772_);
lean_ctor_set(v_reuseFailAlloc_790_, 1, v_a_783_);
v___x_786_ = v_reuseFailAlloc_790_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
lean_object* v___x_788_; 
if (v_isShared_771_ == 0)
{
lean_ctor_set(v___x_770_, 1, v___x_786_);
lean_ctor_set(v___x_770_, 0, v___x_784_);
v___x_788_ = v___x_770_;
goto v_reusejp_787_;
}
else
{
lean_object* v_reuseFailAlloc_789_; 
v_reuseFailAlloc_789_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_789_, 0, v___x_784_);
lean_ctor_set(v_reuseFailAlloc_789_, 1, v___x_786_);
v___x_788_ = v_reuseFailAlloc_789_;
goto v_reusejp_787_;
}
v_reusejp_787_:
{
v_fst_391_ = v_fst_763_;
v_snd_392_ = v___x_788_;
v___y_393_ = v_a_385_;
v___y_394_ = v_a_386_;
v___y_395_ = v_a_387_;
v___y_396_ = v_a_388_;
goto v___jp_390_;
}
}
}
else
{
lean_object* v_a_791_; lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_798_; 
lean_del_object(v___x_775_);
lean_dec(v_fst_772_);
lean_del_object(v___x_770_);
lean_dec(v_fst_768_);
lean_dec(v_fst_763_);
lean_dec_ref(v_e_382_);
v_a_791_ = lean_ctor_get(v___x_782_, 0);
v_isSharedCheck_798_ = !lean_is_exclusive(v___x_782_);
if (v_isSharedCheck_798_ == 0)
{
v___x_793_ = v___x_782_;
v_isShared_794_ = v_isSharedCheck_798_;
goto v_resetjp_792_;
}
else
{
lean_inc(v_a_791_);
lean_dec(v___x_782_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_798_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v___x_796_; 
if (v_isShared_794_ == 0)
{
v___x_796_ = v___x_793_;
goto v_reusejp_795_;
}
else
{
lean_object* v_reuseFailAlloc_797_; 
v_reuseFailAlloc_797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_797_, 0, v_a_791_);
v___x_796_ = v_reuseFailAlloc_797_;
goto v_reusejp_795_;
}
v_reusejp_795_:
{
return v___x_796_;
}
}
}
}
}
}
else
{
lean_object* v_a_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_808_; 
lean_dec(v_fst_763_);
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
v_a_801_ = lean_ctor_get(v___x_765_, 0);
v_isSharedCheck_808_ = !lean_is_exclusive(v___x_765_);
if (v_isSharedCheck_808_ == 0)
{
v___x_803_ = v___x_765_;
v_isShared_804_ = v_isSharedCheck_808_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_a_801_);
lean_dec(v___x_765_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_808_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v___x_806_; 
if (v_isShared_804_ == 0)
{
v___x_806_ = v___x_803_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_807_; 
v_reuseFailAlloc_807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_807_, 0, v_a_801_);
v___x_806_ = v_reuseFailAlloc_807_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
return v___x_806_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_809_; lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_816_; 
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
lean_dec_ref(v_m_381_);
v_a_809_ = lean_ctor_get(v___x_423_, 0);
v_isSharedCheck_816_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_816_ == 0)
{
v___x_811_ = v___x_423_;
v_isShared_812_ = v_isSharedCheck_816_;
goto v_resetjp_810_;
}
else
{
lean_inc(v_a_809_);
lean_dec(v___x_423_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_816_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v___x_814_; 
if (v_isShared_812_ == 0)
{
v___x_814_ = v___x_811_;
goto v_reusejp_813_;
}
else
{
lean_object* v_reuseFailAlloc_815_; 
v_reuseFailAlloc_815_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_815_, 0, v_a_809_);
v___x_814_ = v_reuseFailAlloc_815_;
goto v_reusejp_813_;
}
v_reusejp_813_:
{
return v___x_814_;
}
}
}
}
else
{
lean_object* v_a_817_; lean_object* v___x_819_; uint8_t v_isShared_820_; uint8_t v_isSharedCheck_824_; 
lean_dec_ref(v_pf_383_);
lean_dec_ref(v_e_382_);
lean_dec_ref(v_m_381_);
v_a_817_ = lean_ctor_get(v___x_421_, 0);
v_isSharedCheck_824_ = !lean_is_exclusive(v___x_421_);
if (v_isSharedCheck_824_ == 0)
{
v___x_819_ = v___x_421_;
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
else
{
lean_inc(v_a_817_);
lean_dec(v___x_421_);
v___x_819_ = lean_box(0);
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
v_resetjp_818_:
{
lean_object* v___x_822_; 
if (v_isShared_820_ == 0)
{
v___x_822_ = v___x_819_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_a_817_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
}
v___jp_390_:
{
uint8_t v___x_397_; lean_object* v___x_398_; lean_object* v___f_399_; uint8_t v___x_400_; lean_object* v___x_401_; 
v___x_397_ = 2;
v___x_398_ = lean_box(v___x_397_);
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___lam__0___boxed), 8, 3);
lean_closure_set(v___f_399_, 0, v___x_398_);
lean_closure_set(v___f_399_, 1, v_e_382_);
lean_closure_set(v___f_399_, 2, v_fst_391_);
v___x_400_ = 0;
v___x_401_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_IntervalCases_Methods_getBound_spec__0___redArg(v___f_399_, v___x_400_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_401_) == 0)
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_412_; 
v_a_402_ = lean_ctor_get(v___x_401_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_412_ == 0)
{
v___x_404_ = v___x_401_;
v_isShared_405_ = v_isSharedCheck_412_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_401_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_412_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
uint8_t v___x_406_; 
v___x_406_ = lean_unbox(v_a_402_);
lean_dec(v_a_402_);
if (v___x_406_ == 1)
{
lean_object* v___x_408_; 
if (v_isShared_405_ == 0)
{
lean_ctor_set(v___x_404_, 0, v_snd_392_);
v___x_408_ = v___x_404_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v_snd_392_);
v___x_408_ = v_reuseFailAlloc_409_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
return v___x_408_;
}
}
else
{
lean_object* v___x_410_; lean_object* v___x_411_; 
lean_del_object(v___x_404_);
lean_dec_ref(v_snd_392_);
v___x_410_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9, &lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9);
v___x_411_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_410_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
return v___x_411_;
}
}
}
else
{
lean_object* v_a_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_420_; 
lean_dec_ref(v_snd_392_);
v_a_413_ = lean_ctor_get(v___x_401_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_420_ == 0)
{
v___x_415_ = v___x_401_;
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_a_413_);
lean_dec(v___x_401_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_418_; 
if (v_isShared_416_ == 0)
{
v___x_418_ = v___x_415_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_a_413_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___boxed(lean_object* v_m_825_, lean_object* v_e_826_, lean_object* v_pf_827_, lean_object* v_lb_828_, lean_object* v_a_829_, lean_object* v_a_830_, lean_object* v_a_831_, lean_object* v_a_832_, lean_object* v_a_833_){
_start:
{
uint8_t v_lb_boxed_834_; lean_object* v_res_835_; 
v_lb_boxed_834_ = lean_unbox(v_lb_828_);
v_res_835_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound(v_m_825_, v_e_826_, v_pf_827_, v_lb_boxed_834_, v_a_829_, v_a_830_, v_a_831_, v_a_832_);
lean_dec(v_a_832_);
lean_dec_ref(v_a_831_);
lean_dec(v_a_830_);
lean_dec_ref(v_a_829_);
return v_res_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds(lean_object* v_m_845_, lean_object* v_z1_846_, lean_object* v_z2_847_, lean_object* v_e1_848_, lean_object* v_e2_849_, lean_object* v_p1_850_, lean_object* v_p2_851_, lean_object* v_e_852_, lean_object* v_a_853_, lean_object* v_a_854_, lean_object* v_a_855_, lean_object* v_a_856_){
_start:
{
if (lean_obj_tag(v_z1_846_) == 0)
{
if (lean_obj_tag(v_z2_847_) == 0)
{
lean_object* v_n_858_; lean_object* v_n_859_; uint8_t v___x_860_; 
v_n_858_ = lean_ctor_get(v_z1_846_, 0);
v_n_859_ = lean_ctor_get(v_z2_847_, 0);
lean_inc(v_n_859_);
lean_dec_ref_known(v_z2_847_, 1);
v___x_860_ = lean_int_dec_le(v_n_859_, v_n_858_);
if (v___x_860_ == 0)
{
lean_object* v_proveLE_861_; lean_object* v_roundDown_862_; lean_object* v_mkNumeral_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; 
v_proveLE_861_ = lean_ctor_get(v_m_845_, 2);
lean_inc_ref(v_proveLE_861_);
v_roundDown_862_ = lean_ctor_get(v_m_845_, 5);
lean_inc_ref(v_roundDown_862_);
v_mkNumeral_863_ = lean_ctor_get(v_m_845_, 7);
lean_inc_ref(v_mkNumeral_863_);
lean_dec_ref(v_m_845_);
v___x_864_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0);
v___x_865_ = lean_int_sub(v_n_859_, v___x_864_);
lean_dec(v_n_859_);
lean_inc(v_a_856_);
lean_inc_ref(v_a_855_);
lean_inc(v_a_854_);
lean_inc_ref(v_a_853_);
v___x_866_ = lean_apply_6(v_mkNumeral_863_, v___x_865_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, lean_box(0));
if (lean_obj_tag(v___x_866_) == 0)
{
lean_object* v_a_867_; lean_object* v___x_868_; 
v_a_867_ = lean_ctor_get(v___x_866_, 0);
lean_inc_n(v_a_867_, 2);
lean_dec_ref_known(v___x_866_, 1);
lean_inc(v_a_856_);
lean_inc_ref(v_a_855_);
lean_inc(v_a_854_);
lean_inc_ref(v_a_853_);
v___x_868_ = lean_apply_9(v_roundDown_862_, v_e_852_, v_e2_849_, v_a_867_, v_p2_851_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, lean_box(0));
if (lean_obj_tag(v___x_868_) == 0)
{
lean_object* v_a_869_; lean_object* v___x_870_; 
v_a_869_ = lean_ctor_get(v___x_868_, 0);
lean_inc(v_a_869_);
lean_dec_ref_known(v___x_868_, 1);
lean_inc(v_a_856_);
lean_inc_ref(v_a_855_);
lean_inc(v_a_854_);
lean_inc_ref(v_a_853_);
v___x_870_ = lean_apply_7(v_proveLE_861_, v_a_867_, v_e1_848_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, lean_box(0));
if (lean_obj_tag(v___x_870_) == 0)
{
lean_object* v_a_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; 
v_a_871_ = lean_ctor_get(v___x_870_, 0);
lean_inc(v_a_871_);
lean_dec_ref_known(v___x_870_, 1);
v___x_872_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__1));
v___x_873_ = lean_unsigned_to_nat(2u);
v___x_874_ = lean_mk_empty_array_with_capacity(v___x_873_);
v___x_875_ = lean_array_push(v___x_874_, v_a_869_);
v___x_876_ = lean_array_push(v___x_875_, v_a_871_);
v___x_877_ = l_Lean_Meta_mkAppM(v___x_872_, v___x_876_, v_a_853_, v_a_854_, v_a_855_, v_a_856_);
if (lean_obj_tag(v___x_877_) == 0)
{
lean_object* v_a_878_; lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_886_; 
v_a_878_ = lean_ctor_get(v___x_877_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v___x_877_);
if (v_isSharedCheck_886_ == 0)
{
v___x_880_ = v___x_877_;
v_isShared_881_ = v_isSharedCheck_886_;
goto v_resetjp_879_;
}
else
{
lean_inc(v_a_878_);
lean_dec(v___x_877_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_886_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v___x_882_; lean_object* v___x_884_; 
v___x_882_ = l_Lean_Expr_app___override(v_p1_850_, v_a_878_);
if (v_isShared_881_ == 0)
{
lean_ctor_set(v___x_880_, 0, v___x_882_);
v___x_884_ = v___x_880_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v___x_882_);
v___x_884_ = v_reuseFailAlloc_885_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
return v___x_884_;
}
}
}
else
{
lean_dec_ref(v_p1_850_);
return v___x_877_;
}
}
else
{
lean_dec(v_a_869_);
lean_dec_ref(v_p1_850_);
return v___x_870_;
}
}
else
{
lean_dec(v_a_867_);
lean_dec_ref(v_proveLE_861_);
lean_dec_ref(v_p1_850_);
lean_dec_ref(v_e1_848_);
return v___x_868_;
}
}
else
{
lean_dec_ref(v_roundDown_862_);
lean_dec_ref(v_proveLE_861_);
lean_dec_ref(v_e_852_);
lean_dec_ref(v_p2_851_);
lean_dec_ref(v_p1_850_);
lean_dec_ref(v_e2_849_);
lean_dec_ref(v_e1_848_);
return v___x_866_;
}
}
else
{
lean_object* v_proveLE_887_; lean_object* v___x_888_; 
lean_dec(v_n_859_);
lean_dec_ref(v_e_852_);
v_proveLE_887_ = lean_ctor_get(v_m_845_, 2);
lean_inc_ref(v_proveLE_887_);
lean_dec_ref(v_m_845_);
lean_inc(v_a_856_);
lean_inc_ref(v_a_855_);
lean_inc(v_a_854_);
lean_inc_ref(v_a_853_);
v___x_888_ = lean_apply_7(v_proveLE_887_, v_e2_849_, v_e1_848_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, lean_box(0));
if (lean_obj_tag(v___x_888_) == 0)
{
lean_object* v_a_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; 
v_a_889_ = lean_ctor_get(v___x_888_, 0);
lean_inc(v_a_889_);
lean_dec_ref_known(v___x_888_, 1);
v___x_890_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__3));
v___x_891_ = lean_unsigned_to_nat(2u);
v___x_892_ = lean_mk_empty_array_with_capacity(v___x_891_);
v___x_893_ = lean_array_push(v___x_892_, v_p2_851_);
v___x_894_ = lean_array_push(v___x_893_, v_a_889_);
v___x_895_ = l_Lean_Meta_mkAppM(v___x_890_, v___x_894_, v_a_853_, v_a_854_, v_a_855_, v_a_856_);
if (lean_obj_tag(v___x_895_) == 0)
{
lean_object* v_a_896_; lean_object* v___x_898_; uint8_t v_isShared_899_; uint8_t v_isSharedCheck_904_; 
v_a_896_ = lean_ctor_get(v___x_895_, 0);
v_isSharedCheck_904_ = !lean_is_exclusive(v___x_895_);
if (v_isSharedCheck_904_ == 0)
{
v___x_898_ = v___x_895_;
v_isShared_899_ = v_isSharedCheck_904_;
goto v_resetjp_897_;
}
else
{
lean_inc(v_a_896_);
lean_dec(v___x_895_);
v___x_898_ = lean_box(0);
v_isShared_899_ = v_isSharedCheck_904_;
goto v_resetjp_897_;
}
v_resetjp_897_:
{
lean_object* v___x_900_; lean_object* v___x_902_; 
v___x_900_ = l_Lean_Expr_app___override(v_p1_850_, v_a_896_);
if (v_isShared_899_ == 0)
{
lean_ctor_set(v___x_898_, 0, v___x_900_);
v___x_902_ = v___x_898_;
goto v_reusejp_901_;
}
else
{
lean_object* v_reuseFailAlloc_903_; 
v_reuseFailAlloc_903_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_903_, 0, v___x_900_);
v___x_902_ = v_reuseFailAlloc_903_;
goto v_reusejp_901_;
}
v_reusejp_901_:
{
return v___x_902_;
}
}
}
else
{
lean_dec_ref(v_p1_850_);
return v___x_895_;
}
}
else
{
lean_dec_ref(v_p2_851_);
lean_dec_ref(v_p1_850_);
return v___x_888_;
}
}
}
else
{
lean_object* v_n_905_; lean_object* v_n_906_; lean_object* v___x_908_; uint8_t v_isShared_909_; uint8_t v_isSharedCheck_933_; 
lean_dec_ref(v_e_852_);
v_n_905_ = lean_ctor_get(v_z1_846_, 0);
v_n_906_ = lean_ctor_get(v_z2_847_, 0);
v_isSharedCheck_933_ = !lean_is_exclusive(v_z2_847_);
if (v_isSharedCheck_933_ == 0)
{
v___x_908_ = v_z2_847_;
v_isShared_909_ = v_isSharedCheck_933_;
goto v_resetjp_907_;
}
else
{
lean_inc(v_n_906_);
lean_dec(v_z2_847_);
v___x_908_ = lean_box(0);
v_isShared_909_ = v_isSharedCheck_933_;
goto v_resetjp_907_;
}
v_resetjp_907_:
{
uint8_t v___x_910_; 
v___x_910_ = lean_int_dec_eq(v_n_905_, v_n_906_);
lean_dec(v_n_906_);
if (v___x_910_ == 0)
{
lean_object* v_proveLE_911_; lean_object* v___x_912_; 
lean_del_object(v___x_908_);
v_proveLE_911_ = lean_ctor_get(v_m_845_, 2);
lean_inc_ref(v_proveLE_911_);
lean_dec_ref(v_m_845_);
lean_inc(v_a_856_);
lean_inc_ref(v_a_855_);
lean_inc(v_a_854_);
lean_inc_ref(v_a_853_);
v___x_912_ = lean_apply_7(v_proveLE_911_, v_e2_849_, v_e1_848_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, lean_box(0));
if (lean_obj_tag(v___x_912_) == 0)
{
lean_object* v_a_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; 
v_a_913_ = lean_ctor_get(v___x_912_, 0);
lean_inc(v_a_913_);
lean_dec_ref_known(v___x_912_, 1);
v___x_914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__1));
v___x_915_ = lean_unsigned_to_nat(2u);
v___x_916_ = lean_mk_empty_array_with_capacity(v___x_915_);
v___x_917_ = lean_array_push(v___x_916_, v_p2_851_);
v___x_918_ = lean_array_push(v___x_917_, v_a_913_);
v___x_919_ = l_Lean_Meta_mkAppM(v___x_914_, v___x_918_, v_a_853_, v_a_854_, v_a_855_, v_a_856_);
if (lean_obj_tag(v___x_919_) == 0)
{
lean_object* v_a_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_928_; 
v_a_920_ = lean_ctor_get(v___x_919_, 0);
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_919_);
if (v_isSharedCheck_928_ == 0)
{
v___x_922_ = v___x_919_;
v_isShared_923_ = v_isSharedCheck_928_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_a_920_);
lean_dec(v___x_919_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_928_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_924_; lean_object* v___x_926_; 
v___x_924_ = l_Lean_Expr_app___override(v_p1_850_, v_a_920_);
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 0, v___x_924_);
v___x_926_ = v___x_922_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v___x_924_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
return v___x_926_;
}
}
}
else
{
lean_dec_ref(v_p1_850_);
return v___x_919_;
}
}
else
{
lean_dec_ref(v_p2_851_);
lean_dec_ref(v_p1_850_);
return v___x_912_;
}
}
else
{
lean_object* v___x_929_; lean_object* v___x_931_; 
lean_dec_ref(v_e2_849_);
lean_dec_ref(v_e1_848_);
lean_dec_ref(v_m_845_);
v___x_929_ = l_Lean_Expr_app___override(v_p1_850_, v_p2_851_);
if (v_isShared_909_ == 0)
{
lean_ctor_set_tag(v___x_908_, 0);
lean_ctor_set(v___x_908_, 0, v___x_929_);
v___x_931_ = v___x_908_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_932_; 
v_reuseFailAlloc_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_932_, 0, v___x_929_);
v___x_931_ = v_reuseFailAlloc_932_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
return v___x_931_;
}
}
}
}
}
else
{
lean_dec_ref(v_e_852_);
if (lean_obj_tag(v_z2_847_) == 0)
{
lean_object* v_n_934_; lean_object* v_n_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_962_; 
v_n_934_ = lean_ctor_get(v_z1_846_, 0);
v_n_935_ = lean_ctor_get(v_z2_847_, 0);
v_isSharedCheck_962_ = !lean_is_exclusive(v_z2_847_);
if (v_isSharedCheck_962_ == 0)
{
v___x_937_ = v_z2_847_;
v_isShared_938_ = v_isSharedCheck_962_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_n_935_);
lean_dec(v_z2_847_);
v___x_937_ = lean_box(0);
v_isShared_938_ = v_isSharedCheck_962_;
goto v_resetjp_936_;
}
v_resetjp_936_:
{
uint8_t v___x_939_; 
v___x_939_ = lean_int_dec_eq(v_n_934_, v_n_935_);
lean_dec(v_n_935_);
if (v___x_939_ == 0)
{
lean_object* v_proveLE_940_; lean_object* v___x_941_; 
lean_del_object(v___x_937_);
v_proveLE_940_ = lean_ctor_get(v_m_845_, 2);
lean_inc_ref(v_proveLE_940_);
lean_dec_ref(v_m_845_);
lean_inc(v_a_856_);
lean_inc_ref(v_a_855_);
lean_inc(v_a_854_);
lean_inc_ref(v_a_853_);
v___x_941_ = lean_apply_7(v_proveLE_940_, v_e2_849_, v_e1_848_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, lean_box(0));
if (lean_obj_tag(v___x_941_) == 0)
{
lean_object* v_a_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v_a_942_ = lean_ctor_get(v___x_941_, 0);
lean_inc(v_a_942_);
lean_dec_ref_known(v___x_941_, 1);
v___x_943_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__1));
v___x_944_ = lean_unsigned_to_nat(2u);
v___x_945_ = lean_mk_empty_array_with_capacity(v___x_944_);
v___x_946_ = lean_array_push(v___x_945_, v_a_942_);
v___x_947_ = lean_array_push(v___x_946_, v_p1_850_);
v___x_948_ = l_Lean_Meta_mkAppM(v___x_943_, v___x_947_, v_a_853_, v_a_854_, v_a_855_, v_a_856_);
if (lean_obj_tag(v___x_948_) == 0)
{
lean_object* v_a_949_; lean_object* v___x_951_; uint8_t v_isShared_952_; uint8_t v_isSharedCheck_957_; 
v_a_949_ = lean_ctor_get(v___x_948_, 0);
v_isSharedCheck_957_ = !lean_is_exclusive(v___x_948_);
if (v_isSharedCheck_957_ == 0)
{
v___x_951_ = v___x_948_;
v_isShared_952_ = v_isSharedCheck_957_;
goto v_resetjp_950_;
}
else
{
lean_inc(v_a_949_);
lean_dec(v___x_948_);
v___x_951_ = lean_box(0);
v_isShared_952_ = v_isSharedCheck_957_;
goto v_resetjp_950_;
}
v_resetjp_950_:
{
lean_object* v___x_953_; lean_object* v___x_955_; 
v___x_953_ = l_Lean_Expr_app___override(v_p2_851_, v_a_949_);
if (v_isShared_952_ == 0)
{
lean_ctor_set(v___x_951_, 0, v___x_953_);
v___x_955_ = v___x_951_;
goto v_reusejp_954_;
}
else
{
lean_object* v_reuseFailAlloc_956_; 
v_reuseFailAlloc_956_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_956_, 0, v___x_953_);
v___x_955_ = v_reuseFailAlloc_956_;
goto v_reusejp_954_;
}
v_reusejp_954_:
{
return v___x_955_;
}
}
}
else
{
lean_dec_ref(v_p2_851_);
return v___x_948_;
}
}
else
{
lean_dec_ref(v_p2_851_);
lean_dec_ref(v_p1_850_);
return v___x_941_;
}
}
else
{
lean_object* v___x_958_; lean_object* v___x_960_; 
lean_dec_ref(v_e2_849_);
lean_dec_ref(v_e1_848_);
lean_dec_ref(v_m_845_);
v___x_958_ = l_Lean_Expr_app___override(v_p2_851_, v_p1_850_);
if (v_isShared_938_ == 0)
{
lean_ctor_set(v___x_937_, 0, v___x_958_);
v___x_960_ = v___x_937_;
goto v_reusejp_959_;
}
else
{
lean_object* v_reuseFailAlloc_961_; 
v_reuseFailAlloc_961_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_961_, 0, v___x_958_);
v___x_960_ = v_reuseFailAlloc_961_;
goto v_reusejp_959_;
}
v_reusejp_959_:
{
return v___x_960_;
}
}
}
}
else
{
lean_object* v_proveLT_963_; lean_object* v___x_964_; 
lean_dec_ref_known(v_z2_847_, 1);
v_proveLT_963_ = lean_ctor_get(v_m_845_, 3);
lean_inc_ref(v_proveLT_963_);
lean_dec_ref(v_m_845_);
lean_inc(v_a_856_);
lean_inc_ref(v_a_855_);
lean_inc(v_a_854_);
lean_inc_ref(v_a_853_);
v___x_964_ = lean_apply_7(v_proveLT_963_, v_e2_849_, v_e1_848_, v_a_853_, v_a_854_, v_a_855_, v_a_856_, lean_box(0));
if (lean_obj_tag(v___x_964_) == 0)
{
lean_object* v_a_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; 
v_a_965_ = lean_ctor_get(v___x_964_, 0);
lean_inc(v_a_965_);
lean_dec_ref_known(v___x_964_, 1);
v___x_966_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___closed__1));
v___x_967_ = lean_unsigned_to_nat(2u);
v___x_968_ = lean_mk_empty_array_with_capacity(v___x_967_);
v___x_969_ = lean_array_push(v___x_968_, v_p1_850_);
v___x_970_ = lean_array_push(v___x_969_, v_p2_851_);
v___x_971_ = l_Lean_Meta_mkAppM(v___x_966_, v___x_970_, v_a_853_, v_a_854_, v_a_855_, v_a_856_);
if (lean_obj_tag(v___x_971_) == 0)
{
lean_object* v_a_972_; lean_object* v___x_974_; uint8_t v_isShared_975_; uint8_t v_isSharedCheck_980_; 
v_a_972_ = lean_ctor_get(v___x_971_, 0);
v_isSharedCheck_980_ = !lean_is_exclusive(v___x_971_);
if (v_isSharedCheck_980_ == 0)
{
v___x_974_ = v___x_971_;
v_isShared_975_ = v_isSharedCheck_980_;
goto v_resetjp_973_;
}
else
{
lean_inc(v_a_972_);
lean_dec(v___x_971_);
v___x_974_ = lean_box(0);
v_isShared_975_ = v_isSharedCheck_980_;
goto v_resetjp_973_;
}
v_resetjp_973_:
{
lean_object* v___x_976_; lean_object* v___x_978_; 
v___x_976_ = l_Lean_Expr_app___override(v_a_965_, v_a_972_);
if (v_isShared_975_ == 0)
{
lean_ctor_set(v___x_974_, 0, v___x_976_);
v___x_978_ = v___x_974_;
goto v_reusejp_977_;
}
else
{
lean_object* v_reuseFailAlloc_979_; 
v_reuseFailAlloc_979_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_979_, 0, v___x_976_);
v___x_978_ = v_reuseFailAlloc_979_;
goto v_reusejp_977_;
}
v_reusejp_977_:
{
return v___x_978_;
}
}
}
else
{
lean_dec(v_a_965_);
return v___x_971_;
}
}
else
{
lean_dec_ref(v_p2_851_);
lean_dec_ref(v_p1_850_);
return v___x_964_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds___boxed(lean_object* v_m_981_, lean_object* v_z1_982_, lean_object* v_z2_983_, lean_object* v_e1_984_, lean_object* v_e2_985_, lean_object* v_p1_986_, lean_object* v_p2_987_, lean_object* v_e_988_, lean_object* v_a_989_, lean_object* v_a_990_, lean_object* v_a_991_, lean_object* v_a_992_, lean_object* v_a_993_){
_start:
{
lean_object* v_res_994_; 
v_res_994_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds(v_m_981_, v_z1_982_, v_z2_983_, v_e1_984_, v_e2_985_, v_p1_986_, v_p2_987_, v_e_988_, v_a_989_, v_a_990_, v_a_991_, v_a_992_);
lean_dec(v_a_992_);
lean_dec_ref(v_a_991_);
lean_dec(v_a_990_);
lean_dec_ref(v_a_989_);
lean_dec_ref(v_z1_982_);
return v_res_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0(lean_object* v_msg_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_){
_start:
{
lean_object* v___f_1002_; lean_object* v___x_3130__overap_1003_; lean_object* v___x_1004_; 
v___f_1002_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0___closed__0));
v___x_3130__overap_1003_ = lean_panic_fn_borrowed(v___f_1002_, v_msg_996_);
lean_inc(v___y_1000_);
lean_inc_ref(v___y_999_);
lean_inc(v___y_998_);
lean_inc_ref(v___y_997_);
v___x_1004_ = lean_apply_5(v___x_3130__overap_1003_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_, lean_box(0));
return v___x_1004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0___boxed(lean_object* v_msg_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
lean_object* v_res_1011_; 
v_res_1011_ = lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0(v_msg_1005_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_);
lean_dec(v___y_1009_);
lean_dec_ref(v___y_1008_);
lean_dec(v___y_1007_);
lean_dec_ref(v___y_1006_);
return v_res_1011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__2(lean_object* v_a_1012_){
_start:
{
lean_object* v___x_1013_; 
v___x_1013_ = lean_nat_to_int(v_a_1012_);
return v___x_1013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg(lean_object* v_mvarId_1014_, lean_object* v_x_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_){
_start:
{
lean_object* v___x_1021_; 
v___x_1021_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1014_, v_x_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
if (lean_obj_tag(v___x_1021_) == 0)
{
lean_object* v_a_1022_; lean_object* v___x_1024_; uint8_t v_isShared_1025_; uint8_t v_isSharedCheck_1029_; 
v_a_1022_ = lean_ctor_get(v___x_1021_, 0);
v_isSharedCheck_1029_ = !lean_is_exclusive(v___x_1021_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1024_ = v___x_1021_;
v_isShared_1025_ = v_isSharedCheck_1029_;
goto v_resetjp_1023_;
}
else
{
lean_inc(v_a_1022_);
lean_dec(v___x_1021_);
v___x_1024_ = lean_box(0);
v_isShared_1025_ = v_isSharedCheck_1029_;
goto v_resetjp_1023_;
}
v_resetjp_1023_:
{
lean_object* v___x_1027_; 
if (v_isShared_1025_ == 0)
{
v___x_1027_ = v___x_1024_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v_a_1022_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
}
else
{
lean_object* v_a_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1037_; 
v_a_1030_ = lean_ctor_get(v___x_1021_, 0);
v_isSharedCheck_1037_ = !lean_is_exclusive(v___x_1021_);
if (v_isSharedCheck_1037_ == 0)
{
v___x_1032_ = v___x_1021_;
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_a_1030_);
lean_dec(v___x_1021_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1037_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1035_; 
if (v_isShared_1033_ == 0)
{
v___x_1035_ = v___x_1032_;
goto v_reusejp_1034_;
}
else
{
lean_object* v_reuseFailAlloc_1036_; 
v_reuseFailAlloc_1036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1036_, 0, v_a_1030_);
v___x_1035_ = v_reuseFailAlloc_1036_;
goto v_reusejp_1034_;
}
v_reusejp_1034_:
{
return v___x_1035_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg___boxed(lean_object* v_mvarId_1038_, lean_object* v_x_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_){
_start:
{
lean_object* v_res_1045_; 
v_res_1045_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg(v_mvarId_1038_, v_x_1039_, v___y_1040_, v___y_1041_, v___y_1042_, v___y_1043_);
lean_dec(v___y_1043_);
lean_dec_ref(v___y_1042_);
lean_dec(v___y_1041_);
lean_dec_ref(v___y_1040_);
return v_res_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3(lean_object* v_00_u03b1_1046_, lean_object* v_mvarId_1047_, lean_object* v_x_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_){
_start:
{
lean_object* v___x_1054_; 
v___x_1054_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg(v_mvarId_1047_, v_x_1048_, v___y_1049_, v___y_1050_, v___y_1051_, v___y_1052_);
return v___x_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___boxed(lean_object* v_00_u03b1_1055_, lean_object* v_mvarId_1056_, lean_object* v_x_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v_res_1063_; 
v_res_1063_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3(v_00_u03b1_1055_, v_mvarId_1056_, v_x_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
lean_dec(v___y_1061_);
lean_dec_ref(v___y_1060_);
lean_dec(v___y_1059_);
lean_dec_ref(v___y_1058_);
return v_res_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5_spec__6___redArg(lean_object* v_x_1064_, lean_object* v_x_1065_, lean_object* v_x_1066_, lean_object* v_x_1067_){
_start:
{
lean_object* v_ks_1068_; lean_object* v_vs_1069_; lean_object* v___x_1071_; uint8_t v_isShared_1072_; uint8_t v_isSharedCheck_1093_; 
v_ks_1068_ = lean_ctor_get(v_x_1064_, 0);
v_vs_1069_ = lean_ctor_get(v_x_1064_, 1);
v_isSharedCheck_1093_ = !lean_is_exclusive(v_x_1064_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1071_ = v_x_1064_;
v_isShared_1072_ = v_isSharedCheck_1093_;
goto v_resetjp_1070_;
}
else
{
lean_inc(v_vs_1069_);
lean_inc(v_ks_1068_);
lean_dec(v_x_1064_);
v___x_1071_ = lean_box(0);
v_isShared_1072_ = v_isSharedCheck_1093_;
goto v_resetjp_1070_;
}
v_resetjp_1070_:
{
lean_object* v___x_1073_; uint8_t v___x_1074_; 
v___x_1073_ = lean_array_get_size(v_ks_1068_);
v___x_1074_ = lean_nat_dec_lt(v_x_1065_, v___x_1073_);
if (v___x_1074_ == 0)
{
lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1078_; 
lean_dec(v_x_1065_);
v___x_1075_ = lean_array_push(v_ks_1068_, v_x_1066_);
v___x_1076_ = lean_array_push(v_vs_1069_, v_x_1067_);
if (v_isShared_1072_ == 0)
{
lean_ctor_set(v___x_1071_, 1, v___x_1076_);
lean_ctor_set(v___x_1071_, 0, v___x_1075_);
v___x_1078_ = v___x_1071_;
goto v_reusejp_1077_;
}
else
{
lean_object* v_reuseFailAlloc_1079_; 
v_reuseFailAlloc_1079_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1079_, 0, v___x_1075_);
lean_ctor_set(v_reuseFailAlloc_1079_, 1, v___x_1076_);
v___x_1078_ = v_reuseFailAlloc_1079_;
goto v_reusejp_1077_;
}
v_reusejp_1077_:
{
return v___x_1078_;
}
}
else
{
lean_object* v_k_x27_1080_; uint8_t v___x_1081_; 
v_k_x27_1080_ = lean_array_fget_borrowed(v_ks_1068_, v_x_1065_);
v___x_1081_ = l_Lean_instBEqMVarId_beq(v_x_1066_, v_k_x27_1080_);
if (v___x_1081_ == 0)
{
lean_object* v___x_1083_; 
if (v_isShared_1072_ == 0)
{
v___x_1083_ = v___x_1071_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1087_; 
v_reuseFailAlloc_1087_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1087_, 0, v_ks_1068_);
lean_ctor_set(v_reuseFailAlloc_1087_, 1, v_vs_1069_);
v___x_1083_ = v_reuseFailAlloc_1087_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
lean_object* v___x_1084_; lean_object* v___x_1085_; 
v___x_1084_ = lean_unsigned_to_nat(1u);
v___x_1085_ = lean_nat_add(v_x_1065_, v___x_1084_);
lean_dec(v_x_1065_);
v_x_1064_ = v___x_1083_;
v_x_1065_ = v___x_1085_;
goto _start;
}
}
else
{
lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1091_; 
v___x_1088_ = lean_array_fset(v_ks_1068_, v_x_1065_, v_x_1066_);
v___x_1089_ = lean_array_fset(v_vs_1069_, v_x_1065_, v_x_1067_);
lean_dec(v_x_1065_);
if (v_isShared_1072_ == 0)
{
lean_ctor_set(v___x_1071_, 1, v___x_1089_);
lean_ctor_set(v___x_1071_, 0, v___x_1088_);
v___x_1091_ = v___x_1071_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1092_; 
v_reuseFailAlloc_1092_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1092_, 0, v___x_1088_);
lean_ctor_set(v_reuseFailAlloc_1092_, 1, v___x_1089_);
v___x_1091_ = v_reuseFailAlloc_1092_;
goto v_reusejp_1090_;
}
v_reusejp_1090_:
{
return v___x_1091_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5___redArg(lean_object* v_n_1094_, lean_object* v_k_1095_, lean_object* v_v_1096_){
_start:
{
lean_object* v___x_1097_; lean_object* v___x_1098_; 
v___x_1097_ = lean_unsigned_to_nat(0u);
v___x_1098_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5_spec__6___redArg(v_n_1094_, v___x_1097_, v_k_1095_, v_v_1096_);
return v___x_1098_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_1099_; 
v___x_1099_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg(lean_object* v_x_1100_, size_t v_x_1101_, size_t v_x_1102_, lean_object* v_x_1103_, lean_object* v_x_1104_){
_start:
{
if (lean_obj_tag(v_x_1100_) == 0)
{
lean_object* v_es_1105_; size_t v___x_1106_; size_t v___x_1107_; lean_object* v_j_1108_; lean_object* v___x_1109_; uint8_t v___x_1110_; 
v_es_1105_ = lean_ctor_get(v_x_1100_, 0);
v___x_1106_ = ((size_t)31ULL);
v___x_1107_ = lean_usize_land(v_x_1101_, v___x_1106_);
v_j_1108_ = lean_usize_to_nat(v___x_1107_);
v___x_1109_ = lean_array_get_size(v_es_1105_);
v___x_1110_ = lean_nat_dec_lt(v_j_1108_, v___x_1109_);
if (v___x_1110_ == 0)
{
lean_dec(v_j_1108_);
lean_dec(v_x_1104_);
lean_dec(v_x_1103_);
return v_x_1100_;
}
else
{
lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1149_; 
lean_inc_ref(v_es_1105_);
v_isSharedCheck_1149_ = !lean_is_exclusive(v_x_1100_);
if (v_isSharedCheck_1149_ == 0)
{
lean_object* v_unused_1150_; 
v_unused_1150_ = lean_ctor_get(v_x_1100_, 0);
lean_dec(v_unused_1150_);
v___x_1112_ = v_x_1100_;
v_isShared_1113_ = v_isSharedCheck_1149_;
goto v_resetjp_1111_;
}
else
{
lean_dec(v_x_1100_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1149_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v_v_1114_; lean_object* v___x_1115_; lean_object* v_xs_x27_1116_; lean_object* v___y_1118_; 
v_v_1114_ = lean_array_fget(v_es_1105_, v_j_1108_);
v___x_1115_ = lean_box(0);
v_xs_x27_1116_ = lean_array_fset(v_es_1105_, v_j_1108_, v___x_1115_);
switch(lean_obj_tag(v_v_1114_))
{
case 0:
{
lean_object* v_key_1123_; lean_object* v_val_1124_; lean_object* v___x_1126_; uint8_t v_isShared_1127_; uint8_t v_isSharedCheck_1134_; 
v_key_1123_ = lean_ctor_get(v_v_1114_, 0);
v_val_1124_ = lean_ctor_get(v_v_1114_, 1);
v_isSharedCheck_1134_ = !lean_is_exclusive(v_v_1114_);
if (v_isSharedCheck_1134_ == 0)
{
v___x_1126_ = v_v_1114_;
v_isShared_1127_ = v_isSharedCheck_1134_;
goto v_resetjp_1125_;
}
else
{
lean_inc(v_val_1124_);
lean_inc(v_key_1123_);
lean_dec(v_v_1114_);
v___x_1126_ = lean_box(0);
v_isShared_1127_ = v_isSharedCheck_1134_;
goto v_resetjp_1125_;
}
v_resetjp_1125_:
{
uint8_t v___x_1128_; 
v___x_1128_ = l_Lean_instBEqMVarId_beq(v_x_1103_, v_key_1123_);
if (v___x_1128_ == 0)
{
lean_object* v___x_1129_; lean_object* v___x_1130_; 
lean_del_object(v___x_1126_);
v___x_1129_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1123_, v_val_1124_, v_x_1103_, v_x_1104_);
v___x_1130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1130_, 0, v___x_1129_);
v___y_1118_ = v___x_1130_;
goto v___jp_1117_;
}
else
{
lean_object* v___x_1132_; 
lean_dec(v_val_1124_);
lean_dec(v_key_1123_);
if (v_isShared_1127_ == 0)
{
lean_ctor_set(v___x_1126_, 1, v_x_1104_);
lean_ctor_set(v___x_1126_, 0, v_x_1103_);
v___x_1132_ = v___x_1126_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1133_; 
v_reuseFailAlloc_1133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1133_, 0, v_x_1103_);
lean_ctor_set(v_reuseFailAlloc_1133_, 1, v_x_1104_);
v___x_1132_ = v_reuseFailAlloc_1133_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
v___y_1118_ = v___x_1132_;
goto v___jp_1117_;
}
}
}
}
case 1:
{
lean_object* v_node_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1147_; 
v_node_1135_ = lean_ctor_get(v_v_1114_, 0);
v_isSharedCheck_1147_ = !lean_is_exclusive(v_v_1114_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1137_ = v_v_1114_;
v_isShared_1138_ = v_isSharedCheck_1147_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_node_1135_);
lean_dec(v_v_1114_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1147_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
size_t v___x_1139_; size_t v___x_1140_; size_t v___x_1141_; size_t v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1145_; 
v___x_1139_ = ((size_t)5ULL);
v___x_1140_ = lean_usize_shift_right(v_x_1101_, v___x_1139_);
v___x_1141_ = ((size_t)1ULL);
v___x_1142_ = lean_usize_add(v_x_1102_, v___x_1141_);
v___x_1143_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg(v_node_1135_, v___x_1140_, v___x_1142_, v_x_1103_, v_x_1104_);
if (v_isShared_1138_ == 0)
{
lean_ctor_set(v___x_1137_, 0, v___x_1143_);
v___x_1145_ = v___x_1137_;
goto v_reusejp_1144_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v___x_1143_);
v___x_1145_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1144_;
}
v_reusejp_1144_:
{
v___y_1118_ = v___x_1145_;
goto v___jp_1117_;
}
}
}
default: 
{
lean_object* v___x_1148_; 
v___x_1148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1148_, 0, v_x_1103_);
lean_ctor_set(v___x_1148_, 1, v_x_1104_);
v___y_1118_ = v___x_1148_;
goto v___jp_1117_;
}
}
v___jp_1117_:
{
lean_object* v___x_1119_; lean_object* v___x_1121_; 
v___x_1119_ = lean_array_fset(v_xs_x27_1116_, v_j_1108_, v___y_1118_);
lean_dec(v_j_1108_);
if (v_isShared_1113_ == 0)
{
lean_ctor_set(v___x_1112_, 0, v___x_1119_);
v___x_1121_ = v___x_1112_;
goto v_reusejp_1120_;
}
else
{
lean_object* v_reuseFailAlloc_1122_; 
v_reuseFailAlloc_1122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1122_, 0, v___x_1119_);
v___x_1121_ = v_reuseFailAlloc_1122_;
goto v_reusejp_1120_;
}
v_reusejp_1120_:
{
return v___x_1121_;
}
}
}
}
}
else
{
lean_object* v_ks_1151_; lean_object* v_vs_1152_; lean_object* v___x_1154_; uint8_t v_isShared_1155_; uint8_t v_isSharedCheck_1172_; 
v_ks_1151_ = lean_ctor_get(v_x_1100_, 0);
v_vs_1152_ = lean_ctor_get(v_x_1100_, 1);
v_isSharedCheck_1172_ = !lean_is_exclusive(v_x_1100_);
if (v_isSharedCheck_1172_ == 0)
{
v___x_1154_ = v_x_1100_;
v_isShared_1155_ = v_isSharedCheck_1172_;
goto v_resetjp_1153_;
}
else
{
lean_inc(v_vs_1152_);
lean_inc(v_ks_1151_);
lean_dec(v_x_1100_);
v___x_1154_ = lean_box(0);
v_isShared_1155_ = v_isSharedCheck_1172_;
goto v_resetjp_1153_;
}
v_resetjp_1153_:
{
lean_object* v___x_1157_; 
if (v_isShared_1155_ == 0)
{
v___x_1157_ = v___x_1154_;
goto v_reusejp_1156_;
}
else
{
lean_object* v_reuseFailAlloc_1171_; 
v_reuseFailAlloc_1171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1171_, 0, v_ks_1151_);
lean_ctor_set(v_reuseFailAlloc_1171_, 1, v_vs_1152_);
v___x_1157_ = v_reuseFailAlloc_1171_;
goto v_reusejp_1156_;
}
v_reusejp_1156_:
{
lean_object* v_newNode_1158_; uint8_t v___y_1160_; size_t v___x_1166_; uint8_t v___x_1167_; 
v_newNode_1158_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5___redArg(v___x_1157_, v_x_1103_, v_x_1104_);
v___x_1166_ = ((size_t)7ULL);
v___x_1167_ = lean_usize_dec_le(v___x_1166_, v_x_1102_);
if (v___x_1167_ == 0)
{
lean_object* v___x_1168_; lean_object* v___x_1169_; uint8_t v___x_1170_; 
v___x_1168_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1158_);
v___x_1169_ = lean_unsigned_to_nat(4u);
v___x_1170_ = lean_nat_dec_lt(v___x_1168_, v___x_1169_);
lean_dec(v___x_1168_);
v___y_1160_ = v___x_1170_;
goto v___jp_1159_;
}
else
{
v___y_1160_ = v___x_1167_;
goto v___jp_1159_;
}
v___jp_1159_:
{
if (v___y_1160_ == 0)
{
lean_object* v_ks_1161_; lean_object* v_vs_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; 
v_ks_1161_ = lean_ctor_get(v_newNode_1158_, 0);
lean_inc_ref(v_ks_1161_);
v_vs_1162_ = lean_ctor_get(v_newNode_1158_, 1);
lean_inc_ref(v_vs_1162_);
lean_dec_ref(v_newNode_1158_);
v___x_1163_ = lean_unsigned_to_nat(0u);
v___x_1164_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___closed__0);
v___x_1165_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___redArg(v_x_1102_, v_ks_1161_, v_vs_1162_, v___x_1163_, v___x_1164_);
lean_dec_ref(v_vs_1162_);
lean_dec_ref(v_ks_1161_);
return v___x_1165_;
}
else
{
return v_newNode_1158_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___redArg(size_t v_depth_1173_, lean_object* v_keys_1174_, lean_object* v_vals_1175_, lean_object* v_i_1176_, lean_object* v_entries_1177_){
_start:
{
lean_object* v___x_1178_; uint8_t v___x_1179_; 
v___x_1178_ = lean_array_get_size(v_keys_1174_);
v___x_1179_ = lean_nat_dec_lt(v_i_1176_, v___x_1178_);
if (v___x_1179_ == 0)
{
lean_dec(v_i_1176_);
return v_entries_1177_;
}
else
{
lean_object* v_k_1180_; lean_object* v_v_1181_; uint64_t v___x_1182_; size_t v_h_1183_; size_t v___x_1184_; lean_object* v___x_1185_; size_t v___x_1186_; size_t v___x_1187_; size_t v___x_1188_; size_t v_h_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; 
v_k_1180_ = lean_array_fget_borrowed(v_keys_1174_, v_i_1176_);
v_v_1181_ = lean_array_fget_borrowed(v_vals_1175_, v_i_1176_);
v___x_1182_ = l_Lean_instHashableMVarId_hash(v_k_1180_);
v_h_1183_ = lean_uint64_to_usize(v___x_1182_);
v___x_1184_ = ((size_t)5ULL);
v___x_1185_ = lean_unsigned_to_nat(1u);
v___x_1186_ = ((size_t)1ULL);
v___x_1187_ = lean_usize_sub(v_depth_1173_, v___x_1186_);
v___x_1188_ = lean_usize_mul(v___x_1184_, v___x_1187_);
v_h_1189_ = lean_usize_shift_right(v_h_1183_, v___x_1188_);
v___x_1190_ = lean_nat_add(v_i_1176_, v___x_1185_);
lean_dec(v_i_1176_);
lean_inc(v_v_1181_);
lean_inc(v_k_1180_);
v___x_1191_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg(v_entries_1177_, v_h_1189_, v_depth_1173_, v_k_1180_, v_v_1181_);
v_i_1176_ = v___x_1190_;
v_entries_1177_ = v___x_1191_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___redArg___boxed(lean_object* v_depth_1193_, lean_object* v_keys_1194_, lean_object* v_vals_1195_, lean_object* v_i_1196_, lean_object* v_entries_1197_){
_start:
{
size_t v_depth_boxed_1198_; lean_object* v_res_1199_; 
v_depth_boxed_1198_ = lean_unbox_usize(v_depth_1193_);
lean_dec(v_depth_1193_);
v_res_1199_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___redArg(v_depth_boxed_1198_, v_keys_1194_, v_vals_1195_, v_i_1196_, v_entries_1197_);
lean_dec_ref(v_vals_1195_);
lean_dec_ref(v_keys_1194_);
return v_res_1199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg___boxed(lean_object* v_x_1200_, lean_object* v_x_1201_, lean_object* v_x_1202_, lean_object* v_x_1203_, lean_object* v_x_1204_){
_start:
{
size_t v_x_4151__boxed_1205_; size_t v_x_4152__boxed_1206_; lean_object* v_res_1207_; 
v_x_4151__boxed_1205_ = lean_unbox_usize(v_x_1201_);
lean_dec(v_x_1201_);
v_x_4152__boxed_1206_ = lean_unbox_usize(v_x_1202_);
lean_dec(v_x_1202_);
v_res_1207_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg(v_x_1200_, v_x_4151__boxed_1205_, v_x_4152__boxed_1206_, v_x_1203_, v_x_1204_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1___redArg(lean_object* v_x_1208_, lean_object* v_x_1209_, lean_object* v_x_1210_){
_start:
{
uint64_t v___x_1211_; size_t v___x_1212_; size_t v___x_1213_; lean_object* v___x_1214_; 
v___x_1211_ = l_Lean_instHashableMVarId_hash(v_x_1209_);
v___x_1212_ = lean_uint64_to_usize(v___x_1211_);
v___x_1213_ = ((size_t)1ULL);
v___x_1214_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg(v_x_1208_, v___x_1212_, v___x_1213_, v_x_1209_, v_x_1210_);
return v___x_1214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg(lean_object* v_mvarId_1215_, lean_object* v_val_1216_, lean_object* v___y_1217_){
_start:
{
lean_object* v___x_1219_; lean_object* v_mctx_1220_; lean_object* v_cache_1221_; lean_object* v_zetaDeltaFVarIds_1222_; lean_object* v_postponed_1223_; lean_object* v_diag_1224_; lean_object* v___x_1226_; uint8_t v_isShared_1227_; uint8_t v_isSharedCheck_1252_; 
v___x_1219_ = lean_st_ref_take(v___y_1217_);
v_mctx_1220_ = lean_ctor_get(v___x_1219_, 0);
v_cache_1221_ = lean_ctor_get(v___x_1219_, 1);
v_zetaDeltaFVarIds_1222_ = lean_ctor_get(v___x_1219_, 2);
v_postponed_1223_ = lean_ctor_get(v___x_1219_, 3);
v_diag_1224_ = lean_ctor_get(v___x_1219_, 4);
v_isSharedCheck_1252_ = !lean_is_exclusive(v___x_1219_);
if (v_isSharedCheck_1252_ == 0)
{
v___x_1226_ = v___x_1219_;
v_isShared_1227_ = v_isSharedCheck_1252_;
goto v_resetjp_1225_;
}
else
{
lean_inc(v_diag_1224_);
lean_inc(v_postponed_1223_);
lean_inc(v_zetaDeltaFVarIds_1222_);
lean_inc(v_cache_1221_);
lean_inc(v_mctx_1220_);
lean_dec(v___x_1219_);
v___x_1226_ = lean_box(0);
v_isShared_1227_ = v_isSharedCheck_1252_;
goto v_resetjp_1225_;
}
v_resetjp_1225_:
{
lean_object* v_depth_1228_; lean_object* v_levelAssignDepth_1229_; lean_object* v_lmvarCounter_1230_; lean_object* v_mvarCounter_1231_; lean_object* v_lDecls_1232_; lean_object* v_decls_1233_; lean_object* v_userNames_1234_; lean_object* v_lAssignment_1235_; lean_object* v_eAssignment_1236_; lean_object* v_dAssignment_1237_; lean_object* v___x_1239_; uint8_t v_isShared_1240_; uint8_t v_isSharedCheck_1251_; 
v_depth_1228_ = lean_ctor_get(v_mctx_1220_, 0);
v_levelAssignDepth_1229_ = lean_ctor_get(v_mctx_1220_, 1);
v_lmvarCounter_1230_ = lean_ctor_get(v_mctx_1220_, 2);
v_mvarCounter_1231_ = lean_ctor_get(v_mctx_1220_, 3);
v_lDecls_1232_ = lean_ctor_get(v_mctx_1220_, 4);
v_decls_1233_ = lean_ctor_get(v_mctx_1220_, 5);
v_userNames_1234_ = lean_ctor_get(v_mctx_1220_, 6);
v_lAssignment_1235_ = lean_ctor_get(v_mctx_1220_, 7);
v_eAssignment_1236_ = lean_ctor_get(v_mctx_1220_, 8);
v_dAssignment_1237_ = lean_ctor_get(v_mctx_1220_, 9);
v_isSharedCheck_1251_ = !lean_is_exclusive(v_mctx_1220_);
if (v_isSharedCheck_1251_ == 0)
{
v___x_1239_ = v_mctx_1220_;
v_isShared_1240_ = v_isSharedCheck_1251_;
goto v_resetjp_1238_;
}
else
{
lean_inc(v_dAssignment_1237_);
lean_inc(v_eAssignment_1236_);
lean_inc(v_lAssignment_1235_);
lean_inc(v_userNames_1234_);
lean_inc(v_decls_1233_);
lean_inc(v_lDecls_1232_);
lean_inc(v_mvarCounter_1231_);
lean_inc(v_lmvarCounter_1230_);
lean_inc(v_levelAssignDepth_1229_);
lean_inc(v_depth_1228_);
lean_dec(v_mctx_1220_);
v___x_1239_ = lean_box(0);
v_isShared_1240_ = v_isSharedCheck_1251_;
goto v_resetjp_1238_;
}
v_resetjp_1238_:
{
lean_object* v___x_1241_; lean_object* v___x_1243_; 
v___x_1241_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1___redArg(v_eAssignment_1236_, v_mvarId_1215_, v_val_1216_);
if (v_isShared_1240_ == 0)
{
lean_ctor_set(v___x_1239_, 8, v___x_1241_);
v___x_1243_ = v___x_1239_;
goto v_reusejp_1242_;
}
else
{
lean_object* v_reuseFailAlloc_1250_; 
v_reuseFailAlloc_1250_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1250_, 0, v_depth_1228_);
lean_ctor_set(v_reuseFailAlloc_1250_, 1, v_levelAssignDepth_1229_);
lean_ctor_set(v_reuseFailAlloc_1250_, 2, v_lmvarCounter_1230_);
lean_ctor_set(v_reuseFailAlloc_1250_, 3, v_mvarCounter_1231_);
lean_ctor_set(v_reuseFailAlloc_1250_, 4, v_lDecls_1232_);
lean_ctor_set(v_reuseFailAlloc_1250_, 5, v_decls_1233_);
lean_ctor_set(v_reuseFailAlloc_1250_, 6, v_userNames_1234_);
lean_ctor_set(v_reuseFailAlloc_1250_, 7, v_lAssignment_1235_);
lean_ctor_set(v_reuseFailAlloc_1250_, 8, v___x_1241_);
lean_ctor_set(v_reuseFailAlloc_1250_, 9, v_dAssignment_1237_);
v___x_1243_ = v_reuseFailAlloc_1250_;
goto v_reusejp_1242_;
}
v_reusejp_1242_:
{
lean_object* v___x_1245_; 
if (v_isShared_1227_ == 0)
{
lean_ctor_set(v___x_1226_, 0, v___x_1243_);
v___x_1245_ = v___x_1226_;
goto v_reusejp_1244_;
}
else
{
lean_object* v_reuseFailAlloc_1249_; 
v_reuseFailAlloc_1249_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1249_, 0, v___x_1243_);
lean_ctor_set(v_reuseFailAlloc_1249_, 1, v_cache_1221_);
lean_ctor_set(v_reuseFailAlloc_1249_, 2, v_zetaDeltaFVarIds_1222_);
lean_ctor_set(v_reuseFailAlloc_1249_, 3, v_postponed_1223_);
lean_ctor_set(v_reuseFailAlloc_1249_, 4, v_diag_1224_);
v___x_1245_ = v_reuseFailAlloc_1249_;
goto v_reusejp_1244_;
}
v_reusejp_1244_:
{
lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; 
v___x_1246_ = lean_st_ref_set(v___y_1217_, v___x_1245_);
v___x_1247_ = lean_box(0);
v___x_1248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1248_, 0, v___x_1247_);
return v___x_1248_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg___boxed(lean_object* v_mvarId_1253_, lean_object* v_val_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_){
_start:
{
lean_object* v_res_1257_; 
v_res_1257_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg(v_mvarId_1253_, v_val_1254_, v___y_1255_);
lean_dec(v___y_1255_);
return v_res_1257_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; 
v___x_1261_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__2));
v___x_1262_ = lean_unsigned_to_nat(7u);
v___x_1263_ = lean_unsigned_to_nat(215u);
v___x_1264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__1));
v___x_1265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__0));
v___x_1266_ = l_mkPanicMessageWithDecl(v___x_1265_, v___x_1264_, v___x_1263_, v___x_1262_, v___x_1261_);
return v___x_1266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0(uint8_t v___x_1273_, lean_object* v___x_1274_, lean_object* v_cases_1275_, lean_object* v_g_1276_, lean_object* v_z2_1277_, lean_object* v_m_1278_, lean_object* v_e_1279_, lean_object* v_e2_1280_, lean_object* v_p2_1281_, lean_object* v_z1_1282_, lean_object* v_e1_1283_, lean_object* v_p1_1284_, lean_object* v___x_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_){
_start:
{
if (v___x_1273_ == 0)
{
lean_object* v___x_1291_; uint8_t v___x_1292_; 
v___x_1291_ = lean_unsigned_to_nat(0u);
v___x_1292_ = lean_nat_dec_lt(v___x_1291_, v___x_1274_);
lean_dec(v___x_1274_);
if (v___x_1292_ == 0)
{
lean_object* v___x_1293_; lean_object* v___x_1294_; 
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
v___x_1293_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__3);
v___x_1294_ = lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__0(v___x_1293_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
return v___x_1294_;
}
else
{
lean_object* v___x_1295_; lean_object* v_rhs_1296_; lean_object* v_goal_1297_; lean_object* v___y_1299_; lean_object* v_pf_u2082_1300_; lean_object* v___y_1301_; lean_object* v___y_1302_; lean_object* v___y_1303_; lean_object* v___y_1304_; lean_object* v_pf_u2081_1324_; lean_object* v___y_1325_; lean_object* v___y_1326_; lean_object* v___y_1327_; lean_object* v___y_1328_; 
v___x_1295_ = l_Subarray_get___redArg(v_cases_1275_, v___x_1291_);
lean_dec_ref(v_cases_1275_);
v_rhs_1296_ = lean_ctor_get(v___x_1295_, 0);
lean_inc_ref(v_rhs_1296_);
v_goal_1297_ = lean_ctor_get(v___x_1295_, 2);
lean_inc(v_goal_1297_);
lean_dec(v___x_1295_);
if (lean_obj_tag(v_z1_1282_) == 0)
{
lean_object* v_roundUp_1340_; lean_object* v___x_1341_; 
lean_dec_ref_known(v_z1_1282_, 1);
v_roundUp_1340_ = lean_ctor_get(v_m_1278_, 4);
lean_inc_ref(v_roundUp_1340_);
lean_inc(v___y_1289_);
lean_inc_ref(v___y_1288_);
lean_inc(v___y_1287_);
lean_inc_ref(v___y_1286_);
lean_inc_ref(v_rhs_1296_);
lean_inc_ref(v_e_1279_);
v___x_1341_ = lean_apply_9(v_roundUp_1340_, v_e1_1283_, v_e_1279_, v_rhs_1296_, v_p1_1284_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_, lean_box(0));
if (lean_obj_tag(v___x_1341_) == 0)
{
lean_object* v_a_1342_; 
v_a_1342_ = lean_ctor_get(v___x_1341_, 0);
lean_inc(v_a_1342_);
lean_dec_ref_known(v___x_1341_, 1);
v_pf_u2081_1324_ = v_a_1342_;
v___y_1325_ = v___y_1286_;
v___y_1326_ = v___y_1287_;
v___y_1327_ = v___y_1288_;
v___y_1328_ = v___y_1289_;
goto v___jp_1323_;
}
else
{
lean_object* v_a_1343_; lean_object* v___x_1345_; uint8_t v_isShared_1346_; uint8_t v_isSharedCheck_1350_; 
lean_dec(v_goal_1297_);
lean_dec_ref(v_rhs_1296_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
v_a_1343_ = lean_ctor_get(v___x_1341_, 0);
v_isSharedCheck_1350_ = !lean_is_exclusive(v___x_1341_);
if (v_isSharedCheck_1350_ == 0)
{
v___x_1345_ = v___x_1341_;
v_isShared_1346_ = v_isSharedCheck_1350_;
goto v_resetjp_1344_;
}
else
{
lean_inc(v_a_1343_);
lean_dec(v___x_1341_);
v___x_1345_ = lean_box(0);
v_isShared_1346_ = v_isSharedCheck_1350_;
goto v_resetjp_1344_;
}
v_resetjp_1344_:
{
lean_object* v___x_1348_; 
if (v_isShared_1346_ == 0)
{
v___x_1348_ = v___x_1345_;
goto v_reusejp_1347_;
}
else
{
lean_object* v_reuseFailAlloc_1349_; 
v_reuseFailAlloc_1349_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1349_, 0, v_a_1343_);
v___x_1348_ = v_reuseFailAlloc_1349_;
goto v_reusejp_1347_;
}
v_reusejp_1347_:
{
return v___x_1348_;
}
}
}
}
else
{
lean_dec_ref_known(v_z1_1282_, 1);
lean_dec_ref(v_e1_1283_);
v_pf_u2081_1324_ = v_p1_1284_;
v___y_1325_ = v___y_1286_;
v___y_1326_ = v___y_1287_;
v___y_1327_ = v___y_1288_;
v___y_1328_ = v___y_1289_;
goto v___jp_1323_;
}
v___jp_1298_:
{
lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; 
v___x_1305_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__5));
v___x_1306_ = lean_unsigned_to_nat(2u);
v___x_1307_ = lean_mk_empty_array_with_capacity(v___x_1306_);
v___x_1308_ = lean_array_push(v___x_1307_, v_pf_u2082_1300_);
v___x_1309_ = lean_array_push(v___x_1308_, v___y_1299_);
v___x_1310_ = l_Lean_Meta_mkAppM(v___x_1305_, v___x_1309_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
if (lean_obj_tag(v___x_1310_) == 0)
{
lean_object* v_a_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; 
v_a_1311_ = lean_ctor_get(v___x_1310_, 0);
lean_inc(v_a_1311_);
lean_dec_ref_known(v___x_1310_, 1);
v___x_1312_ = l_Lean_Expr_mvar___override(v_goal_1297_);
v___x_1313_ = l_Lean_Expr_app___override(v___x_1312_, v_a_1311_);
v___x_1314_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg(v_g_1276_, v___x_1313_, v___y_1302_);
return v___x_1314_;
}
else
{
lean_object* v_a_1315_; lean_object* v___x_1317_; uint8_t v_isShared_1318_; uint8_t v_isSharedCheck_1322_; 
lean_dec(v_goal_1297_);
lean_dec(v_g_1276_);
v_a_1315_ = lean_ctor_get(v___x_1310_, 0);
v_isSharedCheck_1322_ = !lean_is_exclusive(v___x_1310_);
if (v_isSharedCheck_1322_ == 0)
{
v___x_1317_ = v___x_1310_;
v_isShared_1318_ = v_isSharedCheck_1322_;
goto v_resetjp_1316_;
}
else
{
lean_inc(v_a_1315_);
lean_dec(v___x_1310_);
v___x_1317_ = lean_box(0);
v_isShared_1318_ = v_isSharedCheck_1322_;
goto v_resetjp_1316_;
}
v_resetjp_1316_:
{
lean_object* v___x_1320_; 
if (v_isShared_1318_ == 0)
{
v___x_1320_ = v___x_1317_;
goto v_reusejp_1319_;
}
else
{
lean_object* v_reuseFailAlloc_1321_; 
v_reuseFailAlloc_1321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1321_, 0, v_a_1315_);
v___x_1320_ = v_reuseFailAlloc_1321_;
goto v_reusejp_1319_;
}
v_reusejp_1319_:
{
return v___x_1320_;
}
}
}
}
v___jp_1323_:
{
if (lean_obj_tag(v_z2_1277_) == 0)
{
lean_object* v_roundDown_1329_; lean_object* v___x_1330_; 
lean_dec_ref_known(v_z2_1277_, 1);
v_roundDown_1329_ = lean_ctor_get(v_m_1278_, 5);
lean_inc_ref(v_roundDown_1329_);
lean_dec_ref(v_m_1278_);
lean_inc(v___y_1328_);
lean_inc_ref(v___y_1327_);
lean_inc(v___y_1326_);
lean_inc_ref(v___y_1325_);
v___x_1330_ = lean_apply_9(v_roundDown_1329_, v_e_1279_, v_e2_1280_, v_rhs_1296_, v_p2_1281_, v___y_1325_, v___y_1326_, v___y_1327_, v___y_1328_, lean_box(0));
if (lean_obj_tag(v___x_1330_) == 0)
{
lean_object* v_a_1331_; 
v_a_1331_ = lean_ctor_get(v___x_1330_, 0);
lean_inc(v_a_1331_);
lean_dec_ref_known(v___x_1330_, 1);
v___y_1299_ = v_pf_u2081_1324_;
v_pf_u2082_1300_ = v_a_1331_;
v___y_1301_ = v___y_1325_;
v___y_1302_ = v___y_1326_;
v___y_1303_ = v___y_1327_;
v___y_1304_ = v___y_1328_;
goto v___jp_1298_;
}
else
{
lean_object* v_a_1332_; lean_object* v___x_1334_; uint8_t v_isShared_1335_; uint8_t v_isSharedCheck_1339_; 
lean_dec_ref(v_pf_u2081_1324_);
lean_dec(v_goal_1297_);
lean_dec(v_g_1276_);
v_a_1332_ = lean_ctor_get(v___x_1330_, 0);
v_isSharedCheck_1339_ = !lean_is_exclusive(v___x_1330_);
if (v_isSharedCheck_1339_ == 0)
{
v___x_1334_ = v___x_1330_;
v_isShared_1335_ = v_isSharedCheck_1339_;
goto v_resetjp_1333_;
}
else
{
lean_inc(v_a_1332_);
lean_dec(v___x_1330_);
v___x_1334_ = lean_box(0);
v_isShared_1335_ = v_isSharedCheck_1339_;
goto v_resetjp_1333_;
}
v_resetjp_1333_:
{
lean_object* v___x_1337_; 
if (v_isShared_1335_ == 0)
{
v___x_1337_ = v___x_1334_;
goto v_reusejp_1336_;
}
else
{
lean_object* v_reuseFailAlloc_1338_; 
v_reuseFailAlloc_1338_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1338_, 0, v_a_1332_);
v___x_1337_ = v_reuseFailAlloc_1338_;
goto v_reusejp_1336_;
}
v_reusejp_1336_:
{
return v___x_1337_;
}
}
}
}
else
{
lean_dec_ref_known(v_z2_1277_, 1);
lean_dec_ref(v_rhs_1296_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
v___y_1299_ = v_pf_u2081_1324_;
v_pf_u2082_1300_ = v_p2_1281_;
v___y_1301_ = v___y_1325_;
v___y_1302_ = v___y_1326_;
v___y_1303_ = v___y_1327_;
v___y_1304_ = v___y_1328_;
goto v___jp_1298_;
}
}
}
}
else
{
lean_object* v___x_1351_; 
lean_inc(v_g_1276_);
v___x_1351_ = l_Lean_MVarId_getType(v_g_1276_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1351_) == 0)
{
lean_object* v_a_1352_; lean_object* v_mkNumeral_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; 
v_a_1352_ = lean_ctor_get(v___x_1351_, 0);
lean_inc(v_a_1352_);
lean_dec_ref_known(v___x_1351_, 1);
v_mkNumeral_1353_ = lean_ctor_get(v_m_1278_, 7);
v___x_1354_ = lean_nat_shiftr(v___x_1274_, v___x_1285_);
v___x_1355_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower(v_z1_1282_);
lean_inc(v___x_1354_);
v___x_1356_ = lean_nat_to_int(v___x_1354_);
v___x_1357_ = lean_int_add(v___x_1355_, v___x_1356_);
lean_dec(v___x_1356_);
lean_dec(v___x_1355_);
lean_inc_ref(v_mkNumeral_1353_);
lean_inc(v___y_1289_);
lean_inc_ref(v___y_1288_);
lean_inc(v___y_1287_);
lean_inc_ref(v___y_1286_);
lean_inc(v___x_1357_);
v___x_1358_ = lean_apply_6(v_mkNumeral_1353_, v___x_1357_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_, lean_box(0));
if (lean_obj_tag(v___x_1358_) == 0)
{
lean_object* v_a_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; 
v_a_1359_ = lean_ctor_get(v___x_1358_, 0);
lean_inc_n(v_a_1359_, 2);
lean_dec_ref_known(v___x_1358_, 1);
v___x_1360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7));
v___x_1361_ = lean_unsigned_to_nat(2u);
v___x_1362_ = lean_mk_empty_array_with_capacity(v___x_1361_);
v___x_1363_ = lean_array_push(v___x_1362_, v_a_1359_);
lean_inc_ref(v_e_1279_);
v___x_1364_ = lean_array_push(v___x_1363_, v_e_1279_);
v___x_1365_ = l_Lean_Meta_mkAppM(v___x_1360_, v___x_1364_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1365_) == 0)
{
lean_object* v_a_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; 
v_a_1366_ = lean_ctor_get(v___x_1365_, 0);
lean_inc_n(v_a_1366_, 2);
lean_dec_ref_known(v___x_1365_, 1);
v___x_1367_ = l_Lean_mkNot(v_a_1366_);
lean_inc(v_a_1352_);
v___x_1368_ = l_Lean_mkArrow(v___x_1367_, v_a_1352_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1368_) == 0)
{
lean_object* v_a_1369_; lean_object* v___x_1370_; uint8_t v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; 
v_a_1369_ = lean_ctor_get(v___x_1368_, 0);
lean_inc(v_a_1369_);
lean_dec_ref_known(v___x_1368_, 1);
v___x_1370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1370_, 0, v_a_1369_);
v___x_1371_ = 2;
v___x_1372_ = lean_box(0);
v___x_1373_ = l_Lean_Meta_mkFreshExprMVar(v___x_1370_, v___x_1371_, v___x_1372_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1373_) == 0)
{
lean_object* v_a_1374_; lean_object* v___x_1375_; 
v_a_1374_ = lean_ctor_get(v___x_1373_, 0);
lean_inc(v_a_1374_);
lean_dec_ref_known(v___x_1373_, 1);
lean_inc(v_a_1366_);
v___x_1375_ = l_Lean_mkArrow(v_a_1366_, v_a_1352_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1375_) == 0)
{
lean_object* v_a_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; 
v_a_1376_ = lean_ctor_get(v___x_1375_, 0);
lean_inc(v_a_1376_);
lean_dec_ref_known(v___x_1375_, 1);
v___x_1377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1377_, 0, v_a_1376_);
v___x_1378_ = l_Lean_Meta_mkFreshExprMVar(v___x_1377_, v___x_1371_, v___x_1372_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1378_) == 0)
{
lean_object* v_a_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; 
v_a_1379_ = lean_ctor_get(v___x_1378_, 0);
lean_inc_n(v_a_1379_, 2);
lean_dec_ref_known(v___x_1378_, 1);
v___x_1380_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___closed__7));
v___x_1381_ = lean_unsigned_to_nat(3u);
v___x_1382_ = lean_mk_empty_array_with_capacity(v___x_1381_);
v___x_1383_ = lean_array_push(v___x_1382_, v_a_1366_);
v___x_1384_ = lean_array_push(v___x_1383_, v_a_1379_);
lean_inc(v_a_1374_);
v___x_1385_ = lean_array_push(v___x_1384_, v_a_1374_);
v___x_1386_ = l_Lean_Meta_mkAppM(v___x_1380_, v___x_1385_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1386_) == 0)
{
lean_object* v_a_1387_; lean_object* v___x_1388_; 
v_a_1387_ = lean_ctor_get(v___x_1386_, 0);
lean_inc(v_a_1387_);
lean_dec_ref_known(v___x_1386_, 1);
v___x_1388_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg(v_g_1276_, v_a_1387_, v___y_1287_);
if (lean_obj_tag(v___x_1388_) == 0)
{
lean_object* v___x_1390_; uint8_t v_isShared_1391_; uint8_t v_isSharedCheck_1438_; 
v_isSharedCheck_1438_ = !lean_is_exclusive(v___x_1388_);
if (v_isSharedCheck_1438_ == 0)
{
lean_object* v_unused_1439_; 
v_unused_1439_ = lean_ctor_get(v___x_1388_, 0);
lean_dec(v_unused_1439_);
v___x_1390_ = v___x_1388_;
v_isShared_1391_ = v_isSharedCheck_1438_;
goto v_resetjp_1389_;
}
else
{
lean_dec(v___x_1388_);
v___x_1390_ = lean_box(0);
v_isShared_1391_ = v_isSharedCheck_1438_;
goto v_resetjp_1389_;
}
v_resetjp_1389_:
{
lean_object* v___x_1392_; uint8_t v___x_1393_; lean_object* v___x_1394_; 
v___x_1392_ = l_Lean_Expr_mvarId_x21(v_a_1374_);
lean_dec(v_a_1374_);
v___x_1393_ = 0;
v___x_1394_ = l_Lean_Meta_intro1Core(v___x_1392_, v___x_1393_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1394_) == 0)
{
lean_object* v_a_1395_; lean_object* v_fst_1396_; lean_object* v_snd_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1402_; 
v_a_1395_ = lean_ctor_get(v___x_1394_, 0);
lean_inc(v_a_1395_);
lean_dec_ref_known(v___x_1394_, 1);
v_fst_1396_ = lean_ctor_get(v_a_1395_, 0);
lean_inc(v_fst_1396_);
v_snd_1397_ = lean_ctor_get(v_a_1395_, 1);
lean_inc(v_snd_1397_);
lean_dec(v_a_1395_);
v___x_1398_ = l_Subarray_copy___redArg(v_cases_1275_);
v___x_1399_ = lean_unsigned_to_nat(0u);
lean_inc(v___x_1354_);
lean_inc_ref(v___x_1398_);
v___x_1400_ = l_Array_toSubarray___redArg(v___x_1398_, v___x_1399_, v___x_1354_);
lean_inc(v___x_1357_);
if (v_isShared_1391_ == 0)
{
lean_ctor_set(v___x_1390_, 0, v___x_1357_);
v___x_1402_ = v___x_1390_;
goto v_reusejp_1401_;
}
else
{
lean_object* v_reuseFailAlloc_1429_; 
v_reuseFailAlloc_1429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1429_, 0, v___x_1357_);
v___x_1402_ = v_reuseFailAlloc_1429_;
goto v_reusejp_1401_;
}
v_reusejp_1401_:
{
lean_object* v___x_1403_; lean_object* v___x_1404_; 
v___x_1403_ = l_Lean_Expr_fvar___override(v_fst_1396_);
lean_inc_ref(v_e_1279_);
lean_inc(v_a_1359_);
lean_inc_ref(v_m_1278_);
v___x_1404_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect(v_m_1278_, v_snd_1397_, v___x_1400_, v_z1_1282_, v___x_1402_, v_e1_1283_, v_a_1359_, v_p1_1284_, v___x_1403_, v_e_1279_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1404_) == 0)
{
lean_object* v___x_1406_; uint8_t v_isShared_1407_; uint8_t v_isSharedCheck_1427_; 
v_isSharedCheck_1427_ = !lean_is_exclusive(v___x_1404_);
if (v_isSharedCheck_1427_ == 0)
{
lean_object* v_unused_1428_; 
v_unused_1428_ = lean_ctor_get(v___x_1404_, 0);
lean_dec(v_unused_1428_);
v___x_1406_ = v___x_1404_;
v_isShared_1407_ = v_isSharedCheck_1427_;
goto v_resetjp_1405_;
}
else
{
lean_dec(v___x_1404_);
v___x_1406_ = lean_box(0);
v_isShared_1407_ = v_isSharedCheck_1427_;
goto v_resetjp_1405_;
}
v_resetjp_1405_:
{
lean_object* v___x_1408_; lean_object* v___x_1409_; 
v___x_1408_ = l_Lean_Expr_mvarId_x21(v_a_1379_);
lean_dec(v_a_1379_);
v___x_1409_ = l_Lean_Meta_intro1Core(v___x_1408_, v___x_1393_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
if (lean_obj_tag(v___x_1409_) == 0)
{
lean_object* v_a_1410_; lean_object* v_fst_1411_; lean_object* v_snd_1412_; lean_object* v___x_1413_; lean_object* v___x_1415_; 
v_a_1410_ = lean_ctor_get(v___x_1409_, 0);
lean_inc(v_a_1410_);
lean_dec_ref_known(v___x_1409_, 1);
v_fst_1411_ = lean_ctor_get(v_a_1410_, 0);
lean_inc(v_fst_1411_);
v_snd_1412_ = lean_ctor_get(v_a_1410_, 1);
lean_inc(v_snd_1412_);
lean_dec(v_a_1410_);
v___x_1413_ = l_Array_toSubarray___redArg(v___x_1398_, v___x_1354_, v___x_1274_);
if (v_isShared_1407_ == 0)
{
lean_ctor_set_tag(v___x_1406_, 1);
lean_ctor_set(v___x_1406_, 0, v___x_1357_);
v___x_1415_ = v___x_1406_;
goto v_reusejp_1414_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v___x_1357_);
v___x_1415_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1414_;
}
v_reusejp_1414_:
{
lean_object* v___x_1416_; lean_object* v___x_1417_; 
v___x_1416_ = l_Lean_Expr_fvar___override(v_fst_1411_);
v___x_1417_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect(v_m_1278_, v_snd_1412_, v___x_1413_, v___x_1415_, v_z2_1277_, v_a_1359_, v_e2_1280_, v___x_1416_, v_p2_1281_, v_e_1279_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_);
return v___x_1417_;
}
}
else
{
lean_object* v_a_1419_; lean_object* v___x_1421_; uint8_t v_isShared_1422_; uint8_t v_isSharedCheck_1426_; 
lean_del_object(v___x_1406_);
lean_dec_ref(v___x_1398_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v___x_1274_);
v_a_1419_ = lean_ctor_get(v___x_1409_, 0);
v_isSharedCheck_1426_ = !lean_is_exclusive(v___x_1409_);
if (v_isSharedCheck_1426_ == 0)
{
v___x_1421_ = v___x_1409_;
v_isShared_1422_ = v_isSharedCheck_1426_;
goto v_resetjp_1420_;
}
else
{
lean_inc(v_a_1419_);
lean_dec(v___x_1409_);
v___x_1421_ = lean_box(0);
v_isShared_1422_ = v_isSharedCheck_1426_;
goto v_resetjp_1420_;
}
v_resetjp_1420_:
{
lean_object* v___x_1424_; 
if (v_isShared_1422_ == 0)
{
v___x_1424_ = v___x_1421_;
goto v_reusejp_1423_;
}
else
{
lean_object* v_reuseFailAlloc_1425_; 
v_reuseFailAlloc_1425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1425_, 0, v_a_1419_);
v___x_1424_ = v_reuseFailAlloc_1425_;
goto v_reusejp_1423_;
}
v_reusejp_1423_:
{
return v___x_1424_;
}
}
}
}
}
else
{
lean_dec_ref(v___x_1398_);
lean_dec(v_a_1379_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v___x_1274_);
return v___x_1404_;
}
}
}
else
{
lean_object* v_a_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1437_; 
lean_del_object(v___x_1390_);
lean_dec(v_a_1379_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1430_ = lean_ctor_get(v___x_1394_, 0);
v_isSharedCheck_1437_ = !lean_is_exclusive(v___x_1394_);
if (v_isSharedCheck_1437_ == 0)
{
v___x_1432_ = v___x_1394_;
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_a_1430_);
lean_dec(v___x_1394_);
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
}
else
{
lean_dec(v_a_1379_);
lean_dec(v_a_1374_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
return v___x_1388_;
}
}
else
{
lean_object* v_a_1440_; lean_object* v___x_1442_; uint8_t v_isShared_1443_; uint8_t v_isSharedCheck_1447_; 
lean_dec(v_a_1379_);
lean_dec(v_a_1374_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1440_ = lean_ctor_get(v___x_1386_, 0);
v_isSharedCheck_1447_ = !lean_is_exclusive(v___x_1386_);
if (v_isSharedCheck_1447_ == 0)
{
v___x_1442_ = v___x_1386_;
v_isShared_1443_ = v_isSharedCheck_1447_;
goto v_resetjp_1441_;
}
else
{
lean_inc(v_a_1440_);
lean_dec(v___x_1386_);
v___x_1442_ = lean_box(0);
v_isShared_1443_ = v_isSharedCheck_1447_;
goto v_resetjp_1441_;
}
v_resetjp_1441_:
{
lean_object* v___x_1445_; 
if (v_isShared_1443_ == 0)
{
v___x_1445_ = v___x_1442_;
goto v_reusejp_1444_;
}
else
{
lean_object* v_reuseFailAlloc_1446_; 
v_reuseFailAlloc_1446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1446_, 0, v_a_1440_);
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
else
{
lean_object* v_a_1448_; lean_object* v___x_1450_; uint8_t v_isShared_1451_; uint8_t v_isSharedCheck_1455_; 
lean_dec(v_a_1374_);
lean_dec(v_a_1366_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1448_ = lean_ctor_get(v___x_1378_, 0);
v_isSharedCheck_1455_ = !lean_is_exclusive(v___x_1378_);
if (v_isSharedCheck_1455_ == 0)
{
v___x_1450_ = v___x_1378_;
v_isShared_1451_ = v_isSharedCheck_1455_;
goto v_resetjp_1449_;
}
else
{
lean_inc(v_a_1448_);
lean_dec(v___x_1378_);
v___x_1450_ = lean_box(0);
v_isShared_1451_ = v_isSharedCheck_1455_;
goto v_resetjp_1449_;
}
v_resetjp_1449_:
{
lean_object* v___x_1453_; 
if (v_isShared_1451_ == 0)
{
v___x_1453_ = v___x_1450_;
goto v_reusejp_1452_;
}
else
{
lean_object* v_reuseFailAlloc_1454_; 
v_reuseFailAlloc_1454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1454_, 0, v_a_1448_);
v___x_1453_ = v_reuseFailAlloc_1454_;
goto v_reusejp_1452_;
}
v_reusejp_1452_:
{
return v___x_1453_;
}
}
}
}
else
{
lean_object* v_a_1456_; lean_object* v___x_1458_; uint8_t v_isShared_1459_; uint8_t v_isSharedCheck_1463_; 
lean_dec(v_a_1374_);
lean_dec(v_a_1366_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1456_ = lean_ctor_get(v___x_1375_, 0);
v_isSharedCheck_1463_ = !lean_is_exclusive(v___x_1375_);
if (v_isSharedCheck_1463_ == 0)
{
v___x_1458_ = v___x_1375_;
v_isShared_1459_ = v_isSharedCheck_1463_;
goto v_resetjp_1457_;
}
else
{
lean_inc(v_a_1456_);
lean_dec(v___x_1375_);
v___x_1458_ = lean_box(0);
v_isShared_1459_ = v_isSharedCheck_1463_;
goto v_resetjp_1457_;
}
v_resetjp_1457_:
{
lean_object* v___x_1461_; 
if (v_isShared_1459_ == 0)
{
v___x_1461_ = v___x_1458_;
goto v_reusejp_1460_;
}
else
{
lean_object* v_reuseFailAlloc_1462_; 
v_reuseFailAlloc_1462_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1462_, 0, v_a_1456_);
v___x_1461_ = v_reuseFailAlloc_1462_;
goto v_reusejp_1460_;
}
v_reusejp_1460_:
{
return v___x_1461_;
}
}
}
}
else
{
lean_object* v_a_1464_; lean_object* v___x_1466_; uint8_t v_isShared_1467_; uint8_t v_isSharedCheck_1471_; 
lean_dec(v_a_1366_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec(v_a_1352_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1464_ = lean_ctor_get(v___x_1373_, 0);
v_isSharedCheck_1471_ = !lean_is_exclusive(v___x_1373_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1466_ = v___x_1373_;
v_isShared_1467_ = v_isSharedCheck_1471_;
goto v_resetjp_1465_;
}
else
{
lean_inc(v_a_1464_);
lean_dec(v___x_1373_);
v___x_1466_ = lean_box(0);
v_isShared_1467_ = v_isSharedCheck_1471_;
goto v_resetjp_1465_;
}
v_resetjp_1465_:
{
lean_object* v___x_1469_; 
if (v_isShared_1467_ == 0)
{
v___x_1469_ = v___x_1466_;
goto v_reusejp_1468_;
}
else
{
lean_object* v_reuseFailAlloc_1470_; 
v_reuseFailAlloc_1470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1470_, 0, v_a_1464_);
v___x_1469_ = v_reuseFailAlloc_1470_;
goto v_reusejp_1468_;
}
v_reusejp_1468_:
{
return v___x_1469_;
}
}
}
}
else
{
lean_object* v_a_1472_; lean_object* v___x_1474_; uint8_t v_isShared_1475_; uint8_t v_isSharedCheck_1479_; 
lean_dec(v_a_1366_);
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec(v_a_1352_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1472_ = lean_ctor_get(v___x_1368_, 0);
v_isSharedCheck_1479_ = !lean_is_exclusive(v___x_1368_);
if (v_isSharedCheck_1479_ == 0)
{
v___x_1474_ = v___x_1368_;
v_isShared_1475_ = v_isSharedCheck_1479_;
goto v_resetjp_1473_;
}
else
{
lean_inc(v_a_1472_);
lean_dec(v___x_1368_);
v___x_1474_ = lean_box(0);
v_isShared_1475_ = v_isSharedCheck_1479_;
goto v_resetjp_1473_;
}
v_resetjp_1473_:
{
lean_object* v___x_1477_; 
if (v_isShared_1475_ == 0)
{
v___x_1477_ = v___x_1474_;
goto v_reusejp_1476_;
}
else
{
lean_object* v_reuseFailAlloc_1478_; 
v_reuseFailAlloc_1478_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1478_, 0, v_a_1472_);
v___x_1477_ = v_reuseFailAlloc_1478_;
goto v_reusejp_1476_;
}
v_reusejp_1476_:
{
return v___x_1477_;
}
}
}
}
else
{
lean_object* v_a_1480_; lean_object* v___x_1482_; uint8_t v_isShared_1483_; uint8_t v_isSharedCheck_1487_; 
lean_dec(v_a_1359_);
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec(v_a_1352_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1480_ = lean_ctor_get(v___x_1365_, 0);
v_isSharedCheck_1487_ = !lean_is_exclusive(v___x_1365_);
if (v_isSharedCheck_1487_ == 0)
{
v___x_1482_ = v___x_1365_;
v_isShared_1483_ = v_isSharedCheck_1487_;
goto v_resetjp_1481_;
}
else
{
lean_inc(v_a_1480_);
lean_dec(v___x_1365_);
v___x_1482_ = lean_box(0);
v_isShared_1483_ = v_isSharedCheck_1487_;
goto v_resetjp_1481_;
}
v_resetjp_1481_:
{
lean_object* v___x_1485_; 
if (v_isShared_1483_ == 0)
{
v___x_1485_ = v___x_1482_;
goto v_reusejp_1484_;
}
else
{
lean_object* v_reuseFailAlloc_1486_; 
v_reuseFailAlloc_1486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1486_, 0, v_a_1480_);
v___x_1485_ = v_reuseFailAlloc_1486_;
goto v_reusejp_1484_;
}
v_reusejp_1484_:
{
return v___x_1485_;
}
}
}
}
else
{
lean_object* v_a_1488_; lean_object* v___x_1490_; uint8_t v_isShared_1491_; uint8_t v_isSharedCheck_1495_; 
lean_dec(v___x_1357_);
lean_dec(v___x_1354_);
lean_dec(v_a_1352_);
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1488_ = lean_ctor_get(v___x_1358_, 0);
v_isSharedCheck_1495_ = !lean_is_exclusive(v___x_1358_);
if (v_isSharedCheck_1495_ == 0)
{
v___x_1490_ = v___x_1358_;
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
else
{
lean_inc(v_a_1488_);
lean_dec(v___x_1358_);
v___x_1490_ = lean_box(0);
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
v_resetjp_1489_:
{
lean_object* v___x_1493_; 
if (v_isShared_1491_ == 0)
{
v___x_1493_ = v___x_1490_;
goto v_reusejp_1492_;
}
else
{
lean_object* v_reuseFailAlloc_1494_; 
v_reuseFailAlloc_1494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1494_, 0, v_a_1488_);
v___x_1493_ = v_reuseFailAlloc_1494_;
goto v_reusejp_1492_;
}
v_reusejp_1492_:
{
return v___x_1493_;
}
}
}
}
else
{
lean_object* v_a_1496_; lean_object* v___x_1498_; uint8_t v_isShared_1499_; uint8_t v_isSharedCheck_1503_; 
lean_dec_ref(v_p1_1284_);
lean_dec_ref(v_e1_1283_);
lean_dec_ref(v_z1_1282_);
lean_dec_ref(v_p2_1281_);
lean_dec_ref(v_e2_1280_);
lean_dec_ref(v_e_1279_);
lean_dec_ref(v_m_1278_);
lean_dec_ref(v_z2_1277_);
lean_dec(v_g_1276_);
lean_dec_ref(v_cases_1275_);
lean_dec(v___x_1274_);
v_a_1496_ = lean_ctor_get(v___x_1351_, 0);
v_isSharedCheck_1503_ = !lean_is_exclusive(v___x_1351_);
if (v_isSharedCheck_1503_ == 0)
{
v___x_1498_ = v___x_1351_;
v_isShared_1499_ = v_isSharedCheck_1503_;
goto v_resetjp_1497_;
}
else
{
lean_inc(v_a_1496_);
lean_dec(v___x_1351_);
v___x_1498_ = lean_box(0);
v_isShared_1499_ = v_isSharedCheck_1503_;
goto v_resetjp_1497_;
}
v_resetjp_1497_:
{
lean_object* v___x_1501_; 
if (v_isShared_1499_ == 0)
{
v___x_1501_ = v___x_1498_;
goto v_reusejp_1500_;
}
else
{
lean_object* v_reuseFailAlloc_1502_; 
v_reuseFailAlloc_1502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1502_, 0, v_a_1496_);
v___x_1501_ = v_reuseFailAlloc_1502_;
goto v_reusejp_1500_;
}
v_reusejp_1500_:
{
return v___x_1501_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___boxed(lean_object** _args){
lean_object* v___x_1504_ = _args[0];
lean_object* v___x_1505_ = _args[1];
lean_object* v_cases_1506_ = _args[2];
lean_object* v_g_1507_ = _args[3];
lean_object* v_z2_1508_ = _args[4];
lean_object* v_m_1509_ = _args[5];
lean_object* v_e_1510_ = _args[6];
lean_object* v_e2_1511_ = _args[7];
lean_object* v_p2_1512_ = _args[8];
lean_object* v_z1_1513_ = _args[9];
lean_object* v_e1_1514_ = _args[10];
lean_object* v_p1_1515_ = _args[11];
lean_object* v___x_1516_ = _args[12];
lean_object* v___y_1517_ = _args[13];
lean_object* v___y_1518_ = _args[14];
lean_object* v___y_1519_ = _args[15];
lean_object* v___y_1520_ = _args[16];
lean_object* v___y_1521_ = _args[17];
_start:
{
uint8_t v___x_4405__boxed_1522_; lean_object* v_res_1523_; 
v___x_4405__boxed_1522_ = lean_unbox(v___x_1504_);
v_res_1523_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0(v___x_4405__boxed_1522_, v___x_1505_, v_cases_1506_, v_g_1507_, v_z2_1508_, v_m_1509_, v_e_1510_, v_e2_1511_, v_p2_1512_, v_z1_1513_, v_e1_1514_, v_p1_1515_, v___x_1516_, v___y_1517_, v___y_1518_, v___y_1519_, v___y_1520_);
lean_dec(v___y_1520_);
lean_dec_ref(v___y_1519_);
lean_dec(v___y_1518_);
lean_dec_ref(v___y_1517_);
lean_dec(v___x_1516_);
return v_res_1523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect(lean_object* v_m_1524_, lean_object* v_g_1525_, lean_object* v_cases_1526_, lean_object* v_z1_1527_, lean_object* v_z2_1528_, lean_object* v_e1_1529_, lean_object* v_e2_1530_, lean_object* v_p1_1531_, lean_object* v_p2_1532_, lean_object* v_e_1533_, lean_object* v_a_1534_, lean_object* v_a_1535_, lean_object* v_a_1536_, lean_object* v_a_1537_){
_start:
{
lean_object* v_start_1539_; lean_object* v_stop_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; uint8_t v___x_1543_; lean_object* v___x_1544_; lean_object* v___y_1545_; lean_object* v___x_1546_; 
v_start_1539_ = lean_ctor_get(v_cases_1526_, 1);
v_stop_1540_ = lean_ctor_get(v_cases_1526_, 2);
v___x_1541_ = lean_unsigned_to_nat(1u);
v___x_1542_ = lean_nat_sub(v_stop_1540_, v_start_1539_);
v___x_1543_ = lean_nat_dec_lt(v___x_1541_, v___x_1542_);
v___x_1544_ = lean_box(v___x_1543_);
lean_inc(v_g_1525_);
v___y_1545_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___lam__0___boxed), 18, 13);
lean_closure_set(v___y_1545_, 0, v___x_1544_);
lean_closure_set(v___y_1545_, 1, v___x_1542_);
lean_closure_set(v___y_1545_, 2, v_cases_1526_);
lean_closure_set(v___y_1545_, 3, v_g_1525_);
lean_closure_set(v___y_1545_, 4, v_z2_1528_);
lean_closure_set(v___y_1545_, 5, v_m_1524_);
lean_closure_set(v___y_1545_, 6, v_e_1533_);
lean_closure_set(v___y_1545_, 7, v_e2_1530_);
lean_closure_set(v___y_1545_, 8, v_p2_1532_);
lean_closure_set(v___y_1545_, 9, v_z1_1527_);
lean_closure_set(v___y_1545_, 10, v_e1_1529_);
lean_closure_set(v___y_1545_, 11, v_p1_1531_);
lean_closure_set(v___y_1545_, 12, v___x_1541_);
v___x_1546_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg(v_g_1525_, v___y_1545_, v_a_1534_, v_a_1535_, v_a_1536_, v_a_1537_);
return v___x_1546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect___boxed(lean_object* v_m_1547_, lean_object* v_g_1548_, lean_object* v_cases_1549_, lean_object* v_z1_1550_, lean_object* v_z2_1551_, lean_object* v_e1_1552_, lean_object* v_e2_1553_, lean_object* v_p1_1554_, lean_object* v_p2_1555_, lean_object* v_e_1556_, lean_object* v_a_1557_, lean_object* v_a_1558_, lean_object* v_a_1559_, lean_object* v_a_1560_, lean_object* v_a_1561_){
_start:
{
lean_object* v_res_1562_; 
v_res_1562_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect(v_m_1547_, v_g_1548_, v_cases_1549_, v_z1_1550_, v_z2_1551_, v_e1_1552_, v_e2_1553_, v_p1_1554_, v_p2_1555_, v_e_1556_, v_a_1557_, v_a_1558_, v_a_1559_, v_a_1560_);
lean_dec(v_a_1560_);
lean_dec_ref(v_a_1559_);
lean_dec(v_a_1558_);
lean_dec_ref(v_a_1557_);
return v_res_1562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1(lean_object* v_mvarId_1563_, lean_object* v_val_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_){
_start:
{
lean_object* v___x_1570_; 
v___x_1570_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg(v_mvarId_1563_, v_val_1564_, v___y_1566_);
return v___x_1570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___boxed(lean_object* v_mvarId_1571_, lean_object* v_val_1572_, lean_object* v___y_1573_, lean_object* v___y_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_){
_start:
{
lean_object* v_res_1578_; 
v_res_1578_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1(v_mvarId_1571_, v_val_1572_, v___y_1573_, v___y_1574_, v___y_1575_, v___y_1576_);
lean_dec(v___y_1576_);
lean_dec_ref(v___y_1575_);
lean_dec(v___y_1574_);
lean_dec_ref(v___y_1573_);
return v_res_1578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1(lean_object* v_00_u03b2_1579_, lean_object* v_x_1580_, lean_object* v_x_1581_, lean_object* v_x_1582_){
_start:
{
lean_object* v___x_1583_; 
v___x_1583_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1___redArg(v_x_1580_, v_x_1581_, v_x_1582_);
return v___x_1583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4(lean_object* v_00_u03b2_1584_, lean_object* v_x_1585_, size_t v_x_1586_, size_t v_x_1587_, lean_object* v_x_1588_, lean_object* v_x_1589_){
_start:
{
lean_object* v___x_1590_; 
v___x_1590_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___redArg(v_x_1585_, v_x_1586_, v_x_1587_, v_x_1588_, v_x_1589_);
return v___x_1590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4___boxed(lean_object* v_00_u03b2_1591_, lean_object* v_x_1592_, lean_object* v_x_1593_, lean_object* v_x_1594_, lean_object* v_x_1595_, lean_object* v_x_1596_){
_start:
{
size_t v_x_4902__boxed_1597_; size_t v_x_4903__boxed_1598_; lean_object* v_res_1599_; 
v_x_4902__boxed_1597_ = lean_unbox_usize(v_x_1593_);
lean_dec(v_x_1593_);
v_x_4903__boxed_1598_ = lean_unbox_usize(v_x_1594_);
lean_dec(v_x_1594_);
v_res_1599_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4(v_00_u03b2_1591_, v_x_1592_, v_x_4902__boxed_1597_, v_x_4903__boxed_1598_, v_x_1595_, v_x_1596_);
return v_res_1599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5(lean_object* v_00_u03b2_1600_, lean_object* v_n_1601_, lean_object* v_k_1602_, lean_object* v_v_1603_){
_start:
{
lean_object* v___x_1604_; 
v___x_1604_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5___redArg(v_n_1601_, v_k_1602_, v_v_1603_);
return v___x_1604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6(lean_object* v_00_u03b2_1605_, size_t v_depth_1606_, lean_object* v_keys_1607_, lean_object* v_vals_1608_, lean_object* v_heq_1609_, lean_object* v_i_1610_, lean_object* v_entries_1611_){
_start:
{
lean_object* v___x_1612_; 
v___x_1612_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___redArg(v_depth_1606_, v_keys_1607_, v_vals_1608_, v_i_1610_, v_entries_1611_);
return v___x_1612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6___boxed(lean_object* v_00_u03b2_1613_, lean_object* v_depth_1614_, lean_object* v_keys_1615_, lean_object* v_vals_1616_, lean_object* v_heq_1617_, lean_object* v_i_1618_, lean_object* v_entries_1619_){
_start:
{
size_t v_depth_boxed_1620_; lean_object* v_res_1621_; 
v_depth_boxed_1620_ = lean_unbox_usize(v_depth_1614_);
lean_dec(v_depth_1614_);
v_res_1621_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__6(v_00_u03b2_1613_, v_depth_boxed_1620_, v_keys_1615_, v_vals_1616_, v_heq_1617_, v_i_1618_, v_entries_1619_);
lean_dec_ref(v_vals_1616_);
lean_dec_ref(v_keys_1615_);
return v_res_1621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5_spec__6(lean_object* v_00_u03b2_1622_, lean_object* v_x_1623_, lean_object* v_x_1624_, lean_object* v_x_1625_, lean_object* v_x_1626_){
_start:
{
lean_object* v___x_1627_; 
v___x_1627_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1_spec__1_spec__4_spec__5_spec__6___redArg(v_x_1623_, v_x_1624_, v_x_1625_, v_x_1626_);
return v___x_1627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0(lean_object* v___x_1629_, lean_object* v_msg_1630_){
_start:
{
lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; 
v___x_1631_ = l_Int_instInhabited;
v___x_1632_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_1633_ = l_Lean_Name_mkStr1(v___x_1632_);
v___x_1634_ = lean_box(0);
v___x_1635_ = l_Lean_Expr_const___override(v___x_1633_, v___x_1634_);
v___x_1636_ = lp_Qq_Qq_instInhabitedQuoted(v___x_1635_);
lean_dec_ref(v___x_1635_);
v___x_1637_ = lp_Qq_Qq_instInhabitedQuoted(v___x_1629_);
v___x_1638_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1638_, 0, v___x_1636_);
lean_ctor_set(v___x_1638_, 1, v___x_1637_);
v___x_1639_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1639_, 0, v___x_1631_);
lean_ctor_set(v___x_1639_, 1, v___x_1638_);
v___x_1640_ = lean_panic_fn_borrowed(v___x_1639_, v_msg_1630_);
lean_dec_ref_known(v___x_1639_, 2);
return v___x_1640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___boxed(lean_object* v___x_1641_, lean_object* v_msg_1642_){
_start:
{
lean_object* v_res_1643_; 
v_res_1643_ = lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0(v___x_1641_, v_msg_1642_);
lean_dec_ref(v___x_1641_);
return v_res_1643_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0(void){
_start:
{
lean_object* v_natZero_1644_; lean_object* v_intZero_1645_; 
v_natZero_1644_ = lean_unsigned_to_nat(0u);
v_intZero_1645_ = lean_nat_to_int(v_natZero_1644_);
return v_intZero_1645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0(lean_object* v_x_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_){
_start:
{
lean_object* v_intZero_1652_; uint8_t v_isNeg_1653_; 
v_intZero_1652_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0);
v_isNeg_1653_ = lean_int_dec_lt(v_x_1646_, v_intZero_1652_);
if (v_isNeg_1653_ == 0)
{
lean_object* v_a_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; 
v_a_1654_ = lean_nat_abs(v_x_1646_);
v___x_1655_ = l_Lean_mkNatLit(v_a_1654_);
v___x_1656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1656_, 0, v___x_1655_);
return v___x_1656_;
}
else
{
lean_object* v___x_1657_; lean_object* v___x_1658_; 
v___x_1657_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9, &lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9);
v___x_1658_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_1657_, v___y_1647_, v___y_1648_, v___y_1649_, v___y_1650_);
return v___x_1658_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___boxed(lean_object* v_x_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_, lean_object* v___y_1663_, lean_object* v___y_1664_){
_start:
{
lean_object* v_res_1665_; 
v_res_1665_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0(v_x_1659_, v___y_1660_, v___y_1661_, v___y_1662_, v___y_1663_);
lean_dec(v___y_1663_);
lean_dec_ref(v___y_1662_);
lean_dec(v___y_1661_);
lean_dec_ref(v___y_1660_);
lean_dec(v_x_1659_);
return v_res_1665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1(lean_object* v_lhs_1667_, lean_object* v_x_1668_, lean_object* v_rhs_x27_1669_, lean_object* v_p_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_, lean_object* v___y_1674_){
_start:
{
lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; 
v___x_1676_ = lean_box(0);
v___x_1677_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_1678_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1___closed__0));
v___x_1679_ = l_Lean_Name_mkStr2(v___x_1677_, v___x_1678_);
v___x_1680_ = l_Lean_Expr_const___override(v___x_1679_, v___x_1676_);
v___x_1681_ = l_Lean_Expr_app___override(v___x_1680_, v_rhs_x27_1669_);
v___x_1682_ = l_Lean_Expr_app___override(v___x_1681_, v_lhs_1667_);
v___x_1683_ = l_Lean_Expr_app___override(v___x_1682_, v_p_1670_);
v___x_1684_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1684_, 0, v___x_1683_);
return v___x_1684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1___boxed(lean_object* v_lhs_1685_, lean_object* v_x_1686_, lean_object* v_rhs_x27_1687_, lean_object* v_p_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_){
_start:
{
lean_object* v_res_1694_; 
v_res_1694_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__1(v_lhs_1685_, v_x_1686_, v_rhs_x27_1687_, v_p_1688_, v___y_1689_, v___y_1690_, v___y_1691_, v___y_1692_);
lean_dec(v___y_1692_);
lean_dec_ref(v___y_1691_);
lean_dec(v___y_1690_);
lean_dec_ref(v___y_1689_);
lean_dec_ref(v_x_1686_);
return v_res_1694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2(lean_object* v_lhs_1696_, lean_object* v_rhs_1697_, lean_object* v_x_1698_, lean_object* v_p_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_){
_start:
{
lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; 
v___x_1705_ = lean_box(0);
v___x_1706_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_1707_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2___closed__0));
v___x_1708_ = l_Lean_Name_mkStr2(v___x_1706_, v___x_1707_);
v___x_1709_ = l_Lean_Expr_const___override(v___x_1708_, v___x_1705_);
v___x_1710_ = l_Lean_Expr_app___override(v___x_1709_, v_rhs_1697_);
v___x_1711_ = l_Lean_Expr_app___override(v___x_1710_, v_lhs_1696_);
v___x_1712_ = l_Lean_Expr_app___override(v___x_1711_, v_p_1699_);
v___x_1713_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1713_, 0, v___x_1712_);
return v___x_1713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2___boxed(lean_object* v_lhs_1714_, lean_object* v_rhs_1715_, lean_object* v_x_1716_, lean_object* v_p_1717_, lean_object* v___y_1718_, lean_object* v___y_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_){
_start:
{
lean_object* v_res_1723_; 
v_res_1723_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__2(v_lhs_1714_, v_rhs_1715_, v_x_1716_, v_p_1717_, v___y_1718_, v___y_1719_, v___y_1720_, v___y_1721_);
lean_dec(v___y_1721_);
lean_dec_ref(v___y_1720_);
lean_dec(v___y_1719_);
lean_dec_ref(v___y_1718_);
lean_dec_ref(v_x_1716_);
return v_res_1723_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0(void){
_start:
{
lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; 
v___x_1724_ = lean_box(0);
v___x_1725_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__1));
v___x_1726_ = l_Lean_Expr_const___override(v___x_1725_, v___x_1724_);
return v___x_1726_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2(void){
_start:
{
lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; 
v___x_1730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__1));
v___x_1731_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__7));
v___x_1732_ = l_Lean_Expr_const___override(v___x_1731_, v___x_1730_);
return v___x_1732_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5(void){
_start:
{
lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; 
v___x_1736_ = lean_box(0);
v___x_1737_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__4));
v___x_1738_ = l_Lean_Expr_const___override(v___x_1737_, v___x_1736_);
return v___x_1738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3(lean_object* v_lhs_1739_, lean_object* v_rhs_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_){
_start:
{
lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; 
v___x_1746_ = lean_box(0);
v___x_1747_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0);
v___x_1748_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2);
v___x_1749_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_1750_ = l_Lean_Name_mkStr1(v___x_1749_);
v___x_1751_ = l_Lean_Expr_const___override(v___x_1750_, v___x_1746_);
v___x_1752_ = l_Lean_Expr_app___override(v___x_1748_, v___x_1751_);
v___x_1753_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5);
v___x_1754_ = l_Lean_Expr_app___override(v___x_1752_, v___x_1753_);
v___x_1755_ = l_Lean_Expr_app___override(v___x_1754_, v_rhs_1740_);
v___x_1756_ = l_Lean_Expr_app___override(v___x_1755_, v_lhs_1739_);
v___x_1757_ = l_Lean_Expr_app___override(v___x_1747_, v___x_1756_);
v___x_1758_ = lp_mathlib_Qq_mkDecideProofQ(v___x_1757_, v___y_1741_, v___y_1742_, v___y_1743_, v___y_1744_);
return v___x_1758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___boxed(lean_object* v_lhs_1759_, lean_object* v_rhs_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_){
_start:
{
lean_object* v_res_1766_; 
v_res_1766_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3(v_lhs_1759_, v_rhs_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_);
lean_dec(v___y_1764_);
lean_dec_ref(v___y_1763_);
lean_dec(v___y_1762_);
lean_dec_ref(v___y_1761_);
return v_res_1766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__4(lean_object* v_lhs_1767_, lean_object* v_rhs_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_){
_start:
{
lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; 
v___x_1774_ = lean_box(0);
v___x_1775_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2);
v___x_1776_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_1777_ = l_Lean_Name_mkStr1(v___x_1776_);
v___x_1778_ = l_Lean_Expr_const___override(v___x_1777_, v___x_1774_);
v___x_1779_ = l_Lean_Expr_app___override(v___x_1775_, v___x_1778_);
v___x_1780_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__5);
v___x_1781_ = l_Lean_Expr_app___override(v___x_1779_, v___x_1780_);
v___x_1782_ = l_Lean_Expr_app___override(v___x_1781_, v_lhs_1767_);
v___x_1783_ = l_Lean_Expr_app___override(v___x_1782_, v_rhs_1768_);
v___x_1784_ = lp_mathlib_Qq_mkDecideProofQ(v___x_1783_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_);
return v___x_1784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__4___boxed(lean_object* v_lhs_1785_, lean_object* v_rhs_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_, lean_object* v___y_1790_, lean_object* v___y_1791_){
_start:
{
lean_object* v_res_1792_; 
v_res_1792_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__4(v_lhs_1785_, v_rhs_1786_, v___y_1787_, v___y_1788_, v___y_1789_, v___y_1790_);
lean_dec(v___y_1790_);
lean_dec_ref(v___y_1789_);
lean_dec(v___y_1788_);
lean_dec_ref(v___y_1787_);
return v_res_1792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__5(lean_object* v_e_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_){
_start:
{
lean_object* v___x_1799_; lean_object* v___x_1800_; 
v___x_1799_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9, &lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound___closed__9);
v___x_1800_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_1799_, v___y_1794_, v___y_1795_, v___y_1796_, v___y_1797_);
return v___x_1800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__5___boxed(lean_object* v_e_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_){
_start:
{
lean_object* v_res_1807_; 
v_res_1807_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__5(v_e_1801_, v___y_1802_, v___y_1803_, v___y_1804_, v___y_1805_);
lean_dec(v___y_1805_);
lean_dec_ref(v___y_1804_);
lean_dec(v___y_1803_);
lean_dec_ref(v___y_1802_);
lean_dec_ref(v_e_1801_);
return v_res_1807_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__0(void){
_start:
{
lean_object* v___x_1808_; lean_object* v___x_1809_; 
v___x_1808_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0);
v___x_1809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1809_, 0, v___x_1808_);
return v___x_1809_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4(void){
_start:
{
lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; 
v___x_1815_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__1));
v___x_1816_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__3));
v___x_1817_ = l_Lean_Expr_const___override(v___x_1816_, v___x_1815_);
return v___x_1817_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6(void){
_start:
{
lean_object* v___x_1820_; lean_object* v___x_1821_; 
v___x_1820_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__5));
v___x_1821_ = l_Lean_Expr_lit___override(v___x_1820_);
return v___x_1821_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__9(void){
_start:
{
lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; 
v___x_1825_ = lean_box(0);
v___x_1826_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__8));
v___x_1827_ = l_Lean_Expr_const___override(v___x_1826_, v___x_1825_);
return v___x_1827_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__10(void){
_start:
{
lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; 
v___x_1828_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6);
v___x_1829_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__9, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__9);
v___x_1830_ = l_Lean_Expr_app___override(v___x_1829_, v___x_1828_);
return v___x_1830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6(lean_object* v_e_1832_, lean_object* v___y_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_){
_start:
{
lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; 
v___x_1838_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__0);
v___x_1839_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_1840_ = l_Lean_Name_mkStr1(v___x_1839_);
v___x_1841_ = lean_box(0);
v___x_1842_ = l_Lean_Expr_const___override(v___x_1840_, v___x_1841_);
v___x_1843_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4);
v___x_1844_ = l_Lean_Expr_app___override(v___x_1843_, v___x_1842_);
v___x_1845_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__6);
v___x_1846_ = l_Lean_Expr_app___override(v___x_1844_, v___x_1845_);
v___x_1847_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__10, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__10);
v___x_1848_ = l_Lean_Expr_app___override(v___x_1846_, v___x_1847_);
v___x_1849_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__11));
v___x_1850_ = l_Lean_Name_mkStr2(v___x_1839_, v___x_1849_);
v___x_1851_ = l_Lean_Expr_const___override(v___x_1850_, v___x_1841_);
v___x_1852_ = l_Lean_Expr_app___override(v___x_1851_, v_e_1832_);
v___x_1853_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1853_, 0, v___x_1848_);
lean_ctor_set(v___x_1853_, 1, v___x_1852_);
v___x_1854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1854_, 0, v___x_1838_);
lean_ctor_set(v___x_1854_, 1, v___x_1853_);
v___x_1855_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1855_, 0, v___x_1854_);
return v___x_1855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___boxed(lean_object* v_e_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_){
_start:
{
lean_object* v_res_1862_; 
v_res_1862_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6(v_e_1856_, v___y_1857_, v___y_1858_, v___y_1859_, v___y_1860_);
lean_dec(v___y_1860_);
lean_dec_ref(v___y_1859_);
lean_dec(v___y_1858_);
lean_dec_ref(v___y_1857_);
return v_res_1862_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__0(void){
_start:
{
lean_object* v___x_1863_; lean_object* v___x_1864_; 
v___x_1863_ = lean_box(0);
v___x_1864_ = l_Lean_Level_succ___override(v___x_1863_);
return v___x_1864_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__3(void){
_start:
{
lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; 
v___x_1868_ = lean_box(0);
v___x_1869_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__0);
v___x_1870_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1870_, 0, v___x_1869_);
lean_ctor_set(v___x_1870_, 1, v___x_1868_);
return v___x_1870_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4(void){
_start:
{
lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; 
v___x_1871_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__3, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__3);
v___x_1872_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__2));
v___x_1873_ = l_Lean_Expr_const___override(v___x_1872_, v___x_1871_);
return v___x_1873_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7(void){
_start:
{
lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; 
v___x_1877_ = lean_box(0);
v___x_1878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__6));
v___x_1879_ = l_Lean_Expr_const___override(v___x_1878_, v___x_1877_);
return v___x_1879_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11(void){
_start:
{
lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; 
v___x_1883_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__10));
v___x_1884_ = lean_unsigned_to_nat(14u);
v___x_1885_ = lean_unsigned_to_nat(22u);
v___x_1886_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__9));
v___x_1887_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__8));
v___x_1888_ = l_mkPanicMessageWithDecl(v___x_1887_, v___x_1886_, v___x_1885_, v___x_1884_, v___x_1883_);
return v___x_1888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7(lean_object* v_e_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_){
_start:
{
lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; uint8_t v___x_1900_; lean_object* v___x_1901_; 
v___x_1895_ = lean_box(0);
v___x_1896_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_1897_ = l_Lean_Name_mkStr1(v___x_1896_);
v___x_1898_ = lean_box(0);
v___x_1899_ = l_Lean_Expr_const___override(v___x_1897_, v___x_1898_);
v___x_1900_ = 0;
lean_inc_ref(v_e_1889_);
lean_inc_ref(v___x_1899_);
v___x_1901_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_1895_, v___x_1899_, v_e_1889_, v___x_1900_, v___y_1890_, v___y_1891_, v___y_1892_, v___y_1893_);
if (lean_obj_tag(v___x_1901_) == 0)
{
lean_object* v_a_1902_; lean_object* v___x_1904_; uint8_t v_isShared_1905_; uint8_t v_isSharedCheck_1938_; 
v_a_1902_ = lean_ctor_get(v___x_1901_, 0);
v_isSharedCheck_1938_ = !lean_is_exclusive(v___x_1901_);
if (v_isSharedCheck_1938_ == 0)
{
v___x_1904_ = v___x_1901_;
v_isShared_1905_ = v_isSharedCheck_1938_;
goto v_resetjp_1903_;
}
else
{
lean_inc(v_a_1902_);
lean_dec(v___x_1901_);
v___x_1904_ = lean_box(0);
v_isShared_1905_ = v_isSharedCheck_1938_;
goto v_resetjp_1903_;
}
v_resetjp_1903_:
{
lean_object* v___y_1907_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; 
v___x_1929_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4);
lean_inc_ref(v___x_1899_);
v___x_1930_ = l_Lean_Expr_app___override(v___x_1929_, v___x_1899_);
lean_inc_ref(v_e_1889_);
v___x_1931_ = l_Lean_Expr_app___override(v___x_1930_, v_e_1889_);
v___x_1932_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRawIntEq(v___x_1895_, v___x_1899_, v_e_1889_, v_a_1902_);
if (lean_obj_tag(v___x_1932_) == 0)
{
lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; 
v___x_1933_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7);
v___x_1934_ = l_Lean_Expr_app___override(v___x_1931_, v___x_1933_);
v___x_1935_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11);
v___x_1936_ = lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0(v___x_1934_, v___x_1935_);
lean_dec_ref(v___x_1934_);
v___y_1907_ = v___x_1936_;
goto v___jp_1906_;
}
else
{
lean_object* v_val_1937_; 
lean_dec_ref(v___x_1931_);
v_val_1937_ = lean_ctor_get(v___x_1932_, 0);
lean_inc(v_val_1937_);
lean_dec_ref_known(v___x_1932_, 1);
v___y_1907_ = v_val_1937_;
goto v___jp_1906_;
}
v___jp_1906_:
{
lean_object* v_snd_1908_; lean_object* v_fst_1909_; lean_object* v___x_1911_; uint8_t v_isShared_1912_; uint8_t v_isSharedCheck_1928_; 
v_snd_1908_ = lean_ctor_get(v___y_1907_, 1);
v_fst_1909_ = lean_ctor_get(v___y_1907_, 0);
v_isSharedCheck_1928_ = !lean_is_exclusive(v___y_1907_);
if (v_isSharedCheck_1928_ == 0)
{
v___x_1911_ = v___y_1907_;
v_isShared_1912_ = v_isSharedCheck_1928_;
goto v_resetjp_1910_;
}
else
{
lean_inc(v_snd_1908_);
lean_inc(v_fst_1909_);
lean_dec(v___y_1907_);
v___x_1911_ = lean_box(0);
v_isShared_1912_ = v_isSharedCheck_1928_;
goto v_resetjp_1910_;
}
v_resetjp_1910_:
{
lean_object* v_fst_1913_; lean_object* v_snd_1914_; lean_object* v___x_1916_; uint8_t v_isShared_1917_; uint8_t v_isSharedCheck_1927_; 
v_fst_1913_ = lean_ctor_get(v_snd_1908_, 0);
v_snd_1914_ = lean_ctor_get(v_snd_1908_, 1);
v_isSharedCheck_1927_ = !lean_is_exclusive(v_snd_1908_);
if (v_isSharedCheck_1927_ == 0)
{
v___x_1916_ = v_snd_1908_;
v_isShared_1917_ = v_isSharedCheck_1927_;
goto v_resetjp_1915_;
}
else
{
lean_inc(v_snd_1914_);
lean_inc(v_fst_1913_);
lean_dec(v_snd_1908_);
v___x_1916_ = lean_box(0);
v_isShared_1917_ = v_isSharedCheck_1927_;
goto v_resetjp_1915_;
}
v_resetjp_1915_:
{
lean_object* v___x_1919_; 
if (v_isShared_1912_ == 0)
{
lean_ctor_set(v___x_1911_, 1, v_snd_1914_);
lean_ctor_set(v___x_1911_, 0, v_fst_1913_);
v___x_1919_ = v___x_1911_;
goto v_reusejp_1918_;
}
else
{
lean_object* v_reuseFailAlloc_1926_; 
v_reuseFailAlloc_1926_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1926_, 0, v_fst_1913_);
lean_ctor_set(v_reuseFailAlloc_1926_, 1, v_snd_1914_);
v___x_1919_ = v_reuseFailAlloc_1926_;
goto v_reusejp_1918_;
}
v_reusejp_1918_:
{
lean_object* v___x_1921_; 
if (v_isShared_1917_ == 0)
{
lean_ctor_set(v___x_1916_, 1, v___x_1919_);
lean_ctor_set(v___x_1916_, 0, v_fst_1909_);
v___x_1921_ = v___x_1916_;
goto v_reusejp_1920_;
}
else
{
lean_object* v_reuseFailAlloc_1925_; 
v_reuseFailAlloc_1925_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1925_, 0, v_fst_1909_);
lean_ctor_set(v_reuseFailAlloc_1925_, 1, v___x_1919_);
v___x_1921_ = v_reuseFailAlloc_1925_;
goto v_reusejp_1920_;
}
v_reusejp_1920_:
{
lean_object* v___x_1923_; 
if (v_isShared_1905_ == 0)
{
lean_ctor_set(v___x_1904_, 0, v___x_1921_);
v___x_1923_ = v___x_1904_;
goto v_reusejp_1922_;
}
else
{
lean_object* v_reuseFailAlloc_1924_; 
v_reuseFailAlloc_1924_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1924_, 0, v___x_1921_);
v___x_1923_ = v_reuseFailAlloc_1924_;
goto v_reusejp_1922_;
}
v_reusejp_1922_:
{
return v___x_1923_;
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
lean_object* v_a_1939_; lean_object* v___x_1941_; uint8_t v_isShared_1942_; uint8_t v_isSharedCheck_1946_; 
lean_dec_ref(v___x_1899_);
lean_dec_ref(v_e_1889_);
v_a_1939_ = lean_ctor_get(v___x_1901_, 0);
v_isSharedCheck_1946_ = !lean_is_exclusive(v___x_1901_);
if (v_isSharedCheck_1946_ == 0)
{
v___x_1941_ = v___x_1901_;
v_isShared_1942_ = v_isSharedCheck_1946_;
goto v_resetjp_1940_;
}
else
{
lean_inc(v_a_1939_);
lean_dec(v___x_1901_);
v___x_1941_ = lean_box(0);
v_isShared_1942_ = v_isSharedCheck_1946_;
goto v_resetjp_1940_;
}
v_resetjp_1940_:
{
lean_object* v___x_1944_; 
if (v_isShared_1942_ == 0)
{
v___x_1944_ = v___x_1941_;
goto v_reusejp_1943_;
}
else
{
lean_object* v_reuseFailAlloc_1945_; 
v_reuseFailAlloc_1945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1945_, 0, v_a_1939_);
v___x_1944_ = v_reuseFailAlloc_1945_;
goto v_reusejp_1943_;
}
v_reusejp_1943_:
{
return v___x_1944_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___boxed(lean_object* v_e_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_){
_start:
{
lean_object* v_res_1953_; 
v_res_1953_ = lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7(v_e_1947_, v___y_1948_, v___y_1949_, v___y_1950_, v___y_1951_);
lean_dec(v___y_1951_);
lean_dec_ref(v___y_1950_);
lean_dec(v___y_1949_);
lean_dec_ref(v___y_1948_);
return v_res_1953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0(lean_object* v___x_1973_, lean_object* v_msg_1974_){
_start:
{
lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; 
v___x_1975_ = l_Int_instInhabited;
v___x_1976_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_1977_ = l_Lean_Name_mkStr1(v___x_1976_);
v___x_1978_ = lean_box(0);
v___x_1979_ = l_Lean_Expr_const___override(v___x_1977_, v___x_1978_);
v___x_1980_ = lp_Qq_Qq_instInhabitedQuoted(v___x_1979_);
lean_dec_ref(v___x_1979_);
v___x_1981_ = lp_Qq_Qq_instInhabitedQuoted(v___x_1973_);
v___x_1982_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1982_, 0, v___x_1980_);
lean_ctor_set(v___x_1982_, 1, v___x_1981_);
v___x_1983_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1983_, 0, v___x_1975_);
lean_ctor_set(v___x_1983_, 1, v___x_1982_);
v___x_1984_ = lean_panic_fn_borrowed(v___x_1983_, v_msg_1974_);
lean_dec_ref_known(v___x_1983_, 2);
return v___x_1984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___boxed(lean_object* v___x_1985_, lean_object* v_msg_1986_){
_start:
{
lean_object* v_res_1987_; 
v_res_1987_ = lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0(v___x_1985_, v_msg_1986_);
lean_dec_ref(v___x_1985_);
return v_res_1987_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2(void){
_start:
{
lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; 
v___x_1991_ = lean_box(0);
v___x_1992_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__1));
v___x_1993_ = l_Lean_Expr_const___override(v___x_1992_, v___x_1991_);
return v___x_1993_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__6(void){
_start:
{
lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; 
v___x_1999_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__1));
v___x_2000_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__5));
v___x_2001_ = l_Lean_Expr_const___override(v___x_2000_, v___x_1999_);
return v___x_2001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0(lean_object* v_x_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_){
_start:
{
lean_object* v_intZero_2009_; uint8_t v_isNeg_2010_; 
v_intZero_2009_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__0___closed__0);
v_isNeg_2010_ = lean_int_dec_lt(v_x_2003_, v_intZero_2009_);
if (v_isNeg_2010_ == 0)
{
lean_object* v_a_2011_; lean_object* v_n_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; 
v_a_2011_ = lean_nat_abs(v_x_2003_);
v_n_2012_ = l_Lean_mkRawNatLit(v_a_2011_);
v___x_2013_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2014_ = l_Lean_Name_mkStr1(v___x_2013_);
v___x_2015_ = lean_box(0);
v___x_2016_ = l_Lean_Expr_const___override(v___x_2014_, v___x_2015_);
v___x_2017_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4);
v___x_2018_ = l_Lean_Expr_app___override(v___x_2017_, v___x_2016_);
lean_inc_ref(v_n_2012_);
v___x_2019_ = l_Lean_Expr_app___override(v___x_2018_, v_n_2012_);
v___x_2020_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2);
v___x_2021_ = l_Lean_Expr_app___override(v___x_2020_, v_n_2012_);
v___x_2022_ = l_Lean_Expr_app___override(v___x_2019_, v___x_2021_);
v___x_2023_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2023_, 0, v___x_2022_);
return v___x_2023_;
}
else
{
lean_object* v_abs_2024_; lean_object* v_one_2025_; lean_object* v_a_2026_; lean_object* v___x_2027_; lean_object* v_n_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; 
v_abs_2024_ = lean_nat_abs(v_x_2003_);
v_one_2025_ = lean_unsigned_to_nat(1u);
v_a_2026_ = lean_nat_sub(v_abs_2024_, v_one_2025_);
lean_dec(v_abs_2024_);
v___x_2027_ = lean_nat_add(v_a_2026_, v_one_2025_);
lean_dec(v_a_2026_);
v_n_2028_ = l_Lean_mkRawNatLit(v___x_2027_);
v___x_2029_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2030_ = l_Lean_Name_mkStr1(v___x_2029_);
v___x_2031_ = lean_box(0);
v___x_2032_ = l_Lean_Expr_const___override(v___x_2030_, v___x_2031_);
v___x_2033_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__6, &lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__6);
lean_inc_ref(v___x_2032_);
v___x_2034_ = l_Lean_Expr_app___override(v___x_2033_, v___x_2032_);
v___x_2035_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__7));
v___x_2036_ = l_Lean_Name_mkStr2(v___x_2029_, v___x_2035_);
v___x_2037_ = l_Lean_Expr_const___override(v___x_2036_, v___x_2031_);
v___x_2038_ = l_Lean_Expr_app___override(v___x_2034_, v___x_2037_);
v___x_2039_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__6___closed__4);
v___x_2040_ = l_Lean_Expr_app___override(v___x_2039_, v___x_2032_);
lean_inc_ref(v_n_2028_);
v___x_2041_ = l_Lean_Expr_app___override(v___x_2040_, v_n_2028_);
v___x_2042_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___closed__2);
v___x_2043_ = l_Lean_Expr_app___override(v___x_2042_, v_n_2028_);
v___x_2044_ = l_Lean_Expr_app___override(v___x_2041_, v___x_2043_);
v___x_2045_ = l_Lean_Expr_app___override(v___x_2038_, v___x_2044_);
v___x_2046_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2046_, 0, v___x_2045_);
return v___x_2046_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0___boxed(lean_object* v_x_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_){
_start:
{
lean_object* v_res_2053_; 
v_res_2053_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__0(v_x_2047_, v___y_2048_, v___y_2049_, v___y_2050_, v___y_2051_);
lean_dec(v___y_2051_);
lean_dec_ref(v___y_2050_);
lean_dec(v___y_2049_);
lean_dec_ref(v___y_2048_);
lean_dec(v_x_2047_);
return v_res_2053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1(lean_object* v_lhs_2055_, lean_object* v_rhs_2056_, lean_object* v_x_2057_, lean_object* v_p_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_){
_start:
{
lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; 
v___x_2064_ = lean_box(0);
v___x_2065_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1___closed__0));
v___x_2067_ = l_Lean_Name_mkStr2(v___x_2065_, v___x_2066_);
v___x_2068_ = l_Lean_Expr_const___override(v___x_2067_, v___x_2064_);
v___x_2069_ = l_Lean_Expr_app___override(v___x_2068_, v_lhs_2055_);
v___x_2070_ = l_Lean_Expr_app___override(v___x_2069_, v_rhs_2056_);
v___x_2071_ = l_Lean_Expr_app___override(v___x_2070_, v_p_2058_);
v___x_2072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2072_, 0, v___x_2071_);
return v___x_2072_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1___boxed(lean_object* v_lhs_2073_, lean_object* v_rhs_2074_, lean_object* v_x_2075_, lean_object* v_p_2076_, lean_object* v___y_2077_, lean_object* v___y_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_){
_start:
{
lean_object* v_res_2082_; 
v_res_2082_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__1(v_lhs_2073_, v_rhs_2074_, v_x_2075_, v_p_2076_, v___y_2077_, v___y_2078_, v___y_2079_, v___y_2080_);
lean_dec(v___y_2080_);
lean_dec_ref(v___y_2079_);
lean_dec(v___y_2078_);
lean_dec_ref(v___y_2077_);
lean_dec_ref(v_x_2075_);
return v_res_2082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2(lean_object* v_lhs_2084_, lean_object* v_rhs_2085_, lean_object* v_x_2086_, lean_object* v_p_2087_, lean_object* v___y_2088_, lean_object* v___y_2089_, lean_object* v___y_2090_, lean_object* v___y_2091_){
_start:
{
lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; 
v___x_2093_ = lean_box(0);
v___x_2094_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2095_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2___closed__0));
v___x_2096_ = l_Lean_Name_mkStr2(v___x_2094_, v___x_2095_);
v___x_2097_ = l_Lean_Expr_const___override(v___x_2096_, v___x_2093_);
v___x_2098_ = l_Lean_Expr_app___override(v___x_2097_, v_lhs_2084_);
v___x_2099_ = l_Lean_Expr_app___override(v___x_2098_, v_rhs_2085_);
v___x_2100_ = l_Lean_Expr_app___override(v___x_2099_, v_p_2087_);
v___x_2101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2101_, 0, v___x_2100_);
return v___x_2101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2___boxed(lean_object* v_lhs_2102_, lean_object* v_rhs_2103_, lean_object* v_x_2104_, lean_object* v_p_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_){
_start:
{
lean_object* v_res_2111_; 
v_res_2111_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__2(v_lhs_2102_, v_rhs_2103_, v_x_2104_, v_p_2105_, v___y_2106_, v___y_2107_, v___y_2108_, v___y_2109_);
lean_dec(v___y_2109_);
lean_dec_ref(v___y_2108_);
lean_dec(v___y_2107_);
lean_dec_ref(v___y_2106_);
lean_dec_ref(v_x_2104_);
return v_res_2111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3(lean_object* v_lhs_2113_, lean_object* v_rhs_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_){
_start:
{
lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; 
v___x_2120_ = lean_box(0);
v___x_2121_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__0);
v___x_2122_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2);
v___x_2123_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2124_ = l_Lean_Name_mkStr1(v___x_2123_);
v___x_2125_ = l_Lean_Expr_const___override(v___x_2124_, v___x_2120_);
v___x_2126_ = l_Lean_Expr_app___override(v___x_2122_, v___x_2125_);
v___x_2127_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___closed__0));
v___x_2128_ = l_Lean_Name_mkStr2(v___x_2123_, v___x_2127_);
v___x_2129_ = l_Lean_Expr_const___override(v___x_2128_, v___x_2120_);
v___x_2130_ = l_Lean_Expr_app___override(v___x_2126_, v___x_2129_);
v___x_2131_ = l_Lean_Expr_app___override(v___x_2130_, v_rhs_2114_);
v___x_2132_ = l_Lean_Expr_app___override(v___x_2131_, v_lhs_2113_);
v___x_2133_ = l_Lean_Expr_app___override(v___x_2121_, v___x_2132_);
v___x_2134_ = lp_mathlib_Qq_mkDecideProofQ(v___x_2133_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_);
return v___x_2134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___boxed(lean_object* v_lhs_2135_, lean_object* v_rhs_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_, lean_object* v___y_2140_, lean_object* v___y_2141_){
_start:
{
lean_object* v_res_2142_; 
v_res_2142_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3(v_lhs_2135_, v_rhs_2136_, v___y_2137_, v___y_2138_, v___y_2139_, v___y_2140_);
lean_dec(v___y_2140_);
lean_dec_ref(v___y_2139_);
lean_dec(v___y_2138_);
lean_dec_ref(v___y_2137_);
return v_res_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__4(lean_object* v_lhs_2143_, lean_object* v_rhs_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_){
_start:
{
lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; 
v___x_2150_ = lean_box(0);
v___x_2151_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__3___closed__2);
v___x_2152_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2153_ = l_Lean_Name_mkStr1(v___x_2152_);
v___x_2154_ = l_Lean_Expr_const___override(v___x_2153_, v___x_2150_);
v___x_2155_ = l_Lean_Expr_app___override(v___x_2151_, v___x_2154_);
v___x_2156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__3___closed__0));
v___x_2157_ = l_Lean_Name_mkStr2(v___x_2152_, v___x_2156_);
v___x_2158_ = l_Lean_Expr_const___override(v___x_2157_, v___x_2150_);
v___x_2159_ = l_Lean_Expr_app___override(v___x_2155_, v___x_2158_);
v___x_2160_ = l_Lean_Expr_app___override(v___x_2159_, v_lhs_2143_);
v___x_2161_ = l_Lean_Expr_app___override(v___x_2160_, v_rhs_2144_);
v___x_2162_ = lp_mathlib_Qq_mkDecideProofQ(v___x_2161_, v___y_2145_, v___y_2146_, v___y_2147_, v___y_2148_);
return v___x_2162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__4___boxed(lean_object* v_lhs_2163_, lean_object* v_rhs_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_, lean_object* v___y_2169_){
_start:
{
lean_object* v_res_2170_; 
v_res_2170_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__4(v_lhs_2163_, v_rhs_2164_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_);
lean_dec(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec(v___y_2166_);
lean_dec_ref(v___y_2165_);
return v_res_2170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__6(lean_object* v_e_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_){
_start:
{
lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; uint8_t v___x_2182_; lean_object* v___x_2183_; 
v___x_2177_ = lean_box(0);
v___x_2178_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2179_ = l_Lean_Name_mkStr1(v___x_2178_);
v___x_2180_ = lean_box(0);
v___x_2181_ = l_Lean_Expr_const___override(v___x_2179_, v___x_2180_);
v___x_2182_ = 0;
lean_inc_ref(v_e_2171_);
lean_inc_ref(v___x_2181_);
v___x_2183_ = lp_mathlib_Mathlib_Meta_NormNum_derive(v___x_2177_, v___x_2181_, v_e_2171_, v___x_2182_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_);
if (lean_obj_tag(v___x_2183_) == 0)
{
lean_object* v_a_2184_; lean_object* v___x_2186_; uint8_t v_isShared_2187_; uint8_t v_isSharedCheck_2220_; 
v_a_2184_ = lean_ctor_get(v___x_2183_, 0);
v_isSharedCheck_2220_ = !lean_is_exclusive(v___x_2183_);
if (v_isSharedCheck_2220_ == 0)
{
v___x_2186_ = v___x_2183_;
v_isShared_2187_ = v_isSharedCheck_2220_;
goto v_resetjp_2185_;
}
else
{
lean_inc(v_a_2184_);
lean_dec(v___x_2183_);
v___x_2186_ = lean_box(0);
v_isShared_2187_ = v_isSharedCheck_2220_;
goto v_resetjp_2185_;
}
v_resetjp_2185_:
{
lean_object* v___y_2189_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; 
v___x_2211_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__4);
lean_inc_ref(v___x_2181_);
v___x_2212_ = l_Lean_Expr_app___override(v___x_2211_, v___x_2181_);
lean_inc_ref(v_e_2171_);
v___x_2213_ = l_Lean_Expr_app___override(v___x_2212_, v_e_2171_);
v___x_2214_ = lp_mathlib_Mathlib_Meta_NormNum_Result_toRawIntEq(v___x_2177_, v___x_2181_, v_e_2171_, v_a_2184_);
if (lean_obj_tag(v___x_2214_) == 0)
{
lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; 
v___x_2215_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__7);
v___x_2216_ = l_Lean_Expr_app___override(v___x_2213_, v___x_2215_);
v___x_2217_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11, &lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods___lam__7___closed__11);
v___x_2218_ = lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0(v___x_2216_, v___x_2217_);
lean_dec_ref(v___x_2216_);
v___y_2189_ = v___x_2218_;
goto v___jp_2188_;
}
else
{
lean_object* v_val_2219_; 
lean_dec_ref(v___x_2213_);
v_val_2219_ = lean_ctor_get(v___x_2214_, 0);
lean_inc(v_val_2219_);
lean_dec_ref_known(v___x_2214_, 1);
v___y_2189_ = v_val_2219_;
goto v___jp_2188_;
}
v___jp_2188_:
{
lean_object* v_snd_2190_; lean_object* v_fst_2191_; lean_object* v___x_2193_; uint8_t v_isShared_2194_; uint8_t v_isSharedCheck_2210_; 
v_snd_2190_ = lean_ctor_get(v___y_2189_, 1);
v_fst_2191_ = lean_ctor_get(v___y_2189_, 0);
v_isSharedCheck_2210_ = !lean_is_exclusive(v___y_2189_);
if (v_isSharedCheck_2210_ == 0)
{
v___x_2193_ = v___y_2189_;
v_isShared_2194_ = v_isSharedCheck_2210_;
goto v_resetjp_2192_;
}
else
{
lean_inc(v_snd_2190_);
lean_inc(v_fst_2191_);
lean_dec(v___y_2189_);
v___x_2193_ = lean_box(0);
v_isShared_2194_ = v_isSharedCheck_2210_;
goto v_resetjp_2192_;
}
v_resetjp_2192_:
{
lean_object* v_fst_2195_; lean_object* v_snd_2196_; lean_object* v___x_2198_; uint8_t v_isShared_2199_; uint8_t v_isSharedCheck_2209_; 
v_fst_2195_ = lean_ctor_get(v_snd_2190_, 0);
v_snd_2196_ = lean_ctor_get(v_snd_2190_, 1);
v_isSharedCheck_2209_ = !lean_is_exclusive(v_snd_2190_);
if (v_isSharedCheck_2209_ == 0)
{
v___x_2198_ = v_snd_2190_;
v_isShared_2199_ = v_isSharedCheck_2209_;
goto v_resetjp_2197_;
}
else
{
lean_inc(v_snd_2196_);
lean_inc(v_fst_2195_);
lean_dec(v_snd_2190_);
v___x_2198_ = lean_box(0);
v_isShared_2199_ = v_isSharedCheck_2209_;
goto v_resetjp_2197_;
}
v_resetjp_2197_:
{
lean_object* v___x_2201_; 
if (v_isShared_2194_ == 0)
{
lean_ctor_set(v___x_2193_, 1, v_snd_2196_);
lean_ctor_set(v___x_2193_, 0, v_fst_2195_);
v___x_2201_ = v___x_2193_;
goto v_reusejp_2200_;
}
else
{
lean_object* v_reuseFailAlloc_2208_; 
v_reuseFailAlloc_2208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2208_, 0, v_fst_2195_);
lean_ctor_set(v_reuseFailAlloc_2208_, 1, v_snd_2196_);
v___x_2201_ = v_reuseFailAlloc_2208_;
goto v_reusejp_2200_;
}
v_reusejp_2200_:
{
lean_object* v___x_2203_; 
if (v_isShared_2199_ == 0)
{
lean_ctor_set(v___x_2198_, 1, v___x_2201_);
lean_ctor_set(v___x_2198_, 0, v_fst_2191_);
v___x_2203_ = v___x_2198_;
goto v_reusejp_2202_;
}
else
{
lean_object* v_reuseFailAlloc_2207_; 
v_reuseFailAlloc_2207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2207_, 0, v_fst_2191_);
lean_ctor_set(v_reuseFailAlloc_2207_, 1, v___x_2201_);
v___x_2203_ = v_reuseFailAlloc_2207_;
goto v_reusejp_2202_;
}
v_reusejp_2202_:
{
lean_object* v___x_2205_; 
if (v_isShared_2187_ == 0)
{
lean_ctor_set(v___x_2186_, 0, v___x_2203_);
v___x_2205_ = v___x_2186_;
goto v_reusejp_2204_;
}
else
{
lean_object* v_reuseFailAlloc_2206_; 
v_reuseFailAlloc_2206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2206_, 0, v___x_2203_);
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
}
}
}
}
else
{
lean_object* v_a_2221_; lean_object* v___x_2223_; uint8_t v_isShared_2224_; uint8_t v_isSharedCheck_2228_; 
lean_dec_ref(v___x_2181_);
lean_dec_ref(v_e_2171_);
v_a_2221_ = lean_ctor_get(v___x_2183_, 0);
v_isSharedCheck_2228_ = !lean_is_exclusive(v___x_2183_);
if (v_isSharedCheck_2228_ == 0)
{
v___x_2223_ = v___x_2183_;
v_isShared_2224_ = v_isSharedCheck_2228_;
goto v_resetjp_2222_;
}
else
{
lean_inc(v_a_2221_);
lean_dec(v___x_2183_);
v___x_2223_ = lean_box(0);
v_isShared_2224_ = v_isSharedCheck_2228_;
goto v_resetjp_2222_;
}
v_resetjp_2222_:
{
lean_object* v___x_2226_; 
if (v_isShared_2224_ == 0)
{
v___x_2226_ = v___x_2223_;
goto v_reusejp_2225_;
}
else
{
lean_object* v_reuseFailAlloc_2227_; 
v_reuseFailAlloc_2227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2227_, 0, v_a_2221_);
v___x_2226_ = v_reuseFailAlloc_2227_;
goto v_reusejp_2225_;
}
v_reusejp_2225_:
{
return v___x_2226_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__6___boxed(lean_object* v_e_2229_, lean_object* v___y_2230_, lean_object* v___y_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_){
_start:
{
lean_object* v_res_2235_; 
v_res_2235_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods___lam__6(v_e_2229_, v___y_2230_, v___y_2231_, v___y_2232_, v___y_2233_);
lean_dec(v___y_2233_);
lean_dec_ref(v___y_2232_);
lean_dec(v___y_2231_);
lean_dec_ref(v___y_2230_);
return v_res_2235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(lean_object* v_x_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_){
_start:
{
lean_object* v___x_2257_; 
v___x_2257_ = l_Lean_Meta_saveState___redArg(v___y_2253_, v___y_2255_);
if (lean_obj_tag(v___x_2257_) == 0)
{
lean_object* v_a_2258_; lean_object* v___x_2259_; 
v_a_2258_ = lean_ctor_get(v___x_2257_, 0);
lean_inc(v_a_2258_);
lean_dec_ref_known(v___x_2257_, 1);
lean_inc(v___y_2255_);
lean_inc_ref(v___y_2254_);
lean_inc(v___y_2253_);
lean_inc_ref(v___y_2252_);
v___x_2259_ = lean_apply_5(v_x_2251_, v___y_2252_, v___y_2253_, v___y_2254_, v___y_2255_, lean_box(0));
if (lean_obj_tag(v___x_2259_) == 0)
{
lean_object* v_a_2260_; lean_object* v___x_2262_; uint8_t v_isShared_2263_; uint8_t v_isSharedCheck_2268_; 
lean_dec(v_a_2258_);
v_a_2260_ = lean_ctor_get(v___x_2259_, 0);
v_isSharedCheck_2268_ = !lean_is_exclusive(v___x_2259_);
if (v_isSharedCheck_2268_ == 0)
{
v___x_2262_ = v___x_2259_;
v_isShared_2263_ = v_isSharedCheck_2268_;
goto v_resetjp_2261_;
}
else
{
lean_inc(v_a_2260_);
lean_dec(v___x_2259_);
v___x_2262_ = lean_box(0);
v_isShared_2263_ = v_isSharedCheck_2268_;
goto v_resetjp_2261_;
}
v_resetjp_2261_:
{
lean_object* v___x_2264_; lean_object* v___x_2266_; 
v___x_2264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2264_, 0, v_a_2260_);
if (v_isShared_2263_ == 0)
{
lean_ctor_set(v___x_2262_, 0, v___x_2264_);
v___x_2266_ = v___x_2262_;
goto v_reusejp_2265_;
}
else
{
lean_object* v_reuseFailAlloc_2267_; 
v_reuseFailAlloc_2267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2267_, 0, v___x_2264_);
v___x_2266_ = v_reuseFailAlloc_2267_;
goto v_reusejp_2265_;
}
v_reusejp_2265_:
{
return v___x_2266_;
}
}
}
else
{
lean_object* v_a_2269_; lean_object* v___x_2271_; uint8_t v_isShared_2272_; uint8_t v_isSharedCheck_2298_; 
v_a_2269_ = lean_ctor_get(v___x_2259_, 0);
v_isSharedCheck_2298_ = !lean_is_exclusive(v___x_2259_);
if (v_isSharedCheck_2298_ == 0)
{
v___x_2271_ = v___x_2259_;
v_isShared_2272_ = v_isSharedCheck_2298_;
goto v_resetjp_2270_;
}
else
{
lean_inc(v_a_2269_);
lean_dec(v___x_2259_);
v___x_2271_ = lean_box(0);
v_isShared_2272_ = v_isSharedCheck_2298_;
goto v_resetjp_2270_;
}
v_resetjp_2270_:
{
lean_object* v___x_2274_; 
lean_inc(v_a_2269_);
if (v_isShared_2272_ == 0)
{
v___x_2274_ = v___x_2271_;
goto v_reusejp_2273_;
}
else
{
lean_object* v_reuseFailAlloc_2297_; 
v_reuseFailAlloc_2297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2297_, 0, v_a_2269_);
v___x_2274_ = v_reuseFailAlloc_2297_;
goto v_reusejp_2273_;
}
v_reusejp_2273_:
{
uint8_t v___y_2276_; uint8_t v___x_2295_; 
v___x_2295_ = l_Lean_Exception_isInterrupt(v_a_2269_);
if (v___x_2295_ == 0)
{
uint8_t v___x_2296_; 
v___x_2296_ = l_Lean_Exception_isRuntime(v_a_2269_);
v___y_2276_ = v___x_2296_;
goto v___jp_2275_;
}
else
{
lean_dec(v_a_2269_);
v___y_2276_ = v___x_2295_;
goto v___jp_2275_;
}
v___jp_2275_:
{
if (v___y_2276_ == 0)
{
lean_object* v___x_2277_; 
lean_dec_ref(v___x_2274_);
v___x_2277_ = l_Lean_Meta_SavedState_restore___redArg(v_a_2258_, v___y_2253_, v___y_2255_);
lean_dec(v_a_2258_);
if (lean_obj_tag(v___x_2277_) == 0)
{
lean_object* v___x_2279_; uint8_t v_isShared_2280_; uint8_t v_isSharedCheck_2285_; 
v_isSharedCheck_2285_ = !lean_is_exclusive(v___x_2277_);
if (v_isSharedCheck_2285_ == 0)
{
lean_object* v_unused_2286_; 
v_unused_2286_ = lean_ctor_get(v___x_2277_, 0);
lean_dec(v_unused_2286_);
v___x_2279_ = v___x_2277_;
v_isShared_2280_ = v_isSharedCheck_2285_;
goto v_resetjp_2278_;
}
else
{
lean_dec(v___x_2277_);
v___x_2279_ = lean_box(0);
v_isShared_2280_ = v_isSharedCheck_2285_;
goto v_resetjp_2278_;
}
v_resetjp_2278_:
{
lean_object* v___x_2281_; lean_object* v___x_2283_; 
v___x_2281_ = lean_box(0);
if (v_isShared_2280_ == 0)
{
lean_ctor_set(v___x_2279_, 0, v___x_2281_);
v___x_2283_ = v___x_2279_;
goto v_reusejp_2282_;
}
else
{
lean_object* v_reuseFailAlloc_2284_; 
v_reuseFailAlloc_2284_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2284_, 0, v___x_2281_);
v___x_2283_ = v_reuseFailAlloc_2284_;
goto v_reusejp_2282_;
}
v_reusejp_2282_:
{
return v___x_2283_;
}
}
}
else
{
lean_object* v_a_2287_; lean_object* v___x_2289_; uint8_t v_isShared_2290_; uint8_t v_isSharedCheck_2294_; 
v_a_2287_ = lean_ctor_get(v___x_2277_, 0);
v_isSharedCheck_2294_ = !lean_is_exclusive(v___x_2277_);
if (v_isSharedCheck_2294_ == 0)
{
v___x_2289_ = v___x_2277_;
v_isShared_2290_ = v_isSharedCheck_2294_;
goto v_resetjp_2288_;
}
else
{
lean_inc(v_a_2287_);
lean_dec(v___x_2277_);
v___x_2289_ = lean_box(0);
v_isShared_2290_ = v_isSharedCheck_2294_;
goto v_resetjp_2288_;
}
v_resetjp_2288_:
{
lean_object* v___x_2292_; 
if (v_isShared_2290_ == 0)
{
v___x_2292_ = v___x_2289_;
goto v_reusejp_2291_;
}
else
{
lean_object* v_reuseFailAlloc_2293_; 
v_reuseFailAlloc_2293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2293_, 0, v_a_2287_);
v___x_2292_ = v_reuseFailAlloc_2293_;
goto v_reusejp_2291_;
}
v_reusejp_2291_:
{
return v___x_2292_;
}
}
}
}
else
{
lean_dec(v_a_2258_);
return v___x_2274_;
}
}
}
}
}
}
else
{
lean_object* v_a_2299_; lean_object* v___x_2301_; uint8_t v_isShared_2302_; uint8_t v_isSharedCheck_2306_; 
lean_dec_ref(v_x_2251_);
v_a_2299_ = lean_ctor_get(v___x_2257_, 0);
v_isSharedCheck_2306_ = !lean_is_exclusive(v___x_2257_);
if (v_isSharedCheck_2306_ == 0)
{
v___x_2301_ = v___x_2257_;
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
else
{
lean_inc(v_a_2299_);
lean_dec(v___x_2257_);
v___x_2301_ = lean_box(0);
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
v_resetjp_2300_:
{
lean_object* v___x_2304_; 
if (v_isShared_2302_ == 0)
{
v___x_2304_ = v___x_2301_;
goto v_reusejp_2303_;
}
else
{
lean_object* v_reuseFailAlloc_2305_; 
v_reuseFailAlloc_2305_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2305_, 0, v_a_2299_);
v___x_2304_ = v_reuseFailAlloc_2305_;
goto v_reusejp_2303_;
}
v_reusejp_2303_:
{
return v___x_2304_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg___boxed(lean_object* v_x_2307_, lean_object* v___y_2308_, lean_object* v___y_2309_, lean_object* v___y_2310_, lean_object* v___y_2311_, lean_object* v___y_2312_){
_start:
{
lean_object* v_res_2313_; 
v_res_2313_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(v_x_2307_, v___y_2308_, v___y_2309_, v___y_2310_, v___y_2311_);
lean_dec(v___y_2311_);
lean_dec_ref(v___y_2310_);
lean_dec(v___y_2309_);
lean_dec_ref(v___y_2308_);
return v_res_2313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0(lean_object* v_00_u03b1_2314_, lean_object* v_x_2315_, lean_object* v___y_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_){
_start:
{
lean_object* v___x_2321_; 
v___x_2321_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(v_x_2315_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_);
return v___x_2321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___boxed(lean_object* v_00_u03b1_2322_, lean_object* v_x_2323_, lean_object* v___y_2324_, lean_object* v___y_2325_, lean_object* v___y_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_){
_start:
{
lean_object* v_res_2329_; 
v_res_2329_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0(v_00_u03b1_2322_, v_x_2323_, v___y_2324_, v___y_2325_, v___y_2326_, v___y_2327_);
lean_dec(v___y_2327_);
lean_dec_ref(v___y_2326_);
lean_dec(v___y_2325_);
lean_dec_ref(v___y_2324_);
return v_res_2329_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1(void){
_start:
{
lean_object* v___x_2331_; lean_object* v___x_2332_; 
v___x_2331_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__0));
v___x_2332_ = l_Lean_stringToMessageData(v___x_2331_);
return v___x_2332_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3(void){
_start:
{
lean_object* v___x_2334_; lean_object* v___x_2335_; 
v___x_2334_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__2));
v___x_2335_ = l_Lean_stringToMessageData(v___x_2334_);
return v___x_2335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2(lean_object* v_m_2336_, lean_object* v_e_2337_, uint8_t v_mustUseBounds_2338_, lean_object* v_as_2339_, size_t v_sz_2340_, size_t v_i_2341_, lean_object* v_b_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_){
_start:
{
lean_object* v_a_2349_; uint8_t v___x_2353_; 
v___x_2353_ = lean_usize_dec_lt(v_i_2341_, v_sz_2340_);
if (v___x_2353_ == 0)
{
lean_object* v___x_2354_; 
lean_dec_ref(v_e_2337_);
lean_dec_ref(v_m_2336_);
v___x_2354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2354_, 0, v_b_2342_);
return v___x_2354_;
}
else
{
lean_object* v_a_2355_; uint8_t v___x_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; 
v_a_2355_ = lean_array_uget_borrowed(v_as_2339_, v_i_2341_);
v___x_2356_ = 0;
v___x_2357_ = lean_box(v___x_2356_);
lean_inc(v_a_2355_);
lean_inc_ref(v_e_2337_);
lean_inc_ref(v_m_2336_);
v___x_2358_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___boxed), 9, 4);
lean_closure_set(v___x_2358_, 0, v_m_2336_);
lean_closure_set(v___x_2358_, 1, v_e_2337_);
lean_closure_set(v___x_2358_, 2, v_a_2355_);
lean_closure_set(v___x_2358_, 3, v___x_2357_);
v___x_2359_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(v___x_2358_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_);
if (lean_obj_tag(v___x_2359_) == 0)
{
lean_object* v_a_2360_; 
v_a_2360_ = lean_ctor_get(v___x_2359_, 0);
lean_inc(v_a_2360_);
lean_dec_ref_known(v___x_2359_, 1);
if (lean_obj_tag(v_a_2360_) == 1)
{
if (lean_obj_tag(v_b_2342_) == 0)
{
v_a_2349_ = v_a_2360_;
goto v___jp_2348_;
}
else
{
lean_object* v_val_2361_; lean_object* v_val_2362_; lean_object* v_fst_2363_; lean_object* v_fst_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; uint8_t v___x_2367_; 
v_val_2361_ = lean_ctor_get(v_a_2360_, 0);
v_val_2362_ = lean_ctor_get(v_b_2342_, 0);
v_fst_2363_ = lean_ctor_get(v_val_2361_, 0);
v_fst_2364_ = lean_ctor_get(v_val_2362_, 0);
v___x_2365_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper(v_fst_2363_);
v___x_2366_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper(v_fst_2364_);
v___x_2367_ = lean_int_dec_lt(v___x_2365_, v___x_2366_);
lean_dec(v___x_2366_);
lean_dec(v___x_2365_);
if (v___x_2367_ == 0)
{
lean_dec_ref_known(v_a_2360_, 1);
v_a_2349_ = v_b_2342_;
goto v___jp_2348_;
}
else
{
lean_dec_ref_known(v_b_2342_, 1);
v_a_2349_ = v_a_2360_;
goto v___jp_2348_;
}
}
}
else
{
lean_dec(v_a_2360_);
if (v_mustUseBounds_2338_ == 0)
{
v_a_2349_ = v_b_2342_;
goto v___jp_2348_;
}
else
{
lean_object* v___x_2368_; 
lean_inc(v___y_2346_);
lean_inc_ref(v___y_2345_);
lean_inc(v___y_2344_);
lean_inc_ref(v___y_2343_);
lean_inc(v_a_2355_);
v___x_2368_ = lean_infer_type(v_a_2355_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_);
if (lean_obj_tag(v___x_2368_) == 0)
{
lean_object* v_a_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; 
v_a_2369_ = lean_ctor_get(v___x_2368_, 0);
lean_inc(v_a_2369_);
lean_dec_ref_known(v___x_2368_, 1);
v___x_2370_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1);
v___x_2371_ = l_Lean_MessageData_ofExpr(v_a_2369_);
v___x_2372_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2372_, 0, v___x_2370_);
lean_ctor_set(v___x_2372_, 1, v___x_2371_);
v___x_2373_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3);
v___x_2374_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2374_, 0, v___x_2372_);
lean_ctor_set(v___x_2374_, 1, v___x_2373_);
v___x_2375_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_2374_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_);
if (lean_obj_tag(v___x_2375_) == 0)
{
lean_dec_ref_known(v___x_2375_, 1);
v_a_2349_ = v_b_2342_;
goto v___jp_2348_;
}
else
{
lean_object* v_a_2376_; lean_object* v___x_2378_; uint8_t v_isShared_2379_; uint8_t v_isSharedCheck_2383_; 
lean_dec(v_b_2342_);
lean_dec_ref(v_e_2337_);
lean_dec_ref(v_m_2336_);
v_a_2376_ = lean_ctor_get(v___x_2375_, 0);
v_isSharedCheck_2383_ = !lean_is_exclusive(v___x_2375_);
if (v_isSharedCheck_2383_ == 0)
{
v___x_2378_ = v___x_2375_;
v_isShared_2379_ = v_isSharedCheck_2383_;
goto v_resetjp_2377_;
}
else
{
lean_inc(v_a_2376_);
lean_dec(v___x_2375_);
v___x_2378_ = lean_box(0);
v_isShared_2379_ = v_isSharedCheck_2383_;
goto v_resetjp_2377_;
}
v_resetjp_2377_:
{
lean_object* v___x_2381_; 
if (v_isShared_2379_ == 0)
{
v___x_2381_ = v___x_2378_;
goto v_reusejp_2380_;
}
else
{
lean_object* v_reuseFailAlloc_2382_; 
v_reuseFailAlloc_2382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2382_, 0, v_a_2376_);
v___x_2381_ = v_reuseFailAlloc_2382_;
goto v_reusejp_2380_;
}
v_reusejp_2380_:
{
return v___x_2381_;
}
}
}
}
else
{
lean_object* v_a_2384_; lean_object* v___x_2386_; uint8_t v_isShared_2387_; uint8_t v_isSharedCheck_2391_; 
lean_dec(v_b_2342_);
lean_dec_ref(v_e_2337_);
lean_dec_ref(v_m_2336_);
v_a_2384_ = lean_ctor_get(v___x_2368_, 0);
v_isSharedCheck_2391_ = !lean_is_exclusive(v___x_2368_);
if (v_isSharedCheck_2391_ == 0)
{
v___x_2386_ = v___x_2368_;
v_isShared_2387_ = v_isSharedCheck_2391_;
goto v_resetjp_2385_;
}
else
{
lean_inc(v_a_2384_);
lean_dec(v___x_2368_);
v___x_2386_ = lean_box(0);
v_isShared_2387_ = v_isSharedCheck_2391_;
goto v_resetjp_2385_;
}
v_resetjp_2385_:
{
lean_object* v___x_2389_; 
if (v_isShared_2387_ == 0)
{
v___x_2389_ = v___x_2386_;
goto v_reusejp_2388_;
}
else
{
lean_object* v_reuseFailAlloc_2390_; 
v_reuseFailAlloc_2390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2390_, 0, v_a_2384_);
v___x_2389_ = v_reuseFailAlloc_2390_;
goto v_reusejp_2388_;
}
v_reusejp_2388_:
{
return v___x_2389_;
}
}
}
}
}
}
else
{
lean_dec(v_b_2342_);
lean_dec_ref(v_e_2337_);
lean_dec_ref(v_m_2336_);
return v___x_2359_;
}
}
v___jp_2348_:
{
size_t v___x_2350_; size_t v___x_2351_; 
v___x_2350_ = ((size_t)1ULL);
v___x_2351_ = lean_usize_add(v_i_2341_, v___x_2350_);
v_i_2341_ = v___x_2351_;
v_b_2342_ = v_a_2349_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___boxed(lean_object* v_m_2392_, lean_object* v_e_2393_, lean_object* v_mustUseBounds_2394_, lean_object* v_as_2395_, lean_object* v_sz_2396_, lean_object* v_i_2397_, lean_object* v_b_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_){
_start:
{
uint8_t v_mustUseBounds_boxed_2404_; size_t v_sz_boxed_2405_; size_t v_i_boxed_2406_; lean_object* v_res_2407_; 
v_mustUseBounds_boxed_2404_ = lean_unbox(v_mustUseBounds_2394_);
v_sz_boxed_2405_ = lean_unbox_usize(v_sz_2396_);
lean_dec(v_sz_2396_);
v_i_boxed_2406_ = lean_unbox_usize(v_i_2397_);
lean_dec(v_i_2397_);
v_res_2407_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2(v_m_2392_, v_e_2393_, v_mustUseBounds_boxed_2404_, v_as_2395_, v_sz_boxed_2405_, v_i_boxed_2406_, v_b_2398_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_);
lean_dec(v___y_2402_);
lean_dec_ref(v___y_2401_);
lean_dec(v___y_2400_);
lean_dec_ref(v___y_2399_);
lean_dec_ref(v_as_2395_);
return v_res_2407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___redArg(lean_object* v___x_2408_, lean_object* v_m_2409_, lean_object* v_e_2410_, lean_object* v_a_2411_, lean_object* v_a_2412_, lean_object* v_range_2413_, lean_object* v_b_2414_, lean_object* v_i_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_, lean_object* v___y_2418_, lean_object* v___y_2419_){
_start:
{
lean_object* v_stop_2421_; lean_object* v_step_2422_; uint8_t v___x_2423_; 
v_stop_2421_ = lean_ctor_get(v_range_2413_, 1);
v_step_2422_ = lean_ctor_get(v_range_2413_, 2);
v___x_2423_ = lean_nat_dec_lt(v_i_2415_, v_stop_2421_);
if (v___x_2423_ == 0)
{
lean_object* v___x_2424_; 
lean_dec(v_i_2415_);
lean_dec(v_a_2412_);
lean_dec_ref(v_a_2411_);
lean_dec_ref(v_e_2410_);
lean_dec_ref(v_m_2409_);
v___x_2424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2424_, 0, v_b_2414_);
return v___x_2424_;
}
else
{
lean_object* v_mkNumeral_2425_; lean_object* v___x_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; 
v_mkNumeral_2425_ = lean_ctor_get(v_m_2409_, 7);
lean_inc(v_i_2415_);
v___x_2426_ = lean_nat_to_int(v_i_2415_);
v___x_2427_ = lean_int_add(v___x_2408_, v___x_2426_);
lean_dec(v___x_2426_);
lean_inc_ref(v_mkNumeral_2425_);
lean_inc(v___y_2419_);
lean_inc_ref(v___y_2418_);
lean_inc(v___y_2417_);
lean_inc_ref(v___y_2416_);
lean_inc(v___x_2427_);
v___x_2428_ = lean_apply_6(v_mkNumeral_2425_, v___x_2427_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_, lean_box(0));
if (lean_obj_tag(v___x_2428_) == 0)
{
lean_object* v_a_2429_; lean_object* v___x_2430_; 
v_a_2429_ = lean_ctor_get(v___x_2428_, 0);
lean_inc_n(v_a_2429_, 2);
lean_dec_ref_known(v___x_2428_, 1);
lean_inc_ref(v_e_2410_);
v___x_2430_ = l_Lean_Meta_mkEq(v_e_2410_, v_a_2429_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_);
if (lean_obj_tag(v___x_2430_) == 0)
{
lean_object* v_a_2431_; lean_object* v___x_2432_; 
v_a_2431_ = lean_ctor_get(v___x_2430_, 0);
lean_inc(v_a_2431_);
lean_dec_ref_known(v___x_2430_, 1);
lean_inc_ref(v_a_2411_);
v___x_2432_ = l_Lean_mkArrow(v_a_2431_, v_a_2411_, v___y_2418_, v___y_2419_);
if (lean_obj_tag(v___x_2432_) == 0)
{
lean_object* v_a_2433_; lean_object* v___x_2434_; uint8_t v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; 
v_a_2433_ = lean_ctor_get(v___x_2432_, 0);
lean_inc(v_a_2433_);
lean_dec_ref_known(v___x_2432_, 1);
v___x_2434_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2434_, 0, v_a_2433_);
v___x_2435_ = 2;
v___x_2436_ = l_Int_repr(v___x_2427_);
v___x_2437_ = lean_box(0);
v___x_2438_ = l_Lean_Name_str___override(v___x_2437_, v___x_2436_);
lean_inc(v_a_2412_);
v___x_2439_ = l_Lean_Meta_appendTag(v_a_2412_, v___x_2438_);
lean_dec(v___x_2438_);
v___x_2440_ = l_Lean_Meta_mkFreshExprMVar(v___x_2434_, v___x_2435_, v___x_2439_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_);
if (lean_obj_tag(v___x_2440_) == 0)
{
lean_object* v_a_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; 
v_a_2441_ = lean_ctor_get(v___x_2440_, 0);
lean_inc(v_a_2441_);
lean_dec_ref_known(v___x_2440_, 1);
v___x_2442_ = l_Lean_Expr_mvarId_x21(v_a_2441_);
lean_dec(v_a_2441_);
v___x_2443_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2443_, 0, v_a_2429_);
lean_ctor_set(v___x_2443_, 1, v___x_2427_);
lean_ctor_set(v___x_2443_, 2, v___x_2442_);
v___x_2444_ = lean_array_push(v_b_2414_, v___x_2443_);
v___x_2445_ = lean_nat_add(v_i_2415_, v_step_2422_);
lean_dec(v_i_2415_);
v_b_2414_ = v___x_2444_;
v_i_2415_ = v___x_2445_;
goto _start;
}
else
{
lean_object* v_a_2447_; lean_object* v___x_2449_; uint8_t v_isShared_2450_; uint8_t v_isSharedCheck_2454_; 
lean_dec(v_a_2429_);
lean_dec(v___x_2427_);
lean_dec(v_i_2415_);
lean_dec_ref(v_b_2414_);
lean_dec(v_a_2412_);
lean_dec_ref(v_a_2411_);
lean_dec_ref(v_e_2410_);
lean_dec_ref(v_m_2409_);
v_a_2447_ = lean_ctor_get(v___x_2440_, 0);
v_isSharedCheck_2454_ = !lean_is_exclusive(v___x_2440_);
if (v_isSharedCheck_2454_ == 0)
{
v___x_2449_ = v___x_2440_;
v_isShared_2450_ = v_isSharedCheck_2454_;
goto v_resetjp_2448_;
}
else
{
lean_inc(v_a_2447_);
lean_dec(v___x_2440_);
v___x_2449_ = lean_box(0);
v_isShared_2450_ = v_isSharedCheck_2454_;
goto v_resetjp_2448_;
}
v_resetjp_2448_:
{
lean_object* v___x_2452_; 
if (v_isShared_2450_ == 0)
{
v___x_2452_ = v___x_2449_;
goto v_reusejp_2451_;
}
else
{
lean_object* v_reuseFailAlloc_2453_; 
v_reuseFailAlloc_2453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2453_, 0, v_a_2447_);
v___x_2452_ = v_reuseFailAlloc_2453_;
goto v_reusejp_2451_;
}
v_reusejp_2451_:
{
return v___x_2452_;
}
}
}
}
else
{
lean_object* v_a_2455_; lean_object* v___x_2457_; uint8_t v_isShared_2458_; uint8_t v_isSharedCheck_2462_; 
lean_dec(v_a_2429_);
lean_dec(v___x_2427_);
lean_dec(v_i_2415_);
lean_dec_ref(v_b_2414_);
lean_dec(v_a_2412_);
lean_dec_ref(v_a_2411_);
lean_dec_ref(v_e_2410_);
lean_dec_ref(v_m_2409_);
v_a_2455_ = lean_ctor_get(v___x_2432_, 0);
v_isSharedCheck_2462_ = !lean_is_exclusive(v___x_2432_);
if (v_isSharedCheck_2462_ == 0)
{
v___x_2457_ = v___x_2432_;
v_isShared_2458_ = v_isSharedCheck_2462_;
goto v_resetjp_2456_;
}
else
{
lean_inc(v_a_2455_);
lean_dec(v___x_2432_);
v___x_2457_ = lean_box(0);
v_isShared_2458_ = v_isSharedCheck_2462_;
goto v_resetjp_2456_;
}
v_resetjp_2456_:
{
lean_object* v___x_2460_; 
if (v_isShared_2458_ == 0)
{
v___x_2460_ = v___x_2457_;
goto v_reusejp_2459_;
}
else
{
lean_object* v_reuseFailAlloc_2461_; 
v_reuseFailAlloc_2461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2461_, 0, v_a_2455_);
v___x_2460_ = v_reuseFailAlloc_2461_;
goto v_reusejp_2459_;
}
v_reusejp_2459_:
{
return v___x_2460_;
}
}
}
}
else
{
lean_object* v_a_2463_; lean_object* v___x_2465_; uint8_t v_isShared_2466_; uint8_t v_isSharedCheck_2470_; 
lean_dec(v_a_2429_);
lean_dec(v___x_2427_);
lean_dec(v_i_2415_);
lean_dec_ref(v_b_2414_);
lean_dec(v_a_2412_);
lean_dec_ref(v_a_2411_);
lean_dec_ref(v_e_2410_);
lean_dec_ref(v_m_2409_);
v_a_2463_ = lean_ctor_get(v___x_2430_, 0);
v_isSharedCheck_2470_ = !lean_is_exclusive(v___x_2430_);
if (v_isSharedCheck_2470_ == 0)
{
v___x_2465_ = v___x_2430_;
v_isShared_2466_ = v_isSharedCheck_2470_;
goto v_resetjp_2464_;
}
else
{
lean_inc(v_a_2463_);
lean_dec(v___x_2430_);
v___x_2465_ = lean_box(0);
v_isShared_2466_ = v_isSharedCheck_2470_;
goto v_resetjp_2464_;
}
v_resetjp_2464_:
{
lean_object* v___x_2468_; 
if (v_isShared_2466_ == 0)
{
v___x_2468_ = v___x_2465_;
goto v_reusejp_2467_;
}
else
{
lean_object* v_reuseFailAlloc_2469_; 
v_reuseFailAlloc_2469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2469_, 0, v_a_2463_);
v___x_2468_ = v_reuseFailAlloc_2469_;
goto v_reusejp_2467_;
}
v_reusejp_2467_:
{
return v___x_2468_;
}
}
}
}
else
{
lean_object* v_a_2471_; lean_object* v___x_2473_; uint8_t v_isShared_2474_; uint8_t v_isSharedCheck_2478_; 
lean_dec(v___x_2427_);
lean_dec(v_i_2415_);
lean_dec_ref(v_b_2414_);
lean_dec(v_a_2412_);
lean_dec_ref(v_a_2411_);
lean_dec_ref(v_e_2410_);
lean_dec_ref(v_m_2409_);
v_a_2471_ = lean_ctor_get(v___x_2428_, 0);
v_isSharedCheck_2478_ = !lean_is_exclusive(v___x_2428_);
if (v_isSharedCheck_2478_ == 0)
{
v___x_2473_ = v___x_2428_;
v_isShared_2474_ = v_isSharedCheck_2478_;
goto v_resetjp_2472_;
}
else
{
lean_inc(v_a_2471_);
lean_dec(v___x_2428_);
v___x_2473_ = lean_box(0);
v_isShared_2474_ = v_isSharedCheck_2478_;
goto v_resetjp_2472_;
}
v_resetjp_2472_:
{
lean_object* v___x_2476_; 
if (v_isShared_2474_ == 0)
{
v___x_2476_ = v___x_2473_;
goto v_reusejp_2475_;
}
else
{
lean_object* v_reuseFailAlloc_2477_; 
v_reuseFailAlloc_2477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2477_, 0, v_a_2471_);
v___x_2476_ = v_reuseFailAlloc_2477_;
goto v_reusejp_2475_;
}
v_reusejp_2475_:
{
return v___x_2476_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___redArg___boxed(lean_object* v___x_2479_, lean_object* v_m_2480_, lean_object* v_e_2481_, lean_object* v_a_2482_, lean_object* v_a_2483_, lean_object* v_range_2484_, lean_object* v_b_2485_, lean_object* v_i_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_, lean_object* v___y_2489_, lean_object* v___y_2490_, lean_object* v___y_2491_){
_start:
{
lean_object* v_res_2492_; 
v_res_2492_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___redArg(v___x_2479_, v_m_2480_, v_e_2481_, v_a_2482_, v_a_2483_, v_range_2484_, v_b_2485_, v_i_2486_, v___y_2487_, v___y_2488_, v___y_2489_, v___y_2490_);
lean_dec(v___y_2490_);
lean_dec_ref(v___y_2489_);
lean_dec(v___y_2488_);
lean_dec_ref(v___y_2487_);
lean_dec_ref(v_range_2484_);
lean_dec(v___x_2479_);
return v_res_2492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___redArg(lean_object* v___x_2493_, lean_object* v_m_2494_, lean_object* v_e_2495_, lean_object* v_a_2496_, lean_object* v_a_2497_, lean_object* v_range_2498_, lean_object* v_b_2499_, lean_object* v_i_2500_, lean_object* v___y_2501_, lean_object* v___y_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_){
_start:
{
lean_object* v_stop_2506_; lean_object* v_step_2507_; uint8_t v___x_2508_; 
v_stop_2506_ = lean_ctor_get(v_range_2498_, 1);
v_step_2507_ = lean_ctor_get(v_range_2498_, 2);
v___x_2508_ = lean_nat_dec_lt(v_i_2500_, v_stop_2506_);
if (v___x_2508_ == 0)
{
lean_object* v___x_2509_; 
lean_dec(v_i_2500_);
lean_dec(v_a_2497_);
lean_dec_ref(v_a_2496_);
lean_dec_ref(v_e_2495_);
lean_dec_ref(v_m_2494_);
v___x_2509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2509_, 0, v_b_2499_);
return v___x_2509_;
}
else
{
lean_object* v_mkNumeral_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; 
v_mkNumeral_2510_ = lean_ctor_get(v_m_2494_, 7);
lean_inc(v_i_2500_);
v___x_2511_ = lean_nat_to_int(v_i_2500_);
v___x_2512_ = lean_int_add(v___x_2493_, v___x_2511_);
lean_dec(v___x_2511_);
lean_inc_ref(v_mkNumeral_2510_);
lean_inc(v___y_2504_);
lean_inc_ref(v___y_2503_);
lean_inc(v___y_2502_);
lean_inc_ref(v___y_2501_);
lean_inc(v___x_2512_);
v___x_2513_ = lean_apply_6(v_mkNumeral_2510_, v___x_2512_, v___y_2501_, v___y_2502_, v___y_2503_, v___y_2504_, lean_box(0));
if (lean_obj_tag(v___x_2513_) == 0)
{
lean_object* v_a_2514_; lean_object* v___x_2515_; 
v_a_2514_ = lean_ctor_get(v___x_2513_, 0);
lean_inc_n(v_a_2514_, 2);
lean_dec_ref_known(v___x_2513_, 1);
lean_inc_ref(v_e_2495_);
v___x_2515_ = l_Lean_Meta_mkEq(v_e_2495_, v_a_2514_, v___y_2501_, v___y_2502_, v___y_2503_, v___y_2504_);
if (lean_obj_tag(v___x_2515_) == 0)
{
lean_object* v_a_2516_; lean_object* v___x_2517_; 
v_a_2516_ = lean_ctor_get(v___x_2515_, 0);
lean_inc(v_a_2516_);
lean_dec_ref_known(v___x_2515_, 1);
lean_inc_ref(v_a_2496_);
v___x_2517_ = l_Lean_mkArrow(v_a_2516_, v_a_2496_, v___y_2503_, v___y_2504_);
if (lean_obj_tag(v___x_2517_) == 0)
{
lean_object* v_a_2518_; lean_object* v___x_2519_; uint8_t v___x_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; 
v_a_2518_ = lean_ctor_get(v___x_2517_, 0);
lean_inc(v_a_2518_);
lean_dec_ref_known(v___x_2517_, 1);
v___x_2519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2519_, 0, v_a_2518_);
v___x_2520_ = 2;
v___x_2521_ = l_Int_repr(v___x_2512_);
v___x_2522_ = lean_box(0);
v___x_2523_ = l_Lean_Name_str___override(v___x_2522_, v___x_2521_);
lean_inc(v_a_2497_);
v___x_2524_ = l_Lean_Meta_appendTag(v_a_2497_, v___x_2523_);
lean_dec(v___x_2523_);
v___x_2525_ = l_Lean_Meta_mkFreshExprMVar(v___x_2519_, v___x_2520_, v___x_2524_, v___y_2501_, v___y_2502_, v___y_2503_, v___y_2504_);
if (lean_obj_tag(v___x_2525_) == 0)
{
lean_object* v_a_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; 
v_a_2526_ = lean_ctor_get(v___x_2525_, 0);
lean_inc(v_a_2526_);
lean_dec_ref_known(v___x_2525_, 1);
v___x_2527_ = l_Lean_Expr_mvarId_x21(v_a_2526_);
lean_dec(v_a_2526_);
v___x_2528_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2528_, 0, v_a_2514_);
lean_ctor_set(v___x_2528_, 1, v___x_2512_);
lean_ctor_set(v___x_2528_, 2, v___x_2527_);
v___x_2529_ = lean_array_push(v_b_2499_, v___x_2528_);
v___x_2530_ = lean_nat_add(v_i_2500_, v_step_2507_);
lean_dec(v_i_2500_);
v___x_2531_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___redArg(v___x_2493_, v_m_2494_, v_e_2495_, v_a_2496_, v_a_2497_, v_range_2498_, v___x_2529_, v___x_2530_, v___y_2501_, v___y_2502_, v___y_2503_, v___y_2504_);
return v___x_2531_;
}
else
{
lean_object* v_a_2532_; lean_object* v___x_2534_; uint8_t v_isShared_2535_; uint8_t v_isSharedCheck_2539_; 
lean_dec(v_a_2514_);
lean_dec(v___x_2512_);
lean_dec(v_i_2500_);
lean_dec_ref(v_b_2499_);
lean_dec(v_a_2497_);
lean_dec_ref(v_a_2496_);
lean_dec_ref(v_e_2495_);
lean_dec_ref(v_m_2494_);
v_a_2532_ = lean_ctor_get(v___x_2525_, 0);
v_isSharedCheck_2539_ = !lean_is_exclusive(v___x_2525_);
if (v_isSharedCheck_2539_ == 0)
{
v___x_2534_ = v___x_2525_;
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
else
{
lean_inc(v_a_2532_);
lean_dec(v___x_2525_);
v___x_2534_ = lean_box(0);
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
v_resetjp_2533_:
{
lean_object* v___x_2537_; 
if (v_isShared_2535_ == 0)
{
v___x_2537_ = v___x_2534_;
goto v_reusejp_2536_;
}
else
{
lean_object* v_reuseFailAlloc_2538_; 
v_reuseFailAlloc_2538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2538_, 0, v_a_2532_);
v___x_2537_ = v_reuseFailAlloc_2538_;
goto v_reusejp_2536_;
}
v_reusejp_2536_:
{
return v___x_2537_;
}
}
}
}
else
{
lean_object* v_a_2540_; lean_object* v___x_2542_; uint8_t v_isShared_2543_; uint8_t v_isSharedCheck_2547_; 
lean_dec(v_a_2514_);
lean_dec(v___x_2512_);
lean_dec(v_i_2500_);
lean_dec_ref(v_b_2499_);
lean_dec(v_a_2497_);
lean_dec_ref(v_a_2496_);
lean_dec_ref(v_e_2495_);
lean_dec_ref(v_m_2494_);
v_a_2540_ = lean_ctor_get(v___x_2517_, 0);
v_isSharedCheck_2547_ = !lean_is_exclusive(v___x_2517_);
if (v_isSharedCheck_2547_ == 0)
{
v___x_2542_ = v___x_2517_;
v_isShared_2543_ = v_isSharedCheck_2547_;
goto v_resetjp_2541_;
}
else
{
lean_inc(v_a_2540_);
lean_dec(v___x_2517_);
v___x_2542_ = lean_box(0);
v_isShared_2543_ = v_isSharedCheck_2547_;
goto v_resetjp_2541_;
}
v_resetjp_2541_:
{
lean_object* v___x_2545_; 
if (v_isShared_2543_ == 0)
{
v___x_2545_ = v___x_2542_;
goto v_reusejp_2544_;
}
else
{
lean_object* v_reuseFailAlloc_2546_; 
v_reuseFailAlloc_2546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2546_, 0, v_a_2540_);
v___x_2545_ = v_reuseFailAlloc_2546_;
goto v_reusejp_2544_;
}
v_reusejp_2544_:
{
return v___x_2545_;
}
}
}
}
else
{
lean_object* v_a_2548_; lean_object* v___x_2550_; uint8_t v_isShared_2551_; uint8_t v_isSharedCheck_2555_; 
lean_dec(v_a_2514_);
lean_dec(v___x_2512_);
lean_dec(v_i_2500_);
lean_dec_ref(v_b_2499_);
lean_dec(v_a_2497_);
lean_dec_ref(v_a_2496_);
lean_dec_ref(v_e_2495_);
lean_dec_ref(v_m_2494_);
v_a_2548_ = lean_ctor_get(v___x_2515_, 0);
v_isSharedCheck_2555_ = !lean_is_exclusive(v___x_2515_);
if (v_isSharedCheck_2555_ == 0)
{
v___x_2550_ = v___x_2515_;
v_isShared_2551_ = v_isSharedCheck_2555_;
goto v_resetjp_2549_;
}
else
{
lean_inc(v_a_2548_);
lean_dec(v___x_2515_);
v___x_2550_ = lean_box(0);
v_isShared_2551_ = v_isSharedCheck_2555_;
goto v_resetjp_2549_;
}
v_resetjp_2549_:
{
lean_object* v___x_2553_; 
if (v_isShared_2551_ == 0)
{
v___x_2553_ = v___x_2550_;
goto v_reusejp_2552_;
}
else
{
lean_object* v_reuseFailAlloc_2554_; 
v_reuseFailAlloc_2554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2554_, 0, v_a_2548_);
v___x_2553_ = v_reuseFailAlloc_2554_;
goto v_reusejp_2552_;
}
v_reusejp_2552_:
{
return v___x_2553_;
}
}
}
}
else
{
lean_object* v_a_2556_; lean_object* v___x_2558_; uint8_t v_isShared_2559_; uint8_t v_isSharedCheck_2563_; 
lean_dec(v___x_2512_);
lean_dec(v_i_2500_);
lean_dec_ref(v_b_2499_);
lean_dec(v_a_2497_);
lean_dec_ref(v_a_2496_);
lean_dec_ref(v_e_2495_);
lean_dec_ref(v_m_2494_);
v_a_2556_ = lean_ctor_get(v___x_2513_, 0);
v_isSharedCheck_2563_ = !lean_is_exclusive(v___x_2513_);
if (v_isSharedCheck_2563_ == 0)
{
v___x_2558_ = v___x_2513_;
v_isShared_2559_ = v_isSharedCheck_2563_;
goto v_resetjp_2557_;
}
else
{
lean_inc(v_a_2556_);
lean_dec(v___x_2513_);
v___x_2558_ = lean_box(0);
v_isShared_2559_ = v_isSharedCheck_2563_;
goto v_resetjp_2557_;
}
v_resetjp_2557_:
{
lean_object* v___x_2561_; 
if (v_isShared_2559_ == 0)
{
v___x_2561_ = v___x_2558_;
goto v_reusejp_2560_;
}
else
{
lean_object* v_reuseFailAlloc_2562_; 
v_reuseFailAlloc_2562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2562_, 0, v_a_2556_);
v___x_2561_ = v_reuseFailAlloc_2562_;
goto v_reusejp_2560_;
}
v_reusejp_2560_:
{
return v___x_2561_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___redArg___boxed(lean_object* v___x_2564_, lean_object* v_m_2565_, lean_object* v_e_2566_, lean_object* v_a_2567_, lean_object* v_a_2568_, lean_object* v_range_2569_, lean_object* v_b_2570_, lean_object* v_i_2571_, lean_object* v___y_2572_, lean_object* v___y_2573_, lean_object* v___y_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_){
_start:
{
lean_object* v_res_2577_; 
v_res_2577_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___redArg(v___x_2564_, v_m_2565_, v_e_2566_, v_a_2567_, v_a_2568_, v_range_2569_, v_b_2570_, v_i_2571_, v___y_2572_, v___y_2573_, v___y_2574_, v___y_2575_);
lean_dec(v___y_2575_);
lean_dec_ref(v___y_2574_);
lean_dec(v___y_2573_);
lean_dec_ref(v___y_2572_);
lean_dec_ref(v_range_2569_);
lean_dec(v___x_2564_);
return v_res_2577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__1(lean_object* v_m_2578_, lean_object* v_e_2579_, uint8_t v_mustUseBounds_2580_, lean_object* v_as_2581_, size_t v_sz_2582_, size_t v_i_2583_, lean_object* v_b_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_){
_start:
{
lean_object* v_a_2591_; uint8_t v___x_2595_; 
v___x_2595_ = lean_usize_dec_lt(v_i_2583_, v_sz_2582_);
if (v___x_2595_ == 0)
{
lean_object* v___x_2596_; 
lean_dec_ref(v_e_2579_);
lean_dec_ref(v_m_2578_);
v___x_2596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2596_, 0, v_b_2584_);
return v___x_2596_;
}
else
{
lean_object* v_a_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; 
v_a_2597_ = lean_array_uget_borrowed(v_as_2581_, v_i_2583_);
v___x_2598_ = lean_box(v___x_2595_);
lean_inc(v_a_2597_);
lean_inc_ref(v_e_2579_);
lean_inc_ref(v_m_2578_);
v___x_2599_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_getBound___boxed), 9, 4);
lean_closure_set(v___x_2599_, 0, v_m_2578_);
lean_closure_set(v___x_2599_, 1, v_e_2579_);
lean_closure_set(v___x_2599_, 2, v_a_2597_);
lean_closure_set(v___x_2599_, 3, v___x_2598_);
v___x_2600_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(v___x_2599_, v___y_2585_, v___y_2586_, v___y_2587_, v___y_2588_);
if (lean_obj_tag(v___x_2600_) == 0)
{
lean_object* v_a_2601_; 
v_a_2601_ = lean_ctor_get(v___x_2600_, 0);
lean_inc(v_a_2601_);
lean_dec_ref_known(v___x_2600_, 1);
if (lean_obj_tag(v_a_2601_) == 1)
{
if (lean_obj_tag(v_b_2584_) == 0)
{
v_a_2591_ = v_a_2601_;
goto v___jp_2590_;
}
else
{
lean_object* v_val_2602_; lean_object* v_val_2603_; lean_object* v_fst_2604_; lean_object* v_fst_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; uint8_t v___x_2608_; 
v_val_2602_ = lean_ctor_get(v_b_2584_, 0);
v_val_2603_ = lean_ctor_get(v_a_2601_, 0);
v_fst_2604_ = lean_ctor_get(v_val_2602_, 0);
v_fst_2605_ = lean_ctor_get(v_val_2603_, 0);
v___x_2606_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower(v_fst_2604_);
v___x_2607_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower(v_fst_2605_);
v___x_2608_ = lean_int_dec_lt(v___x_2606_, v___x_2607_);
lean_dec(v___x_2607_);
lean_dec(v___x_2606_);
if (v___x_2608_ == 0)
{
lean_dec_ref_known(v_a_2601_, 1);
v_a_2591_ = v_b_2584_;
goto v___jp_2590_;
}
else
{
lean_dec_ref_known(v_b_2584_, 1);
v_a_2591_ = v_a_2601_;
goto v___jp_2590_;
}
}
}
else
{
lean_dec(v_a_2601_);
if (v_mustUseBounds_2580_ == 0)
{
v_a_2591_ = v_b_2584_;
goto v___jp_2590_;
}
else
{
lean_object* v___x_2609_; 
lean_inc(v___y_2588_);
lean_inc_ref(v___y_2587_);
lean_inc(v___y_2586_);
lean_inc_ref(v___y_2585_);
lean_inc(v_a_2597_);
v___x_2609_ = lean_infer_type(v_a_2597_, v___y_2585_, v___y_2586_, v___y_2587_, v___y_2588_);
if (lean_obj_tag(v___x_2609_) == 0)
{
lean_object* v_a_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; lean_object* v___x_2616_; 
v_a_2610_ = lean_ctor_get(v___x_2609_, 0);
lean_inc(v_a_2610_);
lean_dec_ref_known(v___x_2609_, 1);
v___x_2611_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__1);
v___x_2612_ = l_Lean_MessageData_ofExpr(v_a_2610_);
v___x_2613_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2613_, 0, v___x_2611_);
lean_ctor_set(v___x_2613_, 1, v___x_2612_);
v___x_2614_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2___closed__3);
v___x_2615_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2615_, 0, v___x_2613_);
lean_ctor_set(v___x_2615_, 1, v___x_2614_);
v___x_2616_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_2615_, v___y_2585_, v___y_2586_, v___y_2587_, v___y_2588_);
if (lean_obj_tag(v___x_2616_) == 0)
{
lean_dec_ref_known(v___x_2616_, 1);
v_a_2591_ = v_b_2584_;
goto v___jp_2590_;
}
else
{
lean_object* v_a_2617_; lean_object* v___x_2619_; uint8_t v_isShared_2620_; uint8_t v_isSharedCheck_2624_; 
lean_dec(v_b_2584_);
lean_dec_ref(v_e_2579_);
lean_dec_ref(v_m_2578_);
v_a_2617_ = lean_ctor_get(v___x_2616_, 0);
v_isSharedCheck_2624_ = !lean_is_exclusive(v___x_2616_);
if (v_isSharedCheck_2624_ == 0)
{
v___x_2619_ = v___x_2616_;
v_isShared_2620_ = v_isSharedCheck_2624_;
goto v_resetjp_2618_;
}
else
{
lean_inc(v_a_2617_);
lean_dec(v___x_2616_);
v___x_2619_ = lean_box(0);
v_isShared_2620_ = v_isSharedCheck_2624_;
goto v_resetjp_2618_;
}
v_resetjp_2618_:
{
lean_object* v___x_2622_; 
if (v_isShared_2620_ == 0)
{
v___x_2622_ = v___x_2619_;
goto v_reusejp_2621_;
}
else
{
lean_object* v_reuseFailAlloc_2623_; 
v_reuseFailAlloc_2623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2623_, 0, v_a_2617_);
v___x_2622_ = v_reuseFailAlloc_2623_;
goto v_reusejp_2621_;
}
v_reusejp_2621_:
{
return v___x_2622_;
}
}
}
}
else
{
lean_object* v_a_2625_; lean_object* v___x_2627_; uint8_t v_isShared_2628_; uint8_t v_isSharedCheck_2632_; 
lean_dec(v_b_2584_);
lean_dec_ref(v_e_2579_);
lean_dec_ref(v_m_2578_);
v_a_2625_ = lean_ctor_get(v___x_2609_, 0);
v_isSharedCheck_2632_ = !lean_is_exclusive(v___x_2609_);
if (v_isSharedCheck_2632_ == 0)
{
v___x_2627_ = v___x_2609_;
v_isShared_2628_ = v_isSharedCheck_2632_;
goto v_resetjp_2626_;
}
else
{
lean_inc(v_a_2625_);
lean_dec(v___x_2609_);
v___x_2627_ = lean_box(0);
v_isShared_2628_ = v_isSharedCheck_2632_;
goto v_resetjp_2626_;
}
v_resetjp_2626_:
{
lean_object* v___x_2630_; 
if (v_isShared_2628_ == 0)
{
v___x_2630_ = v___x_2627_;
goto v_reusejp_2629_;
}
else
{
lean_object* v_reuseFailAlloc_2631_; 
v_reuseFailAlloc_2631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2631_, 0, v_a_2625_);
v___x_2630_ = v_reuseFailAlloc_2631_;
goto v_reusejp_2629_;
}
v_reusejp_2629_:
{
return v___x_2630_;
}
}
}
}
}
}
else
{
lean_dec(v_b_2584_);
lean_dec_ref(v_e_2579_);
lean_dec_ref(v_m_2578_);
return v___x_2600_;
}
}
v___jp_2590_:
{
size_t v___x_2592_; size_t v___x_2593_; 
v___x_2592_ = ((size_t)1ULL);
v___x_2593_ = lean_usize_add(v_i_2583_, v___x_2592_);
v_i_2583_ = v___x_2593_;
v_b_2584_ = v_a_2591_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__1___boxed(lean_object* v_m_2633_, lean_object* v_e_2634_, lean_object* v_mustUseBounds_2635_, lean_object* v_as_2636_, lean_object* v_sz_2637_, lean_object* v_i_2638_, lean_object* v_b_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_){
_start:
{
uint8_t v_mustUseBounds_boxed_2645_; size_t v_sz_boxed_2646_; size_t v_i_boxed_2647_; lean_object* v_res_2648_; 
v_mustUseBounds_boxed_2645_ = lean_unbox(v_mustUseBounds_2635_);
v_sz_boxed_2646_ = lean_unbox_usize(v_sz_2637_);
lean_dec(v_sz_2637_);
v_i_boxed_2647_ = lean_unbox_usize(v_i_2638_);
lean_dec(v_i_2638_);
v_res_2648_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__1(v_m_2633_, v_e_2634_, v_mustUseBounds_boxed_2645_, v_as_2636_, v_sz_boxed_2646_, v_i_boxed_2647_, v_b_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_);
lean_dec(v___y_2643_);
lean_dec_ref(v___y_2642_);
lean_dec(v___y_2641_);
lean_dec_ref(v___y_2640_);
lean_dec_ref(v_as_2636_);
return v_res_2648_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2650_; lean_object* v___x_2651_; 
v___x_2650_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__0));
v___x_2651_ = l_Lean_stringToMessageData(v___x_2650_);
return v___x_2651_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__3(void){
_start:
{
lean_object* v___x_2653_; lean_object* v___x_2654_; 
v___x_2653_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__2));
v___x_2654_ = l_Lean_stringToMessageData(v___x_2653_);
return v___x_2654_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__5(void){
_start:
{
lean_object* v___x_2656_; lean_object* v___x_2657_; 
v___x_2656_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__4));
v___x_2657_ = l_Lean_stringToMessageData(v___x_2656_);
return v___x_2657_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__8(void){
_start:
{
lean_object* v___x_2661_; lean_object* v___x_2662_; 
v___x_2661_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__7));
v___x_2662_ = l_Lean_stringToMessageData(v___x_2661_);
return v___x_2662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0(lean_object* v_e_2663_, lean_object* v_lbs_2664_, uint8_t v_mustUseBounds_2665_, lean_object* v_ubs_2666_, lean_object* v_e_x27_2667_, lean_object* v_g_2668_, lean_object* v___y_2669_, lean_object* v___y_2670_, lean_object* v___y_2671_, lean_object* v___y_2672_){
_start:
{
lean_object* v_m_2675_; lean_object* v___y_2676_; lean_object* v___y_2677_; lean_object* v___y_2678_; lean_object* v___y_2679_; lean_object* v___x_2840_; 
lean_inc(v___y_2672_);
lean_inc_ref(v___y_2671_);
lean_inc(v___y_2670_);
lean_inc_ref(v___y_2669_);
lean_inc_ref(v_e_2663_);
v___x_2840_ = lean_infer_type(v_e_2663_, v___y_2669_, v___y_2670_, v___y_2671_, v___y_2672_);
if (lean_obj_tag(v___x_2840_) == 0)
{
lean_object* v_a_2841_; lean_object* v___x_2842_; 
v_a_2841_ = lean_ctor_get(v___x_2840_, 0);
lean_inc(v_a_2841_);
lean_dec_ref_known(v___x_2840_, 1);
v___x_2842_ = l_Lean_Meta_whnfR(v_a_2841_, v___y_2669_, v___y_2670_, v___y_2671_, v___y_2672_);
if (lean_obj_tag(v___x_2842_) == 0)
{
lean_object* v_a_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; uint8_t v___x_2846_; 
v_a_2843_ = lean_ctor_get(v___x_2842_, 0);
lean_inc(v_a_2843_);
lean_dec_ref_known(v___x_2842_, 1);
v___x_2844_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_natMethods_spec__0___closed__0));
v___x_2845_ = l_Lean_Name_mkStr1(v___x_2844_);
v___x_2846_ = l_Lean_Expr_isConstOf(v_a_2843_, v___x_2845_);
lean_dec(v___x_2845_);
if (v___x_2846_ == 0)
{
lean_object* v___x_2847_; lean_object* v___x_2848_; uint8_t v___x_2849_; 
v___x_2847_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_IntervalCases_intMethods_spec__0___closed__0));
v___x_2848_ = l_Lean_Name_mkStr1(v___x_2847_);
v___x_2849_ = l_Lean_Expr_isConstOf(v_a_2843_, v___x_2848_);
lean_dec(v___x_2848_);
if (v___x_2849_ == 0)
{
lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v_a_2854_; lean_object* v___x_2856_; uint8_t v_isShared_2857_; uint8_t v_isSharedCheck_2861_; 
lean_dec(v_g_2668_);
lean_dec_ref(v_e_x27_2667_);
lean_dec_ref(v_e_2663_);
v___x_2850_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__8, &lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__8);
v___x_2851_ = l_Lean_MessageData_ofExpr(v_a_2843_);
v___x_2852_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2852_, 0, v___x_2850_);
lean_ctor_set(v___x_2852_, 1, v___x_2851_);
v___x_2853_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_2852_, v___y_2669_, v___y_2670_, v___y_2671_, v___y_2672_);
lean_dec(v___y_2672_);
lean_dec_ref(v___y_2671_);
lean_dec(v___y_2670_);
lean_dec_ref(v___y_2669_);
v_a_2854_ = lean_ctor_get(v___x_2853_, 0);
v_isSharedCheck_2861_ = !lean_is_exclusive(v___x_2853_);
if (v_isSharedCheck_2861_ == 0)
{
v___x_2856_ = v___x_2853_;
v_isShared_2857_ = v_isSharedCheck_2861_;
goto v_resetjp_2855_;
}
else
{
lean_inc(v_a_2854_);
lean_dec(v___x_2853_);
v___x_2856_ = lean_box(0);
v_isShared_2857_ = v_isSharedCheck_2861_;
goto v_resetjp_2855_;
}
v_resetjp_2855_:
{
lean_object* v___x_2859_; 
if (v_isShared_2857_ == 0)
{
v___x_2859_ = v___x_2856_;
goto v_reusejp_2858_;
}
else
{
lean_object* v_reuseFailAlloc_2860_; 
v_reuseFailAlloc_2860_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2860_, 0, v_a_2854_);
v___x_2859_ = v_reuseFailAlloc_2860_;
goto v_reusejp_2858_;
}
v_reusejp_2858_:
{
return v___x_2859_;
}
}
}
else
{
lean_object* v___x_2862_; 
lean_dec(v_a_2843_);
v___x_2862_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intMethods));
v_m_2675_ = v___x_2862_;
v___y_2676_ = v___y_2669_;
v___y_2677_ = v___y_2670_;
v___y_2678_ = v___y_2671_;
v___y_2679_ = v___y_2672_;
goto v___jp_2674_;
}
}
else
{
lean_object* v___x_2863_; 
lean_dec(v_a_2843_);
v___x_2863_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_natMethods));
v_m_2675_ = v___x_2863_;
v___y_2676_ = v___y_2669_;
v___y_2677_ = v___y_2670_;
v___y_2678_ = v___y_2671_;
v___y_2679_ = v___y_2672_;
goto v___jp_2674_;
}
}
else
{
lean_object* v_a_2864_; lean_object* v___x_2866_; uint8_t v_isShared_2867_; uint8_t v_isSharedCheck_2871_; 
lean_dec(v___y_2672_);
lean_dec_ref(v___y_2671_);
lean_dec(v___y_2670_);
lean_dec_ref(v___y_2669_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_x27_2667_);
lean_dec_ref(v_e_2663_);
v_a_2864_ = lean_ctor_get(v___x_2842_, 0);
v_isSharedCheck_2871_ = !lean_is_exclusive(v___x_2842_);
if (v_isSharedCheck_2871_ == 0)
{
v___x_2866_ = v___x_2842_;
v_isShared_2867_ = v_isSharedCheck_2871_;
goto v_resetjp_2865_;
}
else
{
lean_inc(v_a_2864_);
lean_dec(v___x_2842_);
v___x_2866_ = lean_box(0);
v_isShared_2867_ = v_isSharedCheck_2871_;
goto v_resetjp_2865_;
}
v_resetjp_2865_:
{
lean_object* v___x_2869_; 
if (v_isShared_2867_ == 0)
{
v___x_2869_ = v___x_2866_;
goto v_reusejp_2868_;
}
else
{
lean_object* v_reuseFailAlloc_2870_; 
v_reuseFailAlloc_2870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2870_, 0, v_a_2864_);
v___x_2869_ = v_reuseFailAlloc_2870_;
goto v_reusejp_2868_;
}
v_reusejp_2868_:
{
return v___x_2869_;
}
}
}
}
else
{
lean_object* v_a_2872_; lean_object* v___x_2874_; uint8_t v_isShared_2875_; uint8_t v_isSharedCheck_2879_; 
lean_dec(v___y_2672_);
lean_dec_ref(v___y_2671_);
lean_dec(v___y_2670_);
lean_dec_ref(v___y_2669_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_x27_2667_);
lean_dec_ref(v_e_2663_);
v_a_2872_ = lean_ctor_get(v___x_2840_, 0);
v_isSharedCheck_2879_ = !lean_is_exclusive(v___x_2840_);
if (v_isSharedCheck_2879_ == 0)
{
v___x_2874_ = v___x_2840_;
v_isShared_2875_ = v_isSharedCheck_2879_;
goto v_resetjp_2873_;
}
else
{
lean_inc(v_a_2872_);
lean_dec(v___x_2840_);
v___x_2874_ = lean_box(0);
v_isShared_2875_ = v_isSharedCheck_2879_;
goto v_resetjp_2873_;
}
v_resetjp_2873_:
{
lean_object* v___x_2877_; 
if (v_isShared_2875_ == 0)
{
v___x_2877_ = v___x_2874_;
goto v_reusejp_2876_;
}
else
{
lean_object* v_reuseFailAlloc_2878_; 
v_reuseFailAlloc_2878_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2878_, 0, v_a_2872_);
v___x_2877_ = v_reuseFailAlloc_2878_;
goto v_reusejp_2876_;
}
v_reusejp_2876_:
{
return v___x_2877_;
}
}
}
v___jp_2674_:
{
lean_object* v_initLB_2680_; lean_object* v_initUB_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; 
v_initLB_2680_ = lean_ctor_get(v_m_2675_, 0);
v_initUB_2681_ = lean_ctor_get(v_m_2675_, 1);
lean_inc_ref(v_initLB_2680_);
lean_inc_ref(v_e_2663_);
v___x_2682_ = lean_apply_1(v_initLB_2680_, v_e_2663_);
v___x_2683_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(v___x_2682_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2683_) == 0)
{
lean_object* v_a_2684_; size_t v_sz_2685_; size_t v___x_2686_; lean_object* v___x_2687_; 
v_a_2684_ = lean_ctor_get(v___x_2683_, 0);
lean_inc(v_a_2684_);
lean_dec_ref_known(v___x_2683_, 1);
v_sz_2685_ = lean_array_size(v_lbs_2664_);
v___x_2686_ = ((size_t)0ULL);
lean_inc_ref(v_e_2663_);
lean_inc_ref(v_m_2675_);
v___x_2687_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__1(v_m_2675_, v_e_2663_, v_mustUseBounds_2665_, v_lbs_2664_, v_sz_2685_, v___x_2686_, v_a_2684_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2687_) == 0)
{
lean_object* v_a_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; 
v_a_2688_ = lean_ctor_get(v___x_2687_, 0);
lean_inc(v_a_2688_);
lean_dec_ref_known(v___x_2687_, 1);
lean_inc_ref(v_initUB_2681_);
lean_inc_ref(v_e_2663_);
v___x_2689_ = lean_apply_1(v_initUB_2681_, v_e_2663_);
v___x_2690_ = lp_mathlib_try_x3f___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__0___redArg(v___x_2689_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2690_) == 0)
{
lean_object* v_a_2691_; size_t v_sz_2692_; lean_object* v___x_2693_; 
v_a_2691_ = lean_ctor_get(v___x_2690_, 0);
lean_inc(v_a_2691_);
lean_dec_ref_known(v___x_2690_, 1);
v_sz_2692_ = lean_array_size(v_ubs_2666_);
lean_inc_ref(v_e_2663_);
lean_inc_ref(v_m_2675_);
v___x_2693_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__2(v_m_2675_, v_e_2663_, v_mustUseBounds_2665_, v_ubs_2666_, v_sz_2692_, v___x_2686_, v_a_2691_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2693_) == 0)
{
if (lean_obj_tag(v_a_2688_) == 0)
{
lean_object* v_a_2694_; 
lean_dec(v_g_2668_);
lean_dec_ref(v_e_2663_);
v_a_2694_ = lean_ctor_get(v___x_2693_, 0);
lean_inc(v_a_2694_);
lean_dec_ref_known(v___x_2693_, 1);
if (lean_obj_tag(v_a_2694_) == 0)
{
lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; 
v___x_2695_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__1);
v___x_2696_ = l_Lean_MessageData_ofExpr(v_e_x27_2667_);
v___x_2697_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2697_, 0, v___x_2695_);
lean_ctor_set(v___x_2697_, 1, v___x_2696_);
v___x_2698_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_2697_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
return v___x_2698_;
}
else
{
lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; 
lean_dec_ref_known(v_a_2694_, 1);
v___x_2699_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__3);
v___x_2700_ = l_Lean_MessageData_ofExpr(v_e_x27_2667_);
v___x_2701_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2701_, 0, v___x_2699_);
lean_ctor_set(v___x_2701_, 1, v___x_2700_);
v___x_2702_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_2701_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
return v___x_2702_;
}
}
else
{
lean_object* v_val_2703_; lean_object* v_snd_2704_; lean_object* v_a_2705_; 
v_val_2703_ = lean_ctor_get(v_a_2688_, 0);
lean_inc(v_val_2703_);
lean_dec_ref_known(v_a_2688_, 1);
v_snd_2704_ = lean_ctor_get(v_val_2703_, 1);
lean_inc(v_snd_2704_);
v_a_2705_ = lean_ctor_get(v___x_2693_, 0);
lean_inc(v_a_2705_);
lean_dec_ref_known(v___x_2693_, 1);
if (lean_obj_tag(v_a_2705_) == 0)
{
lean_object* v___x_2707_; uint8_t v_isShared_2708_; uint8_t v_isSharedCheck_2715_; 
lean_dec(v_val_2703_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_2663_);
v_isSharedCheck_2715_ = !lean_is_exclusive(v_snd_2704_);
if (v_isSharedCheck_2715_ == 0)
{
lean_object* v_unused_2716_; lean_object* v_unused_2717_; 
v_unused_2716_ = lean_ctor_get(v_snd_2704_, 1);
lean_dec(v_unused_2716_);
v_unused_2717_ = lean_ctor_get(v_snd_2704_, 0);
lean_dec(v_unused_2717_);
v___x_2707_ = v_snd_2704_;
v_isShared_2708_ = v_isSharedCheck_2715_;
goto v_resetjp_2706_;
}
else
{
lean_dec(v_snd_2704_);
v___x_2707_ = lean_box(0);
v_isShared_2708_ = v_isSharedCheck_2715_;
goto v_resetjp_2706_;
}
v_resetjp_2706_:
{
lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2712_; 
v___x_2709_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__5);
v___x_2710_ = l_Lean_MessageData_ofExpr(v_e_x27_2667_);
if (v_isShared_2708_ == 0)
{
lean_ctor_set_tag(v___x_2707_, 7);
lean_ctor_set(v___x_2707_, 1, v___x_2710_);
lean_ctor_set(v___x_2707_, 0, v___x_2709_);
v___x_2712_ = v___x_2707_;
goto v_reusejp_2711_;
}
else
{
lean_object* v_reuseFailAlloc_2714_; 
v_reuseFailAlloc_2714_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2714_, 0, v___x_2709_);
lean_ctor_set(v_reuseFailAlloc_2714_, 1, v___x_2710_);
v___x_2712_ = v_reuseFailAlloc_2714_;
goto v_reusejp_2711_;
}
v_reusejp_2711_:
{
lean_object* v___x_2713_; 
v___x_2713_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0___redArg(v___x_2712_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
return v___x_2713_;
}
}
}
else
{
lean_object* v_val_2718_; lean_object* v_snd_2719_; lean_object* v_fst_2720_; lean_object* v_fst_2721_; lean_object* v_snd_2722_; lean_object* v_fst_2723_; lean_object* v_fst_2724_; lean_object* v_snd_2725_; lean_object* v___x_2726_; lean_object* v___x_2727_; uint8_t v___x_2728_; 
lean_dec_ref(v_e_x27_2667_);
v_val_2718_ = lean_ctor_get(v_a_2705_, 0);
lean_inc(v_val_2718_);
lean_dec_ref_known(v_a_2705_, 1);
v_snd_2719_ = lean_ctor_get(v_val_2718_, 1);
lean_inc(v_snd_2719_);
v_fst_2720_ = lean_ctor_get(v_val_2703_, 0);
lean_inc(v_fst_2720_);
lean_dec(v_val_2703_);
v_fst_2721_ = lean_ctor_get(v_snd_2704_, 0);
lean_inc(v_fst_2721_);
v_snd_2722_ = lean_ctor_get(v_snd_2704_, 1);
lean_inc(v_snd_2722_);
lean_dec(v_snd_2704_);
v_fst_2723_ = lean_ctor_get(v_val_2718_, 0);
lean_inc(v_fst_2723_);
lean_dec(v_val_2718_);
v_fst_2724_ = lean_ctor_get(v_snd_2719_, 0);
lean_inc(v_fst_2724_);
v_snd_2725_ = lean_ctor_get(v_snd_2719_, 1);
lean_inc(v_snd_2725_);
lean_dec(v_snd_2719_);
v___x_2726_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asUpper(v_fst_2723_);
v___x_2727_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower(v_fst_2720_);
v___x_2728_ = lean_int_dec_lt(v___x_2726_, v___x_2727_);
if (v___x_2728_ == 0)
{
lean_object* v___x_2729_; 
lean_inc(v_g_2668_);
v___x_2729_ = l_Lean_MVarId_getType(v_g_2668_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2729_) == 0)
{
lean_object* v_a_2730_; lean_object* v___x_2731_; 
v_a_2730_ = lean_ctor_get(v___x_2729_, 0);
lean_inc(v_a_2730_);
lean_dec_ref_known(v___x_2729_, 1);
lean_inc(v_g_2668_);
v___x_2731_ = l_Lean_MVarId_getTag(v_g_2668_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2731_) == 0)
{
lean_object* v_a_2732_; lean_object* v___x_2733_; lean_object* v___x_2734_; lean_object* v___x_2735_; lean_object* v___x_2736_; lean_object* v___x_2737_; lean_object* v___x_2738_; lean_object* v___x_2739_; lean_object* v___x_2740_; lean_object* v___x_2741_; 
v_a_2732_ = lean_ctor_get(v___x_2731_, 0);
lean_inc(v_a_2732_);
lean_dec_ref_known(v___x_2731_, 1);
v___x_2733_ = lean_unsigned_to_nat(0u);
v___x_2734_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__6));
v___x_2735_ = lean_int_sub(v___x_2726_, v___x_2727_);
lean_dec(v___x_2726_);
v___x_2736_ = lean_unsigned_to_nat(1u);
v___x_2737_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0, &lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_IntervalCases_Bound_asLower___closed__0);
v___x_2738_ = lean_int_add(v___x_2735_, v___x_2737_);
lean_dec(v___x_2735_);
v___x_2739_ = l_Int_toNat(v___x_2738_);
lean_dec(v___x_2738_);
v___x_2740_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2740_, 0, v___x_2733_);
lean_ctor_set(v___x_2740_, 1, v___x_2739_);
lean_ctor_set(v___x_2740_, 2, v___x_2736_);
lean_inc_ref(v_e_2663_);
lean_inc_ref(v_m_2675_);
v___x_2741_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___redArg(v___x_2727_, v_m_2675_, v_e_2663_, v_a_2730_, v_a_2732_, v___x_2740_, v___x_2734_, v___x_2733_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
lean_dec_ref_known(v___x_2740_, 3);
lean_dec(v___x_2727_);
if (lean_obj_tag(v___x_2741_) == 0)
{
lean_object* v_a_2742_; lean_object* v___x_2743_; lean_object* v___x_2744_; lean_object* v___x_2745_; 
v_a_2742_ = lean_ctor_get(v___x_2741_, 0);
lean_inc_n(v_a_2742_, 2);
lean_dec_ref_known(v___x_2741_, 1);
v___x_2743_ = lean_array_get_size(v_a_2742_);
v___x_2744_ = l_Array_toSubarray___redArg(v_a_2742_, v___x_2733_, v___x_2743_);
lean_inc_ref(v_m_2675_);
v___x_2745_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_bisect(v_m_2675_, v_g_2668_, v___x_2744_, v_fst_2720_, v_fst_2723_, v_fst_2721_, v_fst_2724_, v_snd_2722_, v_snd_2725_, v_e_2663_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
if (lean_obj_tag(v___x_2745_) == 0)
{
lean_object* v___x_2747_; uint8_t v_isShared_2748_; uint8_t v_isSharedCheck_2752_; 
v_isSharedCheck_2752_ = !lean_is_exclusive(v___x_2745_);
if (v_isSharedCheck_2752_ == 0)
{
lean_object* v_unused_2753_; 
v_unused_2753_ = lean_ctor_get(v___x_2745_, 0);
lean_dec(v_unused_2753_);
v___x_2747_ = v___x_2745_;
v_isShared_2748_ = v_isSharedCheck_2752_;
goto v_resetjp_2746_;
}
else
{
lean_dec(v___x_2745_);
v___x_2747_ = lean_box(0);
v_isShared_2748_ = v_isSharedCheck_2752_;
goto v_resetjp_2746_;
}
v_resetjp_2746_:
{
lean_object* v___x_2750_; 
if (v_isShared_2748_ == 0)
{
lean_ctor_set(v___x_2747_, 0, v_a_2742_);
v___x_2750_ = v___x_2747_;
goto v_reusejp_2749_;
}
else
{
lean_object* v_reuseFailAlloc_2751_; 
v_reuseFailAlloc_2751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2751_, 0, v_a_2742_);
v___x_2750_ = v_reuseFailAlloc_2751_;
goto v_reusejp_2749_;
}
v_reusejp_2749_:
{
return v___x_2750_;
}
}
}
else
{
lean_object* v_a_2754_; lean_object* v___x_2756_; uint8_t v_isShared_2757_; uint8_t v_isSharedCheck_2761_; 
lean_dec(v_a_2742_);
v_a_2754_ = lean_ctor_get(v___x_2745_, 0);
v_isSharedCheck_2761_ = !lean_is_exclusive(v___x_2745_);
if (v_isSharedCheck_2761_ == 0)
{
v___x_2756_ = v___x_2745_;
v_isShared_2757_ = v_isSharedCheck_2761_;
goto v_resetjp_2755_;
}
else
{
lean_inc(v_a_2754_);
lean_dec(v___x_2745_);
v___x_2756_ = lean_box(0);
v_isShared_2757_ = v_isSharedCheck_2761_;
goto v_resetjp_2755_;
}
v_resetjp_2755_:
{
lean_object* v___x_2759_; 
if (v_isShared_2757_ == 0)
{
v___x_2759_ = v___x_2756_;
goto v_reusejp_2758_;
}
else
{
lean_object* v_reuseFailAlloc_2760_; 
v_reuseFailAlloc_2760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2760_, 0, v_a_2754_);
v___x_2759_ = v_reuseFailAlloc_2760_;
goto v_reusejp_2758_;
}
v_reusejp_2758_:
{
return v___x_2759_;
}
}
}
}
else
{
lean_dec(v_snd_2725_);
lean_dec(v_fst_2724_);
lean_dec(v_fst_2723_);
lean_dec(v_snd_2722_);
lean_dec(v_fst_2721_);
lean_dec(v_fst_2720_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_2663_);
return v___x_2741_;
}
}
else
{
lean_object* v_a_2762_; lean_object* v___x_2764_; uint8_t v_isShared_2765_; uint8_t v_isSharedCheck_2769_; 
lean_dec(v_a_2730_);
lean_dec(v___x_2727_);
lean_dec(v___x_2726_);
lean_dec(v_snd_2725_);
lean_dec(v_fst_2724_);
lean_dec(v_fst_2723_);
lean_dec(v_snd_2722_);
lean_dec(v_fst_2721_);
lean_dec(v_fst_2720_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_2663_);
v_a_2762_ = lean_ctor_get(v___x_2731_, 0);
v_isSharedCheck_2769_ = !lean_is_exclusive(v___x_2731_);
if (v_isSharedCheck_2769_ == 0)
{
v___x_2764_ = v___x_2731_;
v_isShared_2765_ = v_isSharedCheck_2769_;
goto v_resetjp_2763_;
}
else
{
lean_inc(v_a_2762_);
lean_dec(v___x_2731_);
v___x_2764_ = lean_box(0);
v_isShared_2765_ = v_isSharedCheck_2769_;
goto v_resetjp_2763_;
}
v_resetjp_2763_:
{
lean_object* v___x_2767_; 
if (v_isShared_2765_ == 0)
{
v___x_2767_ = v___x_2764_;
goto v_reusejp_2766_;
}
else
{
lean_object* v_reuseFailAlloc_2768_; 
v_reuseFailAlloc_2768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2768_, 0, v_a_2762_);
v___x_2767_ = v_reuseFailAlloc_2768_;
goto v_reusejp_2766_;
}
v_reusejp_2766_:
{
return v___x_2767_;
}
}
}
}
else
{
lean_object* v_a_2770_; lean_object* v___x_2772_; uint8_t v_isShared_2773_; uint8_t v_isSharedCheck_2777_; 
lean_dec(v___x_2727_);
lean_dec(v___x_2726_);
lean_dec(v_snd_2725_);
lean_dec(v_fst_2724_);
lean_dec(v_fst_2723_);
lean_dec(v_snd_2722_);
lean_dec(v_fst_2721_);
lean_dec(v_fst_2720_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_2663_);
v_a_2770_ = lean_ctor_get(v___x_2729_, 0);
v_isSharedCheck_2777_ = !lean_is_exclusive(v___x_2729_);
if (v_isSharedCheck_2777_ == 0)
{
v___x_2772_ = v___x_2729_;
v_isShared_2773_ = v_isSharedCheck_2777_;
goto v_resetjp_2771_;
}
else
{
lean_inc(v_a_2770_);
lean_dec(v___x_2729_);
v___x_2772_ = lean_box(0);
v_isShared_2773_ = v_isSharedCheck_2777_;
goto v_resetjp_2771_;
}
v_resetjp_2771_:
{
lean_object* v___x_2775_; 
if (v_isShared_2773_ == 0)
{
v___x_2775_ = v___x_2772_;
goto v_reusejp_2774_;
}
else
{
lean_object* v_reuseFailAlloc_2776_; 
v_reuseFailAlloc_2776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2776_, 0, v_a_2770_);
v___x_2775_ = v_reuseFailAlloc_2776_;
goto v_reusejp_2774_;
}
v_reusejp_2774_:
{
return v___x_2775_;
}
}
}
}
else
{
lean_object* v___x_2778_; 
lean_dec(v___x_2727_);
lean_dec(v___x_2726_);
v___x_2778_ = l_Lean_MVarId_exfalso(v_g_2668_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
if (lean_obj_tag(v___x_2778_) == 0)
{
lean_object* v_a_2779_; lean_object* v___x_2780_; 
v_a_2779_ = lean_ctor_get(v___x_2778_, 0);
lean_inc(v_a_2779_);
lean_dec_ref_known(v___x_2778_, 1);
lean_inc_ref(v_m_2675_);
v___x_2780_ = lp_mathlib_Mathlib_Tactic_IntervalCases_Methods_inconsistentBounds(v_m_2675_, v_fst_2720_, v_fst_2723_, v_fst_2721_, v_fst_2724_, v_snd_2722_, v_snd_2725_, v_e_2663_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec_ref(v___y_2676_);
lean_dec(v_fst_2720_);
if (lean_obj_tag(v___x_2780_) == 0)
{
lean_object* v_a_2781_; lean_object* v___x_2782_; lean_object* v___x_2784_; uint8_t v_isShared_2785_; uint8_t v_isSharedCheck_2790_; 
v_a_2781_ = lean_ctor_get(v___x_2780_, 0);
lean_inc(v_a_2781_);
lean_dec_ref_known(v___x_2780_, 1);
v___x_2782_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__1___redArg(v_a_2779_, v_a_2781_, v___y_2677_);
lean_dec(v___y_2677_);
v_isSharedCheck_2790_ = !lean_is_exclusive(v___x_2782_);
if (v_isSharedCheck_2790_ == 0)
{
lean_object* v_unused_2791_; 
v_unused_2791_ = lean_ctor_get(v___x_2782_, 0);
lean_dec(v_unused_2791_);
v___x_2784_ = v___x_2782_;
v_isShared_2785_ = v_isSharedCheck_2790_;
goto v_resetjp_2783_;
}
else
{
lean_dec(v___x_2782_);
v___x_2784_ = lean_box(0);
v_isShared_2785_ = v_isSharedCheck_2790_;
goto v_resetjp_2783_;
}
v_resetjp_2783_:
{
lean_object* v___x_2786_; lean_object* v___x_2788_; 
v___x_2786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___closed__6));
if (v_isShared_2785_ == 0)
{
lean_ctor_set(v___x_2784_, 0, v___x_2786_);
v___x_2788_ = v___x_2784_;
goto v_reusejp_2787_;
}
else
{
lean_object* v_reuseFailAlloc_2789_; 
v_reuseFailAlloc_2789_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2789_, 0, v___x_2786_);
v___x_2788_ = v_reuseFailAlloc_2789_;
goto v_reusejp_2787_;
}
v_reusejp_2787_:
{
return v___x_2788_;
}
}
}
else
{
lean_object* v_a_2792_; lean_object* v___x_2794_; uint8_t v_isShared_2795_; uint8_t v_isSharedCheck_2799_; 
lean_dec(v_a_2779_);
lean_dec(v___y_2677_);
v_a_2792_ = lean_ctor_get(v___x_2780_, 0);
v_isSharedCheck_2799_ = !lean_is_exclusive(v___x_2780_);
if (v_isSharedCheck_2799_ == 0)
{
v___x_2794_ = v___x_2780_;
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
else
{
lean_inc(v_a_2792_);
lean_dec(v___x_2780_);
v___x_2794_ = lean_box(0);
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
v_resetjp_2793_:
{
lean_object* v___x_2797_; 
if (v_isShared_2795_ == 0)
{
v___x_2797_ = v___x_2794_;
goto v_reusejp_2796_;
}
else
{
lean_object* v_reuseFailAlloc_2798_; 
v_reuseFailAlloc_2798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2798_, 0, v_a_2792_);
v___x_2797_ = v_reuseFailAlloc_2798_;
goto v_reusejp_2796_;
}
v_reusejp_2796_:
{
return v___x_2797_;
}
}
}
}
else
{
lean_object* v_a_2800_; lean_object* v___x_2802_; uint8_t v_isShared_2803_; uint8_t v_isSharedCheck_2807_; 
lean_dec(v_snd_2725_);
lean_dec(v_fst_2724_);
lean_dec(v_fst_2723_);
lean_dec(v_snd_2722_);
lean_dec(v_fst_2721_);
lean_dec(v_fst_2720_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec_ref(v_e_2663_);
v_a_2800_ = lean_ctor_get(v___x_2778_, 0);
v_isSharedCheck_2807_ = !lean_is_exclusive(v___x_2778_);
if (v_isSharedCheck_2807_ == 0)
{
v___x_2802_ = v___x_2778_;
v_isShared_2803_ = v_isSharedCheck_2807_;
goto v_resetjp_2801_;
}
else
{
lean_inc(v_a_2800_);
lean_dec(v___x_2778_);
v___x_2802_ = lean_box(0);
v_isShared_2803_ = v_isSharedCheck_2807_;
goto v_resetjp_2801_;
}
v_resetjp_2801_:
{
lean_object* v___x_2805_; 
if (v_isShared_2803_ == 0)
{
v___x_2805_ = v___x_2802_;
goto v_reusejp_2804_;
}
else
{
lean_object* v_reuseFailAlloc_2806_; 
v_reuseFailAlloc_2806_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2806_, 0, v_a_2800_);
v___x_2805_ = v_reuseFailAlloc_2806_;
goto v_reusejp_2804_;
}
v_reusejp_2804_:
{
return v___x_2805_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2808_; lean_object* v___x_2810_; uint8_t v_isShared_2811_; uint8_t v_isSharedCheck_2815_; 
lean_dec(v_a_2688_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_x27_2667_);
lean_dec_ref(v_e_2663_);
v_a_2808_ = lean_ctor_get(v___x_2693_, 0);
v_isSharedCheck_2815_ = !lean_is_exclusive(v___x_2693_);
if (v_isSharedCheck_2815_ == 0)
{
v___x_2810_ = v___x_2693_;
v_isShared_2811_ = v_isSharedCheck_2815_;
goto v_resetjp_2809_;
}
else
{
lean_inc(v_a_2808_);
lean_dec(v___x_2693_);
v___x_2810_ = lean_box(0);
v_isShared_2811_ = v_isSharedCheck_2815_;
goto v_resetjp_2809_;
}
v_resetjp_2809_:
{
lean_object* v___x_2813_; 
if (v_isShared_2811_ == 0)
{
v___x_2813_ = v___x_2810_;
goto v_reusejp_2812_;
}
else
{
lean_object* v_reuseFailAlloc_2814_; 
v_reuseFailAlloc_2814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2814_, 0, v_a_2808_);
v___x_2813_ = v_reuseFailAlloc_2814_;
goto v_reusejp_2812_;
}
v_reusejp_2812_:
{
return v___x_2813_;
}
}
}
}
else
{
lean_object* v_a_2816_; lean_object* v___x_2818_; uint8_t v_isShared_2819_; uint8_t v_isSharedCheck_2823_; 
lean_dec(v_a_2688_);
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_x27_2667_);
lean_dec_ref(v_e_2663_);
v_a_2816_ = lean_ctor_get(v___x_2690_, 0);
v_isSharedCheck_2823_ = !lean_is_exclusive(v___x_2690_);
if (v_isSharedCheck_2823_ == 0)
{
v___x_2818_ = v___x_2690_;
v_isShared_2819_ = v_isSharedCheck_2823_;
goto v_resetjp_2817_;
}
else
{
lean_inc(v_a_2816_);
lean_dec(v___x_2690_);
v___x_2818_ = lean_box(0);
v_isShared_2819_ = v_isSharedCheck_2823_;
goto v_resetjp_2817_;
}
v_resetjp_2817_:
{
lean_object* v___x_2821_; 
if (v_isShared_2819_ == 0)
{
v___x_2821_ = v___x_2818_;
goto v_reusejp_2820_;
}
else
{
lean_object* v_reuseFailAlloc_2822_; 
v_reuseFailAlloc_2822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2822_, 0, v_a_2816_);
v___x_2821_ = v_reuseFailAlloc_2822_;
goto v_reusejp_2820_;
}
v_reusejp_2820_:
{
return v___x_2821_;
}
}
}
}
else
{
lean_object* v_a_2824_; lean_object* v___x_2826_; uint8_t v_isShared_2827_; uint8_t v_isSharedCheck_2831_; 
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_x27_2667_);
lean_dec_ref(v_e_2663_);
v_a_2824_ = lean_ctor_get(v___x_2687_, 0);
v_isSharedCheck_2831_ = !lean_is_exclusive(v___x_2687_);
if (v_isSharedCheck_2831_ == 0)
{
v___x_2826_ = v___x_2687_;
v_isShared_2827_ = v_isSharedCheck_2831_;
goto v_resetjp_2825_;
}
else
{
lean_inc(v_a_2824_);
lean_dec(v___x_2687_);
v___x_2826_ = lean_box(0);
v_isShared_2827_ = v_isSharedCheck_2831_;
goto v_resetjp_2825_;
}
v_resetjp_2825_:
{
lean_object* v___x_2829_; 
if (v_isShared_2827_ == 0)
{
v___x_2829_ = v___x_2826_;
goto v_reusejp_2828_;
}
else
{
lean_object* v_reuseFailAlloc_2830_; 
v_reuseFailAlloc_2830_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2830_, 0, v_a_2824_);
v___x_2829_ = v_reuseFailAlloc_2830_;
goto v_reusejp_2828_;
}
v_reusejp_2828_:
{
return v___x_2829_;
}
}
}
}
else
{
lean_object* v_a_2832_; lean_object* v___x_2834_; uint8_t v_isShared_2835_; uint8_t v_isSharedCheck_2839_; 
lean_dec(v___y_2679_);
lean_dec_ref(v___y_2678_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v_g_2668_);
lean_dec_ref(v_e_x27_2667_);
lean_dec_ref(v_e_2663_);
v_a_2832_ = lean_ctor_get(v___x_2683_, 0);
v_isSharedCheck_2839_ = !lean_is_exclusive(v___x_2683_);
if (v_isSharedCheck_2839_ == 0)
{
v___x_2834_ = v___x_2683_;
v_isShared_2835_ = v_isSharedCheck_2839_;
goto v_resetjp_2833_;
}
else
{
lean_inc(v_a_2832_);
lean_dec(v___x_2683_);
v___x_2834_ = lean_box(0);
v_isShared_2835_ = v_isSharedCheck_2839_;
goto v_resetjp_2833_;
}
v_resetjp_2833_:
{
lean_object* v___x_2837_; 
if (v_isShared_2835_ == 0)
{
v___x_2837_ = v___x_2834_;
goto v_reusejp_2836_;
}
else
{
lean_object* v_reuseFailAlloc_2838_; 
v_reuseFailAlloc_2838_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2838_, 0, v_a_2832_);
v___x_2837_ = v_reuseFailAlloc_2838_;
goto v_reusejp_2836_;
}
v_reusejp_2836_:
{
return v___x_2837_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___boxed(lean_object* v_e_2880_, lean_object* v_lbs_2881_, lean_object* v_mustUseBounds_2882_, lean_object* v_ubs_2883_, lean_object* v_e_x27_2884_, lean_object* v_g_2885_, lean_object* v___y_2886_, lean_object* v___y_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_){
_start:
{
uint8_t v_mustUseBounds_boxed_2891_; lean_object* v_res_2892_; 
v_mustUseBounds_boxed_2891_ = lean_unbox(v_mustUseBounds_2882_);
v_res_2892_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0(v_e_2880_, v_lbs_2881_, v_mustUseBounds_boxed_2891_, v_ubs_2883_, v_e_x27_2884_, v_g_2885_, v___y_2886_, v___y_2887_, v___y_2888_, v___y_2889_);
lean_dec_ref(v_ubs_2883_);
lean_dec_ref(v_lbs_2881_);
return v_res_2892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases(lean_object* v_g_2893_, lean_object* v_e_2894_, lean_object* v_e_x27_2895_, lean_object* v_lbs_2896_, lean_object* v_ubs_2897_, uint8_t v_mustUseBounds_2898_, lean_object* v_a_2899_, lean_object* v_a_2900_, lean_object* v_a_2901_, lean_object* v_a_2902_){
_start:
{
lean_object* v___x_2904_; lean_object* v___f_2905_; lean_object* v___x_2906_; 
v___x_2904_ = lean_box(v_mustUseBounds_2898_);
lean_inc(v_g_2893_);
v___f_2905_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___lam__0___boxed), 11, 6);
lean_closure_set(v___f_2905_, 0, v_e_2894_);
lean_closure_set(v___f_2905_, 1, v_lbs_2896_);
lean_closure_set(v___f_2905_, 2, v___x_2904_);
lean_closure_set(v___f_2905_, 3, v_ubs_2897_);
lean_closure_set(v___f_2905_, 4, v_e_x27_2895_);
lean_closure_set(v___f_2905_, 5, v_g_2893_);
v___x_2906_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_IntervalCases_Methods_bisect_spec__3___redArg(v_g_2893_, v___f_2905_, v_a_2899_, v_a_2900_, v_a_2901_, v_a_2902_);
return v___x_2906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases___boxed(lean_object* v_g_2907_, lean_object* v_e_2908_, lean_object* v_e_x27_2909_, lean_object* v_lbs_2910_, lean_object* v_ubs_2911_, lean_object* v_mustUseBounds_2912_, lean_object* v_a_2913_, lean_object* v_a_2914_, lean_object* v_a_2915_, lean_object* v_a_2916_, lean_object* v_a_2917_){
_start:
{
uint8_t v_mustUseBounds_boxed_2918_; lean_object* v_res_2919_; 
v_mustUseBounds_boxed_2918_ = lean_unbox(v_mustUseBounds_2912_);
v_res_2919_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases(v_g_2907_, v_e_2908_, v_e_x27_2909_, v_lbs_2910_, v_ubs_2911_, v_mustUseBounds_boxed_2918_, v_a_2913_, v_a_2914_, v_a_2915_, v_a_2916_);
lean_dec(v_a_2916_);
lean_dec_ref(v_a_2915_);
lean_dec(v_a_2914_);
lean_dec_ref(v_a_2913_);
return v_res_2919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3(lean_object* v___x_2920_, lean_object* v_m_2921_, lean_object* v_e_2922_, lean_object* v_a_2923_, lean_object* v_a_2924_, lean_object* v_range_2925_, lean_object* v_b_2926_, lean_object* v_i_2927_, lean_object* v_hs_2928_, lean_object* v_hl_2929_, lean_object* v___y_2930_, lean_object* v___y_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_){
_start:
{
lean_object* v___x_2935_; 
v___x_2935_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___redArg(v___x_2920_, v_m_2921_, v_e_2922_, v_a_2923_, v_a_2924_, v_range_2925_, v_b_2926_, v_i_2927_, v___y_2930_, v___y_2931_, v___y_2932_, v___y_2933_);
return v___x_2935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3___boxed(lean_object* v___x_2936_, lean_object* v_m_2937_, lean_object* v_e_2938_, lean_object* v_a_2939_, lean_object* v_a_2940_, lean_object* v_range_2941_, lean_object* v_b_2942_, lean_object* v_i_2943_, lean_object* v_hs_2944_, lean_object* v_hl_2945_, lean_object* v___y_2946_, lean_object* v___y_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_){
_start:
{
lean_object* v_res_2951_; 
v_res_2951_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3(v___x_2936_, v_m_2937_, v_e_2938_, v_a_2939_, v_a_2940_, v_range_2941_, v_b_2942_, v_i_2943_, v_hs_2944_, v_hl_2945_, v___y_2946_, v___y_2947_, v___y_2948_, v___y_2949_);
lean_dec(v___y_2949_);
lean_dec_ref(v___y_2948_);
lean_dec(v___y_2947_);
lean_dec_ref(v___y_2946_);
lean_dec_ref(v_range_2941_);
lean_dec(v___x_2936_);
return v_res_2951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3(lean_object* v___x_2952_, lean_object* v_m_2953_, lean_object* v_e_2954_, lean_object* v_a_2955_, lean_object* v_a_2956_, lean_object* v_range_2957_, lean_object* v_b_2958_, lean_object* v_i_2959_, lean_object* v_hs_2960_, lean_object* v_hl_2961_, lean_object* v___y_2962_, lean_object* v___y_2963_, lean_object* v___y_2964_, lean_object* v___y_2965_){
_start:
{
lean_object* v___x_2967_; 
v___x_2967_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___redArg(v___x_2952_, v_m_2953_, v_e_2954_, v_a_2955_, v_a_2956_, v_range_2957_, v_b_2958_, v_i_2959_, v___y_2962_, v___y_2963_, v___y_2964_, v___y_2965_);
return v___x_2967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3___boxed(lean_object* v___x_2968_, lean_object* v_m_2969_, lean_object* v_e_2970_, lean_object* v_a_2971_, lean_object* v_a_2972_, lean_object* v_range_2973_, lean_object* v_b_2974_, lean_object* v_i_2975_, lean_object* v_hs_2976_, lean_object* v_hl_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_){
_start:
{
lean_object* v_res_2983_; 
v_res_2983_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_IntervalCases_intervalCases_spec__3_spec__3(v___x_2968_, v_m_2969_, v_e_2970_, v_a_2971_, v_a_2972_, v_range_2973_, v_b_2974_, v_i_2975_, v_hs_2976_, v_hl_2977_, v___y_2978_, v___y_2979_, v___y_2980_, v___y_2981_);
lean_dec(v___y_2981_);
lean_dec_ref(v___y_2980_);
lean_dec(v___y_2979_);
lean_dec_ref(v___y_2978_);
lean_dec_ref(v_range_2973_);
lean_dec(v___x_2968_);
return v_res_2983_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__19(void){
_start:
{
lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; 
v___x_3019_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__18));
v___x_3020_ = l_Lean_binderIdent;
v___x_3021_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__3));
v___x_3022_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3022_, 0, v___x_3021_);
lean_ctor_set(v___x_3022_, 1, v___x_3020_);
lean_ctor_set(v___x_3022_, 2, v___x_3019_);
return v___x_3022_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__20(void){
_start:
{
lean_object* v___x_3023_; lean_object* v___x_3024_; lean_object* v___x_3025_; 
v___x_3023_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__19, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__19);
v___x_3024_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__16));
v___x_3025_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3025_, 0, v___x_3024_);
lean_ctor_set(v___x_3025_, 1, v___x_3023_);
return v___x_3025_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__21(void){
_start:
{
lean_object* v___x_3026_; lean_object* v___x_3027_; lean_object* v___x_3028_; 
v___x_3026_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__20, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__20);
v___x_3027_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__7));
v___x_3028_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3028_, 0, v___x_3027_);
lean_ctor_set(v___x_3028_, 1, v___x_3026_);
return v___x_3028_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__22(void){
_start:
{
lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; 
v___x_3029_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__21, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__21);
v___x_3030_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__14));
v___x_3031_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__3));
v___x_3032_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3032_, 0, v___x_3031_);
lean_ctor_set(v___x_3032_, 1, v___x_3030_);
lean_ctor_set(v___x_3032_, 2, v___x_3029_);
return v___x_3032_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__26(void){
_start:
{
lean_object* v___x_3039_; lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; 
v___x_3039_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__25));
v___x_3040_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__22, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__22);
v___x_3041_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__3));
v___x_3042_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3042_, 0, v___x_3041_);
lean_ctor_set(v___x_3042_, 1, v___x_3040_);
lean_ctor_set(v___x_3042_, 2, v___x_3039_);
return v___x_3042_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__27(void){
_start:
{
lean_object* v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; 
v___x_3043_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__26, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__26);
v___x_3044_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__7));
v___x_3045_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3045_, 0, v___x_3044_);
lean_ctor_set(v___x_3045_, 1, v___x_3043_);
return v___x_3045_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__28(void){
_start:
{
lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; lean_object* v___x_3049_; 
v___x_3046_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__27, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__27);
v___x_3047_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__5));
v___x_3048_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__3));
v___x_3049_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3049_, 0, v___x_3048_);
lean_ctor_set(v___x_3049_, 1, v___x_3047_);
lean_ctor_set(v___x_3049_, 2, v___x_3046_);
return v___x_3049_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__37(void){
_start:
{
lean_object* v___x_3071_; lean_object* v___x_3072_; lean_object* v___x_3073_; lean_object* v___x_3074_; 
v___x_3071_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__36));
v___x_3072_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__28, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__28);
v___x_3073_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__3));
v___x_3074_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3074_, 0, v___x_3073_);
lean_ctor_set(v___x_3074_, 1, v___x_3072_);
lean_ctor_set(v___x_3074_, 2, v___x_3071_);
return v___x_3074_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__38(void){
_start:
{
lean_object* v___x_3075_; lean_object* v___x_3076_; lean_object* v___x_3077_; lean_object* v___x_3078_; 
v___x_3075_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__37, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__37_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__37);
v___x_3076_ = lean_unsigned_to_nat(1022u);
v___x_3077_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__1));
v___x_3078_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3078_, 0, v___x_3077_);
lean_ctor_set(v___x_3078_, 1, v___x_3076_);
lean_ctor_set(v___x_3078_, 2, v___x_3075_);
return v___x_3078_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_intervalCases(void){
_start:
{
lean_object* v___x_3079_; 
v___x_3079_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_intervalCases___closed__38, &lp_mathlib_Mathlib_Tactic_intervalCases___closed__38_once, _init_lp_mathlib_Mathlib_Tactic_intervalCases___closed__38);
return v___x_3079_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3080_; lean_object* v___x_3081_; lean_object* v___x_3082_; 
v___x_3080_ = lean_box(0);
v___x_3081_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3082_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3082_, 0, v___x_3081_);
lean_ctor_set(v___x_3082_, 1, v___x_3080_);
return v___x_3082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3084_; lean_object* v___x_3085_; 
v___x_3084_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___closed__0);
v___x_3085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3085_, 0, v___x_3084_);
return v___x_3085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg___boxed(lean_object* v___y_3086_){
_start:
{
lean_object* v_res_3087_; 
v_res_3087_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v_res_3087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0(lean_object* v_00_u03b1_3088_, lean_object* v___y_3089_, lean_object* v___y_3090_, lean_object* v___y_3091_, lean_object* v___y_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_){
_start:
{
lean_object* v___x_3098_; 
v___x_3098_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_3098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___boxed(lean_object* v_00_u03b1_3099_, lean_object* v___y_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_, lean_object* v___y_3107_, lean_object* v___y_3108_){
_start:
{
lean_object* v_res_3109_; 
v_res_3109_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0(v_00_u03b1_3099_, v___y_3100_, v___y_3101_, v___y_3102_, v___y_3103_, v___y_3104_, v___y_3105_, v___y_3106_, v___y_3107_);
lean_dec(v___y_3107_);
lean_dec_ref(v___y_3106_);
lean_dec(v___y_3105_);
lean_dec_ref(v___y_3104_);
lean_dec(v___y_3103_);
lean_dec_ref(v___y_3102_);
lean_dec(v___y_3101_);
lean_dec_ref(v___y_3100_);
return v_res_3109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___lam__0(lean_object* v_x_3110_, lean_object* v___y_3111_, lean_object* v___y_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_){
_start:
{
lean_object* v___x_3120_; 
lean_inc(v___y_3114_);
lean_inc_ref(v___y_3113_);
lean_inc(v___y_3112_);
lean_inc_ref(v___y_3111_);
v___x_3120_ = lean_apply_9(v_x_3110_, v___y_3111_, v___y_3112_, v___y_3113_, v___y_3114_, v___y_3115_, v___y_3116_, v___y_3117_, v___y_3118_, lean_box(0));
return v___x_3120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___lam__0___boxed(lean_object* v_x_3121_, lean_object* v___y_3122_, lean_object* v___y_3123_, lean_object* v___y_3124_, lean_object* v___y_3125_, lean_object* v___y_3126_, lean_object* v___y_3127_, lean_object* v___y_3128_, lean_object* v___y_3129_, lean_object* v___y_3130_){
_start:
{
lean_object* v_res_3131_; 
v_res_3131_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___lam__0(v_x_3121_, v___y_3122_, v___y_3123_, v___y_3124_, v___y_3125_, v___y_3126_, v___y_3127_, v___y_3128_, v___y_3129_);
lean_dec(v___y_3125_);
lean_dec_ref(v___y_3124_);
lean_dec(v___y_3123_);
lean_dec_ref(v___y_3122_);
return v_res_3131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(lean_object* v_mvarId_3132_, lean_object* v_x_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_, lean_object* v___y_3138_, lean_object* v___y_3139_, lean_object* v___y_3140_, lean_object* v___y_3141_){
_start:
{
lean_object* v___f_3143_; lean_object* v___x_3144_; 
lean_inc(v___y_3137_);
lean_inc_ref(v___y_3136_);
lean_inc(v___y_3135_);
lean_inc_ref(v___y_3134_);
v___f_3143_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_3143_, 0, v_x_3133_);
lean_closure_set(v___f_3143_, 1, v___y_3134_);
lean_closure_set(v___f_3143_, 2, v___y_3135_);
lean_closure_set(v___f_3143_, 3, v___y_3136_);
lean_closure_set(v___f_3143_, 4, v___y_3137_);
v___x_3144_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_3132_, v___f_3143_, v___y_3138_, v___y_3139_, v___y_3140_, v___y_3141_);
if (lean_obj_tag(v___x_3144_) == 0)
{
return v___x_3144_;
}
else
{
lean_object* v_a_3145_; lean_object* v___x_3147_; uint8_t v_isShared_3148_; uint8_t v_isSharedCheck_3152_; 
v_a_3145_ = lean_ctor_get(v___x_3144_, 0);
v_isSharedCheck_3152_ = !lean_is_exclusive(v___x_3144_);
if (v_isSharedCheck_3152_ == 0)
{
v___x_3147_ = v___x_3144_;
v_isShared_3148_ = v_isSharedCheck_3152_;
goto v_resetjp_3146_;
}
else
{
lean_inc(v_a_3145_);
lean_dec(v___x_3144_);
v___x_3147_ = lean_box(0);
v_isShared_3148_ = v_isSharedCheck_3152_;
goto v_resetjp_3146_;
}
v_resetjp_3146_:
{
lean_object* v___x_3150_; 
if (v_isShared_3148_ == 0)
{
v___x_3150_ = v___x_3147_;
goto v_reusejp_3149_;
}
else
{
lean_object* v_reuseFailAlloc_3151_; 
v_reuseFailAlloc_3151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3151_, 0, v_a_3145_);
v___x_3150_ = v_reuseFailAlloc_3151_;
goto v_reusejp_3149_;
}
v_reusejp_3149_:
{
return v___x_3150_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg___boxed(lean_object* v_mvarId_3153_, lean_object* v_x_3154_, lean_object* v___y_3155_, lean_object* v___y_3156_, lean_object* v___y_3157_, lean_object* v___y_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_, lean_object* v___y_3163_){
_start:
{
lean_object* v_res_3164_; 
v_res_3164_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(v_mvarId_3153_, v_x_3154_, v___y_3155_, v___y_3156_, v___y_3157_, v___y_3158_, v___y_3159_, v___y_3160_, v___y_3161_, v___y_3162_);
lean_dec(v___y_3162_);
lean_dec_ref(v___y_3161_);
lean_dec(v___y_3160_);
lean_dec_ref(v___y_3159_);
lean_dec(v___y_3158_);
lean_dec_ref(v___y_3157_);
lean_dec(v___y_3156_);
lean_dec_ref(v___y_3155_);
return v_res_3164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1(lean_object* v_00_u03b1_3165_, lean_object* v_mvarId_3166_, lean_object* v_x_3167_, lean_object* v___y_3168_, lean_object* v___y_3169_, lean_object* v___y_3170_, lean_object* v___y_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_, lean_object* v___y_3175_){
_start:
{
lean_object* v___x_3177_; 
v___x_3177_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(v_mvarId_3166_, v_x_3167_, v___y_3168_, v___y_3169_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_, v___y_3174_, v___y_3175_);
return v___x_3177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___boxed(lean_object* v_00_u03b1_3178_, lean_object* v_mvarId_3179_, lean_object* v_x_3180_, lean_object* v___y_3181_, lean_object* v___y_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_){
_start:
{
lean_object* v_res_3190_; 
v_res_3190_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1(v_00_u03b1_3178_, v_mvarId_3179_, v_x_3180_, v___y_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_, v___y_3187_, v___y_3188_);
lean_dec(v___y_3188_);
lean_dec_ref(v___y_3187_);
lean_dec(v___y_3186_);
lean_dec_ref(v___y_3185_);
lean_dec(v___y_3184_);
lean_dec_ref(v___y_3183_);
lean_dec(v___y_3182_);
lean_dec_ref(v___y_3181_);
return v_res_3190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___lam__0(lean_object* v_k_3191_, lean_object* v___y_3192_, lean_object* v___y_3193_, lean_object* v___y_3194_, lean_object* v___y_3195_, lean_object* v___y_3196_, lean_object* v___y_3197_, lean_object* v___y_3198_, lean_object* v___y_3199_){
_start:
{
lean_object* v___x_3201_; 
lean_inc(v___y_3195_);
lean_inc_ref(v___y_3194_);
lean_inc(v___y_3193_);
lean_inc_ref(v___y_3192_);
v___x_3201_ = lean_apply_9(v_k_3191_, v___y_3192_, v___y_3193_, v___y_3194_, v___y_3195_, v___y_3196_, v___y_3197_, v___y_3198_, v___y_3199_, lean_box(0));
return v___x_3201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___lam__0___boxed(lean_object* v_k_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_, lean_object* v___y_3206_, lean_object* v___y_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_, lean_object* v___y_3210_, lean_object* v___y_3211_){
_start:
{
lean_object* v_res_3212_; 
v_res_3212_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___lam__0(v_k_3202_, v___y_3203_, v___y_3204_, v___y_3205_, v___y_3206_, v___y_3207_, v___y_3208_, v___y_3209_, v___y_3210_);
lean_dec(v___y_3206_);
lean_dec_ref(v___y_3205_);
lean_dec(v___y_3204_);
lean_dec_ref(v___y_3203_);
return v_res_3212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(lean_object* v_k_3213_, uint8_t v_allowLevelAssignments_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_, lean_object* v___y_3219_, lean_object* v___y_3220_, lean_object* v___y_3221_, lean_object* v___y_3222_){
_start:
{
lean_object* v___f_3224_; lean_object* v___x_3225_; 
lean_inc(v___y_3218_);
lean_inc_ref(v___y_3217_);
lean_inc(v___y_3216_);
lean_inc_ref(v___y_3215_);
v___f_3224_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_3224_, 0, v_k_3213_);
lean_closure_set(v___f_3224_, 1, v___y_3215_);
lean_closure_set(v___f_3224_, 2, v___y_3216_);
lean_closure_set(v___f_3224_, 3, v___y_3217_);
lean_closure_set(v___f_3224_, 4, v___y_3218_);
v___x_3225_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_3214_, v___f_3224_, v___y_3219_, v___y_3220_, v___y_3221_, v___y_3222_);
if (lean_obj_tag(v___x_3225_) == 0)
{
return v___x_3225_;
}
else
{
lean_object* v_a_3226_; lean_object* v___x_3228_; uint8_t v_isShared_3229_; uint8_t v_isSharedCheck_3233_; 
v_a_3226_ = lean_ctor_get(v___x_3225_, 0);
v_isSharedCheck_3233_ = !lean_is_exclusive(v___x_3225_);
if (v_isSharedCheck_3233_ == 0)
{
v___x_3228_ = v___x_3225_;
v_isShared_3229_ = v_isSharedCheck_3233_;
goto v_resetjp_3227_;
}
else
{
lean_inc(v_a_3226_);
lean_dec(v___x_3225_);
v___x_3228_ = lean_box(0);
v_isShared_3229_ = v_isSharedCheck_3233_;
goto v_resetjp_3227_;
}
v_resetjp_3227_:
{
lean_object* v___x_3231_; 
if (v_isShared_3229_ == 0)
{
v___x_3231_ = v___x_3228_;
goto v_reusejp_3230_;
}
else
{
lean_object* v_reuseFailAlloc_3232_; 
v_reuseFailAlloc_3232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3232_, 0, v_a_3226_);
v___x_3231_ = v_reuseFailAlloc_3232_;
goto v_reusejp_3230_;
}
v_reusejp_3230_:
{
return v___x_3231_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg___boxed(lean_object* v_k_3234_, lean_object* v_allowLevelAssignments_3235_, lean_object* v___y_3236_, lean_object* v___y_3237_, lean_object* v___y_3238_, lean_object* v___y_3239_, lean_object* v___y_3240_, lean_object* v___y_3241_, lean_object* v___y_3242_, lean_object* v___y_3243_, lean_object* v___y_3244_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_3245_; lean_object* v_res_3246_; 
v_allowLevelAssignments_boxed_3245_ = lean_unbox(v_allowLevelAssignments_3235_);
v_res_3246_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v_k_3234_, v_allowLevelAssignments_boxed_3245_, v___y_3236_, v___y_3237_, v___y_3238_, v___y_3239_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_);
lean_dec(v___y_3243_);
lean_dec_ref(v___y_3242_);
lean_dec(v___y_3241_);
lean_dec_ref(v___y_3240_);
lean_dec(v___y_3239_);
lean_dec_ref(v___y_3238_);
lean_dec(v___y_3237_);
lean_dec_ref(v___y_3236_);
return v_res_3246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3(lean_object* v_00_u03b1_3247_, lean_object* v_k_3248_, uint8_t v_allowLevelAssignments_3249_, lean_object* v___y_3250_, lean_object* v___y_3251_, lean_object* v___y_3252_, lean_object* v___y_3253_, lean_object* v___y_3254_, lean_object* v___y_3255_, lean_object* v___y_3256_, lean_object* v___y_3257_){
_start:
{
lean_object* v___x_3259_; 
v___x_3259_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v_k_3248_, v_allowLevelAssignments_3249_, v___y_3250_, v___y_3251_, v___y_3252_, v___y_3253_, v___y_3254_, v___y_3255_, v___y_3256_, v___y_3257_);
return v___x_3259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___boxed(lean_object* v_00_u03b1_3260_, lean_object* v_k_3261_, lean_object* v_allowLevelAssignments_3262_, lean_object* v___y_3263_, lean_object* v___y_3264_, lean_object* v___y_3265_, lean_object* v___y_3266_, lean_object* v___y_3267_, lean_object* v___y_3268_, lean_object* v___y_3269_, lean_object* v___y_3270_, lean_object* v___y_3271_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_3272_; lean_object* v_res_3273_; 
v_allowLevelAssignments_boxed_3272_ = lean_unbox(v_allowLevelAssignments_3262_);
v_res_3273_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3(v_00_u03b1_3260_, v_k_3261_, v_allowLevelAssignments_boxed_3272_, v___y_3263_, v___y_3264_, v___y_3265_, v___y_3266_, v___y_3267_, v___y_3268_, v___y_3269_, v___y_3270_);
lean_dec(v___y_3270_);
lean_dec_ref(v___y_3269_);
lean_dec(v___y_3268_);
lean_dec_ref(v___y_3267_);
lean_dec(v___y_3266_);
lean_dec_ref(v___y_3265_);
lean_dec(v___y_3264_);
lean_dec_ref(v___y_3263_);
return v_res_3273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___lam__0(lean_object* v___x_3274_, lean_object* v_val_3275_, lean_object* v___y_3276_, lean_object* v___y_3277_, lean_object* v___y_3278_, lean_object* v___y_3279_, lean_object* v___y_3280_, lean_object* v___y_3281_, lean_object* v___y_3282_, lean_object* v___y_3283_){
_start:
{
lean_object* v___x_3285_; 
v___x_3285_ = lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent(v___x_3274_, v_val_3275_, v___y_3280_, v___y_3281_, v___y_3282_, v___y_3283_);
return v___x_3285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___lam__0___boxed(lean_object* v___x_3286_, lean_object* v_val_3287_, lean_object* v___y_3288_, lean_object* v___y_3289_, lean_object* v___y_3290_, lean_object* v___y_3291_, lean_object* v___y_3292_, lean_object* v___y_3293_, lean_object* v___y_3294_, lean_object* v___y_3295_, lean_object* v___y_3296_){
_start:
{
lean_object* v_res_3297_; 
v_res_3297_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___lam__0(v___x_3286_, v_val_3287_, v___y_3288_, v___y_3289_, v___y_3290_, v___y_3291_, v___y_3292_, v___y_3293_, v___y_3294_, v___y_3295_);
lean_dec(v___y_3295_);
lean_dec_ref(v___y_3294_);
lean_dec(v___y_3293_);
lean_dec_ref(v___y_3292_);
lean_dec(v___y_3291_);
lean_dec_ref(v___y_3290_);
lean_dec(v___y_3289_);
lean_dec_ref(v___y_3288_);
return v_res_3297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2(lean_object* v_subst_3298_, uint8_t v___x_3299_, lean_object* v_h_3300_, lean_object* v_h_x3f_3301_, size_t v_sz_3302_, size_t v_i_3303_, lean_object* v_bs_3304_, lean_object* v___y_3305_, lean_object* v___y_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_, lean_object* v___y_3310_, lean_object* v___y_3311_, lean_object* v___y_3312_){
_start:
{
uint8_t v___x_3314_; 
v___x_3314_ = lean_usize_dec_lt(v_i_3303_, v_sz_3302_);
if (v___x_3314_ == 0)
{
lean_object* v___x_3315_; 
lean_dec(v_h_x3f_3301_);
lean_dec(v_h_3300_);
lean_dec(v_subst_3298_);
v___x_3315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3315_, 0, v_bs_3304_);
return v___x_3315_;
}
else
{
lean_object* v_v_3316_; lean_object* v_goal_3317_; uint8_t v___x_3318_; lean_object* v___x_3319_; 
v_v_3316_ = lean_array_uget_borrowed(v_bs_3304_, v_i_3303_);
v_goal_3317_ = lean_ctor_get(v_v_3316_, 2);
v___x_3318_ = 0;
lean_inc(v_goal_3317_);
v___x_3319_ = l_Lean_Meta_intro1Core(v_goal_3317_, v___x_3318_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_);
if (lean_obj_tag(v___x_3319_) == 0)
{
lean_object* v_a_3320_; lean_object* v_fst_3321_; lean_object* v_snd_3322_; lean_object* v___x_3323_; 
v_a_3320_ = lean_ctor_get(v___x_3319_, 0);
lean_inc(v_a_3320_);
lean_dec_ref_known(v___x_3319_, 1);
v_fst_3321_ = lean_ctor_get(v_a_3320_, 0);
lean_inc(v_fst_3321_);
v_snd_3322_ = lean_ctor_get(v_a_3320_, 1);
lean_inc(v_snd_3322_);
lean_dec(v_a_3320_);
lean_inc(v_subst_3298_);
v___x_3323_ = l_Lean_Meta_substCore(v_snd_3322_, v_fst_3321_, v___x_3318_, v_subst_3298_, v___x_3299_, v___x_3318_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_);
if (lean_obj_tag(v___x_3323_) == 0)
{
lean_object* v_a_3324_; lean_object* v_fst_3325_; lean_object* v_snd_3326_; lean_object* v___x_3327_; lean_object* v_bs_x27_3328_; 
v_a_3324_ = lean_ctor_get(v___x_3323_, 0);
lean_inc(v_a_3324_);
lean_dec_ref_known(v___x_3323_, 1);
v_fst_3325_ = lean_ctor_get(v_a_3324_, 0);
lean_inc(v_fst_3325_);
v_snd_3326_ = lean_ctor_get(v_a_3324_, 1);
lean_inc(v_snd_3326_);
lean_dec(v_a_3324_);
v___x_3327_ = lean_unsigned_to_nat(0u);
v_bs_x27_3328_ = lean_array_uset(v_bs_3304_, v_i_3303_, v___x_3327_);
if (lean_obj_tag(v_h_3300_) == 0)
{
lean_dec(v_fst_3325_);
goto v___jp_3329_;
}
else
{
lean_object* v_val_3334_; 
v_val_3334_ = lean_ctor_get(v_h_3300_, 0);
if (lean_obj_tag(v_val_3334_) == 1)
{
if (lean_obj_tag(v_h_x3f_3301_) == 1)
{
lean_object* v_val_3335_; lean_object* v_val_3336_; lean_object* v___x_3337_; lean_object* v___f_3338_; lean_object* v___x_3339_; 
v_val_3335_ = lean_ctor_get(v_val_3334_, 0);
v_val_3336_ = lean_ctor_get(v_h_x3f_3301_, 0);
lean_inc(v_val_3336_);
v___x_3337_ = l_Lean_Meta_FVarSubst_get(v_fst_3325_, v_val_3336_);
lean_dec(v_fst_3325_);
lean_inc(v_val_3335_);
v___f_3338_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___lam__0___boxed), 11, 2);
lean_closure_set(v___f_3338_, 0, v___x_3337_);
lean_closure_set(v___f_3338_, 1, v_val_3335_);
lean_inc(v_snd_3326_);
v___x_3339_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(v_snd_3326_, v___f_3338_, v___y_3305_, v___y_3306_, v___y_3307_, v___y_3308_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_);
if (lean_obj_tag(v___x_3339_) == 0)
{
lean_dec_ref_known(v___x_3339_, 1);
goto v___jp_3329_;
}
else
{
lean_object* v_a_3340_; lean_object* v___x_3342_; uint8_t v_isShared_3343_; uint8_t v_isSharedCheck_3347_; 
lean_dec_ref_known(v_h_x3f_3301_, 1);
lean_dec_ref_known(v_h_3300_, 1);
lean_dec_ref(v_bs_x27_3328_);
lean_dec(v_snd_3326_);
lean_dec(v_subst_3298_);
v_a_3340_ = lean_ctor_get(v___x_3339_, 0);
v_isSharedCheck_3347_ = !lean_is_exclusive(v___x_3339_);
if (v_isSharedCheck_3347_ == 0)
{
v___x_3342_ = v___x_3339_;
v_isShared_3343_ = v_isSharedCheck_3347_;
goto v_resetjp_3341_;
}
else
{
lean_inc(v_a_3340_);
lean_dec(v___x_3339_);
v___x_3342_ = lean_box(0);
v_isShared_3343_ = v_isSharedCheck_3347_;
goto v_resetjp_3341_;
}
v_resetjp_3341_:
{
lean_object* v___x_3345_; 
if (v_isShared_3343_ == 0)
{
v___x_3345_ = v___x_3342_;
goto v_reusejp_3344_;
}
else
{
lean_object* v_reuseFailAlloc_3346_; 
v_reuseFailAlloc_3346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3346_, 0, v_a_3340_);
v___x_3345_ = v_reuseFailAlloc_3346_;
goto v_reusejp_3344_;
}
v_reusejp_3344_:
{
return v___x_3345_;
}
}
}
}
else
{
lean_dec(v_fst_3325_);
goto v___jp_3329_;
}
}
else
{
lean_dec(v_fst_3325_);
goto v___jp_3329_;
}
}
v___jp_3329_:
{
size_t v___x_3330_; size_t v___x_3331_; lean_object* v___x_3332_; 
v___x_3330_ = ((size_t)1ULL);
v___x_3331_ = lean_usize_add(v_i_3303_, v___x_3330_);
v___x_3332_ = lean_array_uset(v_bs_x27_3328_, v_i_3303_, v_snd_3326_);
v_i_3303_ = v___x_3331_;
v_bs_3304_ = v___x_3332_;
goto _start;
}
}
else
{
lean_object* v_a_3348_; lean_object* v___x_3350_; uint8_t v_isShared_3351_; uint8_t v_isSharedCheck_3355_; 
lean_dec_ref(v_bs_3304_);
lean_dec(v_h_x3f_3301_);
lean_dec(v_h_3300_);
lean_dec(v_subst_3298_);
v_a_3348_ = lean_ctor_get(v___x_3323_, 0);
v_isSharedCheck_3355_ = !lean_is_exclusive(v___x_3323_);
if (v_isSharedCheck_3355_ == 0)
{
v___x_3350_ = v___x_3323_;
v_isShared_3351_ = v_isSharedCheck_3355_;
goto v_resetjp_3349_;
}
else
{
lean_inc(v_a_3348_);
lean_dec(v___x_3323_);
v___x_3350_ = lean_box(0);
v_isShared_3351_ = v_isSharedCheck_3355_;
goto v_resetjp_3349_;
}
v_resetjp_3349_:
{
lean_object* v___x_3353_; 
if (v_isShared_3351_ == 0)
{
v___x_3353_ = v___x_3350_;
goto v_reusejp_3352_;
}
else
{
lean_object* v_reuseFailAlloc_3354_; 
v_reuseFailAlloc_3354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3354_, 0, v_a_3348_);
v___x_3353_ = v_reuseFailAlloc_3354_;
goto v_reusejp_3352_;
}
v_reusejp_3352_:
{
return v___x_3353_;
}
}
}
}
else
{
lean_object* v_a_3356_; lean_object* v___x_3358_; uint8_t v_isShared_3359_; uint8_t v_isSharedCheck_3363_; 
lean_dec_ref(v_bs_3304_);
lean_dec(v_h_x3f_3301_);
lean_dec(v_h_3300_);
lean_dec(v_subst_3298_);
v_a_3356_ = lean_ctor_get(v___x_3319_, 0);
v_isSharedCheck_3363_ = !lean_is_exclusive(v___x_3319_);
if (v_isSharedCheck_3363_ == 0)
{
v___x_3358_ = v___x_3319_;
v_isShared_3359_ = v_isSharedCheck_3363_;
goto v_resetjp_3357_;
}
else
{
lean_inc(v_a_3356_);
lean_dec(v___x_3319_);
v___x_3358_ = lean_box(0);
v_isShared_3359_ = v_isSharedCheck_3363_;
goto v_resetjp_3357_;
}
v_resetjp_3357_:
{
lean_object* v___x_3361_; 
if (v_isShared_3359_ == 0)
{
v___x_3361_ = v___x_3358_;
goto v_reusejp_3360_;
}
else
{
lean_object* v_reuseFailAlloc_3362_; 
v_reuseFailAlloc_3362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3362_, 0, v_a_3356_);
v___x_3361_ = v_reuseFailAlloc_3362_;
goto v_reusejp_3360_;
}
v_reusejp_3360_:
{
return v___x_3361_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2___boxed(lean_object* v_subst_3364_, lean_object* v___x_3365_, lean_object* v_h_3366_, lean_object* v_h_x3f_3367_, lean_object* v_sz_3368_, lean_object* v_i_3369_, lean_object* v_bs_3370_, lean_object* v___y_3371_, lean_object* v___y_3372_, lean_object* v___y_3373_, lean_object* v___y_3374_, lean_object* v___y_3375_, lean_object* v___y_3376_, lean_object* v___y_3377_, lean_object* v___y_3378_, lean_object* v___y_3379_){
_start:
{
uint8_t v___x_36398__boxed_3380_; size_t v_sz_boxed_3381_; size_t v_i_boxed_3382_; lean_object* v_res_3383_; 
v___x_36398__boxed_3380_ = lean_unbox(v___x_3365_);
v_sz_boxed_3381_ = lean_unbox_usize(v_sz_3368_);
lean_dec(v_sz_3368_);
v_i_boxed_3382_ = lean_unbox_usize(v_i_3369_);
lean_dec(v_i_3369_);
v_res_3383_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2(v_subst_3364_, v___x_36398__boxed_3380_, v_h_3366_, v_h_x3f_3367_, v_sz_boxed_3381_, v_i_boxed_3382_, v_bs_3370_, v___y_3371_, v___y_3372_, v___y_3373_, v___y_3374_, v___y_3375_, v___y_3376_, v___y_3377_, v___y_3378_);
lean_dec(v___y_3378_);
lean_dec_ref(v___y_3377_);
lean_dec(v___y_3376_);
lean_dec_ref(v___y_3375_);
lean_dec(v___y_3374_);
lean_dec_ref(v___y_3373_);
lean_dec(v___y_3372_);
lean_dec_ref(v___y_3371_);
return v_res_3383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__0(uint8_t v___x_3384_, lean_object* v_h_3385_, lean_object* v_x_3386_, lean_object* v_h_x3f_3387_, lean_object* v_subst_3388_, lean_object* v_g_3389_, lean_object* v_e_3390_, lean_object* v_lbs_3391_, lean_object* v_ubs_3392_, uint8_t v_mustUseBounds_3393_, lean_object* v___y_3394_, lean_object* v___y_3395_, lean_object* v___y_3396_, lean_object* v___y_3397_, lean_object* v___y_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_, lean_object* v___y_3401_){
_start:
{
lean_object* v___x_3403_; lean_object* v___x_3404_; 
v___x_3403_ = l_Lean_Expr_fvar___override(v_x_3386_);
v___x_3404_ = lp_mathlib_Mathlib_Tactic_IntervalCases_intervalCases(v_g_3389_, v___x_3403_, v_e_3390_, v_lbs_3391_, v_ubs_3392_, v_mustUseBounds_3393_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_);
if (lean_obj_tag(v___x_3404_) == 0)
{
lean_object* v_a_3405_; size_t v_sz_3406_; size_t v___x_3407_; lean_object* v___x_3408_; 
v_a_3405_ = lean_ctor_get(v___x_3404_, 0);
lean_inc(v_a_3405_);
lean_dec_ref_known(v___x_3404_, 1);
v_sz_3406_ = lean_array_size(v_a_3405_);
v___x_3407_ = ((size_t)0ULL);
v___x_3408_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__2(v_subst_3388_, v___x_3384_, v_h_3385_, v_h_x3f_3387_, v_sz_3406_, v___x_3407_, v_a_3405_, v___y_3394_, v___y_3395_, v___y_3396_, v___y_3397_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_);
if (lean_obj_tag(v___x_3408_) == 0)
{
lean_object* v_a_3409_; lean_object* v___x_3410_; lean_object* v___x_3411_; 
v_a_3409_ = lean_ctor_get(v___x_3408_, 0);
lean_inc(v_a_3409_);
lean_dec_ref_known(v___x_3408_, 1);
v___x_3410_ = lean_array_to_list(v_a_3409_);
v___x_3411_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_3410_, v___y_3395_, v___y_3398_, v___y_3399_, v___y_3400_, v___y_3401_);
return v___x_3411_;
}
else
{
lean_object* v_a_3412_; lean_object* v___x_3414_; uint8_t v_isShared_3415_; uint8_t v_isSharedCheck_3419_; 
v_a_3412_ = lean_ctor_get(v___x_3408_, 0);
v_isSharedCheck_3419_ = !lean_is_exclusive(v___x_3408_);
if (v_isSharedCheck_3419_ == 0)
{
v___x_3414_ = v___x_3408_;
v_isShared_3415_ = v_isSharedCheck_3419_;
goto v_resetjp_3413_;
}
else
{
lean_inc(v_a_3412_);
lean_dec(v___x_3408_);
v___x_3414_ = lean_box(0);
v_isShared_3415_ = v_isSharedCheck_3419_;
goto v_resetjp_3413_;
}
v_resetjp_3413_:
{
lean_object* v___x_3417_; 
if (v_isShared_3415_ == 0)
{
v___x_3417_ = v___x_3414_;
goto v_reusejp_3416_;
}
else
{
lean_object* v_reuseFailAlloc_3418_; 
v_reuseFailAlloc_3418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3418_, 0, v_a_3412_);
v___x_3417_ = v_reuseFailAlloc_3418_;
goto v_reusejp_3416_;
}
v_reusejp_3416_:
{
return v___x_3417_;
}
}
}
}
else
{
lean_object* v_a_3420_; lean_object* v___x_3422_; uint8_t v_isShared_3423_; uint8_t v_isSharedCheck_3427_; 
lean_dec(v_subst_3388_);
lean_dec(v_h_x3f_3387_);
lean_dec(v_h_3385_);
v_a_3420_ = lean_ctor_get(v___x_3404_, 0);
v_isSharedCheck_3427_ = !lean_is_exclusive(v___x_3404_);
if (v_isSharedCheck_3427_ == 0)
{
v___x_3422_ = v___x_3404_;
v_isShared_3423_ = v_isSharedCheck_3427_;
goto v_resetjp_3421_;
}
else
{
lean_inc(v_a_3420_);
lean_dec(v___x_3404_);
v___x_3422_ = lean_box(0);
v_isShared_3423_ = v_isSharedCheck_3427_;
goto v_resetjp_3421_;
}
v_resetjp_3421_:
{
lean_object* v___x_3425_; 
if (v_isShared_3423_ == 0)
{
v___x_3425_ = v___x_3422_;
goto v_reusejp_3424_;
}
else
{
lean_object* v_reuseFailAlloc_3426_; 
v_reuseFailAlloc_3426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3426_, 0, v_a_3420_);
v___x_3425_ = v_reuseFailAlloc_3426_;
goto v_reusejp_3424_;
}
v_reusejp_3424_:
{
return v___x_3425_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__0___boxed(lean_object** _args){
lean_object* v___x_3428_ = _args[0];
lean_object* v_h_3429_ = _args[1];
lean_object* v_x_3430_ = _args[2];
lean_object* v_h_x3f_3431_ = _args[3];
lean_object* v_subst_3432_ = _args[4];
lean_object* v_g_3433_ = _args[5];
lean_object* v_e_3434_ = _args[6];
lean_object* v_lbs_3435_ = _args[7];
lean_object* v_ubs_3436_ = _args[8];
lean_object* v_mustUseBounds_3437_ = _args[9];
lean_object* v___y_3438_ = _args[10];
lean_object* v___y_3439_ = _args[11];
lean_object* v___y_3440_ = _args[12];
lean_object* v___y_3441_ = _args[13];
lean_object* v___y_3442_ = _args[14];
lean_object* v___y_3443_ = _args[15];
lean_object* v___y_3444_ = _args[16];
lean_object* v___y_3445_ = _args[17];
lean_object* v___y_3446_ = _args[18];
_start:
{
uint8_t v___x_36527__boxed_3447_; uint8_t v_mustUseBounds_boxed_3448_; lean_object* v_res_3449_; 
v___x_36527__boxed_3447_ = lean_unbox(v___x_3428_);
v_mustUseBounds_boxed_3448_ = lean_unbox(v_mustUseBounds_3437_);
v_res_3449_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__0(v___x_36527__boxed_3447_, v_h_3429_, v_x_3430_, v_h_x3f_3431_, v_subst_3432_, v_g_3433_, v_e_3434_, v_lbs_3435_, v_ubs_3436_, v_mustUseBounds_boxed_3448_, v___y_3438_, v___y_3439_, v___y_3440_, v___y_3441_, v___y_3442_, v___y_3443_, v___y_3444_, v___y_3445_);
lean_dec(v___y_3445_);
lean_dec_ref(v___y_3444_);
lean_dec(v___y_3443_);
lean_dec_ref(v___y_3442_);
lean_dec(v___y_3441_);
lean_dec_ref(v___y_3440_);
lean_dec(v___y_3439_);
lean_dec_ref(v___y_3438_);
return v_res_3449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0(uint8_t v___x_3450_, lean_object* v___x_3451_, lean_object* v_fst_3452_, lean_object* v___y_3453_, lean_object* v___y_3454_, lean_object* v___y_3455_, lean_object* v___y_3456_, lean_object* v___y_3457_, lean_object* v___y_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_){
_start:
{
lean_object* v_keyedConfig_3462_; uint8_t v_trackZetaDelta_3463_; lean_object* v_zetaDeltaSet_3464_; lean_object* v_lctx_3465_; lean_object* v_localInstances_3466_; lean_object* v_defEqCtx_x3f_3467_; lean_object* v_synthPendingDepth_3468_; lean_object* v_customCanUnfoldPredicate_x3f_3469_; uint8_t v_univApprox_3470_; uint8_t v_inTypeClassResolution_3471_; uint8_t v_cacheInferType_3472_; lean_object* v___x_3474_; uint8_t v_isShared_3475_; uint8_t v_isSharedCheck_3489_; 
v_keyedConfig_3462_ = lean_ctor_get(v___y_3457_, 0);
v_trackZetaDelta_3463_ = lean_ctor_get_uint8(v___y_3457_, sizeof(void*)*7);
v_zetaDeltaSet_3464_ = lean_ctor_get(v___y_3457_, 1);
v_lctx_3465_ = lean_ctor_get(v___y_3457_, 2);
v_localInstances_3466_ = lean_ctor_get(v___y_3457_, 3);
v_defEqCtx_x3f_3467_ = lean_ctor_get(v___y_3457_, 4);
v_synthPendingDepth_3468_ = lean_ctor_get(v___y_3457_, 5);
v_customCanUnfoldPredicate_x3f_3469_ = lean_ctor_get(v___y_3457_, 6);
v_univApprox_3470_ = lean_ctor_get_uint8(v___y_3457_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3471_ = lean_ctor_get_uint8(v___y_3457_, sizeof(void*)*7 + 2);
v_cacheInferType_3472_ = lean_ctor_get_uint8(v___y_3457_, sizeof(void*)*7 + 3);
v_isSharedCheck_3489_ = !lean_is_exclusive(v___y_3457_);
if (v_isSharedCheck_3489_ == 0)
{
v___x_3474_ = v___y_3457_;
v_isShared_3475_ = v_isSharedCheck_3489_;
goto v_resetjp_3473_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3469_);
lean_inc(v_synthPendingDepth_3468_);
lean_inc(v_defEqCtx_x3f_3467_);
lean_inc(v_localInstances_3466_);
lean_inc(v_lctx_3465_);
lean_inc(v_zetaDeltaSet_3464_);
lean_inc(v_keyedConfig_3462_);
lean_dec(v___y_3457_);
v___x_3474_ = lean_box(0);
v_isShared_3475_ = v_isSharedCheck_3489_;
goto v_resetjp_3473_;
}
v_resetjp_3473_:
{
lean_object* v___x_3476_; lean_object* v___x_3478_; 
v___x_3476_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3450_, v_keyedConfig_3462_);
if (v_isShared_3475_ == 0)
{
lean_ctor_set(v___x_3474_, 0, v___x_3476_);
v___x_3478_ = v___x_3474_;
goto v_reusejp_3477_;
}
else
{
lean_object* v_reuseFailAlloc_3488_; 
v_reuseFailAlloc_3488_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3488_, 0, v___x_3476_);
lean_ctor_set(v_reuseFailAlloc_3488_, 1, v_zetaDeltaSet_3464_);
lean_ctor_set(v_reuseFailAlloc_3488_, 2, v_lctx_3465_);
lean_ctor_set(v_reuseFailAlloc_3488_, 3, v_localInstances_3466_);
lean_ctor_set(v_reuseFailAlloc_3488_, 4, v_defEqCtx_x3f_3467_);
lean_ctor_set(v_reuseFailAlloc_3488_, 5, v_synthPendingDepth_3468_);
lean_ctor_set(v_reuseFailAlloc_3488_, 6, v_customCanUnfoldPredicate_x3f_3469_);
lean_ctor_set_uint8(v_reuseFailAlloc_3488_, sizeof(void*)*7, v_trackZetaDelta_3463_);
lean_ctor_set_uint8(v_reuseFailAlloc_3488_, sizeof(void*)*7 + 1, v_univApprox_3470_);
lean_ctor_set_uint8(v_reuseFailAlloc_3488_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3471_);
lean_ctor_set_uint8(v_reuseFailAlloc_3488_, sizeof(void*)*7 + 3, v_cacheInferType_3472_);
v___x_3478_ = v_reuseFailAlloc_3488_;
goto v_reusejp_3477_;
}
v_reusejp_3477_:
{
lean_object* v___x_3479_; 
v___x_3479_ = l_Lean_Meta_isExprDefEq(v___x_3451_, v_fst_3452_, v___x_3478_, v___y_3458_, v___y_3459_, v___y_3460_);
lean_dec_ref(v___x_3478_);
if (lean_obj_tag(v___x_3479_) == 0)
{
lean_object* v_a_3480_; lean_object* v___x_3482_; uint8_t v_isShared_3483_; uint8_t v_isSharedCheck_3487_; 
v_a_3480_ = lean_ctor_get(v___x_3479_, 0);
v_isSharedCheck_3487_ = !lean_is_exclusive(v___x_3479_);
if (v_isSharedCheck_3487_ == 0)
{
v___x_3482_ = v___x_3479_;
v_isShared_3483_ = v_isSharedCheck_3487_;
goto v_resetjp_3481_;
}
else
{
lean_inc(v_a_3480_);
lean_dec(v___x_3479_);
v___x_3482_ = lean_box(0);
v_isShared_3483_ = v_isSharedCheck_3487_;
goto v_resetjp_3481_;
}
v_resetjp_3481_:
{
lean_object* v___x_3485_; 
if (v_isShared_3483_ == 0)
{
v___x_3485_ = v___x_3482_;
goto v_reusejp_3484_;
}
else
{
lean_object* v_reuseFailAlloc_3486_; 
v_reuseFailAlloc_3486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3486_, 0, v_a_3480_);
v___x_3485_ = v_reuseFailAlloc_3486_;
goto v_reusejp_3484_;
}
v_reusejp_3484_:
{
return v___x_3485_;
}
}
}
else
{
return v___x_3479_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed(lean_object* v___x_3490_, lean_object* v___x_3491_, lean_object* v_fst_3492_, lean_object* v___y_3493_, lean_object* v___y_3494_, lean_object* v___y_3495_, lean_object* v___y_3496_, lean_object* v___y_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_){
_start:
{
uint8_t v___x_36605__boxed_3502_; lean_object* v_res_3503_; 
v___x_36605__boxed_3502_ = lean_unbox(v___x_3490_);
v_res_3503_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0(v___x_36605__boxed_3502_, v___x_3491_, v_fst_3492_, v___y_3493_, v___y_3494_, v___y_3495_, v___y_3496_, v___y_3497_, v___y_3498_, v___y_3499_, v___y_3500_);
lean_dec(v___y_3500_);
lean_dec_ref(v___y_3499_);
lean_dec(v___y_3498_);
lean_dec(v___y_3496_);
lean_dec_ref(v___y_3495_);
lean_dec(v___y_3494_);
lean_dec_ref(v___y_3493_);
return v_res_3503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(lean_object* v_msg_3504_, lean_object* v___y_3505_, lean_object* v___y_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_){
_start:
{
lean_object* v_ref_3510_; lean_object* v___x_3511_; lean_object* v_a_3512_; lean_object* v___x_3514_; uint8_t v_isShared_3515_; uint8_t v_isSharedCheck_3520_; 
v_ref_3510_ = lean_ctor_get(v___y_3507_, 5);
v___x_3511_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_IntervalCases_parseBound_spec__0_spec__0(v_msg_3504_, v___y_3505_, v___y_3506_, v___y_3507_, v___y_3508_);
v_a_3512_ = lean_ctor_get(v___x_3511_, 0);
v_isSharedCheck_3520_ = !lean_is_exclusive(v___x_3511_);
if (v_isSharedCheck_3520_ == 0)
{
v___x_3514_ = v___x_3511_;
v_isShared_3515_ = v_isSharedCheck_3520_;
goto v_resetjp_3513_;
}
else
{
lean_inc(v_a_3512_);
lean_dec(v___x_3511_);
v___x_3514_ = lean_box(0);
v_isShared_3515_ = v_isSharedCheck_3520_;
goto v_resetjp_3513_;
}
v_resetjp_3513_:
{
lean_object* v___x_3516_; lean_object* v___x_3518_; 
lean_inc(v_ref_3510_);
v___x_3516_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3516_, 0, v_ref_3510_);
lean_ctor_set(v___x_3516_, 1, v_a_3512_);
if (v_isShared_3515_ == 0)
{
lean_ctor_set_tag(v___x_3514_, 1);
lean_ctor_set(v___x_3514_, 0, v___x_3516_);
v___x_3518_ = v___x_3514_;
goto v_reusejp_3517_;
}
else
{
lean_object* v_reuseFailAlloc_3519_; 
v_reuseFailAlloc_3519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3519_, 0, v___x_3516_);
v___x_3518_ = v_reuseFailAlloc_3519_;
goto v_reusejp_3517_;
}
v_reusejp_3517_:
{
return v___x_3518_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg___boxed(lean_object* v_msg_3521_, lean_object* v___y_3522_, lean_object* v___y_3523_, lean_object* v___y_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_){
_start:
{
lean_object* v_res_3527_; 
v_res_3527_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v_msg_3521_, v___y_3522_, v___y_3523_, v___y_3524_, v___y_3525_);
lean_dec(v___y_3525_);
lean_dec_ref(v___y_3524_);
lean_dec(v___y_3523_);
lean_dec_ref(v___y_3522_);
return v_res_3527_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1(void){
_start:
{
lean_object* v___x_3529_; lean_object* v___x_3530_; 
v___x_3529_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__0));
v___x_3530_ = l_Lean_stringToMessageData(v___x_3529_);
return v___x_3530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9(lean_object* v___x_3531_, lean_object* v_as_3532_, size_t v_sz_3533_, size_t v_i_3534_, lean_object* v_b_3535_, lean_object* v___y_3536_, lean_object* v___y_3537_, lean_object* v___y_3538_, lean_object* v___y_3539_, lean_object* v___y_3540_, lean_object* v___y_3541_, lean_object* v___y_3542_, lean_object* v___y_3543_){
_start:
{
uint8_t v___x_3545_; 
v___x_3545_ = lean_usize_dec_lt(v_i_3534_, v_sz_3533_);
if (v___x_3545_ == 0)
{
lean_object* v___x_3546_; 
lean_dec(v___x_3531_);
v___x_3546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3546_, 0, v_b_3535_);
return v___x_3546_;
}
else
{
lean_object* v_snd_3547_; lean_object* v___x_3549_; uint8_t v_isShared_3550_; uint8_t v_isSharedCheck_3632_; 
v_snd_3547_ = lean_ctor_get(v_b_3535_, 1);
v_isSharedCheck_3632_ = !lean_is_exclusive(v_b_3535_);
if (v_isSharedCheck_3632_ == 0)
{
lean_object* v_unused_3633_; 
v_unused_3633_ = lean_ctor_get(v_b_3535_, 0);
lean_dec(v_unused_3633_);
v___x_3549_ = v_b_3535_;
v_isShared_3550_ = v_isSharedCheck_3632_;
goto v_resetjp_3548_;
}
else
{
lean_inc(v_snd_3547_);
lean_dec(v_b_3535_);
v___x_3549_ = lean_box(0);
v_isShared_3550_ = v_isSharedCheck_3632_;
goto v_resetjp_3548_;
}
v_resetjp_3548_:
{
lean_object* v___x_3551_; lean_object* v_a_3553_; lean_object* v_fst_3561_; lean_object* v_snd_3562_; lean_object* v_a_3564_; 
v___x_3551_ = lean_box(0);
v_a_3564_ = lean_array_uget_borrowed(v_as_3532_, v_i_3534_);
if (lean_obj_tag(v_a_3564_) == 0)
{
v_a_3553_ = v_snd_3547_;
goto v___jp_3552_;
}
else
{
lean_object* v_val_3565_; lean_object* v___x_3566_; 
v_val_3565_ = lean_ctor_get(v_a_3564_, 0);
v___x_3566_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3537_, v___y_3539_, v___y_3541_, v___y_3543_);
if (lean_obj_tag(v___x_3566_) == 0)
{
lean_object* v_a_3567_; lean_object* v___x_3569_; uint8_t v_isShared_3570_; uint8_t v_isSharedCheck_3623_; 
v_a_3567_ = lean_ctor_get(v___x_3566_, 0);
v_isSharedCheck_3623_ = !lean_is_exclusive(v___x_3566_);
if (v_isSharedCheck_3623_ == 0)
{
v___x_3569_ = v___x_3566_;
v_isShared_3570_ = v_isSharedCheck_3623_;
goto v_resetjp_3568_;
}
else
{
lean_inc(v_a_3567_);
lean_dec(v___x_3566_);
v___x_3569_ = lean_box(0);
v_isShared_3570_ = v_isSharedCheck_3623_;
goto v_resetjp_3568_;
}
v_resetjp_3568_:
{
lean_object* v_fst_3571_; lean_object* v_snd_3572_; lean_object* v___y_3574_; uint8_t v___y_3575_; lean_object* v_a_3589_; lean_object* v___x_3592_; lean_object* v___x_3593_; 
v_fst_3571_ = lean_ctor_get(v_snd_3547_, 0);
lean_inc(v_fst_3571_);
v_snd_3572_ = lean_ctor_get(v_snd_3547_, 1);
lean_inc(v_snd_3572_);
lean_dec(v_snd_3547_);
v___x_3592_ = l_Lean_LocalDecl_type(v_val_3565_);
v___x_3593_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v___x_3592_, v___y_3540_, v___y_3541_, v___y_3542_, v___y_3543_);
if (lean_obj_tag(v___x_3593_) == 0)
{
lean_object* v_a_3594_; lean_object* v_snd_3595_; lean_object* v_fst_3596_; lean_object* v_fst_3597_; uint8_t v___x_3598_; lean_object* v___x_3599_; uint8_t v___x_3600_; lean_object* v___x_3601_; lean_object* v___f_3602_; lean_object* v___x_3603_; 
v_a_3594_ = lean_ctor_get(v___x_3593_, 0);
lean_inc(v_a_3594_);
lean_dec_ref_known(v___x_3593_, 1);
v_snd_3595_ = lean_ctor_get(v_a_3594_, 1);
lean_inc(v_snd_3595_);
v_fst_3596_ = lean_ctor_get(v_a_3594_, 0);
lean_inc(v_fst_3596_);
lean_dec(v_a_3594_);
v_fst_3597_ = lean_ctor_get(v_snd_3595_, 0);
lean_inc(v_fst_3597_);
lean_dec(v_snd_3595_);
v___x_3598_ = 0;
lean_inc(v___x_3531_);
v___x_3599_ = l_Lean_Expr_fvar___override(v___x_3531_);
v___x_3600_ = 2;
v___x_3601_ = lean_box(v___x_3600_);
lean_inc_ref(v___x_3599_);
v___f_3602_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3602_, 0, v___x_3601_);
lean_closure_set(v___f_3602_, 1, v___x_3599_);
lean_closure_set(v___f_3602_, 2, v_fst_3596_);
v___x_3603_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3602_, v___x_3598_, v___y_3536_, v___y_3537_, v___y_3538_, v___y_3539_, v___y_3540_, v___y_3541_, v___y_3542_, v___y_3543_);
if (lean_obj_tag(v___x_3603_) == 0)
{
lean_object* v_a_3604_; uint8_t v___x_3605_; 
v_a_3604_ = lean_ctor_get(v___x_3603_, 0);
lean_inc(v_a_3604_);
lean_dec_ref_known(v___x_3603_, 1);
v___x_3605_ = lean_unbox(v_a_3604_);
lean_dec(v_a_3604_);
if (v___x_3605_ == 0)
{
lean_object* v___x_3606_; lean_object* v___f_3607_; lean_object* v___x_3608_; 
v___x_3606_ = lean_box(v___x_3600_);
v___f_3607_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3607_, 0, v___x_3606_);
lean_closure_set(v___f_3607_, 1, v___x_3599_);
lean_closure_set(v___f_3607_, 2, v_fst_3597_);
v___x_3608_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3607_, v___x_3598_, v___y_3536_, v___y_3537_, v___y_3538_, v___y_3539_, v___y_3540_, v___y_3541_, v___y_3542_, v___y_3543_);
if (lean_obj_tag(v___x_3608_) == 0)
{
lean_object* v_a_3609_; uint8_t v___x_3610_; 
v_a_3609_ = lean_ctor_get(v___x_3608_, 0);
lean_inc(v_a_3609_);
lean_dec_ref_known(v___x_3608_, 1);
v___x_3610_ = lean_unbox(v_a_3609_);
lean_dec(v_a_3609_);
if (v___x_3610_ == 0)
{
lean_object* v___x_3611_; lean_object* v___x_3612_; 
v___x_3611_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1);
v___x_3612_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v___x_3611_, v___y_3540_, v___y_3541_, v___y_3542_, v___y_3543_);
if (lean_obj_tag(v___x_3612_) == 0)
{
lean_dec_ref_known(v___x_3612_, 1);
lean_del_object(v___x_3569_);
lean_dec(v_a_3567_);
v_fst_3561_ = v_fst_3571_;
v_snd_3562_ = v_snd_3572_;
goto v___jp_3560_;
}
else
{
lean_object* v_a_3613_; 
v_a_3613_ = lean_ctor_get(v___x_3612_, 0);
lean_inc(v_a_3613_);
lean_dec_ref_known(v___x_3612_, 1);
v_a_3589_ = v_a_3613_;
goto v___jp_3588_;
}
}
else
{
lean_object* v___x_3614_; lean_object* v___x_3615_; lean_object* v___x_3616_; 
lean_del_object(v___x_3569_);
lean_dec(v_a_3567_);
v___x_3614_ = l_Lean_LocalDecl_fvarId(v_val_3565_);
v___x_3615_ = l_Lean_Expr_fvar___override(v___x_3614_);
v___x_3616_ = lean_array_push(v_fst_3571_, v___x_3615_);
v_fst_3561_ = v___x_3616_;
v_snd_3562_ = v_snd_3572_;
goto v___jp_3560_;
}
}
else
{
lean_object* v_a_3617_; 
v_a_3617_ = lean_ctor_get(v___x_3608_, 0);
lean_inc(v_a_3617_);
lean_dec_ref_known(v___x_3608_, 1);
v_a_3589_ = v_a_3617_;
goto v___jp_3588_;
}
}
else
{
lean_object* v___x_3618_; lean_object* v___x_3619_; lean_object* v___x_3620_; 
lean_dec_ref(v___x_3599_);
lean_dec(v_fst_3597_);
lean_del_object(v___x_3569_);
lean_dec(v_a_3567_);
v___x_3618_ = l_Lean_LocalDecl_fvarId(v_val_3565_);
v___x_3619_ = l_Lean_Expr_fvar___override(v___x_3618_);
v___x_3620_ = lean_array_push(v_snd_3572_, v___x_3619_);
v_fst_3561_ = v_fst_3571_;
v_snd_3562_ = v___x_3620_;
goto v___jp_3560_;
}
}
else
{
lean_object* v_a_3621_; 
lean_dec_ref(v___x_3599_);
lean_dec(v_fst_3597_);
v_a_3621_ = lean_ctor_get(v___x_3603_, 0);
lean_inc(v_a_3621_);
lean_dec_ref_known(v___x_3603_, 1);
v_a_3589_ = v_a_3621_;
goto v___jp_3588_;
}
}
else
{
lean_object* v_a_3622_; 
v_a_3622_ = lean_ctor_get(v___x_3593_, 0);
lean_inc(v_a_3622_);
lean_dec_ref_known(v___x_3593_, 1);
v_a_3589_ = v_a_3622_;
goto v___jp_3588_;
}
v___jp_3573_:
{
if (v___y_3575_ == 0)
{
lean_object* v___x_3576_; 
lean_dec_ref(v___y_3574_);
lean_del_object(v___x_3569_);
v___x_3576_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3567_, v___y_3575_, v___y_3537_, v___y_3538_, v___y_3539_, v___y_3540_, v___y_3541_, v___y_3542_, v___y_3543_);
if (lean_obj_tag(v___x_3576_) == 0)
{
lean_dec_ref_known(v___x_3576_, 1);
v_fst_3561_ = v_fst_3571_;
v_snd_3562_ = v_snd_3572_;
goto v___jp_3560_;
}
else
{
lean_object* v_a_3577_; lean_object* v___x_3579_; uint8_t v_isShared_3580_; uint8_t v_isSharedCheck_3584_; 
lean_dec(v_snd_3572_);
lean_dec(v_fst_3571_);
lean_del_object(v___x_3549_);
lean_dec(v___x_3531_);
v_a_3577_ = lean_ctor_get(v___x_3576_, 0);
v_isSharedCheck_3584_ = !lean_is_exclusive(v___x_3576_);
if (v_isSharedCheck_3584_ == 0)
{
v___x_3579_ = v___x_3576_;
v_isShared_3580_ = v_isSharedCheck_3584_;
goto v_resetjp_3578_;
}
else
{
lean_inc(v_a_3577_);
lean_dec(v___x_3576_);
v___x_3579_ = lean_box(0);
v_isShared_3580_ = v_isSharedCheck_3584_;
goto v_resetjp_3578_;
}
v_resetjp_3578_:
{
lean_object* v___x_3582_; 
if (v_isShared_3580_ == 0)
{
v___x_3582_ = v___x_3579_;
goto v_reusejp_3581_;
}
else
{
lean_object* v_reuseFailAlloc_3583_; 
v_reuseFailAlloc_3583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3583_, 0, v_a_3577_);
v___x_3582_ = v_reuseFailAlloc_3583_;
goto v_reusejp_3581_;
}
v_reusejp_3581_:
{
return v___x_3582_;
}
}
}
}
else
{
lean_object* v___x_3586_; 
lean_dec(v_snd_3572_);
lean_dec(v_fst_3571_);
lean_dec(v_a_3567_);
lean_del_object(v___x_3549_);
lean_dec(v___x_3531_);
if (v_isShared_3570_ == 0)
{
lean_ctor_set_tag(v___x_3569_, 1);
lean_ctor_set(v___x_3569_, 0, v___y_3574_);
v___x_3586_ = v___x_3569_;
goto v_reusejp_3585_;
}
else
{
lean_object* v_reuseFailAlloc_3587_; 
v_reuseFailAlloc_3587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3587_, 0, v___y_3574_);
v___x_3586_ = v_reuseFailAlloc_3587_;
goto v_reusejp_3585_;
}
v_reusejp_3585_:
{
return v___x_3586_;
}
}
}
v___jp_3588_:
{
uint8_t v___x_3590_; 
v___x_3590_ = l_Lean_Exception_isInterrupt(v_a_3589_);
if (v___x_3590_ == 0)
{
uint8_t v___x_3591_; 
lean_inc_ref(v_a_3589_);
v___x_3591_ = l_Lean_Exception_isRuntime(v_a_3589_);
v___y_3574_ = v_a_3589_;
v___y_3575_ = v___x_3591_;
goto v___jp_3573_;
}
else
{
v___y_3574_ = v_a_3589_;
v___y_3575_ = v___x_3590_;
goto v___jp_3573_;
}
}
}
}
else
{
lean_object* v_a_3624_; lean_object* v___x_3626_; uint8_t v_isShared_3627_; uint8_t v_isSharedCheck_3631_; 
lean_del_object(v___x_3549_);
lean_dec(v_snd_3547_);
lean_dec(v___x_3531_);
v_a_3624_ = lean_ctor_get(v___x_3566_, 0);
v_isSharedCheck_3631_ = !lean_is_exclusive(v___x_3566_);
if (v_isSharedCheck_3631_ == 0)
{
v___x_3626_ = v___x_3566_;
v_isShared_3627_ = v_isSharedCheck_3631_;
goto v_resetjp_3625_;
}
else
{
lean_inc(v_a_3624_);
lean_dec(v___x_3566_);
v___x_3626_ = lean_box(0);
v_isShared_3627_ = v_isSharedCheck_3631_;
goto v_resetjp_3625_;
}
v_resetjp_3625_:
{
lean_object* v___x_3629_; 
if (v_isShared_3627_ == 0)
{
v___x_3629_ = v___x_3626_;
goto v_reusejp_3628_;
}
else
{
lean_object* v_reuseFailAlloc_3630_; 
v_reuseFailAlloc_3630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3630_, 0, v_a_3624_);
v___x_3629_ = v_reuseFailAlloc_3630_;
goto v_reusejp_3628_;
}
v_reusejp_3628_:
{
return v___x_3629_;
}
}
}
}
v___jp_3552_:
{
lean_object* v___x_3555_; 
if (v_isShared_3550_ == 0)
{
lean_ctor_set(v___x_3549_, 1, v_a_3553_);
lean_ctor_set(v___x_3549_, 0, v___x_3551_);
v___x_3555_ = v___x_3549_;
goto v_reusejp_3554_;
}
else
{
lean_object* v_reuseFailAlloc_3559_; 
v_reuseFailAlloc_3559_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3559_, 0, v___x_3551_);
lean_ctor_set(v_reuseFailAlloc_3559_, 1, v_a_3553_);
v___x_3555_ = v_reuseFailAlloc_3559_;
goto v_reusejp_3554_;
}
v_reusejp_3554_:
{
size_t v___x_3556_; size_t v___x_3557_; 
v___x_3556_ = ((size_t)1ULL);
v___x_3557_ = lean_usize_add(v_i_3534_, v___x_3556_);
v_i_3534_ = v___x_3557_;
v_b_3535_ = v___x_3555_;
goto _start;
}
}
v___jp_3560_:
{
lean_object* v___x_3563_; 
v___x_3563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3563_, 0, v_fst_3561_);
lean_ctor_set(v___x_3563_, 1, v_snd_3562_);
v_a_3553_ = v___x_3563_;
goto v___jp_3552_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___boxed(lean_object* v___x_3634_, lean_object* v_as_3635_, lean_object* v_sz_3636_, lean_object* v_i_3637_, lean_object* v_b_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_, lean_object* v___y_3642_, lean_object* v___y_3643_, lean_object* v___y_3644_, lean_object* v___y_3645_, lean_object* v___y_3646_, lean_object* v___y_3647_){
_start:
{
size_t v_sz_boxed_3648_; size_t v_i_boxed_3649_; lean_object* v_res_3650_; 
v_sz_boxed_3648_ = lean_unbox_usize(v_sz_3636_);
lean_dec(v_sz_3636_);
v_i_boxed_3649_ = lean_unbox_usize(v_i_3637_);
lean_dec(v_i_3637_);
v_res_3650_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9(v___x_3634_, v_as_3635_, v_sz_boxed_3648_, v_i_boxed_3649_, v_b_3638_, v___y_3639_, v___y_3640_, v___y_3641_, v___y_3642_, v___y_3643_, v___y_3644_, v___y_3645_, v___y_3646_);
lean_dec(v___y_3646_);
lean_dec_ref(v___y_3645_);
lean_dec(v___y_3644_);
lean_dec_ref(v___y_3643_);
lean_dec(v___y_3642_);
lean_dec_ref(v___y_3641_);
lean_dec(v___y_3640_);
lean_dec_ref(v___y_3639_);
lean_dec_ref(v_as_3635_);
return v_res_3650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6(lean_object* v___x_3651_, lean_object* v_as_3652_, size_t v_sz_3653_, size_t v_i_3654_, lean_object* v_b_3655_, lean_object* v___y_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_, lean_object* v___y_3663_){
_start:
{
uint8_t v___x_3665_; 
v___x_3665_ = lean_usize_dec_lt(v_i_3654_, v_sz_3653_);
if (v___x_3665_ == 0)
{
lean_object* v___x_3666_; 
lean_dec(v___x_3651_);
v___x_3666_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3666_, 0, v_b_3655_);
return v___x_3666_;
}
else
{
lean_object* v_snd_3667_; lean_object* v___x_3669_; uint8_t v_isShared_3670_; uint8_t v_isSharedCheck_3752_; 
v_snd_3667_ = lean_ctor_get(v_b_3655_, 1);
v_isSharedCheck_3752_ = !lean_is_exclusive(v_b_3655_);
if (v_isSharedCheck_3752_ == 0)
{
lean_object* v_unused_3753_; 
v_unused_3753_ = lean_ctor_get(v_b_3655_, 0);
lean_dec(v_unused_3753_);
v___x_3669_ = v_b_3655_;
v_isShared_3670_ = v_isSharedCheck_3752_;
goto v_resetjp_3668_;
}
else
{
lean_inc(v_snd_3667_);
lean_dec(v_b_3655_);
v___x_3669_ = lean_box(0);
v_isShared_3670_ = v_isSharedCheck_3752_;
goto v_resetjp_3668_;
}
v_resetjp_3668_:
{
lean_object* v___x_3671_; lean_object* v_a_3673_; lean_object* v_fst_3681_; lean_object* v_snd_3682_; lean_object* v_a_3684_; 
v___x_3671_ = lean_box(0);
v_a_3684_ = lean_array_uget_borrowed(v_as_3652_, v_i_3654_);
if (lean_obj_tag(v_a_3684_) == 0)
{
v_a_3673_ = v_snd_3667_;
goto v___jp_3672_;
}
else
{
lean_object* v_val_3685_; lean_object* v___x_3686_; 
v_val_3685_ = lean_ctor_get(v_a_3684_, 0);
v___x_3686_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3657_, v___y_3659_, v___y_3661_, v___y_3663_);
if (lean_obj_tag(v___x_3686_) == 0)
{
lean_object* v_a_3687_; lean_object* v___x_3689_; uint8_t v_isShared_3690_; uint8_t v_isSharedCheck_3743_; 
v_a_3687_ = lean_ctor_get(v___x_3686_, 0);
v_isSharedCheck_3743_ = !lean_is_exclusive(v___x_3686_);
if (v_isSharedCheck_3743_ == 0)
{
v___x_3689_ = v___x_3686_;
v_isShared_3690_ = v_isSharedCheck_3743_;
goto v_resetjp_3688_;
}
else
{
lean_inc(v_a_3687_);
lean_dec(v___x_3686_);
v___x_3689_ = lean_box(0);
v_isShared_3690_ = v_isSharedCheck_3743_;
goto v_resetjp_3688_;
}
v_resetjp_3688_:
{
lean_object* v_fst_3691_; lean_object* v_snd_3692_; lean_object* v___y_3694_; uint8_t v___y_3695_; lean_object* v_a_3709_; lean_object* v___x_3712_; lean_object* v___x_3713_; 
v_fst_3691_ = lean_ctor_get(v_snd_3667_, 0);
lean_inc(v_fst_3691_);
v_snd_3692_ = lean_ctor_get(v_snd_3667_, 1);
lean_inc(v_snd_3692_);
lean_dec(v_snd_3667_);
v___x_3712_ = l_Lean_LocalDecl_type(v_val_3685_);
v___x_3713_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v___x_3712_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_);
if (lean_obj_tag(v___x_3713_) == 0)
{
lean_object* v_a_3714_; lean_object* v_snd_3715_; lean_object* v_fst_3716_; lean_object* v_fst_3717_; uint8_t v___x_3718_; lean_object* v___x_3719_; uint8_t v___x_3720_; lean_object* v___x_3721_; lean_object* v___f_3722_; lean_object* v___x_3723_; 
v_a_3714_ = lean_ctor_get(v___x_3713_, 0);
lean_inc(v_a_3714_);
lean_dec_ref_known(v___x_3713_, 1);
v_snd_3715_ = lean_ctor_get(v_a_3714_, 1);
lean_inc(v_snd_3715_);
v_fst_3716_ = lean_ctor_get(v_a_3714_, 0);
lean_inc(v_fst_3716_);
lean_dec(v_a_3714_);
v_fst_3717_ = lean_ctor_get(v_snd_3715_, 0);
lean_inc(v_fst_3717_);
lean_dec(v_snd_3715_);
v___x_3718_ = 0;
lean_inc(v___x_3651_);
v___x_3719_ = l_Lean_Expr_fvar___override(v___x_3651_);
v___x_3720_ = 2;
v___x_3721_ = lean_box(v___x_3720_);
lean_inc_ref(v___x_3719_);
v___f_3722_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3722_, 0, v___x_3721_);
lean_closure_set(v___f_3722_, 1, v___x_3719_);
lean_closure_set(v___f_3722_, 2, v_fst_3716_);
v___x_3723_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3722_, v___x_3718_, v___y_3656_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_);
if (lean_obj_tag(v___x_3723_) == 0)
{
lean_object* v_a_3724_; uint8_t v___x_3725_; 
v_a_3724_ = lean_ctor_get(v___x_3723_, 0);
lean_inc(v_a_3724_);
lean_dec_ref_known(v___x_3723_, 1);
v___x_3725_ = lean_unbox(v_a_3724_);
lean_dec(v_a_3724_);
if (v___x_3725_ == 0)
{
lean_object* v___x_3726_; lean_object* v___f_3727_; lean_object* v___x_3728_; 
v___x_3726_ = lean_box(v___x_3720_);
v___f_3727_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3727_, 0, v___x_3726_);
lean_closure_set(v___f_3727_, 1, v___x_3719_);
lean_closure_set(v___f_3727_, 2, v_fst_3717_);
v___x_3728_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3727_, v___x_3718_, v___y_3656_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_);
if (lean_obj_tag(v___x_3728_) == 0)
{
lean_object* v_a_3729_; uint8_t v___x_3730_; 
v_a_3729_ = lean_ctor_get(v___x_3728_, 0);
lean_inc(v_a_3729_);
lean_dec_ref_known(v___x_3728_, 1);
v___x_3730_ = lean_unbox(v_a_3729_);
lean_dec(v_a_3729_);
if (v___x_3730_ == 0)
{
lean_object* v___x_3731_; lean_object* v___x_3732_; 
v___x_3731_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1);
v___x_3732_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v___x_3731_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_);
if (lean_obj_tag(v___x_3732_) == 0)
{
lean_dec_ref_known(v___x_3732_, 1);
lean_del_object(v___x_3689_);
lean_dec(v_a_3687_);
v_fst_3681_ = v_fst_3691_;
v_snd_3682_ = v_snd_3692_;
goto v___jp_3680_;
}
else
{
lean_object* v_a_3733_; 
v_a_3733_ = lean_ctor_get(v___x_3732_, 0);
lean_inc(v_a_3733_);
lean_dec_ref_known(v___x_3732_, 1);
v_a_3709_ = v_a_3733_;
goto v___jp_3708_;
}
}
else
{
lean_object* v___x_3734_; lean_object* v___x_3735_; lean_object* v___x_3736_; 
lean_del_object(v___x_3689_);
lean_dec(v_a_3687_);
v___x_3734_ = l_Lean_LocalDecl_fvarId(v_val_3685_);
v___x_3735_ = l_Lean_Expr_fvar___override(v___x_3734_);
v___x_3736_ = lean_array_push(v_fst_3691_, v___x_3735_);
v_fst_3681_ = v___x_3736_;
v_snd_3682_ = v_snd_3692_;
goto v___jp_3680_;
}
}
else
{
lean_object* v_a_3737_; 
v_a_3737_ = lean_ctor_get(v___x_3728_, 0);
lean_inc(v_a_3737_);
lean_dec_ref_known(v___x_3728_, 1);
v_a_3709_ = v_a_3737_;
goto v___jp_3708_;
}
}
else
{
lean_object* v___x_3738_; lean_object* v___x_3739_; lean_object* v___x_3740_; 
lean_dec_ref(v___x_3719_);
lean_dec(v_fst_3717_);
lean_del_object(v___x_3689_);
lean_dec(v_a_3687_);
v___x_3738_ = l_Lean_LocalDecl_fvarId(v_val_3685_);
v___x_3739_ = l_Lean_Expr_fvar___override(v___x_3738_);
v___x_3740_ = lean_array_push(v_snd_3692_, v___x_3739_);
v_fst_3681_ = v_fst_3691_;
v_snd_3682_ = v___x_3740_;
goto v___jp_3680_;
}
}
else
{
lean_object* v_a_3741_; 
lean_dec_ref(v___x_3719_);
lean_dec(v_fst_3717_);
v_a_3741_ = lean_ctor_get(v___x_3723_, 0);
lean_inc(v_a_3741_);
lean_dec_ref_known(v___x_3723_, 1);
v_a_3709_ = v_a_3741_;
goto v___jp_3708_;
}
}
else
{
lean_object* v_a_3742_; 
v_a_3742_ = lean_ctor_get(v___x_3713_, 0);
lean_inc(v_a_3742_);
lean_dec_ref_known(v___x_3713_, 1);
v_a_3709_ = v_a_3742_;
goto v___jp_3708_;
}
v___jp_3693_:
{
if (v___y_3695_ == 0)
{
lean_object* v___x_3696_; 
lean_dec_ref(v___y_3694_);
lean_del_object(v___x_3689_);
v___x_3696_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3687_, v___y_3695_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_);
if (lean_obj_tag(v___x_3696_) == 0)
{
lean_dec_ref_known(v___x_3696_, 1);
v_fst_3681_ = v_fst_3691_;
v_snd_3682_ = v_snd_3692_;
goto v___jp_3680_;
}
else
{
lean_object* v_a_3697_; lean_object* v___x_3699_; uint8_t v_isShared_3700_; uint8_t v_isSharedCheck_3704_; 
lean_dec(v_snd_3692_);
lean_dec(v_fst_3691_);
lean_del_object(v___x_3669_);
lean_dec(v___x_3651_);
v_a_3697_ = lean_ctor_get(v___x_3696_, 0);
v_isSharedCheck_3704_ = !lean_is_exclusive(v___x_3696_);
if (v_isSharedCheck_3704_ == 0)
{
v___x_3699_ = v___x_3696_;
v_isShared_3700_ = v_isSharedCheck_3704_;
goto v_resetjp_3698_;
}
else
{
lean_inc(v_a_3697_);
lean_dec(v___x_3696_);
v___x_3699_ = lean_box(0);
v_isShared_3700_ = v_isSharedCheck_3704_;
goto v_resetjp_3698_;
}
v_resetjp_3698_:
{
lean_object* v___x_3702_; 
if (v_isShared_3700_ == 0)
{
v___x_3702_ = v___x_3699_;
goto v_reusejp_3701_;
}
else
{
lean_object* v_reuseFailAlloc_3703_; 
v_reuseFailAlloc_3703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3703_, 0, v_a_3697_);
v___x_3702_ = v_reuseFailAlloc_3703_;
goto v_reusejp_3701_;
}
v_reusejp_3701_:
{
return v___x_3702_;
}
}
}
}
else
{
lean_object* v___x_3706_; 
lean_dec(v_snd_3692_);
lean_dec(v_fst_3691_);
lean_dec(v_a_3687_);
lean_del_object(v___x_3669_);
lean_dec(v___x_3651_);
if (v_isShared_3690_ == 0)
{
lean_ctor_set_tag(v___x_3689_, 1);
lean_ctor_set(v___x_3689_, 0, v___y_3694_);
v___x_3706_ = v___x_3689_;
goto v_reusejp_3705_;
}
else
{
lean_object* v_reuseFailAlloc_3707_; 
v_reuseFailAlloc_3707_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3707_, 0, v___y_3694_);
v___x_3706_ = v_reuseFailAlloc_3707_;
goto v_reusejp_3705_;
}
v_reusejp_3705_:
{
return v___x_3706_;
}
}
}
v___jp_3708_:
{
uint8_t v___x_3710_; 
v___x_3710_ = l_Lean_Exception_isInterrupt(v_a_3709_);
if (v___x_3710_ == 0)
{
uint8_t v___x_3711_; 
lean_inc_ref(v_a_3709_);
v___x_3711_ = l_Lean_Exception_isRuntime(v_a_3709_);
v___y_3694_ = v_a_3709_;
v___y_3695_ = v___x_3711_;
goto v___jp_3693_;
}
else
{
v___y_3694_ = v_a_3709_;
v___y_3695_ = v___x_3710_;
goto v___jp_3693_;
}
}
}
}
else
{
lean_object* v_a_3744_; lean_object* v___x_3746_; uint8_t v_isShared_3747_; uint8_t v_isSharedCheck_3751_; 
lean_del_object(v___x_3669_);
lean_dec(v_snd_3667_);
lean_dec(v___x_3651_);
v_a_3744_ = lean_ctor_get(v___x_3686_, 0);
v_isSharedCheck_3751_ = !lean_is_exclusive(v___x_3686_);
if (v_isSharedCheck_3751_ == 0)
{
v___x_3746_ = v___x_3686_;
v_isShared_3747_ = v_isSharedCheck_3751_;
goto v_resetjp_3745_;
}
else
{
lean_inc(v_a_3744_);
lean_dec(v___x_3686_);
v___x_3746_ = lean_box(0);
v_isShared_3747_ = v_isSharedCheck_3751_;
goto v_resetjp_3745_;
}
v_resetjp_3745_:
{
lean_object* v___x_3749_; 
if (v_isShared_3747_ == 0)
{
v___x_3749_ = v___x_3746_;
goto v_reusejp_3748_;
}
else
{
lean_object* v_reuseFailAlloc_3750_; 
v_reuseFailAlloc_3750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3750_, 0, v_a_3744_);
v___x_3749_ = v_reuseFailAlloc_3750_;
goto v_reusejp_3748_;
}
v_reusejp_3748_:
{
return v___x_3749_;
}
}
}
}
v___jp_3672_:
{
lean_object* v___x_3675_; 
if (v_isShared_3670_ == 0)
{
lean_ctor_set(v___x_3669_, 1, v_a_3673_);
lean_ctor_set(v___x_3669_, 0, v___x_3671_);
v___x_3675_ = v___x_3669_;
goto v_reusejp_3674_;
}
else
{
lean_object* v_reuseFailAlloc_3679_; 
v_reuseFailAlloc_3679_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3679_, 0, v___x_3671_);
lean_ctor_set(v_reuseFailAlloc_3679_, 1, v_a_3673_);
v___x_3675_ = v_reuseFailAlloc_3679_;
goto v_reusejp_3674_;
}
v_reusejp_3674_:
{
size_t v___x_3676_; size_t v___x_3677_; lean_object* v___x_3678_; 
v___x_3676_ = ((size_t)1ULL);
v___x_3677_ = lean_usize_add(v_i_3654_, v___x_3676_);
v___x_3678_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9(v___x_3651_, v_as_3652_, v_sz_3653_, v___x_3677_, v___x_3675_, v___y_3656_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, v___y_3663_);
return v___x_3678_;
}
}
v___jp_3680_:
{
lean_object* v___x_3683_; 
v___x_3683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3683_, 0, v_fst_3681_);
lean_ctor_set(v___x_3683_, 1, v_snd_3682_);
v_a_3673_ = v___x_3683_;
goto v___jp_3672_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___boxed(lean_object* v___x_3754_, lean_object* v_as_3755_, lean_object* v_sz_3756_, lean_object* v_i_3757_, lean_object* v_b_3758_, lean_object* v___y_3759_, lean_object* v___y_3760_, lean_object* v___y_3761_, lean_object* v___y_3762_, lean_object* v___y_3763_, lean_object* v___y_3764_, lean_object* v___y_3765_, lean_object* v___y_3766_, lean_object* v___y_3767_){
_start:
{
size_t v_sz_boxed_3768_; size_t v_i_boxed_3769_; lean_object* v_res_3770_; 
v_sz_boxed_3768_ = lean_unbox_usize(v_sz_3756_);
lean_dec(v_sz_3756_);
v_i_boxed_3769_ = lean_unbox_usize(v_i_3757_);
lean_dec(v_i_3757_);
v_res_3770_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6(v___x_3754_, v_as_3755_, v_sz_boxed_3768_, v_i_boxed_3769_, v_b_3758_, v___y_3759_, v___y_3760_, v___y_3761_, v___y_3762_, v___y_3763_, v___y_3764_, v___y_3765_, v___y_3766_);
lean_dec(v___y_3766_);
lean_dec_ref(v___y_3765_);
lean_dec(v___y_3764_);
lean_dec_ref(v___y_3763_);
lean_dec(v___y_3762_);
lean_dec_ref(v___y_3761_);
lean_dec(v___y_3760_);
lean_dec_ref(v___y_3759_);
lean_dec_ref(v_as_3755_);
return v_res_3770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7_spec__9(lean_object* v___x_3771_, lean_object* v_as_3772_, size_t v_sz_3773_, size_t v_i_3774_, lean_object* v_b_3775_, lean_object* v___y_3776_, lean_object* v___y_3777_, lean_object* v___y_3778_, lean_object* v___y_3779_, lean_object* v___y_3780_, lean_object* v___y_3781_, lean_object* v___y_3782_, lean_object* v___y_3783_){
_start:
{
uint8_t v___x_3785_; 
v___x_3785_ = lean_usize_dec_lt(v_i_3774_, v_sz_3773_);
if (v___x_3785_ == 0)
{
lean_object* v___x_3786_; 
lean_dec(v___x_3771_);
v___x_3786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3786_, 0, v_b_3775_);
return v___x_3786_;
}
else
{
lean_object* v_snd_3787_; lean_object* v___x_3789_; uint8_t v_isShared_3790_; uint8_t v_isSharedCheck_3872_; 
v_snd_3787_ = lean_ctor_get(v_b_3775_, 1);
v_isSharedCheck_3872_ = !lean_is_exclusive(v_b_3775_);
if (v_isSharedCheck_3872_ == 0)
{
lean_object* v_unused_3873_; 
v_unused_3873_ = lean_ctor_get(v_b_3775_, 0);
lean_dec(v_unused_3873_);
v___x_3789_ = v_b_3775_;
v_isShared_3790_ = v_isSharedCheck_3872_;
goto v_resetjp_3788_;
}
else
{
lean_inc(v_snd_3787_);
lean_dec(v_b_3775_);
v___x_3789_ = lean_box(0);
v_isShared_3790_ = v_isSharedCheck_3872_;
goto v_resetjp_3788_;
}
v_resetjp_3788_:
{
lean_object* v___x_3791_; lean_object* v_a_3793_; lean_object* v_fst_3801_; lean_object* v_snd_3802_; lean_object* v_a_3804_; 
v___x_3791_ = lean_box(0);
v_a_3804_ = lean_array_uget_borrowed(v_as_3772_, v_i_3774_);
if (lean_obj_tag(v_a_3804_) == 0)
{
v_a_3793_ = v_snd_3787_;
goto v___jp_3792_;
}
else
{
lean_object* v_val_3805_; lean_object* v___x_3806_; 
v_val_3805_ = lean_ctor_get(v_a_3804_, 0);
v___x_3806_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3777_, v___y_3779_, v___y_3781_, v___y_3783_);
if (lean_obj_tag(v___x_3806_) == 0)
{
lean_object* v_a_3807_; lean_object* v___x_3809_; uint8_t v_isShared_3810_; uint8_t v_isSharedCheck_3863_; 
v_a_3807_ = lean_ctor_get(v___x_3806_, 0);
v_isSharedCheck_3863_ = !lean_is_exclusive(v___x_3806_);
if (v_isSharedCheck_3863_ == 0)
{
v___x_3809_ = v___x_3806_;
v_isShared_3810_ = v_isSharedCheck_3863_;
goto v_resetjp_3808_;
}
else
{
lean_inc(v_a_3807_);
lean_dec(v___x_3806_);
v___x_3809_ = lean_box(0);
v_isShared_3810_ = v_isSharedCheck_3863_;
goto v_resetjp_3808_;
}
v_resetjp_3808_:
{
lean_object* v_fst_3811_; lean_object* v_snd_3812_; lean_object* v___y_3814_; uint8_t v___y_3815_; lean_object* v_a_3829_; lean_object* v___x_3832_; lean_object* v___x_3833_; 
v_fst_3811_ = lean_ctor_get(v_snd_3787_, 0);
lean_inc(v_fst_3811_);
v_snd_3812_ = lean_ctor_get(v_snd_3787_, 1);
lean_inc(v_snd_3812_);
lean_dec(v_snd_3787_);
v___x_3832_ = l_Lean_LocalDecl_type(v_val_3805_);
v___x_3833_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v___x_3832_, v___y_3780_, v___y_3781_, v___y_3782_, v___y_3783_);
if (lean_obj_tag(v___x_3833_) == 0)
{
lean_object* v_a_3834_; lean_object* v_snd_3835_; lean_object* v_fst_3836_; lean_object* v_fst_3837_; uint8_t v___x_3838_; lean_object* v___x_3839_; uint8_t v___x_3840_; lean_object* v___x_3841_; lean_object* v___f_3842_; lean_object* v___x_3843_; 
v_a_3834_ = lean_ctor_get(v___x_3833_, 0);
lean_inc(v_a_3834_);
lean_dec_ref_known(v___x_3833_, 1);
v_snd_3835_ = lean_ctor_get(v_a_3834_, 1);
lean_inc(v_snd_3835_);
v_fst_3836_ = lean_ctor_get(v_a_3834_, 0);
lean_inc(v_fst_3836_);
lean_dec(v_a_3834_);
v_fst_3837_ = lean_ctor_get(v_snd_3835_, 0);
lean_inc(v_fst_3837_);
lean_dec(v_snd_3835_);
v___x_3838_ = 0;
lean_inc(v___x_3771_);
v___x_3839_ = l_Lean_Expr_fvar___override(v___x_3771_);
v___x_3840_ = 2;
v___x_3841_ = lean_box(v___x_3840_);
lean_inc_ref(v___x_3839_);
v___f_3842_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3842_, 0, v___x_3841_);
lean_closure_set(v___f_3842_, 1, v___x_3839_);
lean_closure_set(v___f_3842_, 2, v_fst_3836_);
v___x_3843_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3842_, v___x_3838_, v___y_3776_, v___y_3777_, v___y_3778_, v___y_3779_, v___y_3780_, v___y_3781_, v___y_3782_, v___y_3783_);
if (lean_obj_tag(v___x_3843_) == 0)
{
lean_object* v_a_3844_; uint8_t v___x_3845_; 
v_a_3844_ = lean_ctor_get(v___x_3843_, 0);
lean_inc(v_a_3844_);
lean_dec_ref_known(v___x_3843_, 1);
v___x_3845_ = lean_unbox(v_a_3844_);
lean_dec(v_a_3844_);
if (v___x_3845_ == 0)
{
lean_object* v___x_3846_; lean_object* v___f_3847_; lean_object* v___x_3848_; 
v___x_3846_ = lean_box(v___x_3840_);
v___f_3847_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3847_, 0, v___x_3846_);
lean_closure_set(v___f_3847_, 1, v___x_3839_);
lean_closure_set(v___f_3847_, 2, v_fst_3837_);
v___x_3848_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3847_, v___x_3838_, v___y_3776_, v___y_3777_, v___y_3778_, v___y_3779_, v___y_3780_, v___y_3781_, v___y_3782_, v___y_3783_);
if (lean_obj_tag(v___x_3848_) == 0)
{
lean_object* v_a_3849_; uint8_t v___x_3850_; 
v_a_3849_ = lean_ctor_get(v___x_3848_, 0);
lean_inc(v_a_3849_);
lean_dec_ref_known(v___x_3848_, 1);
v___x_3850_ = lean_unbox(v_a_3849_);
lean_dec(v_a_3849_);
if (v___x_3850_ == 0)
{
lean_object* v___x_3851_; lean_object* v___x_3852_; 
v___x_3851_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1);
v___x_3852_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v___x_3851_, v___y_3780_, v___y_3781_, v___y_3782_, v___y_3783_);
if (lean_obj_tag(v___x_3852_) == 0)
{
lean_dec_ref_known(v___x_3852_, 1);
lean_del_object(v___x_3809_);
lean_dec(v_a_3807_);
v_fst_3801_ = v_fst_3811_;
v_snd_3802_ = v_snd_3812_;
goto v___jp_3800_;
}
else
{
lean_object* v_a_3853_; 
v_a_3853_ = lean_ctor_get(v___x_3852_, 0);
lean_inc(v_a_3853_);
lean_dec_ref_known(v___x_3852_, 1);
v_a_3829_ = v_a_3853_;
goto v___jp_3828_;
}
}
else
{
lean_object* v___x_3854_; lean_object* v___x_3855_; lean_object* v___x_3856_; 
lean_del_object(v___x_3809_);
lean_dec(v_a_3807_);
v___x_3854_ = l_Lean_LocalDecl_fvarId(v_val_3805_);
v___x_3855_ = l_Lean_Expr_fvar___override(v___x_3854_);
v___x_3856_ = lean_array_push(v_fst_3811_, v___x_3855_);
v_fst_3801_ = v___x_3856_;
v_snd_3802_ = v_snd_3812_;
goto v___jp_3800_;
}
}
else
{
lean_object* v_a_3857_; 
v_a_3857_ = lean_ctor_get(v___x_3848_, 0);
lean_inc(v_a_3857_);
lean_dec_ref_known(v___x_3848_, 1);
v_a_3829_ = v_a_3857_;
goto v___jp_3828_;
}
}
else
{
lean_object* v___x_3858_; lean_object* v___x_3859_; lean_object* v___x_3860_; 
lean_dec_ref(v___x_3839_);
lean_dec(v_fst_3837_);
lean_del_object(v___x_3809_);
lean_dec(v_a_3807_);
v___x_3858_ = l_Lean_LocalDecl_fvarId(v_val_3805_);
v___x_3859_ = l_Lean_Expr_fvar___override(v___x_3858_);
v___x_3860_ = lean_array_push(v_snd_3812_, v___x_3859_);
v_fst_3801_ = v_fst_3811_;
v_snd_3802_ = v___x_3860_;
goto v___jp_3800_;
}
}
else
{
lean_object* v_a_3861_; 
lean_dec_ref(v___x_3839_);
lean_dec(v_fst_3837_);
v_a_3861_ = lean_ctor_get(v___x_3843_, 0);
lean_inc(v_a_3861_);
lean_dec_ref_known(v___x_3843_, 1);
v_a_3829_ = v_a_3861_;
goto v___jp_3828_;
}
}
else
{
lean_object* v_a_3862_; 
v_a_3862_ = lean_ctor_get(v___x_3833_, 0);
lean_inc(v_a_3862_);
lean_dec_ref_known(v___x_3833_, 1);
v_a_3829_ = v_a_3862_;
goto v___jp_3828_;
}
v___jp_3813_:
{
if (v___y_3815_ == 0)
{
lean_object* v___x_3816_; 
lean_dec_ref(v___y_3814_);
lean_del_object(v___x_3809_);
v___x_3816_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3807_, v___y_3815_, v___y_3777_, v___y_3778_, v___y_3779_, v___y_3780_, v___y_3781_, v___y_3782_, v___y_3783_);
if (lean_obj_tag(v___x_3816_) == 0)
{
lean_dec_ref_known(v___x_3816_, 1);
v_fst_3801_ = v_fst_3811_;
v_snd_3802_ = v_snd_3812_;
goto v___jp_3800_;
}
else
{
lean_object* v_a_3817_; lean_object* v___x_3819_; uint8_t v_isShared_3820_; uint8_t v_isSharedCheck_3824_; 
lean_dec(v_snd_3812_);
lean_dec(v_fst_3811_);
lean_del_object(v___x_3789_);
lean_dec(v___x_3771_);
v_a_3817_ = lean_ctor_get(v___x_3816_, 0);
v_isSharedCheck_3824_ = !lean_is_exclusive(v___x_3816_);
if (v_isSharedCheck_3824_ == 0)
{
v___x_3819_ = v___x_3816_;
v_isShared_3820_ = v_isSharedCheck_3824_;
goto v_resetjp_3818_;
}
else
{
lean_inc(v_a_3817_);
lean_dec(v___x_3816_);
v___x_3819_ = lean_box(0);
v_isShared_3820_ = v_isSharedCheck_3824_;
goto v_resetjp_3818_;
}
v_resetjp_3818_:
{
lean_object* v___x_3822_; 
if (v_isShared_3820_ == 0)
{
v___x_3822_ = v___x_3819_;
goto v_reusejp_3821_;
}
else
{
lean_object* v_reuseFailAlloc_3823_; 
v_reuseFailAlloc_3823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3823_, 0, v_a_3817_);
v___x_3822_ = v_reuseFailAlloc_3823_;
goto v_reusejp_3821_;
}
v_reusejp_3821_:
{
return v___x_3822_;
}
}
}
}
else
{
lean_object* v___x_3826_; 
lean_dec(v_snd_3812_);
lean_dec(v_fst_3811_);
lean_dec(v_a_3807_);
lean_del_object(v___x_3789_);
lean_dec(v___x_3771_);
if (v_isShared_3810_ == 0)
{
lean_ctor_set_tag(v___x_3809_, 1);
lean_ctor_set(v___x_3809_, 0, v___y_3814_);
v___x_3826_ = v___x_3809_;
goto v_reusejp_3825_;
}
else
{
lean_object* v_reuseFailAlloc_3827_; 
v_reuseFailAlloc_3827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3827_, 0, v___y_3814_);
v___x_3826_ = v_reuseFailAlloc_3827_;
goto v_reusejp_3825_;
}
v_reusejp_3825_:
{
return v___x_3826_;
}
}
}
v___jp_3828_:
{
uint8_t v___x_3830_; 
v___x_3830_ = l_Lean_Exception_isInterrupt(v_a_3829_);
if (v___x_3830_ == 0)
{
uint8_t v___x_3831_; 
lean_inc_ref(v_a_3829_);
v___x_3831_ = l_Lean_Exception_isRuntime(v_a_3829_);
v___y_3814_ = v_a_3829_;
v___y_3815_ = v___x_3831_;
goto v___jp_3813_;
}
else
{
v___y_3814_ = v_a_3829_;
v___y_3815_ = v___x_3830_;
goto v___jp_3813_;
}
}
}
}
else
{
lean_object* v_a_3864_; lean_object* v___x_3866_; uint8_t v_isShared_3867_; uint8_t v_isSharedCheck_3871_; 
lean_del_object(v___x_3789_);
lean_dec(v_snd_3787_);
lean_dec(v___x_3771_);
v_a_3864_ = lean_ctor_get(v___x_3806_, 0);
v_isSharedCheck_3871_ = !lean_is_exclusive(v___x_3806_);
if (v_isSharedCheck_3871_ == 0)
{
v___x_3866_ = v___x_3806_;
v_isShared_3867_ = v_isSharedCheck_3871_;
goto v_resetjp_3865_;
}
else
{
lean_inc(v_a_3864_);
lean_dec(v___x_3806_);
v___x_3866_ = lean_box(0);
v_isShared_3867_ = v_isSharedCheck_3871_;
goto v_resetjp_3865_;
}
v_resetjp_3865_:
{
lean_object* v___x_3869_; 
if (v_isShared_3867_ == 0)
{
v___x_3869_ = v___x_3866_;
goto v_reusejp_3868_;
}
else
{
lean_object* v_reuseFailAlloc_3870_; 
v_reuseFailAlloc_3870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3870_, 0, v_a_3864_);
v___x_3869_ = v_reuseFailAlloc_3870_;
goto v_reusejp_3868_;
}
v_reusejp_3868_:
{
return v___x_3869_;
}
}
}
}
v___jp_3792_:
{
lean_object* v___x_3795_; 
if (v_isShared_3790_ == 0)
{
lean_ctor_set(v___x_3789_, 1, v_a_3793_);
lean_ctor_set(v___x_3789_, 0, v___x_3791_);
v___x_3795_ = v___x_3789_;
goto v_reusejp_3794_;
}
else
{
lean_object* v_reuseFailAlloc_3799_; 
v_reuseFailAlloc_3799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3799_, 0, v___x_3791_);
lean_ctor_set(v_reuseFailAlloc_3799_, 1, v_a_3793_);
v___x_3795_ = v_reuseFailAlloc_3799_;
goto v_reusejp_3794_;
}
v_reusejp_3794_:
{
size_t v___x_3796_; size_t v___x_3797_; 
v___x_3796_ = ((size_t)1ULL);
v___x_3797_ = lean_usize_add(v_i_3774_, v___x_3796_);
v_i_3774_ = v___x_3797_;
v_b_3775_ = v___x_3795_;
goto _start;
}
}
v___jp_3800_:
{
lean_object* v___x_3803_; 
v___x_3803_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3803_, 0, v_fst_3801_);
lean_ctor_set(v___x_3803_, 1, v_snd_3802_);
v_a_3793_ = v___x_3803_;
goto v___jp_3792_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7_spec__9___boxed(lean_object* v___x_3874_, lean_object* v_as_3875_, lean_object* v_sz_3876_, lean_object* v_i_3877_, lean_object* v_b_3878_, lean_object* v___y_3879_, lean_object* v___y_3880_, lean_object* v___y_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_, lean_object* v___y_3884_, lean_object* v___y_3885_, lean_object* v___y_3886_, lean_object* v___y_3887_){
_start:
{
size_t v_sz_boxed_3888_; size_t v_i_boxed_3889_; lean_object* v_res_3890_; 
v_sz_boxed_3888_ = lean_unbox_usize(v_sz_3876_);
lean_dec(v_sz_3876_);
v_i_boxed_3889_ = lean_unbox_usize(v_i_3877_);
lean_dec(v_i_3877_);
v_res_3890_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7_spec__9(v___x_3874_, v_as_3875_, v_sz_boxed_3888_, v_i_boxed_3889_, v_b_3878_, v___y_3879_, v___y_3880_, v___y_3881_, v___y_3882_, v___y_3883_, v___y_3884_, v___y_3885_, v___y_3886_);
lean_dec(v___y_3886_);
lean_dec_ref(v___y_3885_);
lean_dec(v___y_3884_);
lean_dec_ref(v___y_3883_);
lean_dec(v___y_3882_);
lean_dec_ref(v___y_3881_);
lean_dec(v___y_3880_);
lean_dec_ref(v___y_3879_);
lean_dec_ref(v_as_3875_);
return v_res_3890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7(lean_object* v___x_3891_, lean_object* v_as_3892_, size_t v_sz_3893_, size_t v_i_3894_, lean_object* v_b_3895_, lean_object* v___y_3896_, lean_object* v___y_3897_, lean_object* v___y_3898_, lean_object* v___y_3899_, lean_object* v___y_3900_, lean_object* v___y_3901_, lean_object* v___y_3902_, lean_object* v___y_3903_){
_start:
{
uint8_t v___x_3905_; 
v___x_3905_ = lean_usize_dec_lt(v_i_3894_, v_sz_3893_);
if (v___x_3905_ == 0)
{
lean_object* v___x_3906_; 
lean_dec(v___x_3891_);
v___x_3906_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3906_, 0, v_b_3895_);
return v___x_3906_;
}
else
{
lean_object* v_snd_3907_; lean_object* v___x_3909_; uint8_t v_isShared_3910_; uint8_t v_isSharedCheck_3992_; 
v_snd_3907_ = lean_ctor_get(v_b_3895_, 1);
v_isSharedCheck_3992_ = !lean_is_exclusive(v_b_3895_);
if (v_isSharedCheck_3992_ == 0)
{
lean_object* v_unused_3993_; 
v_unused_3993_ = lean_ctor_get(v_b_3895_, 0);
lean_dec(v_unused_3993_);
v___x_3909_ = v_b_3895_;
v_isShared_3910_ = v_isSharedCheck_3992_;
goto v_resetjp_3908_;
}
else
{
lean_inc(v_snd_3907_);
lean_dec(v_b_3895_);
v___x_3909_ = lean_box(0);
v_isShared_3910_ = v_isSharedCheck_3992_;
goto v_resetjp_3908_;
}
v_resetjp_3908_:
{
lean_object* v___x_3911_; lean_object* v_a_3913_; lean_object* v_fst_3921_; lean_object* v_snd_3922_; lean_object* v_a_3924_; 
v___x_3911_ = lean_box(0);
v_a_3924_ = lean_array_uget_borrowed(v_as_3892_, v_i_3894_);
if (lean_obj_tag(v_a_3924_) == 0)
{
v_a_3913_ = v_snd_3907_;
goto v___jp_3912_;
}
else
{
lean_object* v_val_3925_; lean_object* v___x_3926_; 
v_val_3925_ = lean_ctor_get(v_a_3924_, 0);
v___x_3926_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3897_, v___y_3899_, v___y_3901_, v___y_3903_);
if (lean_obj_tag(v___x_3926_) == 0)
{
lean_object* v_a_3927_; lean_object* v___x_3929_; uint8_t v_isShared_3930_; uint8_t v_isSharedCheck_3983_; 
v_a_3927_ = lean_ctor_get(v___x_3926_, 0);
v_isSharedCheck_3983_ = !lean_is_exclusive(v___x_3926_);
if (v_isSharedCheck_3983_ == 0)
{
v___x_3929_ = v___x_3926_;
v_isShared_3930_ = v_isSharedCheck_3983_;
goto v_resetjp_3928_;
}
else
{
lean_inc(v_a_3927_);
lean_dec(v___x_3926_);
v___x_3929_ = lean_box(0);
v_isShared_3930_ = v_isSharedCheck_3983_;
goto v_resetjp_3928_;
}
v_resetjp_3928_:
{
lean_object* v_fst_3931_; lean_object* v_snd_3932_; lean_object* v___y_3934_; uint8_t v___y_3935_; lean_object* v_a_3949_; lean_object* v___x_3952_; lean_object* v___x_3953_; 
v_fst_3931_ = lean_ctor_get(v_snd_3907_, 0);
lean_inc(v_fst_3931_);
v_snd_3932_ = lean_ctor_get(v_snd_3907_, 1);
lean_inc(v_snd_3932_);
lean_dec(v_snd_3907_);
v___x_3952_ = l_Lean_LocalDecl_type(v_val_3925_);
v___x_3953_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v___x_3952_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
if (lean_obj_tag(v___x_3953_) == 0)
{
lean_object* v_a_3954_; lean_object* v_snd_3955_; lean_object* v_fst_3956_; lean_object* v_fst_3957_; uint8_t v___x_3958_; lean_object* v___x_3959_; uint8_t v___x_3960_; lean_object* v___x_3961_; lean_object* v___f_3962_; lean_object* v___x_3963_; 
v_a_3954_ = lean_ctor_get(v___x_3953_, 0);
lean_inc(v_a_3954_);
lean_dec_ref_known(v___x_3953_, 1);
v_snd_3955_ = lean_ctor_get(v_a_3954_, 1);
lean_inc(v_snd_3955_);
v_fst_3956_ = lean_ctor_get(v_a_3954_, 0);
lean_inc(v_fst_3956_);
lean_dec(v_a_3954_);
v_fst_3957_ = lean_ctor_get(v_snd_3955_, 0);
lean_inc(v_fst_3957_);
lean_dec(v_snd_3955_);
v___x_3958_ = 0;
lean_inc(v___x_3891_);
v___x_3959_ = l_Lean_Expr_fvar___override(v___x_3891_);
v___x_3960_ = 2;
v___x_3961_ = lean_box(v___x_3960_);
lean_inc_ref(v___x_3959_);
v___f_3962_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3962_, 0, v___x_3961_);
lean_closure_set(v___f_3962_, 1, v___x_3959_);
lean_closure_set(v___f_3962_, 2, v_fst_3956_);
v___x_3963_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3962_, v___x_3958_, v___y_3896_, v___y_3897_, v___y_3898_, v___y_3899_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
if (lean_obj_tag(v___x_3963_) == 0)
{
lean_object* v_a_3964_; uint8_t v___x_3965_; 
v_a_3964_ = lean_ctor_get(v___x_3963_, 0);
lean_inc(v_a_3964_);
lean_dec_ref_known(v___x_3963_, 1);
v___x_3965_ = lean_unbox(v_a_3964_);
lean_dec(v_a_3964_);
if (v___x_3965_ == 0)
{
lean_object* v___x_3966_; lean_object* v___f_3967_; lean_object* v___x_3968_; 
v___x_3966_ = lean_box(v___x_3960_);
v___f_3967_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3967_, 0, v___x_3966_);
lean_closure_set(v___f_3967_, 1, v___x_3959_);
lean_closure_set(v___f_3967_, 2, v_fst_3957_);
v___x_3968_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__3___redArg(v___f_3967_, v___x_3958_, v___y_3896_, v___y_3897_, v___y_3898_, v___y_3899_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
if (lean_obj_tag(v___x_3968_) == 0)
{
lean_object* v_a_3969_; uint8_t v___x_3970_; 
v_a_3969_ = lean_ctor_get(v___x_3968_, 0);
lean_inc(v_a_3969_);
lean_dec_ref_known(v___x_3968_, 1);
v___x_3970_ = lean_unbox(v_a_3969_);
lean_dec(v_a_3969_);
if (v___x_3970_ == 0)
{
lean_object* v___x_3971_; lean_object* v___x_3972_; 
v___x_3971_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1);
v___x_3972_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v___x_3971_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
if (lean_obj_tag(v___x_3972_) == 0)
{
lean_dec_ref_known(v___x_3972_, 1);
lean_del_object(v___x_3929_);
lean_dec(v_a_3927_);
v_fst_3921_ = v_fst_3931_;
v_snd_3922_ = v_snd_3932_;
goto v___jp_3920_;
}
else
{
lean_object* v_a_3973_; 
v_a_3973_ = lean_ctor_get(v___x_3972_, 0);
lean_inc(v_a_3973_);
lean_dec_ref_known(v___x_3972_, 1);
v_a_3949_ = v_a_3973_;
goto v___jp_3948_;
}
}
else
{
lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; 
lean_del_object(v___x_3929_);
lean_dec(v_a_3927_);
v___x_3974_ = l_Lean_LocalDecl_fvarId(v_val_3925_);
v___x_3975_ = l_Lean_Expr_fvar___override(v___x_3974_);
v___x_3976_ = lean_array_push(v_fst_3931_, v___x_3975_);
v_fst_3921_ = v___x_3976_;
v_snd_3922_ = v_snd_3932_;
goto v___jp_3920_;
}
}
else
{
lean_object* v_a_3977_; 
v_a_3977_ = lean_ctor_get(v___x_3968_, 0);
lean_inc(v_a_3977_);
lean_dec_ref_known(v___x_3968_, 1);
v_a_3949_ = v_a_3977_;
goto v___jp_3948_;
}
}
else
{
lean_object* v___x_3978_; lean_object* v___x_3979_; lean_object* v___x_3980_; 
lean_dec_ref(v___x_3959_);
lean_dec(v_fst_3957_);
lean_del_object(v___x_3929_);
lean_dec(v_a_3927_);
v___x_3978_ = l_Lean_LocalDecl_fvarId(v_val_3925_);
v___x_3979_ = l_Lean_Expr_fvar___override(v___x_3978_);
v___x_3980_ = lean_array_push(v_snd_3932_, v___x_3979_);
v_fst_3921_ = v_fst_3931_;
v_snd_3922_ = v___x_3980_;
goto v___jp_3920_;
}
}
else
{
lean_object* v_a_3981_; 
lean_dec_ref(v___x_3959_);
lean_dec(v_fst_3957_);
v_a_3981_ = lean_ctor_get(v___x_3963_, 0);
lean_inc(v_a_3981_);
lean_dec_ref_known(v___x_3963_, 1);
v_a_3949_ = v_a_3981_;
goto v___jp_3948_;
}
}
else
{
lean_object* v_a_3982_; 
v_a_3982_ = lean_ctor_get(v___x_3953_, 0);
lean_inc(v_a_3982_);
lean_dec_ref_known(v___x_3953_, 1);
v_a_3949_ = v_a_3982_;
goto v___jp_3948_;
}
v___jp_3933_:
{
if (v___y_3935_ == 0)
{
lean_object* v___x_3936_; 
lean_dec_ref(v___y_3934_);
lean_del_object(v___x_3929_);
v___x_3936_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3927_, v___y_3935_, v___y_3897_, v___y_3898_, v___y_3899_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
if (lean_obj_tag(v___x_3936_) == 0)
{
lean_dec_ref_known(v___x_3936_, 1);
v_fst_3921_ = v_fst_3931_;
v_snd_3922_ = v_snd_3932_;
goto v___jp_3920_;
}
else
{
lean_object* v_a_3937_; lean_object* v___x_3939_; uint8_t v_isShared_3940_; uint8_t v_isSharedCheck_3944_; 
lean_dec(v_snd_3932_);
lean_dec(v_fst_3931_);
lean_del_object(v___x_3909_);
lean_dec(v___x_3891_);
v_a_3937_ = lean_ctor_get(v___x_3936_, 0);
v_isSharedCheck_3944_ = !lean_is_exclusive(v___x_3936_);
if (v_isSharedCheck_3944_ == 0)
{
v___x_3939_ = v___x_3936_;
v_isShared_3940_ = v_isSharedCheck_3944_;
goto v_resetjp_3938_;
}
else
{
lean_inc(v_a_3937_);
lean_dec(v___x_3936_);
v___x_3939_ = lean_box(0);
v_isShared_3940_ = v_isSharedCheck_3944_;
goto v_resetjp_3938_;
}
v_resetjp_3938_:
{
lean_object* v___x_3942_; 
if (v_isShared_3940_ == 0)
{
v___x_3942_ = v___x_3939_;
goto v_reusejp_3941_;
}
else
{
lean_object* v_reuseFailAlloc_3943_; 
v_reuseFailAlloc_3943_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3943_, 0, v_a_3937_);
v___x_3942_ = v_reuseFailAlloc_3943_;
goto v_reusejp_3941_;
}
v_reusejp_3941_:
{
return v___x_3942_;
}
}
}
}
else
{
lean_object* v___x_3946_; 
lean_dec(v_snd_3932_);
lean_dec(v_fst_3931_);
lean_dec(v_a_3927_);
lean_del_object(v___x_3909_);
lean_dec(v___x_3891_);
if (v_isShared_3930_ == 0)
{
lean_ctor_set_tag(v___x_3929_, 1);
lean_ctor_set(v___x_3929_, 0, v___y_3934_);
v___x_3946_ = v___x_3929_;
goto v_reusejp_3945_;
}
else
{
lean_object* v_reuseFailAlloc_3947_; 
v_reuseFailAlloc_3947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3947_, 0, v___y_3934_);
v___x_3946_ = v_reuseFailAlloc_3947_;
goto v_reusejp_3945_;
}
v_reusejp_3945_:
{
return v___x_3946_;
}
}
}
v___jp_3948_:
{
uint8_t v___x_3950_; 
v___x_3950_ = l_Lean_Exception_isInterrupt(v_a_3949_);
if (v___x_3950_ == 0)
{
uint8_t v___x_3951_; 
lean_inc_ref(v_a_3949_);
v___x_3951_ = l_Lean_Exception_isRuntime(v_a_3949_);
v___y_3934_ = v_a_3949_;
v___y_3935_ = v___x_3951_;
goto v___jp_3933_;
}
else
{
v___y_3934_ = v_a_3949_;
v___y_3935_ = v___x_3950_;
goto v___jp_3933_;
}
}
}
}
else
{
lean_object* v_a_3984_; lean_object* v___x_3986_; uint8_t v_isShared_3987_; uint8_t v_isSharedCheck_3991_; 
lean_del_object(v___x_3909_);
lean_dec(v_snd_3907_);
lean_dec(v___x_3891_);
v_a_3984_ = lean_ctor_get(v___x_3926_, 0);
v_isSharedCheck_3991_ = !lean_is_exclusive(v___x_3926_);
if (v_isSharedCheck_3991_ == 0)
{
v___x_3986_ = v___x_3926_;
v_isShared_3987_ = v_isSharedCheck_3991_;
goto v_resetjp_3985_;
}
else
{
lean_inc(v_a_3984_);
lean_dec(v___x_3926_);
v___x_3986_ = lean_box(0);
v_isShared_3987_ = v_isSharedCheck_3991_;
goto v_resetjp_3985_;
}
v_resetjp_3985_:
{
lean_object* v___x_3989_; 
if (v_isShared_3987_ == 0)
{
v___x_3989_ = v___x_3986_;
goto v_reusejp_3988_;
}
else
{
lean_object* v_reuseFailAlloc_3990_; 
v_reuseFailAlloc_3990_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3990_, 0, v_a_3984_);
v___x_3989_ = v_reuseFailAlloc_3990_;
goto v_reusejp_3988_;
}
v_reusejp_3988_:
{
return v___x_3989_;
}
}
}
}
v___jp_3912_:
{
lean_object* v___x_3915_; 
if (v_isShared_3910_ == 0)
{
lean_ctor_set(v___x_3909_, 1, v_a_3913_);
lean_ctor_set(v___x_3909_, 0, v___x_3911_);
v___x_3915_ = v___x_3909_;
goto v_reusejp_3914_;
}
else
{
lean_object* v_reuseFailAlloc_3919_; 
v_reuseFailAlloc_3919_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3919_, 0, v___x_3911_);
lean_ctor_set(v_reuseFailAlloc_3919_, 1, v_a_3913_);
v___x_3915_ = v_reuseFailAlloc_3919_;
goto v_reusejp_3914_;
}
v_reusejp_3914_:
{
size_t v___x_3916_; size_t v___x_3917_; lean_object* v___x_3918_; 
v___x_3916_ = ((size_t)1ULL);
v___x_3917_ = lean_usize_add(v_i_3894_, v___x_3916_);
v___x_3918_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7_spec__9(v___x_3891_, v_as_3892_, v_sz_3893_, v___x_3917_, v___x_3915_, v___y_3896_, v___y_3897_, v___y_3898_, v___y_3899_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_);
return v___x_3918_;
}
}
v___jp_3920_:
{
lean_object* v___x_3923_; 
v___x_3923_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3923_, 0, v_fst_3921_);
lean_ctor_set(v___x_3923_, 1, v_snd_3922_);
v_a_3913_ = v___x_3923_;
goto v___jp_3912_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7___boxed(lean_object* v___x_3994_, lean_object* v_as_3995_, lean_object* v_sz_3996_, lean_object* v_i_3997_, lean_object* v_b_3998_, lean_object* v___y_3999_, lean_object* v___y_4000_, lean_object* v___y_4001_, lean_object* v___y_4002_, lean_object* v___y_4003_, lean_object* v___y_4004_, lean_object* v___y_4005_, lean_object* v___y_4006_, lean_object* v___y_4007_){
_start:
{
size_t v_sz_boxed_4008_; size_t v_i_boxed_4009_; lean_object* v_res_4010_; 
v_sz_boxed_4008_ = lean_unbox_usize(v_sz_3996_);
lean_dec(v_sz_3996_);
v_i_boxed_4009_ = lean_unbox_usize(v_i_3997_);
lean_dec(v_i_3997_);
v_res_4010_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7(v___x_3994_, v_as_3995_, v_sz_boxed_4008_, v_i_boxed_4009_, v_b_3998_, v___y_3999_, v___y_4000_, v___y_4001_, v___y_4002_, v___y_4003_, v___y_4004_, v___y_4005_, v___y_4006_);
lean_dec(v___y_4006_);
lean_dec_ref(v___y_4005_);
lean_dec(v___y_4004_);
lean_dec_ref(v___y_4003_);
lean_dec(v___y_4002_);
lean_dec_ref(v___y_4001_);
lean_dec(v___y_4000_);
lean_dec_ref(v___y_3999_);
lean_dec_ref(v_as_3995_);
return v_res_4010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5(lean_object* v_init_4011_, lean_object* v___x_4012_, lean_object* v_n_4013_, lean_object* v_b_4014_, lean_object* v___y_4015_, lean_object* v___y_4016_, lean_object* v___y_4017_, lean_object* v___y_4018_, lean_object* v___y_4019_, lean_object* v___y_4020_, lean_object* v___y_4021_, lean_object* v___y_4022_){
_start:
{
if (lean_obj_tag(v_n_4013_) == 0)
{
lean_object* v_cs_4024_; lean_object* v___x_4025_; lean_object* v___x_4026_; size_t v_sz_4027_; size_t v___x_4028_; lean_object* v___x_4029_; 
v_cs_4024_ = lean_ctor_get(v_n_4013_, 0);
v___x_4025_ = lean_box(0);
v___x_4026_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4026_, 0, v___x_4025_);
lean_ctor_set(v___x_4026_, 1, v_b_4014_);
v_sz_4027_ = lean_array_size(v_cs_4024_);
v___x_4028_ = ((size_t)0ULL);
v___x_4029_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__6(v_init_4011_, v___x_4012_, v_cs_4024_, v_sz_4027_, v___x_4028_, v___x_4026_, v___y_4015_, v___y_4016_, v___y_4017_, v___y_4018_, v___y_4019_, v___y_4020_, v___y_4021_, v___y_4022_);
if (lean_obj_tag(v___x_4029_) == 0)
{
lean_object* v_a_4030_; lean_object* v___x_4032_; uint8_t v_isShared_4033_; uint8_t v_isSharedCheck_4044_; 
v_a_4030_ = lean_ctor_get(v___x_4029_, 0);
v_isSharedCheck_4044_ = !lean_is_exclusive(v___x_4029_);
if (v_isSharedCheck_4044_ == 0)
{
v___x_4032_ = v___x_4029_;
v_isShared_4033_ = v_isSharedCheck_4044_;
goto v_resetjp_4031_;
}
else
{
lean_inc(v_a_4030_);
lean_dec(v___x_4029_);
v___x_4032_ = lean_box(0);
v_isShared_4033_ = v_isSharedCheck_4044_;
goto v_resetjp_4031_;
}
v_resetjp_4031_:
{
lean_object* v_fst_4034_; 
v_fst_4034_ = lean_ctor_get(v_a_4030_, 0);
if (lean_obj_tag(v_fst_4034_) == 0)
{
lean_object* v_snd_4035_; lean_object* v___x_4036_; lean_object* v___x_4038_; 
v_snd_4035_ = lean_ctor_get(v_a_4030_, 1);
lean_inc(v_snd_4035_);
lean_dec(v_a_4030_);
v___x_4036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4036_, 0, v_snd_4035_);
if (v_isShared_4033_ == 0)
{
lean_ctor_set(v___x_4032_, 0, v___x_4036_);
v___x_4038_ = v___x_4032_;
goto v_reusejp_4037_;
}
else
{
lean_object* v_reuseFailAlloc_4039_; 
v_reuseFailAlloc_4039_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4039_, 0, v___x_4036_);
v___x_4038_ = v_reuseFailAlloc_4039_;
goto v_reusejp_4037_;
}
v_reusejp_4037_:
{
return v___x_4038_;
}
}
else
{
lean_object* v_val_4040_; lean_object* v___x_4042_; 
lean_inc_ref(v_fst_4034_);
lean_dec(v_a_4030_);
v_val_4040_ = lean_ctor_get(v_fst_4034_, 0);
lean_inc(v_val_4040_);
lean_dec_ref_known(v_fst_4034_, 1);
if (v_isShared_4033_ == 0)
{
lean_ctor_set(v___x_4032_, 0, v_val_4040_);
v___x_4042_ = v___x_4032_;
goto v_reusejp_4041_;
}
else
{
lean_object* v_reuseFailAlloc_4043_; 
v_reuseFailAlloc_4043_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4043_, 0, v_val_4040_);
v___x_4042_ = v_reuseFailAlloc_4043_;
goto v_reusejp_4041_;
}
v_reusejp_4041_:
{
return v___x_4042_;
}
}
}
}
else
{
lean_object* v_a_4045_; lean_object* v___x_4047_; uint8_t v_isShared_4048_; uint8_t v_isSharedCheck_4052_; 
v_a_4045_ = lean_ctor_get(v___x_4029_, 0);
v_isSharedCheck_4052_ = !lean_is_exclusive(v___x_4029_);
if (v_isSharedCheck_4052_ == 0)
{
v___x_4047_ = v___x_4029_;
v_isShared_4048_ = v_isSharedCheck_4052_;
goto v_resetjp_4046_;
}
else
{
lean_inc(v_a_4045_);
lean_dec(v___x_4029_);
v___x_4047_ = lean_box(0);
v_isShared_4048_ = v_isSharedCheck_4052_;
goto v_resetjp_4046_;
}
v_resetjp_4046_:
{
lean_object* v___x_4050_; 
if (v_isShared_4048_ == 0)
{
v___x_4050_ = v___x_4047_;
goto v_reusejp_4049_;
}
else
{
lean_object* v_reuseFailAlloc_4051_; 
v_reuseFailAlloc_4051_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4051_, 0, v_a_4045_);
v___x_4050_ = v_reuseFailAlloc_4051_;
goto v_reusejp_4049_;
}
v_reusejp_4049_:
{
return v___x_4050_;
}
}
}
}
else
{
lean_object* v_vs_4053_; lean_object* v___x_4054_; lean_object* v___x_4055_; size_t v_sz_4056_; size_t v___x_4057_; lean_object* v___x_4058_; 
v_vs_4053_ = lean_ctor_get(v_n_4013_, 0);
v___x_4054_ = lean_box(0);
v___x_4055_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4055_, 0, v___x_4054_);
lean_ctor_set(v___x_4055_, 1, v_b_4014_);
v_sz_4056_ = lean_array_size(v_vs_4053_);
v___x_4057_ = ((size_t)0ULL);
v___x_4058_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__7(v___x_4012_, v_vs_4053_, v_sz_4056_, v___x_4057_, v___x_4055_, v___y_4015_, v___y_4016_, v___y_4017_, v___y_4018_, v___y_4019_, v___y_4020_, v___y_4021_, v___y_4022_);
if (lean_obj_tag(v___x_4058_) == 0)
{
lean_object* v_a_4059_; lean_object* v___x_4061_; uint8_t v_isShared_4062_; uint8_t v_isSharedCheck_4073_; 
v_a_4059_ = lean_ctor_get(v___x_4058_, 0);
v_isSharedCheck_4073_ = !lean_is_exclusive(v___x_4058_);
if (v_isSharedCheck_4073_ == 0)
{
v___x_4061_ = v___x_4058_;
v_isShared_4062_ = v_isSharedCheck_4073_;
goto v_resetjp_4060_;
}
else
{
lean_inc(v_a_4059_);
lean_dec(v___x_4058_);
v___x_4061_ = lean_box(0);
v_isShared_4062_ = v_isSharedCheck_4073_;
goto v_resetjp_4060_;
}
v_resetjp_4060_:
{
lean_object* v_fst_4063_; 
v_fst_4063_ = lean_ctor_get(v_a_4059_, 0);
if (lean_obj_tag(v_fst_4063_) == 0)
{
lean_object* v_snd_4064_; lean_object* v___x_4065_; lean_object* v___x_4067_; 
v_snd_4064_ = lean_ctor_get(v_a_4059_, 1);
lean_inc(v_snd_4064_);
lean_dec(v_a_4059_);
v___x_4065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4065_, 0, v_snd_4064_);
if (v_isShared_4062_ == 0)
{
lean_ctor_set(v___x_4061_, 0, v___x_4065_);
v___x_4067_ = v___x_4061_;
goto v_reusejp_4066_;
}
else
{
lean_object* v_reuseFailAlloc_4068_; 
v_reuseFailAlloc_4068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4068_, 0, v___x_4065_);
v___x_4067_ = v_reuseFailAlloc_4068_;
goto v_reusejp_4066_;
}
v_reusejp_4066_:
{
return v___x_4067_;
}
}
else
{
lean_object* v_val_4069_; lean_object* v___x_4071_; 
lean_inc_ref(v_fst_4063_);
lean_dec(v_a_4059_);
v_val_4069_ = lean_ctor_get(v_fst_4063_, 0);
lean_inc(v_val_4069_);
lean_dec_ref_known(v_fst_4063_, 1);
if (v_isShared_4062_ == 0)
{
lean_ctor_set(v___x_4061_, 0, v_val_4069_);
v___x_4071_ = v___x_4061_;
goto v_reusejp_4070_;
}
else
{
lean_object* v_reuseFailAlloc_4072_; 
v_reuseFailAlloc_4072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4072_, 0, v_val_4069_);
v___x_4071_ = v_reuseFailAlloc_4072_;
goto v_reusejp_4070_;
}
v_reusejp_4070_:
{
return v___x_4071_;
}
}
}
}
else
{
lean_object* v_a_4074_; lean_object* v___x_4076_; uint8_t v_isShared_4077_; uint8_t v_isSharedCheck_4081_; 
v_a_4074_ = lean_ctor_get(v___x_4058_, 0);
v_isSharedCheck_4081_ = !lean_is_exclusive(v___x_4058_);
if (v_isSharedCheck_4081_ == 0)
{
v___x_4076_ = v___x_4058_;
v_isShared_4077_ = v_isSharedCheck_4081_;
goto v_resetjp_4075_;
}
else
{
lean_inc(v_a_4074_);
lean_dec(v___x_4058_);
v___x_4076_ = lean_box(0);
v_isShared_4077_ = v_isSharedCheck_4081_;
goto v_resetjp_4075_;
}
v_resetjp_4075_:
{
lean_object* v___x_4079_; 
if (v_isShared_4077_ == 0)
{
v___x_4079_ = v___x_4076_;
goto v_reusejp_4078_;
}
else
{
lean_object* v_reuseFailAlloc_4080_; 
v_reuseFailAlloc_4080_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4080_, 0, v_a_4074_);
v___x_4079_ = v_reuseFailAlloc_4080_;
goto v_reusejp_4078_;
}
v_reusejp_4078_:
{
return v___x_4079_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__6(lean_object* v_init_4082_, lean_object* v___x_4083_, lean_object* v_as_4084_, size_t v_sz_4085_, size_t v_i_4086_, lean_object* v_b_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_, lean_object* v___y_4090_, lean_object* v___y_4091_, lean_object* v___y_4092_, lean_object* v___y_4093_, lean_object* v___y_4094_, lean_object* v___y_4095_){
_start:
{
uint8_t v___x_4097_; 
v___x_4097_ = lean_usize_dec_lt(v_i_4086_, v_sz_4085_);
if (v___x_4097_ == 0)
{
lean_object* v___x_4098_; 
lean_dec(v___x_4083_);
v___x_4098_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4098_, 0, v_b_4087_);
return v___x_4098_;
}
else
{
lean_object* v_snd_4099_; lean_object* v___x_4101_; uint8_t v_isShared_4102_; uint8_t v_isSharedCheck_4133_; 
v_snd_4099_ = lean_ctor_get(v_b_4087_, 1);
v_isSharedCheck_4133_ = !lean_is_exclusive(v_b_4087_);
if (v_isSharedCheck_4133_ == 0)
{
lean_object* v_unused_4134_; 
v_unused_4134_ = lean_ctor_get(v_b_4087_, 0);
lean_dec(v_unused_4134_);
v___x_4101_ = v_b_4087_;
v_isShared_4102_ = v_isSharedCheck_4133_;
goto v_resetjp_4100_;
}
else
{
lean_inc(v_snd_4099_);
lean_dec(v_b_4087_);
v___x_4101_ = lean_box(0);
v_isShared_4102_ = v_isSharedCheck_4133_;
goto v_resetjp_4100_;
}
v_resetjp_4100_:
{
lean_object* v_a_4103_; lean_object* v___x_4104_; 
v_a_4103_ = lean_array_uget_borrowed(v_as_4084_, v_i_4086_);
lean_inc(v_snd_4099_);
lean_inc(v___x_4083_);
v___x_4104_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5(v_init_4082_, v___x_4083_, v_a_4103_, v_snd_4099_, v___y_4088_, v___y_4089_, v___y_4090_, v___y_4091_, v___y_4092_, v___y_4093_, v___y_4094_, v___y_4095_);
if (lean_obj_tag(v___x_4104_) == 0)
{
lean_object* v_a_4105_; lean_object* v___x_4107_; uint8_t v_isShared_4108_; uint8_t v_isSharedCheck_4124_; 
v_a_4105_ = lean_ctor_get(v___x_4104_, 0);
v_isSharedCheck_4124_ = !lean_is_exclusive(v___x_4104_);
if (v_isSharedCheck_4124_ == 0)
{
v___x_4107_ = v___x_4104_;
v_isShared_4108_ = v_isSharedCheck_4124_;
goto v_resetjp_4106_;
}
else
{
lean_inc(v_a_4105_);
lean_dec(v___x_4104_);
v___x_4107_ = lean_box(0);
v_isShared_4108_ = v_isSharedCheck_4124_;
goto v_resetjp_4106_;
}
v_resetjp_4106_:
{
if (lean_obj_tag(v_a_4105_) == 0)
{
lean_object* v___x_4109_; lean_object* v___x_4111_; 
lean_dec(v___x_4083_);
v___x_4109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4109_, 0, v_a_4105_);
if (v_isShared_4102_ == 0)
{
lean_ctor_set(v___x_4101_, 0, v___x_4109_);
v___x_4111_ = v___x_4101_;
goto v_reusejp_4110_;
}
else
{
lean_object* v_reuseFailAlloc_4115_; 
v_reuseFailAlloc_4115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4115_, 0, v___x_4109_);
lean_ctor_set(v_reuseFailAlloc_4115_, 1, v_snd_4099_);
v___x_4111_ = v_reuseFailAlloc_4115_;
goto v_reusejp_4110_;
}
v_reusejp_4110_:
{
lean_object* v___x_4113_; 
if (v_isShared_4108_ == 0)
{
lean_ctor_set(v___x_4107_, 0, v___x_4111_);
v___x_4113_ = v___x_4107_;
goto v_reusejp_4112_;
}
else
{
lean_object* v_reuseFailAlloc_4114_; 
v_reuseFailAlloc_4114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4114_, 0, v___x_4111_);
v___x_4113_ = v_reuseFailAlloc_4114_;
goto v_reusejp_4112_;
}
v_reusejp_4112_:
{
return v___x_4113_;
}
}
}
else
{
lean_object* v_a_4116_; lean_object* v___x_4117_; lean_object* v___x_4119_; 
lean_del_object(v___x_4107_);
lean_dec(v_snd_4099_);
v_a_4116_ = lean_ctor_get(v_a_4105_, 0);
lean_inc(v_a_4116_);
lean_dec_ref_known(v_a_4105_, 1);
v___x_4117_ = lean_box(0);
if (v_isShared_4102_ == 0)
{
lean_ctor_set(v___x_4101_, 1, v_a_4116_);
lean_ctor_set(v___x_4101_, 0, v___x_4117_);
v___x_4119_ = v___x_4101_;
goto v_reusejp_4118_;
}
else
{
lean_object* v_reuseFailAlloc_4123_; 
v_reuseFailAlloc_4123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4123_, 0, v___x_4117_);
lean_ctor_set(v_reuseFailAlloc_4123_, 1, v_a_4116_);
v___x_4119_ = v_reuseFailAlloc_4123_;
goto v_reusejp_4118_;
}
v_reusejp_4118_:
{
size_t v___x_4120_; size_t v___x_4121_; 
v___x_4120_ = ((size_t)1ULL);
v___x_4121_ = lean_usize_add(v_i_4086_, v___x_4120_);
v_i_4086_ = v___x_4121_;
v_b_4087_ = v___x_4119_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_4125_; lean_object* v___x_4127_; uint8_t v_isShared_4128_; uint8_t v_isSharedCheck_4132_; 
lean_del_object(v___x_4101_);
lean_dec(v_snd_4099_);
lean_dec(v___x_4083_);
v_a_4125_ = lean_ctor_get(v___x_4104_, 0);
v_isSharedCheck_4132_ = !lean_is_exclusive(v___x_4104_);
if (v_isSharedCheck_4132_ == 0)
{
v___x_4127_ = v___x_4104_;
v_isShared_4128_ = v_isSharedCheck_4132_;
goto v_resetjp_4126_;
}
else
{
lean_inc(v_a_4125_);
lean_dec(v___x_4104_);
v___x_4127_ = lean_box(0);
v_isShared_4128_ = v_isSharedCheck_4132_;
goto v_resetjp_4126_;
}
v_resetjp_4126_:
{
lean_object* v___x_4130_; 
if (v_isShared_4128_ == 0)
{
v___x_4130_ = v___x_4127_;
goto v_reusejp_4129_;
}
else
{
lean_object* v_reuseFailAlloc_4131_; 
v_reuseFailAlloc_4131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4131_, 0, v_a_4125_);
v___x_4130_ = v_reuseFailAlloc_4131_;
goto v_reusejp_4129_;
}
v_reusejp_4129_:
{
return v___x_4130_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__6___boxed(lean_object* v_init_4135_, lean_object* v___x_4136_, lean_object* v_as_4137_, lean_object* v_sz_4138_, lean_object* v_i_4139_, lean_object* v_b_4140_, lean_object* v___y_4141_, lean_object* v___y_4142_, lean_object* v___y_4143_, lean_object* v___y_4144_, lean_object* v___y_4145_, lean_object* v___y_4146_, lean_object* v___y_4147_, lean_object* v___y_4148_, lean_object* v___y_4149_){
_start:
{
size_t v_sz_boxed_4150_; size_t v_i_boxed_4151_; lean_object* v_res_4152_; 
v_sz_boxed_4150_ = lean_unbox_usize(v_sz_4138_);
lean_dec(v_sz_4138_);
v_i_boxed_4151_ = lean_unbox_usize(v_i_4139_);
lean_dec(v_i_4139_);
v_res_4152_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5_spec__6(v_init_4135_, v___x_4136_, v_as_4137_, v_sz_boxed_4150_, v_i_boxed_4151_, v_b_4140_, v___y_4141_, v___y_4142_, v___y_4143_, v___y_4144_, v___y_4145_, v___y_4146_, v___y_4147_, v___y_4148_);
lean_dec(v___y_4148_);
lean_dec_ref(v___y_4147_);
lean_dec(v___y_4146_);
lean_dec_ref(v___y_4145_);
lean_dec(v___y_4144_);
lean_dec_ref(v___y_4143_);
lean_dec(v___y_4142_);
lean_dec_ref(v___y_4141_);
lean_dec_ref(v_as_4137_);
lean_dec_ref(v_init_4135_);
return v_res_4152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5___boxed(lean_object* v_init_4153_, lean_object* v___x_4154_, lean_object* v_n_4155_, lean_object* v_b_4156_, lean_object* v___y_4157_, lean_object* v___y_4158_, lean_object* v___y_4159_, lean_object* v___y_4160_, lean_object* v___y_4161_, lean_object* v___y_4162_, lean_object* v___y_4163_, lean_object* v___y_4164_, lean_object* v___y_4165_){
_start:
{
lean_object* v_res_4166_; 
v_res_4166_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5(v_init_4153_, v___x_4154_, v_n_4155_, v_b_4156_, v___y_4157_, v___y_4158_, v___y_4159_, v___y_4160_, v___y_4161_, v___y_4162_, v___y_4163_, v___y_4164_);
lean_dec(v___y_4164_);
lean_dec_ref(v___y_4163_);
lean_dec(v___y_4162_);
lean_dec_ref(v___y_4161_);
lean_dec(v___y_4160_);
lean_dec_ref(v___y_4159_);
lean_dec(v___y_4158_);
lean_dec_ref(v___y_4157_);
lean_dec_ref(v_n_4155_);
lean_dec_ref(v_init_4153_);
return v_res_4166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5(lean_object* v___x_4167_, lean_object* v_t_4168_, lean_object* v_init_4169_, lean_object* v___y_4170_, lean_object* v___y_4171_, lean_object* v___y_4172_, lean_object* v___y_4173_, lean_object* v___y_4174_, lean_object* v___y_4175_, lean_object* v___y_4176_, lean_object* v___y_4177_){
_start:
{
lean_object* v_root_4179_; lean_object* v_tail_4180_; lean_object* v___x_4181_; 
v_root_4179_ = lean_ctor_get(v_t_4168_, 0);
v_tail_4180_ = lean_ctor_get(v_t_4168_, 1);
lean_inc(v___x_4167_);
lean_inc_ref(v_init_4169_);
v___x_4181_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__5(v_init_4169_, v___x_4167_, v_root_4179_, v_init_4169_, v___y_4170_, v___y_4171_, v___y_4172_, v___y_4173_, v___y_4174_, v___y_4175_, v___y_4176_, v___y_4177_);
lean_dec_ref(v_init_4169_);
if (lean_obj_tag(v___x_4181_) == 0)
{
lean_object* v_a_4182_; lean_object* v___x_4184_; uint8_t v_isShared_4185_; uint8_t v_isSharedCheck_4218_; 
v_a_4182_ = lean_ctor_get(v___x_4181_, 0);
v_isSharedCheck_4218_ = !lean_is_exclusive(v___x_4181_);
if (v_isSharedCheck_4218_ == 0)
{
v___x_4184_ = v___x_4181_;
v_isShared_4185_ = v_isSharedCheck_4218_;
goto v_resetjp_4183_;
}
else
{
lean_inc(v_a_4182_);
lean_dec(v___x_4181_);
v___x_4184_ = lean_box(0);
v_isShared_4185_ = v_isSharedCheck_4218_;
goto v_resetjp_4183_;
}
v_resetjp_4183_:
{
if (lean_obj_tag(v_a_4182_) == 0)
{
lean_object* v_a_4186_; lean_object* v___x_4188_; 
lean_dec(v___x_4167_);
v_a_4186_ = lean_ctor_get(v_a_4182_, 0);
lean_inc(v_a_4186_);
lean_dec_ref_known(v_a_4182_, 1);
if (v_isShared_4185_ == 0)
{
lean_ctor_set(v___x_4184_, 0, v_a_4186_);
v___x_4188_ = v___x_4184_;
goto v_reusejp_4187_;
}
else
{
lean_object* v_reuseFailAlloc_4189_; 
v_reuseFailAlloc_4189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4189_, 0, v_a_4186_);
v___x_4188_ = v_reuseFailAlloc_4189_;
goto v_reusejp_4187_;
}
v_reusejp_4187_:
{
return v___x_4188_;
}
}
else
{
lean_object* v_a_4190_; lean_object* v___x_4191_; lean_object* v___x_4192_; size_t v_sz_4193_; size_t v___x_4194_; lean_object* v___x_4195_; 
lean_del_object(v___x_4184_);
v_a_4190_ = lean_ctor_get(v_a_4182_, 0);
lean_inc(v_a_4190_);
lean_dec_ref_known(v_a_4182_, 1);
v___x_4191_ = lean_box(0);
v___x_4192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4192_, 0, v___x_4191_);
lean_ctor_set(v___x_4192_, 1, v_a_4190_);
v_sz_4193_ = lean_array_size(v_tail_4180_);
v___x_4194_ = ((size_t)0ULL);
v___x_4195_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6(v___x_4167_, v_tail_4180_, v_sz_4193_, v___x_4194_, v___x_4192_, v___y_4170_, v___y_4171_, v___y_4172_, v___y_4173_, v___y_4174_, v___y_4175_, v___y_4176_, v___y_4177_);
if (lean_obj_tag(v___x_4195_) == 0)
{
lean_object* v_a_4196_; lean_object* v___x_4198_; uint8_t v_isShared_4199_; uint8_t v_isSharedCheck_4209_; 
v_a_4196_ = lean_ctor_get(v___x_4195_, 0);
v_isSharedCheck_4209_ = !lean_is_exclusive(v___x_4195_);
if (v_isSharedCheck_4209_ == 0)
{
v___x_4198_ = v___x_4195_;
v_isShared_4199_ = v_isSharedCheck_4209_;
goto v_resetjp_4197_;
}
else
{
lean_inc(v_a_4196_);
lean_dec(v___x_4195_);
v___x_4198_ = lean_box(0);
v_isShared_4199_ = v_isSharedCheck_4209_;
goto v_resetjp_4197_;
}
v_resetjp_4197_:
{
lean_object* v_fst_4200_; 
v_fst_4200_ = lean_ctor_get(v_a_4196_, 0);
if (lean_obj_tag(v_fst_4200_) == 0)
{
lean_object* v_snd_4201_; lean_object* v___x_4203_; 
v_snd_4201_ = lean_ctor_get(v_a_4196_, 1);
lean_inc(v_snd_4201_);
lean_dec(v_a_4196_);
if (v_isShared_4199_ == 0)
{
lean_ctor_set(v___x_4198_, 0, v_snd_4201_);
v___x_4203_ = v___x_4198_;
goto v_reusejp_4202_;
}
else
{
lean_object* v_reuseFailAlloc_4204_; 
v_reuseFailAlloc_4204_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4204_, 0, v_snd_4201_);
v___x_4203_ = v_reuseFailAlloc_4204_;
goto v_reusejp_4202_;
}
v_reusejp_4202_:
{
return v___x_4203_;
}
}
else
{
lean_object* v_val_4205_; lean_object* v___x_4207_; 
lean_inc_ref(v_fst_4200_);
lean_dec(v_a_4196_);
v_val_4205_ = lean_ctor_get(v_fst_4200_, 0);
lean_inc(v_val_4205_);
lean_dec_ref_known(v_fst_4200_, 1);
if (v_isShared_4199_ == 0)
{
lean_ctor_set(v___x_4198_, 0, v_val_4205_);
v___x_4207_ = v___x_4198_;
goto v_reusejp_4206_;
}
else
{
lean_object* v_reuseFailAlloc_4208_; 
v_reuseFailAlloc_4208_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4208_, 0, v_val_4205_);
v___x_4207_ = v_reuseFailAlloc_4208_;
goto v_reusejp_4206_;
}
v_reusejp_4206_:
{
return v___x_4207_;
}
}
}
}
else
{
lean_object* v_a_4210_; lean_object* v___x_4212_; uint8_t v_isShared_4213_; uint8_t v_isSharedCheck_4217_; 
v_a_4210_ = lean_ctor_get(v___x_4195_, 0);
v_isSharedCheck_4217_ = !lean_is_exclusive(v___x_4195_);
if (v_isSharedCheck_4217_ == 0)
{
v___x_4212_ = v___x_4195_;
v_isShared_4213_ = v_isSharedCheck_4217_;
goto v_resetjp_4211_;
}
else
{
lean_inc(v_a_4210_);
lean_dec(v___x_4195_);
v___x_4212_ = lean_box(0);
v_isShared_4213_ = v_isSharedCheck_4217_;
goto v_resetjp_4211_;
}
v_resetjp_4211_:
{
lean_object* v___x_4215_; 
if (v_isShared_4213_ == 0)
{
v___x_4215_ = v___x_4212_;
goto v_reusejp_4214_;
}
else
{
lean_object* v_reuseFailAlloc_4216_; 
v_reuseFailAlloc_4216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4216_, 0, v_a_4210_);
v___x_4215_ = v_reuseFailAlloc_4216_;
goto v_reusejp_4214_;
}
v_reusejp_4214_:
{
return v___x_4215_;
}
}
}
}
}
}
else
{
lean_object* v_a_4219_; lean_object* v___x_4221_; uint8_t v_isShared_4222_; uint8_t v_isSharedCheck_4226_; 
lean_dec(v___x_4167_);
v_a_4219_ = lean_ctor_get(v___x_4181_, 0);
v_isSharedCheck_4226_ = !lean_is_exclusive(v___x_4181_);
if (v_isSharedCheck_4226_ == 0)
{
v___x_4221_ = v___x_4181_;
v_isShared_4222_ = v_isSharedCheck_4226_;
goto v_resetjp_4220_;
}
else
{
lean_inc(v_a_4219_);
lean_dec(v___x_4181_);
v___x_4221_ = lean_box(0);
v_isShared_4222_ = v_isSharedCheck_4226_;
goto v_resetjp_4220_;
}
v_resetjp_4220_:
{
lean_object* v___x_4224_; 
if (v_isShared_4222_ == 0)
{
v___x_4224_ = v___x_4221_;
goto v_reusejp_4223_;
}
else
{
lean_object* v_reuseFailAlloc_4225_; 
v_reuseFailAlloc_4225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4225_, 0, v_a_4219_);
v___x_4224_ = v_reuseFailAlloc_4225_;
goto v_reusejp_4223_;
}
v_reusejp_4223_:
{
return v___x_4224_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5___boxed(lean_object* v___x_4227_, lean_object* v_t_4228_, lean_object* v_init_4229_, lean_object* v___y_4230_, lean_object* v___y_4231_, lean_object* v___y_4232_, lean_object* v___y_4233_, lean_object* v___y_4234_, lean_object* v___y_4235_, lean_object* v___y_4236_, lean_object* v___y_4237_, lean_object* v___y_4238_){
_start:
{
lean_object* v_res_4239_; 
v_res_4239_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5(v___x_4227_, v_t_4228_, v_init_4229_, v___y_4230_, v___y_4231_, v___y_4232_, v___y_4233_, v___y_4234_, v___y_4235_, v___y_4236_, v___y_4237_);
lean_dec(v___y_4237_);
lean_dec_ref(v___y_4236_);
lean_dec(v___y_4235_);
lean_dec_ref(v___y_4234_);
lean_dec(v___y_4233_);
lean_dec_ref(v___y_4232_);
lean_dec(v___y_4231_);
lean_dec_ref(v___y_4230_);
lean_dec_ref(v_t_4228_);
return v_res_4239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__1(lean_object* v___x_4240_, lean_object* v___x_4241_, lean_object* v_fst_4242_, lean_object* v___x_4243_, lean_object* v___f_4244_, lean_object* v_fst_4245_, lean_object* v_snd_4246_, lean_object* v___x_4247_, uint8_t v___x_4248_, lean_object* v___y_4249_, lean_object* v___y_4250_, lean_object* v___y_4251_, lean_object* v___y_4252_, lean_object* v___y_4253_, lean_object* v___y_4254_, lean_object* v___y_4255_, lean_object* v___y_4256_){
_start:
{
lean_object* v_lctx_4258_; lean_object* v___x_4259_; lean_object* v_decls_4260_; lean_object* v___x_4261_; 
v_lctx_4258_ = lean_ctor_get(v___y_4253_, 2);
lean_inc_ref(v___x_4240_);
v___x_4259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4259_, 0, v___x_4240_);
lean_ctor_set(v___x_4259_, 1, v___x_4240_);
v_decls_4260_ = lean_ctor_get(v_lctx_4258_, 1);
lean_inc(v___x_4241_);
v___x_4261_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5(v___x_4241_, v_decls_4260_, v___x_4259_, v___y_4249_, v___y_4250_, v___y_4251_, v___y_4252_, v___y_4253_, v___y_4254_, v___y_4255_, v___y_4256_);
if (lean_obj_tag(v___x_4261_) == 0)
{
lean_object* v_a_4262_; lean_object* v_fst_4263_; lean_object* v_snd_4264_; lean_object* v___x_4265_; uint8_t v___x_4266_; 
v_a_4262_ = lean_ctor_get(v___x_4261_, 0);
lean_inc(v_a_4262_);
lean_dec_ref_known(v___x_4261_, 1);
v_fst_4263_ = lean_ctor_get(v_a_4262_, 0);
lean_inc(v_fst_4263_);
v_snd_4264_ = lean_ctor_get(v_a_4262_, 1);
lean_inc(v_snd_4264_);
lean_dec(v_a_4262_);
v___x_4265_ = lean_array_get_size(v_fst_4242_);
v___x_4266_ = lean_nat_dec_lt(v___x_4243_, v___x_4265_);
if (v___x_4266_ == 0)
{
lean_object* v___x_4267_; lean_object* v___x_4268_; lean_object* v___x_4269_; 
v___x_4267_ = lean_box(0);
v___x_4268_ = lean_box(v___x_4248_);
{
lean_object* _aargs[] = {v___x_4241_, v___x_4267_, v_fst_4245_, v_snd_4246_, v___x_4247_, v_fst_4263_, v_snd_4264_, v___x_4268_, v___y_4249_, v___y_4250_, v___y_4251_, v___y_4252_, v___y_4253_, v___y_4254_, v___y_4255_, v___y_4256_, lean_box(0)};
v___x_4269_ = lean_apply_m(v___f_4244_, 17, _aargs);
}
return v___x_4269_;
}
else
{
lean_object* v___x_4270_; lean_object* v___x_4271_; lean_object* v___x_4272_; lean_object* v___x_4273_; 
v___x_4270_ = lean_array_fget_borrowed(v_fst_4242_, v___x_4243_);
lean_inc(v___x_4270_);
v___x_4271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4271_, 0, v___x_4270_);
v___x_4272_ = lean_box(v___x_4248_);
{
lean_object* _aargs[] = {v___x_4241_, v___x_4271_, v_fst_4245_, v_snd_4246_, v___x_4247_, v_fst_4263_, v_snd_4264_, v___x_4272_, v___y_4249_, v___y_4250_, v___y_4251_, v___y_4252_, v___y_4253_, v___y_4254_, v___y_4255_, v___y_4256_, lean_box(0)};
v___x_4273_ = lean_apply_m(v___f_4244_, 17, _aargs);
}
return v___x_4273_;
}
}
else
{
lean_object* v_a_4274_; lean_object* v___x_4276_; uint8_t v_isShared_4277_; uint8_t v_isSharedCheck_4281_; 
lean_dec(v___y_4256_);
lean_dec_ref(v___y_4255_);
lean_dec(v___y_4254_);
lean_dec_ref(v___y_4253_);
lean_dec(v___y_4252_);
lean_dec_ref(v___y_4251_);
lean_dec(v___y_4250_);
lean_dec_ref(v___y_4249_);
lean_dec_ref(v___x_4247_);
lean_dec(v_snd_4246_);
lean_dec(v_fst_4245_);
lean_dec_ref(v___f_4244_);
lean_dec(v___x_4241_);
v_a_4274_ = lean_ctor_get(v___x_4261_, 0);
v_isSharedCheck_4281_ = !lean_is_exclusive(v___x_4261_);
if (v_isSharedCheck_4281_ == 0)
{
v___x_4276_ = v___x_4261_;
v_isShared_4277_ = v_isSharedCheck_4281_;
goto v_resetjp_4275_;
}
else
{
lean_inc(v_a_4274_);
lean_dec(v___x_4261_);
v___x_4276_ = lean_box(0);
v_isShared_4277_ = v_isSharedCheck_4281_;
goto v_resetjp_4275_;
}
v_resetjp_4275_:
{
lean_object* v___x_4279_; 
if (v_isShared_4277_ == 0)
{
v___x_4279_ = v___x_4276_;
goto v_reusejp_4278_;
}
else
{
lean_object* v_reuseFailAlloc_4280_; 
v_reuseFailAlloc_4280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4280_, 0, v_a_4274_);
v___x_4279_ = v_reuseFailAlloc_4280_;
goto v_reusejp_4278_;
}
v_reusejp_4278_:
{
return v___x_4279_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__1___boxed(lean_object** _args){
lean_object* v___x_4282_ = _args[0];
lean_object* v___x_4283_ = _args[1];
lean_object* v_fst_4284_ = _args[2];
lean_object* v___x_4285_ = _args[3];
lean_object* v___f_4286_ = _args[4];
lean_object* v_fst_4287_ = _args[5];
lean_object* v_snd_4288_ = _args[6];
lean_object* v___x_4289_ = _args[7];
lean_object* v___x_4290_ = _args[8];
lean_object* v___y_4291_ = _args[9];
lean_object* v___y_4292_ = _args[10];
lean_object* v___y_4293_ = _args[11];
lean_object* v___y_4294_ = _args[12];
lean_object* v___y_4295_ = _args[13];
lean_object* v___y_4296_ = _args[14];
lean_object* v___y_4297_ = _args[15];
lean_object* v___y_4298_ = _args[16];
lean_object* v___y_4299_ = _args[17];
_start:
{
uint8_t v___x_37890__boxed_4300_; lean_object* v_res_4301_; 
v___x_37890__boxed_4300_ = lean_unbox(v___x_4290_);
v_res_4301_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__1(v___x_4282_, v___x_4283_, v_fst_4284_, v___x_4285_, v___f_4286_, v_fst_4287_, v_snd_4288_, v___x_4289_, v___x_37890__boxed_4300_, v___y_4291_, v___y_4292_, v___y_4293_, v___y_4294_, v___y_4295_, v___y_4296_, v___y_4297_, v___y_4298_);
lean_dec(v___x_4285_);
lean_dec_ref(v_fst_4284_);
return v_res_4301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg(lean_object* v_ref_4302_, lean_object* v_msg_4303_, lean_object* v___y_4304_, lean_object* v___y_4305_, lean_object* v___y_4306_, lean_object* v___y_4307_, lean_object* v___y_4308_, lean_object* v___y_4309_, lean_object* v___y_4310_, lean_object* v___y_4311_){
_start:
{
lean_object* v_fileName_4313_; lean_object* v_fileMap_4314_; lean_object* v_options_4315_; lean_object* v_currRecDepth_4316_; lean_object* v_maxRecDepth_4317_; lean_object* v_ref_4318_; lean_object* v_currNamespace_4319_; lean_object* v_openDecls_4320_; lean_object* v_initHeartbeats_4321_; lean_object* v_maxHeartbeats_4322_; lean_object* v_quotContext_4323_; lean_object* v_currMacroScope_4324_; uint8_t v_diag_4325_; lean_object* v_cancelTk_x3f_4326_; uint8_t v_suppressElabErrors_4327_; lean_object* v_inheritedTraceOptions_4328_; lean_object* v_ref_4329_; lean_object* v___x_4330_; lean_object* v___x_4331_; 
v_fileName_4313_ = lean_ctor_get(v___y_4310_, 0);
v_fileMap_4314_ = lean_ctor_get(v___y_4310_, 1);
v_options_4315_ = lean_ctor_get(v___y_4310_, 2);
v_currRecDepth_4316_ = lean_ctor_get(v___y_4310_, 3);
v_maxRecDepth_4317_ = lean_ctor_get(v___y_4310_, 4);
v_ref_4318_ = lean_ctor_get(v___y_4310_, 5);
v_currNamespace_4319_ = lean_ctor_get(v___y_4310_, 6);
v_openDecls_4320_ = lean_ctor_get(v___y_4310_, 7);
v_initHeartbeats_4321_ = lean_ctor_get(v___y_4310_, 8);
v_maxHeartbeats_4322_ = lean_ctor_get(v___y_4310_, 9);
v_quotContext_4323_ = lean_ctor_get(v___y_4310_, 10);
v_currMacroScope_4324_ = lean_ctor_get(v___y_4310_, 11);
v_diag_4325_ = lean_ctor_get_uint8(v___y_4310_, sizeof(void*)*14);
v_cancelTk_x3f_4326_ = lean_ctor_get(v___y_4310_, 12);
v_suppressElabErrors_4327_ = lean_ctor_get_uint8(v___y_4310_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4328_ = lean_ctor_get(v___y_4310_, 13);
v_ref_4329_ = l_Lean_replaceRef(v_ref_4302_, v_ref_4318_);
lean_inc_ref(v_inheritedTraceOptions_4328_);
lean_inc(v_cancelTk_x3f_4326_);
lean_inc(v_currMacroScope_4324_);
lean_inc(v_quotContext_4323_);
lean_inc(v_maxHeartbeats_4322_);
lean_inc(v_initHeartbeats_4321_);
lean_inc(v_openDecls_4320_);
lean_inc(v_currNamespace_4319_);
lean_inc(v_maxRecDepth_4317_);
lean_inc(v_currRecDepth_4316_);
lean_inc_ref(v_options_4315_);
lean_inc_ref(v_fileMap_4314_);
lean_inc_ref(v_fileName_4313_);
v___x_4330_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4330_, 0, v_fileName_4313_);
lean_ctor_set(v___x_4330_, 1, v_fileMap_4314_);
lean_ctor_set(v___x_4330_, 2, v_options_4315_);
lean_ctor_set(v___x_4330_, 3, v_currRecDepth_4316_);
lean_ctor_set(v___x_4330_, 4, v_maxRecDepth_4317_);
lean_ctor_set(v___x_4330_, 5, v_ref_4329_);
lean_ctor_set(v___x_4330_, 6, v_currNamespace_4319_);
lean_ctor_set(v___x_4330_, 7, v_openDecls_4320_);
lean_ctor_set(v___x_4330_, 8, v_initHeartbeats_4321_);
lean_ctor_set(v___x_4330_, 9, v_maxHeartbeats_4322_);
lean_ctor_set(v___x_4330_, 10, v_quotContext_4323_);
lean_ctor_set(v___x_4330_, 11, v_currMacroScope_4324_);
lean_ctor_set(v___x_4330_, 12, v_cancelTk_x3f_4326_);
lean_ctor_set(v___x_4330_, 13, v_inheritedTraceOptions_4328_);
lean_ctor_set_uint8(v___x_4330_, sizeof(void*)*14, v_diag_4325_);
lean_ctor_set_uint8(v___x_4330_, sizeof(void*)*14 + 1, v_suppressElabErrors_4327_);
v___x_4331_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v_msg_4303_, v___y_4308_, v___y_4309_, v___x_4330_, v___y_4311_);
lean_dec_ref_known(v___x_4330_, 14);
return v___x_4331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg___boxed(lean_object* v_ref_4332_, lean_object* v_msg_4333_, lean_object* v___y_4334_, lean_object* v___y_4335_, lean_object* v___y_4336_, lean_object* v___y_4337_, lean_object* v___y_4338_, lean_object* v___y_4339_, lean_object* v___y_4340_, lean_object* v___y_4341_, lean_object* v___y_4342_){
_start:
{
lean_object* v_res_4343_; 
v_res_4343_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg(v_ref_4332_, v_msg_4333_, v___y_4334_, v___y_4335_, v___y_4336_, v___y_4337_, v___y_4338_, v___y_4339_, v___y_4340_, v___y_4341_);
lean_dec(v___y_4341_);
lean_dec_ref(v___y_4340_);
lean_dec(v___y_4339_);
lean_dec_ref(v___y_4338_);
lean_dec(v___y_4337_);
lean_dec_ref(v___y_4336_);
lean_dec(v___y_4335_);
lean_dec_ref(v___y_4334_);
lean_dec(v_ref_4332_);
return v_res_4343_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__1(void){
_start:
{
lean_object* v___x_4345_; lean_object* v___x_4346_; 
v___x_4345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__0));
v___x_4346_ = l_Lean_stringToMessageData(v___x_4345_);
return v___x_4346_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__3(void){
_start:
{
lean_object* v___x_4348_; lean_object* v___x_4349_; 
v___x_4348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__2));
v___x_4349_ = l_Lean_stringToMessageData(v___x_4348_);
return v___x_4349_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__5(void){
_start:
{
lean_object* v___x_4351_; lean_object* v___x_4352_; 
v___x_4351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__4));
v___x_4352_ = l_Lean_stringToMessageData(v___x_4351_);
return v___x_4352_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__7(void){
_start:
{
lean_object* v___x_4354_; lean_object* v___x_4355_; 
v___x_4354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__6));
v___x_4355_ = l_Lean_stringToMessageData(v___x_4354_);
return v___x_4355_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__9(void){
_start:
{
lean_object* v___x_4357_; lean_object* v___x_4358_; 
v___x_4357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__8));
v___x_4358_ = l_Lean_stringToMessageData(v___x_4357_);
return v___x_4358_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__11(void){
_start:
{
lean_object* v___x_4360_; lean_object* v___x_4361_; 
v___x_4360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__10));
v___x_4361_ = l_Lean_stringToMessageData(v___x_4360_);
return v___x_4361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2(lean_object* v_lb_4362_, lean_object* v_e_4363_, lean_object* v_ub_4364_, lean_object* v_a_4365_, lean_object* v___y_4366_, lean_object* v___x_4367_, lean_object* v___x_4368_, lean_object* v___f_4369_, uint8_t v___x_4370_, lean_object* v___y_4371_, lean_object* v___y_4372_, lean_object* v___y_4373_, lean_object* v___y_4374_, lean_object* v___y_4375_, lean_object* v___y_4376_, lean_object* v___y_4377_, lean_object* v___y_4378_){
_start:
{
lean_object* v___y_4381_; lean_object* v___y_4382_; lean_object* v___y_4383_; lean_object* v___y_4384_; lean_object* v___y_4385_; lean_object* v___y_4386_; lean_object* v___y_4387_; lean_object* v___y_4388_; lean_object* v___y_4389_; lean_object* v___y_4390_; lean_object* v___y_4391_; lean_object* v___y_4392_; lean_object* v___y_4393_; lean_object* v___y_4394_; lean_object* v___y_4395_; lean_object* v___y_4405_; lean_object* v___y_4406_; lean_object* v___y_4407_; lean_object* v___y_4408_; lean_object* v___y_4409_; uint8_t v___y_4410_; lean_object* v___y_4411_; lean_object* v___y_4412_; lean_object* v___y_4413_; lean_object* v___y_4414_; lean_object* v___y_4415_; lean_object* v___y_4416_; lean_object* v___y_4454_; lean_object* v___y_4455_; lean_object* v___y_4456_; lean_object* v___y_4457_; uint8_t v___y_4458_; lean_object* v___y_4459_; lean_object* v___y_4460_; lean_object* v___y_4461_; lean_object* v___y_4462_; lean_object* v___y_4463_; lean_object* v___y_4464_; lean_object* v___y_4465_; lean_object* v___y_4466_; 
if (lean_obj_tag(v_lb_4362_) == 0)
{
if (lean_obj_tag(v_e_4363_) == 1)
{
if (lean_obj_tag(v_ub_4364_) == 0)
{
lean_object* v_val_4467_; lean_object* v___x_4468_; uint8_t v___x_4469_; lean_object* v___x_4470_; 
v_val_4467_ = lean_ctor_get(v_e_4363_, 0);
lean_inc(v_val_4467_);
lean_dec_ref_known(v_e_4363_, 1);
v___x_4468_ = lean_box(0);
v___x_4469_ = 0;
v___x_4470_ = l_Lean_Elab_Tactic_elabTerm(v_val_4467_, v___x_4468_, v___x_4469_, v___y_4371_, v___y_4372_, v___y_4373_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_);
if (lean_obj_tag(v___x_4470_) == 0)
{
lean_object* v_a_4471_; lean_object* v___x_4472_; 
v_a_4471_ = lean_ctor_get(v___x_4470_, 0);
lean_inc(v_a_4471_);
lean_dec_ref_known(v___x_4470_, 1);
lean_inc(v_a_4365_);
v___x_4472_ = lp_mathlib_Lean_Elab_Tactic_getFVarIdsAt(v_a_4365_, v___x_4468_, v___x_4469_, v___y_4371_, v___y_4372_, v___y_4373_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_);
if (lean_obj_tag(v___x_4472_) == 0)
{
lean_object* v_a_4473_; lean_object* v___x_4474_; lean_object* v___x_4475_; lean_object* v___x_4476_; lean_object* v___x_4477_; uint8_t v___x_4478_; lean_object* v___x_4479_; 
v_a_4473_ = lean_ctor_get(v___x_4472_, 0);
lean_inc(v_a_4473_);
lean_dec_ref_known(v___x_4472_, 1);
lean_inc(v_a_4471_);
v___x_4474_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4474_, 0, v_a_4471_);
lean_ctor_set(v___x_4474_, 1, v___x_4468_);
lean_ctor_set(v___x_4474_, 2, v___y_4366_);
v___x_4475_ = lean_mk_empty_array_with_capacity(v___x_4367_);
v___x_4476_ = lean_array_push(v___x_4475_, v___x_4474_);
v___x_4477_ = lean_box(0);
v___x_4478_ = 3;
v___x_4479_ = l_Lean_MVarId_generalizeHyp(v_a_4365_, v___x_4476_, v_a_4473_, v___x_4477_, v___x_4478_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_);
lean_dec(v_a_4473_);
if (lean_obj_tag(v___x_4479_) == 0)
{
lean_object* v_a_4480_; lean_object* v_snd_4481_; lean_object* v_fst_4482_; lean_object* v_fst_4483_; lean_object* v_snd_4484_; lean_object* v___x_4485_; lean_object* v___x_4486_; lean_object* v___x_4487_; lean_object* v___x_4488_; lean_object* v___x_4489_; lean_object* v___f_4490_; lean_object* v___x_4491_; 
v_a_4480_ = lean_ctor_get(v___x_4479_, 0);
lean_inc(v_a_4480_);
lean_dec_ref_known(v___x_4479_, 1);
v_snd_4481_ = lean_ctor_get(v_a_4480_, 1);
lean_inc(v_snd_4481_);
v_fst_4482_ = lean_ctor_get(v_a_4480_, 0);
lean_inc_n(v_fst_4482_, 2);
lean_dec(v_a_4480_);
v_fst_4483_ = lean_ctor_get(v_snd_4481_, 0);
lean_inc(v_fst_4483_);
v_snd_4484_ = lean_ctor_get(v_snd_4481_, 1);
lean_inc_n(v_snd_4484_, 2);
lean_dec(v_snd_4481_);
v___x_4485_ = lean_box(0);
v___x_4486_ = lean_array_get(v___x_4485_, v_fst_4483_, v___x_4368_);
v___x_4487_ = l_Lean_Meta_FVarSubst_apply(v_fst_4482_, v_a_4471_);
lean_dec(v_a_4471_);
v___x_4488_ = lean_mk_empty_array_with_capacity(v___x_4368_);
v___x_4489_ = lean_box(v___x_4469_);
v___f_4490_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__1___boxed), 18, 9);
lean_closure_set(v___f_4490_, 0, v___x_4488_);
lean_closure_set(v___f_4490_, 1, v___x_4486_);
lean_closure_set(v___f_4490_, 2, v_fst_4483_);
lean_closure_set(v___f_4490_, 3, v___x_4367_);
lean_closure_set(v___f_4490_, 4, v___f_4369_);
lean_closure_set(v___f_4490_, 5, v_fst_4482_);
lean_closure_set(v___f_4490_, 6, v_snd_4484_);
lean_closure_set(v___f_4490_, 7, v___x_4487_);
lean_closure_set(v___f_4490_, 8, v___x_4489_);
v___x_4491_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(v_snd_4484_, v___f_4490_, v___y_4371_, v___y_4372_, v___y_4373_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_);
return v___x_4491_;
}
else
{
lean_object* v_a_4492_; lean_object* v___x_4494_; uint8_t v_isShared_4495_; uint8_t v_isSharedCheck_4499_; 
lean_dec(v_a_4471_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
v_a_4492_ = lean_ctor_get(v___x_4479_, 0);
v_isSharedCheck_4499_ = !lean_is_exclusive(v___x_4479_);
if (v_isSharedCheck_4499_ == 0)
{
v___x_4494_ = v___x_4479_;
v_isShared_4495_ = v_isSharedCheck_4499_;
goto v_resetjp_4493_;
}
else
{
lean_inc(v_a_4492_);
lean_dec(v___x_4479_);
v___x_4494_ = lean_box(0);
v_isShared_4495_ = v_isSharedCheck_4499_;
goto v_resetjp_4493_;
}
v_resetjp_4493_:
{
lean_object* v___x_4497_; 
if (v_isShared_4495_ == 0)
{
v___x_4497_ = v___x_4494_;
goto v_reusejp_4496_;
}
else
{
lean_object* v_reuseFailAlloc_4498_; 
v_reuseFailAlloc_4498_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4498_, 0, v_a_4492_);
v___x_4497_ = v_reuseFailAlloc_4498_;
goto v_reusejp_4496_;
}
v_reusejp_4496_:
{
return v___x_4497_;
}
}
}
}
else
{
lean_object* v_a_4500_; lean_object* v___x_4502_; uint8_t v_isShared_4503_; uint8_t v_isSharedCheck_4507_; 
lean_dec(v_a_4471_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4500_ = lean_ctor_get(v___x_4472_, 0);
v_isSharedCheck_4507_ = !lean_is_exclusive(v___x_4472_);
if (v_isSharedCheck_4507_ == 0)
{
v___x_4502_ = v___x_4472_;
v_isShared_4503_ = v_isSharedCheck_4507_;
goto v_resetjp_4501_;
}
else
{
lean_inc(v_a_4500_);
lean_dec(v___x_4472_);
v___x_4502_ = lean_box(0);
v_isShared_4503_ = v_isSharedCheck_4507_;
goto v_resetjp_4501_;
}
v_resetjp_4501_:
{
lean_object* v___x_4505_; 
if (v_isShared_4503_ == 0)
{
v___x_4505_ = v___x_4502_;
goto v_reusejp_4504_;
}
else
{
lean_object* v_reuseFailAlloc_4506_; 
v_reuseFailAlloc_4506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4506_, 0, v_a_4500_);
v___x_4505_ = v_reuseFailAlloc_4506_;
goto v_reusejp_4504_;
}
v_reusejp_4504_:
{
return v___x_4505_;
}
}
}
}
else
{
lean_object* v_a_4508_; lean_object* v___x_4510_; uint8_t v_isShared_4511_; uint8_t v_isSharedCheck_4515_; 
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4508_ = lean_ctor_get(v___x_4470_, 0);
v_isSharedCheck_4515_ = !lean_is_exclusive(v___x_4470_);
if (v_isSharedCheck_4515_ == 0)
{
v___x_4510_ = v___x_4470_;
v_isShared_4511_ = v_isSharedCheck_4515_;
goto v_resetjp_4509_;
}
else
{
lean_inc(v_a_4508_);
lean_dec(v___x_4470_);
v___x_4510_ = lean_box(0);
v_isShared_4511_ = v_isSharedCheck_4515_;
goto v_resetjp_4509_;
}
v_resetjp_4509_:
{
lean_object* v___x_4513_; 
if (v_isShared_4511_ == 0)
{
v___x_4513_ = v___x_4510_;
goto v_reusejp_4512_;
}
else
{
lean_object* v_reuseFailAlloc_4514_; 
v_reuseFailAlloc_4514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4514_, 0, v_a_4508_);
v___x_4513_ = v_reuseFailAlloc_4514_;
goto v_reusejp_4512_;
}
v_reusejp_4512_:
{
return v___x_4513_;
}
}
}
}
else
{
lean_object* v___x_4516_; 
lean_dec_ref_known(v_e_4363_, 1);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
lean_dec(v_ub_4364_);
v___x_4516_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_4516_;
}
}
else
{
lean_object* v___x_4517_; 
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
lean_dec(v_ub_4364_);
lean_dec(v_e_4363_);
v___x_4517_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_4517_;
}
}
else
{
if (lean_obj_tag(v_ub_4364_) == 1)
{
lean_object* v_val_4518_; lean_object* v_val_4519_; lean_object* v___y_4521_; lean_object* v___y_4522_; lean_object* v___y_4523_; lean_object* v___y_4524_; lean_object* v___y_4525_; lean_object* v___y_4526_; lean_object* v___y_4527_; lean_object* v___y_4528_; lean_object* v___y_4529_; lean_object* v___y_4530_; uint8_t v___y_4531_; lean_object* v___y_4532_; lean_object* v___y_4533_; lean_object* v___y_4534_; lean_object* v___y_4535_; uint8_t v___y_4536_; lean_object* v___y_4550_; lean_object* v___y_4551_; lean_object* v___y_4552_; lean_object* v___y_4553_; lean_object* v___y_4554_; lean_object* v___y_4555_; lean_object* v___y_4556_; lean_object* v___y_4557_; lean_object* v___y_4558_; uint8_t v___y_4559_; lean_object* v___y_4560_; lean_object* v___y_4561_; lean_object* v___y_4562_; lean_object* v___y_4563_; lean_object* v___y_4564_; lean_object* v_a_4565_; lean_object* v___y_4569_; lean_object* v___y_4570_; lean_object* v___y_4571_; lean_object* v___y_4572_; lean_object* v___y_4573_; lean_object* v___y_4574_; lean_object* v___y_4575_; lean_object* v___y_4576_; uint8_t v___y_4577_; lean_object* v___y_4578_; lean_object* v___y_4579_; lean_object* v___y_4580_; lean_object* v___y_4581_; lean_object* v___y_4618_; lean_object* v___y_4619_; lean_object* v___y_4620_; lean_object* v___y_4621_; lean_object* v___y_4622_; lean_object* v___y_4623_; lean_object* v___y_4624_; lean_object* v___y_4625_; uint8_t v___y_4626_; lean_object* v___y_4627_; lean_object* v___y_4628_; lean_object* v___y_4629_; lean_object* v___y_4630_; lean_object* v___y_4631_; lean_object* v___y_4633_; lean_object* v___y_4634_; lean_object* v___y_4635_; lean_object* v___y_4636_; lean_object* v___y_4637_; lean_object* v___y_4638_; lean_object* v___y_4639_; lean_object* v___y_4640_; lean_object* v___y_4641_; lean_object* v___y_4642_; lean_object* v___y_4643_; uint8_t v___y_4644_; lean_object* v___y_4645_; lean_object* v___y_4646_; lean_object* v___y_4647_; lean_object* v___y_4648_; uint8_t v___y_4649_; lean_object* v___y_4663_; lean_object* v___y_4664_; lean_object* v___y_4665_; lean_object* v___y_4666_; lean_object* v___y_4667_; lean_object* v___y_4668_; lean_object* v___y_4669_; lean_object* v___y_4670_; lean_object* v___y_4671_; lean_object* v___y_4672_; uint8_t v___y_4673_; lean_object* v___y_4674_; lean_object* v___y_4675_; lean_object* v___y_4676_; lean_object* v___y_4677_; lean_object* v___y_4678_; lean_object* v_a_4679_; lean_object* v_e_4683_; lean_object* v___y_4684_; lean_object* v___y_4685_; lean_object* v___y_4686_; lean_object* v___y_4687_; lean_object* v___y_4688_; lean_object* v___y_4689_; lean_object* v___y_4690_; lean_object* v___y_4691_; 
v_val_4518_ = lean_ctor_get(v_lb_4362_, 0);
lean_inc(v_val_4518_);
lean_dec_ref_known(v_lb_4362_, 1);
v_val_4519_ = lean_ctor_get(v_ub_4364_, 0);
lean_inc(v_val_4519_);
lean_dec_ref_known(v_ub_4364_, 1);
if (lean_obj_tag(v_e_4363_) == 1)
{
lean_object* v_val_4770_; lean_object* v___x_4771_; uint8_t v___x_4772_; lean_object* v___x_4773_; 
v_val_4770_ = lean_ctor_get(v_e_4363_, 0);
lean_inc(v_val_4770_);
lean_dec_ref_known(v_e_4363_, 1);
v___x_4771_ = lean_box(0);
v___x_4772_ = 0;
v___x_4773_ = l_Lean_Elab_Tactic_elabTerm(v_val_4770_, v___x_4771_, v___x_4772_, v___y_4371_, v___y_4372_, v___y_4373_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_);
if (lean_obj_tag(v___x_4773_) == 0)
{
lean_object* v_a_4774_; 
v_a_4774_ = lean_ctor_get(v___x_4773_, 0);
lean_inc(v_a_4774_);
lean_dec_ref_known(v___x_4773_, 1);
v_e_4683_ = v_a_4774_;
v___y_4684_ = v___y_4371_;
v___y_4685_ = v___y_4372_;
v___y_4686_ = v___y_4373_;
v___y_4687_ = v___y_4374_;
v___y_4688_ = v___y_4375_;
v___y_4689_ = v___y_4376_;
v___y_4690_ = v___y_4377_;
v___y_4691_ = v___y_4378_;
goto v___jp_4682_;
}
else
{
lean_object* v_a_4775_; lean_object* v___x_4777_; uint8_t v_isShared_4778_; uint8_t v_isSharedCheck_4782_; 
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4775_ = lean_ctor_get(v___x_4773_, 0);
v_isSharedCheck_4782_ = !lean_is_exclusive(v___x_4773_);
if (v_isSharedCheck_4782_ == 0)
{
v___x_4777_ = v___x_4773_;
v_isShared_4778_ = v_isSharedCheck_4782_;
goto v_resetjp_4776_;
}
else
{
lean_inc(v_a_4775_);
lean_dec(v___x_4773_);
v___x_4777_ = lean_box(0);
v_isShared_4778_ = v_isSharedCheck_4782_;
goto v_resetjp_4776_;
}
v_resetjp_4776_:
{
lean_object* v___x_4780_; 
if (v_isShared_4778_ == 0)
{
v___x_4780_ = v___x_4777_;
goto v_reusejp_4779_;
}
else
{
lean_object* v_reuseFailAlloc_4781_; 
v_reuseFailAlloc_4781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4781_, 0, v_a_4775_);
v___x_4780_ = v_reuseFailAlloc_4781_;
goto v_reusejp_4779_;
}
v_reusejp_4779_:
{
return v___x_4780_;
}
}
}
}
else
{
lean_object* v___x_4783_; uint8_t v___x_4784_; lean_object* v___x_4785_; lean_object* v___x_4786_; 
lean_dec(v_e_4363_);
v___x_4783_ = lean_box(0);
v___x_4784_ = 0;
v___x_4785_ = lean_box(0);
v___x_4786_ = l_Lean_Meta_mkFreshExprMVar(v___x_4783_, v___x_4784_, v___x_4785_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_);
if (lean_obj_tag(v___x_4786_) == 0)
{
lean_object* v_a_4787_; 
v_a_4787_ = lean_ctor_get(v___x_4786_, 0);
lean_inc(v_a_4787_);
lean_dec_ref_known(v___x_4786_, 1);
v_e_4683_ = v_a_4787_;
v___y_4684_ = v___y_4371_;
v___y_4685_ = v___y_4372_;
v___y_4686_ = v___y_4373_;
v___y_4687_ = v___y_4374_;
v___y_4688_ = v___y_4375_;
v___y_4689_ = v___y_4376_;
v___y_4690_ = v___y_4377_;
v___y_4691_ = v___y_4378_;
goto v___jp_4682_;
}
else
{
lean_object* v_a_4788_; lean_object* v___x_4790_; uint8_t v_isShared_4791_; uint8_t v_isSharedCheck_4795_; 
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4788_ = lean_ctor_get(v___x_4786_, 0);
v_isSharedCheck_4795_ = !lean_is_exclusive(v___x_4786_);
if (v_isSharedCheck_4795_ == 0)
{
v___x_4790_ = v___x_4786_;
v_isShared_4791_ = v_isSharedCheck_4795_;
goto v_resetjp_4789_;
}
else
{
lean_inc(v_a_4788_);
lean_dec(v___x_4786_);
v___x_4790_ = lean_box(0);
v_isShared_4791_ = v_isSharedCheck_4795_;
goto v_resetjp_4789_;
}
v_resetjp_4789_:
{
lean_object* v___x_4793_; 
if (v_isShared_4791_ == 0)
{
v___x_4793_ = v___x_4790_;
goto v_reusejp_4792_;
}
else
{
lean_object* v_reuseFailAlloc_4794_; 
v_reuseFailAlloc_4794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4794_, 0, v_a_4788_);
v___x_4793_ = v_reuseFailAlloc_4794_;
goto v_reusejp_4792_;
}
v_reusejp_4792_:
{
return v___x_4793_;
}
}
}
}
v___jp_4520_:
{
if (v___y_4536_ == 0)
{
lean_object* v___x_4537_; 
lean_dec_ref(v___y_4521_);
v___x_4537_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_4524_, v___y_4536_, v___y_4528_, v___y_4534_, v___y_4535_, v___y_4523_, v___y_4525_, v___y_4526_, v___y_4527_);
if (lean_obj_tag(v___x_4537_) == 0)
{
lean_object* v___x_4538_; lean_object* v___x_4539_; lean_object* v___x_4540_; lean_object* v___x_4541_; lean_object* v___x_4542_; lean_object* v___x_4543_; lean_object* v___x_4544_; lean_object* v___x_4545_; lean_object* v___x_4546_; lean_object* v___x_4547_; lean_object* v___x_4548_; 
lean_dec_ref_known(v___x_4537_, 1);
v___x_4538_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__1);
lean_inc_ref(v___y_4532_);
v___x_4539_ = l_Lean_MessageData_ofExpr(v___y_4532_);
lean_inc_ref(v___x_4539_);
v___x_4540_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4540_, 0, v___x_4538_);
lean_ctor_set(v___x_4540_, 1, v___x_4539_);
v___x_4541_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__3, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__3_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__3);
v___x_4542_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4542_, 0, v___x_4540_);
lean_ctor_set(v___x_4542_, 1, v___x_4541_);
v___x_4543_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4543_, 0, v___x_4542_);
lean_ctor_set(v___x_4543_, 1, v___x_4539_);
v___x_4544_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__5);
v___x_4545_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4545_, 0, v___x_4543_);
lean_ctor_set(v___x_4545_, 1, v___x_4544_);
v___x_4546_ = l_Lean_MessageData_ofExpr(v___y_4522_);
v___x_4547_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4547_, 0, v___x_4545_);
lean_ctor_set(v___x_4547_, 1, v___x_4546_);
v___x_4548_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg(v_val_4519_, v___x_4547_, v___y_4533_, v___y_4528_, v___y_4534_, v___y_4535_, v___y_4523_, v___y_4525_, v___y_4526_, v___y_4527_);
lean_dec(v_val_4519_);
v___y_4454_ = v___y_4528_;
v___y_4455_ = v___y_4529_;
v___y_4456_ = v___y_4530_;
v___y_4457_ = v___y_4523_;
v___y_4458_ = v___y_4531_;
v___y_4459_ = v___y_4532_;
v___y_4460_ = v___y_4533_;
v___y_4461_ = v___y_4534_;
v___y_4462_ = v___y_4535_;
v___y_4463_ = v___y_4525_;
v___y_4464_ = v___y_4526_;
v___y_4465_ = v___y_4527_;
v___y_4466_ = v___x_4548_;
goto v___jp_4453_;
}
else
{
lean_dec_ref(v___y_4522_);
lean_dec(v_val_4519_);
v___y_4454_ = v___y_4528_;
v___y_4455_ = v___y_4529_;
v___y_4456_ = v___y_4530_;
v___y_4457_ = v___y_4523_;
v___y_4458_ = v___y_4531_;
v___y_4459_ = v___y_4532_;
v___y_4460_ = v___y_4533_;
v___y_4461_ = v___y_4534_;
v___y_4462_ = v___y_4535_;
v___y_4463_ = v___y_4525_;
v___y_4464_ = v___y_4526_;
v___y_4465_ = v___y_4527_;
v___y_4466_ = v___x_4537_;
goto v___jp_4453_;
}
}
else
{
lean_dec_ref(v___y_4532_);
lean_dec_ref(v___y_4530_);
lean_dec_ref(v___y_4529_);
lean_dec_ref(v___y_4524_);
lean_dec_ref(v___y_4522_);
lean_dec(v_val_4519_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
return v___y_4521_;
}
}
v___jp_4549_:
{
uint8_t v___x_4566_; 
v___x_4566_ = l_Lean_Exception_isInterrupt(v_a_4565_);
if (v___x_4566_ == 0)
{
uint8_t v___x_4567_; 
v___x_4567_ = l_Lean_Exception_isRuntime(v_a_4565_);
v___y_4521_ = v___y_4564_;
v___y_4522_ = v___y_4550_;
v___y_4523_ = v___y_4551_;
v___y_4524_ = v___y_4552_;
v___y_4525_ = v___y_4553_;
v___y_4526_ = v___y_4554_;
v___y_4527_ = v___y_4555_;
v___y_4528_ = v___y_4556_;
v___y_4529_ = v___y_4557_;
v___y_4530_ = v___y_4558_;
v___y_4531_ = v___y_4559_;
v___y_4532_ = v___y_4560_;
v___y_4533_ = v___y_4561_;
v___y_4534_ = v___y_4562_;
v___y_4535_ = v___y_4563_;
v___y_4536_ = v___x_4567_;
goto v___jp_4520_;
}
else
{
lean_dec_ref(v_a_4565_);
v___y_4521_ = v___y_4564_;
v___y_4522_ = v___y_4550_;
v___y_4523_ = v___y_4551_;
v___y_4524_ = v___y_4552_;
v___y_4525_ = v___y_4553_;
v___y_4526_ = v___y_4554_;
v___y_4527_ = v___y_4555_;
v___y_4528_ = v___y_4556_;
v___y_4529_ = v___y_4557_;
v___y_4530_ = v___y_4558_;
v___y_4531_ = v___y_4559_;
v___y_4532_ = v___y_4560_;
v___y_4533_ = v___y_4561_;
v___y_4534_ = v___y_4562_;
v___y_4535_ = v___y_4563_;
v___y_4536_ = v___x_4566_;
goto v___jp_4520_;
}
}
v___jp_4568_:
{
lean_object* v___x_4582_; 
v___x_4582_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_4574_, v___y_4581_, v___y_4571_, v___y_4572_);
if (lean_obj_tag(v___x_4582_) == 0)
{
lean_object* v_a_4583_; lean_object* v___x_4584_; 
v_a_4583_ = lean_ctor_get(v___x_4582_, 0);
lean_inc(v_a_4583_);
lean_dec_ref_known(v___x_4582_, 1);
lean_inc_ref(v___y_4569_);
v___x_4584_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v___y_4569_, v___y_4570_, v___y_4571_, v___y_4573_, v___y_4572_);
if (lean_obj_tag(v___x_4584_) == 0)
{
lean_object* v_a_4585_; lean_object* v_fst_4586_; lean_object* v___x_4587_; 
v_a_4585_ = lean_ctor_get(v___x_4584_, 0);
lean_inc(v_a_4585_);
lean_dec_ref_known(v___x_4584_, 1);
v_fst_4586_ = lean_ctor_get(v_a_4585_, 0);
lean_inc(v_fst_4586_);
lean_dec(v_a_4585_);
lean_inc_ref(v___y_4578_);
v___x_4587_ = l_Lean_Meta_isExprDefEq(v___y_4578_, v_fst_4586_, v___y_4570_, v___y_4571_, v___y_4573_, v___y_4572_);
if (lean_obj_tag(v___x_4587_) == 0)
{
lean_object* v_a_4588_; uint8_t v___x_4589_; 
v_a_4588_ = lean_ctor_get(v___x_4587_, 0);
lean_inc(v_a_4588_);
lean_dec_ref_known(v___x_4587_, 1);
v___x_4589_ = lean_unbox(v_a_4588_);
lean_dec(v_a_4588_);
if (v___x_4589_ == 1)
{
lean_dec(v_a_4583_);
lean_dec_ref(v___y_4569_);
lean_dec(v_val_4519_);
v___y_4405_ = v___y_4574_;
v___y_4406_ = v___y_4575_;
v___y_4407_ = v___y_4576_;
v___y_4408_ = v___y_4570_;
v___y_4409_ = v___y_4578_;
v___y_4410_ = v___y_4577_;
v___y_4411_ = v___y_4579_;
v___y_4412_ = v___y_4580_;
v___y_4413_ = v___y_4572_;
v___y_4414_ = v___y_4573_;
v___y_4415_ = v___y_4571_;
v___y_4416_ = v___y_4581_;
goto v___jp_4404_;
}
else
{
lean_object* v___x_4590_; lean_object* v___x_4591_; lean_object* v_a_4592_; 
v___x_4590_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1);
v___x_4591_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v___x_4590_, v___y_4570_, v___y_4571_, v___y_4573_, v___y_4572_);
v_a_4592_ = lean_ctor_get(v___x_4591_, 0);
lean_inc(v_a_4592_);
v___y_4550_ = v___y_4569_;
v___y_4551_ = v___y_4570_;
v___y_4552_ = v_a_4583_;
v___y_4553_ = v___y_4571_;
v___y_4554_ = v___y_4573_;
v___y_4555_ = v___y_4572_;
v___y_4556_ = v___y_4574_;
v___y_4557_ = v___y_4575_;
v___y_4558_ = v___y_4576_;
v___y_4559_ = v___y_4577_;
v___y_4560_ = v___y_4578_;
v___y_4561_ = v___y_4579_;
v___y_4562_ = v___y_4580_;
v___y_4563_ = v___y_4581_;
v___y_4564_ = v___x_4591_;
v_a_4565_ = v_a_4592_;
goto v___jp_4549_;
}
}
else
{
lean_object* v_a_4593_; lean_object* v___x_4595_; uint8_t v_isShared_4596_; uint8_t v_isSharedCheck_4600_; 
v_a_4593_ = lean_ctor_get(v___x_4587_, 0);
v_isSharedCheck_4600_ = !lean_is_exclusive(v___x_4587_);
if (v_isSharedCheck_4600_ == 0)
{
v___x_4595_ = v___x_4587_;
v_isShared_4596_ = v_isSharedCheck_4600_;
goto v_resetjp_4594_;
}
else
{
lean_inc(v_a_4593_);
lean_dec(v___x_4587_);
v___x_4595_ = lean_box(0);
v_isShared_4596_ = v_isSharedCheck_4600_;
goto v_resetjp_4594_;
}
v_resetjp_4594_:
{
lean_object* v___x_4598_; 
lean_inc(v_a_4593_);
if (v_isShared_4596_ == 0)
{
v___x_4598_ = v___x_4595_;
goto v_reusejp_4597_;
}
else
{
lean_object* v_reuseFailAlloc_4599_; 
v_reuseFailAlloc_4599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4599_, 0, v_a_4593_);
v___x_4598_ = v_reuseFailAlloc_4599_;
goto v_reusejp_4597_;
}
v_reusejp_4597_:
{
v___y_4550_ = v___y_4569_;
v___y_4551_ = v___y_4570_;
v___y_4552_ = v_a_4583_;
v___y_4553_ = v___y_4571_;
v___y_4554_ = v___y_4573_;
v___y_4555_ = v___y_4572_;
v___y_4556_ = v___y_4574_;
v___y_4557_ = v___y_4575_;
v___y_4558_ = v___y_4576_;
v___y_4559_ = v___y_4577_;
v___y_4560_ = v___y_4578_;
v___y_4561_ = v___y_4579_;
v___y_4562_ = v___y_4580_;
v___y_4563_ = v___y_4581_;
v___y_4564_ = v___x_4598_;
v_a_4565_ = v_a_4593_;
goto v___jp_4549_;
}
}
}
}
else
{
lean_object* v_a_4601_; lean_object* v___x_4603_; uint8_t v_isShared_4604_; uint8_t v_isSharedCheck_4608_; 
v_a_4601_ = lean_ctor_get(v___x_4584_, 0);
v_isSharedCheck_4608_ = !lean_is_exclusive(v___x_4584_);
if (v_isSharedCheck_4608_ == 0)
{
v___x_4603_ = v___x_4584_;
v_isShared_4604_ = v_isSharedCheck_4608_;
goto v_resetjp_4602_;
}
else
{
lean_inc(v_a_4601_);
lean_dec(v___x_4584_);
v___x_4603_ = lean_box(0);
v_isShared_4604_ = v_isSharedCheck_4608_;
goto v_resetjp_4602_;
}
v_resetjp_4602_:
{
lean_object* v___x_4606_; 
lean_inc(v_a_4601_);
if (v_isShared_4604_ == 0)
{
v___x_4606_ = v___x_4603_;
goto v_reusejp_4605_;
}
else
{
lean_object* v_reuseFailAlloc_4607_; 
v_reuseFailAlloc_4607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4607_, 0, v_a_4601_);
v___x_4606_ = v_reuseFailAlloc_4607_;
goto v_reusejp_4605_;
}
v_reusejp_4605_:
{
v___y_4550_ = v___y_4569_;
v___y_4551_ = v___y_4570_;
v___y_4552_ = v_a_4583_;
v___y_4553_ = v___y_4571_;
v___y_4554_ = v___y_4573_;
v___y_4555_ = v___y_4572_;
v___y_4556_ = v___y_4574_;
v___y_4557_ = v___y_4575_;
v___y_4558_ = v___y_4576_;
v___y_4559_ = v___y_4577_;
v___y_4560_ = v___y_4578_;
v___y_4561_ = v___y_4579_;
v___y_4562_ = v___y_4580_;
v___y_4563_ = v___y_4581_;
v___y_4564_ = v___x_4606_;
v_a_4565_ = v_a_4601_;
goto v___jp_4549_;
}
}
}
}
else
{
lean_object* v_a_4609_; lean_object* v___x_4611_; uint8_t v_isShared_4612_; uint8_t v_isSharedCheck_4616_; 
lean_dec_ref(v___y_4578_);
lean_dec_ref(v___y_4576_);
lean_dec_ref(v___y_4575_);
lean_dec_ref(v___y_4569_);
lean_dec(v_val_4519_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4609_ = lean_ctor_get(v___x_4582_, 0);
v_isSharedCheck_4616_ = !lean_is_exclusive(v___x_4582_);
if (v_isSharedCheck_4616_ == 0)
{
v___x_4611_ = v___x_4582_;
v_isShared_4612_ = v_isSharedCheck_4616_;
goto v_resetjp_4610_;
}
else
{
lean_inc(v_a_4609_);
lean_dec(v___x_4582_);
v___x_4611_ = lean_box(0);
v_isShared_4612_ = v_isSharedCheck_4616_;
goto v_resetjp_4610_;
}
v_resetjp_4610_:
{
lean_object* v___x_4614_; 
if (v_isShared_4612_ == 0)
{
v___x_4614_ = v___x_4611_;
goto v_reusejp_4613_;
}
else
{
lean_object* v_reuseFailAlloc_4615_; 
v_reuseFailAlloc_4615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4615_, 0, v_a_4609_);
v___x_4614_ = v_reuseFailAlloc_4615_;
goto v_reusejp_4613_;
}
v_reusejp_4613_:
{
return v___x_4614_;
}
}
}
}
v___jp_4617_:
{
if (lean_obj_tag(v___y_4631_) == 0)
{
lean_dec_ref_known(v___y_4631_, 1);
v___y_4569_ = v___y_4618_;
v___y_4570_ = v___y_4619_;
v___y_4571_ = v___y_4620_;
v___y_4572_ = v___y_4621_;
v___y_4573_ = v___y_4622_;
v___y_4574_ = v___y_4623_;
v___y_4575_ = v___y_4624_;
v___y_4576_ = v___y_4625_;
v___y_4577_ = v___y_4626_;
v___y_4578_ = v___y_4627_;
v___y_4579_ = v___y_4628_;
v___y_4580_ = v___y_4629_;
v___y_4581_ = v___y_4630_;
goto v___jp_4568_;
}
else
{
lean_dec_ref(v___y_4627_);
lean_dec_ref(v___y_4625_);
lean_dec_ref(v___y_4624_);
lean_dec_ref(v___y_4618_);
lean_dec(v_val_4519_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
return v___y_4631_;
}
}
v___jp_4632_:
{
if (v___y_4649_ == 0)
{
lean_object* v___x_4650_; 
lean_dec_ref(v___y_4639_);
v___x_4650_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_4634_, v___y_4649_, v___y_4640_, v___y_4647_, v___y_4648_, v___y_4635_, v___y_4636_, v___y_4637_, v___y_4638_);
if (lean_obj_tag(v___x_4650_) == 0)
{
lean_object* v___x_4651_; lean_object* v___x_4652_; lean_object* v___x_4653_; lean_object* v___x_4654_; lean_object* v___x_4655_; lean_object* v___x_4656_; lean_object* v___x_4657_; lean_object* v___x_4658_; lean_object* v___x_4659_; lean_object* v___x_4660_; lean_object* v___x_4661_; 
lean_dec_ref_known(v___x_4650_, 1);
v___x_4651_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__7, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__7_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__7);
lean_inc_ref(v___y_4645_);
v___x_4652_ = l_Lean_MessageData_ofExpr(v___y_4645_);
lean_inc_ref(v___x_4652_);
v___x_4653_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4653_, 0, v___x_4651_);
lean_ctor_set(v___x_4653_, 1, v___x_4652_);
v___x_4654_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__9);
v___x_4655_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4655_, 0, v___x_4653_);
lean_ctor_set(v___x_4655_, 1, v___x_4654_);
v___x_4656_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4656_, 0, v___x_4655_);
lean_ctor_set(v___x_4656_, 1, v___x_4652_);
v___x_4657_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__11, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__11_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___closed__11);
v___x_4658_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4658_, 0, v___x_4656_);
lean_ctor_set(v___x_4658_, 1, v___x_4657_);
v___x_4659_ = l_Lean_MessageData_ofExpr(v___y_4643_);
v___x_4660_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4660_, 0, v___x_4658_);
lean_ctor_set(v___x_4660_, 1, v___x_4659_);
v___x_4661_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg(v_val_4518_, v___x_4660_, v___y_4646_, v___y_4640_, v___y_4647_, v___y_4648_, v___y_4635_, v___y_4636_, v___y_4637_, v___y_4638_);
lean_dec(v_val_4518_);
v___y_4618_ = v___y_4633_;
v___y_4619_ = v___y_4635_;
v___y_4620_ = v___y_4636_;
v___y_4621_ = v___y_4638_;
v___y_4622_ = v___y_4637_;
v___y_4623_ = v___y_4640_;
v___y_4624_ = v___y_4641_;
v___y_4625_ = v___y_4642_;
v___y_4626_ = v___y_4644_;
v___y_4627_ = v___y_4645_;
v___y_4628_ = v___y_4646_;
v___y_4629_ = v___y_4647_;
v___y_4630_ = v___y_4648_;
v___y_4631_ = v___x_4661_;
goto v___jp_4617_;
}
else
{
lean_dec_ref(v___y_4643_);
lean_dec(v_val_4518_);
v___y_4618_ = v___y_4633_;
v___y_4619_ = v___y_4635_;
v___y_4620_ = v___y_4636_;
v___y_4621_ = v___y_4638_;
v___y_4622_ = v___y_4637_;
v___y_4623_ = v___y_4640_;
v___y_4624_ = v___y_4641_;
v___y_4625_ = v___y_4642_;
v___y_4626_ = v___y_4644_;
v___y_4627_ = v___y_4645_;
v___y_4628_ = v___y_4646_;
v___y_4629_ = v___y_4647_;
v___y_4630_ = v___y_4648_;
v___y_4631_ = v___x_4650_;
goto v___jp_4617_;
}
}
else
{
lean_dec_ref(v___y_4645_);
lean_dec_ref(v___y_4643_);
lean_dec_ref(v___y_4642_);
lean_dec_ref(v___y_4641_);
lean_dec_ref(v___y_4634_);
lean_dec_ref(v___y_4633_);
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
return v___y_4639_;
}
}
v___jp_4662_:
{
uint8_t v___x_4680_; 
v___x_4680_ = l_Lean_Exception_isInterrupt(v_a_4679_);
if (v___x_4680_ == 0)
{
uint8_t v___x_4681_; 
v___x_4681_ = l_Lean_Exception_isRuntime(v_a_4679_);
v___y_4633_ = v___y_4663_;
v___y_4634_ = v___y_4664_;
v___y_4635_ = v___y_4665_;
v___y_4636_ = v___y_4666_;
v___y_4637_ = v___y_4667_;
v___y_4638_ = v___y_4668_;
v___y_4639_ = v___y_4678_;
v___y_4640_ = v___y_4669_;
v___y_4641_ = v___y_4670_;
v___y_4642_ = v___y_4671_;
v___y_4643_ = v___y_4672_;
v___y_4644_ = v___y_4673_;
v___y_4645_ = v___y_4674_;
v___y_4646_ = v___y_4675_;
v___y_4647_ = v___y_4676_;
v___y_4648_ = v___y_4677_;
v___y_4649_ = v___x_4681_;
goto v___jp_4632_;
}
else
{
lean_dec_ref(v_a_4679_);
v___y_4633_ = v___y_4663_;
v___y_4634_ = v___y_4664_;
v___y_4635_ = v___y_4665_;
v___y_4636_ = v___y_4666_;
v___y_4637_ = v___y_4667_;
v___y_4638_ = v___y_4668_;
v___y_4639_ = v___y_4678_;
v___y_4640_ = v___y_4669_;
v___y_4641_ = v___y_4670_;
v___y_4642_ = v___y_4671_;
v___y_4643_ = v___y_4672_;
v___y_4644_ = v___y_4673_;
v___y_4645_ = v___y_4674_;
v___y_4646_ = v___y_4675_;
v___y_4647_ = v___y_4676_;
v___y_4648_ = v___y_4677_;
v___y_4649_ = v___x_4680_;
goto v___jp_4632_;
}
}
v___jp_4682_:
{
lean_object* v___x_4692_; uint8_t v___x_4693_; lean_object* v___x_4694_; 
v___x_4692_ = lean_box(0);
v___x_4693_ = 0;
lean_inc(v_val_4518_);
v___x_4694_ = l_Lean_Elab_Tactic_elabTerm(v_val_4518_, v___x_4692_, v___x_4693_, v___y_4684_, v___y_4685_, v___y_4686_, v___y_4687_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
if (lean_obj_tag(v___x_4694_) == 0)
{
lean_object* v_a_4695_; lean_object* v___x_4696_; 
v_a_4695_ = lean_ctor_get(v___x_4694_, 0);
lean_inc(v_a_4695_);
lean_dec_ref_known(v___x_4694_, 1);
lean_inc(v_val_4519_);
v___x_4696_ = l_Lean_Elab_Tactic_elabTerm(v_val_4519_, v___x_4692_, v___x_4693_, v___y_4684_, v___y_4685_, v___y_4686_, v___y_4687_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
if (lean_obj_tag(v___x_4696_) == 0)
{
lean_object* v_a_4697_; lean_object* v___x_4698_; 
v_a_4697_ = lean_ctor_get(v___x_4696_, 0);
lean_inc(v_a_4697_);
lean_dec_ref_known(v___x_4696_, 1);
lean_inc(v___y_4691_);
lean_inc_ref(v___y_4690_);
lean_inc(v___y_4689_);
lean_inc_ref(v___y_4688_);
lean_inc(v_a_4695_);
v___x_4698_ = lean_infer_type(v_a_4695_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
if (lean_obj_tag(v___x_4698_) == 0)
{
lean_object* v_a_4699_; lean_object* v___x_4700_; 
v_a_4699_ = lean_ctor_get(v___x_4698_, 0);
lean_inc(v_a_4699_);
lean_dec_ref_known(v___x_4698_, 1);
lean_inc(v___y_4691_);
lean_inc_ref(v___y_4690_);
lean_inc(v___y_4689_);
lean_inc_ref(v___y_4688_);
lean_inc(v_a_4697_);
v___x_4700_ = lean_infer_type(v_a_4697_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
if (lean_obj_tag(v___x_4700_) == 0)
{
lean_object* v_a_4701_; lean_object* v___x_4702_; 
v_a_4701_ = lean_ctor_get(v___x_4700_, 0);
lean_inc(v_a_4701_);
lean_dec_ref_known(v___x_4700_, 1);
v___x_4702_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_4685_, v___y_4687_, v___y_4689_, v___y_4691_);
if (lean_obj_tag(v___x_4702_) == 0)
{
lean_object* v_a_4703_; lean_object* v___x_4704_; 
v_a_4703_ = lean_ctor_get(v___x_4702_, 0);
lean_inc(v_a_4703_);
lean_dec_ref_known(v___x_4702_, 1);
lean_inc(v_a_4699_);
v___x_4704_ = lp_mathlib_Mathlib_Tactic_IntervalCases_parseBound(v_a_4699_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
if (lean_obj_tag(v___x_4704_) == 0)
{
lean_object* v_a_4705_; lean_object* v_snd_4706_; lean_object* v_fst_4707_; lean_object* v___x_4708_; 
v_a_4705_ = lean_ctor_get(v___x_4704_, 0);
lean_inc(v_a_4705_);
lean_dec_ref_known(v___x_4704_, 1);
v_snd_4706_ = lean_ctor_get(v_a_4705_, 1);
lean_inc(v_snd_4706_);
lean_dec(v_a_4705_);
v_fst_4707_ = lean_ctor_get(v_snd_4706_, 0);
lean_inc(v_fst_4707_);
lean_dec(v_snd_4706_);
lean_inc_ref(v_e_4683_);
v___x_4708_ = l_Lean_Meta_isExprDefEq(v_e_4683_, v_fst_4707_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
if (lean_obj_tag(v___x_4708_) == 0)
{
lean_object* v_a_4709_; uint8_t v___x_4710_; 
v_a_4709_ = lean_ctor_get(v___x_4708_, 0);
lean_inc(v_a_4709_);
lean_dec_ref_known(v___x_4708_, 1);
v___x_4710_ = lean_unbox(v_a_4709_);
lean_dec(v_a_4709_);
if (v___x_4710_ == 1)
{
lean_dec(v_a_4703_);
lean_dec(v_a_4699_);
lean_dec(v_val_4518_);
v___y_4569_ = v_a_4701_;
v___y_4570_ = v___y_4688_;
v___y_4571_ = v___y_4689_;
v___y_4572_ = v___y_4691_;
v___y_4573_ = v___y_4690_;
v___y_4574_ = v___y_4685_;
v___y_4575_ = v_a_4695_;
v___y_4576_ = v_a_4697_;
v___y_4577_ = v___x_4693_;
v___y_4578_ = v_e_4683_;
v___y_4579_ = v___y_4684_;
v___y_4580_ = v___y_4686_;
v___y_4581_ = v___y_4687_;
goto v___jp_4568_;
}
else
{
lean_object* v___x_4711_; lean_object* v___x_4712_; lean_object* v_a_4713_; 
v___x_4711_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__5_spec__6_spec__9___closed__1);
v___x_4712_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v___x_4711_, v___y_4688_, v___y_4689_, v___y_4690_, v___y_4691_);
v_a_4713_ = lean_ctor_get(v___x_4712_, 0);
lean_inc(v_a_4713_);
v___y_4663_ = v_a_4701_;
v___y_4664_ = v_a_4703_;
v___y_4665_ = v___y_4688_;
v___y_4666_ = v___y_4689_;
v___y_4667_ = v___y_4690_;
v___y_4668_ = v___y_4691_;
v___y_4669_ = v___y_4685_;
v___y_4670_ = v_a_4695_;
v___y_4671_ = v_a_4697_;
v___y_4672_ = v_a_4699_;
v___y_4673_ = v___x_4693_;
v___y_4674_ = v_e_4683_;
v___y_4675_ = v___y_4684_;
v___y_4676_ = v___y_4686_;
v___y_4677_ = v___y_4687_;
v___y_4678_ = v___x_4712_;
v_a_4679_ = v_a_4713_;
goto v___jp_4662_;
}
}
else
{
lean_object* v_a_4714_; lean_object* v___x_4716_; uint8_t v_isShared_4717_; uint8_t v_isSharedCheck_4721_; 
v_a_4714_ = lean_ctor_get(v___x_4708_, 0);
v_isSharedCheck_4721_ = !lean_is_exclusive(v___x_4708_);
if (v_isSharedCheck_4721_ == 0)
{
v___x_4716_ = v___x_4708_;
v_isShared_4717_ = v_isSharedCheck_4721_;
goto v_resetjp_4715_;
}
else
{
lean_inc(v_a_4714_);
lean_dec(v___x_4708_);
v___x_4716_ = lean_box(0);
v_isShared_4717_ = v_isSharedCheck_4721_;
goto v_resetjp_4715_;
}
v_resetjp_4715_:
{
lean_object* v___x_4719_; 
lean_inc(v_a_4714_);
if (v_isShared_4717_ == 0)
{
v___x_4719_ = v___x_4716_;
goto v_reusejp_4718_;
}
else
{
lean_object* v_reuseFailAlloc_4720_; 
v_reuseFailAlloc_4720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4720_, 0, v_a_4714_);
v___x_4719_ = v_reuseFailAlloc_4720_;
goto v_reusejp_4718_;
}
v_reusejp_4718_:
{
v___y_4663_ = v_a_4701_;
v___y_4664_ = v_a_4703_;
v___y_4665_ = v___y_4688_;
v___y_4666_ = v___y_4689_;
v___y_4667_ = v___y_4690_;
v___y_4668_ = v___y_4691_;
v___y_4669_ = v___y_4685_;
v___y_4670_ = v_a_4695_;
v___y_4671_ = v_a_4697_;
v___y_4672_ = v_a_4699_;
v___y_4673_ = v___x_4693_;
v___y_4674_ = v_e_4683_;
v___y_4675_ = v___y_4684_;
v___y_4676_ = v___y_4686_;
v___y_4677_ = v___y_4687_;
v___y_4678_ = v___x_4719_;
v_a_4679_ = v_a_4714_;
goto v___jp_4662_;
}
}
}
}
else
{
lean_object* v_a_4722_; lean_object* v___x_4724_; uint8_t v_isShared_4725_; uint8_t v_isSharedCheck_4729_; 
v_a_4722_ = lean_ctor_get(v___x_4704_, 0);
v_isSharedCheck_4729_ = !lean_is_exclusive(v___x_4704_);
if (v_isSharedCheck_4729_ == 0)
{
v___x_4724_ = v___x_4704_;
v_isShared_4725_ = v_isSharedCheck_4729_;
goto v_resetjp_4723_;
}
else
{
lean_inc(v_a_4722_);
lean_dec(v___x_4704_);
v___x_4724_ = lean_box(0);
v_isShared_4725_ = v_isSharedCheck_4729_;
goto v_resetjp_4723_;
}
v_resetjp_4723_:
{
lean_object* v___x_4727_; 
lean_inc(v_a_4722_);
if (v_isShared_4725_ == 0)
{
v___x_4727_ = v___x_4724_;
goto v_reusejp_4726_;
}
else
{
lean_object* v_reuseFailAlloc_4728_; 
v_reuseFailAlloc_4728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4728_, 0, v_a_4722_);
v___x_4727_ = v_reuseFailAlloc_4728_;
goto v_reusejp_4726_;
}
v_reusejp_4726_:
{
v___y_4663_ = v_a_4701_;
v___y_4664_ = v_a_4703_;
v___y_4665_ = v___y_4688_;
v___y_4666_ = v___y_4689_;
v___y_4667_ = v___y_4690_;
v___y_4668_ = v___y_4691_;
v___y_4669_ = v___y_4685_;
v___y_4670_ = v_a_4695_;
v___y_4671_ = v_a_4697_;
v___y_4672_ = v_a_4699_;
v___y_4673_ = v___x_4693_;
v___y_4674_ = v_e_4683_;
v___y_4675_ = v___y_4684_;
v___y_4676_ = v___y_4686_;
v___y_4677_ = v___y_4687_;
v___y_4678_ = v___x_4727_;
v_a_4679_ = v_a_4722_;
goto v___jp_4662_;
}
}
}
}
else
{
lean_object* v_a_4730_; lean_object* v___x_4732_; uint8_t v_isShared_4733_; uint8_t v_isSharedCheck_4737_; 
lean_dec(v_a_4701_);
lean_dec(v_a_4699_);
lean_dec(v_a_4697_);
lean_dec(v_a_4695_);
lean_dec_ref(v_e_4683_);
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4730_ = lean_ctor_get(v___x_4702_, 0);
v_isSharedCheck_4737_ = !lean_is_exclusive(v___x_4702_);
if (v_isSharedCheck_4737_ == 0)
{
v___x_4732_ = v___x_4702_;
v_isShared_4733_ = v_isSharedCheck_4737_;
goto v_resetjp_4731_;
}
else
{
lean_inc(v_a_4730_);
lean_dec(v___x_4702_);
v___x_4732_ = lean_box(0);
v_isShared_4733_ = v_isSharedCheck_4737_;
goto v_resetjp_4731_;
}
v_resetjp_4731_:
{
lean_object* v___x_4735_; 
if (v_isShared_4733_ == 0)
{
v___x_4735_ = v___x_4732_;
goto v_reusejp_4734_;
}
else
{
lean_object* v_reuseFailAlloc_4736_; 
v_reuseFailAlloc_4736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4736_, 0, v_a_4730_);
v___x_4735_ = v_reuseFailAlloc_4736_;
goto v_reusejp_4734_;
}
v_reusejp_4734_:
{
return v___x_4735_;
}
}
}
}
else
{
lean_object* v_a_4738_; lean_object* v___x_4740_; uint8_t v_isShared_4741_; uint8_t v_isSharedCheck_4745_; 
lean_dec(v_a_4699_);
lean_dec(v_a_4697_);
lean_dec(v_a_4695_);
lean_dec_ref(v_e_4683_);
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4738_ = lean_ctor_get(v___x_4700_, 0);
v_isSharedCheck_4745_ = !lean_is_exclusive(v___x_4700_);
if (v_isSharedCheck_4745_ == 0)
{
v___x_4740_ = v___x_4700_;
v_isShared_4741_ = v_isSharedCheck_4745_;
goto v_resetjp_4739_;
}
else
{
lean_inc(v_a_4738_);
lean_dec(v___x_4700_);
v___x_4740_ = lean_box(0);
v_isShared_4741_ = v_isSharedCheck_4745_;
goto v_resetjp_4739_;
}
v_resetjp_4739_:
{
lean_object* v___x_4743_; 
if (v_isShared_4741_ == 0)
{
v___x_4743_ = v___x_4740_;
goto v_reusejp_4742_;
}
else
{
lean_object* v_reuseFailAlloc_4744_; 
v_reuseFailAlloc_4744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4744_, 0, v_a_4738_);
v___x_4743_ = v_reuseFailAlloc_4744_;
goto v_reusejp_4742_;
}
v_reusejp_4742_:
{
return v___x_4743_;
}
}
}
}
else
{
lean_object* v_a_4746_; lean_object* v___x_4748_; uint8_t v_isShared_4749_; uint8_t v_isSharedCheck_4753_; 
lean_dec(v_a_4697_);
lean_dec(v_a_4695_);
lean_dec_ref(v_e_4683_);
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4746_ = lean_ctor_get(v___x_4698_, 0);
v_isSharedCheck_4753_ = !lean_is_exclusive(v___x_4698_);
if (v_isSharedCheck_4753_ == 0)
{
v___x_4748_ = v___x_4698_;
v_isShared_4749_ = v_isSharedCheck_4753_;
goto v_resetjp_4747_;
}
else
{
lean_inc(v_a_4746_);
lean_dec(v___x_4698_);
v___x_4748_ = lean_box(0);
v_isShared_4749_ = v_isSharedCheck_4753_;
goto v_resetjp_4747_;
}
v_resetjp_4747_:
{
lean_object* v___x_4751_; 
if (v_isShared_4749_ == 0)
{
v___x_4751_ = v___x_4748_;
goto v_reusejp_4750_;
}
else
{
lean_object* v_reuseFailAlloc_4752_; 
v_reuseFailAlloc_4752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4752_, 0, v_a_4746_);
v___x_4751_ = v_reuseFailAlloc_4752_;
goto v_reusejp_4750_;
}
v_reusejp_4750_:
{
return v___x_4751_;
}
}
}
}
else
{
lean_object* v_a_4754_; lean_object* v___x_4756_; uint8_t v_isShared_4757_; uint8_t v_isSharedCheck_4761_; 
lean_dec(v_a_4695_);
lean_dec_ref(v_e_4683_);
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4754_ = lean_ctor_get(v___x_4696_, 0);
v_isSharedCheck_4761_ = !lean_is_exclusive(v___x_4696_);
if (v_isSharedCheck_4761_ == 0)
{
v___x_4756_ = v___x_4696_;
v_isShared_4757_ = v_isSharedCheck_4761_;
goto v_resetjp_4755_;
}
else
{
lean_inc(v_a_4754_);
lean_dec(v___x_4696_);
v___x_4756_ = lean_box(0);
v_isShared_4757_ = v_isSharedCheck_4761_;
goto v_resetjp_4755_;
}
v_resetjp_4755_:
{
lean_object* v___x_4759_; 
if (v_isShared_4757_ == 0)
{
v___x_4759_ = v___x_4756_;
goto v_reusejp_4758_;
}
else
{
lean_object* v_reuseFailAlloc_4760_; 
v_reuseFailAlloc_4760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4760_, 0, v_a_4754_);
v___x_4759_ = v_reuseFailAlloc_4760_;
goto v_reusejp_4758_;
}
v_reusejp_4758_:
{
return v___x_4759_;
}
}
}
}
else
{
lean_object* v_a_4762_; lean_object* v___x_4764_; uint8_t v_isShared_4765_; uint8_t v_isSharedCheck_4769_; 
lean_dec_ref(v_e_4683_);
lean_dec(v_val_4519_);
lean_dec(v_val_4518_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4762_ = lean_ctor_get(v___x_4694_, 0);
v_isSharedCheck_4769_ = !lean_is_exclusive(v___x_4694_);
if (v_isSharedCheck_4769_ == 0)
{
v___x_4764_ = v___x_4694_;
v_isShared_4765_ = v_isSharedCheck_4769_;
goto v_resetjp_4763_;
}
else
{
lean_inc(v_a_4762_);
lean_dec(v___x_4694_);
v___x_4764_ = lean_box(0);
v_isShared_4765_ = v_isSharedCheck_4769_;
goto v_resetjp_4763_;
}
v_resetjp_4763_:
{
lean_object* v___x_4767_; 
if (v_isShared_4765_ == 0)
{
v___x_4767_ = v___x_4764_;
goto v_reusejp_4766_;
}
else
{
lean_object* v_reuseFailAlloc_4768_; 
v_reuseFailAlloc_4768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4768_, 0, v_a_4762_);
v___x_4767_ = v_reuseFailAlloc_4768_;
goto v_reusejp_4766_;
}
v_reusejp_4766_:
{
return v___x_4767_;
}
}
}
}
}
else
{
lean_object* v___x_4796_; 
lean_dec_ref_known(v_lb_4362_, 1);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
lean_dec(v_ub_4364_);
lean_dec(v_e_4363_);
v___x_4796_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_4796_;
}
}
v___jp_4380_:
{
lean_object* v___x_4396_; lean_object* v___x_4397_; lean_object* v___x_4398_; lean_object* v___x_4399_; lean_object* v___x_4400_; lean_object* v___x_4401_; lean_object* v___x_4402_; lean_object* v___x_4403_; 
lean_inc_n(v___y_4382_, 2);
v___x_4396_ = l_Lean_Meta_FVarSubst_apply(v___y_4382_, v___y_4387_);
lean_dec_ref(v___y_4387_);
v___x_4397_ = lean_mk_empty_array_with_capacity(v___x_4367_);
lean_dec(v___x_4367_);
lean_inc_ref(v___x_4397_);
v___x_4398_ = lean_array_push(v___x_4397_, v___x_4396_);
v___x_4399_ = l_Lean_Meta_FVarSubst_apply(v___y_4382_, v___y_4388_);
lean_dec_ref(v___y_4388_);
v___x_4400_ = lean_array_push(v___x_4397_, v___x_4399_);
v___x_4401_ = lean_box(v___x_4370_);
lean_inc(v___y_4390_);
v___x_4402_ = lean_apply_8(v___f_4369_, v___y_4392_, v___y_4395_, v___y_4382_, v___y_4390_, v___y_4389_, v___x_4398_, v___x_4400_, v___x_4401_);
v___x_4403_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(v___y_4390_, v___x_4402_, v___y_4391_, v___y_4386_, v___y_4393_, v___y_4394_, v___y_4381_, v___y_4384_, v___y_4385_, v___y_4383_);
return v___x_4403_;
}
v___jp_4404_:
{
lean_object* v___x_4417_; lean_object* v___x_4418_; 
v___x_4417_ = lean_box(0);
lean_inc(v_a_4365_);
v___x_4418_ = lp_mathlib_Lean_Elab_Tactic_getFVarIdsAt(v_a_4365_, v___x_4417_, v___y_4410_, v___y_4411_, v___y_4405_, v___y_4412_, v___y_4416_, v___y_4408_, v___y_4415_, v___y_4414_, v___y_4413_);
if (lean_obj_tag(v___x_4418_) == 0)
{
lean_object* v_a_4419_; lean_object* v___x_4420_; lean_object* v___x_4421_; lean_object* v___x_4422_; lean_object* v___x_4423_; uint8_t v___x_4424_; lean_object* v___x_4425_; 
v_a_4419_ = lean_ctor_get(v___x_4418_, 0);
lean_inc(v_a_4419_);
lean_dec_ref_known(v___x_4418_, 1);
lean_inc_ref(v___y_4409_);
v___x_4420_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4420_, 0, v___y_4409_);
lean_ctor_set(v___x_4420_, 1, v___x_4417_);
lean_ctor_set(v___x_4420_, 2, v___y_4366_);
v___x_4421_ = lean_mk_empty_array_with_capacity(v___x_4367_);
v___x_4422_ = lean_array_push(v___x_4421_, v___x_4420_);
v___x_4423_ = lean_box(0);
v___x_4424_ = 3;
v___x_4425_ = l_Lean_MVarId_generalizeHyp(v_a_4365_, v___x_4422_, v_a_4419_, v___x_4423_, v___x_4424_, v___y_4408_, v___y_4415_, v___y_4414_, v___y_4413_);
lean_dec(v_a_4419_);
if (lean_obj_tag(v___x_4425_) == 0)
{
lean_object* v_a_4426_; lean_object* v_snd_4427_; lean_object* v_fst_4428_; lean_object* v_fst_4429_; lean_object* v_snd_4430_; lean_object* v___x_4431_; lean_object* v___x_4432_; lean_object* v___x_4433_; uint8_t v___x_4434_; 
v_a_4426_ = lean_ctor_get(v___x_4425_, 0);
lean_inc(v_a_4426_);
lean_dec_ref_known(v___x_4425_, 1);
v_snd_4427_ = lean_ctor_get(v_a_4426_, 1);
lean_inc(v_snd_4427_);
v_fst_4428_ = lean_ctor_get(v_a_4426_, 0);
lean_inc(v_fst_4428_);
lean_dec(v_a_4426_);
v_fst_4429_ = lean_ctor_get(v_snd_4427_, 0);
lean_inc(v_fst_4429_);
v_snd_4430_ = lean_ctor_get(v_snd_4427_, 1);
lean_inc(v_snd_4430_);
lean_dec(v_snd_4427_);
v___x_4431_ = lean_box(0);
v___x_4432_ = lean_array_get(v___x_4431_, v_fst_4429_, v___x_4368_);
v___x_4433_ = lean_array_get_size(v_fst_4429_);
v___x_4434_ = lean_nat_dec_lt(v___x_4367_, v___x_4433_);
if (v___x_4434_ == 0)
{
lean_dec(v_fst_4429_);
v___y_4381_ = v___y_4408_;
v___y_4382_ = v_fst_4428_;
v___y_4383_ = v___y_4413_;
v___y_4384_ = v___y_4415_;
v___y_4385_ = v___y_4414_;
v___y_4386_ = v___y_4405_;
v___y_4387_ = v___y_4406_;
v___y_4388_ = v___y_4407_;
v___y_4389_ = v___y_4409_;
v___y_4390_ = v_snd_4430_;
v___y_4391_ = v___y_4411_;
v___y_4392_ = v___x_4432_;
v___y_4393_ = v___y_4412_;
v___y_4394_ = v___y_4416_;
v___y_4395_ = v___x_4417_;
goto v___jp_4380_;
}
else
{
lean_object* v___x_4435_; lean_object* v___x_4436_; 
v___x_4435_ = lean_array_fget(v_fst_4429_, v___x_4367_);
lean_dec(v_fst_4429_);
v___x_4436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4436_, 0, v___x_4435_);
v___y_4381_ = v___y_4408_;
v___y_4382_ = v_fst_4428_;
v___y_4383_ = v___y_4413_;
v___y_4384_ = v___y_4415_;
v___y_4385_ = v___y_4414_;
v___y_4386_ = v___y_4405_;
v___y_4387_ = v___y_4406_;
v___y_4388_ = v___y_4407_;
v___y_4389_ = v___y_4409_;
v___y_4390_ = v_snd_4430_;
v___y_4391_ = v___y_4411_;
v___y_4392_ = v___x_4432_;
v___y_4393_ = v___y_4412_;
v___y_4394_ = v___y_4416_;
v___y_4395_ = v___x_4436_;
goto v___jp_4380_;
}
}
else
{
lean_object* v_a_4437_; lean_object* v___x_4439_; uint8_t v_isShared_4440_; uint8_t v_isSharedCheck_4444_; 
lean_dec_ref(v___y_4409_);
lean_dec_ref(v___y_4407_);
lean_dec_ref(v___y_4406_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
v_a_4437_ = lean_ctor_get(v___x_4425_, 0);
v_isSharedCheck_4444_ = !lean_is_exclusive(v___x_4425_);
if (v_isSharedCheck_4444_ == 0)
{
v___x_4439_ = v___x_4425_;
v_isShared_4440_ = v_isSharedCheck_4444_;
goto v_resetjp_4438_;
}
else
{
lean_inc(v_a_4437_);
lean_dec(v___x_4425_);
v___x_4439_ = lean_box(0);
v_isShared_4440_ = v_isSharedCheck_4444_;
goto v_resetjp_4438_;
}
v_resetjp_4438_:
{
lean_object* v___x_4442_; 
if (v_isShared_4440_ == 0)
{
v___x_4442_ = v___x_4439_;
goto v_reusejp_4441_;
}
else
{
lean_object* v_reuseFailAlloc_4443_; 
v_reuseFailAlloc_4443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4443_, 0, v_a_4437_);
v___x_4442_ = v_reuseFailAlloc_4443_;
goto v_reusejp_4441_;
}
v_reusejp_4441_:
{
return v___x_4442_;
}
}
}
}
else
{
lean_object* v_a_4445_; lean_object* v___x_4447_; uint8_t v_isShared_4448_; uint8_t v_isSharedCheck_4452_; 
lean_dec_ref(v___y_4409_);
lean_dec_ref(v___y_4407_);
lean_dec_ref(v___y_4406_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
v_a_4445_ = lean_ctor_get(v___x_4418_, 0);
v_isSharedCheck_4452_ = !lean_is_exclusive(v___x_4418_);
if (v_isSharedCheck_4452_ == 0)
{
v___x_4447_ = v___x_4418_;
v_isShared_4448_ = v_isSharedCheck_4452_;
goto v_resetjp_4446_;
}
else
{
lean_inc(v_a_4445_);
lean_dec(v___x_4418_);
v___x_4447_ = lean_box(0);
v_isShared_4448_ = v_isSharedCheck_4452_;
goto v_resetjp_4446_;
}
v_resetjp_4446_:
{
lean_object* v___x_4450_; 
if (v_isShared_4448_ == 0)
{
v___x_4450_ = v___x_4447_;
goto v_reusejp_4449_;
}
else
{
lean_object* v_reuseFailAlloc_4451_; 
v_reuseFailAlloc_4451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4451_, 0, v_a_4445_);
v___x_4450_ = v_reuseFailAlloc_4451_;
goto v_reusejp_4449_;
}
v_reusejp_4449_:
{
return v___x_4450_;
}
}
}
}
v___jp_4453_:
{
if (lean_obj_tag(v___y_4466_) == 0)
{
lean_dec_ref_known(v___y_4466_, 1);
v___y_4405_ = v___y_4454_;
v___y_4406_ = v___y_4455_;
v___y_4407_ = v___y_4456_;
v___y_4408_ = v___y_4457_;
v___y_4409_ = v___y_4459_;
v___y_4410_ = v___y_4458_;
v___y_4411_ = v___y_4460_;
v___y_4412_ = v___y_4461_;
v___y_4413_ = v___y_4465_;
v___y_4414_ = v___y_4464_;
v___y_4415_ = v___y_4463_;
v___y_4416_ = v___y_4462_;
goto v___jp_4404_;
}
else
{
lean_dec_ref(v___y_4459_);
lean_dec_ref(v___y_4456_);
lean_dec_ref(v___y_4455_);
lean_dec_ref(v___f_4369_);
lean_dec(v___x_4367_);
lean_dec(v___y_4366_);
lean_dec(v_a_4365_);
return v___y_4466_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___boxed(lean_object** _args){
lean_object* v_lb_4797_ = _args[0];
lean_object* v_e_4798_ = _args[1];
lean_object* v_ub_4799_ = _args[2];
lean_object* v_a_4800_ = _args[3];
lean_object* v___y_4801_ = _args[4];
lean_object* v___x_4802_ = _args[5];
lean_object* v___x_4803_ = _args[6];
lean_object* v___f_4804_ = _args[7];
lean_object* v___x_4805_ = _args[8];
lean_object* v___y_4806_ = _args[9];
lean_object* v___y_4807_ = _args[10];
lean_object* v___y_4808_ = _args[11];
lean_object* v___y_4809_ = _args[12];
lean_object* v___y_4810_ = _args[13];
lean_object* v___y_4811_ = _args[14];
lean_object* v___y_4812_ = _args[15];
lean_object* v___y_4813_ = _args[16];
lean_object* v___y_4814_ = _args[17];
_start:
{
uint8_t v___x_38054__boxed_4815_; lean_object* v_res_4816_; 
v___x_38054__boxed_4815_ = lean_unbox(v___x_4805_);
v_res_4816_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2(v_lb_4797_, v_e_4798_, v_ub_4799_, v_a_4800_, v___y_4801_, v___x_4802_, v___x_4803_, v___f_4804_, v___x_38054__boxed_4815_, v___y_4806_, v___y_4807_, v___y_4808_, v___y_4809_, v___y_4810_, v___y_4811_, v___y_4812_, v___y_4813_);
lean_dec(v___y_4813_);
lean_dec_ref(v___y_4812_);
lean_dec(v___y_4811_);
lean_dec_ref(v___y_4810_);
lean_dec(v___y_4809_);
lean_dec_ref(v___y_4808_);
lean_dec(v___y_4807_);
lean_dec_ref(v___y_4806_);
lean_dec(v___x_4803_);
return v_res_4816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1(lean_object* v_x_4828_, lean_object* v_a_4829_, lean_object* v_a_4830_, lean_object* v_a_4831_, lean_object* v_a_4832_, lean_object* v_a_4833_, lean_object* v_a_4834_, lean_object* v_a_4835_, lean_object* v_a_4836_){
_start:
{
lean_object* v___x_4838_; uint8_t v___x_4839_; 
v___x_4838_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_intervalCases___closed__1));
lean_inc(v_x_4828_);
v___x_4839_ = l_Lean_Syntax_isOfKind(v_x_4828_, v___x_4838_);
if (v___x_4839_ == 0)
{
lean_object* v___x_4840_; 
lean_dec(v_x_4828_);
v___x_4840_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_4840_;
}
else
{
lean_object* v___x_4841_; lean_object* v___x_4842_; lean_object* v___y_4844_; lean_object* v___y_4845_; lean_object* v___y_4846_; lean_object* v___y_4847_; lean_object* v___y_4848_; lean_object* v___y_4849_; lean_object* v___y_4850_; lean_object* v___y_4851_; lean_object* v___y_4852_; lean_object* v___y_4853_; lean_object* v___y_4854_; lean_object* v___y_4855_; lean_object* v___y_4856_; lean_object* v___y_4857_; lean_object* v___y_4858_; lean_object* v___y_4863_; lean_object* v___y_4864_; lean_object* v___y_4865_; lean_object* v___y_4866_; lean_object* v___y_4867_; lean_object* v___y_4868_; lean_object* v___y_4869_; lean_object* v___y_4870_; lean_object* v___y_4871_; lean_object* v___y_4872_; lean_object* v___y_4873_; lean_object* v___y_4874_; lean_object* v___y_4875_; lean_object* v___y_4876_; lean_object* v___y_4877_; lean_object* v___y_4880_; lean_object* v___y_4881_; lean_object* v___y_4882_; lean_object* v___y_4883_; lean_object* v___y_4884_; lean_object* v___y_4885_; lean_object* v___y_4886_; lean_object* v___y_4887_; lean_object* v___y_4888_; lean_object* v___y_4889_; lean_object* v___y_4890_; lean_object* v___y_4891_; lean_object* v___y_4892_; lean_object* v___y_4895_; lean_object* v___y_4896_; lean_object* v___y_4897_; lean_object* v___y_4898_; lean_object* v___y_4899_; lean_object* v___y_4900_; lean_object* v___y_4901_; lean_object* v___y_4902_; lean_object* v___y_4903_; lean_object* v___y_4904_; lean_object* v___y_4905_; lean_object* v_lb_4906_; lean_object* v_ub_4907_; lean_object* v_h_4931_; lean_object* v_e_4932_; lean_object* v___y_4933_; lean_object* v___y_4934_; lean_object* v___y_4935_; lean_object* v___y_4936_; lean_object* v___y_4937_; lean_object* v___y_4938_; lean_object* v___y_4939_; lean_object* v___y_4940_; lean_object* v___x_4955_; lean_object* v_h_4957_; lean_object* v___y_4958_; lean_object* v___y_4959_; lean_object* v___y_4960_; lean_object* v___y_4961_; lean_object* v___y_4962_; lean_object* v___y_4963_; lean_object* v___y_4964_; lean_object* v___y_4965_; uint8_t v___x_4969_; 
v___x_4841_ = lean_unsigned_to_nat(0u);
v___x_4842_ = lean_unsigned_to_nat(1u);
v___x_4955_ = l_Lean_Syntax_getArg(v_x_4828_, v___x_4842_);
v___x_4969_ = l_Lean_Syntax_isNone(v___x_4955_);
if (v___x_4969_ == 0)
{
lean_object* v___x_4970_; uint8_t v___x_4971_; 
v___x_4970_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_4955_);
v___x_4971_ = l_Lean_Syntax_matchesNull(v___x_4955_, v___x_4970_);
if (v___x_4971_ == 0)
{
lean_object* v___x_4972_; 
lean_dec(v___x_4955_);
lean_dec(v_x_4828_);
v___x_4972_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_4972_;
}
else
{
lean_object* v___x_4973_; uint8_t v___x_4974_; 
v___x_4973_ = l_Lean_Syntax_getArg(v___x_4955_, v___x_4841_);
v___x_4974_ = l_Lean_Syntax_isNone(v___x_4973_);
if (v___x_4974_ == 0)
{
uint8_t v___x_4975_; 
lean_inc(v___x_4973_);
v___x_4975_ = l_Lean_Syntax_matchesNull(v___x_4973_, v___x_4970_);
if (v___x_4975_ == 0)
{
lean_object* v___x_4976_; 
lean_dec(v___x_4973_);
lean_dec(v___x_4955_);
lean_dec(v_x_4828_);
v___x_4976_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_4976_;
}
else
{
lean_object* v_h_4977_; lean_object* v___x_4978_; 
v_h_4977_ = l_Lean_Syntax_getArg(v___x_4973_, v___x_4841_);
lean_dec(v___x_4973_);
v___x_4978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4978_, 0, v_h_4977_);
v_h_4957_ = v___x_4978_;
v___y_4958_ = v_a_4829_;
v___y_4959_ = v_a_4830_;
v___y_4960_ = v_a_4831_;
v___y_4961_ = v_a_4832_;
v___y_4962_ = v_a_4833_;
v___y_4963_ = v_a_4834_;
v___y_4964_ = v_a_4835_;
v___y_4965_ = v_a_4836_;
goto v___jp_4956_;
}
}
else
{
lean_object* v___x_4979_; 
lean_dec(v___x_4973_);
v___x_4979_ = lean_box(0);
v_h_4957_ = v___x_4979_;
v___y_4958_ = v_a_4829_;
v___y_4959_ = v_a_4830_;
v___y_4960_ = v_a_4831_;
v___y_4961_ = v_a_4832_;
v___y_4962_ = v_a_4833_;
v___y_4963_ = v_a_4834_;
v___y_4964_ = v_a_4835_;
v___y_4965_ = v_a_4836_;
goto v___jp_4956_;
}
}
}
else
{
lean_object* v___x_4980_; 
lean_dec(v___x_4955_);
v___x_4980_ = lean_box(0);
v_h_4931_ = v___x_4980_;
v_e_4932_ = v___x_4980_;
v___y_4933_ = v_a_4829_;
v___y_4934_ = v_a_4830_;
v___y_4935_ = v_a_4831_;
v___y_4936_ = v_a_4832_;
v___y_4937_ = v_a_4833_;
v___y_4938_ = v_a_4834_;
v___y_4939_ = v_a_4835_;
v___y_4940_ = v_a_4836_;
goto v___jp_4930_;
}
v___jp_4843_:
{
lean_object* v___x_4859_; lean_object* v___y_4860_; lean_object* v___x_4861_; 
v___x_4859_ = lean_box(v___x_4839_);
v___y_4860_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__2___boxed), 18, 9);
lean_closure_set(v___y_4860_, 0, v___y_4844_);
lean_closure_set(v___y_4860_, 1, v___y_4845_);
lean_closure_set(v___y_4860_, 2, v___y_4846_);
lean_closure_set(v___y_4860_, 3, v___y_4847_);
lean_closure_set(v___y_4860_, 4, v___y_4858_);
lean_closure_set(v___y_4860_, 5, v___x_4842_);
lean_closure_set(v___y_4860_, 6, v___x_4841_);
lean_closure_set(v___y_4860_, 7, v___y_4848_);
lean_closure_set(v___y_4860_, 8, v___x_4859_);
v___x_4861_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__1___redArg(v___y_4852_, v___y_4860_, v___y_4850_, v___y_4855_, v___y_4851_, v___y_4853_, v___y_4856_, v___y_4849_, v___y_4854_, v___y_4857_);
return v___x_4861_;
}
v___jp_4862_:
{
lean_object* v___x_4878_; 
v___x_4878_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4878_, 0, v___y_4877_);
v___y_4844_ = v___y_4863_;
v___y_4845_ = v___y_4864_;
v___y_4846_ = v___y_4865_;
v___y_4847_ = v___y_4866_;
v___y_4848_ = v___y_4867_;
v___y_4849_ = v___y_4868_;
v___y_4850_ = v___y_4869_;
v___y_4851_ = v___y_4870_;
v___y_4852_ = v___y_4871_;
v___y_4853_ = v___y_4872_;
v___y_4854_ = v___y_4874_;
v___y_4855_ = v___y_4873_;
v___y_4856_ = v___y_4875_;
v___y_4857_ = v___y_4876_;
v___y_4858_ = v___x_4878_;
goto v___jp_4843_;
}
v___jp_4879_:
{
lean_object* v___x_4893_; 
v___x_4893_ = lean_box(0);
lean_inc(v___y_4883_);
v___y_4844_ = v___y_4880_;
v___y_4845_ = v___y_4881_;
v___y_4846_ = v___y_4882_;
v___y_4847_ = v___y_4883_;
v___y_4848_ = v___y_4884_;
v___y_4849_ = v___y_4885_;
v___y_4850_ = v___y_4886_;
v___y_4851_ = v___y_4887_;
v___y_4852_ = v___y_4883_;
v___y_4853_ = v___y_4888_;
v___y_4854_ = v___y_4890_;
v___y_4855_ = v___y_4889_;
v___y_4856_ = v___y_4891_;
v___y_4857_ = v___y_4892_;
v___y_4858_ = v___x_4893_;
goto v___jp_4843_;
}
v___jp_4894_:
{
lean_object* v___x_4908_; 
v___x_4908_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4899_, v___y_4896_, v___y_4898_, v___y_4895_, v___y_4904_);
if (lean_obj_tag(v___x_4908_) == 0)
{
if (lean_obj_tag(v___y_4900_) == 0)
{
lean_object* v_a_4909_; 
v_a_4909_ = lean_ctor_get(v___x_4908_, 0);
lean_inc(v_a_4909_);
lean_dec_ref_known(v___x_4908_, 1);
v___y_4880_ = v_lb_4906_;
v___y_4881_ = v___y_4897_;
v___y_4882_ = v_ub_4907_;
v___y_4883_ = v_a_4909_;
v___y_4884_ = v___y_4902_;
v___y_4885_ = v___y_4898_;
v___y_4886_ = v___y_4901_;
v___y_4887_ = v___y_4905_;
v___y_4888_ = v___y_4903_;
v___y_4889_ = v___y_4899_;
v___y_4890_ = v___y_4895_;
v___y_4891_ = v___y_4896_;
v___y_4892_ = v___y_4904_;
goto v___jp_4879_;
}
else
{
lean_object* v_val_4910_; 
v_val_4910_ = lean_ctor_get(v___y_4900_, 0);
lean_inc(v_val_4910_);
lean_dec_ref_known(v___y_4900_, 1);
if (lean_obj_tag(v_val_4910_) == 0)
{
lean_object* v_a_4911_; 
v_a_4911_ = lean_ctor_get(v___x_4908_, 0);
lean_inc(v_a_4911_);
lean_dec_ref_known(v___x_4908_, 1);
v___y_4880_ = v_lb_4906_;
v___y_4881_ = v___y_4897_;
v___y_4882_ = v_ub_4907_;
v___y_4883_ = v_a_4911_;
v___y_4884_ = v___y_4902_;
v___y_4885_ = v___y_4898_;
v___y_4886_ = v___y_4901_;
v___y_4887_ = v___y_4905_;
v___y_4888_ = v___y_4903_;
v___y_4889_ = v___y_4899_;
v___y_4890_ = v___y_4895_;
v___y_4891_ = v___y_4896_;
v___y_4892_ = v___y_4904_;
goto v___jp_4879_;
}
else
{
lean_object* v_a_4912_; lean_object* v_val_4913_; lean_object* v___x_4914_; uint8_t v___x_4915_; 
v_a_4912_ = lean_ctor_get(v___x_4908_, 0);
lean_inc(v_a_4912_);
lean_dec_ref_known(v___x_4908_, 1);
v_val_4913_ = lean_ctor_get(v_val_4910_, 0);
lean_inc_n(v_val_4913_, 2);
lean_dec_ref_known(v_val_4910_, 1);
v___x_4914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__2));
v___x_4915_ = l_Lean_Syntax_isOfKind(v_val_4913_, v___x_4914_);
if (v___x_4915_ == 0)
{
lean_object* v___x_4916_; 
lean_dec(v_val_4913_);
v___x_4916_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__4));
lean_inc(v_a_4912_);
v___y_4863_ = v_lb_4906_;
v___y_4864_ = v___y_4897_;
v___y_4865_ = v_ub_4907_;
v___y_4866_ = v_a_4912_;
v___y_4867_ = v___y_4902_;
v___y_4868_ = v___y_4898_;
v___y_4869_ = v___y_4901_;
v___y_4870_ = v___y_4905_;
v___y_4871_ = v_a_4912_;
v___y_4872_ = v___y_4903_;
v___y_4873_ = v___y_4899_;
v___y_4874_ = v___y_4895_;
v___y_4875_ = v___y_4896_;
v___y_4876_ = v___y_4904_;
v___y_4877_ = v___x_4916_;
goto v___jp_4862_;
}
else
{
lean_object* v___x_4917_; lean_object* v___x_4918_; uint8_t v___x_4919_; 
v___x_4917_ = l_Lean_Syntax_getArg(v_val_4913_, v___x_4841_);
lean_dec(v_val_4913_);
v___x_4918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__6));
lean_inc(v___x_4917_);
v___x_4919_ = l_Lean_Syntax_isOfKind(v___x_4917_, v___x_4918_);
if (v___x_4919_ == 0)
{
lean_object* v___x_4920_; 
lean_dec(v___x_4917_);
v___x_4920_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___closed__4));
lean_inc(v_a_4912_);
v___y_4863_ = v_lb_4906_;
v___y_4864_ = v___y_4897_;
v___y_4865_ = v_ub_4907_;
v___y_4866_ = v_a_4912_;
v___y_4867_ = v___y_4902_;
v___y_4868_ = v___y_4898_;
v___y_4869_ = v___y_4901_;
v___y_4870_ = v___y_4905_;
v___y_4871_ = v_a_4912_;
v___y_4872_ = v___y_4903_;
v___y_4873_ = v___y_4899_;
v___y_4874_ = v___y_4895_;
v___y_4875_ = v___y_4896_;
v___y_4876_ = v___y_4904_;
v___y_4877_ = v___x_4920_;
goto v___jp_4862_;
}
else
{
lean_object* v___x_4921_; 
v___x_4921_ = l_Lean_TSyntax_getId(v___x_4917_);
lean_dec(v___x_4917_);
lean_inc(v_a_4912_);
v___y_4863_ = v_lb_4906_;
v___y_4864_ = v___y_4897_;
v___y_4865_ = v_ub_4907_;
v___y_4866_ = v_a_4912_;
v___y_4867_ = v___y_4902_;
v___y_4868_ = v___y_4898_;
v___y_4869_ = v___y_4901_;
v___y_4870_ = v___y_4905_;
v___y_4871_ = v_a_4912_;
v___y_4872_ = v___y_4903_;
v___y_4873_ = v___y_4899_;
v___y_4874_ = v___y_4895_;
v___y_4875_ = v___y_4896_;
v___y_4876_ = v___y_4904_;
v___y_4877_ = v___x_4921_;
goto v___jp_4862_;
}
}
}
}
}
else
{
lean_object* v_a_4922_; lean_object* v___x_4924_; uint8_t v_isShared_4925_; uint8_t v_isSharedCheck_4929_; 
lean_dec(v_ub_4907_);
lean_dec(v_lb_4906_);
lean_dec_ref(v___y_4902_);
lean_dec(v___y_4900_);
lean_dec(v___y_4897_);
v_a_4922_ = lean_ctor_get(v___x_4908_, 0);
v_isSharedCheck_4929_ = !lean_is_exclusive(v___x_4908_);
if (v_isSharedCheck_4929_ == 0)
{
v___x_4924_ = v___x_4908_;
v_isShared_4925_ = v_isSharedCheck_4929_;
goto v_resetjp_4923_;
}
else
{
lean_inc(v_a_4922_);
lean_dec(v___x_4908_);
v___x_4924_ = lean_box(0);
v_isShared_4925_ = v_isSharedCheck_4929_;
goto v_resetjp_4923_;
}
v_resetjp_4923_:
{
lean_object* v___x_4927_; 
if (v_isShared_4925_ == 0)
{
v___x_4927_ = v___x_4924_;
goto v_reusejp_4926_;
}
else
{
lean_object* v_reuseFailAlloc_4928_; 
v_reuseFailAlloc_4928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4928_, 0, v_a_4922_);
v___x_4927_ = v_reuseFailAlloc_4928_;
goto v_reusejp_4926_;
}
v_reusejp_4926_:
{
return v___x_4927_;
}
}
}
}
v___jp_4930_:
{
lean_object* v___x_4941_; lean_object* v___f_4942_; lean_object* v___x_4943_; lean_object* v___x_4944_; uint8_t v___x_4945_; 
v___x_4941_ = lean_box(v___x_4839_);
lean_inc(v_h_4931_);
v___f_4942_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___lam__0___boxed), 19, 2);
lean_closure_set(v___f_4942_, 0, v___x_4941_);
lean_closure_set(v___f_4942_, 1, v_h_4931_);
v___x_4943_ = lean_unsigned_to_nat(2u);
v___x_4944_ = l_Lean_Syntax_getArg(v_x_4828_, v___x_4943_);
lean_dec(v_x_4828_);
v___x_4945_ = l_Lean_Syntax_isNone(v___x_4944_);
if (v___x_4945_ == 0)
{
lean_object* v___x_4946_; uint8_t v___x_4947_; 
v___x_4946_ = lean_unsigned_to_nat(4u);
lean_inc(v___x_4944_);
v___x_4947_ = l_Lean_Syntax_matchesNull(v___x_4944_, v___x_4946_);
if (v___x_4947_ == 0)
{
lean_object* v___x_4948_; 
lean_dec(v___x_4944_);
lean_dec_ref(v___f_4942_);
lean_dec(v_e_4932_);
lean_dec(v_h_4931_);
v___x_4948_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__0___redArg();
return v___x_4948_;
}
else
{
lean_object* v_lb_4949_; lean_object* v___x_4950_; lean_object* v_ub_4951_; lean_object* v___x_4952_; lean_object* v___x_4953_; 
v_lb_4949_ = l_Lean_Syntax_getArg(v___x_4944_, v___x_4842_);
v___x_4950_ = lean_unsigned_to_nat(3u);
v_ub_4951_ = l_Lean_Syntax_getArg(v___x_4944_, v___x_4950_);
lean_dec(v___x_4944_);
v___x_4952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4952_, 0, v_lb_4949_);
v___x_4953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4953_, 0, v_ub_4951_);
v___y_4895_ = v___y_4939_;
v___y_4896_ = v___y_4937_;
v___y_4897_ = v_e_4932_;
v___y_4898_ = v___y_4938_;
v___y_4899_ = v___y_4934_;
v___y_4900_ = v_h_4931_;
v___y_4901_ = v___y_4933_;
v___y_4902_ = v___f_4942_;
v___y_4903_ = v___y_4936_;
v___y_4904_ = v___y_4940_;
v___y_4905_ = v___y_4935_;
v_lb_4906_ = v___x_4952_;
v_ub_4907_ = v___x_4953_;
goto v___jp_4894_;
}
}
else
{
lean_object* v___x_4954_; 
lean_dec(v___x_4944_);
v___x_4954_ = lean_box(0);
v___y_4895_ = v___y_4939_;
v___y_4896_ = v___y_4937_;
v___y_4897_ = v_e_4932_;
v___y_4898_ = v___y_4938_;
v___y_4899_ = v___y_4934_;
v___y_4900_ = v_h_4931_;
v___y_4901_ = v___y_4933_;
v___y_4902_ = v___f_4942_;
v___y_4903_ = v___y_4936_;
v___y_4904_ = v___y_4940_;
v___y_4905_ = v___y_4935_;
v_lb_4906_ = v___x_4954_;
v_ub_4907_ = v___x_4954_;
goto v___jp_4894_;
}
}
v___jp_4956_:
{
lean_object* v_e_4966_; lean_object* v___x_4967_; lean_object* v___x_4968_; 
v_e_4966_ = l_Lean_Syntax_getArg(v___x_4955_, v___x_4842_);
lean_dec(v___x_4955_);
v___x_4967_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4967_, 0, v_h_4957_);
v___x_4968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4968_, 0, v_e_4966_);
v_h_4931_ = v___x_4967_;
v_e_4932_ = v___x_4968_;
v___y_4933_ = v___y_4958_;
v___y_4934_ = v___y_4959_;
v___y_4935_ = v___y_4960_;
v___y_4936_ = v___y_4961_;
v___y_4937_ = v___y_4962_;
v___y_4938_ = v___y_4963_;
v___y_4939_ = v___y_4964_;
v___y_4940_ = v___y_4965_;
goto v___jp_4930_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1___boxed(lean_object* v_x_4981_, lean_object* v_a_4982_, lean_object* v_a_4983_, lean_object* v_a_4984_, lean_object* v_a_4985_, lean_object* v_a_4986_, lean_object* v_a_4987_, lean_object* v_a_4988_, lean_object* v_a_4989_, lean_object* v_a_4990_){
_start:
{
lean_object* v_res_4991_; 
v_res_4991_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1(v_x_4981_, v_a_4982_, v_a_4983_, v_a_4984_, v_a_4985_, v_a_4986_, v_a_4987_, v_a_4988_, v_a_4989_);
lean_dec(v_a_4989_);
lean_dec_ref(v_a_4988_);
lean_dec(v_a_4987_);
lean_dec_ref(v_a_4986_);
lean_dec(v_a_4985_);
lean_dec_ref(v_a_4984_);
lean_dec(v_a_4983_);
lean_dec_ref(v_a_4982_);
return v_res_4991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4(lean_object* v_00_u03b1_4992_, lean_object* v_msg_4993_, lean_object* v___y_4994_, lean_object* v___y_4995_, lean_object* v___y_4996_, lean_object* v___y_4997_, lean_object* v___y_4998_, lean_object* v___y_4999_, lean_object* v___y_5000_, lean_object* v___y_5001_){
_start:
{
lean_object* v___x_5003_; 
v___x_5003_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___redArg(v_msg_4993_, v___y_4998_, v___y_4999_, v___y_5000_, v___y_5001_);
return v___x_5003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4___boxed(lean_object* v_00_u03b1_5004_, lean_object* v_msg_5005_, lean_object* v___y_5006_, lean_object* v___y_5007_, lean_object* v___y_5008_, lean_object* v___y_5009_, lean_object* v___y_5010_, lean_object* v___y_5011_, lean_object* v___y_5012_, lean_object* v___y_5013_, lean_object* v___y_5014_){
_start:
{
lean_object* v_res_5015_; 
v_res_5015_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__4(v_00_u03b1_5004_, v_msg_5005_, v___y_5006_, v___y_5007_, v___y_5008_, v___y_5009_, v___y_5010_, v___y_5011_, v___y_5012_, v___y_5013_);
lean_dec(v___y_5013_);
lean_dec_ref(v___y_5012_);
lean_dec(v___y_5011_);
lean_dec_ref(v___y_5010_);
lean_dec(v___y_5009_);
lean_dec_ref(v___y_5008_);
lean_dec(v___y_5007_);
lean_dec_ref(v___y_5006_);
return v_res_5015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6(lean_object* v_00_u03b1_5016_, lean_object* v_ref_5017_, lean_object* v_msg_5018_, lean_object* v___y_5019_, lean_object* v___y_5020_, lean_object* v___y_5021_, lean_object* v___y_5022_, lean_object* v___y_5023_, lean_object* v___y_5024_, lean_object* v___y_5025_, lean_object* v___y_5026_){
_start:
{
lean_object* v___x_5028_; 
v___x_5028_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___redArg(v_ref_5017_, v_msg_5018_, v___y_5019_, v___y_5020_, v___y_5021_, v___y_5022_, v___y_5023_, v___y_5024_, v___y_5025_, v___y_5026_);
return v___x_5028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6___boxed(lean_object* v_00_u03b1_5029_, lean_object* v_ref_5030_, lean_object* v_msg_5031_, lean_object* v___y_5032_, lean_object* v___y_5033_, lean_object* v___y_5034_, lean_object* v___y_5035_, lean_object* v___y_5036_, lean_object* v___y_5037_, lean_object* v___y_5038_, lean_object* v___y_5039_, lean_object* v___y_5040_){
_start:
{
lean_object* v_res_5041_; 
v_res_5041_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic___aux__Mathlib__Tactic__IntervalCases______elabRules__Mathlib__Tactic__intervalCases__1_spec__6(v_00_u03b1_5029_, v_ref_5030_, v_msg_5031_, v___y_5032_, v___y_5033_, v___y_5034_, v___y_5035_, v___y_5036_, v___y_5037_, v___y_5038_, v___y_5039_);
lean_dec(v___y_5039_);
lean_dec_ref(v___y_5038_);
lean_dec(v___y_5037_);
lean_dec_ref(v___y_5036_);
lean_dec(v___y_5035_);
lean_dec_ref(v___y_5034_);
lean_dec(v___y_5033_);
lean_dec_ref(v___y_5032_);
lean_dec(v_ref_5030_);
return v_res_5041_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_IntervalCases(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_IntervalCases(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_intervalCases = _init_lp_mathlib_Mathlib_Tactic_intervalCases();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_intervalCases);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_IntervalCases(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_IntervalCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_IntervalCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_IntervalCases(builtin);
}
#ifdef __cplusplus
}
#endif
