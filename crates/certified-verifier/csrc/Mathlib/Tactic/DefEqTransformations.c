// Lean compiler output
// Module: Mathlib.Tactic.DefEqTransformations
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Conv.Basic
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
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_expr_equal(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_replaceTargetDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_LocalDecl_index(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_withReverted___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Expr_isLambda(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate(lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* lean_expr_abstract(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_etaExpandedStrict_x3f(lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_change(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_FVarId_getValue_x3f___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_kabstract(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_findDecl_x3f___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getProjectionFnInfo_x3f(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_ExprStructEq_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
uint8_t l_Lean_ExprStructEq_beq(lean_object*, lean_object*);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConst(lean_object*);
size_t lean_ptr_addr(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_unfoldProjInst_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_betaReduce(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Elab_Tactic_getFVarIds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_Meta_reduce(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
uint8_t l_Lean_isStructure(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_getLhs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_changeLhs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "given type"};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__1;
static const lean_string_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "\nis not definitionally equal to"};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_changeLocalDecl_x27_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_changeLocalDecl_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "unexpected auxiliary target"};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__2;
static lean_once_cell_t lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "changeLocalDecl"};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(138, 31, 202, 231, 182, 71, 213, 201)}};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__1 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__1_value;
static const lean_array_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__2 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__2_value;
static const lean_string_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "local variable "};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__3 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4;
static const lean_string_object lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = " is not present in local context"};
static const lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__5 = (const lean_object*)&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticWhnf__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(115, 153, 74, 13, 145, 162, 115, 46)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "whnf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__15;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__16;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticWhnf____;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "betaReduceStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 208, 186, 183, 59, 224, 98, 68)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "beta_reduce"};
static const lean_object* lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_betaReduceStx;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "convBeta_reduce"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__0_value),LEAN_SCALAR_PTR_LITERAL(60, 176, 152, 135, 214, 198, 129, 190)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_convBeta__reduce = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticReduce__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(77, 183, 4, 136, 160, 219, 119, 160)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "reduce"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticReduce____;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__19___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19_spec__20___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "transform"};
static const lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___closed__0_value;
static const lean_array_object lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__0(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__10(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_unfoldFVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldFVars___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__19(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19_spec__20(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = " has no value to refold"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_refoldFVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_refoldFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "refoldLetStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(190, 79, 50, 141, 208, 19, 62, 142)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "refold_let"};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__4_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__11_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__16;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_refoldLetStx;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__refoldLetStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__refoldLetStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "convRefold_let___"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(112, 175, 104, 86, 247, 122, 15, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_convRefold__let______ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convRefold__let________1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convRefold__let________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_unfoldProjs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_unfoldProjs___lam__1___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjs___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "unfoldProjsStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 189, 244, 196, 232, 184, 119, 59)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "unfold_projs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjsStx;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "convUnfold_projs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 235, 229, 253, 146, 242, 92, 90)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_convUnfold__projs = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_unfoldProjs___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_etaReduceAll___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_etaReduceAll___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceAll___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "etaReduceStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(180, 231, 249, 180, 153, 47, 31, 44)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eta_reduce"};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceStx;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "convEta_reduce"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__0_value),LEAN_SCALAR_PTR_LITERAL(156, 35, 71, 234, 48, 21, 127, 107)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_convEta__reduce = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_etaReduceAll___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__2(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "etaExpandStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(11, 166, 189, 20, 114, 129, 62, 184)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eta_expand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandStx;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convEta__expand___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "convEta_expand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__expand___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 63, 149, 106, 139, 246, 229, 107)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__expand___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__expand___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_convEta__expand = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__expand___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_etaExpandAll___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_getProjectedExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_getProjectedExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_etaStruct_x3f___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_whnfR___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_etaStructAll___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_etaStructAll___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructAll___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_etaStructStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "etaStructStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaStructStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(169, 165, 134, 178, 236, 20, 132, 145)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_etaStructStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "eta_struct"};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaStructStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_etaStructStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_etaStructStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_etaStructStx___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_etaStructStx___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_etaStructStx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_etaStructStx___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructStx;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_convEta__struct___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "convEta_struct"};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__struct___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__0_value),LEAN_SCALAR_PTR_LITERAL(120, 177, 15, 27, 243, 16, 62, 36)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_convEta__struct___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_convEta__struct___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_convEta__struct = (const lean_object*)&lp_mathlib_Mathlib_Tactic_convEta__struct___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_etaStructAll___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___redArg(lean_object* v_mvarId_1_, lean_object* v_x_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1_, v_x_2_, v___y_3_, v___y_4_, v___y_5_, v___y_6_);
if (lean_obj_tag(v___x_8_) == 0)
{
lean_object* v_a_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_16_; 
v_a_9_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_16_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_16_ == 0)
{
v___x_11_ = v___x_8_;
v_isShared_12_ = v_isSharedCheck_16_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_a_9_);
lean_dec(v___x_8_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_16_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___x_14_; 
if (v_isShared_12_ == 0)
{
v___x_14_ = v___x_11_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v_a_9_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
else
{
lean_object* v_a_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_24_; 
v_a_17_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_24_ == 0)
{
v___x_19_ = v___x_8_;
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_a_17_);
lean_dec(v___x_8_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___x_22_; 
if (v_isShared_20_ == 0)
{
v___x_22_ = v___x_19_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_a_17_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___redArg___boxed(lean_object* v_mvarId_25_, lean_object* v_x_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___redArg(v_mvarId_25_, v_x_26_, v___y_27_, v___y_28_, v___y_29_, v___y_30_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
lean_dec(v___y_28_);
lean_dec_ref(v___y_27_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1(lean_object* v_00_u03b1_33_, lean_object* v_mvarId_34_, lean_object* v_x_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___redArg(v_mvarId_34_, v_x_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___boxed(lean_object* v_00_u03b1_42_, lean_object* v_mvarId_43_, lean_object* v_x_44_, lean_object* v___y_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1(v_00_u03b1_42_, v_mvarId_43_, v_x_44_, v___y_45_, v___y_46_, v___y_47_, v___y_48_);
lean_dec(v___y_48_);
lean_dec_ref(v___y_47_);
lean_dec(v___y_46_);
lean_dec_ref(v___y_45_);
return v_res_50_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__1(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_52_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__0));
v___x_53_ = l_Lean_stringToMessageData(v___x_52_);
return v___x_53_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__3(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__2));
v___x_56_ = l_Lean_stringToMessageData(v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0(uint8_t v_checkDefEq_57_, lean_object* v_typeNew_58_, lean_object* v___x_59_, lean_object* v_mvarId_60_, lean_object* v_typeOld_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
if (v_checkDefEq_57_ == 0)
{
lean_object* v___x_67_; lean_object* v___x_68_; 
lean_dec_ref(v_typeOld_61_);
lean_dec(v_mvarId_60_);
lean_dec(v___x_59_);
lean_dec_ref(v_typeNew_58_);
v___x_67_ = lean_box(0);
v___x_68_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
return v___x_68_;
}
else
{
lean_object* v___x_69_; 
lean_inc_ref(v_typeOld_61_);
lean_inc_ref(v_typeNew_58_);
v___x_69_ = l_Lean_Meta_isExprDefEq(v_typeNew_58_, v_typeOld_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
if (lean_obj_tag(v___x_69_) == 0)
{
lean_object* v_a_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_88_; 
v_a_70_ = lean_ctor_get(v___x_69_, 0);
v_isSharedCheck_88_ = !lean_is_exclusive(v___x_69_);
if (v_isSharedCheck_88_ == 0)
{
v___x_72_ = v___x_69_;
v_isShared_73_ = v_isSharedCheck_88_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_a_70_);
lean_dec(v___x_69_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_88_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
uint8_t v___x_74_; 
v___x_74_ = lean_unbox(v_a_70_);
lean_dec(v_a_70_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
lean_del_object(v___x_72_);
v___x_75_ = lean_obj_once(&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__1, &lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__1_once, _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__1);
v___x_76_ = l_Lean_indentExpr(v_typeNew_58_);
v___x_77_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_77_, 0, v___x_75_);
lean_ctor_set(v___x_77_, 1, v___x_76_);
v___x_78_ = lean_obj_once(&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__3, &lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__3_once, _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___closed__3);
v___x_79_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_77_);
lean_ctor_set(v___x_79_, 1, v___x_78_);
v___x_80_ = l_Lean_indentExpr(v_typeOld_61_);
v___x_81_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_79_);
lean_ctor_set(v___x_81_, 1, v___x_80_);
v___x_82_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
v___x_83_ = l_Lean_Meta_throwTacticEx___redArg(v___x_59_, v_mvarId_60_, v___x_82_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
return v___x_83_;
}
else
{
lean_object* v___x_84_; lean_object* v___x_86_; 
lean_dec_ref(v_typeOld_61_);
lean_dec(v_mvarId_60_);
lean_dec(v___x_59_);
lean_dec_ref(v_typeNew_58_);
v___x_84_ = lean_box(0);
if (v_isShared_73_ == 0)
{
lean_ctor_set(v___x_72_, 0, v___x_84_);
v___x_86_ = v___x_72_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v___x_84_);
v___x_86_ = v_reuseFailAlloc_87_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
return v___x_86_;
}
}
}
}
else
{
lean_object* v_a_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_96_; 
lean_dec_ref(v_typeOld_61_);
lean_dec(v_mvarId_60_);
lean_dec(v___x_59_);
lean_dec_ref(v_typeNew_58_);
v_a_89_ = lean_ctor_get(v___x_69_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_69_);
if (v_isSharedCheck_96_ == 0)
{
v___x_91_ = v___x_69_;
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_a_89_);
lean_dec(v___x_69_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_96_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_94_; 
if (v_isShared_92_ == 0)
{
v___x_94_ = v___x_91_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_a_89_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___boxed(lean_object* v_checkDefEq_97_, lean_object* v_typeNew_98_, lean_object* v___x_99_, lean_object* v_mvarId_100_, lean_object* v_typeOld_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_){
_start:
{
uint8_t v_checkDefEq_boxed_107_; lean_object* v_res_108_; 
v_checkDefEq_boxed_107_ = lean_unbox(v_checkDefEq_97_);
v_res_108_ = lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0(v_checkDefEq_boxed_107_, v_typeNew_98_, v___x_99_, v_mvarId_100_, v_typeOld_101_, v___y_102_, v___y_103_, v___y_104_, v___y_105_);
lean_dec(v___y_105_);
lean_dec_ref(v___y_104_);
lean_dec(v___y_103_);
lean_dec_ref(v___y_102_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_changeLocalDecl_x27_spec__0(size_t v_sz_109_, size_t v_i_110_, lean_object* v_bs_111_){
_start:
{
uint8_t v___x_112_; 
v___x_112_ = lean_usize_dec_lt(v_i_110_, v_sz_109_);
if (v___x_112_ == 0)
{
return v_bs_111_;
}
else
{
lean_object* v_v_113_; lean_object* v___x_114_; lean_object* v_bs_x27_115_; lean_object* v___x_116_; size_t v___x_117_; size_t v___x_118_; lean_object* v___x_119_; 
v_v_113_ = lean_array_uget(v_bs_111_, v_i_110_);
v___x_114_ = lean_unsigned_to_nat(0u);
v_bs_x27_115_ = lean_array_uset(v_bs_111_, v_i_110_, v___x_114_);
v___x_116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_116_, 0, v_v_113_);
v___x_117_ = ((size_t)1ULL);
v___x_118_ = lean_usize_add(v_i_110_, v___x_117_);
v___x_119_ = lean_array_uset(v_bs_x27_115_, v_i_110_, v___x_116_);
v_i_110_ = v___x_118_;
v_bs_111_ = v___x_119_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_changeLocalDecl_x27_spec__0___boxed(lean_object* v_sz_121_, lean_object* v_i_122_, lean_object* v_bs_123_){
_start:
{
size_t v_sz_boxed_124_; size_t v_i_boxed_125_; lean_object* v_res_126_; 
v_sz_boxed_124_ = lean_unbox_usize(v_sz_121_);
lean_dec(v_sz_121_);
v_i_boxed_125_ = lean_unbox_usize(v_i_122_);
lean_dec(v_i_122_);
v_res_126_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_changeLocalDecl_x27_spec__0(v_sz_boxed_124_, v_i_boxed_125_, v_bs_123_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__1(lean_object* v_mvarId_127_, lean_object* v_fvars_128_, lean_object* v_targetNew_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = l_Lean_MVarId_replaceTargetDefEq(v_mvarId_127_, v_targetNew_129_, v___y_130_, v___y_131_, v___y_132_, v___y_133_);
if (lean_obj_tag(v___x_135_) == 0)
{
lean_object* v_a_136_; lean_object* v___x_138_; uint8_t v_isShared_139_; uint8_t v_isSharedCheck_149_; 
v_a_136_ = lean_ctor_get(v___x_135_, 0);
v_isSharedCheck_149_ = !lean_is_exclusive(v___x_135_);
if (v_isSharedCheck_149_ == 0)
{
v___x_138_ = v___x_135_;
v_isShared_139_ = v_isSharedCheck_149_;
goto v_resetjp_137_;
}
else
{
lean_inc(v_a_136_);
lean_dec(v___x_135_);
v___x_138_ = lean_box(0);
v_isShared_139_ = v_isSharedCheck_149_;
goto v_resetjp_137_;
}
v_resetjp_137_:
{
lean_object* v___x_140_; size_t v_sz_141_; size_t v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_147_; 
v___x_140_ = lean_box(0);
v_sz_141_ = lean_array_size(v_fvars_128_);
v___x_142_ = ((size_t)0ULL);
v___x_143_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_MVarId_changeLocalDecl_x27_spec__0(v_sz_141_, v___x_142_, v_fvars_128_);
v___x_144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
lean_ctor_set(v___x_144_, 1, v_a_136_);
v___x_145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_140_);
lean_ctor_set(v___x_145_, 1, v___x_144_);
if (v_isShared_139_ == 0)
{
lean_ctor_set(v___x_138_, 0, v___x_145_);
v___x_147_ = v___x_138_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_148_; 
v_reuseFailAlloc_148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_148_, 0, v___x_145_);
v___x_147_ = v_reuseFailAlloc_148_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
return v___x_147_;
}
}
}
else
{
lean_object* v_a_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_157_; 
lean_dec_ref(v_fvars_128_);
v_a_150_ = lean_ctor_get(v___x_135_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v___x_135_);
if (v_isSharedCheck_157_ == 0)
{
v___x_152_ = v___x_135_;
v_isShared_153_ = v_isSharedCheck_157_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_a_150_);
lean_dec(v___x_135_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_157_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v___x_155_; 
if (v_isShared_153_ == 0)
{
v___x_155_ = v___x_152_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v_a_150_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__1___boxed(lean_object* v_mvarId_158_, lean_object* v_fvars_159_, lean_object* v_targetNew_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__1(v_mvarId_158_, v_fvars_159_, v_targetNew_160_, v___y_161_, v___y_162_, v___y_163_, v___y_164_);
lean_dec(v___y_164_);
lean_dec_ref(v___y_163_);
lean_dec(v___y_162_);
lean_dec_ref(v___y_161_);
return v_res_166_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__2(void){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__1));
v___x_171_ = l_Lean_MessageData_ofFormat(v___x_170_);
return v___x_171_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__3(void){
_start:
{
lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_172_ = lean_obj_once(&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__2, &lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__2_once, _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__2);
v___x_173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2(lean_object* v_mvarId_174_, lean_object* v___f_175_, lean_object* v_typeNew_176_, lean_object* v___f_177_, lean_object* v___x_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_){
_start:
{
lean_object* v___x_184_; 
lean_inc(v_mvarId_174_);
v___x_184_ = l_Lean_MVarId_getType(v_mvarId_174_, v___y_179_, v___y_180_, v___y_181_, v___y_182_);
if (lean_obj_tag(v___x_184_) == 0)
{
lean_object* v_a_185_; 
v_a_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_a_185_);
lean_dec_ref_known(v___x_184_, 1);
switch(lean_obj_tag(v_a_185_))
{
case 7:
{
lean_object* v_binderName_186_; lean_object* v_binderType_187_; lean_object* v_body_188_; uint8_t v_binderInfo_189_; lean_object* v___x_190_; 
lean_dec(v___x_178_);
lean_dec(v_mvarId_174_);
v_binderName_186_ = lean_ctor_get(v_a_185_, 0);
lean_inc(v_binderName_186_);
v_binderType_187_ = lean_ctor_get(v_a_185_, 1);
lean_inc_ref(v_binderType_187_);
v_body_188_ = lean_ctor_get(v_a_185_, 2);
lean_inc_ref(v_body_188_);
v_binderInfo_189_ = lean_ctor_get_uint8(v_a_185_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_a_185_, 3);
lean_inc(v___y_182_);
lean_inc_ref(v___y_181_);
lean_inc(v___y_180_);
lean_inc_ref(v___y_179_);
v___x_190_ = lean_apply_6(v___f_175_, v_binderType_187_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, lean_box(0));
if (lean_obj_tag(v___x_190_) == 0)
{
lean_object* v___x_191_; lean_object* v___x_192_; 
lean_dec_ref_known(v___x_190_, 1);
v___x_191_ = l_Lean_Expr_forallE___override(v_binderName_186_, v_typeNew_176_, v_body_188_, v_binderInfo_189_);
v___x_192_ = lean_apply_6(v___f_177_, v___x_191_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, lean_box(0));
return v___x_192_;
}
else
{
lean_object* v_a_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_200_; 
lean_dec_ref(v_body_188_);
lean_dec(v_binderName_186_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec_ref(v___f_177_);
lean_dec_ref(v_typeNew_176_);
v_a_193_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_200_ == 0)
{
v___x_195_ = v___x_190_;
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_a_193_);
lean_dec(v___x_190_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_200_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_198_; 
if (v_isShared_196_ == 0)
{
v___x_198_ = v___x_195_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v_a_193_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
case 8:
{
lean_object* v_declName_201_; lean_object* v_type_202_; lean_object* v_value_203_; lean_object* v_body_204_; uint8_t v_nondep_205_; lean_object* v___x_206_; 
lean_dec(v___x_178_);
lean_dec(v_mvarId_174_);
v_declName_201_ = lean_ctor_get(v_a_185_, 0);
lean_inc(v_declName_201_);
v_type_202_ = lean_ctor_get(v_a_185_, 1);
lean_inc_ref(v_type_202_);
v_value_203_ = lean_ctor_get(v_a_185_, 2);
lean_inc_ref(v_value_203_);
v_body_204_ = lean_ctor_get(v_a_185_, 3);
lean_inc_ref(v_body_204_);
v_nondep_205_ = lean_ctor_get_uint8(v_a_185_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_a_185_, 4);
lean_inc(v___y_182_);
lean_inc_ref(v___y_181_);
lean_inc(v___y_180_);
lean_inc_ref(v___y_179_);
v___x_206_ = lean_apply_6(v___f_175_, v_type_202_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, lean_box(0));
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v___x_207_; lean_object* v___x_208_; 
lean_dec_ref_known(v___x_206_, 1);
v___x_207_ = l_Lean_Expr_letE___override(v_declName_201_, v_typeNew_176_, v_value_203_, v_body_204_, v_nondep_205_);
v___x_208_ = lean_apply_6(v___f_177_, v___x_207_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, lean_box(0));
return v___x_208_;
}
else
{
lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
lean_dec_ref(v_body_204_);
lean_dec_ref(v_value_203_);
lean_dec(v_declName_201_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec_ref(v___f_177_);
lean_dec_ref(v_typeNew_176_);
v_a_209_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_216_ == 0)
{
v___x_211_ = v___x_206_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_206_);
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
default: 
{
lean_object* v___x_217_; lean_object* v___x_218_; 
lean_dec(v_a_185_);
lean_dec_ref(v___f_177_);
lean_dec_ref(v_typeNew_176_);
lean_dec_ref(v___f_175_);
v___x_217_ = lean_obj_once(&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__3, &lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__3_once, _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___closed__3);
v___x_218_ = l_Lean_Meta_throwTacticEx___redArg(v___x_178_, v_mvarId_174_, v___x_217_, v___y_179_, v___y_180_, v___y_181_, v___y_182_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
return v___x_218_;
}
}
}
else
{
lean_object* v_a_219_; lean_object* v___x_221_; uint8_t v_isShared_222_; uint8_t v_isSharedCheck_226_; 
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___x_178_);
lean_dec_ref(v___f_177_);
lean_dec_ref(v_typeNew_176_);
lean_dec_ref(v___f_175_);
lean_dec(v_mvarId_174_);
v_a_219_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_226_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_226_ == 0)
{
v___x_221_ = v___x_184_;
v_isShared_222_ = v_isSharedCheck_226_;
goto v_resetjp_220_;
}
else
{
lean_inc(v_a_219_);
lean_dec(v___x_184_);
v___x_221_ = lean_box(0);
v_isShared_222_ = v_isSharedCheck_226_;
goto v_resetjp_220_;
}
v_resetjp_220_:
{
lean_object* v___x_224_; 
if (v_isShared_222_ == 0)
{
v___x_224_ = v___x_221_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v_a_219_);
v___x_224_ = v_reuseFailAlloc_225_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
return v___x_224_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___boxed(lean_object* v_mvarId_227_, lean_object* v___f_228_, lean_object* v_typeNew_229_, lean_object* v___f_230_, lean_object* v___x_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2(v_mvarId_227_, v___f_228_, v_typeNew_229_, v___f_230_, v___x_231_, v___y_232_, v___y_233_, v___y_234_, v___y_235_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__3(uint8_t v_checkDefEq_238_, lean_object* v_typeNew_239_, lean_object* v___x_240_, lean_object* v_mvarId_241_, lean_object* v_fvars_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v___x_248_; lean_object* v___f_249_; lean_object* v___f_250_; lean_object* v___f_251_; lean_object* v___x_252_; 
v___x_248_ = lean_box(v_checkDefEq_238_);
lean_inc_n(v_mvarId_241_, 3);
lean_inc(v___x_240_);
lean_inc_ref(v_typeNew_239_);
v___f_249_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__0___boxed), 10, 4);
lean_closure_set(v___f_249_, 0, v___x_248_);
lean_closure_set(v___f_249_, 1, v_typeNew_239_);
lean_closure_set(v___f_249_, 2, v___x_240_);
lean_closure_set(v___f_249_, 3, v_mvarId_241_);
v___f_250_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__1___boxed), 8, 2);
lean_closure_set(v___f_250_, 0, v_mvarId_241_);
lean_closure_set(v___f_250_, 1, v_fvars_242_);
v___f_251_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__2___boxed), 10, 5);
lean_closure_set(v___f_251_, 0, v_mvarId_241_);
lean_closure_set(v___f_251_, 1, v___f_249_);
lean_closure_set(v___f_251_, 2, v_typeNew_239_);
lean_closure_set(v___f_251_, 3, v___f_250_);
lean_closure_set(v___f_251_, 4, v___x_240_);
v___x_252_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_changeLocalDecl_x27_spec__1___redArg(v_mvarId_241_, v___f_251_, v___y_243_, v___y_244_, v___y_245_, v___y_246_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__3___boxed(lean_object* v_checkDefEq_253_, lean_object* v_typeNew_254_, lean_object* v___x_255_, lean_object* v_mvarId_256_, lean_object* v_fvars_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
uint8_t v_checkDefEq_boxed_263_; lean_object* v_res_264_; 
v_checkDefEq_boxed_263_ = lean_unbox(v_checkDefEq_253_);
v_res_264_ = lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__3(v_checkDefEq_boxed_263_, v_typeNew_254_, v___x_255_, v_mvarId_256_, v_fvars_257_, v___y_258_, v___y_259_, v___y_260_, v___y_261_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(lean_object* v_val_265_, lean_object* v_as_266_, size_t v_i_267_, size_t v_stop_268_, lean_object* v_b_269_){
_start:
{
lean_object* v___y_271_; uint8_t v___x_275_; 
v___x_275_ = lean_usize_dec_eq(v_i_267_, v_stop_268_);
if (v___x_275_ == 0)
{
lean_object* v___x_276_; 
v___x_276_ = lean_array_uget_borrowed(v_as_266_, v_i_267_);
if (lean_obj_tag(v___x_276_) == 0)
{
v___y_271_ = v_b_269_;
goto v___jp_270_;
}
else
{
lean_object* v_val_277_; lean_object* v___x_278_; lean_object* v___x_279_; uint8_t v___x_280_; 
v_val_277_ = lean_ctor_get(v___x_276_, 0);
v___x_278_ = l_Lean_LocalDecl_index(v_val_265_);
v___x_279_ = l_Lean_LocalDecl_index(v_val_277_);
v___x_280_ = lean_nat_dec_le(v___x_278_, v___x_279_);
lean_dec(v___x_279_);
lean_dec(v___x_278_);
if (v___x_280_ == 0)
{
v___y_271_ = v_b_269_;
goto v___jp_270_;
}
else
{
lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_281_ = l_Lean_LocalDecl_fvarId(v_val_277_);
v___x_282_ = lean_array_push(v_b_269_, v___x_281_);
v___y_271_ = v___x_282_;
goto v___jp_270_;
}
}
}
else
{
return v_b_269_;
}
v___jp_270_:
{
size_t v___x_272_; size_t v___x_273_; 
v___x_272_ = ((size_t)1ULL);
v___x_273_ = lean_usize_add(v_i_267_, v___x_272_);
v_i_267_ = v___x_273_;
v_b_269_ = v___y_271_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4___boxed(lean_object* v_val_283_, lean_object* v_as_284_, lean_object* v_i_285_, lean_object* v_stop_286_, lean_object* v_b_287_){
_start:
{
size_t v_i_boxed_288_; size_t v_stop_boxed_289_; lean_object* v_res_290_; 
v_i_boxed_288_ = lean_unbox_usize(v_i_285_);
lean_dec(v_i_285_);
v_stop_boxed_289_ = lean_unbox_usize(v_stop_286_);
lean_dec(v_stop_286_);
v_res_290_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_283_, v_as_284_, v_i_boxed_288_, v_stop_boxed_289_, v_b_287_);
lean_dec_ref(v_as_284_);
lean_dec_ref(v_val_283_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__5(lean_object* v_val_291_, lean_object* v_x_292_, lean_object* v_x_293_){
_start:
{
if (lean_obj_tag(v_x_292_) == 0)
{
lean_object* v_cs_294_; lean_object* v___x_295_; lean_object* v___x_296_; uint8_t v___x_297_; 
v_cs_294_ = lean_ctor_get(v_x_292_, 0);
v___x_295_ = lean_unsigned_to_nat(0u);
v___x_296_ = lean_array_get_size(v_cs_294_);
v___x_297_ = lean_nat_dec_lt(v___x_295_, v___x_296_);
if (v___x_297_ == 0)
{
return v_x_293_;
}
else
{
uint8_t v___x_298_; 
v___x_298_ = lean_nat_dec_le(v___x_296_, v___x_296_);
if (v___x_298_ == 0)
{
if (v___x_297_ == 0)
{
return v_x_293_;
}
else
{
size_t v___x_299_; size_t v___x_300_; lean_object* v___x_301_; 
v___x_299_ = ((size_t)0ULL);
v___x_300_ = lean_usize_of_nat(v___x_296_);
v___x_301_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4(v_val_291_, v_cs_294_, v___x_299_, v___x_300_, v_x_293_);
return v___x_301_;
}
}
else
{
size_t v___x_302_; size_t v___x_303_; lean_object* v___x_304_; 
v___x_302_ = ((size_t)0ULL);
v___x_303_ = lean_usize_of_nat(v___x_296_);
v___x_304_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4(v_val_291_, v_cs_294_, v___x_302_, v___x_303_, v_x_293_);
return v___x_304_;
}
}
}
else
{
lean_object* v_vs_305_; lean_object* v___x_306_; lean_object* v___x_307_; uint8_t v___x_308_; 
v_vs_305_ = lean_ctor_get(v_x_292_, 0);
v___x_306_ = lean_unsigned_to_nat(0u);
v___x_307_ = lean_array_get_size(v_vs_305_);
v___x_308_ = lean_nat_dec_lt(v___x_306_, v___x_307_);
if (v___x_308_ == 0)
{
return v_x_293_;
}
else
{
uint8_t v___x_309_; 
v___x_309_ = lean_nat_dec_le(v___x_307_, v___x_307_);
if (v___x_309_ == 0)
{
if (v___x_308_ == 0)
{
return v_x_293_;
}
else
{
size_t v___x_310_; size_t v___x_311_; lean_object* v___x_312_; 
v___x_310_ = ((size_t)0ULL);
v___x_311_ = lean_usize_of_nat(v___x_307_);
v___x_312_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_291_, v_vs_305_, v___x_310_, v___x_311_, v_x_293_);
return v___x_312_;
}
}
else
{
size_t v___x_313_; size_t v___x_314_; lean_object* v___x_315_; 
v___x_313_ = ((size_t)0ULL);
v___x_314_ = lean_usize_of_nat(v___x_307_);
v___x_315_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_291_, v_vs_305_, v___x_313_, v___x_314_, v_x_293_);
return v___x_315_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4(lean_object* v_val_316_, lean_object* v_as_317_, size_t v_i_318_, size_t v_stop_319_, lean_object* v_b_320_){
_start:
{
uint8_t v___x_321_; 
v___x_321_ = lean_usize_dec_eq(v_i_318_, v_stop_319_);
if (v___x_321_ == 0)
{
lean_object* v___x_322_; lean_object* v___x_323_; size_t v___x_324_; size_t v___x_325_; 
v___x_322_ = lean_array_uget_borrowed(v_as_317_, v_i_318_);
v___x_323_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__5(v_val_316_, v___x_322_, v_b_320_);
v___x_324_ = ((size_t)1ULL);
v___x_325_ = lean_usize_add(v_i_318_, v___x_324_);
v_i_318_ = v___x_325_;
v_b_320_ = v___x_323_;
goto _start;
}
else
{
return v_b_320_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4___boxed(lean_object* v_val_327_, lean_object* v_as_328_, lean_object* v_i_329_, lean_object* v_stop_330_, lean_object* v_b_331_){
_start:
{
size_t v_i_boxed_332_; size_t v_stop_boxed_333_; lean_object* v_res_334_; 
v_i_boxed_332_ = lean_unbox_usize(v_i_329_);
lean_dec(v_i_329_);
v_stop_boxed_333_ = lean_unbox_usize(v_stop_330_);
lean_dec(v_stop_330_);
v_res_334_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4(v_val_327_, v_as_328_, v_i_boxed_332_, v_stop_boxed_333_, v_b_331_);
lean_dec_ref(v_as_328_);
lean_dec_ref(v_val_327_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__5___boxed(lean_object* v_val_335_, lean_object* v_x_336_, lean_object* v_x_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__5(v_val_335_, v_x_336_, v_x_337_);
lean_dec_ref(v_x_336_);
lean_dec_ref(v_val_335_);
return v_res_338_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___closed__0(void){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3(lean_object* v_val_340_, lean_object* v_x_341_, size_t v_x_342_, size_t v_x_343_, lean_object* v_x_344_){
_start:
{
if (lean_obj_tag(v_x_341_) == 0)
{
lean_object* v_cs_345_; lean_object* v___x_346_; size_t v___x_347_; lean_object* v_j_348_; lean_object* v___x_349_; size_t v___x_350_; size_t v___x_351_; size_t v___x_352_; size_t v___x_353_; size_t v___x_354_; size_t v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; uint8_t v___x_360_; 
v_cs_345_ = lean_ctor_get(v_x_341_, 0);
v___x_346_ = lean_obj_once(&lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___closed__0, &lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___closed__0_once, _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___closed__0);
v___x_347_ = lean_usize_shift_right(v_x_342_, v_x_343_);
v_j_348_ = lean_usize_to_nat(v___x_347_);
v___x_349_ = lean_array_get_borrowed(v___x_346_, v_cs_345_, v_j_348_);
v___x_350_ = ((size_t)1ULL);
v___x_351_ = lean_usize_shift_left(v___x_350_, v_x_343_);
v___x_352_ = lean_usize_sub(v___x_351_, v___x_350_);
v___x_353_ = lean_usize_land(v_x_342_, v___x_352_);
v___x_354_ = ((size_t)5ULL);
v___x_355_ = lean_usize_sub(v_x_343_, v___x_354_);
v___x_356_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3(v_val_340_, v___x_349_, v___x_353_, v___x_355_, v_x_344_);
v___x_357_ = lean_unsigned_to_nat(1u);
v___x_358_ = lean_nat_add(v_j_348_, v___x_357_);
lean_dec(v_j_348_);
v___x_359_ = lean_array_get_size(v_cs_345_);
v___x_360_ = lean_nat_dec_lt(v___x_358_, v___x_359_);
if (v___x_360_ == 0)
{
lean_dec(v___x_358_);
return v___x_356_;
}
else
{
uint8_t v___x_361_; 
v___x_361_ = lean_nat_dec_le(v___x_359_, v___x_359_);
if (v___x_361_ == 0)
{
if (v___x_360_ == 0)
{
lean_dec(v___x_358_);
return v___x_356_;
}
else
{
size_t v___x_362_; size_t v___x_363_; lean_object* v___x_364_; 
v___x_362_ = lean_usize_of_nat(v___x_358_);
lean_dec(v___x_358_);
v___x_363_ = lean_usize_of_nat(v___x_359_);
v___x_364_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4(v_val_340_, v_cs_345_, v___x_362_, v___x_363_, v___x_356_);
return v___x_364_;
}
}
else
{
size_t v___x_365_; size_t v___x_366_; lean_object* v___x_367_; 
v___x_365_ = lean_usize_of_nat(v___x_358_);
lean_dec(v___x_358_);
v___x_366_ = lean_usize_of_nat(v___x_359_);
v___x_367_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3_spec__4(v_val_340_, v_cs_345_, v___x_365_, v___x_366_, v___x_356_);
return v___x_367_;
}
}
}
else
{
lean_object* v_vs_368_; lean_object* v___x_369_; lean_object* v___x_370_; uint8_t v___x_371_; 
v_vs_368_ = lean_ctor_get(v_x_341_, 0);
v___x_369_ = lean_usize_to_nat(v_x_342_);
v___x_370_ = lean_array_get_size(v_vs_368_);
v___x_371_ = lean_nat_dec_lt(v___x_369_, v___x_370_);
if (v___x_371_ == 0)
{
lean_dec(v___x_369_);
return v_x_344_;
}
else
{
uint8_t v___x_372_; 
v___x_372_ = lean_nat_dec_le(v___x_370_, v___x_370_);
if (v___x_372_ == 0)
{
if (v___x_371_ == 0)
{
lean_dec(v___x_369_);
return v_x_344_;
}
else
{
size_t v___x_373_; size_t v___x_374_; lean_object* v___x_375_; 
v___x_373_ = lean_usize_of_nat(v___x_369_);
lean_dec(v___x_369_);
v___x_374_ = lean_usize_of_nat(v___x_370_);
v___x_375_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_340_, v_vs_368_, v___x_373_, v___x_374_, v_x_344_);
return v___x_375_;
}
}
else
{
size_t v___x_376_; size_t v___x_377_; lean_object* v___x_378_; 
v___x_376_ = lean_usize_of_nat(v___x_369_);
lean_dec(v___x_369_);
v___x_377_ = lean_usize_of_nat(v___x_370_);
v___x_378_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_340_, v_vs_368_, v___x_376_, v___x_377_, v_x_344_);
return v___x_378_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3___boxed(lean_object* v_val_379_, lean_object* v_x_380_, lean_object* v_x_381_, lean_object* v_x_382_, lean_object* v_x_383_){
_start:
{
size_t v_x_4474__boxed_384_; size_t v_x_4475__boxed_385_; lean_object* v_res_386_; 
v_x_4474__boxed_384_ = lean_unbox_usize(v_x_381_);
lean_dec(v_x_381_);
v_x_4475__boxed_385_ = lean_unbox_usize(v_x_382_);
lean_dec(v_x_382_);
v_res_386_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3(v_val_379_, v_x_380_, v_x_4474__boxed_384_, v_x_4475__boxed_385_, v_x_383_);
lean_dec_ref(v_x_380_);
lean_dec_ref(v_val_379_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2(lean_object* v_val_387_, lean_object* v_t_388_, lean_object* v_init_389_, lean_object* v_start_390_){
_start:
{
lean_object* v___x_391_; uint8_t v___x_392_; 
v___x_391_ = lean_unsigned_to_nat(0u);
v___x_392_ = lean_nat_dec_eq(v_start_390_, v___x_391_);
if (v___x_392_ == 0)
{
lean_object* v_root_393_; lean_object* v_tail_394_; size_t v_shift_395_; lean_object* v_tailOff_396_; uint8_t v___x_397_; 
v_root_393_ = lean_ctor_get(v_t_388_, 0);
v_tail_394_ = lean_ctor_get(v_t_388_, 1);
v_shift_395_ = lean_ctor_get_usize(v_t_388_, 4);
v_tailOff_396_ = lean_ctor_get(v_t_388_, 3);
v___x_397_ = lean_nat_dec_le(v_tailOff_396_, v_start_390_);
if (v___x_397_ == 0)
{
size_t v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; uint8_t v___x_401_; 
v___x_398_ = lean_usize_of_nat(v_start_390_);
v___x_399_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__3(v_val_387_, v_root_393_, v___x_398_, v_shift_395_, v_init_389_);
v___x_400_ = lean_array_get_size(v_tail_394_);
v___x_401_ = lean_nat_dec_lt(v___x_391_, v___x_400_);
if (v___x_401_ == 0)
{
return v___x_399_;
}
else
{
uint8_t v___x_402_; 
v___x_402_ = lean_nat_dec_le(v___x_400_, v___x_400_);
if (v___x_402_ == 0)
{
if (v___x_401_ == 0)
{
return v___x_399_;
}
else
{
size_t v___x_403_; size_t v___x_404_; lean_object* v___x_405_; 
v___x_403_ = ((size_t)0ULL);
v___x_404_ = lean_usize_of_nat(v___x_400_);
v___x_405_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_387_, v_tail_394_, v___x_403_, v___x_404_, v___x_399_);
return v___x_405_;
}
}
else
{
size_t v___x_406_; size_t v___x_407_; lean_object* v___x_408_; 
v___x_406_ = ((size_t)0ULL);
v___x_407_ = lean_usize_of_nat(v___x_400_);
v___x_408_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_387_, v_tail_394_, v___x_406_, v___x_407_, v___x_399_);
return v___x_408_;
}
}
}
else
{
lean_object* v___x_409_; lean_object* v___x_410_; uint8_t v___x_411_; 
v___x_409_ = lean_nat_sub(v_start_390_, v_tailOff_396_);
v___x_410_ = lean_array_get_size(v_tail_394_);
v___x_411_ = lean_nat_dec_lt(v___x_409_, v___x_410_);
if (v___x_411_ == 0)
{
lean_dec(v___x_409_);
return v_init_389_;
}
else
{
uint8_t v___x_412_; 
v___x_412_ = lean_nat_dec_le(v___x_410_, v___x_410_);
if (v___x_412_ == 0)
{
if (v___x_411_ == 0)
{
lean_dec(v___x_409_);
return v_init_389_;
}
else
{
size_t v___x_413_; size_t v___x_414_; lean_object* v___x_415_; 
v___x_413_ = lean_usize_of_nat(v___x_409_);
lean_dec(v___x_409_);
v___x_414_ = lean_usize_of_nat(v___x_410_);
v___x_415_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_387_, v_tail_394_, v___x_413_, v___x_414_, v_init_389_);
return v___x_415_;
}
}
else
{
size_t v___x_416_; size_t v___x_417_; lean_object* v___x_418_; 
v___x_416_ = lean_usize_of_nat(v___x_409_);
lean_dec(v___x_409_);
v___x_417_ = lean_usize_of_nat(v___x_410_);
v___x_418_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_387_, v_tail_394_, v___x_416_, v___x_417_, v_init_389_);
return v___x_418_;
}
}
}
}
else
{
lean_object* v_root_419_; lean_object* v_tail_420_; lean_object* v___x_421_; lean_object* v___x_422_; uint8_t v___x_423_; 
v_root_419_ = lean_ctor_get(v_t_388_, 0);
v_tail_420_ = lean_ctor_get(v_t_388_, 1);
v___x_421_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__5(v_val_387_, v_root_419_, v_init_389_);
v___x_422_ = lean_array_get_size(v_tail_420_);
v___x_423_ = lean_nat_dec_lt(v___x_391_, v___x_422_);
if (v___x_423_ == 0)
{
return v___x_421_;
}
else
{
uint8_t v___x_424_; 
v___x_424_ = lean_nat_dec_le(v___x_422_, v___x_422_);
if (v___x_424_ == 0)
{
if (v___x_423_ == 0)
{
return v___x_421_;
}
else
{
size_t v___x_425_; size_t v___x_426_; lean_object* v___x_427_; 
v___x_425_ = ((size_t)0ULL);
v___x_426_ = lean_usize_of_nat(v___x_422_);
v___x_427_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_387_, v_tail_420_, v___x_425_, v___x_426_, v___x_421_);
return v___x_427_;
}
}
else
{
size_t v___x_428_; size_t v___x_429_; lean_object* v___x_430_; 
v___x_428_ = ((size_t)0ULL);
v___x_429_ = lean_usize_of_nat(v___x_422_);
v___x_430_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2_spec__4(v_val_387_, v_tail_420_, v___x_428_, v___x_429_, v___x_421_);
return v___x_430_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2___boxed(lean_object* v_val_431_, lean_object* v_t_432_, lean_object* v_init_433_, lean_object* v_start_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2(v_val_431_, v_t_432_, v_init_433_, v_start_434_);
lean_dec(v_start_434_);
lean_dec_ref(v_t_432_);
lean_dec_ref(v_val_431_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2(lean_object* v_val_436_, lean_object* v_lctx_437_, lean_object* v_init_438_, lean_object* v_start_439_){
_start:
{
lean_object* v_decls_440_; lean_object* v___x_441_; 
v_decls_440_ = lean_ctor_get(v_lctx_437_, 1);
v___x_441_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2_spec__2(v_val_436_, v_decls_440_, v_init_438_, v_start_439_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2___boxed(lean_object* v_val_442_, lean_object* v_lctx_443_, lean_object* v_init_444_, lean_object* v_start_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2(v_val_442_, v_lctx_443_, v_init_444_, v_start_445_);
lean_dec(v_start_445_);
lean_dec_ref(v_lctx_443_);
lean_dec_ref(v_val_442_);
return v_res_446_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4(void){
_start:
{
lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_453_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__3));
v___x_454_ = l_Lean_stringToMessageData(v___x_453_);
return v___x_454_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__6(void){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_456_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__5));
v___x_457_ = l_Lean_stringToMessageData(v___x_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27(lean_object* v_mvarId_458_, lean_object* v_fvarId_459_, lean_object* v_typeNew_460_, uint8_t v_checkDefEq_461_, lean_object* v_a_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_){
_start:
{
lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_467_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__1));
lean_inc(v_mvarId_458_);
v___x_468_ = l_Lean_MVarId_checkNotAssigned(v_mvarId_458_, v___x_467_, v_a_462_, v_a_463_, v_a_464_, v_a_465_);
if (lean_obj_tag(v___x_468_) == 0)
{
lean_object* v___x_469_; 
lean_dec_ref_known(v___x_468_, 1);
lean_inc(v_mvarId_458_);
v___x_469_ = l_Lean_MVarId_getDecl(v_mvarId_458_, v_a_462_, v_a_463_, v_a_464_, v_a_465_);
if (lean_obj_tag(v___x_469_) == 0)
{
lean_object* v_a_470_; lean_object* v_lctx_471_; lean_object* v___x_472_; 
v_a_470_ = lean_ctor_get(v___x_469_, 0);
lean_inc(v_a_470_);
lean_dec_ref_known(v___x_469_, 1);
v_lctx_471_ = lean_ctor_get(v_a_470_, 1);
lean_inc_ref_n(v_lctx_471_, 2);
lean_dec(v_a_470_);
lean_inc(v_fvarId_459_);
v___x_472_ = lean_local_ctx_find(v_lctx_471_, v_fvarId_459_);
if (lean_obj_tag(v___x_472_) == 1)
{
lean_object* v_val_473_; lean_object* v___x_474_; lean_object* v___f_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; uint8_t v___x_479_; lean_object* v___x_480_; 
lean_dec(v_fvarId_459_);
v_val_473_ = lean_ctor_get(v___x_472_, 0);
lean_inc(v_val_473_);
lean_dec_ref_known(v___x_472_, 1);
v___x_474_ = lean_box(v_checkDefEq_461_);
v___f_475_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___lam__3___boxed), 10, 3);
lean_closure_set(v___f_475_, 0, v___x_474_);
lean_closure_set(v___f_475_, 1, v_typeNew_460_);
lean_closure_set(v___f_475_, 2, v___x_467_);
v___x_476_ = lean_unsigned_to_nat(0u);
v___x_477_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__2));
v___x_478_ = lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_MVarId_changeLocalDecl_x27_spec__2(v_val_473_, v_lctx_471_, v___x_477_, v___x_476_);
lean_dec_ref(v_lctx_471_);
lean_dec(v_val_473_);
v___x_479_ = 0;
v___x_480_ = l_Lean_MVarId_withReverted___redArg(v_mvarId_458_, v___x_478_, v___f_475_, v___x_479_, v_a_462_, v_a_463_, v_a_464_, v_a_465_);
if (lean_obj_tag(v___x_480_) == 0)
{
lean_object* v_a_481_; lean_object* v___x_483_; uint8_t v_isShared_484_; uint8_t v_isSharedCheck_489_; 
v_a_481_ = lean_ctor_get(v___x_480_, 0);
v_isSharedCheck_489_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_489_ == 0)
{
v___x_483_ = v___x_480_;
v_isShared_484_ = v_isSharedCheck_489_;
goto v_resetjp_482_;
}
else
{
lean_inc(v_a_481_);
lean_dec(v___x_480_);
v___x_483_ = lean_box(0);
v_isShared_484_ = v_isSharedCheck_489_;
goto v_resetjp_482_;
}
v_resetjp_482_:
{
lean_object* v_snd_485_; lean_object* v___x_487_; 
v_snd_485_ = lean_ctor_get(v_a_481_, 1);
lean_inc(v_snd_485_);
lean_dec(v_a_481_);
if (v_isShared_484_ == 0)
{
lean_ctor_set(v___x_483_, 0, v_snd_485_);
v___x_487_ = v___x_483_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v_snd_485_);
v___x_487_ = v_reuseFailAlloc_488_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
return v___x_487_;
}
}
}
else
{
lean_object* v_a_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_497_; 
v_a_490_ = lean_ctor_get(v___x_480_, 0);
v_isSharedCheck_497_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_497_ == 0)
{
v___x_492_ = v___x_480_;
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_a_490_);
lean_dec(v___x_480_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_495_; 
if (v_isShared_493_ == 0)
{
v___x_495_ = v___x_492_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v_a_490_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
}
else
{
lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; 
lean_dec(v___x_472_);
lean_dec_ref(v_lctx_471_);
lean_dec_ref(v_typeNew_460_);
v___x_498_ = lean_obj_once(&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4, &lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4_once, _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4);
v___x_499_ = l_Lean_Expr_fvar___override(v_fvarId_459_);
v___x_500_ = l_Lean_MessageData_ofExpr(v___x_499_);
v___x_501_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_501_, 0, v___x_498_);
lean_ctor_set(v___x_501_, 1, v___x_500_);
v___x_502_ = lean_obj_once(&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__6, &lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__6_once, _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__6);
v___x_503_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_501_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
lean_inc(v_mvarId_458_);
v___x_504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_504_, 0, v_mvarId_458_);
v___x_505_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_505_, 0, v___x_503_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
v___x_506_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_506_, 0, v___x_505_);
v___x_507_ = l_Lean_Meta_throwTacticEx___redArg(v___x_467_, v_mvarId_458_, v___x_506_, v_a_462_, v_a_463_, v_a_464_, v_a_465_);
return v___x_507_;
}
}
else
{
lean_object* v_a_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_515_; 
lean_dec_ref(v_typeNew_460_);
lean_dec(v_fvarId_459_);
lean_dec(v_mvarId_458_);
v_a_508_ = lean_ctor_get(v___x_469_, 0);
v_isSharedCheck_515_ = !lean_is_exclusive(v___x_469_);
if (v_isSharedCheck_515_ == 0)
{
v___x_510_ = v___x_469_;
v_isShared_511_ = v_isSharedCheck_515_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_a_508_);
lean_dec(v___x_469_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_515_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v___x_513_; 
if (v_isShared_511_ == 0)
{
v___x_513_ = v___x_510_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v_a_508_);
v___x_513_ = v_reuseFailAlloc_514_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
return v___x_513_;
}
}
}
}
else
{
lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_523_; 
lean_dec_ref(v_typeNew_460_);
lean_dec(v_fvarId_459_);
lean_dec(v_mvarId_458_);
v_a_516_ = lean_ctor_get(v___x_468_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_468_);
if (v_isSharedCheck_523_ == 0)
{
v___x_518_ = v___x_468_;
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_468_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_521_; 
if (v_isShared_519_ == 0)
{
v___x_521_ = v___x_518_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_a_516_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_changeLocalDecl_x27___boxed(lean_object* v_mvarId_524_, lean_object* v_fvarId_525_, lean_object* v_typeNew_526_, lean_object* v_checkDefEq_527_, lean_object* v_a_528_, lean_object* v_a_529_, lean_object* v_a_530_, lean_object* v_a_531_, lean_object* v_a_532_){
_start:
{
uint8_t v_checkDefEq_boxed_533_; lean_object* v_res_534_; 
v_checkDefEq_boxed_533_ = lean_unbox(v_checkDefEq_527_);
v_res_534_ = lp_mathlib_Lean_MVarId_changeLocalDecl_x27(v_mvarId_524_, v_fvarId_525_, v_typeNew_526_, v_checkDefEq_boxed_533_, v_a_528_, v_a_529_, v_a_530_, v_a_531_);
lean_dec(v_a_531_);
lean_dec_ref(v_a_530_);
lean_dec(v_a_529_);
lean_dec_ref(v_a_528_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(lean_object* v_e_535_, lean_object* v___y_536_){
_start:
{
uint8_t v___x_538_; 
v___x_538_ = l_Lean_Expr_hasMVar(v_e_535_);
if (v___x_538_ == 0)
{
lean_object* v___x_539_; 
v___x_539_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_539_, 0, v_e_535_);
return v___x_539_;
}
else
{
lean_object* v___x_540_; lean_object* v_mctx_541_; lean_object* v___x_542_; lean_object* v_fst_543_; lean_object* v_snd_544_; lean_object* v___x_545_; lean_object* v_cache_546_; lean_object* v_zetaDeltaFVarIds_547_; lean_object* v_postponed_548_; lean_object* v_diag_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_558_; 
v___x_540_ = lean_st_ref_get(v___y_536_);
v_mctx_541_ = lean_ctor_get(v___x_540_, 0);
lean_inc_ref(v_mctx_541_);
lean_dec(v___x_540_);
v___x_542_ = l_Lean_instantiateMVarsCore(v_mctx_541_, v_e_535_);
v_fst_543_ = lean_ctor_get(v___x_542_, 0);
lean_inc(v_fst_543_);
v_snd_544_ = lean_ctor_get(v___x_542_, 1);
lean_inc(v_snd_544_);
lean_dec_ref(v___x_542_);
v___x_545_ = lean_st_ref_take(v___y_536_);
v_cache_546_ = lean_ctor_get(v___x_545_, 1);
v_zetaDeltaFVarIds_547_ = lean_ctor_get(v___x_545_, 2);
v_postponed_548_ = lean_ctor_get(v___x_545_, 3);
v_diag_549_ = lean_ctor_get(v___x_545_, 4);
v_isSharedCheck_558_ = !lean_is_exclusive(v___x_545_);
if (v_isSharedCheck_558_ == 0)
{
lean_object* v_unused_559_; 
v_unused_559_ = lean_ctor_get(v___x_545_, 0);
lean_dec(v_unused_559_);
v___x_551_ = v___x_545_;
v_isShared_552_ = v_isSharedCheck_558_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_diag_549_);
lean_inc(v_postponed_548_);
lean_inc(v_zetaDeltaFVarIds_547_);
lean_inc(v_cache_546_);
lean_dec(v___x_545_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_558_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_554_; 
if (v_isShared_552_ == 0)
{
lean_ctor_set(v___x_551_, 0, v_snd_544_);
v___x_554_ = v___x_551_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_557_; 
v_reuseFailAlloc_557_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_557_, 0, v_snd_544_);
lean_ctor_set(v_reuseFailAlloc_557_, 1, v_cache_546_);
lean_ctor_set(v_reuseFailAlloc_557_, 2, v_zetaDeltaFVarIds_547_);
lean_ctor_set(v_reuseFailAlloc_557_, 3, v_postponed_548_);
lean_ctor_set(v_reuseFailAlloc_557_, 4, v_diag_549_);
v___x_554_ = v_reuseFailAlloc_557_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_555_ = lean_st_ref_set(v___y_536_, v___x_554_);
v___x_556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_556_, 0, v_fst_543_);
return v___x_556_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg___boxed(lean_object* v_e_560_, lean_object* v___y_561_, lean_object* v___y_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(v_e_560_, v___y_561_);
lean_dec(v___y_561_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0(lean_object* v_e_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_){
_start:
{
lean_object* v___x_570_; 
v___x_570_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(v_e_564_, v___y_566_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___boxed(lean_object* v_e_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_){
_start:
{
lean_object* v_res_577_; 
v_res_577_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0(v_e_571_, v___y_572_, v___y_573_, v___y_574_, v___y_575_);
lean_dec(v___y_575_);
lean_dec_ref(v___y_574_);
lean_dec(v___y_573_);
lean_dec_ref(v___y_572_);
return v_res_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__0(lean_object* v_h_578_, lean_object* v_m_579_, uint8_t v_checkDefEq_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_){
_start:
{
lean_object* v_val_591_; lean_object* v___x_595_; 
v___x_595_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_582_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
if (lean_obj_tag(v___x_595_) == 0)
{
lean_object* v_a_596_; lean_object* v___x_597_; 
v_a_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc(v_a_596_);
lean_dec_ref_known(v___x_595_, 1);
lean_inc(v_h_578_);
v___x_597_ = l_Lean_FVarId_getType___redArg(v_h_578_, v___y_585_, v___y_587_, v___y_588_);
if (lean_obj_tag(v___x_597_) == 0)
{
lean_object* v_a_598_; lean_object* v___x_599_; lean_object* v_a_600_; lean_object* v___x_602_; uint8_t v_isShared_603_; uint8_t v_isSharedCheck_628_; 
v_a_598_ = lean_ctor_get(v___x_597_, 0);
lean_inc_n(v_a_598_, 2);
lean_dec_ref_known(v___x_597_, 1);
v___x_599_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(v_a_598_, v___y_586_);
v_a_600_ = lean_ctor_get(v___x_599_, 0);
v_isSharedCheck_628_ = !lean_is_exclusive(v___x_599_);
if (v_isSharedCheck_628_ == 0)
{
v___x_602_ = v___x_599_;
v_isShared_603_ = v_isSharedCheck_628_;
goto v_resetjp_601_;
}
else
{
lean_inc(v_a_600_);
lean_dec(v___x_599_);
v___x_602_ = lean_box(0);
v_isShared_603_ = v_isSharedCheck_628_;
goto v_resetjp_601_;
}
v_resetjp_601_:
{
lean_object* v___x_605_; 
lean_inc(v_h_578_);
if (v_isShared_603_ == 0)
{
lean_ctor_set_tag(v___x_602_, 1);
lean_ctor_set(v___x_602_, 0, v_h_578_);
v___x_605_ = v___x_602_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_627_; 
v_reuseFailAlloc_627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_627_, 0, v_h_578_);
v___x_605_ = v_reuseFailAlloc_627_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
lean_object* v___x_606_; 
lean_inc(v___y_588_);
lean_inc_ref(v___y_587_);
lean_inc(v___y_586_);
lean_inc_ref(v___y_585_);
v___x_606_ = lean_apply_7(v_m_579_, v___x_605_, v_a_600_, v___y_585_, v___y_586_, v___y_587_, v___y_588_, lean_box(0));
if (lean_obj_tag(v___x_606_) == 0)
{
lean_object* v_a_607_; uint8_t v___x_608_; 
v_a_607_ = lean_ctor_get(v___x_606_, 0);
lean_inc(v_a_607_);
lean_dec_ref_known(v___x_606_, 1);
v___x_608_ = lean_expr_equal(v_a_598_, v_a_607_);
lean_dec(v_a_598_);
if (v___x_608_ == 0)
{
lean_object* v___x_609_; 
v___x_609_ = lp_mathlib_Lean_MVarId_changeLocalDecl_x27(v_a_596_, v_h_578_, v_a_607_, v_checkDefEq_580_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
if (lean_obj_tag(v___x_609_) == 0)
{
lean_object* v_a_610_; 
v_a_610_ = lean_ctor_get(v___x_609_, 0);
lean_inc(v_a_610_);
lean_dec_ref_known(v___x_609_, 1);
v_val_591_ = v_a_610_;
goto v___jp_590_;
}
else
{
lean_object* v_a_611_; lean_object* v___x_613_; uint8_t v_isShared_614_; uint8_t v_isSharedCheck_618_; 
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
v_a_611_ = lean_ctor_get(v___x_609_, 0);
v_isSharedCheck_618_ = !lean_is_exclusive(v___x_609_);
if (v_isSharedCheck_618_ == 0)
{
v___x_613_ = v___x_609_;
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
else
{
lean_inc(v_a_611_);
lean_dec(v___x_609_);
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
else
{
lean_dec(v_a_607_);
lean_dec(v_h_578_);
v_val_591_ = v_a_596_;
goto v___jp_590_;
}
}
else
{
lean_object* v_a_619_; lean_object* v___x_621_; uint8_t v_isShared_622_; uint8_t v_isSharedCheck_626_; 
lean_dec(v_a_598_);
lean_dec(v_a_596_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v_h_578_);
v_a_619_ = lean_ctor_get(v___x_606_, 0);
v_isSharedCheck_626_ = !lean_is_exclusive(v___x_606_);
if (v_isSharedCheck_626_ == 0)
{
v___x_621_ = v___x_606_;
v_isShared_622_ = v_isSharedCheck_626_;
goto v_resetjp_620_;
}
else
{
lean_inc(v_a_619_);
lean_dec(v___x_606_);
v___x_621_ = lean_box(0);
v_isShared_622_ = v_isSharedCheck_626_;
goto v_resetjp_620_;
}
v_resetjp_620_:
{
lean_object* v___x_624_; 
if (v_isShared_622_ == 0)
{
v___x_624_ = v___x_621_;
goto v_reusejp_623_;
}
else
{
lean_object* v_reuseFailAlloc_625_; 
v_reuseFailAlloc_625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_625_, 0, v_a_619_);
v___x_624_ = v_reuseFailAlloc_625_;
goto v_reusejp_623_;
}
v_reusejp_623_:
{
return v___x_624_;
}
}
}
}
}
}
else
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
lean_dec(v_a_596_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec_ref(v_m_579_);
lean_dec(v_h_578_);
v_a_629_ = lean_ctor_get(v___x_597_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_597_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_597_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_597_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_a_629_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
}
else
{
lean_object* v_a_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_644_; 
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec_ref(v_m_579_);
lean_dec(v_h_578_);
v_a_637_ = lean_ctor_get(v___x_595_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_595_);
if (v_isSharedCheck_644_ == 0)
{
v___x_639_ = v___x_595_;
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_a_637_);
lean_dec(v___x_595_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_642_; 
if (v_isShared_640_ == 0)
{
v___x_642_ = v___x_639_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v_a_637_);
v___x_642_ = v_reuseFailAlloc_643_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
return v___x_642_;
}
}
}
v___jp_590_:
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_592_ = lean_box(0);
v___x_593_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_593_, 0, v_val_591_);
lean_ctor_set(v___x_593_, 1, v___x_592_);
v___x_594_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_593_, v___y_582_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
return v___x_594_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__0___boxed(lean_object* v_h_645_, lean_object* v_m_646_, lean_object* v_checkDefEq_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
uint8_t v_checkDefEq_boxed_657_; lean_object* v_res_658_; 
v_checkDefEq_boxed_657_ = lean_unbox(v_checkDefEq_647_);
v_res_658_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__0(v_h_645_, v_m_646_, v_checkDefEq_boxed_657_, v___y_648_, v___y_649_, v___y_650_, v___y_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__1(lean_object* v_m_659_, uint8_t v_checkDefEq_660_, lean_object* v_h_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_){
_start:
{
lean_object* v___x_671_; lean_object* v___f_672_; lean_object* v___x_673_; 
v___x_671_ = lean_box(v_checkDefEq_660_);
v___f_672_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__0___boxed), 12, 3);
lean_closure_set(v___f_672_, 0, v_h_661_);
lean_closure_set(v___f_672_, 1, v_m_659_);
lean_closure_set(v___f_672_, 2, v___x_671_);
v___x_673_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_672_, v___y_662_, v___y_663_, v___y_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_);
return v___x_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__1___boxed(lean_object* v_m_674_, lean_object* v_checkDefEq_675_, lean_object* v_h_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
uint8_t v_checkDefEq_boxed_686_; lean_object* v_res_687_; 
v_checkDefEq_boxed_686_ = lean_unbox(v_checkDefEq_675_);
v_res_687_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__1(v_m_674_, v_checkDefEq_boxed_686_, v_h_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
lean_dec(v___y_684_);
lean_dec_ref(v___y_683_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
return v_res_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1_spec__1(lean_object* v_msgData_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_){
_start:
{
lean_object* v___x_694_; lean_object* v_env_695_; lean_object* v___x_696_; lean_object* v_mctx_697_; lean_object* v_lctx_698_; lean_object* v_options_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_694_ = lean_st_ref_get(v___y_692_);
v_env_695_ = lean_ctor_get(v___x_694_, 0);
lean_inc_ref(v_env_695_);
lean_dec(v___x_694_);
v___x_696_ = lean_st_ref_get(v___y_690_);
v_mctx_697_ = lean_ctor_get(v___x_696_, 0);
lean_inc_ref(v_mctx_697_);
lean_dec(v___x_696_);
v_lctx_698_ = lean_ctor_get(v___y_689_, 2);
v_options_699_ = lean_ctor_get(v___y_691_, 2);
lean_inc_ref(v_options_699_);
lean_inc_ref(v_lctx_698_);
v___x_700_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_700_, 0, v_env_695_);
lean_ctor_set(v___x_700_, 1, v_mctx_697_);
lean_ctor_set(v___x_700_, 2, v_lctx_698_);
lean_ctor_set(v___x_700_, 3, v_options_699_);
v___x_701_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_701_, 0, v___x_700_);
lean_ctor_set(v___x_701_, 1, v_msgData_688_);
v___x_702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_702_, 0, v___x_701_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1_spec__1___boxed(lean_object* v_msgData_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_){
_start:
{
lean_object* v_res_709_; 
v_res_709_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1_spec__1(v_msgData_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_);
lean_dec(v___y_707_);
lean_dec_ref(v___y_706_);
lean_dec(v___y_705_);
lean_dec_ref(v___y_704_);
return v_res_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___redArg(lean_object* v_msg_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_){
_start:
{
lean_object* v_ref_716_; lean_object* v___x_717_; lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_726_; 
v_ref_716_ = lean_ctor_get(v___y_713_, 5);
v___x_717_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1_spec__1(v_msg_710_, v___y_711_, v___y_712_, v___y_713_, v___y_714_);
v_a_718_ = lean_ctor_get(v___x_717_, 0);
v_isSharedCheck_726_ = !lean_is_exclusive(v___x_717_);
if (v_isSharedCheck_726_ == 0)
{
v___x_720_ = v___x_717_;
v_isShared_721_ = v_isSharedCheck_726_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_dec(v___x_717_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_726_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_722_; lean_object* v___x_724_; 
lean_inc(v_ref_716_);
v___x_722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_722_, 0, v_ref_716_);
lean_ctor_set(v___x_722_, 1, v_a_718_);
if (v_isShared_721_ == 0)
{
lean_ctor_set_tag(v___x_720_, 1);
lean_ctor_set(v___x_720_, 0, v___x_722_);
v___x_724_ = v___x_720_;
goto v_reusejp_723_;
}
else
{
lean_object* v_reuseFailAlloc_725_; 
v_reuseFailAlloc_725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_725_, 0, v___x_722_);
v___x_724_ = v_reuseFailAlloc_725_;
goto v_reusejp_723_;
}
v_reusejp_723_:
{
return v___x_724_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___redArg___boxed(lean_object* v_msg_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___redArg(v_msg_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec(v___y_729_);
lean_dec_ref(v___y_728_);
return v_res_733_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__1(void){
_start:
{
lean_object* v___x_735_; lean_object* v___x_736_; 
v___x_735_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__0));
v___x_736_ = l_Lean_stringToMessageData(v___x_735_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2(lean_object* v_tacticName_737_, lean_object* v_x_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_){
_start:
{
lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; 
v___x_748_ = l_Lean_stringToMessageData(v_tacticName_737_);
v___x_749_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__1, &lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___closed__1);
v___x_750_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_750_, 0, v___x_748_);
lean_ctor_set(v___x_750_, 1, v___x_749_);
v___x_751_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___redArg(v___x_750_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
return v___x_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___boxed(lean_object* v_tacticName_752_, lean_object* v_x_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_){
_start:
{
lean_object* v_res_763_; 
v_res_763_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2(v_tacticName_752_, v_x_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_, v___y_758_, v___y_759_, v___y_760_, v___y_761_);
lean_dec(v___y_761_);
lean_dec_ref(v___y_760_);
lean_dec(v___y_759_);
lean_dec_ref(v___y_758_);
lean_dec(v___y_757_);
lean_dec_ref(v___y_756_);
lean_dec(v___y_755_);
lean_dec_ref(v___y_754_);
lean_dec(v_x_753_);
return v_res_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__3(lean_object* v_m_764_, uint8_t v_checkDefEq_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_){
_start:
{
lean_object* v___x_775_; 
v___x_775_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_767_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
if (lean_obj_tag(v___x_775_) == 0)
{
lean_object* v_a_776_; lean_object* v___x_777_; 
v_a_776_ = lean_ctor_get(v___x_775_, 0);
lean_inc_n(v_a_776_, 2);
lean_dec_ref_known(v___x_775_, 1);
v___x_777_ = l_Lean_MVarId_getType(v_a_776_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
if (lean_obj_tag(v___x_777_) == 0)
{
lean_object* v_a_778_; lean_object* v___x_779_; lean_object* v_a_780_; lean_object* v___x_781_; lean_object* v___x_782_; 
v_a_778_ = lean_ctor_get(v___x_777_, 0);
lean_inc(v_a_778_);
lean_dec_ref_known(v___x_777_, 1);
v___x_779_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(v_a_778_, v___y_771_);
v_a_780_ = lean_ctor_get(v___x_779_, 0);
lean_inc(v_a_780_);
lean_dec_ref(v___x_779_);
v___x_781_ = lean_box(0);
lean_inc(v___y_773_);
lean_inc_ref(v___y_772_);
lean_inc(v___y_771_);
lean_inc_ref(v___y_770_);
v___x_782_ = lean_apply_7(v_m_764_, v___x_781_, v_a_780_, v___y_770_, v___y_771_, v___y_772_, v___y_773_, lean_box(0));
if (lean_obj_tag(v___x_782_) == 0)
{
lean_object* v_a_783_; lean_object* v___x_784_; 
v_a_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc(v_a_783_);
lean_dec_ref_known(v___x_782_, 1);
v___x_784_ = l_Lean_MVarId_change(v_a_776_, v_a_783_, v_checkDefEq_765_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
if (lean_obj_tag(v___x_784_) == 0)
{
lean_object* v_a_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; 
v_a_785_ = lean_ctor_get(v___x_784_, 0);
lean_inc(v_a_785_);
lean_dec_ref_known(v___x_784_, 1);
v___x_786_ = lean_box(0);
v___x_787_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_787_, 0, v_a_785_);
lean_ctor_set(v___x_787_, 1, v___x_786_);
v___x_788_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_787_, v___y_767_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
return v___x_788_;
}
else
{
lean_object* v_a_789_; lean_object* v___x_791_; uint8_t v_isShared_792_; uint8_t v_isSharedCheck_796_; 
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
v_a_789_ = lean_ctor_get(v___x_784_, 0);
v_isSharedCheck_796_ = !lean_is_exclusive(v___x_784_);
if (v_isSharedCheck_796_ == 0)
{
v___x_791_ = v___x_784_;
v_isShared_792_ = v_isSharedCheck_796_;
goto v_resetjp_790_;
}
else
{
lean_inc(v_a_789_);
lean_dec(v___x_784_);
v___x_791_ = lean_box(0);
v_isShared_792_ = v_isSharedCheck_796_;
goto v_resetjp_790_;
}
v_resetjp_790_:
{
lean_object* v___x_794_; 
if (v_isShared_792_ == 0)
{
v___x_794_ = v___x_791_;
goto v_reusejp_793_;
}
else
{
lean_object* v_reuseFailAlloc_795_; 
v_reuseFailAlloc_795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_795_, 0, v_a_789_);
v___x_794_ = v_reuseFailAlloc_795_;
goto v_reusejp_793_;
}
v_reusejp_793_:
{
return v___x_794_;
}
}
}
}
else
{
lean_object* v_a_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_804_; 
lean_dec(v_a_776_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
v_a_797_ = lean_ctor_get(v___x_782_, 0);
v_isSharedCheck_804_ = !lean_is_exclusive(v___x_782_);
if (v_isSharedCheck_804_ == 0)
{
v___x_799_ = v___x_782_;
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_a_797_);
lean_dec(v___x_782_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_802_; 
if (v_isShared_800_ == 0)
{
v___x_802_ = v___x_799_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v_a_797_);
v___x_802_ = v_reuseFailAlloc_803_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
return v___x_802_;
}
}
}
}
else
{
lean_object* v_a_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_812_; 
lean_dec(v_a_776_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
lean_dec_ref(v_m_764_);
v_a_805_ = lean_ctor_get(v___x_777_, 0);
v_isSharedCheck_812_ = !lean_is_exclusive(v___x_777_);
if (v_isSharedCheck_812_ == 0)
{
v___x_807_ = v___x_777_;
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_a_805_);
lean_dec(v___x_777_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v___x_810_; 
if (v_isShared_808_ == 0)
{
v___x_810_ = v___x_807_;
goto v_reusejp_809_;
}
else
{
lean_object* v_reuseFailAlloc_811_; 
v_reuseFailAlloc_811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_811_, 0, v_a_805_);
v___x_810_ = v_reuseFailAlloc_811_;
goto v_reusejp_809_;
}
v_reusejp_809_:
{
return v___x_810_;
}
}
}
}
else
{
lean_object* v_a_813_; lean_object* v___x_815_; uint8_t v_isShared_816_; uint8_t v_isSharedCheck_820_; 
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
lean_dec_ref(v_m_764_);
v_a_813_ = lean_ctor_get(v___x_775_, 0);
v_isSharedCheck_820_ = !lean_is_exclusive(v___x_775_);
if (v_isSharedCheck_820_ == 0)
{
v___x_815_ = v___x_775_;
v_isShared_816_ = v_isSharedCheck_820_;
goto v_resetjp_814_;
}
else
{
lean_inc(v_a_813_);
lean_dec(v___x_775_);
v___x_815_ = lean_box(0);
v_isShared_816_ = v_isSharedCheck_820_;
goto v_resetjp_814_;
}
v_resetjp_814_:
{
lean_object* v___x_818_; 
if (v_isShared_816_ == 0)
{
v___x_818_ = v___x_815_;
goto v_reusejp_817_;
}
else
{
lean_object* v_reuseFailAlloc_819_; 
v_reuseFailAlloc_819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_819_, 0, v_a_813_);
v___x_818_ = v_reuseFailAlloc_819_;
goto v_reusejp_817_;
}
v_reusejp_817_:
{
return v___x_818_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__3___boxed(lean_object* v_m_821_, lean_object* v_checkDefEq_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_){
_start:
{
uint8_t v_checkDefEq_boxed_832_; lean_object* v_res_833_; 
v_checkDefEq_boxed_832_ = lean_unbox(v_checkDefEq_822_);
v_res_833_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__3(v_m_821_, v_checkDefEq_boxed_832_, v___y_823_, v___y_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_, v___y_830_);
lean_dec(v___y_826_);
lean_dec_ref(v___y_825_);
lean_dec(v___y_824_);
lean_dec_ref(v___y_823_);
return v_res_833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__4(lean_object* v___f_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_){
_start:
{
lean_object* v___x_844_; 
v___x_844_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_834_, v___y_835_, v___y_836_, v___y_837_, v___y_838_, v___y_839_, v___y_840_, v___y_841_, v___y_842_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__4___boxed(lean_object* v___f_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_){
_start:
{
lean_object* v_res_855_; 
v_res_855_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__4(v___f_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_, v___y_852_, v___y_853_);
lean_dec(v___y_853_);
lean_dec_ref(v___y_852_);
lean_dec(v___y_851_);
lean_dec_ref(v___y_850_);
lean_dec(v___y_849_);
lean_dec_ref(v___y_848_);
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic(lean_object* v_m_856_, lean_object* v_loc_x3f_857_, lean_object* v_tacticName_858_, uint8_t v_checkDefEq_859_, lean_object* v_a_860_, lean_object* v_a_861_, lean_object* v_a_862_, lean_object* v_a_863_, lean_object* v_a_864_, lean_object* v_a_865_, lean_object* v_a_866_, lean_object* v_a_867_){
_start:
{
lean_object* v___x_869_; lean_object* v___f_870_; lean_object* v___f_871_; lean_object* v___x_872_; lean_object* v___f_873_; lean_object* v___f_874_; lean_object* v___y_876_; 
v___x_869_ = lean_box(v_checkDefEq_859_);
lean_inc_ref(v_m_856_);
v___f_870_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__1___boxed), 12, 2);
lean_closure_set(v___f_870_, 0, v_m_856_);
lean_closure_set(v___f_870_, 1, v___x_869_);
v___f_871_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__2___boxed), 11, 1);
lean_closure_set(v___f_871_, 0, v_tacticName_858_);
v___x_872_ = lean_box(v_checkDefEq_859_);
v___f_873_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__3___boxed), 11, 2);
lean_closure_set(v___f_873_, 0, v_m_856_);
lean_closure_set(v___f_873_, 1, v___x_872_);
v___f_874_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runDefEqTactic___lam__4___boxed), 10, 1);
lean_closure_set(v___f_874_, 0, v___f_873_);
if (lean_obj_tag(v_loc_x3f_857_) == 0)
{
lean_object* v___x_881_; 
v___x_881_ = lean_box(0);
v___y_876_ = v___x_881_;
goto v___jp_875_;
}
else
{
lean_object* v_val_882_; lean_object* v___x_884_; uint8_t v_isShared_885_; uint8_t v_isSharedCheck_889_; 
v_val_882_ = lean_ctor_get(v_loc_x3f_857_, 0);
v_isSharedCheck_889_ = !lean_is_exclusive(v_loc_x3f_857_);
if (v_isSharedCheck_889_ == 0)
{
v___x_884_ = v_loc_x3f_857_;
v_isShared_885_ = v_isSharedCheck_889_;
goto v_resetjp_883_;
}
else
{
lean_inc(v_val_882_);
lean_dec(v_loc_x3f_857_);
v___x_884_ = lean_box(0);
v_isShared_885_ = v_isSharedCheck_889_;
goto v_resetjp_883_;
}
v_resetjp_883_:
{
lean_object* v___x_887_; 
if (v_isShared_885_ == 0)
{
v___x_887_ = v___x_884_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_888_; 
v_reuseFailAlloc_888_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_888_, 0, v_val_882_);
v___x_887_ = v_reuseFailAlloc_888_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
v___y_876_ = v___x_887_;
goto v___jp_875_;
}
}
}
v___jp_875_:
{
lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; 
v___x_877_ = l_Lean_mkOptionalNode(v___y_876_);
v___x_878_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_877_);
lean_dec(v___x_877_);
v___x_879_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withLocation___boxed), 13, 4);
lean_closure_set(v___x_879_, 0, v___x_878_);
lean_closure_set(v___x_879_, 1, v___f_870_);
lean_closure_set(v___x_879_, 2, v___f_874_);
lean_closure_set(v___x_879_, 3, v___f_871_);
v___x_880_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___x_879_, v_a_860_, v_a_861_, v_a_862_, v_a_863_, v_a_864_, v_a_865_, v_a_866_, v_a_867_);
return v___x_880_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqTactic___boxed(lean_object* v_m_890_, lean_object* v_loc_x3f_891_, lean_object* v_tacticName_892_, lean_object* v_checkDefEq_893_, lean_object* v_a_894_, lean_object* v_a_895_, lean_object* v_a_896_, lean_object* v_a_897_, lean_object* v_a_898_, lean_object* v_a_899_, lean_object* v_a_900_, lean_object* v_a_901_, lean_object* v_a_902_){
_start:
{
uint8_t v_checkDefEq_boxed_903_; lean_object* v_res_904_; 
v_checkDefEq_boxed_903_ = lean_unbox(v_checkDefEq_893_);
v_res_904_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v_m_890_, v_loc_x3f_891_, v_tacticName_892_, v_checkDefEq_boxed_903_, v_a_894_, v_a_895_, v_a_896_, v_a_897_, v_a_898_, v_a_899_, v_a_900_, v_a_901_);
lean_dec(v_a_901_);
lean_dec_ref(v_a_900_);
lean_dec(v_a_899_);
lean_dec_ref(v_a_898_);
lean_dec(v_a_897_);
lean_dec_ref(v_a_896_);
lean_dec(v_a_895_);
lean_dec_ref(v_a_894_);
return v_res_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1(lean_object* v_00_u03b1_905_, lean_object* v_msg_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
lean_object* v___x_916_; 
v___x_916_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___redArg(v_msg_906_, v___y_911_, v___y_912_, v___y_913_, v___y_914_);
return v___x_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1___boxed(lean_object* v_00_u03b1_917_, lean_object* v_msg_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_){
_start:
{
lean_object* v_res_928_; 
v_res_928_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1(v_00_u03b1_917_, v_msg_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_, v___y_923_, v___y_924_, v___y_925_, v___y_926_);
lean_dec(v___y_926_);
lean_dec_ref(v___y_925_);
lean_dec(v___y_924_);
lean_dec_ref(v___y_923_);
lean_dec(v___y_922_);
lean_dec_ref(v___y_921_);
lean_dec(v___y_920_);
lean_dec_ref(v___y_919_);
return v_res_928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___redArg(lean_object* v_e_929_, lean_object* v___y_930_){
_start:
{
uint8_t v___x_932_; 
v___x_932_ = l_Lean_Expr_hasMVar(v_e_929_);
if (v___x_932_ == 0)
{
lean_object* v___x_933_; 
v___x_933_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_933_, 0, v_e_929_);
return v___x_933_;
}
else
{
lean_object* v___x_934_; lean_object* v_mctx_935_; lean_object* v___x_936_; lean_object* v_fst_937_; lean_object* v_snd_938_; lean_object* v___x_939_; lean_object* v_cache_940_; lean_object* v_zetaDeltaFVarIds_941_; lean_object* v_postponed_942_; lean_object* v_diag_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_952_; 
v___x_934_ = lean_st_ref_get(v___y_930_);
v_mctx_935_ = lean_ctor_get(v___x_934_, 0);
lean_inc_ref(v_mctx_935_);
lean_dec(v___x_934_);
v___x_936_ = l_Lean_instantiateMVarsCore(v_mctx_935_, v_e_929_);
v_fst_937_ = lean_ctor_get(v___x_936_, 0);
lean_inc(v_fst_937_);
v_snd_938_ = lean_ctor_get(v___x_936_, 1);
lean_inc(v_snd_938_);
lean_dec_ref(v___x_936_);
v___x_939_ = lean_st_ref_take(v___y_930_);
v_cache_940_ = lean_ctor_get(v___x_939_, 1);
v_zetaDeltaFVarIds_941_ = lean_ctor_get(v___x_939_, 2);
v_postponed_942_ = lean_ctor_get(v___x_939_, 3);
v_diag_943_ = lean_ctor_get(v___x_939_, 4);
v_isSharedCheck_952_ = !lean_is_exclusive(v___x_939_);
if (v_isSharedCheck_952_ == 0)
{
lean_object* v_unused_953_; 
v_unused_953_ = lean_ctor_get(v___x_939_, 0);
lean_dec(v_unused_953_);
v___x_945_ = v___x_939_;
v_isShared_946_ = v_isSharedCheck_952_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_diag_943_);
lean_inc(v_postponed_942_);
lean_inc(v_zetaDeltaFVarIds_941_);
lean_inc(v_cache_940_);
lean_dec(v___x_939_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_952_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
lean_object* v___x_948_; 
if (v_isShared_946_ == 0)
{
lean_ctor_set(v___x_945_, 0, v_snd_938_);
v___x_948_ = v___x_945_;
goto v_reusejp_947_;
}
else
{
lean_object* v_reuseFailAlloc_951_; 
v_reuseFailAlloc_951_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_951_, 0, v_snd_938_);
lean_ctor_set(v_reuseFailAlloc_951_, 1, v_cache_940_);
lean_ctor_set(v_reuseFailAlloc_951_, 2, v_zetaDeltaFVarIds_941_);
lean_ctor_set(v_reuseFailAlloc_951_, 3, v_postponed_942_);
lean_ctor_set(v_reuseFailAlloc_951_, 4, v_diag_943_);
v___x_948_ = v_reuseFailAlloc_951_;
goto v_reusejp_947_;
}
v_reusejp_947_:
{
lean_object* v___x_949_; lean_object* v___x_950_; 
v___x_949_ = lean_st_ref_set(v___y_930_, v___x_948_);
v___x_950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_950_, 0, v_fst_937_);
return v___x_950_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___redArg___boxed(lean_object* v_e_954_, lean_object* v___y_955_, lean_object* v___y_956_){
_start:
{
lean_object* v_res_957_; 
v_res_957_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___redArg(v_e_954_, v___y_955_);
lean_dec(v___y_955_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0(lean_object* v_e_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_){
_start:
{
lean_object* v___x_968_; 
v___x_968_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___redArg(v_e_958_, v___y_964_);
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___boxed(lean_object* v_e_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_){
_start:
{
lean_object* v_res_979_; 
v_res_979_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0(v_e_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_, v___y_977_);
lean_dec(v___y_977_);
lean_dec_ref(v___y_976_);
lean_dec(v___y_975_);
lean_dec_ref(v___y_974_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
return v_res_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___lam__0(lean_object* v_m_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_){
_start:
{
lean_object* v___x_990_; 
v___x_990_ = l_Lean_Elab_Tactic_Conv_getLhs___redArg(v___y_982_, v___y_985_, v___y_986_, v___y_987_, v___y_988_);
if (lean_obj_tag(v___x_990_) == 0)
{
lean_object* v_a_991_; lean_object* v___x_992_; lean_object* v_a_993_; lean_object* v___x_994_; 
v_a_991_ = lean_ctor_get(v___x_990_, 0);
lean_inc(v_a_991_);
lean_dec_ref_known(v___x_990_, 1);
v___x_992_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqConvTactic_spec__0___redArg(v_a_991_, v___y_986_);
v_a_993_ = lean_ctor_get(v___x_992_, 0);
lean_inc(v_a_993_);
lean_dec_ref(v___x_992_);
lean_inc(v___y_988_);
lean_inc_ref(v___y_987_);
lean_inc(v___y_986_);
lean_inc_ref(v___y_985_);
v___x_994_ = lean_apply_6(v_m_980_, v_a_993_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, lean_box(0));
if (lean_obj_tag(v___x_994_) == 0)
{
lean_object* v_a_995_; lean_object* v___x_996_; 
v_a_995_ = lean_ctor_get(v___x_994_, 0);
lean_inc(v_a_995_);
lean_dec_ref_known(v___x_994_, 1);
v___x_996_ = l_Lean_Elab_Tactic_Conv_changeLhs(v_a_995_, v___y_981_, v___y_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_);
lean_dec(v___y_988_);
lean_dec_ref(v___y_987_);
lean_dec(v___y_986_);
lean_dec_ref(v___y_985_);
return v___x_996_;
}
else
{
lean_object* v_a_997_; lean_object* v___x_999_; uint8_t v_isShared_1000_; uint8_t v_isSharedCheck_1004_; 
lean_dec(v___y_988_);
lean_dec_ref(v___y_987_);
lean_dec(v___y_986_);
lean_dec_ref(v___y_985_);
v_a_997_ = lean_ctor_get(v___x_994_, 0);
v_isSharedCheck_1004_ = !lean_is_exclusive(v___x_994_);
if (v_isSharedCheck_1004_ == 0)
{
v___x_999_ = v___x_994_;
v_isShared_1000_ = v_isSharedCheck_1004_;
goto v_resetjp_998_;
}
else
{
lean_inc(v_a_997_);
lean_dec(v___x_994_);
v___x_999_ = lean_box(0);
v_isShared_1000_ = v_isSharedCheck_1004_;
goto v_resetjp_998_;
}
v_resetjp_998_:
{
lean_object* v___x_1002_; 
if (v_isShared_1000_ == 0)
{
v___x_1002_ = v___x_999_;
goto v_reusejp_1001_;
}
else
{
lean_object* v_reuseFailAlloc_1003_; 
v_reuseFailAlloc_1003_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1003_, 0, v_a_997_);
v___x_1002_ = v_reuseFailAlloc_1003_;
goto v_reusejp_1001_;
}
v_reusejp_1001_:
{
return v___x_1002_;
}
}
}
}
else
{
lean_object* v_a_1005_; lean_object* v___x_1007_; uint8_t v_isShared_1008_; uint8_t v_isSharedCheck_1012_; 
lean_dec(v___y_988_);
lean_dec_ref(v___y_987_);
lean_dec(v___y_986_);
lean_dec_ref(v___y_985_);
lean_dec_ref(v_m_980_);
v_a_1005_ = lean_ctor_get(v___x_990_, 0);
v_isSharedCheck_1012_ = !lean_is_exclusive(v___x_990_);
if (v_isSharedCheck_1012_ == 0)
{
v___x_1007_ = v___x_990_;
v_isShared_1008_ = v_isSharedCheck_1012_;
goto v_resetjp_1006_;
}
else
{
lean_inc(v_a_1005_);
lean_dec(v___x_990_);
v___x_1007_ = lean_box(0);
v_isShared_1008_ = v_isSharedCheck_1012_;
goto v_resetjp_1006_;
}
v_resetjp_1006_:
{
lean_object* v___x_1010_; 
if (v_isShared_1008_ == 0)
{
v___x_1010_ = v___x_1007_;
goto v_reusejp_1009_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v_a_1005_);
v___x_1010_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1009_;
}
v_reusejp_1009_:
{
return v___x_1010_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___lam__0___boxed(lean_object* v_m_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_){
_start:
{
lean_object* v_res_1023_; 
v_res_1023_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___lam__0(v_m_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
return v_res_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(lean_object* v_m_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_, lean_object* v_a_1031_, lean_object* v_a_1032_){
_start:
{
lean_object* v___f_1034_; lean_object* v___x_1035_; 
v___f_1034_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___lam__0___boxed), 10, 1);
lean_closure_set(v___f_1034_, 0, v_m_1024_);
v___x_1035_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1034_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_);
return v___x_1035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runDefEqConvTactic___boxed(lean_object* v_m_1036_, lean_object* v_a_1037_, lean_object* v_a_1038_, lean_object* v_a_1039_, lean_object* v_a_1040_, lean_object* v_a_1041_, lean_object* v_a_1042_, lean_object* v_a_1043_, lean_object* v_a_1044_, lean_object* v_a_1045_){
_start:
{
lean_object* v_res_1046_; 
v_res_1046_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(v_m_1036_, v_a_1037_, v_a_1038_, v_a_1039_, v_a_1040_, v_a_1041_, v_a_1042_, v_a_1043_, v_a_1044_);
lean_dec(v_a_1044_);
lean_dec_ref(v_a_1043_);
lean_dec(v_a_1042_);
lean_dec_ref(v_a_1041_);
lean_dec(v_a_1040_);
lean_dec_ref(v_a_1039_);
lean_dec(v_a_1038_);
lean_dec_ref(v_a_1037_);
return v_res_1046_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__13(void){
_start:
{
lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; 
v___x_1069_ = l_Lean_Parser_Tactic_location;
v___x_1070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__12));
v___x_1071_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_1072_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1072_, 0, v___x_1071_);
lean_ctor_set(v___x_1072_, 1, v___x_1070_);
lean_ctor_set(v___x_1072_, 2, v___x_1069_);
return v___x_1072_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14(void){
_start:
{
lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; 
v___x_1073_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__13, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__13_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__13);
v___x_1074_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__9));
v___x_1075_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1075_, 0, v___x_1074_);
lean_ctor_set(v___x_1075_, 1, v___x_1073_);
return v___x_1075_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__15(void){
_start:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; 
v___x_1076_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_1077_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__7));
v___x_1078_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_1079_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1079_, 0, v___x_1078_);
lean_ctor_set(v___x_1079_, 1, v___x_1077_);
lean_ctor_set(v___x_1079_, 2, v___x_1076_);
return v___x_1079_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__16(void){
_start:
{
lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; 
v___x_1080_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__15, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__15_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__15);
v___x_1081_ = lean_unsigned_to_nat(1022u);
v___x_1082_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3));
v___x_1083_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1083_, 0, v___x_1082_);
lean_ctor_set(v___x_1083_, 1, v___x_1081_);
lean_ctor_set(v___x_1083_, 2, v___x_1080_);
return v___x_1083_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticWhnf____(void){
_start:
{
lean_object* v___x_1084_; 
v___x_1084_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__16, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__16_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__16);
return v___x_1084_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1085_ = lean_box(0);
v___x_1086_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1087_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1087_, 0, v___x_1086_);
lean_ctor_set(v___x_1087_, 1, v___x_1085_);
return v___x_1087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg(){
_start:
{
lean_object* v___x_1089_; lean_object* v___x_1090_; 
v___x_1089_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___closed__0);
v___x_1090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1090_, 0, v___x_1089_);
return v___x_1090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg___boxed(lean_object* v___y_1091_){
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0(lean_object* v_00_u03b1_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_){
_start:
{
lean_object* v___x_1103_; 
v___x_1103_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___boxed(lean_object* v_00_u03b1_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
lean_object* v_res_1114_; 
v_res_1114_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0(v_00_u03b1_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_);
lean_dec(v___y_1112_);
lean_dec_ref(v___y_1111_);
lean_dec(v___y_1110_);
lean_dec_ref(v___y_1109_);
lean_dec(v___y_1108_);
lean_dec_ref(v___y_1107_);
lean_dec(v___y_1106_);
lean_dec_ref(v___y_1105_);
return v_res_1114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___lam__0(lean_object* v_x_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_){
_start:
{
lean_object* v___x_1122_; 
lean_inc(v___y_1120_);
lean_inc_ref(v___y_1119_);
lean_inc(v___y_1118_);
lean_inc_ref(v___y_1117_);
v___x_1122_ = lean_whnf(v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_);
return v___x_1122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___lam__0___boxed(lean_object* v_x_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_){
_start:
{
lean_object* v_res_1130_; 
v_res_1130_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___lam__0(v_x_1123_, v___y_1124_, v___y_1125_, v___y_1126_, v___y_1127_, v___y_1128_);
lean_dec(v___y_1128_);
lean_dec_ref(v___y_1127_);
lean_dec(v___y_1126_);
lean_dec_ref(v___y_1125_);
lean_dec(v_x_1123_);
return v_res_1130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1(lean_object* v_x_1132_, lean_object* v_a_1133_, lean_object* v_a_1134_, lean_object* v_a_1135_, lean_object* v_a_1136_, lean_object* v_a_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_){
_start:
{
lean_object* v___x_1142_; uint8_t v___x_1143_; 
v___x_1142_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__3));
lean_inc(v_x_1132_);
v___x_1143_ = l_Lean_Syntax_isOfKind(v_x_1132_, v___x_1142_);
if (v___x_1143_ == 0)
{
lean_object* v___x_1144_; 
lean_dec(v_x_1132_);
v___x_1144_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_1144_;
}
else
{
lean_object* v___f_1145_; lean_object* v___y_1147_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; 
v___f_1145_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___closed__0));
v___x_1151_ = lean_unsigned_to_nat(1u);
v___x_1152_ = l_Lean_Syntax_getArg(v_x_1132_, v___x_1151_);
lean_dec(v_x_1132_);
v___x_1153_ = l_Lean_Syntax_getOptional_x3f(v___x_1152_);
lean_dec(v___x_1152_);
if (lean_obj_tag(v___x_1153_) == 0)
{
lean_object* v___x_1154_; 
v___x_1154_ = lean_box(0);
v___y_1147_ = v___x_1154_;
goto v___jp_1146_;
}
else
{
lean_object* v_val_1155_; lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1162_; 
v_val_1155_ = lean_ctor_get(v___x_1153_, 0);
v_isSharedCheck_1162_ = !lean_is_exclusive(v___x_1153_);
if (v_isSharedCheck_1162_ == 0)
{
v___x_1157_ = v___x_1153_;
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
else
{
lean_inc(v_val_1155_);
lean_dec(v___x_1153_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v___x_1160_; 
if (v_isShared_1158_ == 0)
{
v___x_1160_ = v___x_1157_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v_val_1155_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
v___y_1147_ = v___x_1160_;
goto v___jp_1146_;
}
}
}
v___jp_1146_:
{
lean_object* v___x_1148_; uint8_t v___x_1149_; lean_object* v___x_1150_; 
v___x_1148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__6));
v___x_1149_ = 0;
v___x_1150_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___f_1145_, v___y_1147_, v___x_1148_, v___x_1149_, v_a_1133_, v_a_1134_, v_a_1135_, v_a_1136_, v_a_1137_, v_a_1138_, v_a_1139_, v_a_1140_);
return v___x_1150_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1___boxed(lean_object* v_x_1163_, lean_object* v_a_1164_, lean_object* v_a_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_, lean_object* v_a_1168_, lean_object* v_a_1169_, lean_object* v_a_1170_, lean_object* v_a_1171_, lean_object* v_a_1172_){
_start:
{
lean_object* v_res_1173_; 
v_res_1173_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1(v_x_1163_, v_a_1164_, v_a_1165_, v_a_1166_, v_a_1167_, v_a_1168_, v_a_1169_, v_a_1170_, v_a_1171_);
lean_dec(v_a_1171_);
lean_dec_ref(v_a_1170_);
lean_dec(v_a_1169_);
lean_dec_ref(v_a_1168_);
lean_dec(v_a_1167_);
lean_dec_ref(v_a_1166_);
lean_dec(v_a_1165_);
lean_dec_ref(v_a_1164_);
return v_res_1173_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__4(void){
_start:
{
lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; 
v___x_1183_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_1184_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__3));
v___x_1185_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_1186_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1186_, 0, v___x_1185_);
lean_ctor_set(v___x_1186_, 1, v___x_1184_);
lean_ctor_set(v___x_1186_, 2, v___x_1183_);
return v___x_1186_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__5(void){
_start:
{
lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; 
v___x_1187_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__4, &lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__4);
v___x_1188_ = lean_unsigned_to_nat(1022u);
v___x_1189_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1));
v___x_1190_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1189_);
lean_ctor_set(v___x_1190_, 1, v___x_1188_);
lean_ctor_set(v___x_1190_, 2, v___x_1187_);
return v___x_1190_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_betaReduceStx(void){
_start:
{
lean_object* v___x_1191_; 
v___x_1191_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__5, &lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__5);
return v___x_1191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___lam__0(lean_object* v_x_1192_, lean_object* v_e_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_){
_start:
{
lean_object* v___x_1199_; 
v___x_1199_ = l_Lean_Core_betaReduce(v_e_1193_, v___y_1196_, v___y_1197_);
return v___x_1199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___lam__0___boxed(lean_object* v_x_1200_, lean_object* v_e_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___lam__0(v_x_1200_, v_e_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
lean_dec(v_x_1200_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1(lean_object* v_x_1209_, lean_object* v_a_1210_, lean_object* v_a_1211_, lean_object* v_a_1212_, lean_object* v_a_1213_, lean_object* v_a_1214_, lean_object* v_a_1215_, lean_object* v_a_1216_, lean_object* v_a_1217_){
_start:
{
lean_object* v___x_1219_; uint8_t v___x_1220_; 
v___x_1219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__1));
lean_inc(v_x_1209_);
v___x_1220_ = l_Lean_Syntax_isOfKind(v_x_1209_, v___x_1219_);
if (v___x_1220_ == 0)
{
lean_object* v___x_1221_; 
lean_dec(v_x_1209_);
v___x_1221_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_1221_;
}
else
{
lean_object* v___f_1222_; lean_object* v___y_1224_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; 
v___f_1222_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___closed__0));
v___x_1228_ = lean_unsigned_to_nat(1u);
v___x_1229_ = l_Lean_Syntax_getArg(v_x_1209_, v___x_1228_);
lean_dec(v_x_1209_);
v___x_1230_ = l_Lean_Syntax_getOptional_x3f(v___x_1229_);
lean_dec(v___x_1229_);
if (lean_obj_tag(v___x_1230_) == 0)
{
lean_object* v___x_1231_; 
v___x_1231_ = lean_box(0);
v___y_1224_ = v___x_1231_;
goto v___jp_1223_;
}
else
{
lean_object* v_val_1232_; lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1239_; 
v_val_1232_ = lean_ctor_get(v___x_1230_, 0);
v_isSharedCheck_1239_ = !lean_is_exclusive(v___x_1230_);
if (v_isSharedCheck_1239_ == 0)
{
v___x_1234_ = v___x_1230_;
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
else
{
lean_inc(v_val_1232_);
lean_dec(v___x_1230_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v___x_1237_; 
if (v_isShared_1235_ == 0)
{
v___x_1237_ = v___x_1234_;
goto v_reusejp_1236_;
}
else
{
lean_object* v_reuseFailAlloc_1238_; 
v_reuseFailAlloc_1238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1238_, 0, v_val_1232_);
v___x_1237_ = v_reuseFailAlloc_1238_;
goto v_reusejp_1236_;
}
v_reusejp_1236_:
{
v___y_1224_ = v___x_1237_;
goto v___jp_1223_;
}
}
}
v___jp_1223_:
{
lean_object* v___x_1225_; uint8_t v___x_1226_; lean_object* v___x_1227_; 
v___x_1225_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_betaReduceStx___closed__2));
v___x_1226_ = 0;
v___x_1227_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___f_1222_, v___y_1224_, v___x_1225_, v___x_1226_, v_a_1210_, v_a_1211_, v_a_1212_, v_a_1213_, v_a_1214_, v_a_1215_, v_a_1216_, v_a_1217_);
return v___x_1227_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1___boxed(lean_object* v_x_1240_, lean_object* v_a_1241_, lean_object* v_a_1242_, lean_object* v_a_1243_, lean_object* v_a_1244_, lean_object* v_a_1245_, lean_object* v_a_1246_, lean_object* v_a_1247_, lean_object* v_a_1248_, lean_object* v_a_1249_){
_start:
{
lean_object* v_res_1250_; 
v_res_1250_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__betaReduceStx__1(v_x_1240_, v_a_1241_, v_a_1242_, v_a_1243_, v_a_1244_, v_a_1245_, v_a_1246_, v_a_1247_, v_a_1248_);
lean_dec(v_a_1248_);
lean_dec_ref(v_a_1247_);
lean_dec(v_a_1246_);
lean_dec_ref(v_a_1245_);
lean_dec(v_a_1244_);
lean_dec_ref(v_a_1243_);
lean_dec(v_a_1242_);
lean_dec_ref(v_a_1241_);
return v_res_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___lam__0(lean_object* v_x_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_){
_start:
{
lean_object* v___x_1267_; 
v___x_1267_ = l_Lean_Core_betaReduce(v_x_1261_, v___y_1264_, v___y_1265_);
return v___x_1267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___lam__0___boxed(lean_object* v_x_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
lean_object* v_res_1274_; 
v_res_1274_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___lam__0(v_x_1268_, v___y_1269_, v___y_1270_, v___y_1271_, v___y_1272_);
lean_dec(v___y_1272_);
lean_dec_ref(v___y_1271_);
lean_dec(v___y_1270_);
lean_dec_ref(v___y_1269_);
return v_res_1274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1(lean_object* v_x_1276_, lean_object* v_a_1277_, lean_object* v_a_1278_, lean_object* v_a_1279_, lean_object* v_a_1280_, lean_object* v_a_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_, lean_object* v_a_1284_){
_start:
{
lean_object* v___x_1286_; uint8_t v___x_1287_; 
v___x_1286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convBeta__reduce___closed__1));
v___x_1287_ = l_Lean_Syntax_isOfKind(v_x_1276_, v___x_1286_);
if (v___x_1287_ == 0)
{
lean_object* v___x_1288_; 
v___x_1288_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_1288_;
}
else
{
lean_object* v___f_1289_; lean_object* v___x_1290_; 
v___f_1289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___closed__0));
v___x_1290_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(v___f_1289_, v_a_1277_, v_a_1278_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_, v_a_1283_, v_a_1284_);
return v___x_1290_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1___boxed(lean_object* v_x_1291_, lean_object* v_a_1292_, lean_object* v_a_1293_, lean_object* v_a_1294_, lean_object* v_a_1295_, lean_object* v_a_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_, lean_object* v_a_1299_, lean_object* v_a_1300_){
_start:
{
lean_object* v_res_1301_; 
v_res_1301_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convBeta__reduce__1(v_x_1291_, v_a_1292_, v_a_1293_, v_a_1294_, v_a_1295_, v_a_1296_, v_a_1297_, v_a_1298_, v_a_1299_);
lean_dec(v_a_1299_);
lean_dec_ref(v_a_1298_);
lean_dec(v_a_1297_);
lean_dec_ref(v_a_1296_);
lean_dec(v_a_1295_);
lean_dec_ref(v_a_1294_);
lean_dec(v_a_1293_);
lean_dec_ref(v_a_1292_);
return v_res_1301_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__4(void){
_start:
{
lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; 
v___x_1311_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_1312_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__3));
v___x_1313_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_1314_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1314_, 0, v___x_1313_);
lean_ctor_set(v___x_1314_, 1, v___x_1312_);
lean_ctor_set(v___x_1314_, 2, v___x_1311_);
return v___x_1314_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__5(void){
_start:
{
lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; 
v___x_1315_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__4, &lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__4);
v___x_1316_ = lean_unsigned_to_nat(1022u);
v___x_1317_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1));
v___x_1318_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1318_, 0, v___x_1317_);
lean_ctor_set(v___x_1318_, 1, v___x_1316_);
lean_ctor_set(v___x_1318_, 2, v___x_1315_);
return v___x_1318_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticReduce____(void){
_start:
{
lean_object* v___x_1319_; 
v___x_1319_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__5, &lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__5);
return v___x_1319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___lam__0(uint8_t v___x_1320_, lean_object* v_x_1321_, lean_object* v_e_1322_, lean_object* v___y_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_){
_start:
{
uint8_t v___x_1328_; lean_object* v___x_1329_; 
v___x_1328_ = 0;
v___x_1329_ = l_Lean_Meta_reduce(v_e_1322_, v___x_1320_, v___x_1328_, v___x_1328_, v___y_1323_, v___y_1324_, v___y_1325_, v___y_1326_);
return v___x_1329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___lam__0___boxed(lean_object* v___x_1330_, lean_object* v_x_1331_, lean_object* v_e_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_){
_start:
{
uint8_t v___x_213__boxed_1338_; lean_object* v_res_1339_; 
v___x_213__boxed_1338_ = lean_unbox(v___x_1330_);
v_res_1339_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___lam__0(v___x_213__boxed_1338_, v_x_1331_, v_e_1332_, v___y_1333_, v___y_1334_, v___y_1335_, v___y_1336_);
lean_dec(v___y_1336_);
lean_dec_ref(v___y_1335_);
lean_dec(v___y_1334_);
lean_dec_ref(v___y_1333_);
lean_dec(v_x_1331_);
return v_res_1339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1(lean_object* v_x_1340_, lean_object* v_a_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_, lean_object* v_a_1348_){
_start:
{
lean_object* v___x_1350_; uint8_t v___x_1351_; 
v___x_1350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__1));
lean_inc(v_x_1340_);
v___x_1351_ = l_Lean_Syntax_isOfKind(v_x_1340_, v___x_1350_);
if (v___x_1351_ == 0)
{
lean_object* v___x_1352_; 
lean_dec(v_x_1340_);
v___x_1352_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_1352_;
}
else
{
lean_object* v___x_1353_; lean_object* v___f_1354_; lean_object* v___y_1356_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; 
v___x_1353_ = lean_box(v___x_1351_);
v___f_1354_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1354_, 0, v___x_1353_);
v___x_1359_ = lean_unsigned_to_nat(1u);
v___x_1360_ = l_Lean_Syntax_getArg(v_x_1340_, v___x_1359_);
lean_dec(v_x_1340_);
v___x_1361_ = l_Lean_Syntax_getOptional_x3f(v___x_1360_);
lean_dec(v___x_1360_);
if (lean_obj_tag(v___x_1361_) == 0)
{
lean_object* v___x_1362_; 
v___x_1362_ = lean_box(0);
v___y_1356_ = v___x_1362_;
goto v___jp_1355_;
}
else
{
lean_object* v_val_1363_; lean_object* v___x_1365_; uint8_t v_isShared_1366_; uint8_t v_isSharedCheck_1370_; 
v_val_1363_ = lean_ctor_get(v___x_1361_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1361_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1365_ = v___x_1361_;
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
else
{
lean_inc(v_val_1363_);
lean_dec(v___x_1361_);
v___x_1365_ = lean_box(0);
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
v_resetjp_1364_:
{
lean_object* v___x_1368_; 
if (v_isShared_1366_ == 0)
{
v___x_1368_ = v___x_1365_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_val_1363_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
v___y_1356_ = v___x_1368_;
goto v___jp_1355_;
}
}
}
v___jp_1355_:
{
lean_object* v___x_1357_; lean_object* v___x_1358_; 
v___x_1357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticReduce_____00__closed__2));
v___x_1358_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___f_1354_, v___y_1356_, v___x_1357_, v___x_1351_, v_a_1341_, v_a_1342_, v_a_1343_, v_a_1344_, v_a_1345_, v_a_1346_, v_a_1347_, v_a_1348_);
return v___x_1358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1___boxed(lean_object* v_x_1371_, lean_object* v_a_1372_, lean_object* v_a_1373_, lean_object* v_a_1374_, lean_object* v_a_1375_, lean_object* v_a_1376_, lean_object* v_a_1377_, lean_object* v_a_1378_, lean_object* v_a_1379_, lean_object* v_a_1380_){
_start:
{
lean_object* v_res_1381_; 
v_res_1381_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticReduce______1(v_x_1371_, v_a_1372_, v_a_1373_, v_a_1374_, v_a_1375_, v_a_1376_, v_a_1377_, v_a_1378_, v_a_1379_);
lean_dec(v_a_1379_);
lean_dec_ref(v_a_1378_);
lean_dec(v_a_1377_);
lean_dec_ref(v_a_1376_);
lean_dec(v_a_1375_);
lean_dec_ref(v_a_1374_);
lean_dec(v_a_1373_);
lean_dec_ref(v_a_1372_);
return v_res_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__0(lean_object* v_e_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_){
_start:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1388_, 0, v_e_1382_);
v___x_1389_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1389_, 0, v___x_1388_);
return v___x_1389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__0___boxed(lean_object* v_e_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_){
_start:
{
lean_object* v_res_1396_; 
v_res_1396_ = lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__0(v_e_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
return v_res_1396_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0_spec__0(lean_object* v_a_1397_, lean_object* v_as_1398_, size_t v_i_1399_, size_t v_stop_1400_){
_start:
{
uint8_t v___x_1401_; 
v___x_1401_ = lean_usize_dec_eq(v_i_1399_, v_stop_1400_);
if (v___x_1401_ == 0)
{
lean_object* v___x_1402_; uint8_t v___x_1403_; 
v___x_1402_ = lean_array_uget_borrowed(v_as_1398_, v_i_1399_);
v___x_1403_ = l_Lean_instBEqFVarId_beq(v_a_1397_, v___x_1402_);
if (v___x_1403_ == 0)
{
size_t v___x_1404_; size_t v___x_1405_; 
v___x_1404_ = ((size_t)1ULL);
v___x_1405_ = lean_usize_add(v_i_1399_, v___x_1404_);
v_i_1399_ = v___x_1405_;
goto _start;
}
else
{
return v___x_1403_;
}
}
else
{
uint8_t v___x_1407_; 
v___x_1407_ = 0;
return v___x_1407_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0_spec__0___boxed(lean_object* v_a_1408_, lean_object* v_as_1409_, lean_object* v_i_1410_, lean_object* v_stop_1411_){
_start:
{
size_t v_i_boxed_1412_; size_t v_stop_boxed_1413_; uint8_t v_res_1414_; lean_object* v_r_1415_; 
v_i_boxed_1412_ = lean_unbox_usize(v_i_1410_);
lean_dec(v_i_1410_);
v_stop_boxed_1413_ = lean_unbox_usize(v_stop_1411_);
lean_dec(v_stop_1411_);
v_res_1414_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0_spec__0(v_a_1408_, v_as_1409_, v_i_boxed_1412_, v_stop_boxed_1413_);
lean_dec_ref(v_as_1409_);
lean_dec(v_a_1408_);
v_r_1415_ = lean_box(v_res_1414_);
return v_r_1415_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0(lean_object* v_as_1416_, lean_object* v_a_1417_){
_start:
{
lean_object* v___x_1418_; lean_object* v___x_1419_; uint8_t v___x_1420_; 
v___x_1418_ = lean_unsigned_to_nat(0u);
v___x_1419_ = lean_array_get_size(v_as_1416_);
v___x_1420_ = lean_nat_dec_lt(v___x_1418_, v___x_1419_);
if (v___x_1420_ == 0)
{
return v___x_1420_;
}
else
{
if (v___x_1420_ == 0)
{
return v___x_1420_;
}
else
{
size_t v___x_1421_; size_t v___x_1422_; uint8_t v___x_1423_; 
v___x_1421_ = ((size_t)0ULL);
v___x_1422_ = lean_usize_of_nat(v___x_1419_);
v___x_1423_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0_spec__0(v_a_1417_, v_as_1416_, v___x_1421_, v___x_1422_);
return v___x_1423_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0___boxed(lean_object* v_as_1424_, lean_object* v_a_1425_){
_start:
{
uint8_t v_res_1426_; lean_object* v_r_1427_; 
v_res_1426_ = lp_mathlib_Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0(v_as_1424_, v_a_1425_);
lean_dec(v_a_1425_);
lean_dec_ref(v_as_1424_);
v_r_1427_ = lean_box(v_res_1426_);
return v_r_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1(lean_object* v_fvars_1430_, lean_object* v_node_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_){
_start:
{
if (lean_obj_tag(v_node_1431_) == 1)
{
lean_object* v_fvarId_1437_; uint8_t v___x_1438_; 
v_fvarId_1437_ = lean_ctor_get(v_node_1431_, 0);
lean_inc(v_fvarId_1437_);
lean_dec_ref_known(v_node_1431_, 1);
v___x_1438_ = lp_mathlib_Array_contains___at___00Mathlib_Tactic_unfoldFVars_spec__0(v_fvars_1430_, v_fvarId_1437_);
if (v___x_1438_ == 0)
{
lean_object* v___x_1439_; lean_object* v___x_1440_; 
lean_dec(v_fvarId_1437_);
v___x_1439_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0));
v___x_1440_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1440_, 0, v___x_1439_);
return v___x_1440_;
}
else
{
uint8_t v___x_1441_; lean_object* v___x_1442_; 
v___x_1441_ = 0;
v___x_1442_ = l_Lean_FVarId_getValue_x3f___redArg(v_fvarId_1437_, v___x_1441_, v___y_1432_, v___y_1434_, v___y_1435_);
if (lean_obj_tag(v___x_1442_) == 0)
{
lean_object* v_a_1443_; lean_object* v___x_1445_; uint8_t v_isShared_1446_; uint8_t v_isSharedCheck_1468_; 
v_a_1443_ = lean_ctor_get(v___x_1442_, 0);
v_isSharedCheck_1468_ = !lean_is_exclusive(v___x_1442_);
if (v_isSharedCheck_1468_ == 0)
{
v___x_1445_ = v___x_1442_;
v_isShared_1446_ = v_isSharedCheck_1468_;
goto v_resetjp_1444_;
}
else
{
lean_inc(v_a_1443_);
lean_dec(v___x_1442_);
v___x_1445_ = lean_box(0);
v_isShared_1446_ = v_isSharedCheck_1468_;
goto v_resetjp_1444_;
}
v_resetjp_1444_:
{
if (lean_obj_tag(v_a_1443_) == 1)
{
lean_object* v_val_1447_; lean_object* v___x_1449_; uint8_t v_isShared_1450_; uint8_t v_isSharedCheck_1463_; 
lean_del_object(v___x_1445_);
v_val_1447_ = lean_ctor_get(v_a_1443_, 0);
v_isSharedCheck_1463_ = !lean_is_exclusive(v_a_1443_);
if (v_isSharedCheck_1463_ == 0)
{
v___x_1449_ = v_a_1443_;
v_isShared_1450_ = v_isSharedCheck_1463_;
goto v_resetjp_1448_;
}
else
{
lean_inc(v_val_1447_);
lean_dec(v_a_1443_);
v___x_1449_ = lean_box(0);
v_isShared_1450_ = v_isSharedCheck_1463_;
goto v_resetjp_1448_;
}
v_resetjp_1448_:
{
lean_object* v___x_1451_; lean_object* v_a_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1462_; 
v___x_1451_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(v_val_1447_, v___y_1433_);
v_a_1452_ = lean_ctor_get(v___x_1451_, 0);
v_isSharedCheck_1462_ = !lean_is_exclusive(v___x_1451_);
if (v_isSharedCheck_1462_ == 0)
{
v___x_1454_ = v___x_1451_;
v_isShared_1455_ = v_isSharedCheck_1462_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_a_1452_);
lean_dec(v___x_1451_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1462_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1457_; 
if (v_isShared_1450_ == 0)
{
lean_ctor_set(v___x_1449_, 0, v_a_1452_);
v___x_1457_ = v___x_1449_;
goto v_reusejp_1456_;
}
else
{
lean_object* v_reuseFailAlloc_1461_; 
v_reuseFailAlloc_1461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1461_, 0, v_a_1452_);
v___x_1457_ = v_reuseFailAlloc_1461_;
goto v_reusejp_1456_;
}
v_reusejp_1456_:
{
lean_object* v___x_1459_; 
if (v_isShared_1455_ == 0)
{
lean_ctor_set(v___x_1454_, 0, v___x_1457_);
v___x_1459_ = v___x_1454_;
goto v_reusejp_1458_;
}
else
{
lean_object* v_reuseFailAlloc_1460_; 
v_reuseFailAlloc_1460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1460_, 0, v___x_1457_);
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
}
else
{
lean_object* v___x_1464_; lean_object* v___x_1466_; 
lean_dec(v_a_1443_);
v___x_1464_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0));
if (v_isShared_1446_ == 0)
{
lean_ctor_set(v___x_1445_, 0, v___x_1464_);
v___x_1466_ = v___x_1445_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v___x_1464_);
v___x_1466_ = v_reuseFailAlloc_1467_;
goto v_reusejp_1465_;
}
v_reusejp_1465_:
{
return v___x_1466_;
}
}
}
}
else
{
lean_object* v_a_1469_; lean_object* v___x_1471_; uint8_t v_isShared_1472_; uint8_t v_isSharedCheck_1476_; 
v_a_1469_ = lean_ctor_get(v___x_1442_, 0);
v_isSharedCheck_1476_ = !lean_is_exclusive(v___x_1442_);
if (v_isSharedCheck_1476_ == 0)
{
v___x_1471_ = v___x_1442_;
v_isShared_1472_ = v_isSharedCheck_1476_;
goto v_resetjp_1470_;
}
else
{
lean_inc(v_a_1469_);
lean_dec(v___x_1442_);
v___x_1471_ = lean_box(0);
v_isShared_1472_ = v_isSharedCheck_1476_;
goto v_resetjp_1470_;
}
v_resetjp_1470_:
{
lean_object* v___x_1474_; 
if (v_isShared_1472_ == 0)
{
v___x_1474_ = v___x_1471_;
goto v_reusejp_1473_;
}
else
{
lean_object* v_reuseFailAlloc_1475_; 
v_reuseFailAlloc_1475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1475_, 0, v_a_1469_);
v___x_1474_ = v_reuseFailAlloc_1475_;
goto v_reusejp_1473_;
}
v_reusejp_1473_:
{
return v___x_1474_;
}
}
}
}
}
else
{
lean_object* v___x_1477_; lean_object* v___x_1478_; 
lean_dec_ref(v_node_1431_);
v___x_1477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0));
v___x_1478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1478_, 0, v___x_1477_);
return v___x_1478_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___boxed(lean_object* v_fvars_1479_, lean_object* v_node_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_){
_start:
{
lean_object* v_res_1486_; 
v_res_1486_ = lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1(v_fvars_1479_, v_node_1480_, v___y_1481_, v___y_1482_, v___y_1483_, v___y_1484_);
lean_dec(v___y_1484_);
lean_dec_ref(v___y_1483_);
lean_dec(v___y_1482_);
lean_dec_ref(v___y_1481_);
lean_dec_ref(v_fvars_1479_);
return v_res_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___lam__0(lean_object* v_00_u03b1_1487_, lean_object* v_x_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_){
_start:
{
lean_object* v___x_1494_; lean_object* v___x_1495_; 
v___x_1494_ = lean_apply_1(v_x_1488_, lean_box(0));
v___x_1495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1495_, 0, v___x_1494_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___lam__0___boxed(lean_object* v_00_u03b1_1496_, lean_object* v_x_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_){
_start:
{
lean_object* v_res_1503_; 
v_res_1503_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___lam__0(v_00_u03b1_1496_, v_x_1497_, v___y_1498_, v___y_1499_, v___y_1500_, v___y_1501_);
lean_dec(v___y_1501_);
lean_dec_ref(v___y_1500_);
lean_dec(v___y_1499_);
lean_dec_ref(v___y_1498_);
return v_res_1503_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___redArg(lean_object* v_a_1504_, lean_object* v_x_1505_){
_start:
{
if (lean_obj_tag(v_x_1505_) == 0)
{
uint8_t v___x_1506_; 
v___x_1506_ = 0;
return v___x_1506_;
}
else
{
lean_object* v_key_1507_; lean_object* v_tail_1508_; uint8_t v___x_1509_; 
v_key_1507_ = lean_ctor_get(v_x_1505_, 0);
v_tail_1508_ = lean_ctor_get(v_x_1505_, 2);
v___x_1509_ = l_Lean_ExprStructEq_beq(v_key_1507_, v_a_1504_);
if (v___x_1509_ == 0)
{
v_x_1505_ = v_tail_1508_;
goto _start;
}
else
{
return v___x_1509_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___redArg___boxed(lean_object* v_a_1511_, lean_object* v_x_1512_){
_start:
{
uint8_t v_res_1513_; lean_object* v_r_1514_; 
v_res_1513_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___redArg(v_a_1511_, v_x_1512_);
lean_dec(v_x_1512_);
lean_dec_ref(v_a_1511_);
v_r_1514_ = lean_box(v_res_1513_);
return v_r_1514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__19___redArg(lean_object* v_a_1515_, lean_object* v_b_1516_, lean_object* v_x_1517_){
_start:
{
if (lean_obj_tag(v_x_1517_) == 0)
{
lean_dec(v_b_1516_);
lean_dec_ref(v_a_1515_);
return v_x_1517_;
}
else
{
lean_object* v_key_1518_; lean_object* v_value_1519_; lean_object* v_tail_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1532_; 
v_key_1518_ = lean_ctor_get(v_x_1517_, 0);
v_value_1519_ = lean_ctor_get(v_x_1517_, 1);
v_tail_1520_ = lean_ctor_get(v_x_1517_, 2);
v_isSharedCheck_1532_ = !lean_is_exclusive(v_x_1517_);
if (v_isSharedCheck_1532_ == 0)
{
v___x_1522_ = v_x_1517_;
v_isShared_1523_ = v_isSharedCheck_1532_;
goto v_resetjp_1521_;
}
else
{
lean_inc(v_tail_1520_);
lean_inc(v_value_1519_);
lean_inc(v_key_1518_);
lean_dec(v_x_1517_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1532_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
uint8_t v___x_1524_; 
v___x_1524_ = l_Lean_ExprStructEq_beq(v_key_1518_, v_a_1515_);
if (v___x_1524_ == 0)
{
lean_object* v___x_1525_; lean_object* v___x_1527_; 
v___x_1525_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__19___redArg(v_a_1515_, v_b_1516_, v_tail_1520_);
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 2, v___x_1525_);
v___x_1527_ = v___x_1522_;
goto v_reusejp_1526_;
}
else
{
lean_object* v_reuseFailAlloc_1528_; 
v_reuseFailAlloc_1528_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1528_, 0, v_key_1518_);
lean_ctor_set(v_reuseFailAlloc_1528_, 1, v_value_1519_);
lean_ctor_set(v_reuseFailAlloc_1528_, 2, v___x_1525_);
v___x_1527_ = v_reuseFailAlloc_1528_;
goto v_reusejp_1526_;
}
v_reusejp_1526_:
{
return v___x_1527_;
}
}
else
{
lean_object* v___x_1530_; 
lean_dec(v_value_1519_);
lean_dec(v_key_1518_);
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 1, v_b_1516_);
lean_ctor_set(v___x_1522_, 0, v_a_1515_);
v___x_1530_ = v___x_1522_;
goto v_reusejp_1529_;
}
else
{
lean_object* v_reuseFailAlloc_1531_; 
v_reuseFailAlloc_1531_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1531_, 0, v_a_1515_);
lean_ctor_set(v_reuseFailAlloc_1531_, 1, v_b_1516_);
lean_ctor_set(v_reuseFailAlloc_1531_, 2, v_tail_1520_);
v___x_1530_ = v_reuseFailAlloc_1531_;
goto v_reusejp_1529_;
}
v_reusejp_1529_:
{
return v___x_1530_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19_spec__20___redArg(lean_object* v_x_1533_, lean_object* v_x_1534_){
_start:
{
if (lean_obj_tag(v_x_1534_) == 0)
{
return v_x_1533_;
}
else
{
lean_object* v_key_1535_; lean_object* v_value_1536_; lean_object* v_tail_1537_; lean_object* v___x_1539_; uint8_t v_isShared_1540_; uint8_t v_isSharedCheck_1560_; 
v_key_1535_ = lean_ctor_get(v_x_1534_, 0);
v_value_1536_ = lean_ctor_get(v_x_1534_, 1);
v_tail_1537_ = lean_ctor_get(v_x_1534_, 2);
v_isSharedCheck_1560_ = !lean_is_exclusive(v_x_1534_);
if (v_isSharedCheck_1560_ == 0)
{
v___x_1539_ = v_x_1534_;
v_isShared_1540_ = v_isSharedCheck_1560_;
goto v_resetjp_1538_;
}
else
{
lean_inc(v_tail_1537_);
lean_inc(v_value_1536_);
lean_inc(v_key_1535_);
lean_dec(v_x_1534_);
v___x_1539_ = lean_box(0);
v_isShared_1540_ = v_isSharedCheck_1560_;
goto v_resetjp_1538_;
}
v_resetjp_1538_:
{
lean_object* v___x_1541_; uint64_t v___x_1542_; uint64_t v___x_1543_; uint64_t v___x_1544_; uint64_t v_fold_1545_; uint64_t v___x_1546_; uint64_t v___x_1547_; uint64_t v___x_1548_; size_t v___x_1549_; size_t v___x_1550_; size_t v___x_1551_; size_t v___x_1552_; size_t v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1556_; 
v___x_1541_ = lean_array_get_size(v_x_1533_);
v___x_1542_ = l_Lean_ExprStructEq_hash(v_key_1535_);
v___x_1543_ = 32ULL;
v___x_1544_ = lean_uint64_shift_right(v___x_1542_, v___x_1543_);
v_fold_1545_ = lean_uint64_xor(v___x_1542_, v___x_1544_);
v___x_1546_ = 16ULL;
v___x_1547_ = lean_uint64_shift_right(v_fold_1545_, v___x_1546_);
v___x_1548_ = lean_uint64_xor(v_fold_1545_, v___x_1547_);
v___x_1549_ = lean_uint64_to_usize(v___x_1548_);
v___x_1550_ = lean_usize_of_nat(v___x_1541_);
v___x_1551_ = ((size_t)1ULL);
v___x_1552_ = lean_usize_sub(v___x_1550_, v___x_1551_);
v___x_1553_ = lean_usize_land(v___x_1549_, v___x_1552_);
v___x_1554_ = lean_array_uget_borrowed(v_x_1533_, v___x_1553_);
lean_inc(v___x_1554_);
if (v_isShared_1540_ == 0)
{
lean_ctor_set(v___x_1539_, 2, v___x_1554_);
v___x_1556_ = v___x_1539_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1559_; 
v_reuseFailAlloc_1559_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1559_, 0, v_key_1535_);
lean_ctor_set(v_reuseFailAlloc_1559_, 1, v_value_1536_);
lean_ctor_set(v_reuseFailAlloc_1559_, 2, v___x_1554_);
v___x_1556_ = v_reuseFailAlloc_1559_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
lean_object* v___x_1557_; 
v___x_1557_ = lean_array_uset(v_x_1533_, v___x_1553_, v___x_1556_);
v_x_1533_ = v___x_1557_;
v_x_1534_ = v_tail_1537_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19___redArg(lean_object* v_i_1561_, lean_object* v_source_1562_, lean_object* v_target_1563_){
_start:
{
lean_object* v___x_1564_; uint8_t v___x_1565_; 
v___x_1564_ = lean_array_get_size(v_source_1562_);
v___x_1565_ = lean_nat_dec_lt(v_i_1561_, v___x_1564_);
if (v___x_1565_ == 0)
{
lean_dec_ref(v_source_1562_);
lean_dec(v_i_1561_);
return v_target_1563_;
}
else
{
lean_object* v_es_1566_; lean_object* v___x_1567_; lean_object* v_source_1568_; lean_object* v_target_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; 
v_es_1566_ = lean_array_fget(v_source_1562_, v_i_1561_);
v___x_1567_ = lean_box(0);
v_source_1568_ = lean_array_fset(v_source_1562_, v_i_1561_, v___x_1567_);
v_target_1569_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19_spec__20___redArg(v_target_1563_, v_es_1566_);
v___x_1570_ = lean_unsigned_to_nat(1u);
v___x_1571_ = lean_nat_add(v_i_1561_, v___x_1570_);
lean_dec(v_i_1561_);
v_i_1561_ = v___x_1571_;
v_source_1562_ = v_source_1568_;
v_target_1563_ = v_target_1569_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18___redArg(lean_object* v_data_1573_){
_start:
{
lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v_nbuckets_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; 
v___x_1574_ = lean_array_get_size(v_data_1573_);
v___x_1575_ = lean_unsigned_to_nat(2u);
v_nbuckets_1576_ = lean_nat_mul(v___x_1574_, v___x_1575_);
v___x_1577_ = lean_unsigned_to_nat(0u);
v___x_1578_ = lean_box(0);
v___x_1579_ = lean_mk_array(v_nbuckets_1576_, v___x_1578_);
v___x_1580_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19___redArg(v___x_1577_, v_data_1573_, v___x_1579_);
return v___x_1580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12___redArg(lean_object* v_m_1581_, lean_object* v_a_1582_, lean_object* v_b_1583_){
_start:
{
lean_object* v_size_1584_; lean_object* v_buckets_1585_; lean_object* v___x_1587_; uint8_t v_isShared_1588_; uint8_t v_isSharedCheck_1628_; 
v_size_1584_ = lean_ctor_get(v_m_1581_, 0);
v_buckets_1585_ = lean_ctor_get(v_m_1581_, 1);
v_isSharedCheck_1628_ = !lean_is_exclusive(v_m_1581_);
if (v_isSharedCheck_1628_ == 0)
{
v___x_1587_ = v_m_1581_;
v_isShared_1588_ = v_isSharedCheck_1628_;
goto v_resetjp_1586_;
}
else
{
lean_inc(v_buckets_1585_);
lean_inc(v_size_1584_);
lean_dec(v_m_1581_);
v___x_1587_ = lean_box(0);
v_isShared_1588_ = v_isSharedCheck_1628_;
goto v_resetjp_1586_;
}
v_resetjp_1586_:
{
lean_object* v___x_1589_; uint64_t v___x_1590_; uint64_t v___x_1591_; uint64_t v___x_1592_; uint64_t v_fold_1593_; uint64_t v___x_1594_; uint64_t v___x_1595_; uint64_t v___x_1596_; size_t v___x_1597_; size_t v___x_1598_; size_t v___x_1599_; size_t v___x_1600_; size_t v___x_1601_; lean_object* v_bkt_1602_; uint8_t v___x_1603_; 
v___x_1589_ = lean_array_get_size(v_buckets_1585_);
v___x_1590_ = l_Lean_ExprStructEq_hash(v_a_1582_);
v___x_1591_ = 32ULL;
v___x_1592_ = lean_uint64_shift_right(v___x_1590_, v___x_1591_);
v_fold_1593_ = lean_uint64_xor(v___x_1590_, v___x_1592_);
v___x_1594_ = 16ULL;
v___x_1595_ = lean_uint64_shift_right(v_fold_1593_, v___x_1594_);
v___x_1596_ = lean_uint64_xor(v_fold_1593_, v___x_1595_);
v___x_1597_ = lean_uint64_to_usize(v___x_1596_);
v___x_1598_ = lean_usize_of_nat(v___x_1589_);
v___x_1599_ = ((size_t)1ULL);
v___x_1600_ = lean_usize_sub(v___x_1598_, v___x_1599_);
v___x_1601_ = lean_usize_land(v___x_1597_, v___x_1600_);
v_bkt_1602_ = lean_array_uget_borrowed(v_buckets_1585_, v___x_1601_);
v___x_1603_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___redArg(v_a_1582_, v_bkt_1602_);
if (v___x_1603_ == 0)
{
lean_object* v___x_1604_; lean_object* v_size_x27_1605_; lean_object* v___x_1606_; lean_object* v_buckets_x27_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; uint8_t v___x_1613_; 
v___x_1604_ = lean_unsigned_to_nat(1u);
v_size_x27_1605_ = lean_nat_add(v_size_1584_, v___x_1604_);
lean_dec(v_size_1584_);
lean_inc(v_bkt_1602_);
v___x_1606_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1606_, 0, v_a_1582_);
lean_ctor_set(v___x_1606_, 1, v_b_1583_);
lean_ctor_set(v___x_1606_, 2, v_bkt_1602_);
v_buckets_x27_1607_ = lean_array_uset(v_buckets_1585_, v___x_1601_, v___x_1606_);
v___x_1608_ = lean_unsigned_to_nat(4u);
v___x_1609_ = lean_nat_mul(v_size_x27_1605_, v___x_1608_);
v___x_1610_ = lean_unsigned_to_nat(3u);
v___x_1611_ = lean_nat_div(v___x_1609_, v___x_1610_);
lean_dec(v___x_1609_);
v___x_1612_ = lean_array_get_size(v_buckets_x27_1607_);
v___x_1613_ = lean_nat_dec_le(v___x_1611_, v___x_1612_);
lean_dec(v___x_1611_);
if (v___x_1613_ == 0)
{
lean_object* v_val_1614_; lean_object* v___x_1616_; 
v_val_1614_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18___redArg(v_buckets_x27_1607_);
if (v_isShared_1588_ == 0)
{
lean_ctor_set(v___x_1587_, 1, v_val_1614_);
lean_ctor_set(v___x_1587_, 0, v_size_x27_1605_);
v___x_1616_ = v___x_1587_;
goto v_reusejp_1615_;
}
else
{
lean_object* v_reuseFailAlloc_1617_; 
v_reuseFailAlloc_1617_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1617_, 0, v_size_x27_1605_);
lean_ctor_set(v_reuseFailAlloc_1617_, 1, v_val_1614_);
v___x_1616_ = v_reuseFailAlloc_1617_;
goto v_reusejp_1615_;
}
v_reusejp_1615_:
{
return v___x_1616_;
}
}
else
{
lean_object* v___x_1619_; 
if (v_isShared_1588_ == 0)
{
lean_ctor_set(v___x_1587_, 1, v_buckets_x27_1607_);
lean_ctor_set(v___x_1587_, 0, v_size_x27_1605_);
v___x_1619_ = v___x_1587_;
goto v_reusejp_1618_;
}
else
{
lean_object* v_reuseFailAlloc_1620_; 
v_reuseFailAlloc_1620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1620_, 0, v_size_x27_1605_);
lean_ctor_set(v_reuseFailAlloc_1620_, 1, v_buckets_x27_1607_);
v___x_1619_ = v_reuseFailAlloc_1620_;
goto v_reusejp_1618_;
}
v_reusejp_1618_:
{
return v___x_1619_;
}
}
}
else
{
lean_object* v___x_1621_; lean_object* v_buckets_x27_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1626_; 
lean_inc(v_bkt_1602_);
v___x_1621_ = lean_box(0);
v_buckets_x27_1622_ = lean_array_uset(v_buckets_1585_, v___x_1601_, v___x_1621_);
v___x_1623_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__19___redArg(v_a_1582_, v_b_1583_, v_bkt_1602_);
v___x_1624_ = lean_array_uset(v_buckets_x27_1622_, v___x_1601_, v___x_1623_);
if (v_isShared_1588_ == 0)
{
lean_ctor_set(v___x_1587_, 1, v___x_1624_);
v___x_1626_ = v___x_1587_;
goto v_reusejp_1625_;
}
else
{
lean_object* v_reuseFailAlloc_1627_; 
v_reuseFailAlloc_1627_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1627_, 0, v_size_1584_);
lean_ctor_set(v_reuseFailAlloc_1627_, 1, v___x_1624_);
v___x_1626_ = v_reuseFailAlloc_1627_;
goto v_reusejp_1625_;
}
v_reusejp_1625_:
{
return v___x_1626_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__2(lean_object* v_a_1629_, lean_object* v_e_1630_, lean_object* v_a_1631_){
_start:
{
lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; 
v___x_1633_ = lean_st_ref_take(v_a_1629_);
v___x_1634_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12___redArg(v___x_1633_, v_e_1630_, v_a_1631_);
v___x_1635_ = lean_st_ref_set(v_a_1629_, v___x_1634_);
v___x_1636_ = lean_box(0);
return v___x_1636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__2___boxed(lean_object* v_a_1637_, lean_object* v_e_1638_, lean_object* v_a_1639_, lean_object* v___y_1640_){
_start:
{
lean_object* v_res_1641_; 
v_res_1641_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__2(v_a_1637_, v_e_1638_, v_a_1639_);
lean_dec(v_a_1637_);
return v_res_1641_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__3(void){
_start:
{
lean_object* v___x_1647_; lean_object* v___x_1648_; 
v___x_1647_ = l_Lean_maxRecDepthErrorMessage;
v___x_1648_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1648_, 0, v___x_1647_);
return v___x_1648_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__4(void){
_start:
{
lean_object* v___x_1649_; lean_object* v___x_1650_; 
v___x_1649_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__3, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__3_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__3);
v___x_1650_ = l_Lean_MessageData_ofFormat(v___x_1649_);
return v___x_1650_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__5(void){
_start:
{
lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; 
v___x_1651_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__4, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__4_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__4);
v___x_1652_ = ((lean_object*)(lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__2));
v___x_1653_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1653_, 0, v___x_1652_);
lean_ctor_set(v___x_1653_, 1, v___x_1651_);
return v___x_1653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg(lean_object* v_ref_1654_){
_start:
{
lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; 
v___x_1656_ = lean_obj_once(&lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__5, &lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__5_once, _init_lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___closed__5);
v___x_1657_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1657_, 0, v_ref_1654_);
lean_ctor_set(v___x_1657_, 1, v___x_1656_);
v___x_1658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1658_, 0, v___x_1657_);
return v___x_1658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg___boxed(lean_object* v_ref_1659_, lean_object* v___y_1660_){
_start:
{
lean_object* v_res_1661_; 
v_res_1661_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg(v_ref_1659_);
return v_res_1661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___redArg(lean_object* v_x_1662_, lean_object* v___y_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_, lean_object* v___y_1666_, lean_object* v___y_1667_){
_start:
{
lean_object* v___y_1670_; lean_object* v_fileName_1679_; lean_object* v_fileMap_1680_; lean_object* v_options_1681_; lean_object* v_currRecDepth_1682_; lean_object* v_maxRecDepth_1683_; lean_object* v_ref_1684_; lean_object* v_currNamespace_1685_; lean_object* v_openDecls_1686_; lean_object* v_initHeartbeats_1687_; lean_object* v_maxHeartbeats_1688_; lean_object* v_quotContext_1689_; lean_object* v_currMacroScope_1690_; uint8_t v_diag_1691_; lean_object* v_cancelTk_x3f_1692_; uint8_t v_suppressElabErrors_1693_; lean_object* v_inheritedTraceOptions_1694_; lean_object* v___x_1700_; uint8_t v___x_1701_; 
v_fileName_1679_ = lean_ctor_get(v___y_1666_, 0);
v_fileMap_1680_ = lean_ctor_get(v___y_1666_, 1);
v_options_1681_ = lean_ctor_get(v___y_1666_, 2);
v_currRecDepth_1682_ = lean_ctor_get(v___y_1666_, 3);
v_maxRecDepth_1683_ = lean_ctor_get(v___y_1666_, 4);
v_ref_1684_ = lean_ctor_get(v___y_1666_, 5);
v_currNamespace_1685_ = lean_ctor_get(v___y_1666_, 6);
v_openDecls_1686_ = lean_ctor_get(v___y_1666_, 7);
v_initHeartbeats_1687_ = lean_ctor_get(v___y_1666_, 8);
v_maxHeartbeats_1688_ = lean_ctor_get(v___y_1666_, 9);
v_quotContext_1689_ = lean_ctor_get(v___y_1666_, 10);
v_currMacroScope_1690_ = lean_ctor_get(v___y_1666_, 11);
v_diag_1691_ = lean_ctor_get_uint8(v___y_1666_, sizeof(void*)*14);
v_cancelTk_x3f_1692_ = lean_ctor_get(v___y_1666_, 12);
v_suppressElabErrors_1693_ = lean_ctor_get_uint8(v___y_1666_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1694_ = lean_ctor_get(v___y_1666_, 13);
v___x_1700_ = lean_unsigned_to_nat(0u);
v___x_1701_ = lean_nat_dec_eq(v_maxRecDepth_1683_, v___x_1700_);
if (v___x_1701_ == 0)
{
uint8_t v___x_1702_; 
v___x_1702_ = lean_nat_dec_eq(v_currRecDepth_1682_, v_maxRecDepth_1683_);
if (v___x_1702_ == 0)
{
goto v___jp_1695_;
}
else
{
lean_object* v___x_1703_; 
lean_dec_ref(v_x_1662_);
lean_inc(v_ref_1684_);
v___x_1703_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg(v_ref_1684_);
v___y_1670_ = v___x_1703_;
goto v___jp_1669_;
}
}
else
{
goto v___jp_1695_;
}
v___jp_1669_:
{
if (lean_obj_tag(v___y_1670_) == 0)
{
return v___y_1670_;
}
else
{
lean_object* v_a_1671_; lean_object* v___x_1673_; uint8_t v_isShared_1674_; uint8_t v_isSharedCheck_1678_; 
v_a_1671_ = lean_ctor_get(v___y_1670_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___y_1670_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1673_ = v___y_1670_;
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
else
{
lean_inc(v_a_1671_);
lean_dec(v___y_1670_);
v___x_1673_ = lean_box(0);
v_isShared_1674_ = v_isSharedCheck_1678_;
goto v_resetjp_1672_;
}
v_resetjp_1672_:
{
lean_object* v___x_1676_; 
if (v_isShared_1674_ == 0)
{
v___x_1676_ = v___x_1673_;
goto v_reusejp_1675_;
}
else
{
lean_object* v_reuseFailAlloc_1677_; 
v_reuseFailAlloc_1677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1677_, 0, v_a_1671_);
v___x_1676_ = v_reuseFailAlloc_1677_;
goto v_reusejp_1675_;
}
v_reusejp_1675_:
{
return v___x_1676_;
}
}
}
}
v___jp_1695_:
{
lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; 
v___x_1696_ = lean_unsigned_to_nat(1u);
v___x_1697_ = lean_nat_add(v_currRecDepth_1682_, v___x_1696_);
lean_inc_ref(v_inheritedTraceOptions_1694_);
lean_inc(v_cancelTk_x3f_1692_);
lean_inc(v_currMacroScope_1690_);
lean_inc(v_quotContext_1689_);
lean_inc(v_maxHeartbeats_1688_);
lean_inc(v_initHeartbeats_1687_);
lean_inc(v_openDecls_1686_);
lean_inc(v_currNamespace_1685_);
lean_inc(v_ref_1684_);
lean_inc(v_maxRecDepth_1683_);
lean_inc_ref(v_options_1681_);
lean_inc_ref(v_fileMap_1680_);
lean_inc_ref(v_fileName_1679_);
v___x_1698_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1698_, 0, v_fileName_1679_);
lean_ctor_set(v___x_1698_, 1, v_fileMap_1680_);
lean_ctor_set(v___x_1698_, 2, v_options_1681_);
lean_ctor_set(v___x_1698_, 3, v___x_1697_);
lean_ctor_set(v___x_1698_, 4, v_maxRecDepth_1683_);
lean_ctor_set(v___x_1698_, 5, v_ref_1684_);
lean_ctor_set(v___x_1698_, 6, v_currNamespace_1685_);
lean_ctor_set(v___x_1698_, 7, v_openDecls_1686_);
lean_ctor_set(v___x_1698_, 8, v_initHeartbeats_1687_);
lean_ctor_set(v___x_1698_, 9, v_maxHeartbeats_1688_);
lean_ctor_set(v___x_1698_, 10, v_quotContext_1689_);
lean_ctor_set(v___x_1698_, 11, v_currMacroScope_1690_);
lean_ctor_set(v___x_1698_, 12, v_cancelTk_x3f_1692_);
lean_ctor_set(v___x_1698_, 13, v_inheritedTraceOptions_1694_);
lean_ctor_set_uint8(v___x_1698_, sizeof(void*)*14, v_diag_1691_);
lean_ctor_set_uint8(v___x_1698_, sizeof(void*)*14 + 1, v_suppressElabErrors_1693_);
lean_inc(v___y_1667_);
lean_inc(v___y_1665_);
lean_inc_ref(v___y_1664_);
lean_inc(v___y_1663_);
v___x_1699_ = lean_apply_6(v_x_1662_, v___y_1663_, v___y_1664_, v___y_1665_, v___x_1698_, v___y_1667_, lean_box(0));
v___y_1670_ = v___x_1699_;
goto v___jp_1669_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___redArg___boxed(lean_object* v_x_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_){
_start:
{
lean_object* v_res_1711_; 
v_res_1711_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___redArg(v_x_1704_, v___y_1705_, v___y_1706_, v___y_1707_, v___y_1708_, v___y_1709_);
lean_dec(v___y_1709_);
lean_dec_ref(v___y_1708_);
lean_dec(v___y_1707_);
lean_dec_ref(v___y_1706_);
lean_dec(v___y_1705_);
return v_res_1711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___lam__0(lean_object* v_k_1712_, lean_object* v___y_1713_, lean_object* v_b_1714_, lean_object* v___y_1715_, lean_object* v___y_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_){
_start:
{
lean_object* v___x_1720_; 
lean_inc(v___y_1718_);
lean_inc_ref(v___y_1717_);
lean_inc(v___y_1716_);
lean_inc_ref(v___y_1715_);
lean_inc(v___y_1713_);
v___x_1720_ = lean_apply_7(v_k_1712_, v_b_1714_, v___y_1713_, v___y_1715_, v___y_1716_, v___y_1717_, v___y_1718_, lean_box(0));
return v___x_1720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___lam__0___boxed(lean_object* v_k_1721_, lean_object* v___y_1722_, lean_object* v_b_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_, lean_object* v___y_1728_){
_start:
{
lean_object* v_res_1729_; 
v_res_1729_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___lam__0(v_k_1721_, v___y_1722_, v_b_1723_, v___y_1724_, v___y_1725_, v___y_1726_, v___y_1727_);
lean_dec(v___y_1727_);
lean_dec_ref(v___y_1726_);
lean_dec(v___y_1725_);
lean_dec_ref(v___y_1724_);
lean_dec(v___y_1722_);
return v_res_1729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg(lean_object* v_name_1730_, uint8_t v_bi_1731_, lean_object* v_type_1732_, lean_object* v_k_1733_, uint8_t v_kind_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_){
_start:
{
lean_object* v___f_1741_; lean_object* v___x_1742_; 
lean_inc(v___y_1735_);
v___f_1741_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1741_, 0, v_k_1733_);
lean_closure_set(v___f_1741_, 1, v___y_1735_);
v___x_1742_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1730_, v_bi_1731_, v_type_1732_, v___f_1741_, v_kind_1734_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_);
if (lean_obj_tag(v___x_1742_) == 0)
{
return v___x_1742_;
}
else
{
lean_object* v_a_1743_; lean_object* v___x_1745_; uint8_t v_isShared_1746_; uint8_t v_isSharedCheck_1750_; 
v_a_1743_ = lean_ctor_get(v___x_1742_, 0);
v_isSharedCheck_1750_ = !lean_is_exclusive(v___x_1742_);
if (v_isSharedCheck_1750_ == 0)
{
v___x_1745_ = v___x_1742_;
v_isShared_1746_ = v_isSharedCheck_1750_;
goto v_resetjp_1744_;
}
else
{
lean_inc(v_a_1743_);
lean_dec(v___x_1742_);
v___x_1745_ = lean_box(0);
v_isShared_1746_ = v_isSharedCheck_1750_;
goto v_resetjp_1744_;
}
v_resetjp_1744_:
{
lean_object* v___x_1748_; 
if (v_isShared_1746_ == 0)
{
v___x_1748_ = v___x_1745_;
goto v_reusejp_1747_;
}
else
{
lean_object* v_reuseFailAlloc_1749_; 
v_reuseFailAlloc_1749_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1749_, 0, v_a_1743_);
v___x_1748_ = v_reuseFailAlloc_1749_;
goto v_reusejp_1747_;
}
v_reusejp_1747_:
{
return v___x_1748_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___boxed(lean_object* v_name_1751_, lean_object* v_bi_1752_, lean_object* v_type_1753_, lean_object* v_k_1754_, lean_object* v_kind_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_){
_start:
{
uint8_t v_bi_boxed_1762_; uint8_t v_kind_boxed_1763_; lean_object* v_res_1764_; 
v_bi_boxed_1762_ = lean_unbox(v_bi_1752_);
v_kind_boxed_1763_ = lean_unbox(v_kind_1755_);
v_res_1764_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg(v_name_1751_, v_bi_boxed_1762_, v_type_1753_, v_k_1754_, v_kind_boxed_1763_, v___y_1756_, v___y_1757_, v___y_1758_, v___y_1759_, v___y_1760_);
lean_dec(v___y_1760_);
lean_dec_ref(v___y_1759_);
lean_dec(v___y_1758_);
lean_dec_ref(v___y_1757_);
lean_dec(v___y_1756_);
return v_res_1764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__2(lean_object* v___x_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_){
_start:
{
lean_object* v___x_1771_; 
v___x_1771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1771_, 0, v___x_1765_);
return v___x_1771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__2___boxed(lean_object* v___x_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_){
_start:
{
lean_object* v_res_1778_; 
v_res_1778_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__2(v___x_1772_, v___y_1773_, v___y_1774_, v___y_1775_, v___y_1776_);
lean_dec(v___y_1776_);
lean_dec_ref(v___y_1775_);
lean_dec(v___y_1774_);
lean_dec_ref(v___y_1773_);
return v_res_1778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___redArg(lean_object* v_name_1779_, lean_object* v_type_1780_, lean_object* v_val_1781_, lean_object* v_k_1782_, uint8_t v_nondep_1783_, uint8_t v_kind_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_){
_start:
{
lean_object* v___f_1791_; lean_object* v___x_1792_; 
lean_inc(v___y_1785_);
v___f_1791_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1791_, 0, v_k_1782_);
lean_closure_set(v___f_1791_, 1, v___y_1785_);
v___x_1792_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_1779_, v_type_1780_, v_val_1781_, v___f_1791_, v_nondep_1783_, v_kind_1784_, v___y_1786_, v___y_1787_, v___y_1788_, v___y_1789_);
if (lean_obj_tag(v___x_1792_) == 0)
{
return v___x_1792_;
}
else
{
lean_object* v_a_1793_; lean_object* v___x_1795_; uint8_t v_isShared_1796_; uint8_t v_isSharedCheck_1800_; 
v_a_1793_ = lean_ctor_get(v___x_1792_, 0);
v_isSharedCheck_1800_ = !lean_is_exclusive(v___x_1792_);
if (v_isSharedCheck_1800_ == 0)
{
v___x_1795_ = v___x_1792_;
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
else
{
lean_inc(v_a_1793_);
lean_dec(v___x_1792_);
v___x_1795_ = lean_box(0);
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
v_resetjp_1794_:
{
lean_object* v___x_1798_; 
if (v_isShared_1796_ == 0)
{
v___x_1798_ = v___x_1795_;
goto v_reusejp_1797_;
}
else
{
lean_object* v_reuseFailAlloc_1799_; 
v_reuseFailAlloc_1799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1799_, 0, v_a_1793_);
v___x_1798_ = v_reuseFailAlloc_1799_;
goto v_reusejp_1797_;
}
v_reusejp_1797_:
{
return v___x_1798_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___redArg___boxed(lean_object* v_name_1801_, lean_object* v_type_1802_, lean_object* v_val_1803_, lean_object* v_k_1804_, lean_object* v_nondep_1805_, lean_object* v_kind_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_){
_start:
{
uint8_t v_nondep_boxed_1813_; uint8_t v_kind_boxed_1814_; lean_object* v_res_1815_; 
v_nondep_boxed_1813_ = lean_unbox(v_nondep_1805_);
v_kind_boxed_1814_ = lean_unbox(v_kind_1806_);
v_res_1815_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___redArg(v_name_1801_, v_type_1802_, v_val_1803_, v_k_1804_, v_nondep_boxed_1813_, v_kind_boxed_1814_, v___y_1807_, v___y_1808_, v___y_1809_, v___y_1810_, v___y_1811_);
lean_dec(v___y_1811_);
lean_dec_ref(v___y_1810_);
lean_dec(v___y_1809_);
lean_dec_ref(v___y_1808_);
lean_dec(v___y_1807_);
return v_res_1815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__0(lean_object* v_00_u03b1_1816_, lean_object* v_x_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_){
_start:
{
lean_object* v___x_1823_; lean_object* v___x_1824_; 
v___x_1823_ = lean_apply_1(v_x_1817_, lean_box(0));
v___x_1824_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1824_, 0, v___x_1823_);
return v___x_1824_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__0___boxed(lean_object* v_00_u03b1_1825_, lean_object* v_x_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_, lean_object* v___y_1831_){
_start:
{
lean_object* v_res_1832_; 
v_res_1832_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__0(v_00_u03b1_1825_, v_x_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_);
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec_ref(v___y_1827_);
return v_res_1832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___redArg(lean_object* v_a_1833_, lean_object* v_x_1834_){
_start:
{
if (lean_obj_tag(v_x_1834_) == 0)
{
lean_object* v___x_1835_; 
v___x_1835_ = lean_box(0);
return v___x_1835_;
}
else
{
lean_object* v_key_1836_; lean_object* v_value_1837_; lean_object* v_tail_1838_; uint8_t v___x_1839_; 
v_key_1836_ = lean_ctor_get(v_x_1834_, 0);
v_value_1837_ = lean_ctor_get(v_x_1834_, 1);
v_tail_1838_ = lean_ctor_get(v_x_1834_, 2);
v___x_1839_ = l_Lean_ExprStructEq_beq(v_key_1836_, v_a_1833_);
if (v___x_1839_ == 0)
{
v_x_1834_ = v_tail_1838_;
goto _start;
}
else
{
lean_object* v___x_1841_; 
lean_inc(v_value_1837_);
v___x_1841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1841_, 0, v_value_1837_);
return v___x_1841_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___redArg___boxed(lean_object* v_a_1842_, lean_object* v_x_1843_){
_start:
{
lean_object* v_res_1844_; 
v_res_1844_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___redArg(v_a_1842_, v_x_1843_);
lean_dec(v_x_1843_);
lean_dec_ref(v_a_1842_);
return v_res_1844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___redArg(lean_object* v_m_1845_, lean_object* v_a_1846_){
_start:
{
lean_object* v_buckets_1847_; lean_object* v___x_1848_; uint64_t v___x_1849_; uint64_t v___x_1850_; uint64_t v___x_1851_; uint64_t v_fold_1852_; uint64_t v___x_1853_; uint64_t v___x_1854_; uint64_t v___x_1855_; size_t v___x_1856_; size_t v___x_1857_; size_t v___x_1858_; size_t v___x_1859_; size_t v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; 
v_buckets_1847_ = lean_ctor_get(v_m_1845_, 1);
v___x_1848_ = lean_array_get_size(v_buckets_1847_);
v___x_1849_ = l_Lean_ExprStructEq_hash(v_a_1846_);
v___x_1850_ = 32ULL;
v___x_1851_ = lean_uint64_shift_right(v___x_1849_, v___x_1850_);
v_fold_1852_ = lean_uint64_xor(v___x_1849_, v___x_1851_);
v___x_1853_ = 16ULL;
v___x_1854_ = lean_uint64_shift_right(v_fold_1852_, v___x_1853_);
v___x_1855_ = lean_uint64_xor(v_fold_1852_, v___x_1854_);
v___x_1856_ = lean_uint64_to_usize(v___x_1855_);
v___x_1857_ = lean_usize_of_nat(v___x_1848_);
v___x_1858_ = ((size_t)1ULL);
v___x_1859_ = lean_usize_sub(v___x_1857_, v___x_1858_);
v___x_1860_ = lean_usize_land(v___x_1856_, v___x_1859_);
v___x_1861_ = lean_array_uget_borrowed(v_buckets_1847_, v___x_1860_);
v___x_1862_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___redArg(v_a_1846_, v___x_1861_);
return v___x_1862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___redArg___boxed(lean_object* v_m_1863_, lean_object* v_a_1864_){
_start:
{
lean_object* v_res_1865_; 
v_res_1865_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___redArg(v_m_1863_, v_a_1864_);
lean_dec_ref(v_a_1864_);
lean_dec_ref(v_m_1863_);
return v_res_1865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___lam__0(lean_object* v_fvars_1869_, lean_object* v_pre_1870_, lean_object* v_post_1871_, uint8_t v_usedLetOnly_1872_, uint8_t v_skipConstInApp_1873_, uint8_t v_skipInstances_1874_, lean_object* v_body_1875_, lean_object* v_x_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_){
_start:
{
lean_object* v___x_1883_; lean_object* v___x_1884_; 
v___x_1883_ = lean_array_push(v_fvars_1869_, v_x_1876_);
v___x_1884_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8(v_pre_1870_, v_post_1871_, v_usedLetOnly_1872_, v_skipConstInApp_1873_, v_skipInstances_1874_, v___x_1883_, v_body_1875_, v___y_1877_, v___y_1878_, v___y_1879_, v___y_1880_, v___y_1881_);
return v___x_1884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___lam__0___boxed(lean_object* v_fvars_1885_, lean_object* v_pre_1886_, lean_object* v_post_1887_, lean_object* v_usedLetOnly_1888_, lean_object* v_skipConstInApp_1889_, lean_object* v_skipInstances_1890_, lean_object* v_body_1891_, lean_object* v_x_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_){
_start:
{
uint8_t v_usedLetOnly_boxed_1899_; uint8_t v_skipConstInApp_boxed_1900_; uint8_t v_skipInstances_boxed_1901_; lean_object* v_res_1902_; 
v_usedLetOnly_boxed_1899_ = lean_unbox(v_usedLetOnly_1888_);
v_skipConstInApp_boxed_1900_ = lean_unbox(v_skipConstInApp_1889_);
v_skipInstances_boxed_1901_ = lean_unbox(v_skipInstances_1890_);
v_res_1902_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___lam__0(v_fvars_1885_, v_pre_1886_, v_post_1887_, v_usedLetOnly_boxed_1899_, v_skipConstInApp_boxed_1900_, v_skipInstances_boxed_1901_, v_body_1891_, v_x_1892_, v___y_1893_, v___y_1894_, v___y_1895_, v___y_1896_, v___y_1897_);
lean_dec(v___y_1897_);
lean_dec_ref(v___y_1896_);
lean_dec(v___y_1895_);
lean_dec_ref(v___y_1894_);
lean_dec(v___y_1893_);
return v_res_1902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(lean_object* v_pre_1903_, lean_object* v_post_1904_, uint8_t v_usedLetOnly_1905_, uint8_t v_skipConstInApp_1906_, uint8_t v_skipInstances_1907_, lean_object* v_e_1908_, lean_object* v_a_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_){
_start:
{
lean_object* v___x_1915_; 
lean_inc_ref(v_post_1904_);
lean_inc(v___y_1913_);
lean_inc_ref(v___y_1912_);
lean_inc(v___y_1911_);
lean_inc_ref(v___y_1910_);
lean_inc_ref(v_e_1908_);
v___x_1915_ = lean_apply_6(v_post_1904_, v_e_1908_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, lean_box(0));
if (lean_obj_tag(v___x_1915_) == 0)
{
lean_object* v_a_1916_; lean_object* v___x_1918_; uint8_t v_isShared_1919_; uint8_t v_isSharedCheck_1934_; 
v_a_1916_ = lean_ctor_get(v___x_1915_, 0);
v_isSharedCheck_1934_ = !lean_is_exclusive(v___x_1915_);
if (v_isSharedCheck_1934_ == 0)
{
v___x_1918_ = v___x_1915_;
v_isShared_1919_ = v_isSharedCheck_1934_;
goto v_resetjp_1917_;
}
else
{
lean_inc(v_a_1916_);
lean_dec(v___x_1915_);
v___x_1918_ = lean_box(0);
v_isShared_1919_ = v_isSharedCheck_1934_;
goto v_resetjp_1917_;
}
v_resetjp_1917_:
{
switch(lean_obj_tag(v_a_1916_))
{
case 0:
{
lean_object* v_e_1920_; lean_object* v___x_1922_; 
lean_dec_ref(v_e_1908_);
lean_dec_ref(v_post_1904_);
lean_dec_ref(v_pre_1903_);
v_e_1920_ = lean_ctor_get(v_a_1916_, 0);
lean_inc_ref(v_e_1920_);
lean_dec_ref_known(v_a_1916_, 1);
if (v_isShared_1919_ == 0)
{
lean_ctor_set(v___x_1918_, 0, v_e_1920_);
v___x_1922_ = v___x_1918_;
goto v_reusejp_1921_;
}
else
{
lean_object* v_reuseFailAlloc_1923_; 
v_reuseFailAlloc_1923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1923_, 0, v_e_1920_);
v___x_1922_ = v_reuseFailAlloc_1923_;
goto v_reusejp_1921_;
}
v_reusejp_1921_:
{
return v___x_1922_;
}
}
case 1:
{
lean_object* v_e_1924_; lean_object* v___x_1925_; 
lean_del_object(v___x_1918_);
lean_dec_ref(v_e_1908_);
v_e_1924_ = lean_ctor_get(v_a_1916_, 0);
lean_inc_ref(v_e_1924_);
lean_dec_ref_known(v_a_1916_, 1);
v___x_1925_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_1903_, v_post_1904_, v_usedLetOnly_1905_, v_skipConstInApp_1906_, v_skipInstances_1907_, v_e_1924_, v_a_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_);
return v___x_1925_;
}
default: 
{
lean_object* v_e_x3f_1926_; 
lean_dec_ref(v_post_1904_);
lean_dec_ref(v_pre_1903_);
v_e_x3f_1926_ = lean_ctor_get(v_a_1916_, 0);
lean_inc(v_e_x3f_1926_);
lean_dec_ref_known(v_a_1916_, 1);
if (lean_obj_tag(v_e_x3f_1926_) == 0)
{
lean_object* v___x_1928_; 
if (v_isShared_1919_ == 0)
{
lean_ctor_set(v___x_1918_, 0, v_e_1908_);
v___x_1928_ = v___x_1918_;
goto v_reusejp_1927_;
}
else
{
lean_object* v_reuseFailAlloc_1929_; 
v_reuseFailAlloc_1929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1929_, 0, v_e_1908_);
v___x_1928_ = v_reuseFailAlloc_1929_;
goto v_reusejp_1927_;
}
v_reusejp_1927_:
{
return v___x_1928_;
}
}
else
{
lean_object* v_val_1930_; lean_object* v___x_1932_; 
lean_dec_ref(v_e_1908_);
v_val_1930_ = lean_ctor_get(v_e_x3f_1926_, 0);
lean_inc(v_val_1930_);
lean_dec_ref_known(v_e_x3f_1926_, 1);
if (v_isShared_1919_ == 0)
{
lean_ctor_set(v___x_1918_, 0, v_val_1930_);
v___x_1932_ = v___x_1918_;
goto v_reusejp_1931_;
}
else
{
lean_object* v_reuseFailAlloc_1933_; 
v_reuseFailAlloc_1933_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1933_, 0, v_val_1930_);
v___x_1932_ = v_reuseFailAlloc_1933_;
goto v_reusejp_1931_;
}
v_reusejp_1931_:
{
return v___x_1932_;
}
}
}
}
}
}
else
{
lean_object* v_a_1935_; lean_object* v___x_1937_; uint8_t v_isShared_1938_; uint8_t v_isSharedCheck_1942_; 
lean_dec_ref(v_e_1908_);
lean_dec_ref(v_post_1904_);
lean_dec_ref(v_pre_1903_);
v_a_1935_ = lean_ctor_get(v___x_1915_, 0);
v_isSharedCheck_1942_ = !lean_is_exclusive(v___x_1915_);
if (v_isSharedCheck_1942_ == 0)
{
v___x_1937_ = v___x_1915_;
v_isShared_1938_ = v_isSharedCheck_1942_;
goto v_resetjp_1936_;
}
else
{
lean_inc(v_a_1935_);
lean_dec(v___x_1915_);
v___x_1937_ = lean_box(0);
v_isShared_1938_ = v_isSharedCheck_1942_;
goto v_resetjp_1936_;
}
v_resetjp_1936_:
{
lean_object* v___x_1940_; 
if (v_isShared_1938_ == 0)
{
v___x_1940_ = v___x_1937_;
goto v_reusejp_1939_;
}
else
{
lean_object* v_reuseFailAlloc_1941_; 
v_reuseFailAlloc_1941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1941_, 0, v_a_1935_);
v___x_1940_ = v_reuseFailAlloc_1941_;
goto v_reusejp_1939_;
}
v_reusejp_1939_:
{
return v___x_1940_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8(lean_object* v_pre_1943_, lean_object* v_post_1944_, uint8_t v_usedLetOnly_1945_, uint8_t v_skipConstInApp_1946_, uint8_t v_skipInstances_1947_, lean_object* v_fvars_1948_, lean_object* v_e_1949_, lean_object* v_a_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_){
_start:
{
if (lean_obj_tag(v_e_1949_) == 6)
{
lean_object* v_binderName_1956_; lean_object* v_binderType_1957_; lean_object* v_body_1958_; uint8_t v_binderInfo_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; 
v_binderName_1956_ = lean_ctor_get(v_e_1949_, 0);
lean_inc(v_binderName_1956_);
v_binderType_1957_ = lean_ctor_get(v_e_1949_, 1);
lean_inc_ref(v_binderType_1957_);
v_body_1958_ = lean_ctor_get(v_e_1949_, 2);
lean_inc_ref(v_body_1958_);
v_binderInfo_1959_ = lean_ctor_get_uint8(v_e_1949_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_1949_, 3);
v___x_1960_ = lean_expr_instantiate_rev(v_binderType_1957_, v_fvars_1948_);
lean_dec_ref(v_binderType_1957_);
lean_inc_ref(v_post_1944_);
lean_inc_ref(v_pre_1943_);
v___x_1961_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_1943_, v_post_1944_, v_usedLetOnly_1945_, v_skipConstInApp_1946_, v_skipInstances_1947_, v___x_1960_, v_a_1950_, v___y_1951_, v___y_1952_, v___y_1953_, v___y_1954_);
if (lean_obj_tag(v___x_1961_) == 0)
{
lean_object* v_a_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___f_1966_; uint8_t v___x_1967_; lean_object* v___x_1968_; 
v_a_1962_ = lean_ctor_get(v___x_1961_, 0);
lean_inc(v_a_1962_);
lean_dec_ref_known(v___x_1961_, 1);
v___x_1963_ = lean_box(v_usedLetOnly_1945_);
v___x_1964_ = lean_box(v_skipConstInApp_1946_);
v___x_1965_ = lean_box(v_skipInstances_1947_);
v___f_1966_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___lam__0___boxed), 14, 7);
lean_closure_set(v___f_1966_, 0, v_fvars_1948_);
lean_closure_set(v___f_1966_, 1, v_pre_1943_);
lean_closure_set(v___f_1966_, 2, v_post_1944_);
lean_closure_set(v___f_1966_, 3, v___x_1963_);
lean_closure_set(v___f_1966_, 4, v___x_1964_);
lean_closure_set(v___f_1966_, 5, v___x_1965_);
lean_closure_set(v___f_1966_, 6, v_body_1958_);
v___x_1967_ = 0;
v___x_1968_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg(v_binderName_1956_, v_binderInfo_1959_, v_a_1962_, v___f_1966_, v___x_1967_, v_a_1950_, v___y_1951_, v___y_1952_, v___y_1953_, v___y_1954_);
return v___x_1968_;
}
else
{
lean_dec_ref(v_body_1958_);
lean_dec(v_binderName_1956_);
lean_dec_ref(v_fvars_1948_);
lean_dec_ref(v_post_1944_);
lean_dec_ref(v_pre_1943_);
return v___x_1961_;
}
}
else
{
lean_object* v___x_1969_; lean_object* v___x_1970_; 
v___x_1969_ = lean_expr_instantiate_rev(v_e_1949_, v_fvars_1948_);
lean_dec_ref(v_e_1949_);
lean_inc_ref(v_post_1944_);
lean_inc_ref(v_pre_1943_);
v___x_1970_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_1943_, v_post_1944_, v_usedLetOnly_1945_, v_skipConstInApp_1946_, v_skipInstances_1947_, v___x_1969_, v_a_1950_, v___y_1951_, v___y_1952_, v___y_1953_, v___y_1954_);
if (lean_obj_tag(v___x_1970_) == 0)
{
lean_object* v_a_1971_; uint8_t v___x_1972_; uint8_t v___x_1973_; uint8_t v___x_1974_; lean_object* v___x_1975_; 
v_a_1971_ = lean_ctor_get(v___x_1970_, 0);
lean_inc(v_a_1971_);
lean_dec_ref_known(v___x_1970_, 1);
v___x_1972_ = 0;
v___x_1973_ = 1;
v___x_1974_ = 1;
v___x_1975_ = l_Lean_Meta_mkLambdaFVars(v_fvars_1948_, v_a_1971_, v___x_1972_, v_usedLetOnly_1945_, v___x_1972_, v___x_1973_, v___x_1974_, v___y_1951_, v___y_1952_, v___y_1953_, v___y_1954_);
lean_dec_ref(v_fvars_1948_);
if (lean_obj_tag(v___x_1975_) == 0)
{
lean_object* v_a_1976_; lean_object* v___x_1977_; 
v_a_1976_ = lean_ctor_get(v___x_1975_, 0);
lean_inc(v_a_1976_);
lean_dec_ref_known(v___x_1975_, 1);
v___x_1977_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_1943_, v_post_1944_, v_usedLetOnly_1945_, v_skipConstInApp_1946_, v_skipInstances_1947_, v_a_1976_, v_a_1950_, v___y_1951_, v___y_1952_, v___y_1953_, v___y_1954_);
return v___x_1977_;
}
else
{
lean_dec_ref(v_post_1944_);
lean_dec_ref(v_pre_1943_);
return v___x_1975_;
}
}
else
{
lean_dec_ref(v_fvars_1948_);
lean_dec_ref(v_post_1944_);
lean_dec_ref(v_pre_1943_);
return v___x_1970_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___lam__0(lean_object* v_fvars_1978_, lean_object* v_pre_1979_, lean_object* v_post_1980_, uint8_t v_usedLetOnly_1981_, uint8_t v_skipConstInApp_1982_, uint8_t v_skipInstances_1983_, lean_object* v_body_1984_, lean_object* v_x_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_){
_start:
{
lean_object* v___x_1992_; lean_object* v___x_1993_; 
v___x_1992_ = lean_array_push(v_fvars_1978_, v_x_1985_);
v___x_1993_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9(v_pre_1979_, v_post_1980_, v_usedLetOnly_1981_, v_skipConstInApp_1982_, v_skipInstances_1983_, v___x_1992_, v_body_1984_, v___y_1986_, v___y_1987_, v___y_1988_, v___y_1989_, v___y_1990_);
return v___x_1993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___lam__0___boxed(lean_object* v_fvars_1994_, lean_object* v_pre_1995_, lean_object* v_post_1996_, lean_object* v_usedLetOnly_1997_, lean_object* v_skipConstInApp_1998_, lean_object* v_skipInstances_1999_, lean_object* v_body_2000_, lean_object* v_x_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_){
_start:
{
uint8_t v_usedLetOnly_boxed_2008_; uint8_t v_skipConstInApp_boxed_2009_; uint8_t v_skipInstances_boxed_2010_; lean_object* v_res_2011_; 
v_usedLetOnly_boxed_2008_ = lean_unbox(v_usedLetOnly_1997_);
v_skipConstInApp_boxed_2009_ = lean_unbox(v_skipConstInApp_1998_);
v_skipInstances_boxed_2010_ = lean_unbox(v_skipInstances_1999_);
v_res_2011_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___lam__0(v_fvars_1994_, v_pre_1995_, v_post_1996_, v_usedLetOnly_boxed_2008_, v_skipConstInApp_boxed_2009_, v_skipInstances_boxed_2010_, v_body_2000_, v_x_2001_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_, v___y_2006_);
lean_dec(v___y_2006_);
lean_dec_ref(v___y_2005_);
lean_dec(v___y_2004_);
lean_dec_ref(v___y_2003_);
lean_dec(v___y_2002_);
return v_res_2011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9(lean_object* v_pre_2012_, lean_object* v_post_2013_, uint8_t v_usedLetOnly_2014_, uint8_t v_skipConstInApp_2015_, uint8_t v_skipInstances_2016_, lean_object* v_fvars_2017_, lean_object* v_e_2018_, lean_object* v_a_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_, lean_object* v___y_2023_){
_start:
{
if (lean_obj_tag(v_e_2018_) == 8)
{
lean_object* v_declName_2025_; lean_object* v_type_2026_; lean_object* v_value_2027_; lean_object* v_body_2028_; uint8_t v_nondep_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; 
v_declName_2025_ = lean_ctor_get(v_e_2018_, 0);
lean_inc(v_declName_2025_);
v_type_2026_ = lean_ctor_get(v_e_2018_, 1);
lean_inc_ref(v_type_2026_);
v_value_2027_ = lean_ctor_get(v_e_2018_, 2);
lean_inc_ref(v_value_2027_);
v_body_2028_ = lean_ctor_get(v_e_2018_, 3);
lean_inc_ref(v_body_2028_);
v_nondep_2029_ = lean_ctor_get_uint8(v_e_2018_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_e_2018_, 4);
v___x_2030_ = lean_expr_instantiate_rev(v_type_2026_, v_fvars_2017_);
lean_dec_ref(v_type_2026_);
lean_inc_ref(v_post_2013_);
lean_inc_ref(v_pre_2012_);
v___x_2031_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2012_, v_post_2013_, v_usedLetOnly_2014_, v_skipConstInApp_2015_, v_skipInstances_2016_, v___x_2030_, v_a_2019_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_);
if (lean_obj_tag(v___x_2031_) == 0)
{
lean_object* v_a_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; 
v_a_2032_ = lean_ctor_get(v___x_2031_, 0);
lean_inc(v_a_2032_);
lean_dec_ref_known(v___x_2031_, 1);
v___x_2033_ = lean_expr_instantiate_rev(v_value_2027_, v_fvars_2017_);
lean_dec_ref(v_value_2027_);
lean_inc_ref(v_post_2013_);
lean_inc_ref(v_pre_2012_);
v___x_2034_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2012_, v_post_2013_, v_usedLetOnly_2014_, v_skipConstInApp_2015_, v_skipInstances_2016_, v___x_2033_, v_a_2019_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_);
if (lean_obj_tag(v___x_2034_) == 0)
{
lean_object* v_a_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___f_2039_; uint8_t v___x_2040_; lean_object* v___x_2041_; 
v_a_2035_ = lean_ctor_get(v___x_2034_, 0);
lean_inc(v_a_2035_);
lean_dec_ref_known(v___x_2034_, 1);
v___x_2036_ = lean_box(v_usedLetOnly_2014_);
v___x_2037_ = lean_box(v_skipConstInApp_2015_);
v___x_2038_ = lean_box(v_skipInstances_2016_);
v___f_2039_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___lam__0___boxed), 14, 7);
lean_closure_set(v___f_2039_, 0, v_fvars_2017_);
lean_closure_set(v___f_2039_, 1, v_pre_2012_);
lean_closure_set(v___f_2039_, 2, v_post_2013_);
lean_closure_set(v___f_2039_, 3, v___x_2036_);
lean_closure_set(v___f_2039_, 4, v___x_2037_);
lean_closure_set(v___f_2039_, 5, v___x_2038_);
lean_closure_set(v___f_2039_, 6, v_body_2028_);
v___x_2040_ = 0;
v___x_2041_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___redArg(v_declName_2025_, v_a_2032_, v_a_2035_, v___f_2039_, v_nondep_2029_, v___x_2040_, v_a_2019_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_);
return v___x_2041_;
}
else
{
lean_dec(v_a_2032_);
lean_dec_ref(v_body_2028_);
lean_dec(v_declName_2025_);
lean_dec_ref(v_fvars_2017_);
lean_dec_ref(v_post_2013_);
lean_dec_ref(v_pre_2012_);
return v___x_2034_;
}
}
else
{
lean_dec_ref(v_body_2028_);
lean_dec_ref(v_value_2027_);
lean_dec(v_declName_2025_);
lean_dec_ref(v_fvars_2017_);
lean_dec_ref(v_post_2013_);
lean_dec_ref(v_pre_2012_);
return v___x_2031_;
}
}
else
{
lean_object* v___x_2042_; lean_object* v___x_2043_; 
v___x_2042_ = lean_expr_instantiate_rev(v_e_2018_, v_fvars_2017_);
lean_dec_ref(v_e_2018_);
lean_inc_ref(v_post_2013_);
lean_inc_ref(v_pre_2012_);
v___x_2043_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2012_, v_post_2013_, v_usedLetOnly_2014_, v_skipConstInApp_2015_, v_skipInstances_2016_, v___x_2042_, v_a_2019_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_);
if (lean_obj_tag(v___x_2043_) == 0)
{
lean_object* v_a_2044_; uint8_t v___x_2045_; uint8_t v___x_2046_; lean_object* v___x_2047_; 
v_a_2044_ = lean_ctor_get(v___x_2043_, 0);
lean_inc(v_a_2044_);
lean_dec_ref_known(v___x_2043_, 1);
v___x_2045_ = 0;
v___x_2046_ = 1;
v___x_2047_ = l_Lean_Meta_mkLetFVars(v_fvars_2017_, v_a_2044_, v_usedLetOnly_2014_, v___x_2045_, v___x_2046_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_);
lean_dec_ref(v_fvars_2017_);
if (lean_obj_tag(v___x_2047_) == 0)
{
lean_object* v_a_2048_; lean_object* v___x_2049_; 
v_a_2048_ = lean_ctor_get(v___x_2047_, 0);
lean_inc(v_a_2048_);
lean_dec_ref_known(v___x_2047_, 1);
v___x_2049_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2012_, v_post_2013_, v_usedLetOnly_2014_, v_skipConstInApp_2015_, v_skipInstances_2016_, v_a_2048_, v_a_2019_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_);
return v___x_2049_;
}
else
{
lean_dec_ref(v_post_2013_);
lean_dec_ref(v_pre_2012_);
return v___x_2047_;
}
}
else
{
lean_dec_ref(v_fvars_2017_);
lean_dec_ref(v_post_2013_);
lean_dec_ref(v_pre_2012_);
return v___x_2043_;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1(void){
_start:
{
lean_object* v___x_2050_; lean_object* v_dummy_2051_; 
v___x_2050_ = lean_box(0);
v_dummy_2051_ = l_Lean_Expr_sort___override(v___x_2050_);
return v_dummy_2051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__3(lean_object* v_pre_2052_, lean_object* v_post_2053_, uint8_t v_usedLetOnly_2054_, uint8_t v_skipConstInApp_2055_, uint8_t v_skipInstances_2056_, size_t v_sz_2057_, size_t v_i_2058_, lean_object* v_bs_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_){
_start:
{
uint8_t v___x_2066_; 
v___x_2066_ = lean_usize_dec_lt(v_i_2058_, v_sz_2057_);
if (v___x_2066_ == 0)
{
lean_object* v___x_2067_; 
lean_dec_ref(v_post_2053_);
lean_dec_ref(v_pre_2052_);
v___x_2067_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2067_, 0, v_bs_2059_);
return v___x_2067_;
}
else
{
lean_object* v_v_2068_; lean_object* v___x_2069_; 
v_v_2068_ = lean_array_uget_borrowed(v_bs_2059_, v_i_2058_);
lean_inc(v_v_2068_);
lean_inc_ref(v_post_2053_);
lean_inc_ref(v_pre_2052_);
v___x_2069_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2052_, v_post_2053_, v_usedLetOnly_2054_, v_skipConstInApp_2055_, v_skipInstances_2056_, v_v_2068_, v___y_2060_, v___y_2061_, v___y_2062_, v___y_2063_, v___y_2064_);
if (lean_obj_tag(v___x_2069_) == 0)
{
lean_object* v_a_2070_; lean_object* v___x_2071_; lean_object* v_bs_x27_2072_; size_t v___x_2073_; size_t v___x_2074_; lean_object* v___x_2075_; 
v_a_2070_ = lean_ctor_get(v___x_2069_, 0);
lean_inc(v_a_2070_);
lean_dec_ref_known(v___x_2069_, 1);
v___x_2071_ = lean_unsigned_to_nat(0u);
v_bs_x27_2072_ = lean_array_uset(v_bs_2059_, v_i_2058_, v___x_2071_);
v___x_2073_ = ((size_t)1ULL);
v___x_2074_ = lean_usize_add(v_i_2058_, v___x_2073_);
v___x_2075_ = lean_array_uset(v_bs_x27_2072_, v_i_2058_, v_a_2070_);
v_i_2058_ = v___x_2074_;
v_bs_2059_ = v___x_2075_;
goto _start;
}
else
{
lean_object* v_a_2077_; lean_object* v___x_2079_; uint8_t v_isShared_2080_; uint8_t v_isSharedCheck_2084_; 
lean_dec_ref(v_bs_2059_);
lean_dec_ref(v_post_2053_);
lean_dec_ref(v_pre_2052_);
v_a_2077_ = lean_ctor_get(v___x_2069_, 0);
v_isSharedCheck_2084_ = !lean_is_exclusive(v___x_2069_);
if (v_isSharedCheck_2084_ == 0)
{
v___x_2079_ = v___x_2069_;
v_isShared_2080_ = v_isSharedCheck_2084_;
goto v_resetjp_2078_;
}
else
{
lean_inc(v_a_2077_);
lean_dec(v___x_2069_);
v___x_2079_ = lean_box(0);
v_isShared_2080_ = v_isSharedCheck_2084_;
goto v_resetjp_2078_;
}
v_resetjp_2078_:
{
lean_object* v___x_2082_; 
if (v_isShared_2080_ == 0)
{
v___x_2082_ = v___x_2079_;
goto v_reusejp_2081_;
}
else
{
lean_object* v_reuseFailAlloc_2083_; 
v_reuseFailAlloc_2083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2083_, 0, v_a_2077_);
v___x_2082_ = v_reuseFailAlloc_2083_;
goto v_reusejp_2081_;
}
v_reusejp_2081_:
{
return v___x_2082_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__0(lean_object* v_pre_2085_, lean_object* v_post_2086_, uint8_t v_usedLetOnly_2087_, uint8_t v_skipConstInApp_2088_, uint8_t v_skipInstances_2089_, lean_object* v___x_2090_, lean_object* v___y_2091_, lean_object* v_b_2092_, lean_object* v_a_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_){
_start:
{
lean_object* v___x_2099_; 
v___x_2099_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2085_, v_post_2086_, v_usedLetOnly_2087_, v_skipConstInApp_2088_, v_skipInstances_2089_, v___x_2090_, v___y_2091_, v___y_2094_, v___y_2095_, v___y_2096_, v___y_2097_);
if (lean_obj_tag(v___x_2099_) == 0)
{
lean_object* v_a_2100_; lean_object* v___x_2102_; uint8_t v_isShared_2103_; uint8_t v_isSharedCheck_2109_; 
v_a_2100_ = lean_ctor_get(v___x_2099_, 0);
v_isSharedCheck_2109_ = !lean_is_exclusive(v___x_2099_);
if (v_isSharedCheck_2109_ == 0)
{
v___x_2102_ = v___x_2099_;
v_isShared_2103_ = v_isSharedCheck_2109_;
goto v_resetjp_2101_;
}
else
{
lean_inc(v_a_2100_);
lean_dec(v___x_2099_);
v___x_2102_ = lean_box(0);
v_isShared_2103_ = v_isSharedCheck_2109_;
goto v_resetjp_2101_;
}
v_resetjp_2101_:
{
lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2107_; 
v___x_2104_ = lean_array_fset(v_b_2092_, v_a_2093_, v_a_2100_);
v___x_2105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2105_, 0, v___x_2104_);
if (v_isShared_2103_ == 0)
{
lean_ctor_set(v___x_2102_, 0, v___x_2105_);
v___x_2107_ = v___x_2102_;
goto v_reusejp_2106_;
}
else
{
lean_object* v_reuseFailAlloc_2108_; 
v_reuseFailAlloc_2108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2108_, 0, v___x_2105_);
v___x_2107_ = v_reuseFailAlloc_2108_;
goto v_reusejp_2106_;
}
v_reusejp_2106_:
{
return v___x_2107_;
}
}
}
else
{
lean_object* v_a_2110_; lean_object* v___x_2112_; uint8_t v_isShared_2113_; uint8_t v_isSharedCheck_2117_; 
lean_dec_ref(v_b_2092_);
v_a_2110_ = lean_ctor_get(v___x_2099_, 0);
v_isSharedCheck_2117_ = !lean_is_exclusive(v___x_2099_);
if (v_isSharedCheck_2117_ == 0)
{
v___x_2112_ = v___x_2099_;
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
else
{
lean_inc(v_a_2110_);
lean_dec(v___x_2099_);
v___x_2112_ = lean_box(0);
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
v_resetjp_2111_:
{
lean_object* v___x_2115_; 
if (v_isShared_2113_ == 0)
{
v___x_2115_ = v___x_2112_;
goto v_reusejp_2114_;
}
else
{
lean_object* v_reuseFailAlloc_2116_; 
v_reuseFailAlloc_2116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2116_, 0, v_a_2110_);
v___x_2115_ = v_reuseFailAlloc_2116_;
goto v_reusejp_2114_;
}
v_reusejp_2114_:
{
return v___x_2115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__0___boxed(lean_object* v_pre_2118_, lean_object* v_post_2119_, lean_object* v_usedLetOnly_2120_, lean_object* v_skipConstInApp_2121_, lean_object* v_skipInstances_2122_, lean_object* v___x_2123_, lean_object* v___y_2124_, lean_object* v_b_2125_, lean_object* v_a_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_){
_start:
{
uint8_t v_usedLetOnly_boxed_2132_; uint8_t v_skipConstInApp_boxed_2133_; uint8_t v_skipInstances_boxed_2134_; lean_object* v_res_2135_; 
v_usedLetOnly_boxed_2132_ = lean_unbox(v_usedLetOnly_2120_);
v_skipConstInApp_boxed_2133_ = lean_unbox(v_skipConstInApp_2121_);
v_skipInstances_boxed_2134_ = lean_unbox(v_skipInstances_2122_);
v_res_2135_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__0(v_pre_2118_, v_post_2119_, v_usedLetOnly_boxed_2132_, v_skipConstInApp_boxed_2133_, v_skipInstances_boxed_2134_, v___x_2123_, v___y_2124_, v_b_2125_, v_a_2126_, v___y_2127_, v___y_2128_, v___y_2129_, v___y_2130_);
lean_dec(v___y_2130_);
lean_dec_ref(v___y_2129_);
lean_dec(v___y_2128_);
lean_dec_ref(v___y_2127_);
lean_dec(v_a_2126_);
lean_dec(v___y_2124_);
return v_res_2135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg(lean_object* v_upperBound_2136_, lean_object* v___x_2137_, lean_object* v_pre_2138_, lean_object* v_post_2139_, uint8_t v_usedLetOnly_2140_, uint8_t v_skipConstInApp_2141_, uint8_t v_skipInstances_2142_, lean_object* v_a_2143_, lean_object* v_b_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_){
_start:
{
lean_object* v___y_2152_; uint8_t v___x_2175_; 
v___x_2175_ = lean_nat_dec_lt(v_a_2143_, v_upperBound_2136_);
if (v___x_2175_ == 0)
{
lean_object* v___x_2176_; 
lean_dec(v_a_2143_);
lean_dec_ref(v_post_2139_);
lean_dec_ref(v_pre_2138_);
v___x_2176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2176_, 0, v_b_2144_);
return v___x_2176_;
}
else
{
lean_object* v___x_2177_; lean_object* v___x_2178_; uint8_t v___x_2179_; 
v___x_2177_ = lean_array_fget_borrowed(v_b_2144_, v_a_2143_);
v___x_2178_ = lean_array_get_size(v___x_2137_);
v___x_2179_ = lean_nat_dec_lt(v_a_2143_, v___x_2178_);
if (v___x_2179_ == 0)
{
lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___f_2183_; 
lean_inc(v___x_2177_);
v___x_2180_ = lean_box(v_usedLetOnly_2140_);
v___x_2181_ = lean_box(v_skipConstInApp_2141_);
v___x_2182_ = lean_box(v_skipInstances_2142_);
lean_inc(v_a_2143_);
lean_inc(v___y_2145_);
lean_inc_ref(v_post_2139_);
lean_inc_ref(v_pre_2138_);
v___f_2183_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__0___boxed), 14, 9);
lean_closure_set(v___f_2183_, 0, v_pre_2138_);
lean_closure_set(v___f_2183_, 1, v_post_2139_);
lean_closure_set(v___f_2183_, 2, v___x_2180_);
lean_closure_set(v___f_2183_, 3, v___x_2181_);
lean_closure_set(v___f_2183_, 4, v___x_2182_);
lean_closure_set(v___f_2183_, 5, v___x_2177_);
lean_closure_set(v___f_2183_, 6, v___y_2145_);
lean_closure_set(v___f_2183_, 7, v_b_2144_);
lean_closure_set(v___f_2183_, 8, v_a_2143_);
v___y_2152_ = v___f_2183_;
goto v___jp_2151_;
}
else
{
lean_object* v___x_2184_; uint8_t v_isInstance_2185_; 
v___x_2184_ = lean_array_fget_borrowed(v___x_2137_, v_a_2143_);
v_isInstance_2185_ = lean_ctor_get_uint8(v___x_2184_, sizeof(void*)*1 + 4);
if (v_isInstance_2185_ == 0)
{
lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___f_2189_; 
lean_inc(v___x_2177_);
v___x_2186_ = lean_box(v_usedLetOnly_2140_);
v___x_2187_ = lean_box(v_skipConstInApp_2141_);
v___x_2188_ = lean_box(v_skipInstances_2142_);
lean_inc(v_a_2143_);
lean_inc(v___y_2145_);
lean_inc_ref(v_post_2139_);
lean_inc_ref(v_pre_2138_);
v___f_2189_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__0___boxed), 14, 9);
lean_closure_set(v___f_2189_, 0, v_pre_2138_);
lean_closure_set(v___f_2189_, 1, v_post_2139_);
lean_closure_set(v___f_2189_, 2, v___x_2186_);
lean_closure_set(v___f_2189_, 3, v___x_2187_);
lean_closure_set(v___f_2189_, 4, v___x_2188_);
lean_closure_set(v___f_2189_, 5, v___x_2177_);
lean_closure_set(v___f_2189_, 6, v___y_2145_);
lean_closure_set(v___f_2189_, 7, v_b_2144_);
lean_closure_set(v___f_2189_, 8, v_a_2143_);
v___y_2152_ = v___f_2189_;
goto v___jp_2151_;
}
else
{
lean_object* v___x_2190_; lean_object* v___f_2191_; 
v___x_2190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2190_, 0, v_b_2144_);
v___f_2191_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___lam__2___boxed), 6, 1);
lean_closure_set(v___f_2191_, 0, v___x_2190_);
v___y_2152_ = v___f_2191_;
goto v___jp_2151_;
}
}
}
v___jp_2151_:
{
lean_object* v___x_2153_; 
lean_inc(v___y_2149_);
lean_inc_ref(v___y_2148_);
lean_inc(v___y_2147_);
lean_inc_ref(v___y_2146_);
v___x_2153_ = lean_apply_5(v___y_2152_, v___y_2146_, v___y_2147_, v___y_2148_, v___y_2149_, lean_box(0));
if (lean_obj_tag(v___x_2153_) == 0)
{
lean_object* v_a_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2166_; 
v_a_2154_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2166_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2166_ == 0)
{
v___x_2156_ = v___x_2153_;
v_isShared_2157_ = v_isSharedCheck_2166_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_a_2154_);
lean_dec(v___x_2153_);
v___x_2156_ = lean_box(0);
v_isShared_2157_ = v_isSharedCheck_2166_;
goto v_resetjp_2155_;
}
v_resetjp_2155_:
{
if (lean_obj_tag(v_a_2154_) == 0)
{
lean_object* v_a_2158_; lean_object* v___x_2160_; 
lean_dec(v_a_2143_);
lean_dec_ref(v_post_2139_);
lean_dec_ref(v_pre_2138_);
v_a_2158_ = lean_ctor_get(v_a_2154_, 0);
lean_inc(v_a_2158_);
lean_dec_ref_known(v_a_2154_, 1);
if (v_isShared_2157_ == 0)
{
lean_ctor_set(v___x_2156_, 0, v_a_2158_);
v___x_2160_ = v___x_2156_;
goto v_reusejp_2159_;
}
else
{
lean_object* v_reuseFailAlloc_2161_; 
v_reuseFailAlloc_2161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2161_, 0, v_a_2158_);
v___x_2160_ = v_reuseFailAlloc_2161_;
goto v_reusejp_2159_;
}
v_reusejp_2159_:
{
return v___x_2160_;
}
}
else
{
lean_object* v_a_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; 
lean_del_object(v___x_2156_);
v_a_2162_ = lean_ctor_get(v_a_2154_, 0);
lean_inc(v_a_2162_);
lean_dec_ref_known(v_a_2154_, 1);
v___x_2163_ = lean_unsigned_to_nat(1u);
v___x_2164_ = lean_nat_add(v_a_2143_, v___x_2163_);
lean_dec(v_a_2143_);
v_a_2143_ = v___x_2164_;
v_b_2144_ = v_a_2162_;
goto _start;
}
}
}
else
{
lean_object* v_a_2167_; lean_object* v___x_2169_; uint8_t v_isShared_2170_; uint8_t v_isSharedCheck_2174_; 
lean_dec(v_a_2143_);
lean_dec_ref(v_post_2139_);
lean_dec_ref(v_pre_2138_);
v_a_2167_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2174_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2174_ == 0)
{
v___x_2169_ = v___x_2153_;
v_isShared_2170_ = v_isSharedCheck_2174_;
goto v_resetjp_2168_;
}
else
{
lean_inc(v_a_2167_);
lean_dec(v___x_2153_);
v___x_2169_ = lean_box(0);
v_isShared_2170_ = v_isSharedCheck_2174_;
goto v_resetjp_2168_;
}
v_resetjp_2168_:
{
lean_object* v___x_2172_; 
if (v_isShared_2170_ == 0)
{
v___x_2172_ = v___x_2169_;
goto v_reusejp_2171_;
}
else
{
lean_object* v_reuseFailAlloc_2173_; 
v_reuseFailAlloc_2173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2173_, 0, v_a_2167_);
v___x_2172_ = v_reuseFailAlloc_2173_;
goto v_reusejp_2171_;
}
v_reusejp_2171_:
{
return v___x_2172_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__10(uint8_t v_skipInstances_2192_, lean_object* v_pre_2193_, lean_object* v_post_2194_, uint8_t v_usedLetOnly_2195_, uint8_t v_skipConstInApp_2196_, lean_object* v_x_2197_, lean_object* v_x_2198_, lean_object* v_x_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_){
_start:
{
lean_object* v_f_2207_; lean_object* v___y_2208_; lean_object* v___y_2209_; lean_object* v___y_2210_; lean_object* v___y_2211_; lean_object* v___y_2212_; 
if (lean_obj_tag(v_x_2197_) == 5)
{
lean_object* v_fn_2255_; lean_object* v_arg_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; 
v_fn_2255_ = lean_ctor_get(v_x_2197_, 0);
lean_inc_ref(v_fn_2255_);
v_arg_2256_ = lean_ctor_get(v_x_2197_, 1);
lean_inc_ref(v_arg_2256_);
lean_dec_ref_known(v_x_2197_, 2);
v___x_2257_ = lean_array_set(v_x_2198_, v_x_2199_, v_arg_2256_);
v___x_2258_ = lean_unsigned_to_nat(1u);
v___x_2259_ = lean_nat_sub(v_x_2199_, v___x_2258_);
lean_dec(v_x_2199_);
v_x_2197_ = v_fn_2255_;
v_x_2198_ = v___x_2257_;
v_x_2199_ = v___x_2259_;
goto _start;
}
else
{
lean_dec(v_x_2199_);
if (v_skipConstInApp_2196_ == 0)
{
goto v___jp_2252_;
}
else
{
uint8_t v___x_2261_; 
v___x_2261_ = l_Lean_Expr_isConst(v_x_2197_);
if (v___x_2261_ == 0)
{
goto v___jp_2252_;
}
else
{
v_f_2207_ = v_x_2197_;
v___y_2208_ = v___y_2200_;
v___y_2209_ = v___y_2201_;
v___y_2210_ = v___y_2202_;
v___y_2211_ = v___y_2203_;
v___y_2212_ = v___y_2204_;
goto v___jp_2206_;
}
}
}
v___jp_2206_:
{
if (v_skipInstances_2192_ == 0)
{
size_t v_sz_2213_; size_t v___x_2214_; lean_object* v___x_2215_; 
v_sz_2213_ = lean_array_size(v_x_2198_);
v___x_2214_ = ((size_t)0ULL);
lean_inc_ref(v_post_2194_);
lean_inc_ref(v_pre_2193_);
v___x_2215_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__3(v_pre_2193_, v_post_2194_, v_usedLetOnly_2195_, v_skipConstInApp_2196_, v_skipInstances_2192_, v_sz_2213_, v___x_2214_, v_x_2198_, v___y_2208_, v___y_2209_, v___y_2210_, v___y_2211_, v___y_2212_);
if (lean_obj_tag(v___x_2215_) == 0)
{
lean_object* v_a_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; 
v_a_2216_ = lean_ctor_get(v___x_2215_, 0);
lean_inc(v_a_2216_);
lean_dec_ref_known(v___x_2215_, 1);
v___x_2217_ = l_Lean_mkAppN(v_f_2207_, v_a_2216_);
lean_dec(v_a_2216_);
v___x_2218_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2193_, v_post_2194_, v_usedLetOnly_2195_, v_skipConstInApp_2196_, v_skipInstances_2192_, v___x_2217_, v___y_2208_, v___y_2209_, v___y_2210_, v___y_2211_, v___y_2212_);
return v___x_2218_;
}
else
{
lean_object* v_a_2219_; lean_object* v___x_2221_; uint8_t v_isShared_2222_; uint8_t v_isSharedCheck_2226_; 
lean_dec_ref(v_f_2207_);
lean_dec_ref(v_post_2194_);
lean_dec_ref(v_pre_2193_);
v_a_2219_ = lean_ctor_get(v___x_2215_, 0);
v_isSharedCheck_2226_ = !lean_is_exclusive(v___x_2215_);
if (v_isSharedCheck_2226_ == 0)
{
v___x_2221_ = v___x_2215_;
v_isShared_2222_ = v_isSharedCheck_2226_;
goto v_resetjp_2220_;
}
else
{
lean_inc(v_a_2219_);
lean_dec(v___x_2215_);
v___x_2221_ = lean_box(0);
v_isShared_2222_ = v_isSharedCheck_2226_;
goto v_resetjp_2220_;
}
v_resetjp_2220_:
{
lean_object* v___x_2224_; 
if (v_isShared_2222_ == 0)
{
v___x_2224_ = v___x_2221_;
goto v_reusejp_2223_;
}
else
{
lean_object* v_reuseFailAlloc_2225_; 
v_reuseFailAlloc_2225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2225_, 0, v_a_2219_);
v___x_2224_ = v_reuseFailAlloc_2225_;
goto v_reusejp_2223_;
}
v_reusejp_2223_:
{
return v___x_2224_;
}
}
}
}
else
{
lean_object* v___x_2227_; lean_object* v___x_2228_; 
v___x_2227_ = lean_array_get_size(v_x_2198_);
lean_inc_ref(v_f_2207_);
v___x_2228_ = l_Lean_Meta_getFunInfoNArgs(v_f_2207_, v___x_2227_, v___y_2209_, v___y_2210_, v___y_2211_, v___y_2212_);
if (lean_obj_tag(v___x_2228_) == 0)
{
lean_object* v_a_2229_; lean_object* v_paramInfo_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; 
v_a_2229_ = lean_ctor_get(v___x_2228_, 0);
lean_inc(v_a_2229_);
lean_dec_ref_known(v___x_2228_, 1);
v_paramInfo_2230_ = lean_ctor_get(v_a_2229_, 0);
lean_inc_ref(v_paramInfo_2230_);
lean_dec(v_a_2229_);
v___x_2231_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_post_2194_);
lean_inc_ref(v_pre_2193_);
v___x_2232_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg(v___x_2227_, v_paramInfo_2230_, v_pre_2193_, v_post_2194_, v_usedLetOnly_2195_, v_skipConstInApp_2196_, v_skipInstances_2192_, v___x_2231_, v_x_2198_, v___y_2208_, v___y_2209_, v___y_2210_, v___y_2211_, v___y_2212_);
lean_dec_ref(v_paramInfo_2230_);
if (lean_obj_tag(v___x_2232_) == 0)
{
lean_object* v_a_2233_; lean_object* v___x_2234_; lean_object* v___x_2235_; 
v_a_2233_ = lean_ctor_get(v___x_2232_, 0);
lean_inc(v_a_2233_);
lean_dec_ref_known(v___x_2232_, 1);
v___x_2234_ = l_Lean_mkAppN(v_f_2207_, v_a_2233_);
lean_dec(v_a_2233_);
v___x_2235_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2193_, v_post_2194_, v_usedLetOnly_2195_, v_skipConstInApp_2196_, v_skipInstances_2192_, v___x_2234_, v___y_2208_, v___y_2209_, v___y_2210_, v___y_2211_, v___y_2212_);
return v___x_2235_;
}
else
{
lean_object* v_a_2236_; lean_object* v___x_2238_; uint8_t v_isShared_2239_; uint8_t v_isSharedCheck_2243_; 
lean_dec_ref(v_f_2207_);
lean_dec_ref(v_post_2194_);
lean_dec_ref(v_pre_2193_);
v_a_2236_ = lean_ctor_get(v___x_2232_, 0);
v_isSharedCheck_2243_ = !lean_is_exclusive(v___x_2232_);
if (v_isSharedCheck_2243_ == 0)
{
v___x_2238_ = v___x_2232_;
v_isShared_2239_ = v_isSharedCheck_2243_;
goto v_resetjp_2237_;
}
else
{
lean_inc(v_a_2236_);
lean_dec(v___x_2232_);
v___x_2238_ = lean_box(0);
v_isShared_2239_ = v_isSharedCheck_2243_;
goto v_resetjp_2237_;
}
v_resetjp_2237_:
{
lean_object* v___x_2241_; 
if (v_isShared_2239_ == 0)
{
v___x_2241_ = v___x_2238_;
goto v_reusejp_2240_;
}
else
{
lean_object* v_reuseFailAlloc_2242_; 
v_reuseFailAlloc_2242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2242_, 0, v_a_2236_);
v___x_2241_ = v_reuseFailAlloc_2242_;
goto v_reusejp_2240_;
}
v_reusejp_2240_:
{
return v___x_2241_;
}
}
}
}
else
{
lean_object* v_a_2244_; lean_object* v___x_2246_; uint8_t v_isShared_2247_; uint8_t v_isSharedCheck_2251_; 
lean_dec_ref(v_f_2207_);
lean_dec_ref(v_x_2198_);
lean_dec_ref(v_post_2194_);
lean_dec_ref(v_pre_2193_);
v_a_2244_ = lean_ctor_get(v___x_2228_, 0);
v_isSharedCheck_2251_ = !lean_is_exclusive(v___x_2228_);
if (v_isSharedCheck_2251_ == 0)
{
v___x_2246_ = v___x_2228_;
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
else
{
lean_inc(v_a_2244_);
lean_dec(v___x_2228_);
v___x_2246_ = lean_box(0);
v_isShared_2247_ = v_isSharedCheck_2251_;
goto v_resetjp_2245_;
}
v_resetjp_2245_:
{
lean_object* v___x_2249_; 
if (v_isShared_2247_ == 0)
{
v___x_2249_ = v___x_2246_;
goto v_reusejp_2248_;
}
else
{
lean_object* v_reuseFailAlloc_2250_; 
v_reuseFailAlloc_2250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2250_, 0, v_a_2244_);
v___x_2249_ = v_reuseFailAlloc_2250_;
goto v_reusejp_2248_;
}
v_reusejp_2248_:
{
return v___x_2249_;
}
}
}
}
}
v___jp_2252_:
{
lean_object* v___x_2253_; 
lean_inc_ref(v_post_2194_);
lean_inc_ref(v_pre_2193_);
v___x_2253_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2193_, v_post_2194_, v_usedLetOnly_2195_, v_skipConstInApp_2196_, v_skipInstances_2192_, v_x_2197_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_, v___y_2204_);
if (lean_obj_tag(v___x_2253_) == 0)
{
lean_object* v_a_2254_; 
v_a_2254_ = lean_ctor_get(v___x_2253_, 0);
lean_inc(v_a_2254_);
lean_dec_ref_known(v___x_2253_, 1);
v_f_2207_ = v_a_2254_;
v___y_2208_ = v___y_2200_;
v___y_2209_ = v___y_2201_;
v___y_2210_ = v___y_2202_;
v___y_2211_ = v___y_2203_;
v___y_2212_ = v___y_2204_;
goto v___jp_2206_;
}
else
{
lean_dec_ref(v_x_2198_);
lean_dec_ref(v_post_2194_);
lean_dec_ref(v_pre_2193_);
return v___x_2253_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1(lean_object* v___x_2262_, lean_object* v_pre_2263_, lean_object* v_e_2264_, lean_object* v_post_2265_, uint8_t v_usedLetOnly_2266_, uint8_t v_skipConstInApp_2267_, uint8_t v_skipInstances_2268_, lean_object* v___y_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_){
_start:
{
lean_object* v___x_2275_; 
v___x_2275_ = l_Lean_Core_checkSystem(v___x_2262_, v___y_2272_, v___y_2273_);
if (lean_obj_tag(v___x_2275_) == 0)
{
lean_object* v___x_2276_; 
lean_dec_ref_known(v___x_2275_, 1);
lean_inc_ref(v_pre_2263_);
lean_inc(v___y_2273_);
lean_inc_ref(v___y_2272_);
lean_inc(v___y_2271_);
lean_inc_ref(v___y_2270_);
lean_inc_ref(v_e_2264_);
v___x_2276_ = lean_apply_6(v_pre_2263_, v_e_2264_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_, lean_box(0));
if (lean_obj_tag(v___x_2276_) == 0)
{
lean_object* v_a_2277_; lean_object* v___x_2279_; uint8_t v_isShared_2280_; uint8_t v_isSharedCheck_2325_; 
v_a_2277_ = lean_ctor_get(v___x_2276_, 0);
v_isSharedCheck_2325_ = !lean_is_exclusive(v___x_2276_);
if (v_isSharedCheck_2325_ == 0)
{
v___x_2279_ = v___x_2276_;
v_isShared_2280_ = v_isSharedCheck_2325_;
goto v_resetjp_2278_;
}
else
{
lean_inc(v_a_2277_);
lean_dec(v___x_2276_);
v___x_2279_ = lean_box(0);
v_isShared_2280_ = v_isSharedCheck_2325_;
goto v_resetjp_2278_;
}
v_resetjp_2278_:
{
lean_object* v___y_2282_; 
switch(lean_obj_tag(v_a_2277_))
{
case 0:
{
lean_object* v_e_2317_; lean_object* v___x_2319_; 
lean_dec_ref(v_post_2265_);
lean_dec_ref(v_e_2264_);
lean_dec_ref(v_pre_2263_);
v_e_2317_ = lean_ctor_get(v_a_2277_, 0);
lean_inc_ref(v_e_2317_);
lean_dec_ref_known(v_a_2277_, 1);
if (v_isShared_2280_ == 0)
{
lean_ctor_set(v___x_2279_, 0, v_e_2317_);
v___x_2319_ = v___x_2279_;
goto v_reusejp_2318_;
}
else
{
lean_object* v_reuseFailAlloc_2320_; 
v_reuseFailAlloc_2320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2320_, 0, v_e_2317_);
v___x_2319_ = v_reuseFailAlloc_2320_;
goto v_reusejp_2318_;
}
v_reusejp_2318_:
{
return v___x_2319_;
}
}
case 1:
{
lean_object* v_e_2321_; lean_object* v___x_2322_; 
lean_del_object(v___x_2279_);
lean_dec_ref(v_e_2264_);
v_e_2321_ = lean_ctor_get(v_a_2277_, 0);
lean_inc_ref(v_e_2321_);
lean_dec_ref_known(v_a_2277_, 1);
v___x_2322_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v_e_2321_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2322_;
}
default: 
{
lean_object* v_e_x3f_2323_; 
lean_del_object(v___x_2279_);
v_e_x3f_2323_ = lean_ctor_get(v_a_2277_, 0);
lean_inc(v_e_x3f_2323_);
lean_dec_ref_known(v_a_2277_, 1);
if (lean_obj_tag(v_e_x3f_2323_) == 0)
{
v___y_2282_ = v_e_2264_;
goto v___jp_2281_;
}
else
{
lean_object* v_val_2324_; 
lean_dec_ref(v_e_2264_);
v_val_2324_ = lean_ctor_get(v_e_x3f_2323_, 0);
lean_inc(v_val_2324_);
lean_dec_ref_known(v_e_x3f_2323_, 1);
v___y_2282_ = v_val_2324_;
goto v___jp_2281_;
}
}
}
v___jp_2281_:
{
switch(lean_obj_tag(v___y_2282_))
{
case 7:
{
lean_object* v___x_2283_; lean_object* v___x_2284_; 
v___x_2283_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__0));
v___x_2284_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___x_2283_, v___y_2282_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2284_;
}
case 6:
{
lean_object* v___x_2285_; lean_object* v___x_2286_; 
v___x_2285_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__0));
v___x_2286_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___x_2285_, v___y_2282_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2286_;
}
case 8:
{
lean_object* v___x_2287_; lean_object* v___x_2288_; 
v___x_2287_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__0));
v___x_2288_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___x_2287_, v___y_2282_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2288_;
}
case 5:
{
lean_object* v_dummy_2289_; lean_object* v_nargs_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; 
v_dummy_2289_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1);
v_nargs_2290_ = l_Lean_Expr_getAppNumArgs(v___y_2282_);
lean_inc(v_nargs_2290_);
v___x_2291_ = lean_mk_array(v_nargs_2290_, v_dummy_2289_);
v___x_2292_ = lean_unsigned_to_nat(1u);
v___x_2293_ = lean_nat_sub(v_nargs_2290_, v___x_2292_);
lean_dec(v_nargs_2290_);
v___x_2294_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__10(v_skipInstances_2268_, v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v___y_2282_, v___x_2291_, v___x_2293_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2294_;
}
case 10:
{
lean_object* v_data_2295_; lean_object* v_expr_2296_; lean_object* v___x_2297_; 
v_data_2295_ = lean_ctor_get(v___y_2282_, 0);
v_expr_2296_ = lean_ctor_get(v___y_2282_, 1);
lean_inc_ref(v_expr_2296_);
lean_inc_ref(v_post_2265_);
lean_inc_ref(v_pre_2263_);
v___x_2297_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v_expr_2296_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
if (lean_obj_tag(v___x_2297_) == 0)
{
lean_object* v_a_2298_; size_t v___x_2299_; size_t v___x_2300_; uint8_t v___x_2301_; 
v_a_2298_ = lean_ctor_get(v___x_2297_, 0);
lean_inc(v_a_2298_);
lean_dec_ref_known(v___x_2297_, 1);
v___x_2299_ = lean_ptr_addr(v_expr_2296_);
v___x_2300_ = lean_ptr_addr(v_a_2298_);
v___x_2301_ = lean_usize_dec_eq(v___x_2299_, v___x_2300_);
if (v___x_2301_ == 0)
{
lean_object* v___x_2302_; lean_object* v___x_2303_; 
lean_inc(v_data_2295_);
lean_dec_ref_known(v___y_2282_, 2);
v___x_2302_ = l_Lean_Expr_mdata___override(v_data_2295_, v_a_2298_);
v___x_2303_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___x_2302_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2303_;
}
else
{
lean_object* v___x_2304_; 
lean_dec(v_a_2298_);
v___x_2304_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___y_2282_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2304_;
}
}
else
{
lean_dec_ref_known(v___y_2282_, 2);
lean_dec_ref(v_post_2265_);
lean_dec_ref(v_pre_2263_);
return v___x_2297_;
}
}
case 11:
{
lean_object* v_typeName_2305_; lean_object* v_idx_2306_; lean_object* v_struct_2307_; lean_object* v___x_2308_; 
v_typeName_2305_ = lean_ctor_get(v___y_2282_, 0);
v_idx_2306_ = lean_ctor_get(v___y_2282_, 1);
v_struct_2307_ = lean_ctor_get(v___y_2282_, 2);
lean_inc_ref(v_struct_2307_);
lean_inc_ref(v_post_2265_);
lean_inc_ref(v_pre_2263_);
v___x_2308_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v_struct_2307_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
if (lean_obj_tag(v___x_2308_) == 0)
{
lean_object* v_a_2309_; size_t v___x_2310_; size_t v___x_2311_; uint8_t v___x_2312_; 
v_a_2309_ = lean_ctor_get(v___x_2308_, 0);
lean_inc(v_a_2309_);
lean_dec_ref_known(v___x_2308_, 1);
v___x_2310_ = lean_ptr_addr(v_struct_2307_);
v___x_2311_ = lean_ptr_addr(v_a_2309_);
v___x_2312_ = lean_usize_dec_eq(v___x_2310_, v___x_2311_);
if (v___x_2312_ == 0)
{
lean_object* v___x_2313_; lean_object* v___x_2314_; 
lean_inc(v_idx_2306_);
lean_inc(v_typeName_2305_);
lean_dec_ref_known(v___y_2282_, 3);
v___x_2313_ = l_Lean_Expr_proj___override(v_typeName_2305_, v_idx_2306_, v_a_2309_);
v___x_2314_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___x_2313_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2314_;
}
else
{
lean_object* v___x_2315_; 
lean_dec(v_a_2309_);
v___x_2315_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___y_2282_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2315_;
}
}
else
{
lean_dec_ref_known(v___y_2282_, 3);
lean_dec_ref(v_post_2265_);
lean_dec_ref(v_pre_2263_);
return v___x_2308_;
}
}
default: 
{
lean_object* v___x_2316_; 
v___x_2316_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2263_, v_post_2265_, v_usedLetOnly_2266_, v_skipConstInApp_2267_, v_skipInstances_2268_, v___y_2282_, v___y_2269_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
return v___x_2316_;
}
}
}
}
}
else
{
lean_object* v_a_2326_; lean_object* v___x_2328_; uint8_t v_isShared_2329_; uint8_t v_isSharedCheck_2333_; 
lean_dec_ref(v_post_2265_);
lean_dec_ref(v_e_2264_);
lean_dec_ref(v_pre_2263_);
v_a_2326_ = lean_ctor_get(v___x_2276_, 0);
v_isSharedCheck_2333_ = !lean_is_exclusive(v___x_2276_);
if (v_isSharedCheck_2333_ == 0)
{
v___x_2328_ = v___x_2276_;
v_isShared_2329_ = v_isSharedCheck_2333_;
goto v_resetjp_2327_;
}
else
{
lean_inc(v_a_2326_);
lean_dec(v___x_2276_);
v___x_2328_ = lean_box(0);
v_isShared_2329_ = v_isSharedCheck_2333_;
goto v_resetjp_2327_;
}
v_resetjp_2327_:
{
lean_object* v___x_2331_; 
if (v_isShared_2329_ == 0)
{
v___x_2331_ = v___x_2328_;
goto v_reusejp_2330_;
}
else
{
lean_object* v_reuseFailAlloc_2332_; 
v_reuseFailAlloc_2332_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2332_, 0, v_a_2326_);
v___x_2331_ = v_reuseFailAlloc_2332_;
goto v_reusejp_2330_;
}
v_reusejp_2330_:
{
return v___x_2331_;
}
}
}
}
else
{
lean_object* v_a_2334_; lean_object* v___x_2336_; uint8_t v_isShared_2337_; uint8_t v_isSharedCheck_2341_; 
lean_dec_ref(v_post_2265_);
lean_dec_ref(v_e_2264_);
lean_dec_ref(v_pre_2263_);
v_a_2334_ = lean_ctor_get(v___x_2275_, 0);
v_isSharedCheck_2341_ = !lean_is_exclusive(v___x_2275_);
if (v_isSharedCheck_2341_ == 0)
{
v___x_2336_ = v___x_2275_;
v_isShared_2337_ = v_isSharedCheck_2341_;
goto v_resetjp_2335_;
}
else
{
lean_inc(v_a_2334_);
lean_dec(v___x_2275_);
v___x_2336_ = lean_box(0);
v_isShared_2337_ = v_isSharedCheck_2341_;
goto v_resetjp_2335_;
}
v_resetjp_2335_:
{
lean_object* v___x_2339_; 
if (v_isShared_2337_ == 0)
{
v___x_2339_ = v___x_2336_;
goto v_reusejp_2338_;
}
else
{
lean_object* v_reuseFailAlloc_2340_; 
v_reuseFailAlloc_2340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2340_, 0, v_a_2334_);
v___x_2339_ = v_reuseFailAlloc_2340_;
goto v_reusejp_2338_;
}
v_reusejp_2338_:
{
return v___x_2339_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___boxed(lean_object* v___x_2342_, lean_object* v_pre_2343_, lean_object* v_e_2344_, lean_object* v_post_2345_, lean_object* v_usedLetOnly_2346_, lean_object* v_skipConstInApp_2347_, lean_object* v_skipInstances_2348_, lean_object* v___y_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_){
_start:
{
uint8_t v_usedLetOnly_boxed_2355_; uint8_t v_skipConstInApp_boxed_2356_; uint8_t v_skipInstances_boxed_2357_; lean_object* v_res_2358_; 
v_usedLetOnly_boxed_2355_ = lean_unbox(v_usedLetOnly_2346_);
v_skipConstInApp_boxed_2356_ = lean_unbox(v_skipConstInApp_2347_);
v_skipInstances_boxed_2357_ = lean_unbox(v_skipInstances_2348_);
v_res_2358_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1(v___x_2342_, v_pre_2343_, v_e_2344_, v_post_2345_, v_usedLetOnly_boxed_2355_, v_skipConstInApp_boxed_2356_, v_skipInstances_boxed_2357_, v___y_2349_, v___y_2350_, v___y_2351_, v___y_2352_, v___y_2353_);
lean_dec(v___y_2353_);
lean_dec_ref(v___y_2352_);
lean_dec(v___y_2351_);
lean_dec_ref(v___y_2350_);
lean_dec(v___y_2349_);
return v_res_2358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(lean_object* v_pre_2359_, lean_object* v_post_2360_, uint8_t v_usedLetOnly_2361_, uint8_t v_skipConstInApp_2362_, uint8_t v_skipInstances_2363_, lean_object* v_e_2364_, lean_object* v_a_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_){
_start:
{
lean_object* v___x_2371_; lean_object* v___x_2372_; 
lean_inc(v_a_2365_);
v___x_2371_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_2371_, 0, lean_box(0));
lean_closure_set(v___x_2371_, 1, lean_box(0));
lean_closure_set(v___x_2371_, 2, v_a_2365_);
v___x_2372_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__0(lean_box(0), v___x_2371_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_);
if (lean_obj_tag(v___x_2372_) == 0)
{
lean_object* v_a_2373_; lean_object* v___x_2375_; uint8_t v_isShared_2376_; uint8_t v_isSharedCheck_2407_; 
v_a_2373_ = lean_ctor_get(v___x_2372_, 0);
v_isSharedCheck_2407_ = !lean_is_exclusive(v___x_2372_);
if (v_isSharedCheck_2407_ == 0)
{
v___x_2375_ = v___x_2372_;
v_isShared_2376_ = v_isSharedCheck_2407_;
goto v_resetjp_2374_;
}
else
{
lean_inc(v_a_2373_);
lean_dec(v___x_2372_);
v___x_2375_ = lean_box(0);
v_isShared_2376_ = v_isSharedCheck_2407_;
goto v_resetjp_2374_;
}
v_resetjp_2374_:
{
lean_object* v___x_2377_; 
v___x_2377_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___redArg(v_a_2373_, v_e_2364_);
lean_dec(v_a_2373_);
if (lean_obj_tag(v___x_2377_) == 0)
{
lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___f_2382_; lean_object* v___x_2383_; 
lean_del_object(v___x_2375_);
v___x_2378_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___closed__0));
v___x_2379_ = lean_box(v_usedLetOnly_2361_);
v___x_2380_ = lean_box(v_skipConstInApp_2362_);
v___x_2381_ = lean_box(v_skipInstances_2363_);
lean_inc_ref(v_e_2364_);
v___f_2382_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___boxed), 13, 7);
lean_closure_set(v___f_2382_, 0, v___x_2378_);
lean_closure_set(v___f_2382_, 1, v_pre_2359_);
lean_closure_set(v___f_2382_, 2, v_e_2364_);
lean_closure_set(v___f_2382_, 3, v_post_2360_);
lean_closure_set(v___f_2382_, 4, v___x_2379_);
lean_closure_set(v___f_2382_, 5, v___x_2380_);
lean_closure_set(v___f_2382_, 6, v___x_2381_);
v___x_2383_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___redArg(v___f_2382_, v_a_2365_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_);
if (lean_obj_tag(v___x_2383_) == 0)
{
lean_object* v_a_2384_; lean_object* v___f_2385_; lean_object* v___x_2386_; 
v_a_2384_ = lean_ctor_get(v___x_2383_, 0);
lean_inc_n(v_a_2384_, 2);
lean_dec_ref_known(v___x_2383_, 1);
lean_inc(v_a_2365_);
v___f_2385_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__2___boxed), 4, 3);
lean_closure_set(v___f_2385_, 0, v_a_2365_);
lean_closure_set(v___f_2385_, 1, v_e_2364_);
lean_closure_set(v___f_2385_, 2, v_a_2384_);
v___x_2386_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__0(lean_box(0), v___f_2385_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_);
if (lean_obj_tag(v___x_2386_) == 0)
{
lean_object* v___x_2388_; uint8_t v_isShared_2389_; uint8_t v_isSharedCheck_2393_; 
v_isSharedCheck_2393_ = !lean_is_exclusive(v___x_2386_);
if (v_isSharedCheck_2393_ == 0)
{
lean_object* v_unused_2394_; 
v_unused_2394_ = lean_ctor_get(v___x_2386_, 0);
lean_dec(v_unused_2394_);
v___x_2388_ = v___x_2386_;
v_isShared_2389_ = v_isSharedCheck_2393_;
goto v_resetjp_2387_;
}
else
{
lean_dec(v___x_2386_);
v___x_2388_ = lean_box(0);
v_isShared_2389_ = v_isSharedCheck_2393_;
goto v_resetjp_2387_;
}
v_resetjp_2387_:
{
lean_object* v___x_2391_; 
if (v_isShared_2389_ == 0)
{
lean_ctor_set(v___x_2388_, 0, v_a_2384_);
v___x_2391_ = v___x_2388_;
goto v_reusejp_2390_;
}
else
{
lean_object* v_reuseFailAlloc_2392_; 
v_reuseFailAlloc_2392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2392_, 0, v_a_2384_);
v___x_2391_ = v_reuseFailAlloc_2392_;
goto v_reusejp_2390_;
}
v_reusejp_2390_:
{
return v___x_2391_;
}
}
}
else
{
lean_object* v_a_2395_; lean_object* v___x_2397_; uint8_t v_isShared_2398_; uint8_t v_isSharedCheck_2402_; 
lean_dec(v_a_2384_);
v_a_2395_ = lean_ctor_get(v___x_2386_, 0);
v_isSharedCheck_2402_ = !lean_is_exclusive(v___x_2386_);
if (v_isSharedCheck_2402_ == 0)
{
v___x_2397_ = v___x_2386_;
v_isShared_2398_ = v_isSharedCheck_2402_;
goto v_resetjp_2396_;
}
else
{
lean_inc(v_a_2395_);
lean_dec(v___x_2386_);
v___x_2397_ = lean_box(0);
v_isShared_2398_ = v_isSharedCheck_2402_;
goto v_resetjp_2396_;
}
v_resetjp_2396_:
{
lean_object* v___x_2400_; 
if (v_isShared_2398_ == 0)
{
v___x_2400_ = v___x_2397_;
goto v_reusejp_2399_;
}
else
{
lean_object* v_reuseFailAlloc_2401_; 
v_reuseFailAlloc_2401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2401_, 0, v_a_2395_);
v___x_2400_ = v_reuseFailAlloc_2401_;
goto v_reusejp_2399_;
}
v_reusejp_2399_:
{
return v___x_2400_;
}
}
}
}
else
{
lean_dec_ref(v_e_2364_);
return v___x_2383_;
}
}
else
{
lean_object* v_val_2403_; lean_object* v___x_2405_; 
lean_dec_ref(v_e_2364_);
lean_dec_ref(v_post_2360_);
lean_dec_ref(v_pre_2359_);
v_val_2403_ = lean_ctor_get(v___x_2377_, 0);
lean_inc(v_val_2403_);
lean_dec_ref_known(v___x_2377_, 1);
if (v_isShared_2376_ == 0)
{
lean_ctor_set(v___x_2375_, 0, v_val_2403_);
v___x_2405_ = v___x_2375_;
goto v_reusejp_2404_;
}
else
{
lean_object* v_reuseFailAlloc_2406_; 
v_reuseFailAlloc_2406_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2406_, 0, v_val_2403_);
v___x_2405_ = v_reuseFailAlloc_2406_;
goto v_reusejp_2404_;
}
v_reusejp_2404_:
{
return v___x_2405_;
}
}
}
}
else
{
lean_object* v_a_2408_; lean_object* v___x_2410_; uint8_t v_isShared_2411_; uint8_t v_isSharedCheck_2415_; 
lean_dec_ref(v_e_2364_);
lean_dec_ref(v_post_2360_);
lean_dec_ref(v_pre_2359_);
v_a_2408_ = lean_ctor_get(v___x_2372_, 0);
v_isSharedCheck_2415_ = !lean_is_exclusive(v___x_2372_);
if (v_isSharedCheck_2415_ == 0)
{
v___x_2410_ = v___x_2372_;
v_isShared_2411_ = v_isSharedCheck_2415_;
goto v_resetjp_2409_;
}
else
{
lean_inc(v_a_2408_);
lean_dec(v___x_2372_);
v___x_2410_ = lean_box(0);
v_isShared_2411_ = v_isSharedCheck_2415_;
goto v_resetjp_2409_;
}
v_resetjp_2409_:
{
lean_object* v___x_2413_; 
if (v_isShared_2411_ == 0)
{
v___x_2413_ = v___x_2410_;
goto v_reusejp_2412_;
}
else
{
lean_object* v_reuseFailAlloc_2414_; 
v_reuseFailAlloc_2414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2414_, 0, v_a_2408_);
v___x_2413_ = v_reuseFailAlloc_2414_;
goto v_reusejp_2412_;
}
v_reusejp_2412_:
{
return v___x_2413_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___lam__0___boxed(lean_object* v_fvars_2416_, lean_object* v_pre_2417_, lean_object* v_post_2418_, lean_object* v_usedLetOnly_2419_, lean_object* v_skipConstInApp_2420_, lean_object* v_skipInstances_2421_, lean_object* v_body_2422_, lean_object* v_x_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_, lean_object* v___y_2426_, lean_object* v___y_2427_, lean_object* v___y_2428_, lean_object* v___y_2429_){
_start:
{
uint8_t v_usedLetOnly_boxed_2430_; uint8_t v_skipConstInApp_boxed_2431_; uint8_t v_skipInstances_boxed_2432_; lean_object* v_res_2433_; 
v_usedLetOnly_boxed_2430_ = lean_unbox(v_usedLetOnly_2419_);
v_skipConstInApp_boxed_2431_ = lean_unbox(v_skipConstInApp_2420_);
v_skipInstances_boxed_2432_ = lean_unbox(v_skipInstances_2421_);
v_res_2433_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___lam__0(v_fvars_2416_, v_pre_2417_, v_post_2418_, v_usedLetOnly_boxed_2430_, v_skipConstInApp_boxed_2431_, v_skipInstances_boxed_2432_, v_body_2422_, v_x_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_, v___y_2428_);
lean_dec(v___y_2428_);
lean_dec_ref(v___y_2427_);
lean_dec(v___y_2426_);
lean_dec_ref(v___y_2425_);
lean_dec(v___y_2424_);
return v_res_2433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7(lean_object* v_pre_2434_, lean_object* v_post_2435_, uint8_t v_usedLetOnly_2436_, uint8_t v_skipConstInApp_2437_, uint8_t v_skipInstances_2438_, lean_object* v_fvars_2439_, lean_object* v_e_2440_, lean_object* v_a_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_, lean_object* v___y_2444_, lean_object* v___y_2445_){
_start:
{
if (lean_obj_tag(v_e_2440_) == 7)
{
lean_object* v_binderName_2447_; lean_object* v_binderType_2448_; lean_object* v_body_2449_; uint8_t v_binderInfo_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; 
v_binderName_2447_ = lean_ctor_get(v_e_2440_, 0);
lean_inc(v_binderName_2447_);
v_binderType_2448_ = lean_ctor_get(v_e_2440_, 1);
lean_inc_ref(v_binderType_2448_);
v_body_2449_ = lean_ctor_get(v_e_2440_, 2);
lean_inc_ref(v_body_2449_);
v_binderInfo_2450_ = lean_ctor_get_uint8(v_e_2440_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_2440_, 3);
v___x_2451_ = lean_expr_instantiate_rev(v_binderType_2448_, v_fvars_2439_);
lean_dec_ref(v_binderType_2448_);
lean_inc_ref(v_post_2435_);
lean_inc_ref(v_pre_2434_);
v___x_2452_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2434_, v_post_2435_, v_usedLetOnly_2436_, v_skipConstInApp_2437_, v_skipInstances_2438_, v___x_2451_, v_a_2441_, v___y_2442_, v___y_2443_, v___y_2444_, v___y_2445_);
if (lean_obj_tag(v___x_2452_) == 0)
{
lean_object* v_a_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___f_2457_; uint8_t v___x_2458_; lean_object* v___x_2459_; 
v_a_2453_ = lean_ctor_get(v___x_2452_, 0);
lean_inc(v_a_2453_);
lean_dec_ref_known(v___x_2452_, 1);
v___x_2454_ = lean_box(v_usedLetOnly_2436_);
v___x_2455_ = lean_box(v_skipConstInApp_2437_);
v___x_2456_ = lean_box(v_skipInstances_2438_);
v___f_2457_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___lam__0___boxed), 14, 7);
lean_closure_set(v___f_2457_, 0, v_fvars_2439_);
lean_closure_set(v___f_2457_, 1, v_pre_2434_);
lean_closure_set(v___f_2457_, 2, v_post_2435_);
lean_closure_set(v___f_2457_, 3, v___x_2454_);
lean_closure_set(v___f_2457_, 4, v___x_2455_);
lean_closure_set(v___f_2457_, 5, v___x_2456_);
lean_closure_set(v___f_2457_, 6, v_body_2449_);
v___x_2458_ = 0;
v___x_2459_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg(v_binderName_2447_, v_binderInfo_2450_, v_a_2453_, v___f_2457_, v___x_2458_, v_a_2441_, v___y_2442_, v___y_2443_, v___y_2444_, v___y_2445_);
return v___x_2459_;
}
else
{
lean_dec_ref(v_body_2449_);
lean_dec(v_binderName_2447_);
lean_dec_ref(v_fvars_2439_);
lean_dec_ref(v_post_2435_);
lean_dec_ref(v_pre_2434_);
return v___x_2452_;
}
}
else
{
lean_object* v___x_2460_; lean_object* v___x_2461_; 
v___x_2460_ = lean_expr_instantiate_rev(v_e_2440_, v_fvars_2439_);
lean_dec_ref(v_e_2440_);
lean_inc_ref(v_post_2435_);
lean_inc_ref(v_pre_2434_);
v___x_2461_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2434_, v_post_2435_, v_usedLetOnly_2436_, v_skipConstInApp_2437_, v_skipInstances_2438_, v___x_2460_, v_a_2441_, v___y_2442_, v___y_2443_, v___y_2444_, v___y_2445_);
if (lean_obj_tag(v___x_2461_) == 0)
{
lean_object* v_a_2462_; uint8_t v___x_2463_; uint8_t v___x_2464_; uint8_t v___x_2465_; lean_object* v___x_2466_; 
v_a_2462_ = lean_ctor_get(v___x_2461_, 0);
lean_inc(v_a_2462_);
lean_dec_ref_known(v___x_2461_, 1);
v___x_2463_ = 0;
v___x_2464_ = 1;
v___x_2465_ = 1;
v___x_2466_ = l_Lean_Meta_mkForallFVars(v_fvars_2439_, v_a_2462_, v___x_2463_, v_usedLetOnly_2436_, v___x_2464_, v___x_2465_, v___y_2442_, v___y_2443_, v___y_2444_, v___y_2445_);
lean_dec_ref(v_fvars_2439_);
if (lean_obj_tag(v___x_2466_) == 0)
{
lean_object* v_a_2467_; lean_object* v___x_2468_; 
v_a_2467_ = lean_ctor_get(v___x_2466_, 0);
lean_inc(v_a_2467_);
lean_dec_ref_known(v___x_2466_, 1);
v___x_2468_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2434_, v_post_2435_, v_usedLetOnly_2436_, v_skipConstInApp_2437_, v_skipInstances_2438_, v_a_2467_, v_a_2441_, v___y_2442_, v___y_2443_, v___y_2444_, v___y_2445_);
return v___x_2468_;
}
else
{
lean_dec_ref(v_post_2435_);
lean_dec_ref(v_pre_2434_);
return v___x_2466_;
}
}
else
{
lean_dec_ref(v_fvars_2439_);
lean_dec_ref(v_post_2435_);
lean_dec_ref(v_pre_2434_);
return v___x_2461_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___lam__0(lean_object* v_fvars_2469_, lean_object* v_pre_2470_, lean_object* v_post_2471_, uint8_t v_usedLetOnly_2472_, uint8_t v_skipConstInApp_2473_, uint8_t v_skipInstances_2474_, lean_object* v_body_2475_, lean_object* v_x_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_){
_start:
{
lean_object* v___x_2483_; lean_object* v___x_2484_; 
v___x_2483_ = lean_array_push(v_fvars_2469_, v_x_2476_);
v___x_2484_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7(v_pre_2470_, v_post_2471_, v_usedLetOnly_2472_, v_skipConstInApp_2473_, v_skipInstances_2474_, v___x_2483_, v_body_2475_, v___y_2477_, v___y_2478_, v___y_2479_, v___y_2480_, v___y_2481_);
return v___x_2484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4___boxed(lean_object* v_pre_2485_, lean_object* v_post_2486_, lean_object* v_usedLetOnly_2487_, lean_object* v_skipConstInApp_2488_, lean_object* v_skipInstances_2489_, lean_object* v_e_2490_, lean_object* v_a_2491_, lean_object* v___y_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_){
_start:
{
uint8_t v_usedLetOnly_boxed_2497_; uint8_t v_skipConstInApp_boxed_2498_; uint8_t v_skipInstances_boxed_2499_; lean_object* v_res_2500_; 
v_usedLetOnly_boxed_2497_ = lean_unbox(v_usedLetOnly_2487_);
v_skipConstInApp_boxed_2498_ = lean_unbox(v_skipConstInApp_2488_);
v_skipInstances_boxed_2499_ = lean_unbox(v_skipInstances_2489_);
v_res_2500_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__4(v_pre_2485_, v_post_2486_, v_usedLetOnly_boxed_2497_, v_skipConstInApp_boxed_2498_, v_skipInstances_boxed_2499_, v_e_2490_, v_a_2491_, v___y_2492_, v___y_2493_, v___y_2494_, v___y_2495_);
lean_dec(v___y_2495_);
lean_dec_ref(v___y_2494_);
lean_dec(v___y_2493_);
lean_dec_ref(v___y_2492_);
lean_dec(v_a_2491_);
return v_res_2500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__3___boxed(lean_object* v_pre_2501_, lean_object* v_post_2502_, lean_object* v_usedLetOnly_2503_, lean_object* v_skipConstInApp_2504_, lean_object* v_skipInstances_2505_, lean_object* v_sz_2506_, lean_object* v_i_2507_, lean_object* v_bs_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_){
_start:
{
uint8_t v_usedLetOnly_boxed_2515_; uint8_t v_skipConstInApp_boxed_2516_; uint8_t v_skipInstances_boxed_2517_; size_t v_sz_boxed_2518_; size_t v_i_boxed_2519_; lean_object* v_res_2520_; 
v_usedLetOnly_boxed_2515_ = lean_unbox(v_usedLetOnly_2503_);
v_skipConstInApp_boxed_2516_ = lean_unbox(v_skipConstInApp_2504_);
v_skipInstances_boxed_2517_ = lean_unbox(v_skipInstances_2505_);
v_sz_boxed_2518_ = lean_unbox_usize(v_sz_2506_);
lean_dec(v_sz_2506_);
v_i_boxed_2519_ = lean_unbox_usize(v_i_2507_);
lean_dec(v_i_2507_);
v_res_2520_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__3(v_pre_2501_, v_post_2502_, v_usedLetOnly_boxed_2515_, v_skipConstInApp_boxed_2516_, v_skipInstances_boxed_2517_, v_sz_boxed_2518_, v_i_boxed_2519_, v_bs_2508_, v___y_2509_, v___y_2510_, v___y_2511_, v___y_2512_, v___y_2513_);
lean_dec(v___y_2513_);
lean_dec_ref(v___y_2512_);
lean_dec(v___y_2511_);
lean_dec_ref(v___y_2510_);
lean_dec(v___y_2509_);
return v_res_2520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___boxed(lean_object* v_pre_2521_, lean_object* v_post_2522_, lean_object* v_usedLetOnly_2523_, lean_object* v_skipConstInApp_2524_, lean_object* v_skipInstances_2525_, lean_object* v_e_2526_, lean_object* v_a_2527_, lean_object* v___y_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_, lean_object* v___y_2531_, lean_object* v___y_2532_){
_start:
{
uint8_t v_usedLetOnly_boxed_2533_; uint8_t v_skipConstInApp_boxed_2534_; uint8_t v_skipInstances_boxed_2535_; lean_object* v_res_2536_; 
v_usedLetOnly_boxed_2533_ = lean_unbox(v_usedLetOnly_2523_);
v_skipConstInApp_boxed_2534_ = lean_unbox(v_skipConstInApp_2524_);
v_skipInstances_boxed_2535_ = lean_unbox(v_skipInstances_2525_);
v_res_2536_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2521_, v_post_2522_, v_usedLetOnly_boxed_2533_, v_skipConstInApp_boxed_2534_, v_skipInstances_boxed_2535_, v_e_2526_, v_a_2527_, v___y_2528_, v___y_2529_, v___y_2530_, v___y_2531_);
lean_dec(v___y_2531_);
lean_dec_ref(v___y_2530_);
lean_dec(v___y_2529_);
lean_dec_ref(v___y_2528_);
lean_dec(v_a_2527_);
return v_res_2536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7___boxed(lean_object* v_pre_2537_, lean_object* v_post_2538_, lean_object* v_usedLetOnly_2539_, lean_object* v_skipConstInApp_2540_, lean_object* v_skipInstances_2541_, lean_object* v_fvars_2542_, lean_object* v_e_2543_, lean_object* v_a_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_){
_start:
{
uint8_t v_usedLetOnly_boxed_2550_; uint8_t v_skipConstInApp_boxed_2551_; uint8_t v_skipInstances_boxed_2552_; lean_object* v_res_2553_; 
v_usedLetOnly_boxed_2550_ = lean_unbox(v_usedLetOnly_2539_);
v_skipConstInApp_boxed_2551_ = lean_unbox(v_skipConstInApp_2540_);
v_skipInstances_boxed_2552_ = lean_unbox(v_skipInstances_2541_);
v_res_2553_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7(v_pre_2537_, v_post_2538_, v_usedLetOnly_boxed_2550_, v_skipConstInApp_boxed_2551_, v_skipInstances_boxed_2552_, v_fvars_2542_, v_e_2543_, v_a_2544_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_);
lean_dec(v___y_2548_);
lean_dec_ref(v___y_2547_);
lean_dec(v___y_2546_);
lean_dec_ref(v___y_2545_);
lean_dec(v_a_2544_);
return v_res_2553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8___boxed(lean_object* v_pre_2554_, lean_object* v_post_2555_, lean_object* v_usedLetOnly_2556_, lean_object* v_skipConstInApp_2557_, lean_object* v_skipInstances_2558_, lean_object* v_fvars_2559_, lean_object* v_e_2560_, lean_object* v_a_2561_, lean_object* v___y_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_){
_start:
{
uint8_t v_usedLetOnly_boxed_2567_; uint8_t v_skipConstInApp_boxed_2568_; uint8_t v_skipInstances_boxed_2569_; lean_object* v_res_2570_; 
v_usedLetOnly_boxed_2567_ = lean_unbox(v_usedLetOnly_2556_);
v_skipConstInApp_boxed_2568_ = lean_unbox(v_skipConstInApp_2557_);
v_skipInstances_boxed_2569_ = lean_unbox(v_skipInstances_2558_);
v_res_2570_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__8(v_pre_2554_, v_post_2555_, v_usedLetOnly_boxed_2567_, v_skipConstInApp_boxed_2568_, v_skipInstances_boxed_2569_, v_fvars_2559_, v_e_2560_, v_a_2561_, v___y_2562_, v___y_2563_, v___y_2564_, v___y_2565_);
lean_dec(v___y_2565_);
lean_dec_ref(v___y_2564_);
lean_dec(v___y_2563_);
lean_dec_ref(v___y_2562_);
lean_dec(v_a_2561_);
return v_res_2570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9___boxed(lean_object* v_pre_2571_, lean_object* v_post_2572_, lean_object* v_usedLetOnly_2573_, lean_object* v_skipConstInApp_2574_, lean_object* v_skipInstances_2575_, lean_object* v_fvars_2576_, lean_object* v_e_2577_, lean_object* v_a_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_){
_start:
{
uint8_t v_usedLetOnly_boxed_2584_; uint8_t v_skipConstInApp_boxed_2585_; uint8_t v_skipInstances_boxed_2586_; lean_object* v_res_2587_; 
v_usedLetOnly_boxed_2584_ = lean_unbox(v_usedLetOnly_2573_);
v_skipConstInApp_boxed_2585_ = lean_unbox(v_skipConstInApp_2574_);
v_skipInstances_boxed_2586_ = lean_unbox(v_skipInstances_2575_);
v_res_2587_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9(v_pre_2571_, v_post_2572_, v_usedLetOnly_boxed_2584_, v_skipConstInApp_boxed_2585_, v_skipInstances_boxed_2586_, v_fvars_2576_, v_e_2577_, v_a_2578_, v___y_2579_, v___y_2580_, v___y_2581_, v___y_2582_);
lean_dec(v___y_2582_);
lean_dec_ref(v___y_2581_);
lean_dec(v___y_2580_);
lean_dec_ref(v___y_2579_);
lean_dec(v_a_2578_);
return v_res_2587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_upperBound_2588_, lean_object* v___x_2589_, lean_object* v_pre_2590_, lean_object* v_post_2591_, lean_object* v_usedLetOnly_2592_, lean_object* v_skipConstInApp_2593_, lean_object* v_skipInstances_2594_, lean_object* v_a_2595_, lean_object* v_b_2596_, lean_object* v___y_2597_, lean_object* v___y_2598_, lean_object* v___y_2599_, lean_object* v___y_2600_, lean_object* v___y_2601_, lean_object* v___y_2602_){
_start:
{
uint8_t v_usedLetOnly_boxed_2603_; uint8_t v_skipConstInApp_boxed_2604_; uint8_t v_skipInstances_boxed_2605_; lean_object* v_res_2606_; 
v_usedLetOnly_boxed_2603_ = lean_unbox(v_usedLetOnly_2592_);
v_skipConstInApp_boxed_2604_ = lean_unbox(v_skipConstInApp_2593_);
v_skipInstances_boxed_2605_ = lean_unbox(v_skipInstances_2594_);
v_res_2606_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg(v_upperBound_2588_, v___x_2589_, v_pre_2590_, v_post_2591_, v_usedLetOnly_boxed_2603_, v_skipConstInApp_boxed_2604_, v_skipInstances_boxed_2605_, v_a_2595_, v_b_2596_, v___y_2597_, v___y_2598_, v___y_2599_, v___y_2600_, v___y_2601_);
lean_dec(v___y_2601_);
lean_dec_ref(v___y_2600_);
lean_dec(v___y_2599_);
lean_dec_ref(v___y_2598_);
lean_dec(v___y_2597_);
lean_dec_ref(v___x_2589_);
lean_dec(v_upperBound_2588_);
return v_res_2606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__10___boxed(lean_object* v_skipInstances_2607_, lean_object* v_pre_2608_, lean_object* v_post_2609_, lean_object* v_usedLetOnly_2610_, lean_object* v_skipConstInApp_2611_, lean_object* v_x_2612_, lean_object* v_x_2613_, lean_object* v_x_2614_, lean_object* v___y_2615_, lean_object* v___y_2616_, lean_object* v___y_2617_, lean_object* v___y_2618_, lean_object* v___y_2619_, lean_object* v___y_2620_){
_start:
{
uint8_t v_skipInstances_boxed_2621_; uint8_t v_usedLetOnly_boxed_2622_; uint8_t v_skipConstInApp_boxed_2623_; lean_object* v_res_2624_; 
v_skipInstances_boxed_2621_ = lean_unbox(v_skipInstances_2607_);
v_usedLetOnly_boxed_2622_ = lean_unbox(v_usedLetOnly_2610_);
v_skipConstInApp_boxed_2623_ = lean_unbox(v_skipConstInApp_2611_);
v_res_2624_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__10(v_skipInstances_boxed_2621_, v_pre_2608_, v_post_2609_, v_usedLetOnly_boxed_2622_, v_skipConstInApp_boxed_2623_, v_x_2612_, v_x_2613_, v_x_2614_, v___y_2615_, v___y_2616_, v___y_2617_, v___y_2618_, v___y_2619_);
lean_dec(v___y_2619_);
lean_dec_ref(v___y_2618_);
lean_dec(v___y_2617_);
lean_dec_ref(v___y_2616_);
lean_dec(v___y_2615_);
return v_res_2624_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__0(void){
_start:
{
lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; 
v___x_2625_ = lean_box(0);
v___x_2626_ = lean_unsigned_to_nat(16u);
v___x_2627_ = lean_mk_array(v___x_2626_, v___x_2625_);
return v___x_2627_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__1(void){
_start:
{
lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; 
v___x_2628_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__0, &lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__0_once, _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__0);
v___x_2629_ = lean_unsigned_to_nat(0u);
v___x_2630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2630_, 0, v___x_2629_);
lean_ctor_set(v___x_2630_, 1, v___x_2628_);
return v___x_2630_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__2(void){
_start:
{
lean_object* v___x_2631_; lean_object* v___x_2632_; 
v___x_2631_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__1, &lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__1_once, _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__1);
v___x_2632_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_2632_, 0, lean_box(0));
lean_closure_set(v___x_2632_, 1, lean_box(0));
lean_closure_set(v___x_2632_, 2, v___x_2631_);
return v___x_2632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1(lean_object* v_input_2633_, lean_object* v_pre_2634_, lean_object* v_post_2635_, uint8_t v_usedLetOnly_2636_, uint8_t v_skipConstInApp_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_){
_start:
{
lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v_a_2645_; uint8_t v___x_2646_; lean_object* v___x_2647_; 
v___x_2643_ = lean_obj_once(&lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__2, &lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__2_once, _init_lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___closed__2);
v___x_2644_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___lam__0(lean_box(0), v___x_2643_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_);
v_a_2645_ = lean_ctor_get(v___x_2644_, 0);
lean_inc(v_a_2645_);
lean_dec_ref(v___x_2644_);
v___x_2646_ = 0;
v___x_2647_ = lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2(v_pre_2634_, v_post_2635_, v_usedLetOnly_2636_, v_skipConstInApp_2637_, v___x_2646_, v_input_2633_, v_a_2645_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_);
if (lean_obj_tag(v___x_2647_) == 0)
{
lean_object* v_a_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2652_; uint8_t v_isShared_2653_; uint8_t v_isSharedCheck_2657_; 
v_a_2648_ = lean_ctor_get(v___x_2647_, 0);
lean_inc(v_a_2648_);
lean_dec_ref_known(v___x_2647_, 1);
v___x_2649_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_2649_, 0, lean_box(0));
lean_closure_set(v___x_2649_, 1, lean_box(0));
lean_closure_set(v___x_2649_, 2, v_a_2645_);
v___x_2650_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___lam__0(lean_box(0), v___x_2649_, v___y_2638_, v___y_2639_, v___y_2640_, v___y_2641_);
v_isSharedCheck_2657_ = !lean_is_exclusive(v___x_2650_);
if (v_isSharedCheck_2657_ == 0)
{
lean_object* v_unused_2658_; 
v_unused_2658_ = lean_ctor_get(v___x_2650_, 0);
lean_dec(v_unused_2658_);
v___x_2652_ = v___x_2650_;
v_isShared_2653_ = v_isSharedCheck_2657_;
goto v_resetjp_2651_;
}
else
{
lean_dec(v___x_2650_);
v___x_2652_ = lean_box(0);
v_isShared_2653_ = v_isSharedCheck_2657_;
goto v_resetjp_2651_;
}
v_resetjp_2651_:
{
lean_object* v___x_2655_; 
if (v_isShared_2653_ == 0)
{
lean_ctor_set(v___x_2652_, 0, v_a_2648_);
v___x_2655_ = v___x_2652_;
goto v_reusejp_2654_;
}
else
{
lean_object* v_reuseFailAlloc_2656_; 
v_reuseFailAlloc_2656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2656_, 0, v_a_2648_);
v___x_2655_ = v_reuseFailAlloc_2656_;
goto v_reusejp_2654_;
}
v_reusejp_2654_:
{
return v___x_2655_;
}
}
}
else
{
lean_dec(v_a_2645_);
return v___x_2647_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1___boxed(lean_object* v_input_2659_, lean_object* v_pre_2660_, lean_object* v_post_2661_, lean_object* v_usedLetOnly_2662_, lean_object* v_skipConstInApp_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_){
_start:
{
uint8_t v_usedLetOnly_boxed_2669_; uint8_t v_skipConstInApp_boxed_2670_; lean_object* v_res_2671_; 
v_usedLetOnly_boxed_2669_ = lean_unbox(v_usedLetOnly_2662_);
v_skipConstInApp_boxed_2670_ = lean_unbox(v_skipConstInApp_2663_);
v_res_2671_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1(v_input_2659_, v_pre_2660_, v_post_2661_, v_usedLetOnly_boxed_2669_, v_skipConstInApp_boxed_2670_, v___y_2664_, v___y_2665_, v___y_2666_, v___y_2667_);
lean_dec(v___y_2667_);
lean_dec_ref(v___y_2666_);
lean_dec(v___y_2665_);
lean_dec_ref(v___y_2664_);
return v_res_2671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars(lean_object* v_fvars_2673_, lean_object* v_e_2674_, lean_object* v_a_2675_, lean_object* v_a_2676_, lean_object* v_a_2677_, lean_object* v_a_2678_){
_start:
{
lean_object* v___f_2680_; lean_object* v___f_2681_; uint8_t v___x_2682_; uint8_t v___x_2683_; lean_object* v___x_2684_; 
v___f_2680_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___closed__0));
v___f_2681_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___boxed), 7, 1);
lean_closure_set(v___f_2681_, 0, v_fvars_2673_);
v___x_2682_ = 1;
v___x_2683_ = 0;
v___x_2684_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1(v_e_2674_, v___f_2681_, v___f_2680_, v___x_2682_, v___x_2683_, v_a_2675_, v_a_2676_, v_a_2677_, v_a_2678_);
return v___x_2684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldFVars___boxed(lean_object* v_fvars_2685_, lean_object* v_e_2686_, lean_object* v_a_2687_, lean_object* v_a_2688_, lean_object* v_a_2689_, lean_object* v_a_2690_, lean_object* v_a_2691_){
_start:
{
lean_object* v_res_2692_; 
v_res_2692_ = lp_mathlib_Mathlib_Tactic_unfoldFVars(v_fvars_2685_, v_e_2686_, v_a_2687_, v_a_2688_, v_a_2689_, v_a_2690_);
lean_dec(v_a_2690_);
lean_dec_ref(v_a_2689_);
lean_dec(v_a_2688_);
lean_dec_ref(v_a_2687_);
return v_res_2692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5(lean_object* v_upperBound_2693_, lean_object* v___x_2694_, lean_object* v_pre_2695_, lean_object* v_post_2696_, uint8_t v_usedLetOnly_2697_, uint8_t v_skipConstInApp_2698_, uint8_t v_skipInstances_2699_, lean_object* v___x_2700_, lean_object* v_inst_2701_, lean_object* v_R_2702_, lean_object* v_a_2703_, lean_object* v_b_2704_, lean_object* v_c_2705_, lean_object* v___y_2706_, lean_object* v___y_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_){
_start:
{
lean_object* v___x_2712_; 
v___x_2712_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___redArg(v_upperBound_2693_, v___x_2694_, v_pre_2695_, v_post_2696_, v_usedLetOnly_2697_, v_skipConstInApp_2698_, v_skipInstances_2699_, v_a_2703_, v_b_2704_, v___y_2706_, v___y_2707_, v___y_2708_, v___y_2709_, v___y_2710_);
return v___x_2712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5___boxed(lean_object** _args){
lean_object* v_upperBound_2713_ = _args[0];
lean_object* v___x_2714_ = _args[1];
lean_object* v_pre_2715_ = _args[2];
lean_object* v_post_2716_ = _args[3];
lean_object* v_usedLetOnly_2717_ = _args[4];
lean_object* v_skipConstInApp_2718_ = _args[5];
lean_object* v_skipInstances_2719_ = _args[6];
lean_object* v___x_2720_ = _args[7];
lean_object* v_inst_2721_ = _args[8];
lean_object* v_R_2722_ = _args[9];
lean_object* v_a_2723_ = _args[10];
lean_object* v_b_2724_ = _args[11];
lean_object* v_c_2725_ = _args[12];
lean_object* v___y_2726_ = _args[13];
lean_object* v___y_2727_ = _args[14];
lean_object* v___y_2728_ = _args[15];
lean_object* v___y_2729_ = _args[16];
lean_object* v___y_2730_ = _args[17];
lean_object* v___y_2731_ = _args[18];
_start:
{
uint8_t v_usedLetOnly_boxed_2732_; uint8_t v_skipConstInApp_boxed_2733_; uint8_t v_skipInstances_boxed_2734_; lean_object* v_res_2735_; 
v_usedLetOnly_boxed_2732_ = lean_unbox(v_usedLetOnly_2717_);
v_skipConstInApp_boxed_2733_ = lean_unbox(v_skipConstInApp_2718_);
v_skipInstances_boxed_2734_ = lean_unbox(v_skipInstances_2719_);
v_res_2735_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__5(v_upperBound_2713_, v___x_2714_, v_pre_2715_, v_post_2716_, v_usedLetOnly_boxed_2732_, v_skipConstInApp_boxed_2733_, v_skipInstances_boxed_2734_, v___x_2720_, v_inst_2721_, v_R_2722_, v_a_2723_, v_b_2724_, v_c_2725_, v___y_2726_, v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_);
lean_dec(v___y_2730_);
lean_dec_ref(v___y_2729_);
lean_dec(v___y_2728_);
lean_dec_ref(v___y_2727_);
lean_dec(v___y_2726_);
lean_dec(v___x_2720_);
lean_dec_ref(v___x_2714_);
lean_dec(v_upperBound_2713_);
return v_res_2735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6(lean_object* v_00_u03b2_2736_, lean_object* v_m_2737_, lean_object* v_a_2738_){
_start:
{
lean_object* v___x_2739_; 
v___x_2739_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___redArg(v_m_2737_, v_a_2738_);
return v___x_2739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6___boxed(lean_object* v_00_u03b2_2740_, lean_object* v_m_2741_, lean_object* v_a_2742_){
_start:
{
lean_object* v_res_2743_; 
v_res_2743_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6(v_00_u03b2_2740_, v_m_2741_, v_a_2742_);
lean_dec_ref(v_a_2742_);
lean_dec_ref(v_m_2741_);
return v_res_2743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9(lean_object* v_00_u03b1_2744_, lean_object* v_name_2745_, uint8_t v_bi_2746_, lean_object* v_type_2747_, lean_object* v_k_2748_, uint8_t v_kind_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_){
_start:
{
lean_object* v___x_2756_; 
v___x_2756_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___redArg(v_name_2745_, v_bi_2746_, v_type_2747_, v_k_2748_, v_kind_2749_, v___y_2750_, v___y_2751_, v___y_2752_, v___y_2753_, v___y_2754_);
return v___x_2756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9___boxed(lean_object* v_00_u03b1_2757_, lean_object* v_name_2758_, lean_object* v_bi_2759_, lean_object* v_type_2760_, lean_object* v_k_2761_, lean_object* v_kind_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_, lean_object* v___y_2768_){
_start:
{
uint8_t v_bi_boxed_2769_; uint8_t v_kind_boxed_2770_; lean_object* v_res_2771_; 
v_bi_boxed_2769_ = lean_unbox(v_bi_2759_);
v_kind_boxed_2770_ = lean_unbox(v_kind_2762_);
v_res_2771_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__7_spec__9(v_00_u03b1_2757_, v_name_2758_, v_bi_boxed_2769_, v_type_2760_, v_k_2761_, v_kind_boxed_2770_, v___y_2763_, v___y_2764_, v___y_2765_, v___y_2766_, v___y_2767_);
lean_dec(v___y_2767_);
lean_dec_ref(v___y_2766_);
lean_dec(v___y_2765_);
lean_dec_ref(v___y_2764_);
lean_dec(v___y_2763_);
return v_res_2771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12(lean_object* v_00_u03b1_2772_, lean_object* v_name_2773_, lean_object* v_type_2774_, lean_object* v_val_2775_, lean_object* v_k_2776_, uint8_t v_nondep_2777_, uint8_t v_kind_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_){
_start:
{
lean_object* v___x_2785_; 
v___x_2785_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___redArg(v_name_2773_, v_type_2774_, v_val_2775_, v_k_2776_, v_nondep_2777_, v_kind_2778_, v___y_2779_, v___y_2780_, v___y_2781_, v___y_2782_, v___y_2783_);
return v___x_2785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12___boxed(lean_object* v_00_u03b1_2786_, lean_object* v_name_2787_, lean_object* v_type_2788_, lean_object* v_val_2789_, lean_object* v_k_2790_, lean_object* v_nondep_2791_, lean_object* v_kind_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_){
_start:
{
uint8_t v_nondep_boxed_2799_; uint8_t v_kind_boxed_2800_; lean_object* v_res_2801_; 
v_nondep_boxed_2799_ = lean_unbox(v_nondep_2791_);
v_kind_boxed_2800_ = lean_unbox(v_kind_2792_);
v_res_2801_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__9_spec__12(v_00_u03b1_2786_, v_name_2787_, v_type_2788_, v_val_2789_, v_k_2790_, v_nondep_boxed_2799_, v_kind_boxed_2800_, v___y_2793_, v___y_2794_, v___y_2795_, v___y_2796_, v___y_2797_);
lean_dec(v___y_2797_);
lean_dec_ref(v___y_2796_);
lean_dec(v___y_2795_);
lean_dec_ref(v___y_2794_);
lean_dec(v___y_2793_);
return v_res_2801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15(lean_object* v_00_u03b1_2802_, lean_object* v_ref_2803_, lean_object* v___y_2804_, lean_object* v___y_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_){
_start:
{
lean_object* v___x_2809_; 
v___x_2809_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___redArg(v_ref_2803_);
return v___x_2809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15___boxed(lean_object* v_00_u03b1_2810_, lean_object* v_ref_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_){
_start:
{
lean_object* v_res_2817_; 
v_res_2817_ = lp_mathlib_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11_spec__15(v_00_u03b1_2810_, v_ref_2811_, v___y_2812_, v___y_2813_, v___y_2814_, v___y_2815_);
lean_dec(v___y_2815_);
lean_dec_ref(v___y_2814_);
lean_dec(v___y_2813_);
lean_dec_ref(v___y_2812_);
return v_res_2817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11(lean_object* v_00_u03b1_2818_, lean_object* v_x_2819_, lean_object* v___y_2820_, lean_object* v___y_2821_, lean_object* v___y_2822_, lean_object* v___y_2823_, lean_object* v___y_2824_){
_start:
{
lean_object* v___x_2826_; 
v___x_2826_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___redArg(v_x_2819_, v___y_2820_, v___y_2821_, v___y_2822_, v___y_2823_, v___y_2824_);
return v___x_2826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11___boxed(lean_object* v_00_u03b1_2827_, lean_object* v_x_2828_, lean_object* v___y_2829_, lean_object* v___y_2830_, lean_object* v___y_2831_, lean_object* v___y_2832_, lean_object* v___y_2833_, lean_object* v___y_2834_){
_start:
{
lean_object* v_res_2835_; 
v_res_2835_ = lp_mathlib_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__11(v_00_u03b1_2827_, v_x_2828_, v___y_2829_, v___y_2830_, v___y_2831_, v___y_2832_, v___y_2833_);
lean_dec(v___y_2833_);
lean_dec_ref(v___y_2832_);
lean_dec(v___y_2831_);
lean_dec_ref(v___y_2830_);
lean_dec(v___y_2829_);
return v_res_2835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12(lean_object* v_00_u03b2_2836_, lean_object* v_m_2837_, lean_object* v_a_2838_, lean_object* v_b_2839_){
_start:
{
lean_object* v___x_2840_; 
v___x_2840_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12___redArg(v_m_2837_, v_a_2838_, v_b_2839_);
return v___x_2840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7(lean_object* v_00_u03b2_2841_, lean_object* v_a_2842_, lean_object* v_x_2843_){
_start:
{
lean_object* v___x_2844_; 
v___x_2844_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___redArg(v_a_2842_, v_x_2843_);
return v___x_2844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7___boxed(lean_object* v_00_u03b2_2845_, lean_object* v_a_2846_, lean_object* v_x_2847_){
_start:
{
lean_object* v_res_2848_; 
v_res_2848_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__6_spec__7(v_00_u03b2_2845_, v_a_2846_, v_x_2847_);
lean_dec(v_x_2847_);
lean_dec_ref(v_a_2846_);
return v_res_2848_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17(lean_object* v_00_u03b2_2849_, lean_object* v_a_2850_, lean_object* v_x_2851_){
_start:
{
uint8_t v___x_2852_; 
v___x_2852_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___redArg(v_a_2850_, v_x_2851_);
return v___x_2852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17___boxed(lean_object* v_00_u03b2_2853_, lean_object* v_a_2854_, lean_object* v_x_2855_){
_start:
{
uint8_t v_res_2856_; lean_object* v_r_2857_; 
v_res_2856_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__17(v_00_u03b2_2853_, v_a_2854_, v_x_2855_);
lean_dec(v_x_2855_);
lean_dec_ref(v_a_2854_);
v_r_2857_ = lean_box(v_res_2856_);
return v_r_2857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18(lean_object* v_00_u03b2_2858_, lean_object* v_data_2859_){
_start:
{
lean_object* v___x_2860_; 
v___x_2860_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18___redArg(v_data_2859_);
return v___x_2860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__19(lean_object* v_00_u03b2_2861_, lean_object* v_a_2862_, lean_object* v_b_2863_, lean_object* v_x_2864_){
_start:
{
lean_object* v___x_2865_; 
v___x_2865_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__19___redArg(v_a_2862_, v_b_2863_, v_x_2864_);
return v___x_2865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19(lean_object* v_00_u03b2_2866_, lean_object* v_i_2867_, lean_object* v_source_2868_, lean_object* v_target_2869_){
_start:
{
lean_object* v___x_2870_; 
v___x_2870_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19___redArg(v_i_2867_, v_source_2868_, v_target_2869_);
return v___x_2870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19_spec__20(lean_object* v_00_u03b2_2871_, lean_object* v_x_2872_, lean_object* v_x_2873_){
_start:
{
lean_object* v___x_2874_; 
v___x_2874_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2_spec__12_spec__18_spec__19_spec__20___redArg(v_x_2872_, v_x_2873_);
return v___x_2874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___redArg(lean_object* v_msg_2875_, lean_object* v___y_2876_, lean_object* v___y_2877_, lean_object* v___y_2878_, lean_object* v___y_2879_){
_start:
{
lean_object* v_ref_2881_; lean_object* v___x_2882_; lean_object* v_a_2883_; lean_object* v___x_2885_; uint8_t v_isShared_2886_; uint8_t v_isSharedCheck_2891_; 
v_ref_2881_ = lean_ctor_get(v___y_2878_, 5);
v___x_2882_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_runDefEqTactic_spec__1_spec__1(v_msg_2875_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_);
v_a_2883_ = lean_ctor_get(v___x_2882_, 0);
v_isSharedCheck_2891_ = !lean_is_exclusive(v___x_2882_);
if (v_isSharedCheck_2891_ == 0)
{
v___x_2885_ = v___x_2882_;
v_isShared_2886_ = v_isSharedCheck_2891_;
goto v_resetjp_2884_;
}
else
{
lean_inc(v_a_2883_);
lean_dec(v___x_2882_);
v___x_2885_ = lean_box(0);
v_isShared_2886_ = v_isSharedCheck_2891_;
goto v_resetjp_2884_;
}
v_resetjp_2884_:
{
lean_object* v___x_2887_; lean_object* v___x_2889_; 
lean_inc(v_ref_2881_);
v___x_2887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2887_, 0, v_ref_2881_);
lean_ctor_set(v___x_2887_, 1, v_a_2883_);
if (v_isShared_2886_ == 0)
{
lean_ctor_set_tag(v___x_2885_, 1);
lean_ctor_set(v___x_2885_, 0, v___x_2887_);
v___x_2889_ = v___x_2885_;
goto v_reusejp_2888_;
}
else
{
lean_object* v_reuseFailAlloc_2890_; 
v_reuseFailAlloc_2890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2890_, 0, v___x_2887_);
v___x_2889_ = v_reuseFailAlloc_2890_;
goto v_reusejp_2888_;
}
v_reusejp_2888_:
{
return v___x_2889_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___redArg___boxed(lean_object* v_msg_2892_, lean_object* v___y_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_, lean_object* v___y_2897_){
_start:
{
lean_object* v_res_2898_; 
v_res_2898_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___redArg(v_msg_2892_, v___y_2893_, v___y_2894_, v___y_2895_, v___y_2896_);
lean_dec(v___y_2896_);
lean_dec_ref(v___y_2895_);
lean_dec(v___y_2894_);
lean_dec_ref(v___y_2893_);
return v_res_2898_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__1(void){
_start:
{
lean_object* v___x_2900_; lean_object* v___x_2901_; 
v___x_2900_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__0));
v___x_2901_ = l_Lean_stringToMessageData(v___x_2900_);
return v___x_2901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1(lean_object* v_as_2902_, size_t v_sz_2903_, size_t v_i_2904_, lean_object* v_b_2905_, lean_object* v___y_2906_, lean_object* v___y_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_){
_start:
{
lean_object* v_a_2912_; uint8_t v___x_2916_; 
v___x_2916_ = lean_usize_dec_lt(v_i_2904_, v_sz_2903_);
if (v___x_2916_ == 0)
{
lean_object* v___x_2917_; 
v___x_2917_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2917_, 0, v_b_2905_);
return v___x_2917_;
}
else
{
lean_object* v_a_2918_; uint8_t v___x_2919_; lean_object* v___x_2920_; 
v_a_2918_ = lean_array_uget_borrowed(v_as_2902_, v_i_2904_);
v___x_2919_ = 0;
lean_inc(v_a_2918_);
v___x_2920_ = l_Lean_FVarId_getValue_x3f___redArg(v_a_2918_, v___x_2919_, v___y_2906_, v___y_2908_, v___y_2909_);
if (lean_obj_tag(v___x_2920_) == 0)
{
lean_object* v_a_2921_; 
v_a_2921_ = lean_ctor_get(v___x_2920_, 0);
lean_inc(v_a_2921_);
lean_dec_ref_known(v___x_2920_, 1);
if (lean_obj_tag(v_a_2921_) == 1)
{
lean_object* v_val_2922_; lean_object* v___x_2923_; lean_object* v___x_2924_; 
v_val_2922_ = lean_ctor_get(v_a_2921_, 0);
lean_inc(v_val_2922_);
lean_dec_ref_known(v_a_2921_, 1);
v___x_2923_ = lean_box(0);
v___x_2924_ = l_Lean_Meta_kabstract(v_b_2905_, v_val_2922_, v___x_2923_, v___y_2906_, v___y_2907_, v___y_2908_, v___y_2909_);
if (lean_obj_tag(v___x_2924_) == 0)
{
lean_object* v_a_2925_; lean_object* v___x_2926_; lean_object* v___x_2927_; 
v_a_2925_ = lean_ctor_get(v___x_2924_, 0);
lean_inc(v_a_2925_);
lean_dec_ref_known(v___x_2924_, 1);
lean_inc(v_a_2918_);
v___x_2926_ = l_Lean_Expr_fvar___override(v_a_2918_);
v___x_2927_ = lean_expr_instantiate1(v_a_2925_, v___x_2926_);
lean_dec_ref(v___x_2926_);
lean_dec(v_a_2925_);
v_a_2912_ = v___x_2927_;
goto v___jp_2911_;
}
else
{
return v___x_2924_;
}
}
else
{
lean_object* v___x_2928_; lean_object* v___x_2929_; lean_object* v___x_2930_; lean_object* v___x_2931_; lean_object* v___x_2932_; lean_object* v___x_2933_; lean_object* v___x_2934_; 
lean_dec(v_a_2921_);
v___x_2928_ = lean_obj_once(&lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4, &lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4_once, _init_lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__4);
lean_inc(v_a_2918_);
v___x_2929_ = l_Lean_Expr_fvar___override(v_a_2918_);
v___x_2930_ = l_Lean_MessageData_ofExpr(v___x_2929_);
v___x_2931_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2931_, 0, v___x_2928_);
lean_ctor_set(v___x_2931_, 1, v___x_2930_);
v___x_2932_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___closed__1);
v___x_2933_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2933_, 0, v___x_2931_);
lean_ctor_set(v___x_2933_, 1, v___x_2932_);
v___x_2934_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___redArg(v___x_2933_, v___y_2906_, v___y_2907_, v___y_2908_, v___y_2909_);
if (lean_obj_tag(v___x_2934_) == 0)
{
lean_dec_ref_known(v___x_2934_, 1);
v_a_2912_ = v_b_2905_;
goto v___jp_2911_;
}
else
{
lean_object* v_a_2935_; lean_object* v___x_2937_; uint8_t v_isShared_2938_; uint8_t v_isSharedCheck_2942_; 
lean_dec_ref(v_b_2905_);
v_a_2935_ = lean_ctor_get(v___x_2934_, 0);
v_isSharedCheck_2942_ = !lean_is_exclusive(v___x_2934_);
if (v_isSharedCheck_2942_ == 0)
{
v___x_2937_ = v___x_2934_;
v_isShared_2938_ = v_isSharedCheck_2942_;
goto v_resetjp_2936_;
}
else
{
lean_inc(v_a_2935_);
lean_dec(v___x_2934_);
v___x_2937_ = lean_box(0);
v_isShared_2938_ = v_isSharedCheck_2942_;
goto v_resetjp_2936_;
}
v_resetjp_2936_:
{
lean_object* v___x_2940_; 
if (v_isShared_2938_ == 0)
{
v___x_2940_ = v___x_2937_;
goto v_reusejp_2939_;
}
else
{
lean_object* v_reuseFailAlloc_2941_; 
v_reuseFailAlloc_2941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2941_, 0, v_a_2935_);
v___x_2940_ = v_reuseFailAlloc_2941_;
goto v_reusejp_2939_;
}
v_reusejp_2939_:
{
return v___x_2940_;
}
}
}
}
}
else
{
lean_object* v_a_2943_; lean_object* v___x_2945_; uint8_t v_isShared_2946_; uint8_t v_isSharedCheck_2950_; 
lean_dec_ref(v_b_2905_);
v_a_2943_ = lean_ctor_get(v___x_2920_, 0);
v_isSharedCheck_2950_ = !lean_is_exclusive(v___x_2920_);
if (v_isSharedCheck_2950_ == 0)
{
v___x_2945_ = v___x_2920_;
v_isShared_2946_ = v_isSharedCheck_2950_;
goto v_resetjp_2944_;
}
else
{
lean_inc(v_a_2943_);
lean_dec(v___x_2920_);
v___x_2945_ = lean_box(0);
v_isShared_2946_ = v_isSharedCheck_2950_;
goto v_resetjp_2944_;
}
v_resetjp_2944_:
{
lean_object* v___x_2948_; 
if (v_isShared_2946_ == 0)
{
v___x_2948_ = v___x_2945_;
goto v_reusejp_2947_;
}
else
{
lean_object* v_reuseFailAlloc_2949_; 
v_reuseFailAlloc_2949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2949_, 0, v_a_2943_);
v___x_2948_ = v_reuseFailAlloc_2949_;
goto v_reusejp_2947_;
}
v_reusejp_2947_:
{
return v___x_2948_;
}
}
}
}
v___jp_2911_:
{
size_t v___x_2913_; size_t v___x_2914_; 
v___x_2913_ = ((size_t)1ULL);
v___x_2914_ = lean_usize_add(v_i_2904_, v___x_2913_);
v_i_2904_ = v___x_2914_;
v_b_2905_ = v_a_2912_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1___boxed(lean_object* v_as_2951_, lean_object* v_sz_2952_, lean_object* v_i_2953_, lean_object* v_b_2954_, lean_object* v___y_2955_, lean_object* v___y_2956_, lean_object* v___y_2957_, lean_object* v___y_2958_, lean_object* v___y_2959_){
_start:
{
size_t v_sz_boxed_2960_; size_t v_i_boxed_2961_; lean_object* v_res_2962_; 
v_sz_boxed_2960_ = lean_unbox_usize(v_sz_2952_);
lean_dec(v_sz_2952_);
v_i_boxed_2961_ = lean_unbox_usize(v_i_2953_);
lean_dec(v_i_2953_);
v_res_2962_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1(v_as_2951_, v_sz_boxed_2960_, v_i_boxed_2961_, v_b_2954_, v___y_2955_, v___y_2956_, v___y_2957_, v___y_2958_);
lean_dec(v___y_2958_);
lean_dec_ref(v___y_2957_);
lean_dec(v___y_2956_);
lean_dec_ref(v___y_2955_);
lean_dec_ref(v_as_2951_);
return v_res_2962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg(lean_object* v___x_2963_, lean_object* v_as_2964_, size_t v_i_2965_, size_t v_stop_2966_, lean_object* v_b_2967_, lean_object* v___y_2968_){
_start:
{
uint8_t v___x_2970_; 
v___x_2970_ = lean_usize_dec_eq(v_i_2965_, v_stop_2966_);
if (v___x_2970_ == 0)
{
lean_object* v___x_2971_; lean_object* v___x_2972_; 
v___x_2971_ = lean_array_uget_borrowed(v_as_2964_, v_i_2965_);
lean_inc(v___x_2971_);
v___x_2972_ = l_Lean_FVarId_findDecl_x3f___redArg(v___x_2971_, v___y_2968_);
if (lean_obj_tag(v___x_2972_) == 0)
{
lean_object* v_a_2973_; lean_object* v_a_2975_; 
v_a_2973_ = lean_ctor_get(v___x_2972_, 0);
lean_inc(v_a_2973_);
lean_dec_ref_known(v___x_2972_, 1);
if (lean_obj_tag(v_a_2973_) == 1)
{
lean_object* v_val_2979_; lean_object* v___x_2980_; uint8_t v___x_2981_; 
v_val_2979_ = lean_ctor_get(v_a_2973_, 0);
lean_inc(v_val_2979_);
lean_dec_ref_known(v_a_2973_, 1);
v___x_2980_ = l_Lean_LocalDecl_index(v_val_2979_);
lean_dec(v_val_2979_);
v___x_2981_ = lean_nat_dec_lt(v___x_2980_, v___x_2963_);
lean_dec(v___x_2980_);
if (v___x_2981_ == 0)
{
v_a_2975_ = v_b_2967_;
goto v___jp_2974_;
}
else
{
lean_object* v___x_2982_; 
lean_inc(v___x_2971_);
v___x_2982_ = lean_array_push(v_b_2967_, v___x_2971_);
v_a_2975_ = v___x_2982_;
goto v___jp_2974_;
}
}
else
{
lean_dec(v_a_2973_);
v_a_2975_ = v_b_2967_;
goto v___jp_2974_;
}
v___jp_2974_:
{
size_t v___x_2976_; size_t v___x_2977_; 
v___x_2976_ = ((size_t)1ULL);
v___x_2977_ = lean_usize_add(v_i_2965_, v___x_2976_);
v_i_2965_ = v___x_2977_;
v_b_2967_ = v_a_2975_;
goto _start;
}
}
else
{
lean_object* v_a_2983_; lean_object* v___x_2985_; uint8_t v_isShared_2986_; uint8_t v_isSharedCheck_2990_; 
lean_dec_ref(v_b_2967_);
v_a_2983_ = lean_ctor_get(v___x_2972_, 0);
v_isSharedCheck_2990_ = !lean_is_exclusive(v___x_2972_);
if (v_isSharedCheck_2990_ == 0)
{
v___x_2985_ = v___x_2972_;
v_isShared_2986_ = v_isSharedCheck_2990_;
goto v_resetjp_2984_;
}
else
{
lean_inc(v_a_2983_);
lean_dec(v___x_2972_);
v___x_2985_ = lean_box(0);
v_isShared_2986_ = v_isSharedCheck_2990_;
goto v_resetjp_2984_;
}
v_resetjp_2984_:
{
lean_object* v___x_2988_; 
if (v_isShared_2986_ == 0)
{
v___x_2988_ = v___x_2985_;
goto v_reusejp_2987_;
}
else
{
lean_object* v_reuseFailAlloc_2989_; 
v_reuseFailAlloc_2989_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2989_, 0, v_a_2983_);
v___x_2988_ = v_reuseFailAlloc_2989_;
goto v_reusejp_2987_;
}
v_reusejp_2987_:
{
return v___x_2988_;
}
}
}
}
else
{
lean_object* v___x_2991_; 
v___x_2991_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2991_, 0, v_b_2967_);
return v___x_2991_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg___boxed(lean_object* v___x_2992_, lean_object* v_as_2993_, lean_object* v_i_2994_, lean_object* v_stop_2995_, lean_object* v_b_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_){
_start:
{
size_t v_i_boxed_2999_; size_t v_stop_boxed_3000_; lean_object* v_res_3001_; 
v_i_boxed_2999_ = lean_unbox_usize(v_i_2994_);
lean_dec(v_i_2994_);
v_stop_boxed_3000_ = lean_unbox_usize(v_stop_2995_);
lean_dec(v_stop_2995_);
v_res_3001_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg(v___x_2992_, v_as_2993_, v_i_boxed_2999_, v_stop_boxed_3000_, v_b_2996_, v___y_2997_);
lean_dec_ref(v___y_2997_);
lean_dec_ref(v_as_2993_);
lean_dec(v___x_2992_);
return v_res_3001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_refoldFVars(lean_object* v_fvars_3002_, lean_object* v_loc_x3f_3003_, lean_object* v_e_3004_, lean_object* v_a_3005_, lean_object* v_a_3006_, lean_object* v_a_3007_, lean_object* v_a_3008_){
_start:
{
lean_object* v_fvars_3011_; lean_object* v___y_3012_; lean_object* v___y_3013_; lean_object* v___y_3014_; lean_object* v___y_3015_; lean_object* v___y_3020_; 
if (lean_obj_tag(v_loc_x3f_3003_) == 1)
{
lean_object* v_val_3030_; lean_object* v___x_3031_; 
v_val_3030_ = lean_ctor_get(v_loc_x3f_3003_, 0);
lean_inc(v_val_3030_);
lean_dec_ref_known(v_loc_x3f_3003_, 1);
v___x_3031_ = l_Lean_FVarId_getDecl___redArg(v_val_3030_, v_a_3005_, v_a_3007_, v_a_3008_);
if (lean_obj_tag(v___x_3031_) == 0)
{
lean_object* v_a_3032_; lean_object* v___x_3033_; lean_object* v___x_3034_; lean_object* v___x_3035_; uint8_t v___x_3036_; 
v_a_3032_ = lean_ctor_get(v___x_3031_, 0);
lean_inc(v_a_3032_);
lean_dec_ref_known(v___x_3031_, 1);
v___x_3033_ = lean_unsigned_to_nat(0u);
v___x_3034_ = lean_array_get_size(v_fvars_3002_);
v___x_3035_ = ((lean_object*)(lp_mathlib_Lean_MVarId_changeLocalDecl_x27___closed__2));
v___x_3036_ = lean_nat_dec_lt(v___x_3033_, v___x_3034_);
if (v___x_3036_ == 0)
{
lean_dec(v_a_3032_);
lean_dec_ref(v_fvars_3002_);
v_fvars_3011_ = v___x_3035_;
v___y_3012_ = v_a_3005_;
v___y_3013_ = v_a_3006_;
v___y_3014_ = v_a_3007_;
v___y_3015_ = v_a_3008_;
goto v___jp_3010_;
}
else
{
lean_object* v___x_3037_; uint8_t v___x_3038_; 
v___x_3037_ = l_Lean_LocalDecl_index(v_a_3032_);
lean_dec(v_a_3032_);
v___x_3038_ = lean_nat_dec_le(v___x_3034_, v___x_3034_);
if (v___x_3038_ == 0)
{
if (v___x_3036_ == 0)
{
lean_dec(v___x_3037_);
lean_dec_ref(v_fvars_3002_);
v_fvars_3011_ = v___x_3035_;
v___y_3012_ = v_a_3005_;
v___y_3013_ = v_a_3006_;
v___y_3014_ = v_a_3007_;
v___y_3015_ = v_a_3008_;
goto v___jp_3010_;
}
else
{
size_t v___x_3039_; size_t v___x_3040_; lean_object* v___x_3041_; 
v___x_3039_ = ((size_t)0ULL);
v___x_3040_ = lean_usize_of_nat(v___x_3034_);
v___x_3041_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg(v___x_3037_, v_fvars_3002_, v___x_3039_, v___x_3040_, v___x_3035_, v_a_3005_);
lean_dec_ref(v_fvars_3002_);
lean_dec(v___x_3037_);
v___y_3020_ = v___x_3041_;
goto v___jp_3019_;
}
}
else
{
size_t v___x_3042_; size_t v___x_3043_; lean_object* v___x_3044_; 
v___x_3042_ = ((size_t)0ULL);
v___x_3043_ = lean_usize_of_nat(v___x_3034_);
v___x_3044_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg(v___x_3037_, v_fvars_3002_, v___x_3042_, v___x_3043_, v___x_3035_, v_a_3005_);
lean_dec_ref(v_fvars_3002_);
lean_dec(v___x_3037_);
v___y_3020_ = v___x_3044_;
goto v___jp_3019_;
}
}
}
else
{
lean_object* v_a_3045_; lean_object* v___x_3047_; uint8_t v_isShared_3048_; uint8_t v_isSharedCheck_3052_; 
lean_dec_ref(v_e_3004_);
lean_dec_ref(v_fvars_3002_);
v_a_3045_ = lean_ctor_get(v___x_3031_, 0);
v_isSharedCheck_3052_ = !lean_is_exclusive(v___x_3031_);
if (v_isSharedCheck_3052_ == 0)
{
v___x_3047_ = v___x_3031_;
v_isShared_3048_ = v_isSharedCheck_3052_;
goto v_resetjp_3046_;
}
else
{
lean_inc(v_a_3045_);
lean_dec(v___x_3031_);
v___x_3047_ = lean_box(0);
v_isShared_3048_ = v_isSharedCheck_3052_;
goto v_resetjp_3046_;
}
v_resetjp_3046_:
{
lean_object* v___x_3050_; 
if (v_isShared_3048_ == 0)
{
v___x_3050_ = v___x_3047_;
goto v_reusejp_3049_;
}
else
{
lean_object* v_reuseFailAlloc_3051_; 
v_reuseFailAlloc_3051_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3051_, 0, v_a_3045_);
v___x_3050_ = v_reuseFailAlloc_3051_;
goto v_reusejp_3049_;
}
v_reusejp_3049_:
{
return v___x_3050_;
}
}
}
}
else
{
lean_dec(v_loc_x3f_3003_);
v_fvars_3011_ = v_fvars_3002_;
v___y_3012_ = v_a_3005_;
v___y_3013_ = v_a_3006_;
v___y_3014_ = v_a_3007_;
v___y_3015_ = v_a_3008_;
goto v___jp_3010_;
}
v___jp_3010_:
{
size_t v_sz_3016_; size_t v___x_3017_; lean_object* v___x_3018_; 
v_sz_3016_ = lean_array_size(v_fvars_3011_);
v___x_3017_ = ((size_t)0ULL);
v___x_3018_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_refoldFVars_spec__1(v_fvars_3011_, v_sz_3016_, v___x_3017_, v_e_3004_, v___y_3012_, v___y_3013_, v___y_3014_, v___y_3015_);
lean_dec_ref(v_fvars_3011_);
return v___x_3018_;
}
v___jp_3019_:
{
if (lean_obj_tag(v___y_3020_) == 0)
{
lean_object* v_a_3021_; 
v_a_3021_ = lean_ctor_get(v___y_3020_, 0);
lean_inc(v_a_3021_);
lean_dec_ref_known(v___y_3020_, 1);
v_fvars_3011_ = v_a_3021_;
v___y_3012_ = v_a_3005_;
v___y_3013_ = v_a_3006_;
v___y_3014_ = v_a_3007_;
v___y_3015_ = v_a_3008_;
goto v___jp_3010_;
}
else
{
lean_object* v_a_3022_; lean_object* v___x_3024_; uint8_t v_isShared_3025_; uint8_t v_isSharedCheck_3029_; 
lean_dec_ref(v_e_3004_);
v_a_3022_ = lean_ctor_get(v___y_3020_, 0);
v_isSharedCheck_3029_ = !lean_is_exclusive(v___y_3020_);
if (v_isSharedCheck_3029_ == 0)
{
v___x_3024_ = v___y_3020_;
v_isShared_3025_ = v_isSharedCheck_3029_;
goto v_resetjp_3023_;
}
else
{
lean_inc(v_a_3022_);
lean_dec(v___y_3020_);
v___x_3024_ = lean_box(0);
v_isShared_3025_ = v_isSharedCheck_3029_;
goto v_resetjp_3023_;
}
v_resetjp_3023_:
{
lean_object* v___x_3027_; 
if (v_isShared_3025_ == 0)
{
v___x_3027_ = v___x_3024_;
goto v_reusejp_3026_;
}
else
{
lean_object* v_reuseFailAlloc_3028_; 
v_reuseFailAlloc_3028_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3028_, 0, v_a_3022_);
v___x_3027_ = v_reuseFailAlloc_3028_;
goto v_reusejp_3026_;
}
v_reusejp_3026_:
{
return v___x_3027_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_refoldFVars___boxed(lean_object* v_fvars_3053_, lean_object* v_loc_x3f_3054_, lean_object* v_e_3055_, lean_object* v_a_3056_, lean_object* v_a_3057_, lean_object* v_a_3058_, lean_object* v_a_3059_, lean_object* v_a_3060_){
_start:
{
lean_object* v_res_3061_; 
v_res_3061_ = lp_mathlib_Mathlib_Tactic_refoldFVars(v_fvars_3053_, v_loc_x3f_3054_, v_e_3055_, v_a_3056_, v_a_3057_, v_a_3058_, v_a_3059_);
lean_dec(v_a_3059_);
lean_dec_ref(v_a_3058_);
lean_dec(v_a_3057_);
lean_dec_ref(v_a_3056_);
return v_res_3061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0(lean_object* v_00_u03b1_3062_, lean_object* v_msg_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_){
_start:
{
lean_object* v___x_3069_; 
v___x_3069_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___redArg(v_msg_3063_, v___y_3064_, v___y_3065_, v___y_3066_, v___y_3067_);
return v___x_3069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0___boxed(lean_object* v_00_u03b1_3070_, lean_object* v_msg_3071_, lean_object* v___y_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_){
_start:
{
lean_object* v_res_3077_; 
v_res_3077_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_refoldFVars_spec__0(v_00_u03b1_3070_, v_msg_3071_, v___y_3072_, v___y_3073_, v___y_3074_, v___y_3075_);
lean_dec(v___y_3075_);
lean_dec_ref(v___y_3074_);
lean_dec(v___y_3073_);
lean_dec_ref(v___y_3072_);
return v_res_3077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2(lean_object* v___x_3078_, lean_object* v_as_3079_, size_t v_i_3080_, size_t v_stop_3081_, lean_object* v_b_3082_, lean_object* v___y_3083_, lean_object* v___y_3084_, lean_object* v___y_3085_, lean_object* v___y_3086_){
_start:
{
lean_object* v___x_3088_; 
v___x_3088_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___redArg(v___x_3078_, v_as_3079_, v_i_3080_, v_stop_3081_, v_b_3082_, v___y_3083_);
return v___x_3088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2___boxed(lean_object* v___x_3089_, lean_object* v_as_3090_, lean_object* v_i_3091_, lean_object* v_stop_3092_, lean_object* v_b_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_, lean_object* v___y_3097_, lean_object* v___y_3098_){
_start:
{
size_t v_i_boxed_3099_; size_t v_stop_boxed_3100_; lean_object* v_res_3101_; 
v_i_boxed_3099_ = lean_unbox_usize(v_i_3091_);
lean_dec(v_i_3091_);
v_stop_boxed_3100_ = lean_unbox_usize(v_stop_3092_);
lean_dec(v_stop_3092_);
v_res_3101_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_refoldFVars_spec__2(v___x_3089_, v_as_3090_, v_i_boxed_3099_, v_stop_boxed_3100_, v_b_3093_, v___y_3094_, v___y_3095_, v___y_3096_, v___y_3097_);
lean_dec(v___y_3097_);
lean_dec_ref(v___y_3096_);
lean_dec(v___y_3095_);
lean_dec_ref(v___y_3094_);
lean_dec_ref(v_as_3090_);
lean_dec(v___x_3089_);
return v_res_3101_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__16(void){
_start:
{
lean_object* v___x_3140_; lean_object* v___x_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; 
v___x_3140_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_3141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__15));
v___x_3142_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_3143_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3143_, 0, v___x_3142_);
lean_ctor_set(v___x_3143_, 1, v___x_3141_);
lean_ctor_set(v___x_3143_, 2, v___x_3140_);
return v___x_3143_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__17(void){
_start:
{
lean_object* v___x_3144_; lean_object* v___x_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; 
v___x_3144_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__16, &lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__16);
v___x_3145_ = lean_unsigned_to_nat(1022u);
v___x_3146_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1));
v___x_3147_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3147_, 0, v___x_3146_);
lean_ctor_set(v___x_3147_, 1, v___x_3145_);
lean_ctor_set(v___x_3147_, 2, v___x_3144_);
return v___x_3147_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_refoldLetStx(void){
_start:
{
lean_object* v___x_3148_; 
v___x_3148_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__17, &lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__17);
return v___x_3148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__refoldLetStx__1(lean_object* v_x_3149_, lean_object* v_a_3150_, lean_object* v_a_3151_, lean_object* v_a_3152_, lean_object* v_a_3153_, lean_object* v_a_3154_, lean_object* v_a_3155_, lean_object* v_a_3156_, lean_object* v_a_3157_){
_start:
{
lean_object* v___x_3159_; uint8_t v___x_3160_; 
v___x_3159_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__1));
lean_inc(v_x_3149_);
v___x_3160_ = l_Lean_Syntax_isOfKind(v_x_3149_, v___x_3159_);
if (v___x_3160_ == 0)
{
lean_object* v___x_3161_; 
lean_dec(v_x_3149_);
v___x_3161_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_3161_;
}
else
{
lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v_loc_x3f_3165_; lean_object* v___y_3166_; lean_object* v___y_3167_; lean_object* v___y_3168_; lean_object* v___y_3169_; lean_object* v___y_3170_; lean_object* v___y_3171_; lean_object* v___y_3172_; lean_object* v___y_3173_; lean_object* v___x_3188_; lean_object* v___x_3189_; uint8_t v___x_3190_; 
v___x_3162_ = lean_unsigned_to_nat(1u);
v___x_3163_ = l_Lean_Syntax_getArg(v_x_3149_, v___x_3162_);
v___x_3188_ = lean_unsigned_to_nat(2u);
v___x_3189_ = l_Lean_Syntax_getArg(v_x_3149_, v___x_3188_);
lean_dec(v_x_3149_);
v___x_3190_ = l_Lean_Syntax_isNone(v___x_3189_);
if (v___x_3190_ == 0)
{
uint8_t v___x_3191_; 
lean_inc(v___x_3189_);
v___x_3191_ = l_Lean_Syntax_matchesNull(v___x_3189_, v___x_3162_);
if (v___x_3191_ == 0)
{
lean_object* v___x_3192_; 
lean_dec(v___x_3189_);
lean_dec(v___x_3163_);
v___x_3192_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_3192_;
}
else
{
lean_object* v___x_3193_; lean_object* v_loc_x3f_3194_; lean_object* v___x_3195_; 
v___x_3193_ = lean_unsigned_to_nat(0u);
v_loc_x3f_3194_ = l_Lean_Syntax_getArg(v___x_3189_, v___x_3193_);
lean_dec(v___x_3189_);
v___x_3195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3195_, 0, v_loc_x3f_3194_);
v_loc_x3f_3165_ = v___x_3195_;
v___y_3166_ = v_a_3150_;
v___y_3167_ = v_a_3151_;
v___y_3168_ = v_a_3152_;
v___y_3169_ = v_a_3153_;
v___y_3170_ = v_a_3154_;
v___y_3171_ = v_a_3155_;
v___y_3172_ = v_a_3156_;
v___y_3173_ = v_a_3157_;
goto v___jp_3164_;
}
}
else
{
lean_object* v___x_3196_; 
lean_dec(v___x_3189_);
v___x_3196_ = lean_box(0);
v_loc_x3f_3165_ = v___x_3196_;
v___y_3166_ = v_a_3150_;
v___y_3167_ = v_a_3151_;
v___y_3168_ = v_a_3152_;
v___y_3169_ = v_a_3153_;
v___y_3170_ = v_a_3154_;
v___y_3171_ = v_a_3155_;
v___y_3172_ = v_a_3156_;
v___y_3173_ = v_a_3157_;
goto v___jp_3164_;
}
v___jp_3164_:
{
lean_object* v_hs_3174_; lean_object* v___x_3175_; 
v_hs_3174_ = l_Lean_Syntax_getArgs(v___x_3163_);
lean_dec(v___x_3163_);
v___x_3175_ = l_Lean_Elab_Tactic_getFVarIds(v_hs_3174_, v___y_3166_, v___y_3167_, v___y_3168_, v___y_3169_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
if (lean_obj_tag(v___x_3175_) == 0)
{
lean_object* v_a_3176_; lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; 
v_a_3176_ = lean_ctor_get(v___x_3175_, 0);
lean_inc(v_a_3176_);
lean_dec_ref_known(v___x_3175_, 1);
v___x_3177_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_refoldFVars___boxed), 8, 1);
lean_closure_set(v___x_3177_, 0, v_a_3176_);
v___x_3178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_refoldLetStx___closed__2));
v___x_3179_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___x_3177_, v_loc_x3f_3165_, v___x_3178_, v___x_3160_, v___y_3166_, v___y_3167_, v___y_3168_, v___y_3169_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
return v___x_3179_;
}
else
{
lean_object* v_a_3180_; lean_object* v___x_3182_; uint8_t v_isShared_3183_; uint8_t v_isSharedCheck_3187_; 
lean_dec(v_loc_x3f_3165_);
v_a_3180_ = lean_ctor_get(v___x_3175_, 0);
v_isSharedCheck_3187_ = !lean_is_exclusive(v___x_3175_);
if (v_isSharedCheck_3187_ == 0)
{
v___x_3182_ = v___x_3175_;
v_isShared_3183_ = v_isSharedCheck_3187_;
goto v_resetjp_3181_;
}
else
{
lean_inc(v_a_3180_);
lean_dec(v___x_3175_);
v___x_3182_ = lean_box(0);
v_isShared_3183_ = v_isSharedCheck_3187_;
goto v_resetjp_3181_;
}
v_resetjp_3181_:
{
lean_object* v___x_3185_; 
if (v_isShared_3183_ == 0)
{
v___x_3185_ = v___x_3182_;
goto v_reusejp_3184_;
}
else
{
lean_object* v_reuseFailAlloc_3186_; 
v_reuseFailAlloc_3186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3186_, 0, v_a_3180_);
v___x_3185_ = v_reuseFailAlloc_3186_;
goto v_reusejp_3184_;
}
v_reusejp_3184_:
{
return v___x_3185_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__refoldLetStx__1___boxed(lean_object* v_x_3197_, lean_object* v_a_3198_, lean_object* v_a_3199_, lean_object* v_a_3200_, lean_object* v_a_3201_, lean_object* v_a_3202_, lean_object* v_a_3203_, lean_object* v_a_3204_, lean_object* v_a_3205_, lean_object* v_a_3206_){
_start:
{
lean_object* v_res_3207_; 
v_res_3207_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__refoldLetStx__1(v_x_3197_, v_a_3198_, v_a_3199_, v_a_3200_, v_a_3201_, v_a_3202_, v_a_3203_, v_a_3204_, v_a_3205_);
lean_dec(v_a_3205_);
lean_dec_ref(v_a_3204_);
lean_dec(v_a_3203_);
lean_dec_ref(v_a_3202_);
lean_dec(v_a_3201_);
lean_dec_ref(v_a_3200_);
lean_dec(v_a_3199_);
lean_dec_ref(v_a_3198_);
return v_res_3207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convRefold__let________1(lean_object* v_x_3218_, lean_object* v_a_3219_, lean_object* v_a_3220_, lean_object* v_a_3221_, lean_object* v_a_3222_, lean_object* v_a_3223_, lean_object* v_a_3224_, lean_object* v_a_3225_, lean_object* v_a_3226_){
_start:
{
lean_object* v___x_3228_; uint8_t v___x_3229_; 
v___x_3228_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convRefold__let_______00__closed__1));
lean_inc(v_x_3218_);
v___x_3229_ = l_Lean_Syntax_isOfKind(v_x_3218_, v___x_3228_);
if (v___x_3229_ == 0)
{
lean_object* v___x_3230_; 
lean_dec(v_x_3218_);
v___x_3230_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_3230_;
}
else
{
lean_object* v___x_3231_; lean_object* v___x_3232_; lean_object* v_hs_3233_; lean_object* v___x_3234_; 
v___x_3231_ = lean_unsigned_to_nat(1u);
v___x_3232_ = l_Lean_Syntax_getArg(v_x_3218_, v___x_3231_);
lean_dec(v_x_3218_);
v_hs_3233_ = l_Lean_Syntax_getArgs(v___x_3232_);
lean_dec(v___x_3232_);
v___x_3234_ = l_Lean_Elab_Tactic_getFVarIds(v_hs_3233_, v_a_3219_, v_a_3220_, v_a_3221_, v_a_3222_, v_a_3223_, v_a_3224_, v_a_3225_, v_a_3226_);
if (lean_obj_tag(v___x_3234_) == 0)
{
lean_object* v_a_3235_; lean_object* v___x_3236_; lean_object* v___x_3237_; lean_object* v___x_3238_; 
v_a_3235_ = lean_ctor_get(v___x_3234_, 0);
lean_inc(v_a_3235_);
lean_dec_ref_known(v___x_3234_, 1);
v___x_3236_ = lean_box(0);
v___x_3237_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_refoldFVars___boxed), 8, 2);
lean_closure_set(v___x_3237_, 0, v_a_3235_);
lean_closure_set(v___x_3237_, 1, v___x_3236_);
v___x_3238_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(v___x_3237_, v_a_3219_, v_a_3220_, v_a_3221_, v_a_3222_, v_a_3223_, v_a_3224_, v_a_3225_, v_a_3226_);
return v___x_3238_;
}
else
{
lean_object* v_a_3239_; lean_object* v___x_3241_; uint8_t v_isShared_3242_; uint8_t v_isSharedCheck_3246_; 
v_a_3239_ = lean_ctor_get(v___x_3234_, 0);
v_isSharedCheck_3246_ = !lean_is_exclusive(v___x_3234_);
if (v_isSharedCheck_3246_ == 0)
{
v___x_3241_ = v___x_3234_;
v_isShared_3242_ = v_isSharedCheck_3246_;
goto v_resetjp_3240_;
}
else
{
lean_inc(v_a_3239_);
lean_dec(v___x_3234_);
v___x_3241_ = lean_box(0);
v_isShared_3242_ = v_isSharedCheck_3246_;
goto v_resetjp_3240_;
}
v_resetjp_3240_:
{
lean_object* v___x_3244_; 
if (v_isShared_3242_ == 0)
{
v___x_3244_ = v___x_3241_;
goto v_reusejp_3243_;
}
else
{
lean_object* v_reuseFailAlloc_3245_; 
v_reuseFailAlloc_3245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3245_, 0, v_a_3239_);
v___x_3244_ = v_reuseFailAlloc_3245_;
goto v_reusejp_3243_;
}
v_reusejp_3243_:
{
return v___x_3244_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convRefold__let________1___boxed(lean_object* v_x_3247_, lean_object* v_a_3248_, lean_object* v_a_3249_, lean_object* v_a_3250_, lean_object* v_a_3251_, lean_object* v_a_3252_, lean_object* v_a_3253_, lean_object* v_a_3254_, lean_object* v_a_3255_, lean_object* v_a_3256_){
_start:
{
lean_object* v_res_3257_; 
v_res_3257_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convRefold__let________1(v_x_3247_, v_a_3248_, v_a_3249_, v_a_3250_, v_a_3251_, v_a_3252_, v_a_3253_, v_a_3254_, v_a_3255_);
lean_dec(v_a_3255_);
lean_dec_ref(v_a_3254_);
lean_dec(v_a_3253_);
lean_dec_ref(v_a_3252_);
lean_dec(v_a_3251_);
lean_dec_ref(v_a_3250_);
lean_dec(v_a_3249_);
lean_dec_ref(v_a_3248_);
return v_res_3257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs___lam__1(lean_object* v_node_3258_, lean_object* v___y_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_, lean_object* v___y_3262_){
_start:
{
lean_object* v___x_3264_; 
v___x_3264_ = l_Lean_Meta_unfoldProjInst_x3f(v_node_3258_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_);
if (lean_obj_tag(v___x_3264_) == 0)
{
lean_object* v_a_3265_; lean_object* v___x_3267_; uint8_t v_isShared_3268_; uint8_t v_isSharedCheck_3290_; 
v_a_3265_ = lean_ctor_get(v___x_3264_, 0);
v_isSharedCheck_3290_ = !lean_is_exclusive(v___x_3264_);
if (v_isSharedCheck_3290_ == 0)
{
v___x_3267_ = v___x_3264_;
v_isShared_3268_ = v_isSharedCheck_3290_;
goto v_resetjp_3266_;
}
else
{
lean_inc(v_a_3265_);
lean_dec(v___x_3264_);
v___x_3267_ = lean_box(0);
v_isShared_3268_ = v_isSharedCheck_3290_;
goto v_resetjp_3266_;
}
v_resetjp_3266_:
{
if (lean_obj_tag(v_a_3265_) == 1)
{
lean_object* v_val_3269_; lean_object* v___x_3271_; uint8_t v_isShared_3272_; uint8_t v_isSharedCheck_3285_; 
lean_del_object(v___x_3267_);
v_val_3269_ = lean_ctor_get(v_a_3265_, 0);
v_isSharedCheck_3285_ = !lean_is_exclusive(v_a_3265_);
if (v_isSharedCheck_3285_ == 0)
{
v___x_3271_ = v_a_3265_;
v_isShared_3272_ = v_isSharedCheck_3285_;
goto v_resetjp_3270_;
}
else
{
lean_inc(v_val_3269_);
lean_dec(v_a_3265_);
v___x_3271_ = lean_box(0);
v_isShared_3272_ = v_isSharedCheck_3285_;
goto v_resetjp_3270_;
}
v_resetjp_3270_:
{
lean_object* v___x_3273_; lean_object* v_a_3274_; lean_object* v___x_3276_; uint8_t v_isShared_3277_; uint8_t v_isSharedCheck_3284_; 
v___x_3273_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_runDefEqTactic_spec__0___redArg(v_val_3269_, v___y_3260_);
v_a_3274_ = lean_ctor_get(v___x_3273_, 0);
v_isSharedCheck_3284_ = !lean_is_exclusive(v___x_3273_);
if (v_isSharedCheck_3284_ == 0)
{
v___x_3276_ = v___x_3273_;
v_isShared_3277_ = v_isSharedCheck_3284_;
goto v_resetjp_3275_;
}
else
{
lean_inc(v_a_3274_);
lean_dec(v___x_3273_);
v___x_3276_ = lean_box(0);
v_isShared_3277_ = v_isSharedCheck_3284_;
goto v_resetjp_3275_;
}
v_resetjp_3275_:
{
lean_object* v___x_3279_; 
if (v_isShared_3272_ == 0)
{
lean_ctor_set(v___x_3271_, 0, v_a_3274_);
v___x_3279_ = v___x_3271_;
goto v_reusejp_3278_;
}
else
{
lean_object* v_reuseFailAlloc_3283_; 
v_reuseFailAlloc_3283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3283_, 0, v_a_3274_);
v___x_3279_ = v_reuseFailAlloc_3283_;
goto v_reusejp_3278_;
}
v_reusejp_3278_:
{
lean_object* v___x_3281_; 
if (v_isShared_3277_ == 0)
{
lean_ctor_set(v___x_3276_, 0, v___x_3279_);
v___x_3281_ = v___x_3276_;
goto v_reusejp_3280_;
}
else
{
lean_object* v_reuseFailAlloc_3282_; 
v_reuseFailAlloc_3282_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3282_, 0, v___x_3279_);
v___x_3281_ = v_reuseFailAlloc_3282_;
goto v_reusejp_3280_;
}
v_reusejp_3280_:
{
return v___x_3281_;
}
}
}
}
}
else
{
lean_object* v___x_3286_; lean_object* v___x_3288_; 
lean_dec(v_a_3265_);
v___x_3286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0));
if (v_isShared_3268_ == 0)
{
lean_ctor_set(v___x_3267_, 0, v___x_3286_);
v___x_3288_ = v___x_3267_;
goto v_reusejp_3287_;
}
else
{
lean_object* v_reuseFailAlloc_3289_; 
v_reuseFailAlloc_3289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3289_, 0, v___x_3286_);
v___x_3288_ = v_reuseFailAlloc_3289_;
goto v_reusejp_3287_;
}
v_reusejp_3287_:
{
return v___x_3288_;
}
}
}
}
else
{
lean_object* v_a_3291_; lean_object* v___x_3293_; uint8_t v_isShared_3294_; uint8_t v_isSharedCheck_3298_; 
v_a_3291_ = lean_ctor_get(v___x_3264_, 0);
v_isSharedCheck_3298_ = !lean_is_exclusive(v___x_3264_);
if (v_isSharedCheck_3298_ == 0)
{
v___x_3293_ = v___x_3264_;
v_isShared_3294_ = v_isSharedCheck_3298_;
goto v_resetjp_3292_;
}
else
{
lean_inc(v_a_3291_);
lean_dec(v___x_3264_);
v___x_3293_ = lean_box(0);
v_isShared_3294_ = v_isSharedCheck_3298_;
goto v_resetjp_3292_;
}
v_resetjp_3292_:
{
lean_object* v___x_3296_; 
if (v_isShared_3294_ == 0)
{
v___x_3296_ = v___x_3293_;
goto v_reusejp_3295_;
}
else
{
lean_object* v_reuseFailAlloc_3297_; 
v_reuseFailAlloc_3297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3297_, 0, v_a_3291_);
v___x_3296_ = v_reuseFailAlloc_3297_;
goto v_reusejp_3295_;
}
v_reusejp_3295_:
{
return v___x_3296_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs___lam__1___boxed(lean_object* v_node_3299_, lean_object* v___y_3300_, lean_object* v___y_3301_, lean_object* v___y_3302_, lean_object* v___y_3303_, lean_object* v___y_3304_){
_start:
{
lean_object* v_res_3305_; 
v_res_3305_ = lp_mathlib_Mathlib_Tactic_unfoldProjs___lam__1(v_node_3299_, v___y_3300_, v___y_3301_, v___y_3302_, v___y_3303_);
lean_dec(v___y_3303_);
lean_dec_ref(v___y_3302_);
lean_dec(v___y_3301_);
lean_dec_ref(v___y_3300_);
return v_res_3305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs(lean_object* v_e_3307_, lean_object* v_a_3308_, lean_object* v_a_3309_, lean_object* v_a_3310_, lean_object* v_a_3311_){
_start:
{
lean_object* v___f_3313_; lean_object* v___f_3314_; uint8_t v___x_3315_; lean_object* v___x_3316_; 
v___f_3313_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___closed__0));
v___f_3314_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldProjs___closed__0));
v___x_3315_ = 0;
v___x_3316_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1(v_e_3307_, v___f_3314_, v___f_3313_, v___x_3315_, v___x_3315_, v_a_3308_, v_a_3309_, v_a_3310_, v_a_3311_);
return v___x_3316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_unfoldProjs___boxed(lean_object* v_e_3317_, lean_object* v_a_3318_, lean_object* v_a_3319_, lean_object* v_a_3320_, lean_object* v_a_3321_, lean_object* v_a_3322_){
_start:
{
lean_object* v_res_3323_; 
v_res_3323_ = lp_mathlib_Mathlib_Tactic_unfoldProjs(v_e_3317_, v_a_3318_, v_a_3319_, v_a_3320_, v_a_3321_);
lean_dec(v_a_3321_);
lean_dec_ref(v_a_3320_);
lean_dec(v_a_3319_);
lean_dec_ref(v_a_3318_);
return v_res_3323_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__4(void){
_start:
{
lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___x_3336_; 
v___x_3333_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_3334_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__3));
v___x_3335_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_3336_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3336_, 0, v___x_3335_);
lean_ctor_set(v___x_3336_, 1, v___x_3334_);
lean_ctor_set(v___x_3336_, 2, v___x_3333_);
return v___x_3336_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__5(void){
_start:
{
lean_object* v___x_3337_; lean_object* v___x_3338_; lean_object* v___x_3339_; lean_object* v___x_3340_; 
v___x_3337_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__4, &lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__4);
v___x_3338_ = lean_unsigned_to_nat(1022u);
v___x_3339_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1));
v___x_3340_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3340_, 0, v___x_3339_);
lean_ctor_set(v___x_3340_, 1, v___x_3338_);
lean_ctor_set(v___x_3340_, 2, v___x_3337_);
return v___x_3340_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_unfoldProjsStx(void){
_start:
{
lean_object* v___x_3341_; 
v___x_3341_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__5, &lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__5);
return v___x_3341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___lam__0(lean_object* v_x_3342_, lean_object* v___y_3343_, lean_object* v___y_3344_, lean_object* v___y_3345_, lean_object* v___y_3346_, lean_object* v___y_3347_){
_start:
{
lean_object* v___x_3349_; 
v___x_3349_ = lp_mathlib_Mathlib_Tactic_unfoldProjs(v___y_3343_, v___y_3344_, v___y_3345_, v___y_3346_, v___y_3347_);
return v___x_3349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___lam__0___boxed(lean_object* v_x_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_, lean_object* v___y_3355_, lean_object* v___y_3356_){
_start:
{
lean_object* v_res_3357_; 
v_res_3357_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___lam__0(v_x_3350_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_, v___y_3355_);
lean_dec(v___y_3355_);
lean_dec_ref(v___y_3354_);
lean_dec(v___y_3353_);
lean_dec_ref(v___y_3352_);
lean_dec(v_x_3350_);
return v_res_3357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1(lean_object* v_x_3359_, lean_object* v_a_3360_, lean_object* v_a_3361_, lean_object* v_a_3362_, lean_object* v_a_3363_, lean_object* v_a_3364_, lean_object* v_a_3365_, lean_object* v_a_3366_, lean_object* v_a_3367_){
_start:
{
lean_object* v___x_3369_; uint8_t v___x_3370_; 
v___x_3369_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__1));
lean_inc(v_x_3359_);
v___x_3370_ = l_Lean_Syntax_isOfKind(v_x_3359_, v___x_3369_);
if (v___x_3370_ == 0)
{
lean_object* v___x_3371_; 
lean_dec(v_x_3359_);
v___x_3371_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_3371_;
}
else
{
lean_object* v___f_3372_; lean_object* v___y_3374_; lean_object* v___x_3377_; lean_object* v___x_3378_; lean_object* v___x_3379_; 
v___f_3372_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___closed__0));
v___x_3377_ = lean_unsigned_to_nat(1u);
v___x_3378_ = l_Lean_Syntax_getArg(v_x_3359_, v___x_3377_);
lean_dec(v_x_3359_);
v___x_3379_ = l_Lean_Syntax_getOptional_x3f(v___x_3378_);
lean_dec(v___x_3378_);
if (lean_obj_tag(v___x_3379_) == 0)
{
lean_object* v___x_3380_; 
v___x_3380_ = lean_box(0);
v___y_3374_ = v___x_3380_;
goto v___jp_3373_;
}
else
{
lean_object* v_val_3381_; lean_object* v___x_3383_; uint8_t v_isShared_3384_; uint8_t v_isSharedCheck_3388_; 
v_val_3381_ = lean_ctor_get(v___x_3379_, 0);
v_isSharedCheck_3388_ = !lean_is_exclusive(v___x_3379_);
if (v_isSharedCheck_3388_ == 0)
{
v___x_3383_ = v___x_3379_;
v_isShared_3384_ = v_isSharedCheck_3388_;
goto v_resetjp_3382_;
}
else
{
lean_inc(v_val_3381_);
lean_dec(v___x_3379_);
v___x_3383_ = lean_box(0);
v_isShared_3384_ = v_isSharedCheck_3388_;
goto v_resetjp_3382_;
}
v_resetjp_3382_:
{
lean_object* v___x_3386_; 
if (v_isShared_3384_ == 0)
{
v___x_3386_ = v___x_3383_;
goto v_reusejp_3385_;
}
else
{
lean_object* v_reuseFailAlloc_3387_; 
v_reuseFailAlloc_3387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3387_, 0, v_val_3381_);
v___x_3386_ = v_reuseFailAlloc_3387_;
goto v_reusejp_3385_;
}
v_reusejp_3385_:
{
v___y_3374_ = v___x_3386_;
goto v___jp_3373_;
}
}
}
v___jp_3373_:
{
lean_object* v___x_3375_; lean_object* v___x_3376_; 
v___x_3375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldProjsStx___closed__2));
v___x_3376_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___f_3372_, v___y_3374_, v___x_3375_, v___x_3370_, v_a_3360_, v_a_3361_, v_a_3362_, v_a_3363_, v_a_3364_, v_a_3365_, v_a_3366_, v_a_3367_);
return v___x_3376_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1___boxed(lean_object* v_x_3389_, lean_object* v_a_3390_, lean_object* v_a_3391_, lean_object* v_a_3392_, lean_object* v_a_3393_, lean_object* v_a_3394_, lean_object* v_a_3395_, lean_object* v_a_3396_, lean_object* v_a_3397_, lean_object* v_a_3398_){
_start:
{
lean_object* v_res_3399_; 
v_res_3399_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__unfoldProjsStx__1(v_x_3389_, v_a_3390_, v_a_3391_, v_a_3392_, v_a_3393_, v_a_3394_, v_a_3395_, v_a_3396_, v_a_3397_);
lean_dec(v_a_3397_);
lean_dec_ref(v_a_3396_);
lean_dec(v_a_3395_);
lean_dec_ref(v_a_3394_);
lean_dec(v_a_3393_);
lean_dec_ref(v_a_3392_);
lean_dec(v_a_3391_);
lean_dec_ref(v_a_3390_);
return v_res_3399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1(lean_object* v_x_3411_, lean_object* v_a_3412_, lean_object* v_a_3413_, lean_object* v_a_3414_, lean_object* v_a_3415_, lean_object* v_a_3416_, lean_object* v_a_3417_, lean_object* v_a_3418_, lean_object* v_a_3419_){
_start:
{
lean_object* v___x_3421_; uint8_t v___x_3422_; 
v___x_3421_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convUnfold__projs___closed__1));
v___x_3422_ = l_Lean_Syntax_isOfKind(v_x_3411_, v___x_3421_);
if (v___x_3422_ == 0)
{
lean_object* v___x_3423_; 
v___x_3423_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_3423_;
}
else
{
lean_object* v___x_3424_; lean_object* v___x_3425_; 
v___x_3424_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1___closed__0));
v___x_3425_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(v___x_3424_, v_a_3412_, v_a_3413_, v_a_3414_, v_a_3415_, v_a_3416_, v_a_3417_, v_a_3418_, v_a_3419_);
return v___x_3425_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1___boxed(lean_object* v_x_3426_, lean_object* v_a_3427_, lean_object* v_a_3428_, lean_object* v_a_3429_, lean_object* v_a_3430_, lean_object* v_a_3431_, lean_object* v_a_3432_, lean_object* v_a_3433_, lean_object* v_a_3434_, lean_object* v_a_3435_){
_start:
{
lean_object* v_res_3436_; 
v_res_3436_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convUnfold__projs__1(v_x_3426_, v_a_3427_, v_a_3428_, v_a_3429_, v_a_3430_, v_a_3431_, v_a_3432_, v_a_3433_, v_a_3434_);
lean_dec(v_a_3434_);
lean_dec_ref(v_a_3433_);
lean_dec(v_a_3432_);
lean_dec_ref(v_a_3431_);
lean_dec(v_a_3430_);
lean_dec_ref(v_a_3429_);
lean_dec(v_a_3428_);
lean_dec_ref(v_a_3427_);
return v_res_3436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll___lam__0(lean_object* v_node_3437_, lean_object* v___y_3438_, lean_object* v___y_3439_, lean_object* v___y_3440_, lean_object* v___y_3441_){
_start:
{
lean_object* v___x_3443_; 
v___x_3443_ = l_Lean_Expr_etaExpandedStrict_x3f(v_node_3437_);
if (lean_obj_tag(v___x_3443_) == 0)
{
lean_object* v___x_3444_; lean_object* v___x_3445_; 
v___x_3444_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_3444_, 0, v___x_3443_);
v___x_3445_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3445_, 0, v___x_3444_);
return v___x_3445_;
}
else
{
lean_object* v_val_3446_; lean_object* v___x_3448_; uint8_t v_isShared_3449_; uint8_t v_isSharedCheck_3454_; 
v_val_3446_ = lean_ctor_get(v___x_3443_, 0);
v_isSharedCheck_3454_ = !lean_is_exclusive(v___x_3443_);
if (v_isSharedCheck_3454_ == 0)
{
v___x_3448_ = v___x_3443_;
v_isShared_3449_ = v_isSharedCheck_3454_;
goto v_resetjp_3447_;
}
else
{
lean_inc(v_val_3446_);
lean_dec(v___x_3443_);
v___x_3448_ = lean_box(0);
v_isShared_3449_ = v_isSharedCheck_3454_;
goto v_resetjp_3447_;
}
v_resetjp_3447_:
{
lean_object* v___x_3451_; 
if (v_isShared_3449_ == 0)
{
v___x_3451_ = v___x_3448_;
goto v_reusejp_3450_;
}
else
{
lean_object* v_reuseFailAlloc_3453_; 
v_reuseFailAlloc_3453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3453_, 0, v_val_3446_);
v___x_3451_ = v_reuseFailAlloc_3453_;
goto v_reusejp_3450_;
}
v_reusejp_3450_:
{
lean_object* v___x_3452_; 
v___x_3452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3452_, 0, v___x_3451_);
return v___x_3452_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll___lam__0___boxed(lean_object* v_node_3455_, lean_object* v___y_3456_, lean_object* v___y_3457_, lean_object* v___y_3458_, lean_object* v___y_3459_, lean_object* v___y_3460_){
_start:
{
lean_object* v_res_3461_; 
v_res_3461_ = lp_mathlib_Mathlib_Tactic_etaReduceAll___lam__0(v_node_3455_, v___y_3456_, v___y_3457_, v___y_3458_, v___y_3459_);
lean_dec(v___y_3459_);
lean_dec_ref(v___y_3458_);
lean_dec(v___y_3457_);
lean_dec_ref(v___y_3456_);
return v_res_3461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll(lean_object* v_e_3463_, lean_object* v_a_3464_, lean_object* v_a_3465_, lean_object* v_a_3466_, lean_object* v_a_3467_){
_start:
{
lean_object* v___f_3469_; lean_object* v___f_3470_; uint8_t v___x_3471_; lean_object* v___x_3472_; 
v___f_3469_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaReduceAll___closed__0));
v___f_3470_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___closed__0));
v___x_3471_ = 0;
v___x_3472_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1(v_e_3463_, v___f_3469_, v___f_3470_, v___x_3471_, v___x_3471_, v_a_3464_, v_a_3465_, v_a_3466_, v_a_3467_);
return v___x_3472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaReduceAll___boxed(lean_object* v_e_3473_, lean_object* v_a_3474_, lean_object* v_a_3475_, lean_object* v_a_3476_, lean_object* v_a_3477_, lean_object* v_a_3478_){
_start:
{
lean_object* v_res_3479_; 
v_res_3479_ = lp_mathlib_Mathlib_Tactic_etaReduceAll(v_e_3473_, v_a_3474_, v_a_3475_, v_a_3476_, v_a_3477_);
lean_dec(v_a_3477_);
lean_dec_ref(v_a_3476_);
lean_dec(v_a_3475_);
lean_dec_ref(v_a_3474_);
return v_res_3479_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__4(void){
_start:
{
lean_object* v___x_3489_; lean_object* v___x_3490_; lean_object* v___x_3491_; lean_object* v___x_3492_; 
v___x_3489_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_3490_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__3));
v___x_3491_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_3492_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3492_, 0, v___x_3491_);
lean_ctor_set(v___x_3492_, 1, v___x_3490_);
lean_ctor_set(v___x_3492_, 2, v___x_3489_);
return v___x_3492_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__5(void){
_start:
{
lean_object* v___x_3493_; lean_object* v___x_3494_; lean_object* v___x_3495_; lean_object* v___x_3496_; 
v___x_3493_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__4, &lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__4);
v___x_3494_ = lean_unsigned_to_nat(1022u);
v___x_3495_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1));
v___x_3496_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3496_, 0, v___x_3495_);
lean_ctor_set(v___x_3496_, 1, v___x_3494_);
lean_ctor_set(v___x_3496_, 2, v___x_3493_);
return v___x_3496_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaReduceStx(void){
_start:
{
lean_object* v___x_3497_; 
v___x_3497_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__5, &lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__5);
return v___x_3497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___lam__0(lean_object* v_x_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_, lean_object* v___y_3502_, lean_object* v___y_3503_){
_start:
{
lean_object* v___x_3505_; 
v___x_3505_ = lp_mathlib_Mathlib_Tactic_etaReduceAll(v___y_3499_, v___y_3500_, v___y_3501_, v___y_3502_, v___y_3503_);
return v___x_3505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___lam__0___boxed(lean_object* v_x_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_, lean_object* v___y_3509_, lean_object* v___y_3510_, lean_object* v___y_3511_, lean_object* v___y_3512_){
_start:
{
lean_object* v_res_3513_; 
v_res_3513_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___lam__0(v_x_3506_, v___y_3507_, v___y_3508_, v___y_3509_, v___y_3510_, v___y_3511_);
lean_dec(v___y_3511_);
lean_dec_ref(v___y_3510_);
lean_dec(v___y_3509_);
lean_dec_ref(v___y_3508_);
lean_dec(v_x_3506_);
return v_res_3513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1(lean_object* v_x_3515_, lean_object* v_a_3516_, lean_object* v_a_3517_, lean_object* v_a_3518_, lean_object* v_a_3519_, lean_object* v_a_3520_, lean_object* v_a_3521_, lean_object* v_a_3522_, lean_object* v_a_3523_){
_start:
{
lean_object* v___x_3525_; uint8_t v___x_3526_; 
v___x_3525_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__1));
lean_inc(v_x_3515_);
v___x_3526_ = l_Lean_Syntax_isOfKind(v_x_3515_, v___x_3525_);
if (v___x_3526_ == 0)
{
lean_object* v___x_3527_; 
lean_dec(v_x_3515_);
v___x_3527_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_3527_;
}
else
{
lean_object* v___f_3528_; lean_object* v___y_3530_; lean_object* v___x_3533_; lean_object* v___x_3534_; lean_object* v___x_3535_; 
v___f_3528_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___closed__0));
v___x_3533_ = lean_unsigned_to_nat(1u);
v___x_3534_ = l_Lean_Syntax_getArg(v_x_3515_, v___x_3533_);
lean_dec(v_x_3515_);
v___x_3535_ = l_Lean_Syntax_getOptional_x3f(v___x_3534_);
lean_dec(v___x_3534_);
if (lean_obj_tag(v___x_3535_) == 0)
{
lean_object* v___x_3536_; 
v___x_3536_ = lean_box(0);
v___y_3530_ = v___x_3536_;
goto v___jp_3529_;
}
else
{
lean_object* v_val_3537_; lean_object* v___x_3539_; uint8_t v_isShared_3540_; uint8_t v_isSharedCheck_3544_; 
v_val_3537_ = lean_ctor_get(v___x_3535_, 0);
v_isSharedCheck_3544_ = !lean_is_exclusive(v___x_3535_);
if (v_isSharedCheck_3544_ == 0)
{
v___x_3539_ = v___x_3535_;
v_isShared_3540_ = v_isSharedCheck_3544_;
goto v_resetjp_3538_;
}
else
{
lean_inc(v_val_3537_);
lean_dec(v___x_3535_);
v___x_3539_ = lean_box(0);
v_isShared_3540_ = v_isSharedCheck_3544_;
goto v_resetjp_3538_;
}
v_resetjp_3538_:
{
lean_object* v___x_3542_; 
if (v_isShared_3540_ == 0)
{
v___x_3542_ = v___x_3539_;
goto v_reusejp_3541_;
}
else
{
lean_object* v_reuseFailAlloc_3543_; 
v_reuseFailAlloc_3543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3543_, 0, v_val_3537_);
v___x_3542_ = v_reuseFailAlloc_3543_;
goto v_reusejp_3541_;
}
v_reusejp_3541_:
{
v___y_3530_ = v___x_3542_;
goto v___jp_3529_;
}
}
}
v___jp_3529_:
{
lean_object* v___x_3531_; lean_object* v___x_3532_; 
v___x_3531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaReduceStx___closed__2));
v___x_3532_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___f_3528_, v___y_3530_, v___x_3531_, v___x_3526_, v_a_3516_, v_a_3517_, v_a_3518_, v_a_3519_, v_a_3520_, v_a_3521_, v_a_3522_, v_a_3523_);
return v___x_3532_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1___boxed(lean_object* v_x_3545_, lean_object* v_a_3546_, lean_object* v_a_3547_, lean_object* v_a_3548_, lean_object* v_a_3549_, lean_object* v_a_3550_, lean_object* v_a_3551_, lean_object* v_a_3552_, lean_object* v_a_3553_, lean_object* v_a_3554_){
_start:
{
lean_object* v_res_3555_; 
v_res_3555_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaReduceStx__1(v_x_3545_, v_a_3546_, v_a_3547_, v_a_3548_, v_a_3549_, v_a_3550_, v_a_3551_, v_a_3552_, v_a_3553_);
lean_dec(v_a_3553_);
lean_dec_ref(v_a_3552_);
lean_dec(v_a_3551_);
lean_dec_ref(v_a_3550_);
lean_dec(v_a_3549_);
lean_dec_ref(v_a_3548_);
lean_dec(v_a_3547_);
lean_dec_ref(v_a_3546_);
return v_res_3555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1(lean_object* v_x_3567_, lean_object* v_a_3568_, lean_object* v_a_3569_, lean_object* v_a_3570_, lean_object* v_a_3571_, lean_object* v_a_3572_, lean_object* v_a_3573_, lean_object* v_a_3574_, lean_object* v_a_3575_){
_start:
{
lean_object* v___x_3577_; uint8_t v___x_3578_; 
v___x_3577_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convEta__reduce___closed__1));
v___x_3578_ = l_Lean_Syntax_isOfKind(v_x_3567_, v___x_3577_);
if (v___x_3578_ == 0)
{
lean_object* v___x_3579_; 
v___x_3579_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_3579_;
}
else
{
lean_object* v___x_3580_; lean_object* v___x_3581_; 
v___x_3580_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1___closed__0));
v___x_3581_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(v___x_3580_, v_a_3568_, v_a_3569_, v_a_3570_, v_a_3571_, v_a_3572_, v_a_3573_, v_a_3574_, v_a_3575_);
return v___x_3581_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1___boxed(lean_object* v_x_3582_, lean_object* v_a_3583_, lean_object* v_a_3584_, lean_object* v_a_3585_, lean_object* v_a_3586_, lean_object* v_a_3587_, lean_object* v_a_3588_, lean_object* v_a_3589_, lean_object* v_a_3590_, lean_object* v_a_3591_){
_start:
{
lean_object* v_res_3592_; 
v_res_3592_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__reduce__1(v_x_3582_, v_a_3583_, v_a_3584_, v_a_3585_, v_a_3586_, v_a_3587_, v_a_3588_, v_a_3589_, v_a_3590_);
lean_dec(v_a_3590_);
lean_dec_ref(v_a_3589_);
lean_dec(v_a_3588_);
lean_dec_ref(v_a_3587_);
lean_dec(v_a_3586_);
lean_dec_ref(v_a_3585_);
lean_dec(v_a_3584_);
lean_dec_ref(v_a_3583_);
return v_res_3592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___lam__0(lean_object* v_k_3593_, lean_object* v_b_3594_, lean_object* v___y_3595_, lean_object* v___y_3596_, lean_object* v___y_3597_, lean_object* v___y_3598_){
_start:
{
lean_object* v___x_3600_; 
lean_inc(v___y_3598_);
lean_inc_ref(v___y_3597_);
lean_inc(v___y_3596_);
lean_inc_ref(v___y_3595_);
v___x_3600_ = lean_apply_6(v_k_3593_, v_b_3594_, v___y_3595_, v___y_3596_, v___y_3597_, v___y_3598_, lean_box(0));
return v___x_3600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___lam__0___boxed(lean_object* v_k_3601_, lean_object* v_b_3602_, lean_object* v___y_3603_, lean_object* v___y_3604_, lean_object* v___y_3605_, lean_object* v___y_3606_, lean_object* v___y_3607_){
_start:
{
lean_object* v_res_3608_; 
v_res_3608_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___lam__0(v_k_3601_, v_b_3602_, v___y_3603_, v___y_3604_, v___y_3605_, v___y_3606_);
lean_dec(v___y_3606_);
lean_dec_ref(v___y_3605_);
lean_dec(v___y_3604_);
lean_dec_ref(v___y_3603_);
return v_res_3608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg(lean_object* v_name_3609_, uint8_t v_bi_3610_, lean_object* v_type_3611_, lean_object* v_k_3612_, uint8_t v_kind_3613_, lean_object* v___y_3614_, lean_object* v___y_3615_, lean_object* v___y_3616_, lean_object* v___y_3617_){
_start:
{
lean_object* v___f_3619_; lean_object* v___x_3620_; 
v___f_3619_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3619_, 0, v_k_3612_);
v___x_3620_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_3609_, v_bi_3610_, v_type_3611_, v___f_3619_, v_kind_3613_, v___y_3614_, v___y_3615_, v___y_3616_, v___y_3617_);
if (lean_obj_tag(v___x_3620_) == 0)
{
lean_object* v_a_3621_; lean_object* v___x_3623_; uint8_t v_isShared_3624_; uint8_t v_isSharedCheck_3628_; 
v_a_3621_ = lean_ctor_get(v___x_3620_, 0);
v_isSharedCheck_3628_ = !lean_is_exclusive(v___x_3620_);
if (v_isSharedCheck_3628_ == 0)
{
v___x_3623_ = v___x_3620_;
v_isShared_3624_ = v_isSharedCheck_3628_;
goto v_resetjp_3622_;
}
else
{
lean_inc(v_a_3621_);
lean_dec(v___x_3620_);
v___x_3623_ = lean_box(0);
v_isShared_3624_ = v_isSharedCheck_3628_;
goto v_resetjp_3622_;
}
v_resetjp_3622_:
{
lean_object* v___x_3626_; 
if (v_isShared_3624_ == 0)
{
v___x_3626_ = v___x_3623_;
goto v_reusejp_3625_;
}
else
{
lean_object* v_reuseFailAlloc_3627_; 
v_reuseFailAlloc_3627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3627_, 0, v_a_3621_);
v___x_3626_ = v_reuseFailAlloc_3627_;
goto v_reusejp_3625_;
}
v_reusejp_3625_:
{
return v___x_3626_;
}
}
}
else
{
lean_object* v_a_3629_; lean_object* v___x_3631_; uint8_t v_isShared_3632_; uint8_t v_isSharedCheck_3636_; 
v_a_3629_ = lean_ctor_get(v___x_3620_, 0);
v_isSharedCheck_3636_ = !lean_is_exclusive(v___x_3620_);
if (v_isSharedCheck_3636_ == 0)
{
v___x_3631_ = v___x_3620_;
v_isShared_3632_ = v_isSharedCheck_3636_;
goto v_resetjp_3630_;
}
else
{
lean_inc(v_a_3629_);
lean_dec(v___x_3620_);
v___x_3631_ = lean_box(0);
v_isShared_3632_ = v_isSharedCheck_3636_;
goto v_resetjp_3630_;
}
v_resetjp_3630_:
{
lean_object* v___x_3634_; 
if (v_isShared_3632_ == 0)
{
v___x_3634_ = v___x_3631_;
goto v_reusejp_3633_;
}
else
{
lean_object* v_reuseFailAlloc_3635_; 
v_reuseFailAlloc_3635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3635_, 0, v_a_3629_);
v___x_3634_ = v_reuseFailAlloc_3635_;
goto v_reusejp_3633_;
}
v_reusejp_3633_:
{
return v___x_3634_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___boxed(lean_object* v_name_3637_, lean_object* v_bi_3638_, lean_object* v_type_3639_, lean_object* v_k_3640_, lean_object* v_kind_3641_, lean_object* v___y_3642_, lean_object* v___y_3643_, lean_object* v___y_3644_, lean_object* v___y_3645_, lean_object* v___y_3646_){
_start:
{
uint8_t v_bi_boxed_3647_; uint8_t v_kind_boxed_3648_; lean_object* v_res_3649_; 
v_bi_boxed_3647_ = lean_unbox(v_bi_3638_);
v_kind_boxed_3648_ = lean_unbox(v_kind_3641_);
v_res_3649_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg(v_name_3637_, v_bi_boxed_3647_, v_type_3639_, v_k_3640_, v_kind_boxed_3648_, v___y_3642_, v___y_3643_, v___y_3644_, v___y_3645_);
lean_dec(v___y_3645_);
lean_dec_ref(v___y_3644_);
lean_dec(v___y_3643_);
lean_dec_ref(v___y_3642_);
return v_res_3649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0(lean_object* v_00_u03b1_3650_, lean_object* v_name_3651_, uint8_t v_bi_3652_, lean_object* v_type_3653_, lean_object* v_k_3654_, uint8_t v_kind_3655_, lean_object* v___y_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_){
_start:
{
lean_object* v___x_3661_; 
v___x_3661_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg(v_name_3651_, v_bi_3652_, v_type_3653_, v_k_3654_, v_kind_3655_, v___y_3656_, v___y_3657_, v___y_3658_, v___y_3659_);
return v___x_3661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___boxed(lean_object* v_00_u03b1_3662_, lean_object* v_name_3663_, lean_object* v_bi_3664_, lean_object* v_type_3665_, lean_object* v_k_3666_, lean_object* v_kind_3667_, lean_object* v___y_3668_, lean_object* v___y_3669_, lean_object* v___y_3670_, lean_object* v___y_3671_, lean_object* v___y_3672_){
_start:
{
uint8_t v_bi_boxed_3673_; uint8_t v_kind_boxed_3674_; lean_object* v_res_3675_; 
v_bi_boxed_3673_ = lean_unbox(v_bi_3664_);
v_kind_boxed_3674_ = lean_unbox(v_kind_3667_);
v_res_3675_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0(v_00_u03b1_3662_, v_name_3663_, v_bi_boxed_3673_, v_type_3665_, v_k_3666_, v_kind_boxed_3674_, v___y_3668_, v___y_3669_, v___y_3670_, v___y_3671_);
lean_dec(v___y_3671_);
lean_dec_ref(v___y_3670_);
lean_dec(v___y_3669_);
lean_dec_ref(v___y_3668_);
return v_res_3675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___redArg(lean_object* v_name_3676_, lean_object* v_type_3677_, lean_object* v_val_3678_, lean_object* v_k_3679_, uint8_t v_nondep_3680_, uint8_t v_kind_3681_, lean_object* v___y_3682_, lean_object* v___y_3683_, lean_object* v___y_3684_, lean_object* v___y_3685_){
_start:
{
lean_object* v___f_3687_; lean_object* v___x_3688_; 
v___f_3687_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3687_, 0, v_k_3679_);
v___x_3688_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_3676_, v_type_3677_, v_val_3678_, v___f_3687_, v_nondep_3680_, v_kind_3681_, v___y_3682_, v___y_3683_, v___y_3684_, v___y_3685_);
if (lean_obj_tag(v___x_3688_) == 0)
{
lean_object* v_a_3689_; lean_object* v___x_3691_; uint8_t v_isShared_3692_; uint8_t v_isSharedCheck_3696_; 
v_a_3689_ = lean_ctor_get(v___x_3688_, 0);
v_isSharedCheck_3696_ = !lean_is_exclusive(v___x_3688_);
if (v_isSharedCheck_3696_ == 0)
{
v___x_3691_ = v___x_3688_;
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
else
{
lean_inc(v_a_3689_);
lean_dec(v___x_3688_);
v___x_3691_ = lean_box(0);
v_isShared_3692_ = v_isSharedCheck_3696_;
goto v_resetjp_3690_;
}
v_resetjp_3690_:
{
lean_object* v___x_3694_; 
if (v_isShared_3692_ == 0)
{
v___x_3694_ = v___x_3691_;
goto v_reusejp_3693_;
}
else
{
lean_object* v_reuseFailAlloc_3695_; 
v_reuseFailAlloc_3695_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3695_, 0, v_a_3689_);
v___x_3694_ = v_reuseFailAlloc_3695_;
goto v_reusejp_3693_;
}
v_reusejp_3693_:
{
return v___x_3694_;
}
}
}
else
{
lean_object* v_a_3697_; lean_object* v___x_3699_; uint8_t v_isShared_3700_; uint8_t v_isSharedCheck_3704_; 
v_a_3697_ = lean_ctor_get(v___x_3688_, 0);
v_isSharedCheck_3704_ = !lean_is_exclusive(v___x_3688_);
if (v_isSharedCheck_3704_ == 0)
{
v___x_3699_ = v___x_3688_;
v_isShared_3700_ = v_isSharedCheck_3704_;
goto v_resetjp_3698_;
}
else
{
lean_inc(v_a_3697_);
lean_dec(v___x_3688_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___redArg___boxed(lean_object* v_name_3705_, lean_object* v_type_3706_, lean_object* v_val_3707_, lean_object* v_k_3708_, lean_object* v_nondep_3709_, lean_object* v_kind_3710_, lean_object* v___y_3711_, lean_object* v___y_3712_, lean_object* v___y_3713_, lean_object* v___y_3714_, lean_object* v___y_3715_){
_start:
{
uint8_t v_nondep_boxed_3716_; uint8_t v_kind_boxed_3717_; lean_object* v_res_3718_; 
v_nondep_boxed_3716_ = lean_unbox(v_nondep_3709_);
v_kind_boxed_3717_ = lean_unbox(v_kind_3710_);
v_res_3718_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___redArg(v_name_3705_, v_type_3706_, v_val_3707_, v_k_3708_, v_nondep_boxed_3716_, v_kind_boxed_3717_, v___y_3711_, v___y_3712_, v___y_3713_, v___y_3714_);
lean_dec(v___y_3714_);
lean_dec_ref(v___y_3713_);
lean_dec(v___y_3712_);
lean_dec_ref(v___y_3711_);
return v_res_3718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1(lean_object* v_00_u03b1_3719_, lean_object* v_name_3720_, lean_object* v_type_3721_, lean_object* v_val_3722_, lean_object* v_k_3723_, uint8_t v_nondep_3724_, uint8_t v_kind_3725_, lean_object* v___y_3726_, lean_object* v___y_3727_, lean_object* v___y_3728_, lean_object* v___y_3729_){
_start:
{
lean_object* v___x_3731_; 
v___x_3731_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___redArg(v_name_3720_, v_type_3721_, v_val_3722_, v_k_3723_, v_nondep_3724_, v_kind_3725_, v___y_3726_, v___y_3727_, v___y_3728_, v___y_3729_);
return v___x_3731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___boxed(lean_object* v_00_u03b1_3732_, lean_object* v_name_3733_, lean_object* v_type_3734_, lean_object* v_val_3735_, lean_object* v_k_3736_, lean_object* v_nondep_3737_, lean_object* v_kind_3738_, lean_object* v___y_3739_, lean_object* v___y_3740_, lean_object* v___y_3741_, lean_object* v___y_3742_, lean_object* v___y_3743_){
_start:
{
uint8_t v_nondep_boxed_3744_; uint8_t v_kind_boxed_3745_; lean_object* v_res_3746_; 
v_nondep_boxed_3744_ = lean_unbox(v_nondep_3737_);
v_kind_boxed_3745_ = lean_unbox(v_kind_3738_);
v_res_3746_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1(v_00_u03b1_3732_, v_name_3733_, v_type_3734_, v_val_3735_, v_k_3736_, v_nondep_boxed_3744_, v_kind_boxed_3745_, v___y_3739_, v___y_3740_, v___y_3741_, v___y_3742_);
lean_dec(v___y_3742_);
lean_dec_ref(v___y_3741_);
lean_dec(v___y_3740_);
lean_dec_ref(v___y_3739_);
return v_res_3746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___lam__0(lean_object* v_k_3747_, lean_object* v_b_3748_, lean_object* v_c_3749_, lean_object* v___y_3750_, lean_object* v___y_3751_, lean_object* v___y_3752_, lean_object* v___y_3753_){
_start:
{
lean_object* v___x_3755_; 
lean_inc(v___y_3753_);
lean_inc_ref(v___y_3752_);
lean_inc(v___y_3751_);
lean_inc_ref(v___y_3750_);
v___x_3755_ = lean_apply_7(v_k_3747_, v_b_3748_, v_c_3749_, v___y_3750_, v___y_3751_, v___y_3752_, v___y_3753_, lean_box(0));
return v___x_3755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___lam__0___boxed(lean_object* v_k_3756_, lean_object* v_b_3757_, lean_object* v_c_3758_, lean_object* v___y_3759_, lean_object* v___y_3760_, lean_object* v___y_3761_, lean_object* v___y_3762_, lean_object* v___y_3763_){
_start:
{
lean_object* v_res_3764_; 
v_res_3764_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___lam__0(v_k_3756_, v_b_3757_, v_c_3758_, v___y_3759_, v___y_3760_, v___y_3761_, v___y_3762_);
lean_dec(v___y_3762_);
lean_dec_ref(v___y_3761_);
lean_dec(v___y_3760_);
lean_dec_ref(v___y_3759_);
return v_res_3764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg(lean_object* v_type_3765_, lean_object* v_k_3766_, uint8_t v_cleanupAnnotations_3767_, uint8_t v_whnfType_3768_, lean_object* v___y_3769_, lean_object* v___y_3770_, lean_object* v___y_3771_, lean_object* v___y_3772_){
_start:
{
lean_object* v___f_3774_; lean_object* v___x_3775_; 
v___f_3774_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_3774_, 0, v_k_3766_);
v___x_3775_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_3765_, v___f_3774_, v_cleanupAnnotations_3767_, v_whnfType_3768_, v___y_3769_, v___y_3770_, v___y_3771_, v___y_3772_);
if (lean_obj_tag(v___x_3775_) == 0)
{
lean_object* v_a_3776_; lean_object* v___x_3778_; uint8_t v_isShared_3779_; uint8_t v_isSharedCheck_3783_; 
v_a_3776_ = lean_ctor_get(v___x_3775_, 0);
v_isSharedCheck_3783_ = !lean_is_exclusive(v___x_3775_);
if (v_isSharedCheck_3783_ == 0)
{
v___x_3778_ = v___x_3775_;
v_isShared_3779_ = v_isSharedCheck_3783_;
goto v_resetjp_3777_;
}
else
{
lean_inc(v_a_3776_);
lean_dec(v___x_3775_);
v___x_3778_ = lean_box(0);
v_isShared_3779_ = v_isSharedCheck_3783_;
goto v_resetjp_3777_;
}
v_resetjp_3777_:
{
lean_object* v___x_3781_; 
if (v_isShared_3779_ == 0)
{
v___x_3781_ = v___x_3778_;
goto v_reusejp_3780_;
}
else
{
lean_object* v_reuseFailAlloc_3782_; 
v_reuseFailAlloc_3782_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3782_, 0, v_a_3776_);
v___x_3781_ = v_reuseFailAlloc_3782_;
goto v_reusejp_3780_;
}
v_reusejp_3780_:
{
return v___x_3781_;
}
}
}
else
{
lean_object* v_a_3784_; lean_object* v___x_3786_; uint8_t v_isShared_3787_; uint8_t v_isSharedCheck_3791_; 
v_a_3784_ = lean_ctor_get(v___x_3775_, 0);
v_isSharedCheck_3791_ = !lean_is_exclusive(v___x_3775_);
if (v_isSharedCheck_3791_ == 0)
{
v___x_3786_ = v___x_3775_;
v_isShared_3787_ = v_isSharedCheck_3791_;
goto v_resetjp_3785_;
}
else
{
lean_inc(v_a_3784_);
lean_dec(v___x_3775_);
v___x_3786_ = lean_box(0);
v_isShared_3787_ = v_isSharedCheck_3791_;
goto v_resetjp_3785_;
}
v_resetjp_3785_:
{
lean_object* v___x_3789_; 
if (v_isShared_3787_ == 0)
{
v___x_3789_ = v___x_3786_;
goto v_reusejp_3788_;
}
else
{
lean_object* v_reuseFailAlloc_3790_; 
v_reuseFailAlloc_3790_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3790_, 0, v_a_3784_);
v___x_3789_ = v_reuseFailAlloc_3790_;
goto v_reusejp_3788_;
}
v_reusejp_3788_:
{
return v___x_3789_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg___boxed(lean_object* v_type_3792_, lean_object* v_k_3793_, lean_object* v_cleanupAnnotations_3794_, lean_object* v_whnfType_3795_, lean_object* v___y_3796_, lean_object* v___y_3797_, lean_object* v___y_3798_, lean_object* v___y_3799_, lean_object* v___y_3800_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_3801_; uint8_t v_whnfType_boxed_3802_; lean_object* v_res_3803_; 
v_cleanupAnnotations_boxed_3801_ = lean_unbox(v_cleanupAnnotations_3794_);
v_whnfType_boxed_3802_ = lean_unbox(v_whnfType_3795_);
v_res_3803_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg(v_type_3792_, v_k_3793_, v_cleanupAnnotations_boxed_3801_, v_whnfType_boxed_3802_, v___y_3796_, v___y_3797_, v___y_3798_, v___y_3799_);
lean_dec(v___y_3799_);
lean_dec_ref(v___y_3798_);
lean_dec(v___y_3797_);
lean_dec_ref(v___y_3796_);
return v_res_3803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll___lam__0___boxed(lean_object* v_e_3804_, lean_object* v___x_3805_, lean_object* v___x_3806_, lean_object* v_xs_3807_, lean_object* v_x_3808_, lean_object* v___y_3809_, lean_object* v___y_3810_, lean_object* v___y_3811_, lean_object* v___y_3812_, lean_object* v___y_3813_){
_start:
{
uint8_t v___x_4676__boxed_3814_; uint8_t v___x_4677__boxed_3815_; lean_object* v_res_3816_; 
v___x_4676__boxed_3814_ = lean_unbox(v___x_3805_);
v___x_4677__boxed_3815_ = lean_unbox(v___x_3806_);
v_res_3816_ = lp_mathlib_Mathlib_Tactic_etaExpandAll___lam__0(v_e_3804_, v___x_4676__boxed_3814_, v___x_4677__boxed_3815_, v_xs_3807_, v_x_3808_, v___y_3809_, v___y_3810_, v___y_3811_, v___y_3812_);
lean_dec(v___y_3812_);
lean_dec_ref(v___y_3811_);
lean_dec(v___y_3810_);
lean_dec_ref(v___y_3809_);
lean_dec_ref(v_x_3808_);
lean_dec_ref(v_xs_3807_);
lean_dec_ref(v_e_3804_);
return v_res_3816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll(lean_object* v_e_3817_, lean_object* v_a_3818_, lean_object* v_a_3819_, lean_object* v_a_3820_, lean_object* v_a_3821_){
_start:
{
uint8_t v___x_3823_; 
v___x_3823_ = l_Lean_Expr_isLambda(v_e_3817_);
if (v___x_3823_ == 0)
{
lean_object* v___x_3824_; 
lean_inc(v_a_3821_);
lean_inc_ref(v_a_3820_);
lean_inc(v_a_3819_);
lean_inc_ref(v_a_3818_);
lean_inc_ref(v_e_3817_);
v___x_3824_ = lean_infer_type(v_e_3817_, v_a_3818_, v_a_3819_, v_a_3820_, v_a_3821_);
if (lean_obj_tag(v___x_3824_) == 0)
{
lean_object* v_a_3825_; uint8_t v___x_3826_; lean_object* v___x_3827_; lean_object* v___x_3828_; lean_object* v___f_3829_; lean_object* v___x_3830_; 
v_a_3825_ = lean_ctor_get(v___x_3824_, 0);
lean_inc(v_a_3825_);
lean_dec_ref_known(v___x_3824_, 1);
v___x_3826_ = 1;
v___x_3827_ = lean_box(v___x_3823_);
v___x_3828_ = lean_box(v___x_3826_);
v___f_3829_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_etaExpandAll___lam__0___boxed), 10, 3);
lean_closure_set(v___f_3829_, 0, v_e_3817_);
lean_closure_set(v___f_3829_, 1, v___x_3827_);
lean_closure_set(v___f_3829_, 2, v___x_3828_);
v___x_3830_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg(v_a_3825_, v___f_3829_, v___x_3823_, v___x_3823_, v_a_3818_, v_a_3819_, v_a_3820_, v_a_3821_);
return v___x_3830_;
}
else
{
lean_dec_ref(v_e_3817_);
return v___x_3824_;
}
}
else
{
lean_object* v___x_3831_; 
v___x_3831_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms(v_e_3817_, v_a_3818_, v_a_3819_, v_a_3820_, v_a_3821_);
return v___x_3831_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0(lean_object* v_body_3832_, lean_object* v_x_3833_, lean_object* v___y_3834_, lean_object* v___y_3835_, lean_object* v___y_3836_, lean_object* v___y_3837_){
_start:
{
lean_object* v___x_3839_; lean_object* v___x_3840_; 
v___x_3839_ = lean_expr_instantiate1(v_body_3832_, v_x_3833_);
v___x_3840_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v___x_3839_, v___y_3834_, v___y_3835_, v___y_3836_, v___y_3837_);
if (lean_obj_tag(v___x_3840_) == 0)
{
lean_object* v_a_3841_; lean_object* v___x_3843_; uint8_t v_isShared_3844_; uint8_t v_isSharedCheck_3852_; 
v_a_3841_ = lean_ctor_get(v___x_3840_, 0);
v_isSharedCheck_3852_ = !lean_is_exclusive(v___x_3840_);
if (v_isSharedCheck_3852_ == 0)
{
v___x_3843_ = v___x_3840_;
v_isShared_3844_ = v_isSharedCheck_3852_;
goto v_resetjp_3842_;
}
else
{
lean_inc(v_a_3841_);
lean_dec(v___x_3840_);
v___x_3843_ = lean_box(0);
v_isShared_3844_ = v_isSharedCheck_3852_;
goto v_resetjp_3842_;
}
v_resetjp_3842_:
{
lean_object* v___x_3845_; lean_object* v___x_3846_; lean_object* v___x_3847_; lean_object* v___x_3848_; lean_object* v___x_3850_; 
v___x_3845_ = lean_unsigned_to_nat(1u);
v___x_3846_ = lean_mk_empty_array_with_capacity(v___x_3845_);
v___x_3847_ = lean_array_push(v___x_3846_, v_x_3833_);
v___x_3848_ = lean_expr_abstract(v_a_3841_, v___x_3847_);
lean_dec_ref(v___x_3847_);
lean_dec(v_a_3841_);
if (v_isShared_3844_ == 0)
{
lean_ctor_set(v___x_3843_, 0, v___x_3848_);
v___x_3850_ = v___x_3843_;
goto v_reusejp_3849_;
}
else
{
lean_object* v_reuseFailAlloc_3851_; 
v_reuseFailAlloc_3851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3851_, 0, v___x_3848_);
v___x_3850_ = v_reuseFailAlloc_3851_;
goto v_reusejp_3849_;
}
v_reusejp_3849_:
{
return v___x_3850_;
}
}
}
else
{
lean_dec_ref(v_x_3833_);
return v___x_3840_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0___boxed(lean_object* v_body_3853_, lean_object* v_x_3854_, lean_object* v___y_3855_, lean_object* v___y_3856_, lean_object* v___y_3857_, lean_object* v___y_3858_, lean_object* v___y_3859_){
_start:
{
lean_object* v_res_3860_; 
v_res_3860_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0(v_body_3853_, v_x_3854_, v___y_3855_, v___y_3856_, v___y_3857_, v___y_3858_);
lean_dec(v___y_3858_);
lean_dec_ref(v___y_3857_);
lean_dec(v___y_3856_);
lean_dec_ref(v___y_3855_);
lean_dec_ref(v_body_3853_);
return v_res_3860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__2(size_t v_sz_3861_, size_t v_i_3862_, lean_object* v_bs_3863_, lean_object* v___y_3864_, lean_object* v___y_3865_, lean_object* v___y_3866_, lean_object* v___y_3867_){
_start:
{
uint8_t v___x_3869_; 
v___x_3869_ = lean_usize_dec_lt(v_i_3862_, v_sz_3861_);
if (v___x_3869_ == 0)
{
lean_object* v___x_3870_; 
v___x_3870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3870_, 0, v_bs_3863_);
return v___x_3870_;
}
else
{
lean_object* v_v_3871_; lean_object* v___x_3872_; 
v_v_3871_ = lean_array_uget_borrowed(v_bs_3863_, v_i_3862_);
lean_inc(v_v_3871_);
v___x_3872_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_v_3871_, v___y_3864_, v___y_3865_, v___y_3866_, v___y_3867_);
if (lean_obj_tag(v___x_3872_) == 0)
{
lean_object* v_a_3873_; lean_object* v___x_3874_; lean_object* v_bs_x27_3875_; size_t v___x_3876_; size_t v___x_3877_; lean_object* v___x_3878_; 
v_a_3873_ = lean_ctor_get(v___x_3872_, 0);
lean_inc(v_a_3873_);
lean_dec_ref_known(v___x_3872_, 1);
v___x_3874_ = lean_unsigned_to_nat(0u);
v_bs_x27_3875_ = lean_array_uset(v_bs_3863_, v_i_3862_, v___x_3874_);
v___x_3876_ = ((size_t)1ULL);
v___x_3877_ = lean_usize_add(v_i_3862_, v___x_3876_);
v___x_3878_ = lean_array_uset(v_bs_x27_3875_, v_i_3862_, v_a_3873_);
v_i_3862_ = v___x_3877_;
v_bs_3863_ = v___x_3878_;
goto _start;
}
else
{
lean_object* v_a_3880_; lean_object* v___x_3882_; uint8_t v_isShared_3883_; uint8_t v_isSharedCheck_3887_; 
lean_dec_ref(v_bs_3863_);
v_a_3880_ = lean_ctor_get(v___x_3872_, 0);
v_isSharedCheck_3887_ = !lean_is_exclusive(v___x_3872_);
if (v_isSharedCheck_3887_ == 0)
{
v___x_3882_ = v___x_3872_;
v_isShared_3883_ = v_isSharedCheck_3887_;
goto v_resetjp_3881_;
}
else
{
lean_inc(v_a_3880_);
lean_dec(v___x_3872_);
v___x_3882_ = lean_box(0);
v_isShared_3883_ = v_isSharedCheck_3887_;
goto v_resetjp_3881_;
}
v_resetjp_3881_:
{
lean_object* v___x_3885_; 
if (v_isShared_3883_ == 0)
{
v___x_3885_ = v___x_3882_;
goto v_reusejp_3884_;
}
else
{
lean_object* v_reuseFailAlloc_3886_; 
v_reuseFailAlloc_3886_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3886_, 0, v_a_3880_);
v___x_3885_ = v_reuseFailAlloc_3886_;
goto v_reusejp_3884_;
}
v_reusejp_3884_:
{
return v___x_3885_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms(lean_object* v_x_3888_, lean_object* v_a_3889_, lean_object* v_a_3890_, lean_object* v_a_3891_, lean_object* v_a_3892_){
_start:
{
switch(lean_obj_tag(v_x_3888_))
{
case 7:
{
lean_object* v_binderName_3894_; lean_object* v_binderType_3895_; lean_object* v_body_3896_; uint8_t v_binderInfo_3897_; lean_object* v___x_3898_; 
v_binderName_3894_ = lean_ctor_get(v_x_3888_, 0);
lean_inc(v_binderName_3894_);
v_binderType_3895_ = lean_ctor_get(v_x_3888_, 1);
lean_inc_ref_n(v_binderType_3895_, 2);
v_body_3896_ = lean_ctor_get(v_x_3888_, 2);
lean_inc_ref(v_body_3896_);
v_binderInfo_3897_ = lean_ctor_get_uint8(v_x_3888_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_3888_, 3);
v___x_3898_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_binderType_3895_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3898_) == 0)
{
lean_object* v_a_3899_; lean_object* v___f_3900_; uint8_t v___x_3901_; lean_object* v___x_3902_; 
v_a_3899_ = lean_ctor_get(v___x_3898_, 0);
lean_inc(v_a_3899_);
lean_dec_ref_known(v___x_3898_, 1);
v___f_3900_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3900_, 0, v_body_3896_);
v___x_3901_ = 0;
lean_inc(v_binderName_3894_);
v___x_3902_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg(v_binderName_3894_, v_binderInfo_3897_, v_binderType_3895_, v___f_3900_, v___x_3901_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3902_) == 0)
{
lean_object* v_a_3903_; lean_object* v___x_3905_; uint8_t v_isShared_3906_; uint8_t v_isSharedCheck_3911_; 
v_a_3903_ = lean_ctor_get(v___x_3902_, 0);
v_isSharedCheck_3911_ = !lean_is_exclusive(v___x_3902_);
if (v_isSharedCheck_3911_ == 0)
{
v___x_3905_ = v___x_3902_;
v_isShared_3906_ = v_isSharedCheck_3911_;
goto v_resetjp_3904_;
}
else
{
lean_inc(v_a_3903_);
lean_dec(v___x_3902_);
v___x_3905_ = lean_box(0);
v_isShared_3906_ = v_isSharedCheck_3911_;
goto v_resetjp_3904_;
}
v_resetjp_3904_:
{
lean_object* v___x_3907_; lean_object* v___x_3909_; 
v___x_3907_ = l_Lean_Expr_forallE___override(v_binderName_3894_, v_a_3899_, v_a_3903_, v_binderInfo_3897_);
if (v_isShared_3906_ == 0)
{
lean_ctor_set(v___x_3905_, 0, v___x_3907_);
v___x_3909_ = v___x_3905_;
goto v_reusejp_3908_;
}
else
{
lean_object* v_reuseFailAlloc_3910_; 
v_reuseFailAlloc_3910_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3910_, 0, v___x_3907_);
v___x_3909_ = v_reuseFailAlloc_3910_;
goto v_reusejp_3908_;
}
v_reusejp_3908_:
{
return v___x_3909_;
}
}
}
else
{
lean_dec(v_a_3899_);
lean_dec(v_binderName_3894_);
return v___x_3902_;
}
}
else
{
lean_dec_ref(v_body_3896_);
lean_dec_ref(v_binderType_3895_);
lean_dec(v_binderName_3894_);
return v___x_3898_;
}
}
case 6:
{
lean_object* v_binderName_3912_; lean_object* v_binderType_3913_; lean_object* v_body_3914_; uint8_t v_binderInfo_3915_; lean_object* v___x_3916_; 
v_binderName_3912_ = lean_ctor_get(v_x_3888_, 0);
lean_inc(v_binderName_3912_);
v_binderType_3913_ = lean_ctor_get(v_x_3888_, 1);
lean_inc_ref_n(v_binderType_3913_, 2);
v_body_3914_ = lean_ctor_get(v_x_3888_, 2);
lean_inc_ref(v_body_3914_);
v_binderInfo_3915_ = lean_ctor_get_uint8(v_x_3888_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_3888_, 3);
v___x_3916_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_binderType_3913_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3916_) == 0)
{
lean_object* v_a_3917_; lean_object* v___f_3918_; uint8_t v___x_3919_; lean_object* v___x_3920_; 
v_a_3917_ = lean_ctor_get(v___x_3916_, 0);
lean_inc(v_a_3917_);
lean_dec_ref_known(v___x_3916_, 1);
v___f_3918_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3918_, 0, v_body_3914_);
v___x_3919_ = 0;
lean_inc(v_binderName_3912_);
v___x_3920_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__0___redArg(v_binderName_3912_, v_binderInfo_3915_, v_binderType_3913_, v___f_3918_, v___x_3919_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3920_) == 0)
{
lean_object* v_a_3921_; lean_object* v___x_3923_; uint8_t v_isShared_3924_; uint8_t v_isSharedCheck_3929_; 
v_a_3921_ = lean_ctor_get(v___x_3920_, 0);
v_isSharedCheck_3929_ = !lean_is_exclusive(v___x_3920_);
if (v_isSharedCheck_3929_ == 0)
{
v___x_3923_ = v___x_3920_;
v_isShared_3924_ = v_isSharedCheck_3929_;
goto v_resetjp_3922_;
}
else
{
lean_inc(v_a_3921_);
lean_dec(v___x_3920_);
v___x_3923_ = lean_box(0);
v_isShared_3924_ = v_isSharedCheck_3929_;
goto v_resetjp_3922_;
}
v_resetjp_3922_:
{
lean_object* v___x_3925_; lean_object* v___x_3927_; 
v___x_3925_ = l_Lean_Expr_lam___override(v_binderName_3912_, v_a_3917_, v_a_3921_, v_binderInfo_3915_);
if (v_isShared_3924_ == 0)
{
lean_ctor_set(v___x_3923_, 0, v___x_3925_);
v___x_3927_ = v___x_3923_;
goto v_reusejp_3926_;
}
else
{
lean_object* v_reuseFailAlloc_3928_; 
v_reuseFailAlloc_3928_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3928_, 0, v___x_3925_);
v___x_3927_ = v_reuseFailAlloc_3928_;
goto v_reusejp_3926_;
}
v_reusejp_3926_:
{
return v___x_3927_;
}
}
}
else
{
lean_dec(v_a_3917_);
lean_dec(v_binderName_3912_);
return v___x_3920_;
}
}
else
{
lean_dec_ref(v_body_3914_);
lean_dec_ref(v_binderType_3913_);
lean_dec(v_binderName_3912_);
return v___x_3916_;
}
}
case 8:
{
lean_object* v_declName_3930_; lean_object* v_type_3931_; lean_object* v_value_3932_; lean_object* v_body_3933_; uint8_t v_nondep_3934_; lean_object* v___x_3935_; 
v_declName_3930_ = lean_ctor_get(v_x_3888_, 0);
lean_inc(v_declName_3930_);
v_type_3931_ = lean_ctor_get(v_x_3888_, 1);
lean_inc_ref_n(v_type_3931_, 2);
v_value_3932_ = lean_ctor_get(v_x_3888_, 2);
lean_inc_ref(v_value_3932_);
v_body_3933_ = lean_ctor_get(v_x_3888_, 3);
lean_inc_ref(v_body_3933_);
v_nondep_3934_ = lean_ctor_get_uint8(v_x_3888_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_x_3888_, 4);
v___x_3935_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_type_3931_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3935_) == 0)
{
lean_object* v_a_3936_; lean_object* v___x_3937_; 
v_a_3936_ = lean_ctor_get(v___x_3935_, 0);
lean_inc(v_a_3936_);
lean_dec_ref_known(v___x_3935_, 1);
lean_inc_ref(v_value_3932_);
v___x_3937_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_value_3932_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3937_) == 0)
{
lean_object* v_a_3938_; lean_object* v___f_3939_; uint8_t v___x_3940_; lean_object* v___x_3941_; 
v_a_3938_ = lean_ctor_get(v___x_3937_, 0);
lean_inc(v_a_3938_);
lean_dec_ref_known(v___x_3937_, 1);
v___f_3939_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3939_, 0, v_body_3933_);
v___x_3940_ = 0;
lean_inc(v_declName_3930_);
v___x_3941_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__1___redArg(v_declName_3930_, v_type_3931_, v_value_3932_, v___f_3939_, v_nondep_3934_, v___x_3940_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3941_) == 0)
{
lean_object* v_a_3942_; lean_object* v___x_3944_; uint8_t v_isShared_3945_; uint8_t v_isSharedCheck_3950_; 
v_a_3942_ = lean_ctor_get(v___x_3941_, 0);
v_isSharedCheck_3950_ = !lean_is_exclusive(v___x_3941_);
if (v_isSharedCheck_3950_ == 0)
{
v___x_3944_ = v___x_3941_;
v_isShared_3945_ = v_isSharedCheck_3950_;
goto v_resetjp_3943_;
}
else
{
lean_inc(v_a_3942_);
lean_dec(v___x_3941_);
v___x_3944_ = lean_box(0);
v_isShared_3945_ = v_isSharedCheck_3950_;
goto v_resetjp_3943_;
}
v_resetjp_3943_:
{
lean_object* v___x_3946_; lean_object* v___x_3948_; 
v___x_3946_ = l_Lean_Expr_letE___override(v_declName_3930_, v_a_3936_, v_a_3938_, v_a_3942_, v_nondep_3934_);
if (v_isShared_3945_ == 0)
{
lean_ctor_set(v___x_3944_, 0, v___x_3946_);
v___x_3948_ = v___x_3944_;
goto v_reusejp_3947_;
}
else
{
lean_object* v_reuseFailAlloc_3949_; 
v_reuseFailAlloc_3949_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3949_, 0, v___x_3946_);
v___x_3948_ = v_reuseFailAlloc_3949_;
goto v_reusejp_3947_;
}
v_reusejp_3947_:
{
return v___x_3948_;
}
}
}
else
{
lean_dec(v_a_3938_);
lean_dec(v_a_3936_);
lean_dec(v_declName_3930_);
return v___x_3941_;
}
}
else
{
lean_dec(v_a_3936_);
lean_dec_ref(v_body_3933_);
lean_dec_ref(v_value_3932_);
lean_dec_ref(v_type_3931_);
lean_dec(v_declName_3930_);
return v___x_3937_;
}
}
else
{
lean_dec_ref(v_body_3933_);
lean_dec_ref(v_value_3932_);
lean_dec_ref(v_type_3931_);
lean_dec(v_declName_3930_);
return v___x_3935_;
}
}
case 5:
{
lean_object* v_f_3951_; lean_object* v_dummy_3952_; lean_object* v_nargs_3953_; lean_object* v___x_3954_; lean_object* v___x_3955_; lean_object* v___x_3956_; lean_object* v_args_3957_; lean_object* v___x_3958_; 
v_f_3951_ = l_Lean_Expr_getAppFn(v_x_3888_);
v_dummy_3952_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1);
v_nargs_3953_ = l_Lean_Expr_getAppNumArgs(v_x_3888_);
lean_inc(v_nargs_3953_);
v___x_3954_ = lean_mk_array(v_nargs_3953_, v_dummy_3952_);
v___x_3955_ = lean_unsigned_to_nat(1u);
v___x_3956_ = lean_nat_sub(v_nargs_3953_, v___x_3955_);
lean_dec(v_nargs_3953_);
v_args_3957_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_x_3888_, v___x_3954_, v___x_3956_);
lean_inc_ref(v_f_3951_);
v___x_3958_ = l_Lean_Expr_etaExpandedStrict_x3f(v_f_3951_);
if (lean_obj_tag(v___x_3958_) == 0)
{
lean_object* v___x_3959_; 
v___x_3959_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms(v_f_3951_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3959_) == 0)
{
lean_object* v_a_3960_; size_t v_sz_3961_; size_t v___x_3962_; lean_object* v___x_3963_; 
v_a_3960_ = lean_ctor_get(v___x_3959_, 0);
lean_inc(v_a_3960_);
lean_dec_ref_known(v___x_3959_, 1);
v_sz_3961_ = lean_array_size(v_args_3957_);
v___x_3962_ = ((size_t)0ULL);
v___x_3963_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__2(v_sz_3961_, v___x_3962_, v_args_3957_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3963_) == 0)
{
lean_object* v_a_3964_; lean_object* v___x_3966_; uint8_t v_isShared_3967_; uint8_t v_isSharedCheck_3972_; 
v_a_3964_ = lean_ctor_get(v___x_3963_, 0);
v_isSharedCheck_3972_ = !lean_is_exclusive(v___x_3963_);
if (v_isSharedCheck_3972_ == 0)
{
v___x_3966_ = v___x_3963_;
v_isShared_3967_ = v_isSharedCheck_3972_;
goto v_resetjp_3965_;
}
else
{
lean_inc(v_a_3964_);
lean_dec(v___x_3963_);
v___x_3966_ = lean_box(0);
v_isShared_3967_ = v_isSharedCheck_3972_;
goto v_resetjp_3965_;
}
v_resetjp_3965_:
{
lean_object* v___x_3968_; lean_object* v___x_3970_; 
v___x_3968_ = l_Lean_mkAppN(v_a_3960_, v_a_3964_);
lean_dec(v_a_3964_);
if (v_isShared_3967_ == 0)
{
lean_ctor_set(v___x_3966_, 0, v___x_3968_);
v___x_3970_ = v___x_3966_;
goto v_reusejp_3969_;
}
else
{
lean_object* v_reuseFailAlloc_3971_; 
v_reuseFailAlloc_3971_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3971_, 0, v___x_3968_);
v___x_3970_ = v_reuseFailAlloc_3971_;
goto v_reusejp_3969_;
}
v_reusejp_3969_:
{
return v___x_3970_;
}
}
}
else
{
lean_object* v_a_3973_; lean_object* v___x_3975_; uint8_t v_isShared_3976_; uint8_t v_isSharedCheck_3980_; 
lean_dec(v_a_3960_);
v_a_3973_ = lean_ctor_get(v___x_3963_, 0);
v_isSharedCheck_3980_ = !lean_is_exclusive(v___x_3963_);
if (v_isSharedCheck_3980_ == 0)
{
v___x_3975_ = v___x_3963_;
v_isShared_3976_ = v_isSharedCheck_3980_;
goto v_resetjp_3974_;
}
else
{
lean_inc(v_a_3973_);
lean_dec(v___x_3963_);
v___x_3975_ = lean_box(0);
v_isShared_3976_ = v_isSharedCheck_3980_;
goto v_resetjp_3974_;
}
v_resetjp_3974_:
{
lean_object* v___x_3978_; 
if (v_isShared_3976_ == 0)
{
v___x_3978_ = v___x_3975_;
goto v_reusejp_3977_;
}
else
{
lean_object* v_reuseFailAlloc_3979_; 
v_reuseFailAlloc_3979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3979_, 0, v_a_3973_);
v___x_3978_ = v_reuseFailAlloc_3979_;
goto v_reusejp_3977_;
}
v_reusejp_3977_:
{
return v___x_3978_;
}
}
}
}
else
{
lean_dec_ref(v_args_3957_);
return v___x_3959_;
}
}
else
{
lean_object* v___x_3981_; 
lean_dec_ref_known(v___x_3958_, 1);
v___x_3981_ = l_Lean_Expr_beta(v_f_3951_, v_args_3957_);
v_x_3888_ = v___x_3981_;
goto _start;
}
}
case 10:
{
lean_object* v_data_3983_; lean_object* v_expr_3984_; lean_object* v___x_3985_; 
v_data_3983_ = lean_ctor_get(v_x_3888_, 0);
lean_inc(v_data_3983_);
v_expr_3984_ = lean_ctor_get(v_x_3888_, 1);
lean_inc_ref(v_expr_3984_);
lean_dec_ref_known(v_x_3888_, 2);
v___x_3985_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_expr_3984_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3985_) == 0)
{
lean_object* v_a_3986_; lean_object* v___x_3988_; uint8_t v_isShared_3989_; uint8_t v_isSharedCheck_3994_; 
v_a_3986_ = lean_ctor_get(v___x_3985_, 0);
v_isSharedCheck_3994_ = !lean_is_exclusive(v___x_3985_);
if (v_isSharedCheck_3994_ == 0)
{
v___x_3988_ = v___x_3985_;
v_isShared_3989_ = v_isSharedCheck_3994_;
goto v_resetjp_3987_;
}
else
{
lean_inc(v_a_3986_);
lean_dec(v___x_3985_);
v___x_3988_ = lean_box(0);
v_isShared_3989_ = v_isSharedCheck_3994_;
goto v_resetjp_3987_;
}
v_resetjp_3987_:
{
lean_object* v___x_3990_; lean_object* v___x_3992_; 
v___x_3990_ = l_Lean_Expr_mdata___override(v_data_3983_, v_a_3986_);
if (v_isShared_3989_ == 0)
{
lean_ctor_set(v___x_3988_, 0, v___x_3990_);
v___x_3992_ = v___x_3988_;
goto v_reusejp_3991_;
}
else
{
lean_object* v_reuseFailAlloc_3993_; 
v_reuseFailAlloc_3993_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3993_, 0, v___x_3990_);
v___x_3992_ = v_reuseFailAlloc_3993_;
goto v_reusejp_3991_;
}
v_reusejp_3991_:
{
return v___x_3992_;
}
}
}
else
{
lean_dec(v_data_3983_);
return v___x_3985_;
}
}
case 11:
{
lean_object* v_typeName_3995_; lean_object* v_idx_3996_; lean_object* v_struct_3997_; lean_object* v___x_3998_; 
v_typeName_3995_ = lean_ctor_get(v_x_3888_, 0);
lean_inc(v_typeName_3995_);
v_idx_3996_ = lean_ctor_get(v_x_3888_, 1);
lean_inc(v_idx_3996_);
v_struct_3997_ = lean_ctor_get(v_x_3888_, 2);
lean_inc_ref(v_struct_3997_);
lean_dec_ref_known(v_x_3888_, 3);
v___x_3998_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_struct_3997_, v_a_3889_, v_a_3890_, v_a_3891_, v_a_3892_);
if (lean_obj_tag(v___x_3998_) == 0)
{
lean_object* v_a_3999_; lean_object* v___x_4001_; uint8_t v_isShared_4002_; uint8_t v_isSharedCheck_4007_; 
v_a_3999_ = lean_ctor_get(v___x_3998_, 0);
v_isSharedCheck_4007_ = !lean_is_exclusive(v___x_3998_);
if (v_isSharedCheck_4007_ == 0)
{
v___x_4001_ = v___x_3998_;
v_isShared_4002_ = v_isSharedCheck_4007_;
goto v_resetjp_4000_;
}
else
{
lean_inc(v_a_3999_);
lean_dec(v___x_3998_);
v___x_4001_ = lean_box(0);
v_isShared_4002_ = v_isSharedCheck_4007_;
goto v_resetjp_4000_;
}
v_resetjp_4000_:
{
lean_object* v___x_4003_; lean_object* v___x_4005_; 
v___x_4003_ = l_Lean_Expr_proj___override(v_typeName_3995_, v_idx_3996_, v_a_3999_);
if (v_isShared_4002_ == 0)
{
lean_ctor_set(v___x_4001_, 0, v___x_4003_);
v___x_4005_ = v___x_4001_;
goto v_reusejp_4004_;
}
else
{
lean_object* v_reuseFailAlloc_4006_; 
v_reuseFailAlloc_4006_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4006_, 0, v___x_4003_);
v___x_4005_ = v_reuseFailAlloc_4006_;
goto v_reusejp_4004_;
}
v_reusejp_4004_:
{
return v___x_4005_;
}
}
}
else
{
lean_dec(v_idx_3996_);
lean_dec(v_typeName_3995_);
return v___x_3998_;
}
}
default: 
{
lean_object* v___x_4008_; 
v___x_4008_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4008_, 0, v_x_3888_);
return v___x_4008_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll___lam__0(lean_object* v_e_4009_, uint8_t v___x_4010_, uint8_t v___x_4011_, lean_object* v_xs_4012_, lean_object* v_x_4013_, lean_object* v___y_4014_, lean_object* v___y_4015_, lean_object* v___y_4016_, lean_object* v___y_4017_){
_start:
{
lean_object* v___x_4019_; lean_object* v___x_4020_; lean_object* v___x_4021_; 
v___x_4019_ = lean_expr_instantiate(v_e_4009_, v_xs_4012_);
v___x_4020_ = l_Lean_mkAppN(v___x_4019_, v_xs_4012_);
v___x_4021_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms(v___x_4020_, v___y_4014_, v___y_4015_, v___y_4016_, v___y_4017_);
if (lean_obj_tag(v___x_4021_) == 0)
{
lean_object* v_a_4022_; uint8_t v___x_4023_; lean_object* v___x_4024_; 
v_a_4022_ = lean_ctor_get(v___x_4021_, 0);
lean_inc(v_a_4022_);
lean_dec_ref_known(v___x_4021_, 1);
v___x_4023_ = 1;
v___x_4024_ = l_Lean_Meta_mkLambdaFVars(v_xs_4012_, v_a_4022_, v___x_4010_, v___x_4011_, v___x_4010_, v___x_4011_, v___x_4023_, v___y_4014_, v___y_4015_, v___y_4016_, v___y_4017_);
return v___x_4024_;
}
else
{
return v___x_4021_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaExpandAll___boxed(lean_object* v_e_4025_, lean_object* v_a_4026_, lean_object* v_a_4027_, lean_object* v_a_4028_, lean_object* v_a_4029_, lean_object* v_a_4030_){
_start:
{
lean_object* v_res_4031_; 
v_res_4031_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v_e_4025_, v_a_4026_, v_a_4027_, v_a_4028_, v_a_4029_);
lean_dec(v_a_4029_);
lean_dec_ref(v_a_4028_);
lean_dec(v_a_4027_);
lean_dec_ref(v_a_4026_);
return v_res_4031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__2___boxed(lean_object* v_sz_4032_, lean_object* v_i_4033_, lean_object* v_bs_4034_, lean_object* v___y_4035_, lean_object* v___y_4036_, lean_object* v___y_4037_, lean_object* v___y_4038_, lean_object* v___y_4039_){
_start:
{
size_t v_sz_boxed_4040_; size_t v_i_boxed_4041_; lean_object* v_res_4042_; 
v_sz_boxed_4040_ = lean_unbox_usize(v_sz_4032_);
lean_dec(v_sz_4032_);
v_i_boxed_4041_ = lean_unbox_usize(v_i_4033_);
lean_dec(v_i_4033_);
v_res_4042_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms_spec__2(v_sz_boxed_4040_, v_i_boxed_4041_, v_bs_4034_, v___y_4035_, v___y_4036_, v___y_4037_, v___y_4038_);
lean_dec(v___y_4038_);
lean_dec_ref(v___y_4037_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
return v_res_4042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms___boxed(lean_object* v_x_4043_, lean_object* v_a_4044_, lean_object* v_a_4045_, lean_object* v_a_4046_, lean_object* v_a_4047_, lean_object* v_a_4048_){
_start:
{
lean_object* v_res_4049_; 
v_res_4049_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaExpandAll_expandSubterms(v_x_4043_, v_a_4044_, v_a_4045_, v_a_4046_, v_a_4047_);
lean_dec(v_a_4047_);
lean_dec_ref(v_a_4046_);
lean_dec(v_a_4045_);
lean_dec_ref(v_a_4044_);
return v_res_4049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4(lean_object* v_00_u03b1_4050_, lean_object* v_type_4051_, lean_object* v_k_4052_, uint8_t v_cleanupAnnotations_4053_, uint8_t v_whnfType_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_, lean_object* v___y_4057_, lean_object* v___y_4058_){
_start:
{
lean_object* v___x_4060_; 
v___x_4060_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___redArg(v_type_4051_, v_k_4052_, v_cleanupAnnotations_4053_, v_whnfType_4054_, v___y_4055_, v___y_4056_, v___y_4057_, v___y_4058_);
return v___x_4060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4___boxed(lean_object* v_00_u03b1_4061_, lean_object* v_type_4062_, lean_object* v_k_4063_, lean_object* v_cleanupAnnotations_4064_, lean_object* v_whnfType_4065_, lean_object* v___y_4066_, lean_object* v___y_4067_, lean_object* v___y_4068_, lean_object* v___y_4069_, lean_object* v___y_4070_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_4071_; uint8_t v_whnfType_boxed_4072_; lean_object* v_res_4073_; 
v_cleanupAnnotations_boxed_4071_ = lean_unbox(v_cleanupAnnotations_4064_);
v_whnfType_boxed_4072_ = lean_unbox(v_whnfType_4065_);
v_res_4073_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_etaExpandAll_spec__4(v_00_u03b1_4061_, v_type_4062_, v_k_4063_, v_cleanupAnnotations_boxed_4071_, v_whnfType_boxed_4072_, v___y_4066_, v___y_4067_, v___y_4068_, v___y_4069_);
lean_dec(v___y_4069_);
lean_dec_ref(v___y_4068_);
lean_dec(v___y_4067_);
lean_dec_ref(v___y_4066_);
return v_res_4073_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__4(void){
_start:
{
lean_object* v___x_4083_; lean_object* v___x_4084_; lean_object* v___x_4085_; lean_object* v___x_4086_; 
v___x_4083_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_4084_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__3));
v___x_4085_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_4086_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4086_, 0, v___x_4085_);
lean_ctor_set(v___x_4086_, 1, v___x_4084_);
lean_ctor_set(v___x_4086_, 2, v___x_4083_);
return v___x_4086_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__5(void){
_start:
{
lean_object* v___x_4087_; lean_object* v___x_4088_; lean_object* v___x_4089_; lean_object* v___x_4090_; 
v___x_4087_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__4, &lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__4);
v___x_4088_ = lean_unsigned_to_nat(1022u);
v___x_4089_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1));
v___x_4090_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4090_, 0, v___x_4089_);
lean_ctor_set(v___x_4090_, 1, v___x_4088_);
lean_ctor_set(v___x_4090_, 2, v___x_4087_);
return v___x_4090_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaExpandStx(void){
_start:
{
lean_object* v___x_4091_; 
v___x_4091_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__5, &lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__5);
return v___x_4091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___lam__0(lean_object* v_x_4092_, lean_object* v___y_4093_, lean_object* v___y_4094_, lean_object* v___y_4095_, lean_object* v___y_4096_, lean_object* v___y_4097_){
_start:
{
lean_object* v___x_4099_; 
v___x_4099_ = lp_mathlib_Mathlib_Tactic_etaExpandAll(v___y_4093_, v___y_4094_, v___y_4095_, v___y_4096_, v___y_4097_);
return v___x_4099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___lam__0___boxed(lean_object* v_x_4100_, lean_object* v___y_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_, lean_object* v___y_4104_, lean_object* v___y_4105_, lean_object* v___y_4106_){
_start:
{
lean_object* v_res_4107_; 
v_res_4107_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___lam__0(v_x_4100_, v___y_4101_, v___y_4102_, v___y_4103_, v___y_4104_, v___y_4105_);
lean_dec(v___y_4105_);
lean_dec_ref(v___y_4104_);
lean_dec(v___y_4103_);
lean_dec_ref(v___y_4102_);
lean_dec(v_x_4100_);
return v_res_4107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1(lean_object* v_x_4109_, lean_object* v_a_4110_, lean_object* v_a_4111_, lean_object* v_a_4112_, lean_object* v_a_4113_, lean_object* v_a_4114_, lean_object* v_a_4115_, lean_object* v_a_4116_, lean_object* v_a_4117_){
_start:
{
lean_object* v___x_4119_; uint8_t v___x_4120_; 
v___x_4119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__1));
lean_inc(v_x_4109_);
v___x_4120_ = l_Lean_Syntax_isOfKind(v_x_4109_, v___x_4119_);
if (v___x_4120_ == 0)
{
lean_object* v___x_4121_; 
lean_dec(v_x_4109_);
v___x_4121_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_4121_;
}
else
{
lean_object* v___f_4122_; lean_object* v___y_4124_; lean_object* v___x_4127_; lean_object* v___x_4128_; lean_object* v___x_4129_; 
v___f_4122_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___closed__0));
v___x_4127_ = lean_unsigned_to_nat(1u);
v___x_4128_ = l_Lean_Syntax_getArg(v_x_4109_, v___x_4127_);
lean_dec(v_x_4109_);
v___x_4129_ = l_Lean_Syntax_getOptional_x3f(v___x_4128_);
lean_dec(v___x_4128_);
if (lean_obj_tag(v___x_4129_) == 0)
{
lean_object* v___x_4130_; 
v___x_4130_ = lean_box(0);
v___y_4124_ = v___x_4130_;
goto v___jp_4123_;
}
else
{
lean_object* v_val_4131_; lean_object* v___x_4133_; uint8_t v_isShared_4134_; uint8_t v_isSharedCheck_4138_; 
v_val_4131_ = lean_ctor_get(v___x_4129_, 0);
v_isSharedCheck_4138_ = !lean_is_exclusive(v___x_4129_);
if (v_isSharedCheck_4138_ == 0)
{
v___x_4133_ = v___x_4129_;
v_isShared_4134_ = v_isSharedCheck_4138_;
goto v_resetjp_4132_;
}
else
{
lean_inc(v_val_4131_);
lean_dec(v___x_4129_);
v___x_4133_ = lean_box(0);
v_isShared_4134_ = v_isSharedCheck_4138_;
goto v_resetjp_4132_;
}
v_resetjp_4132_:
{
lean_object* v___x_4136_; 
if (v_isShared_4134_ == 0)
{
v___x_4136_ = v___x_4133_;
goto v_reusejp_4135_;
}
else
{
lean_object* v_reuseFailAlloc_4137_; 
v_reuseFailAlloc_4137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4137_, 0, v_val_4131_);
v___x_4136_ = v_reuseFailAlloc_4137_;
goto v_reusejp_4135_;
}
v_reusejp_4135_:
{
v___y_4124_ = v___x_4136_;
goto v___jp_4123_;
}
}
}
v___jp_4123_:
{
lean_object* v___x_4125_; lean_object* v___x_4126_; 
v___x_4125_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaExpandStx___closed__2));
v___x_4126_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___f_4122_, v___y_4124_, v___x_4125_, v___x_4120_, v_a_4110_, v_a_4111_, v_a_4112_, v_a_4113_, v_a_4114_, v_a_4115_, v_a_4116_, v_a_4117_);
return v___x_4126_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1___boxed(lean_object* v_x_4139_, lean_object* v_a_4140_, lean_object* v_a_4141_, lean_object* v_a_4142_, lean_object* v_a_4143_, lean_object* v_a_4144_, lean_object* v_a_4145_, lean_object* v_a_4146_, lean_object* v_a_4147_, lean_object* v_a_4148_){
_start:
{
lean_object* v_res_4149_; 
v_res_4149_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaExpandStx__1(v_x_4139_, v_a_4140_, v_a_4141_, v_a_4142_, v_a_4143_, v_a_4144_, v_a_4145_, v_a_4146_, v_a_4147_);
lean_dec(v_a_4147_);
lean_dec_ref(v_a_4146_);
lean_dec(v_a_4145_);
lean_dec_ref(v_a_4144_);
lean_dec(v_a_4143_);
lean_dec_ref(v_a_4142_);
lean_dec(v_a_4141_);
lean_dec_ref(v_a_4140_);
return v_res_4149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1(lean_object* v_x_4161_, lean_object* v_a_4162_, lean_object* v_a_4163_, lean_object* v_a_4164_, lean_object* v_a_4165_, lean_object* v_a_4166_, lean_object* v_a_4167_, lean_object* v_a_4168_, lean_object* v_a_4169_){
_start:
{
lean_object* v___x_4171_; uint8_t v___x_4172_; 
v___x_4171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convEta__expand___closed__1));
v___x_4172_ = l_Lean_Syntax_isOfKind(v_x_4161_, v___x_4171_);
if (v___x_4172_ == 0)
{
lean_object* v___x_4173_; 
v___x_4173_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_4173_;
}
else
{
lean_object* v___x_4174_; lean_object* v___x_4175_; 
v___x_4174_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1___closed__0));
v___x_4175_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(v___x_4174_, v_a_4162_, v_a_4163_, v_a_4164_, v_a_4165_, v_a_4166_, v_a_4167_, v_a_4168_, v_a_4169_);
return v___x_4175_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1___boxed(lean_object* v_x_4176_, lean_object* v_a_4177_, lean_object* v_a_4178_, lean_object* v_a_4179_, lean_object* v_a_4180_, lean_object* v_a_4181_, lean_object* v_a_4182_, lean_object* v_a_4183_, lean_object* v_a_4184_, lean_object* v_a_4185_){
_start:
{
lean_object* v_res_4186_; 
v_res_4186_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__expand__1(v_x_4176_, v_a_4177_, v_a_4178_, v_a_4179_, v_a_4180_, v_a_4181_, v_a_4182_, v_a_4183_, v_a_4184_);
lean_dec(v_a_4184_);
lean_dec_ref(v_a_4183_);
lean_dec(v_a_4182_);
lean_dec_ref(v_a_4181_);
lean_dec(v_a_4180_);
lean_dec_ref(v_a_4179_);
lean_dec(v_a_4178_);
lean_dec_ref(v_a_4177_);
return v_res_4186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___redArg(lean_object* v_declName_4187_, lean_object* v___y_4188_){
_start:
{
lean_object* v___x_4190_; lean_object* v_env_4191_; lean_object* v___x_4192_; lean_object* v___x_4193_; 
v___x_4190_ = lean_st_ref_get(v___y_4188_);
v_env_4191_ = lean_ctor_get(v___x_4190_, 0);
lean_inc_ref(v_env_4191_);
lean_dec(v___x_4190_);
v___x_4192_ = l_Lean_Environment_getProjectionFnInfo_x3f(v_env_4191_, v_declName_4187_);
v___x_4193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4193_, 0, v___x_4192_);
return v___x_4193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___redArg___boxed(lean_object* v_declName_4194_, lean_object* v___y_4195_, lean_object* v___y_4196_){
_start:
{
lean_object* v_res_4197_; 
v_res_4197_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___redArg(v_declName_4194_, v___y_4195_);
lean_dec(v___y_4195_);
return v_res_4197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0(lean_object* v_declName_4198_, lean_object* v___y_4199_, lean_object* v___y_4200_, lean_object* v___y_4201_, lean_object* v___y_4202_){
_start:
{
lean_object* v___x_4204_; 
v___x_4204_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___redArg(v_declName_4198_, v___y_4202_);
return v___x_4204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___boxed(lean_object* v_declName_4205_, lean_object* v___y_4206_, lean_object* v___y_4207_, lean_object* v___y_4208_, lean_object* v___y_4209_, lean_object* v___y_4210_){
_start:
{
lean_object* v_res_4211_; 
v_res_4211_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0(v_declName_4205_, v___y_4206_, v___y_4207_, v___y_4208_, v___y_4209_);
lean_dec(v___y_4209_);
lean_dec_ref(v___y_4208_);
lean_dec(v___y_4207_);
lean_dec_ref(v___y_4206_);
return v_res_4211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_getProjectedExpr(lean_object* v_e_4212_, lean_object* v_a_4213_, lean_object* v_a_4214_, lean_object* v_a_4215_, lean_object* v_a_4216_){
_start:
{
if (lean_obj_tag(v_e_4212_) == 11)
{
lean_object* v_typeName_4221_; lean_object* v_idx_4222_; lean_object* v_struct_4223_; lean_object* v___x_4224_; lean_object* v___x_4225_; lean_object* v___x_4226_; lean_object* v___x_4227_; 
v_typeName_4221_ = lean_ctor_get(v_e_4212_, 0);
v_idx_4222_ = lean_ctor_get(v_e_4212_, 1);
v_struct_4223_ = lean_ctor_get(v_e_4212_, 2);
lean_inc_ref(v_struct_4223_);
lean_inc(v_idx_4222_);
v___x_4224_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4224_, 0, v_idx_4222_);
lean_ctor_set(v___x_4224_, 1, v_struct_4223_);
lean_inc(v_typeName_4221_);
v___x_4225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4225_, 0, v_typeName_4221_);
lean_ctor_set(v___x_4225_, 1, v___x_4224_);
v___x_4226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4226_, 0, v___x_4225_);
v___x_4227_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4227_, 0, v___x_4226_);
return v___x_4227_;
}
else
{
lean_object* v___x_4228_; 
v___x_4228_ = l_Lean_Expr_getAppFn(v_e_4212_);
if (lean_obj_tag(v___x_4228_) == 4)
{
lean_object* v_declName_4229_; lean_object* v___x_4230_; lean_object* v_a_4231_; lean_object* v___x_4233_; uint8_t v_isShared_4234_; uint8_t v_isSharedCheck_4263_; 
v_declName_4229_ = lean_ctor_get(v___x_4228_, 0);
lean_inc(v_declName_4229_);
lean_dec_ref_known(v___x_4228_, 2);
v___x_4230_ = lp_mathlib_Lean_getProjectionFnInfo_x3f___at___00Mathlib_Tactic_getProjectedExpr_spec__0___redArg(v_declName_4229_, v_a_4216_);
v_a_4231_ = lean_ctor_get(v___x_4230_, 0);
v_isSharedCheck_4263_ = !lean_is_exclusive(v___x_4230_);
if (v_isSharedCheck_4263_ == 0)
{
v___x_4233_ = v___x_4230_;
v_isShared_4234_ = v_isSharedCheck_4263_;
goto v_resetjp_4232_;
}
else
{
lean_inc(v_a_4231_);
lean_dec(v___x_4230_);
v___x_4233_ = lean_box(0);
v_isShared_4234_ = v_isSharedCheck_4263_;
goto v_resetjp_4232_;
}
v_resetjp_4232_:
{
if (lean_obj_tag(v_a_4231_) == 1)
{
lean_object* v_val_4235_; lean_object* v_ctorName_4236_; lean_object* v_numParams_4237_; lean_object* v_i_4238_; lean_object* v___x_4239_; lean_object* v___x_4240_; lean_object* v___x_4241_; uint8_t v___x_4242_; 
v_val_4235_ = lean_ctor_get(v_a_4231_, 0);
lean_inc(v_val_4235_);
lean_dec_ref_known(v_a_4231_, 1);
v_ctorName_4236_ = lean_ctor_get(v_val_4235_, 0);
lean_inc(v_ctorName_4236_);
v_numParams_4237_ = lean_ctor_get(v_val_4235_, 1);
lean_inc(v_numParams_4237_);
v_i_4238_ = lean_ctor_get(v_val_4235_, 2);
lean_inc(v_i_4238_);
lean_dec(v_val_4235_);
v___x_4239_ = l_Lean_Expr_getAppNumArgs(v_e_4212_);
v___x_4240_ = lean_unsigned_to_nat(1u);
v___x_4241_ = lean_nat_add(v_numParams_4237_, v___x_4240_);
lean_dec(v_numParams_4237_);
v___x_4242_ = lean_nat_dec_eq(v___x_4239_, v___x_4241_);
lean_dec(v___x_4241_);
lean_dec(v___x_4239_);
if (v___x_4242_ == 0)
{
lean_dec(v_i_4238_);
lean_dec(v_ctorName_4236_);
lean_del_object(v___x_4233_);
goto v___jp_4218_;
}
else
{
lean_object* v___x_4243_; lean_object* v_env_4244_; uint8_t v___x_4245_; lean_object* v___x_4246_; 
v___x_4243_ = lean_st_ref_get(v_a_4216_);
v_env_4244_ = lean_ctor_get(v___x_4243_, 0);
lean_inc_ref(v_env_4244_);
lean_dec(v___x_4243_);
v___x_4245_ = 0;
v___x_4246_ = l_Lean_Environment_find_x3f(v_env_4244_, v_ctorName_4236_, v___x_4245_);
if (lean_obj_tag(v___x_4246_) == 1)
{
lean_object* v_val_4247_; lean_object* v___x_4249_; uint8_t v_isShared_4250_; uint8_t v_isSharedCheck_4262_; 
v_val_4247_ = lean_ctor_get(v___x_4246_, 0);
v_isSharedCheck_4262_ = !lean_is_exclusive(v___x_4246_);
if (v_isSharedCheck_4262_ == 0)
{
v___x_4249_ = v___x_4246_;
v_isShared_4250_ = v_isSharedCheck_4262_;
goto v_resetjp_4248_;
}
else
{
lean_inc(v_val_4247_);
lean_dec(v___x_4246_);
v___x_4249_ = lean_box(0);
v_isShared_4250_ = v_isSharedCheck_4262_;
goto v_resetjp_4248_;
}
v_resetjp_4248_:
{
if (lean_obj_tag(v_val_4247_) == 6)
{
lean_object* v_val_4251_; lean_object* v_induct_4252_; lean_object* v___x_4253_; lean_object* v___x_4254_; lean_object* v___x_4255_; lean_object* v___x_4257_; 
v_val_4251_ = lean_ctor_get(v_val_4247_, 0);
lean_inc_ref(v_val_4251_);
lean_dec_ref_known(v_val_4247_, 1);
v_induct_4252_ = lean_ctor_get(v_val_4251_, 1);
lean_inc(v_induct_4252_);
lean_dec_ref(v_val_4251_);
v___x_4253_ = l_Lean_Expr_appArg_x21(v_e_4212_);
v___x_4254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4254_, 0, v_i_4238_);
lean_ctor_set(v___x_4254_, 1, v___x_4253_);
v___x_4255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4255_, 0, v_induct_4252_);
lean_ctor_set(v___x_4255_, 1, v___x_4254_);
if (v_isShared_4250_ == 0)
{
lean_ctor_set(v___x_4249_, 0, v___x_4255_);
v___x_4257_ = v___x_4249_;
goto v_reusejp_4256_;
}
else
{
lean_object* v_reuseFailAlloc_4261_; 
v_reuseFailAlloc_4261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4261_, 0, v___x_4255_);
v___x_4257_ = v_reuseFailAlloc_4261_;
goto v_reusejp_4256_;
}
v_reusejp_4256_:
{
lean_object* v___x_4259_; 
if (v_isShared_4234_ == 0)
{
lean_ctor_set(v___x_4233_, 0, v___x_4257_);
v___x_4259_ = v___x_4233_;
goto v_reusejp_4258_;
}
else
{
lean_object* v_reuseFailAlloc_4260_; 
v_reuseFailAlloc_4260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4260_, 0, v___x_4257_);
v___x_4259_ = v_reuseFailAlloc_4260_;
goto v_reusejp_4258_;
}
v_reusejp_4258_:
{
return v___x_4259_;
}
}
}
else
{
lean_del_object(v___x_4249_);
lean_dec(v_val_4247_);
lean_dec(v_i_4238_);
lean_del_object(v___x_4233_);
goto v___jp_4218_;
}
}
}
else
{
lean_dec(v___x_4246_);
lean_dec(v_i_4238_);
lean_del_object(v___x_4233_);
goto v___jp_4218_;
}
}
}
else
{
lean_del_object(v___x_4233_);
lean_dec(v_a_4231_);
goto v___jp_4218_;
}
}
}
else
{
lean_dec_ref(v___x_4228_);
goto v___jp_4218_;
}
}
v___jp_4218_:
{
lean_object* v___x_4219_; lean_object* v___x_4220_; 
v___x_4219_ = lean_box(0);
v___x_4220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4220_, 0, v___x_4219_);
return v___x_4220_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_getProjectedExpr___boxed(lean_object* v_e_4264_, lean_object* v_a_4265_, lean_object* v_a_4266_, lean_object* v_a_4267_, lean_object* v_a_4268_, lean_object* v_a_4269_){
_start:
{
lean_object* v_res_4270_; 
v_res_4270_ = lp_mathlib_Mathlib_Tactic_getProjectedExpr(v_e_4264_, v_a_4265_, v_a_4266_, v_a_4267_, v_a_4268_);
lean_dec(v_a_4268_);
lean_dec_ref(v_a_4267_);
lean_dec(v_a_4266_);
lean_dec_ref(v_a_4265_);
lean_dec_ref(v_e_4264_);
return v_res_4270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg(lean_object* v_fVal_4279_, lean_object* v_args_4280_, lean_object* v_m_4281_, lean_object* v_range_4282_, lean_object* v_b_4283_, lean_object* v_i_4284_, lean_object* v___y_4285_, lean_object* v___y_4286_, lean_object* v___y_4287_, lean_object* v___y_4288_){
_start:
{
lean_object* v_stop_4290_; lean_object* v_step_4291_; uint8_t v___x_4292_; 
v_stop_4290_ = lean_ctor_get(v_range_4282_, 1);
v_step_4291_ = lean_ctor_get(v_range_4282_, 2);
v___x_4292_ = lean_nat_dec_lt(v_i_4284_, v_stop_4290_);
if (v___x_4292_ == 0)
{
lean_object* v___x_4293_; 
lean_dec(v_i_4284_);
lean_dec_ref(v_m_4281_);
v___x_4293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4293_, 0, v_b_4283_);
return v___x_4293_;
}
else
{
lean_object* v_induct_4294_; lean_object* v_numParams_4295_; lean_object* v___x_4296_; lean_object* v___x_4297_; lean_object* v___x_4298_; lean_object* v___x_4299_; 
lean_dec_ref(v_b_4283_);
v_induct_4294_ = lean_ctor_get(v_fVal_4279_, 1);
v_numParams_4295_ = lean_ctor_get(v_fVal_4279_, 3);
v___x_4296_ = l_Lean_instInhabitedExpr;
v___x_4297_ = lean_nat_add(v_numParams_4295_, v_i_4284_);
v___x_4298_ = lean_array_get_borrowed(v___x_4296_, v_args_4280_, v___x_4297_);
lean_dec(v___x_4297_);
lean_inc_ref(v_m_4281_);
lean_inc(v___y_4288_);
lean_inc_ref(v___y_4287_);
lean_inc(v___y_4286_);
lean_inc_ref(v___y_4285_);
lean_inc(v___x_4298_);
v___x_4299_ = lean_apply_6(v_m_4281_, v___x_4298_, v___y_4285_, v___y_4286_, v___y_4287_, v___y_4288_, lean_box(0));
if (lean_obj_tag(v___x_4299_) == 0)
{
lean_object* v_a_4300_; lean_object* v___x_4301_; 
v_a_4300_ = lean_ctor_get(v___x_4299_, 0);
lean_inc(v_a_4300_);
lean_dec_ref_known(v___x_4299_, 1);
v___x_4301_ = lp_mathlib_Mathlib_Tactic_getProjectedExpr(v_a_4300_, v___y_4285_, v___y_4286_, v___y_4287_, v___y_4288_);
lean_dec(v_a_4300_);
if (lean_obj_tag(v___x_4301_) == 0)
{
lean_object* v_a_4302_; lean_object* v___x_4304_; uint8_t v_isShared_4305_; uint8_t v_isSharedCheck_4341_; 
v_a_4302_ = lean_ctor_get(v___x_4301_, 0);
v_isSharedCheck_4341_ = !lean_is_exclusive(v___x_4301_);
if (v_isSharedCheck_4341_ == 0)
{
v___x_4304_ = v___x_4301_;
v_isShared_4305_ = v_isSharedCheck_4341_;
goto v_resetjp_4303_;
}
else
{
lean_inc(v_a_4302_);
lean_dec(v___x_4301_);
v___x_4304_ = lean_box(0);
v_isShared_4305_ = v_isSharedCheck_4341_;
goto v_resetjp_4303_;
}
v_resetjp_4303_:
{
lean_object* v___x_4306_; 
v___x_4306_ = lean_box(0);
if (lean_obj_tag(v_a_4302_) == 1)
{
lean_object* v_val_4307_; lean_object* v___x_4309_; uint8_t v_isShared_4310_; uint8_t v_isSharedCheck_4337_; 
lean_dec_ref(v_m_4281_);
v_val_4307_ = lean_ctor_get(v_a_4302_, 0);
v_isSharedCheck_4337_ = !lean_is_exclusive(v_a_4302_);
if (v_isSharedCheck_4337_ == 0)
{
v___x_4309_ = v_a_4302_;
v_isShared_4310_ = v_isSharedCheck_4337_;
goto v_resetjp_4308_;
}
else
{
lean_inc(v_val_4307_);
lean_dec(v_a_4302_);
v___x_4309_ = lean_box(0);
v_isShared_4310_ = v_isSharedCheck_4337_;
goto v_resetjp_4308_;
}
v_resetjp_4308_:
{
lean_object* v_snd_4311_; lean_object* v_fst_4312_; lean_object* v_fst_4313_; lean_object* v_snd_4314_; lean_object* v___x_4316_; uint8_t v_isShared_4317_; uint8_t v_isSharedCheck_4336_; 
v_snd_4311_ = lean_ctor_get(v_val_4307_, 1);
lean_inc(v_snd_4311_);
v_fst_4312_ = lean_ctor_get(v_val_4307_, 0);
lean_inc(v_fst_4312_);
lean_dec(v_val_4307_);
v_fst_4313_ = lean_ctor_get(v_snd_4311_, 0);
v_snd_4314_ = lean_ctor_get(v_snd_4311_, 1);
v_isSharedCheck_4336_ = !lean_is_exclusive(v_snd_4311_);
if (v_isSharedCheck_4336_ == 0)
{
v___x_4316_ = v_snd_4311_;
v_isShared_4317_ = v_isSharedCheck_4336_;
goto v_resetjp_4315_;
}
else
{
lean_inc(v_snd_4314_);
lean_inc(v_fst_4313_);
lean_dec(v_snd_4311_);
v___x_4316_ = lean_box(0);
v_isShared_4317_ = v_isSharedCheck_4336_;
goto v_resetjp_4315_;
}
v_resetjp_4315_:
{
uint8_t v___y_4319_; uint8_t v___x_4334_; 
v___x_4334_ = lean_name_eq(v_fst_4312_, v_induct_4294_);
lean_dec(v_fst_4312_);
if (v___x_4334_ == 0)
{
lean_dec(v_fst_4313_);
lean_dec(v_i_4284_);
v___y_4319_ = v___x_4334_;
goto v___jp_4318_;
}
else
{
uint8_t v___x_4335_; 
v___x_4335_ = lean_nat_dec_eq(v_i_4284_, v_fst_4313_);
lean_dec(v_fst_4313_);
lean_dec(v_i_4284_);
v___y_4319_ = v___x_4335_;
goto v___jp_4318_;
}
v___jp_4318_:
{
if (v___y_4319_ == 0)
{
lean_object* v___x_4320_; lean_object* v___x_4322_; 
lean_del_object(v___x_4316_);
lean_dec(v_snd_4314_);
lean_del_object(v___x_4309_);
v___x_4320_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__1));
if (v_isShared_4305_ == 0)
{
lean_ctor_set(v___x_4304_, 0, v___x_4320_);
v___x_4322_ = v___x_4304_;
goto v_reusejp_4321_;
}
else
{
lean_object* v_reuseFailAlloc_4323_; 
v_reuseFailAlloc_4323_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4323_, 0, v___x_4320_);
v___x_4322_ = v_reuseFailAlloc_4323_;
goto v_reusejp_4321_;
}
v_reusejp_4321_:
{
return v___x_4322_;
}
}
else
{
lean_object* v___x_4324_; lean_object* v___x_4326_; 
v___x_4324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4324_, 0, v_snd_4314_);
if (v_isShared_4310_ == 0)
{
lean_ctor_set(v___x_4309_, 0, v___x_4324_);
v___x_4326_ = v___x_4309_;
goto v_reusejp_4325_;
}
else
{
lean_object* v_reuseFailAlloc_4333_; 
v_reuseFailAlloc_4333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4333_, 0, v___x_4324_);
v___x_4326_ = v_reuseFailAlloc_4333_;
goto v_reusejp_4325_;
}
v_reusejp_4325_:
{
lean_object* v___x_4328_; 
if (v_isShared_4317_ == 0)
{
lean_ctor_set(v___x_4316_, 1, v___x_4306_);
lean_ctor_set(v___x_4316_, 0, v___x_4326_);
v___x_4328_ = v___x_4316_;
goto v_reusejp_4327_;
}
else
{
lean_object* v_reuseFailAlloc_4332_; 
v_reuseFailAlloc_4332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4332_, 0, v___x_4326_);
lean_ctor_set(v_reuseFailAlloc_4332_, 1, v___x_4306_);
v___x_4328_ = v_reuseFailAlloc_4332_;
goto v_reusejp_4327_;
}
v_reusejp_4327_:
{
lean_object* v___x_4330_; 
if (v_isShared_4305_ == 0)
{
lean_ctor_set(v___x_4304_, 0, v___x_4328_);
v___x_4330_ = v___x_4304_;
goto v_reusejp_4329_;
}
else
{
lean_object* v_reuseFailAlloc_4331_; 
v_reuseFailAlloc_4331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4331_, 0, v___x_4328_);
v___x_4330_ = v_reuseFailAlloc_4331_;
goto v_reusejp_4329_;
}
v_reusejp_4329_:
{
return v___x_4330_;
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
lean_object* v___x_4338_; lean_object* v___x_4339_; 
lean_del_object(v___x_4304_);
lean_dec(v_a_4302_);
v___x_4338_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__2));
v___x_4339_ = lean_nat_add(v_i_4284_, v_step_4291_);
lean_dec(v_i_4284_);
v_b_4283_ = v___x_4338_;
v_i_4284_ = v___x_4339_;
goto _start;
}
}
}
else
{
lean_object* v_a_4342_; lean_object* v___x_4344_; uint8_t v_isShared_4345_; uint8_t v_isSharedCheck_4349_; 
lean_dec(v_i_4284_);
lean_dec_ref(v_m_4281_);
v_a_4342_ = lean_ctor_get(v___x_4301_, 0);
v_isSharedCheck_4349_ = !lean_is_exclusive(v___x_4301_);
if (v_isSharedCheck_4349_ == 0)
{
v___x_4344_ = v___x_4301_;
v_isShared_4345_ = v_isSharedCheck_4349_;
goto v_resetjp_4343_;
}
else
{
lean_inc(v_a_4342_);
lean_dec(v___x_4301_);
v___x_4344_ = lean_box(0);
v_isShared_4345_ = v_isSharedCheck_4349_;
goto v_resetjp_4343_;
}
v_resetjp_4343_:
{
lean_object* v___x_4347_; 
if (v_isShared_4345_ == 0)
{
v___x_4347_ = v___x_4344_;
goto v_reusejp_4346_;
}
else
{
lean_object* v_reuseFailAlloc_4348_; 
v_reuseFailAlloc_4348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4348_, 0, v_a_4342_);
v___x_4347_ = v_reuseFailAlloc_4348_;
goto v_reusejp_4346_;
}
v_reusejp_4346_:
{
return v___x_4347_;
}
}
}
}
else
{
lean_object* v_a_4350_; lean_object* v___x_4352_; uint8_t v_isShared_4353_; uint8_t v_isSharedCheck_4357_; 
lean_dec(v_i_4284_);
lean_dec_ref(v_m_4281_);
v_a_4350_ = lean_ctor_get(v___x_4299_, 0);
v_isSharedCheck_4357_ = !lean_is_exclusive(v___x_4299_);
if (v_isSharedCheck_4357_ == 0)
{
v___x_4352_ = v___x_4299_;
v_isShared_4353_ = v_isSharedCheck_4357_;
goto v_resetjp_4351_;
}
else
{
lean_inc(v_a_4350_);
lean_dec(v___x_4299_);
v___x_4352_ = lean_box(0);
v_isShared_4353_ = v_isSharedCheck_4357_;
goto v_resetjp_4351_;
}
v_resetjp_4351_:
{
lean_object* v___x_4355_; 
if (v_isShared_4353_ == 0)
{
v___x_4355_ = v___x_4352_;
goto v_reusejp_4354_;
}
else
{
lean_object* v_reuseFailAlloc_4356_; 
v_reuseFailAlloc_4356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4356_, 0, v_a_4350_);
v___x_4355_ = v_reuseFailAlloc_4356_;
goto v_reusejp_4354_;
}
v_reusejp_4354_:
{
return v___x_4355_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___boxed(lean_object* v_fVal_4358_, lean_object* v_args_4359_, lean_object* v_m_4360_, lean_object* v_range_4361_, lean_object* v_b_4362_, lean_object* v_i_4363_, lean_object* v___y_4364_, lean_object* v___y_4365_, lean_object* v___y_4366_, lean_object* v___y_4367_, lean_object* v___y_4368_){
_start:
{
lean_object* v_res_4369_; 
v_res_4369_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg(v_fVal_4358_, v_args_4359_, v_m_4360_, v_range_4361_, v_b_4362_, v_i_4363_, v___y_4364_, v___y_4365_, v___y_4366_, v___y_4367_);
lean_dec(v___y_4367_);
lean_dec_ref(v___y_4366_);
lean_dec(v___y_4365_);
lean_dec_ref(v___y_4364_);
lean_dec_ref(v_range_4361_);
lean_dec_ref(v_args_4359_);
lean_dec_ref(v_fVal_4358_);
return v_res_4369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj(lean_object* v_fVal_4370_, lean_object* v_args_4371_, lean_object* v_m_4372_, lean_object* v_a_4373_, lean_object* v_a_4374_, lean_object* v_a_4375_, lean_object* v_a_4376_){
_start:
{
lean_object* v_numFields_4378_; lean_object* v___x_4379_; lean_object* v___x_4380_; lean_object* v___x_4381_; lean_object* v___x_4382_; lean_object* v___x_4383_; 
v_numFields_4378_ = lean_ctor_get(v_fVal_4370_, 4);
v___x_4379_ = lean_unsigned_to_nat(0u);
v___x_4380_ = lean_unsigned_to_nat(1u);
lean_inc(v_numFields_4378_);
v___x_4381_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4381_, 0, v___x_4379_);
lean_ctor_set(v___x_4381_, 1, v_numFields_4378_);
lean_ctor_set(v___x_4381_, 2, v___x_4380_);
v___x_4382_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg___closed__2));
v___x_4383_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg(v_fVal_4370_, v_args_4371_, v_m_4372_, v___x_4381_, v___x_4382_, v___x_4379_, v_a_4373_, v_a_4374_, v_a_4375_, v_a_4376_);
lean_dec_ref_known(v___x_4381_, 3);
if (lean_obj_tag(v___x_4383_) == 0)
{
lean_object* v_a_4384_; lean_object* v___x_4386_; uint8_t v_isShared_4387_; uint8_t v_isSharedCheck_4397_; 
v_a_4384_ = lean_ctor_get(v___x_4383_, 0);
v_isSharedCheck_4397_ = !lean_is_exclusive(v___x_4383_);
if (v_isSharedCheck_4397_ == 0)
{
v___x_4386_ = v___x_4383_;
v_isShared_4387_ = v_isSharedCheck_4397_;
goto v_resetjp_4385_;
}
else
{
lean_inc(v_a_4384_);
lean_dec(v___x_4383_);
v___x_4386_ = lean_box(0);
v_isShared_4387_ = v_isSharedCheck_4397_;
goto v_resetjp_4385_;
}
v_resetjp_4385_:
{
lean_object* v_fst_4388_; 
v_fst_4388_ = lean_ctor_get(v_a_4384_, 0);
lean_inc(v_fst_4388_);
lean_dec(v_a_4384_);
if (lean_obj_tag(v_fst_4388_) == 0)
{
lean_object* v___x_4389_; lean_object* v___x_4391_; 
v___x_4389_ = lean_box(2);
if (v_isShared_4387_ == 0)
{
lean_ctor_set(v___x_4386_, 0, v___x_4389_);
v___x_4391_ = v___x_4386_;
goto v_reusejp_4390_;
}
else
{
lean_object* v_reuseFailAlloc_4392_; 
v_reuseFailAlloc_4392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4392_, 0, v___x_4389_);
v___x_4391_ = v_reuseFailAlloc_4392_;
goto v_reusejp_4390_;
}
v_reusejp_4390_:
{
return v___x_4391_;
}
}
else
{
lean_object* v_val_4393_; lean_object* v___x_4395_; 
v_val_4393_ = lean_ctor_get(v_fst_4388_, 0);
lean_inc(v_val_4393_);
lean_dec_ref_known(v_fst_4388_, 1);
if (v_isShared_4387_ == 0)
{
lean_ctor_set(v___x_4386_, 0, v_val_4393_);
v___x_4395_ = v___x_4386_;
goto v_reusejp_4394_;
}
else
{
lean_object* v_reuseFailAlloc_4396_; 
v_reuseFailAlloc_4396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4396_, 0, v_val_4393_);
v___x_4395_ = v_reuseFailAlloc_4396_;
goto v_reusejp_4394_;
}
v_reusejp_4394_:
{
return v___x_4395_;
}
}
}
}
else
{
lean_object* v_a_4398_; lean_object* v___x_4400_; uint8_t v_isShared_4401_; uint8_t v_isSharedCheck_4405_; 
v_a_4398_ = lean_ctor_get(v___x_4383_, 0);
v_isSharedCheck_4405_ = !lean_is_exclusive(v___x_4383_);
if (v_isSharedCheck_4405_ == 0)
{
v___x_4400_ = v___x_4383_;
v_isShared_4401_ = v_isSharedCheck_4405_;
goto v_resetjp_4399_;
}
else
{
lean_inc(v_a_4398_);
lean_dec(v___x_4383_);
v___x_4400_ = lean_box(0);
v_isShared_4401_ = v_isSharedCheck_4405_;
goto v_resetjp_4399_;
}
v_resetjp_4399_:
{
lean_object* v___x_4403_; 
if (v_isShared_4401_ == 0)
{
v___x_4403_ = v___x_4400_;
goto v_reusejp_4402_;
}
else
{
lean_object* v_reuseFailAlloc_4404_; 
v_reuseFailAlloc_4404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4404_, 0, v_a_4398_);
v___x_4403_ = v_reuseFailAlloc_4404_;
goto v_reusejp_4402_;
}
v_reusejp_4402_:
{
return v___x_4403_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj___boxed(lean_object* v_fVal_4406_, lean_object* v_args_4407_, lean_object* v_m_4408_, lean_object* v_a_4409_, lean_object* v_a_4410_, lean_object* v_a_4411_, lean_object* v_a_4412_, lean_object* v_a_4413_){
_start:
{
lean_object* v_res_4414_; 
v_res_4414_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj(v_fVal_4406_, v_args_4407_, v_m_4408_, v_a_4409_, v_a_4410_, v_a_4411_, v_a_4412_);
lean_dec(v_a_4412_);
lean_dec_ref(v_a_4411_);
lean_dec(v_a_4410_);
lean_dec_ref(v_a_4409_);
lean_dec_ref(v_args_4407_);
lean_dec_ref(v_fVal_4406_);
return v_res_4414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0(lean_object* v_fVal_4415_, lean_object* v_args_4416_, lean_object* v_m_4417_, lean_object* v_range_4418_, lean_object* v_b_4419_, lean_object* v_i_4420_, lean_object* v_hs_4421_, lean_object* v_hl_4422_, lean_object* v___y_4423_, lean_object* v___y_4424_, lean_object* v___y_4425_, lean_object* v___y_4426_){
_start:
{
lean_object* v___x_4428_; 
v___x_4428_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___redArg(v_fVal_4415_, v_args_4416_, v_m_4417_, v_range_4418_, v_b_4419_, v_i_4420_, v___y_4423_, v___y_4424_, v___y_4425_, v___y_4426_);
return v___x_4428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0___boxed(lean_object* v_fVal_4429_, lean_object* v_args_4430_, lean_object* v_m_4431_, lean_object* v_range_4432_, lean_object* v_b_4433_, lean_object* v_i_4434_, lean_object* v_hs_4435_, lean_object* v_hl_4436_, lean_object* v___y_4437_, lean_object* v___y_4438_, lean_object* v___y_4439_, lean_object* v___y_4440_, lean_object* v___y_4441_){
_start:
{
lean_object* v_res_4442_; 
v_res_4442_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj_spec__0(v_fVal_4429_, v_args_4430_, v_m_4431_, v_range_4432_, v_b_4433_, v_i_4434_, v_hs_4435_, v_hl_4436_, v___y_4437_, v___y_4438_, v___y_4439_, v___y_4440_);
lean_dec(v___y_4440_);
lean_dec_ref(v___y_4439_);
lean_dec(v___y_4438_);
lean_dec_ref(v___y_4437_);
lean_dec_ref(v_range_4432_);
lean_dec_ref(v_args_4430_);
lean_dec_ref(v_fVal_4429_);
return v_res_4442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___lam__0(lean_object* v___y_4443_, lean_object* v___y_4444_, lean_object* v___y_4445_, lean_object* v___y_4446_, lean_object* v___y_4447_){
_start:
{
lean_object* v___x_4449_; 
v___x_4449_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4449_, 0, v___y_4443_);
return v___x_4449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___lam__0___boxed(lean_object* v___y_4450_, lean_object* v___y_4451_, lean_object* v___y_4452_, lean_object* v___y_4453_, lean_object* v___y_4454_, lean_object* v___y_4455_){
_start:
{
lean_object* v_res_4456_; 
v_res_4456_ = lp_mathlib_Mathlib_Tactic_etaStruct_x3f___lam__0(v___y_4450_, v___y_4451_, v___y_4452_, v___y_4453_, v___y_4454_);
lean_dec(v___y_4454_);
lean_dec_ref(v___y_4453_);
lean_dec(v___y_4452_);
lean_dec_ref(v___y_4451_);
return v_res_4456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f(lean_object* v_e_4459_, uint8_t v_tryWhnfR_4460_, lean_object* v_a_4461_, lean_object* v_a_4462_, lean_object* v_a_4463_, lean_object* v_a_4464_){
_start:
{
lean_object* v_x_x3f_4470_; lean_object* v___y_4471_; lean_object* v___y_4472_; lean_object* v___y_4473_; lean_object* v___y_4474_; lean_object* v___x_4504_; 
v___x_4504_ = l_Lean_Expr_getAppFn(v_e_4459_);
if (lean_obj_tag(v___x_4504_) == 4)
{
lean_object* v_declName_4505_; lean_object* v___x_4506_; lean_object* v_env_4507_; uint8_t v___x_4508_; lean_object* v___x_4509_; 
v_declName_4505_ = lean_ctor_get(v___x_4504_, 0);
lean_inc(v_declName_4505_);
lean_dec_ref_known(v___x_4504_, 2);
v___x_4506_ = lean_st_ref_get(v_a_4464_);
v_env_4507_ = lean_ctor_get(v___x_4506_, 0);
lean_inc_ref(v_env_4507_);
lean_dec(v___x_4506_);
v___x_4508_ = 0;
v___x_4509_ = l_Lean_Environment_find_x3f(v_env_4507_, v_declName_4505_, v___x_4508_);
if (lean_obj_tag(v___x_4509_) == 1)
{
lean_object* v_val_4510_; 
v_val_4510_ = lean_ctor_get(v___x_4509_, 0);
lean_inc(v_val_4510_);
lean_dec_ref_known(v___x_4509_, 1);
if (lean_obj_tag(v_val_4510_) == 6)
{
lean_object* v_val_4511_; lean_object* v___x_4513_; uint8_t v_isShared_4514_; uint8_t v_isSharedCheck_4565_; 
v_val_4511_ = lean_ctor_get(v_val_4510_, 0);
v_isSharedCheck_4565_ = !lean_is_exclusive(v_val_4510_);
if (v_isSharedCheck_4565_ == 0)
{
v___x_4513_ = v_val_4510_;
v_isShared_4514_ = v_isSharedCheck_4565_;
goto v_resetjp_4512_;
}
else
{
lean_inc(v_val_4511_);
lean_dec(v_val_4510_);
v___x_4513_ = lean_box(0);
v_isShared_4514_ = v_isSharedCheck_4565_;
goto v_resetjp_4512_;
}
v_resetjp_4512_:
{
lean_object* v_induct_4515_; lean_object* v_numParams_4516_; lean_object* v_numFields_4517_; lean_object* v___f_4518_; uint8_t v___y_4520_; lean_object* v___x_4560_; uint8_t v___x_4561_; 
v_induct_4515_ = lean_ctor_get(v_val_4511_, 1);
v_numParams_4516_ = lean_ctor_get(v_val_4511_, 3);
v_numFields_4517_ = lean_ctor_get(v_val_4511_, 4);
v___f_4518_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__0));
v___x_4560_ = lean_unsigned_to_nat(0u);
v___x_4561_ = lean_nat_dec_lt(v___x_4560_, v_numFields_4517_);
if (v___x_4561_ == 0)
{
v___y_4520_ = v___x_4561_;
goto v___jp_4519_;
}
else
{
lean_object* v___x_4562_; lean_object* v___x_4563_; uint8_t v___x_4564_; 
v___x_4562_ = l_Lean_Expr_getAppNumArgs(v_e_4459_);
v___x_4563_ = lean_nat_add(v_numParams_4516_, v_numFields_4517_);
v___x_4564_ = lean_nat_dec_eq(v___x_4562_, v___x_4563_);
lean_dec(v___x_4563_);
lean_dec(v___x_4562_);
v___y_4520_ = v___x_4564_;
goto v___jp_4519_;
}
v___jp_4519_:
{
if (v___y_4520_ == 0)
{
lean_object* v___x_4521_; lean_object* v___x_4523_; 
lean_dec_ref(v_val_4511_);
lean_dec_ref(v_e_4459_);
v___x_4521_ = lean_box(0);
if (v_isShared_4514_ == 0)
{
lean_ctor_set_tag(v___x_4513_, 0);
lean_ctor_set(v___x_4513_, 0, v___x_4521_);
v___x_4523_ = v___x_4513_;
goto v_reusejp_4522_;
}
else
{
lean_object* v_reuseFailAlloc_4524_; 
v_reuseFailAlloc_4524_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4524_, 0, v___x_4521_);
v___x_4523_ = v_reuseFailAlloc_4524_;
goto v_reusejp_4522_;
}
v_reusejp_4522_:
{
return v___x_4523_;
}
}
else
{
lean_object* v___x_4525_; lean_object* v_env_4526_; uint8_t v___x_4527_; 
v___x_4525_ = lean_st_ref_get(v_a_4464_);
v_env_4526_ = lean_ctor_get(v___x_4525_, 0);
lean_inc_ref(v_env_4526_);
lean_dec(v___x_4525_);
lean_inc(v_induct_4515_);
v___x_4527_ = l_Lean_isStructure(v_env_4526_, v_induct_4515_);
if (v___x_4527_ == 0)
{
lean_object* v___x_4528_; lean_object* v___x_4530_; 
lean_dec_ref(v_val_4511_);
lean_dec_ref(v_e_4459_);
v___x_4528_ = lean_box(0);
if (v_isShared_4514_ == 0)
{
lean_ctor_set_tag(v___x_4513_, 0);
lean_ctor_set(v___x_4513_, 0, v___x_4528_);
v___x_4530_ = v___x_4513_;
goto v_reusejp_4529_;
}
else
{
lean_object* v_reuseFailAlloc_4531_; 
v_reuseFailAlloc_4531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4531_, 0, v___x_4528_);
v___x_4530_ = v_reuseFailAlloc_4531_;
goto v_reusejp_4529_;
}
v_reusejp_4529_:
{
return v___x_4530_;
}
}
else
{
lean_object* v_dummy_4532_; lean_object* v_nargs_4533_; lean_object* v___x_4534_; lean_object* v___x_4535_; lean_object* v___x_4536_; lean_object* v___x_4537_; lean_object* v___x_4538_; 
lean_del_object(v___x_4513_);
v_dummy_4532_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1, &lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1_spec__2___lam__1___closed__1);
v_nargs_4533_ = l_Lean_Expr_getAppNumArgs(v_e_4459_);
lean_inc(v_nargs_4533_);
v___x_4534_ = lean_mk_array(v_nargs_4533_, v_dummy_4532_);
v___x_4535_ = lean_unsigned_to_nat(1u);
v___x_4536_ = lean_nat_sub(v_nargs_4533_, v___x_4535_);
lean_dec(v_nargs_4533_);
lean_inc_ref(v_e_4459_);
v___x_4537_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_e_4459_, v___x_4534_, v___x_4536_);
v___x_4538_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj(v_val_4511_, v___x_4537_, v___f_4518_, v_a_4461_, v_a_4462_, v_a_4463_, v_a_4464_);
if (lean_obj_tag(v___x_4538_) == 0)
{
if (v_tryWhnfR_4460_ == 0)
{
lean_object* v_a_4539_; 
lean_dec_ref(v___x_4537_);
lean_dec_ref(v_val_4511_);
v_a_4539_ = lean_ctor_get(v___x_4538_, 0);
lean_inc(v_a_4539_);
lean_dec_ref_known(v___x_4538_, 1);
v_x_x3f_4470_ = v_a_4539_;
v___y_4471_ = v_a_4461_;
v___y_4472_ = v_a_4462_;
v___y_4473_ = v_a_4463_;
v___y_4474_ = v_a_4464_;
goto v___jp_4469_;
}
else
{
lean_object* v_a_4540_; 
v_a_4540_ = lean_ctor_get(v___x_4538_, 0);
lean_inc(v_a_4540_);
lean_dec_ref_known(v___x_4538_, 1);
if (lean_obj_tag(v_a_4540_) == 2)
{
lean_object* v___x_4541_; lean_object* v___x_4542_; 
v___x_4541_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaStruct_x3f___closed__1));
v___x_4542_ = lp_mathlib___private_Mathlib_Tactic_DefEqTransformations_0__Mathlib_Tactic_etaStruct_x3f_findProj(v_val_4511_, v___x_4537_, v___x_4541_, v_a_4461_, v_a_4462_, v_a_4463_, v_a_4464_);
lean_dec_ref(v___x_4537_);
lean_dec_ref(v_val_4511_);
if (lean_obj_tag(v___x_4542_) == 0)
{
lean_object* v_a_4543_; 
v_a_4543_ = lean_ctor_get(v___x_4542_, 0);
lean_inc(v_a_4543_);
lean_dec_ref_known(v___x_4542_, 1);
v_x_x3f_4470_ = v_a_4543_;
v___y_4471_ = v_a_4461_;
v___y_4472_ = v_a_4462_;
v___y_4473_ = v_a_4463_;
v___y_4474_ = v_a_4464_;
goto v___jp_4469_;
}
else
{
lean_object* v_a_4544_; lean_object* v___x_4546_; uint8_t v_isShared_4547_; uint8_t v_isSharedCheck_4551_; 
lean_dec_ref(v_e_4459_);
v_a_4544_ = lean_ctor_get(v___x_4542_, 0);
v_isSharedCheck_4551_ = !lean_is_exclusive(v___x_4542_);
if (v_isSharedCheck_4551_ == 0)
{
v___x_4546_ = v___x_4542_;
v_isShared_4547_ = v_isSharedCheck_4551_;
goto v_resetjp_4545_;
}
else
{
lean_inc(v_a_4544_);
lean_dec(v___x_4542_);
v___x_4546_ = lean_box(0);
v_isShared_4547_ = v_isSharedCheck_4551_;
goto v_resetjp_4545_;
}
v_resetjp_4545_:
{
lean_object* v___x_4549_; 
if (v_isShared_4547_ == 0)
{
v___x_4549_ = v___x_4546_;
goto v_reusejp_4548_;
}
else
{
lean_object* v_reuseFailAlloc_4550_; 
v_reuseFailAlloc_4550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4550_, 0, v_a_4544_);
v___x_4549_ = v_reuseFailAlloc_4550_;
goto v_reusejp_4548_;
}
v_reusejp_4548_:
{
return v___x_4549_;
}
}
}
}
else
{
lean_dec_ref(v___x_4537_);
lean_dec_ref(v_val_4511_);
v_x_x3f_4470_ = v_a_4540_;
v___y_4471_ = v_a_4461_;
v___y_4472_ = v_a_4462_;
v___y_4473_ = v_a_4463_;
v___y_4474_ = v_a_4464_;
goto v___jp_4469_;
}
}
}
else
{
lean_object* v_a_4552_; lean_object* v___x_4554_; uint8_t v_isShared_4555_; uint8_t v_isSharedCheck_4559_; 
lean_dec_ref(v___x_4537_);
lean_dec_ref(v_val_4511_);
lean_dec_ref(v_e_4459_);
v_a_4552_ = lean_ctor_get(v___x_4538_, 0);
v_isSharedCheck_4559_ = !lean_is_exclusive(v___x_4538_);
if (v_isSharedCheck_4559_ == 0)
{
v___x_4554_ = v___x_4538_;
v_isShared_4555_ = v_isSharedCheck_4559_;
goto v_resetjp_4553_;
}
else
{
lean_inc(v_a_4552_);
lean_dec(v___x_4538_);
v___x_4554_ = lean_box(0);
v_isShared_4555_ = v_isSharedCheck_4559_;
goto v_resetjp_4553_;
}
v_resetjp_4553_:
{
lean_object* v___x_4557_; 
if (v_isShared_4555_ == 0)
{
v___x_4557_ = v___x_4554_;
goto v_reusejp_4556_;
}
else
{
lean_object* v_reuseFailAlloc_4558_; 
v_reuseFailAlloc_4558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4558_, 0, v_a_4552_);
v___x_4557_ = v_reuseFailAlloc_4558_;
goto v_reusejp_4556_;
}
v_reusejp_4556_:
{
return v___x_4557_;
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
lean_dec(v_val_4510_);
lean_dec_ref(v_e_4459_);
goto v___jp_4501_;
}
}
else
{
lean_dec(v___x_4509_);
lean_dec_ref(v_e_4459_);
goto v___jp_4501_;
}
}
else
{
lean_object* v___x_4566_; lean_object* v___x_4567_; 
lean_dec_ref(v___x_4504_);
lean_dec_ref(v_e_4459_);
v___x_4566_ = lean_box(0);
v___x_4567_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4567_, 0, v___x_4566_);
return v___x_4567_;
}
v___jp_4466_:
{
lean_object* v___x_4467_; lean_object* v___x_4468_; 
v___x_4467_ = lean_box(0);
v___x_4468_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4468_, 0, v___x_4467_);
return v___x_4468_;
}
v___jp_4469_:
{
if (lean_obj_tag(v_x_x3f_4470_) == 1)
{
lean_object* v_a_4475_; lean_object* v___x_4477_; uint8_t v_isShared_4478_; uint8_t v_isSharedCheck_4500_; 
v_a_4475_ = lean_ctor_get(v_x_x3f_4470_, 0);
v_isSharedCheck_4500_ = !lean_is_exclusive(v_x_x3f_4470_);
if (v_isSharedCheck_4500_ == 0)
{
v___x_4477_ = v_x_x3f_4470_;
v_isShared_4478_ = v_isSharedCheck_4500_;
goto v_resetjp_4476_;
}
else
{
lean_inc(v_a_4475_);
lean_dec(v_x_x3f_4470_);
v___x_4477_ = lean_box(0);
v_isShared_4478_ = v_isSharedCheck_4500_;
goto v_resetjp_4476_;
}
v_resetjp_4476_:
{
lean_object* v___x_4479_; 
lean_inc(v_a_4475_);
v___x_4479_ = l_Lean_Meta_isExprDefEq(v_a_4475_, v_e_4459_, v___y_4471_, v___y_4472_, v___y_4473_, v___y_4474_);
if (lean_obj_tag(v___x_4479_) == 0)
{
lean_object* v_a_4480_; lean_object* v___x_4482_; uint8_t v_isShared_4483_; uint8_t v_isSharedCheck_4491_; 
v_a_4480_ = lean_ctor_get(v___x_4479_, 0);
v_isSharedCheck_4491_ = !lean_is_exclusive(v___x_4479_);
if (v_isSharedCheck_4491_ == 0)
{
v___x_4482_ = v___x_4479_;
v_isShared_4483_ = v_isSharedCheck_4491_;
goto v_resetjp_4481_;
}
else
{
lean_inc(v_a_4480_);
lean_dec(v___x_4479_);
v___x_4482_ = lean_box(0);
v_isShared_4483_ = v_isSharedCheck_4491_;
goto v_resetjp_4481_;
}
v_resetjp_4481_:
{
uint8_t v___x_4484_; 
v___x_4484_ = lean_unbox(v_a_4480_);
lean_dec(v_a_4480_);
if (v___x_4484_ == 0)
{
lean_del_object(v___x_4482_);
lean_del_object(v___x_4477_);
lean_dec(v_a_4475_);
goto v___jp_4466_;
}
else
{
lean_object* v___x_4486_; 
if (v_isShared_4478_ == 0)
{
v___x_4486_ = v___x_4477_;
goto v_reusejp_4485_;
}
else
{
lean_object* v_reuseFailAlloc_4490_; 
v_reuseFailAlloc_4490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4490_, 0, v_a_4475_);
v___x_4486_ = v_reuseFailAlloc_4490_;
goto v_reusejp_4485_;
}
v_reusejp_4485_:
{
lean_object* v___x_4488_; 
if (v_isShared_4483_ == 0)
{
lean_ctor_set(v___x_4482_, 0, v___x_4486_);
v___x_4488_ = v___x_4482_;
goto v_reusejp_4487_;
}
else
{
lean_object* v_reuseFailAlloc_4489_; 
v_reuseFailAlloc_4489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4489_, 0, v___x_4486_);
v___x_4488_ = v_reuseFailAlloc_4489_;
goto v_reusejp_4487_;
}
v_reusejp_4487_:
{
return v___x_4488_;
}
}
}
}
}
else
{
lean_object* v_a_4492_; lean_object* v___x_4494_; uint8_t v_isShared_4495_; uint8_t v_isSharedCheck_4499_; 
lean_del_object(v___x_4477_);
lean_dec(v_a_4475_);
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
}
else
{
lean_dec(v_x_x3f_4470_);
lean_dec_ref(v_e_4459_);
goto v___jp_4466_;
}
}
v___jp_4501_:
{
lean_object* v___x_4502_; lean_object* v___x_4503_; 
v___x_4502_ = lean_box(0);
v___x_4503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4503_, 0, v___x_4502_);
return v___x_4503_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStruct_x3f___boxed(lean_object* v_e_4568_, lean_object* v_tryWhnfR_4569_, lean_object* v_a_4570_, lean_object* v_a_4571_, lean_object* v_a_4572_, lean_object* v_a_4573_, lean_object* v_a_4574_){
_start:
{
uint8_t v_tryWhnfR_boxed_4575_; lean_object* v_res_4576_; 
v_tryWhnfR_boxed_4575_ = lean_unbox(v_tryWhnfR_4569_);
v_res_4576_ = lp_mathlib_Mathlib_Tactic_etaStruct_x3f(v_e_4568_, v_tryWhnfR_boxed_4575_, v_a_4570_, v_a_4571_, v_a_4572_, v_a_4573_);
lean_dec(v_a_4573_);
lean_dec_ref(v_a_4572_);
lean_dec(v_a_4571_);
lean_dec_ref(v_a_4570_);
return v_res_4576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll___lam__0(lean_object* v_node_4577_, lean_object* v___y_4578_, lean_object* v___y_4579_, lean_object* v___y_4580_, lean_object* v___y_4581_){
_start:
{
uint8_t v___x_4583_; lean_object* v___x_4584_; 
v___x_4583_ = 1;
v___x_4584_ = lp_mathlib_Mathlib_Tactic_etaStruct_x3f(v_node_4577_, v___x_4583_, v___y_4578_, v___y_4579_, v___y_4580_, v___y_4581_);
if (lean_obj_tag(v___x_4584_) == 0)
{
lean_object* v_a_4585_; lean_object* v___x_4587_; uint8_t v_isShared_4588_; uint8_t v_isSharedCheck_4604_; 
v_a_4585_ = lean_ctor_get(v___x_4584_, 0);
v_isSharedCheck_4604_ = !lean_is_exclusive(v___x_4584_);
if (v_isSharedCheck_4604_ == 0)
{
v___x_4587_ = v___x_4584_;
v_isShared_4588_ = v_isSharedCheck_4604_;
goto v_resetjp_4586_;
}
else
{
lean_inc(v_a_4585_);
lean_dec(v___x_4584_);
v___x_4587_ = lean_box(0);
v_isShared_4588_ = v_isSharedCheck_4604_;
goto v_resetjp_4586_;
}
v_resetjp_4586_:
{
if (lean_obj_tag(v_a_4585_) == 1)
{
lean_object* v_val_4589_; lean_object* v___x_4591_; uint8_t v_isShared_4592_; uint8_t v_isSharedCheck_4599_; 
v_val_4589_ = lean_ctor_get(v_a_4585_, 0);
v_isSharedCheck_4599_ = !lean_is_exclusive(v_a_4585_);
if (v_isSharedCheck_4599_ == 0)
{
v___x_4591_ = v_a_4585_;
v_isShared_4592_ = v_isSharedCheck_4599_;
goto v_resetjp_4590_;
}
else
{
lean_inc(v_val_4589_);
lean_dec(v_a_4585_);
v___x_4591_ = lean_box(0);
v_isShared_4592_ = v_isSharedCheck_4599_;
goto v_resetjp_4590_;
}
v_resetjp_4590_:
{
lean_object* v___x_4594_; 
if (v_isShared_4592_ == 0)
{
v___x_4594_ = v___x_4591_;
goto v_reusejp_4593_;
}
else
{
lean_object* v_reuseFailAlloc_4598_; 
v_reuseFailAlloc_4598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4598_, 0, v_val_4589_);
v___x_4594_ = v_reuseFailAlloc_4598_;
goto v_reusejp_4593_;
}
v_reusejp_4593_:
{
lean_object* v___x_4596_; 
if (v_isShared_4588_ == 0)
{
lean_ctor_set(v___x_4587_, 0, v___x_4594_);
v___x_4596_ = v___x_4587_;
goto v_reusejp_4595_;
}
else
{
lean_object* v_reuseFailAlloc_4597_; 
v_reuseFailAlloc_4597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4597_, 0, v___x_4594_);
v___x_4596_ = v_reuseFailAlloc_4597_;
goto v_reusejp_4595_;
}
v_reusejp_4595_:
{
return v___x_4596_;
}
}
}
}
else
{
lean_object* v___x_4600_; lean_object* v___x_4602_; 
lean_dec(v_a_4585_);
v___x_4600_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___lam__1___closed__0));
if (v_isShared_4588_ == 0)
{
lean_ctor_set(v___x_4587_, 0, v___x_4600_);
v___x_4602_ = v___x_4587_;
goto v_reusejp_4601_;
}
else
{
lean_object* v_reuseFailAlloc_4603_; 
v_reuseFailAlloc_4603_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4603_, 0, v___x_4600_);
v___x_4602_ = v_reuseFailAlloc_4603_;
goto v_reusejp_4601_;
}
v_reusejp_4601_:
{
return v___x_4602_;
}
}
}
}
else
{
lean_object* v_a_4605_; lean_object* v___x_4607_; uint8_t v_isShared_4608_; uint8_t v_isSharedCheck_4612_; 
v_a_4605_ = lean_ctor_get(v___x_4584_, 0);
v_isSharedCheck_4612_ = !lean_is_exclusive(v___x_4584_);
if (v_isSharedCheck_4612_ == 0)
{
v___x_4607_ = v___x_4584_;
v_isShared_4608_ = v_isSharedCheck_4612_;
goto v_resetjp_4606_;
}
else
{
lean_inc(v_a_4605_);
lean_dec(v___x_4584_);
v___x_4607_ = lean_box(0);
v_isShared_4608_ = v_isSharedCheck_4612_;
goto v_resetjp_4606_;
}
v_resetjp_4606_:
{
lean_object* v___x_4610_; 
if (v_isShared_4608_ == 0)
{
v___x_4610_ = v___x_4607_;
goto v_reusejp_4609_;
}
else
{
lean_object* v_reuseFailAlloc_4611_; 
v_reuseFailAlloc_4611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4611_, 0, v_a_4605_);
v___x_4610_ = v_reuseFailAlloc_4611_;
goto v_reusejp_4609_;
}
v_reusejp_4609_:
{
return v___x_4610_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll___lam__0___boxed(lean_object* v_node_4613_, lean_object* v___y_4614_, lean_object* v___y_4615_, lean_object* v___y_4616_, lean_object* v___y_4617_, lean_object* v___y_4618_){
_start:
{
lean_object* v_res_4619_; 
v_res_4619_ = lp_mathlib_Mathlib_Tactic_etaStructAll___lam__0(v_node_4613_, v___y_4614_, v___y_4615_, v___y_4616_, v___y_4617_);
lean_dec(v___y_4617_);
lean_dec_ref(v___y_4616_);
lean_dec(v___y_4615_);
lean_dec_ref(v___y_4614_);
return v_res_4619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll(lean_object* v_e_4621_, lean_object* v_a_4622_, lean_object* v_a_4623_, lean_object* v_a_4624_, lean_object* v_a_4625_){
_start:
{
lean_object* v___f_4627_; lean_object* v___f_4628_; uint8_t v___x_4629_; lean_object* v___x_4630_; 
v___f_4627_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaStructAll___closed__0));
v___f_4628_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_unfoldFVars___closed__0));
v___x_4629_ = 0;
v___x_4630_ = lp_mathlib_Lean_Meta_transform___at___00Mathlib_Tactic_unfoldFVars_spec__1(v_e_4621_, v___f_4627_, v___f_4628_, v___x_4629_, v___x_4629_, v_a_4622_, v_a_4623_, v_a_4624_, v_a_4625_);
return v___x_4630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_etaStructAll___boxed(lean_object* v_e_4631_, lean_object* v_a_4632_, lean_object* v_a_4633_, lean_object* v_a_4634_, lean_object* v_a_4635_, lean_object* v_a_4636_){
_start:
{
lean_object* v_res_4637_; 
v_res_4637_ = lp_mathlib_Mathlib_Tactic_etaStructAll(v_e_4631_, v_a_4632_, v_a_4633_, v_a_4634_, v_a_4635_);
lean_dec(v_a_4635_);
lean_dec_ref(v_a_4634_);
lean_dec(v_a_4633_);
lean_dec_ref(v_a_4632_);
return v_res_4637_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaStructStx___closed__4(void){
_start:
{
lean_object* v___x_4647_; lean_object* v___x_4648_; lean_object* v___x_4649_; lean_object* v___x_4650_; 
v___x_4647_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14, &lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14_once, _init_lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__14);
v___x_4648_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaStructStx___closed__3));
v___x_4649_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticWhnf_____00__closed__5));
v___x_4650_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4650_, 0, v___x_4649_);
lean_ctor_set(v___x_4650_, 1, v___x_4648_);
lean_ctor_set(v___x_4650_, 2, v___x_4647_);
return v___x_4650_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaStructStx___closed__5(void){
_start:
{
lean_object* v___x_4651_; lean_object* v___x_4652_; lean_object* v___x_4653_; lean_object* v___x_4654_; 
v___x_4651_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__4, &lp_mathlib_Mathlib_Tactic_etaStructStx___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_etaStructStx___closed__4);
v___x_4652_ = lean_unsigned_to_nat(1022u);
v___x_4653_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1));
v___x_4654_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4654_, 0, v___x_4653_);
lean_ctor_set(v___x_4654_, 1, v___x_4652_);
lean_ctor_set(v___x_4654_, 2, v___x_4651_);
return v___x_4654_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_etaStructStx(void){
_start:
{
lean_object* v___x_4655_; 
v___x_4655_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_etaStructStx___closed__5, &lp_mathlib_Mathlib_Tactic_etaStructStx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_etaStructStx___closed__5);
return v___x_4655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___lam__0(lean_object* v_x_4656_, lean_object* v___y_4657_, lean_object* v___y_4658_, lean_object* v___y_4659_, lean_object* v___y_4660_, lean_object* v___y_4661_){
_start:
{
lean_object* v___x_4663_; 
v___x_4663_ = lp_mathlib_Mathlib_Tactic_etaStructAll(v___y_4657_, v___y_4658_, v___y_4659_, v___y_4660_, v___y_4661_);
return v___x_4663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___lam__0___boxed(lean_object* v_x_4664_, lean_object* v___y_4665_, lean_object* v___y_4666_, lean_object* v___y_4667_, lean_object* v___y_4668_, lean_object* v___y_4669_, lean_object* v___y_4670_){
_start:
{
lean_object* v_res_4671_; 
v_res_4671_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___lam__0(v_x_4664_, v___y_4665_, v___y_4666_, v___y_4667_, v___y_4668_, v___y_4669_);
lean_dec(v___y_4669_);
lean_dec_ref(v___y_4668_);
lean_dec(v___y_4667_);
lean_dec_ref(v___y_4666_);
lean_dec(v_x_4664_);
return v_res_4671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1(lean_object* v_x_4673_, lean_object* v_a_4674_, lean_object* v_a_4675_, lean_object* v_a_4676_, lean_object* v_a_4677_, lean_object* v_a_4678_, lean_object* v_a_4679_, lean_object* v_a_4680_, lean_object* v_a_4681_){
_start:
{
lean_object* v___x_4683_; uint8_t v___x_4684_; 
v___x_4683_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaStructStx___closed__1));
lean_inc(v_x_4673_);
v___x_4684_ = l_Lean_Syntax_isOfKind(v_x_4673_, v___x_4683_);
if (v___x_4684_ == 0)
{
lean_object* v___x_4685_; 
lean_dec(v_x_4673_);
v___x_4685_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_4685_;
}
else
{
lean_object* v___f_4686_; lean_object* v___y_4688_; lean_object* v___x_4691_; lean_object* v___x_4692_; lean_object* v___x_4693_; 
v___f_4686_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___closed__0));
v___x_4691_ = lean_unsigned_to_nat(1u);
v___x_4692_ = l_Lean_Syntax_getArg(v_x_4673_, v___x_4691_);
lean_dec(v_x_4673_);
v___x_4693_ = l_Lean_Syntax_getOptional_x3f(v___x_4692_);
lean_dec(v___x_4692_);
if (lean_obj_tag(v___x_4693_) == 0)
{
lean_object* v___x_4694_; 
v___x_4694_ = lean_box(0);
v___y_4688_ = v___x_4694_;
goto v___jp_4687_;
}
else
{
lean_object* v_val_4695_; lean_object* v___x_4697_; uint8_t v_isShared_4698_; uint8_t v_isSharedCheck_4702_; 
v_val_4695_ = lean_ctor_get(v___x_4693_, 0);
v_isSharedCheck_4702_ = !lean_is_exclusive(v___x_4693_);
if (v_isSharedCheck_4702_ == 0)
{
v___x_4697_ = v___x_4693_;
v_isShared_4698_ = v_isSharedCheck_4702_;
goto v_resetjp_4696_;
}
else
{
lean_inc(v_val_4695_);
lean_dec(v___x_4693_);
v___x_4697_ = lean_box(0);
v_isShared_4698_ = v_isSharedCheck_4702_;
goto v_resetjp_4696_;
}
v_resetjp_4696_:
{
lean_object* v___x_4700_; 
if (v_isShared_4698_ == 0)
{
v___x_4700_ = v___x_4697_;
goto v_reusejp_4699_;
}
else
{
lean_object* v_reuseFailAlloc_4701_; 
v_reuseFailAlloc_4701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4701_, 0, v_val_4695_);
v___x_4700_ = v_reuseFailAlloc_4701_;
goto v_reusejp_4699_;
}
v_reusejp_4699_:
{
v___y_4688_ = v___x_4700_;
goto v___jp_4687_;
}
}
}
v___jp_4687_:
{
lean_object* v___x_4689_; lean_object* v___x_4690_; 
v___x_4689_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_etaStructStx___closed__2));
v___x_4690_ = lp_mathlib_Mathlib_Tactic_runDefEqTactic(v___f_4686_, v___y_4688_, v___x_4689_, v___x_4684_, v_a_4674_, v_a_4675_, v_a_4676_, v_a_4677_, v_a_4678_, v_a_4679_, v_a_4680_, v_a_4681_);
return v___x_4690_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1___boxed(lean_object* v_x_4703_, lean_object* v_a_4704_, lean_object* v_a_4705_, lean_object* v_a_4706_, lean_object* v_a_4707_, lean_object* v_a_4708_, lean_object* v_a_4709_, lean_object* v_a_4710_, lean_object* v_a_4711_, lean_object* v_a_4712_){
_start:
{
lean_object* v_res_4713_; 
v_res_4713_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__etaStructStx__1(v_x_4703_, v_a_4704_, v_a_4705_, v_a_4706_, v_a_4707_, v_a_4708_, v_a_4709_, v_a_4710_, v_a_4711_);
lean_dec(v_a_4711_);
lean_dec_ref(v_a_4710_);
lean_dec(v_a_4709_);
lean_dec_ref(v_a_4708_);
lean_dec(v_a_4707_);
lean_dec_ref(v_a_4706_);
lean_dec(v_a_4705_);
lean_dec_ref(v_a_4704_);
return v_res_4713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1(lean_object* v_x_4725_, lean_object* v_a_4726_, lean_object* v_a_4727_, lean_object* v_a_4728_, lean_object* v_a_4729_, lean_object* v_a_4730_, lean_object* v_a_4731_, lean_object* v_a_4732_, lean_object* v_a_4733_){
_start:
{
lean_object* v___x_4735_; uint8_t v___x_4736_; 
v___x_4735_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_convEta__struct___closed__1));
v___x_4736_ = l_Lean_Syntax_isOfKind(v_x_4725_, v___x_4735_);
if (v___x_4736_ == 0)
{
lean_object* v___x_4737_; 
v___x_4737_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__tacticWhnf______1_spec__0___redArg();
return v___x_4737_;
}
else
{
lean_object* v___x_4738_; lean_object* v___x_4739_; 
v___x_4738_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1___closed__0));
v___x_4739_ = lp_mathlib_Mathlib_Tactic_runDefEqConvTactic(v___x_4738_, v_a_4726_, v_a_4727_, v_a_4728_, v_a_4729_, v_a_4730_, v_a_4731_, v_a_4732_, v_a_4733_);
return v___x_4739_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1___boxed(lean_object* v_x_4740_, lean_object* v_a_4741_, lean_object* v_a_4742_, lean_object* v_a_4743_, lean_object* v_a_4744_, lean_object* v_a_4745_, lean_object* v_a_4746_, lean_object* v_a_4747_, lean_object* v_a_4748_, lean_object* v_a_4749_){
_start:
{
lean_object* v_res_4750_; 
v_res_4750_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__DefEqTransformations______elabRules__Mathlib__Tactic__convEta__struct__1(v_x_4740_, v_a_4741_, v_a_4742_, v_a_4743_, v_a_4744_, v_a_4745_, v_a_4746_, v_a_4747_, v_a_4748_);
lean_dec(v_a_4748_);
lean_dec_ref(v_a_4747_);
lean_dec(v_a_4746_);
lean_dec_ref(v_a_4745_);
lean_dec(v_a_4744_);
lean_dec_ref(v_a_4743_);
lean_dec(v_a_4742_);
lean_dec_ref(v_a_4741_);
return v_res_4750_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DefEqTransformations(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_Conv_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_DefEqTransformations(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Conv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_tacticWhnf____ = _init_lp_mathlib_Mathlib_Tactic_tacticWhnf____();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticWhnf____);
lp_mathlib_Mathlib_Tactic_betaReduceStx = _init_lp_mathlib_Mathlib_Tactic_betaReduceStx();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_betaReduceStx);
lp_mathlib_Mathlib_Tactic_tacticReduce____ = _init_lp_mathlib_Mathlib_Tactic_tacticReduce____();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticReduce____);
lp_mathlib_Mathlib_Tactic_refoldLetStx = _init_lp_mathlib_Mathlib_Tactic_refoldLetStx();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_refoldLetStx);
lp_mathlib_Mathlib_Tactic_unfoldProjsStx = _init_lp_mathlib_Mathlib_Tactic_unfoldProjsStx();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_unfoldProjsStx);
lp_mathlib_Mathlib_Tactic_etaReduceStx = _init_lp_mathlib_Mathlib_Tactic_etaReduceStx();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_etaReduceStx);
lp_mathlib_Mathlib_Tactic_etaExpandStx = _init_lp_mathlib_Mathlib_Tactic_etaExpandStx();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_etaExpandStx);
lp_mathlib_Mathlib_Tactic_etaStructStx = _init_lp_mathlib_Mathlib_Tactic_etaStructStx();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_etaStructStx);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Conv_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_DefEqTransformations(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Conv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DefEqTransformations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_DefEqTransformations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_DefEqTransformations(builtin);
}
#ifdef __cplusplus
}
#endif
