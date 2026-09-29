// Lean compiler output
// Module: Mathlib.Tactic.TermCongr
// Imports: public import Init public meta import Init public import Mathlib.Lean.Meta.CongrTheorems
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_KVMap_find(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_find_expr(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Meta_isRefl_x3f(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkSort(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkPropExt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* lp_mathlib_Lean_Expr_sides_x3f(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkApp4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqRefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_mkImpCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingName_x21(lean_object*);
uint8_t l_Lean_Expr_bindingInfo_x21(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
uint8_t l_Lean_Expr_isArrow(lean_object*);
lean_object* l_Lean_Expr_letBody_x21(lean_object*);
lean_object* l_Lean_Expr_letValue_x21(lean_object*);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getBoundedAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_extract___redArg(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_replace_expr(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkCongrFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkHEqRefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkHEqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_mkHCongrWithArity_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Meta_mkCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedParamInfo_default;
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint8_t l_Lean_Meta_TransparencyMode_lt(uint8_t, uint8_t);
lean_object* l_Lean_Meta_FunInfo_getArity(lean_object*);
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Expr_eqv___boxed(lean_object*, lean_object*);
lean_object* l_instBEqProd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_hash___boxed(lean_object*);
lean_object* l_instHashableProd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MonadCacheT_instMonad___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshTypeMVar(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_empty;
lean_object* l_Lean_KVMap_insert(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getCanonicalAntiquot(lean_object*);
lean_object* l_Lean_Syntax_getAntiquotTerm(lean_object*);
uint8_t l_Lean_Syntax_isAntiquots(lean_object*);
lean_object* l_Lean_Syntax_antiquotKinds(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_instInhabitedTermElabM(lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkExpectedTypeHint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isEq(lean_object*);
uint8_t l_Lean_Expr_isHEq(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__0_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__0_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__0_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__1_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "congr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__1_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__1_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__0_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 84, 199, 228, 250, 36, 60, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__1_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(134, 149, 36, 3, 10, 102, 232, 229)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__3_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__3_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__3_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__4_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__3_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__4_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__4_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__6_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__4_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__6_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__6_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__8_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__6_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__8_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__8_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "TermCongr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__10_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__8_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(84, 54, 17, 2, 45, 226, 46, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__10_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__10_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__11_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__10_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(85, 228, 121, 16, 78, 16, 159, 167)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__11_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__11_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__12_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__11_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(176, 167, 37, 119, 104, 184, 176, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__12_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__12_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__13_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__12_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(205, 229, 49, 112, 111, 129, 58, 157)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__13_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__13_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__14_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__13_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(174, 25, 212, 209, 33, 83, 154, 227)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__14_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__14_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__15_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__15_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__15_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__16_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__14_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__15_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(195, 53, 71, 113, 210, 100, 118, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__16_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__16_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__17_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__17_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__17_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__18_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__16_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__17_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 187, 43, 241, 119, 237, 187, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__18_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__18_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__19_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__18_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(143, 255, 200, 139, 234, 190, 186, 79)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__19_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__19_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__20_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__19_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(94, 170, 21, 98, 124, 13, 146, 236)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__20_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__20_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__21_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__20_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(73, 94, 93, 130, 219, 91, 158, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__21_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__21_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__22_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__21_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)(((size_t)(920020848) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(157, 155, 11, 89, 211, 66, 23, 78)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__22_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__22_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__23_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__23_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__23_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__24_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__22_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__23_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(30, 69, 39, 206, 155, 55, 146, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__24_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__24_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__25_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__25_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__25_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__26_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__24_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__25_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(18, 153, 129, 206, 12, 144, 24, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__26_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__26_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__27_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__26_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(171, 40, 79, 227, 191, 12, 230, 205)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__27_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__27_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "termCongr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(240, 51, 13, 100, 49, 117, 250, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(188, 76, 23, 43, 255, 180, 59, 181)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "congr("};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "withoutForbidden"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__6_value),LEAN_SCALAR_PTR_LITERAL(36, 202, 249, 244, 227, 198, 135, 34)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "ppDedentIfGrouped"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__8_value),LEAN_SCALAR_PTR_LITERAL(195, 164, 225, 181, 149, 187, 81, 113)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_termCongr = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "congrHoleForLhsKey"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__14_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__0_value),LEAN_SCALAR_PTR_LITERAL(238, 215, 26, 221, 179, 214, 146, 6)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "congrHoleIndex"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__14_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__0_value),LEAN_SCALAR_PTR_LITERAL(175, 151, 244, 6, 106, 208, 249, 237)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cHole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(240, 51, 13, 100, 49, 117, 250, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__0_value),LEAN_SCALAR_PTR_LITERAL(106, 103, 102, 75, 11, 249, 10, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Hole has type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "\nbut is expected to be a Prop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "cHoleExpand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__5_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__7_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__9_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(240, 51, 13, 100, 49, 117, 250, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__0_value),LEAN_SCALAR_PTR_LITERAL(135, 66, 63, 38, 145, 44, 188, 127)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "cHole% "};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__4_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "token"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__7_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__6_value),LEAN_SCALAR_PTR_LITERAL(44, 186, 227, 4, 127, 55, 112, 53)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__7_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__11_value),LEAN_SCALAR_PTR_LITERAL(31, 17, 90, 48, 59, 130, 19, 107)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__11_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__18_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHoleExpand(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHoleExpand___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Expecting term"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_processAntiquot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_processAntiquot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cHole%"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "\nis expected to be an equality."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "HEq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 180, 169, 191, 74, 196, 152, 188)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "\nis expected to be a `HEq`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__1_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "\nis expected to be an `Iff`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_ensureIff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_ensureIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Expecting"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "\nand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "\nto have definitionally equal types."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "iff_of_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__0_value),LEAN_SCALAR_PTR_LITERAL(186, 65, 13, 14, 191, 127, 32, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "\nto be a proposition."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "eq_of_heq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(38, 61, 104, 192, 47, 1, 246, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Cannot turn HEq proof into an equality proof. Has type"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "heq_of_eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(76, 243, 206, 193, 60, 85, 181, 135)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Congruence hole has type"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 75, .m_capacity = 75, .m_length = 74, .m_data = "\nbut its right-hand side is not definitionally equal to the expected value"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 74, .m_capacity = 74, .m_length = 73, .m_data = "\nbut its left-hand side is not definitionally equal to the expected value"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Mathlib.Tactic.TermCongr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 94, .m_capacity = 94, .m_length = 93, .m_data = "_private.Mathlib.Tactic.TermCongr.0.Mathlib.Tactic.TermCongr.CongrResult.mk'.ensureSidesDefeq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Unexpectedly did not generate an eq or heq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__9;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Cannot generate congruence because we need"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "\nto be definitionally equal to"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Subsingleton"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "elim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 130, 42, 228, 248, 162, 23, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__1_value),LEAN_SCALAR_PTR_LITERAL(79, 85, 152, 16, 239, 41, 62, 212)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "proof_irrel_heq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__3_value),LEAN_SCALAR_PTR_LITERAL(180, 105, 248, 247, 187, 48, 190, 226)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Could not generate congruence between"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Right-hand side"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "\nstill has a congruence hole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Left-hand side"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "congr(...) failed with left-hand side"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "\nand right-hand side "};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Left-hand side lost its congruence hole annotation."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Right-hand side lost its congruence hole annotation."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Right-hand side of congruence hole is"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "\nbut is expected to be"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Left-hand side of congruence hole is"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Elaborated types of congruence holes are not defeq."};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "A LHS congruence hole leaked into the RHS"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "A RHS congruence hole leaked into the LHS"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__21_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "mkCongrOfCHole, both holes"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__24_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_getJointAppFns(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Expr_eqv___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__0_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Expr_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__1_value;
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__2;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__4_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__5 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__5_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__6 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12_spec__15___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "pi_congr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 59, 165, 47, 128, 36, 68, 242)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "hole processing succeeded"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Incompatible primitive projections"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "base case"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "funext"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(226, 251, 226, 140, 5, 134, 146, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__0_value;
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "unexpected hcongr argument kind"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__1 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__1_value;
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 77, .m_capacity = 77, .m_length = 76, .m_data = "_private.Mathlib.Tactic.TermCongr.0.Mathlib.Tactic.TermCongr.mkCongrOfApp.go"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "app, args "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " by hcongr, "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " arguments"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "app, arg "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " by eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__11_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__12;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "app, hcongr needs transitivity"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__15;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " by rfl"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__17;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "app desync (function types)"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "app desync (arity)"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "app, updated arity "};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "app, arity "};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lam"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forallE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "letE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mdata"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "congr(...) internal error: out of gas"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mkCongrOf: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12_spec__15(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "Mathlib.Tactic.TermCongr.elabTermCongr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "unreachable case, sides\? guarantees Iff, Eq, and HEq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Right-hand side of elaborated pattern"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "\nis not definitionally equal to right-hand side of expected type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Left-hand side of elaborated pattern"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "\nis not definitionally equal to left-hand side of expected type"};
static const lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_66_; uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_66_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_67_ = 0;
v___x_68_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__27_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_69_ = l_Lean_registerTraceClass(v___x_66_, v___x_67_, v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2____boxed(lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_();
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole___redArg(lean_object* v_val_128_){
_start:
{
lean_inc(v_val_128_);
return v_val_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole___redArg___boxed(lean_object* v_val_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_Mathlib_Tactic_TermCongr_cHole___redArg(v_val_129_);
lean_dec(v_val_129_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole(lean_object* v_00_u03b1_131_, lean_object* v_val_132_, lean_object* v_p_133_, lean_object* v___pf_134_){
_start:
{
lean_inc(v_val_132_);
return v_val_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole___boxed(lean_object* v_00_u03b1_135_, lean_object* v_val_136_, lean_object* v_p_137_, lean_object* v___pf_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Mathlib_Tactic_TermCongr_cHole(v_00_u03b1_135_, v_val_136_, v_p_137_, v___pf_138_);
lean_dec(v_val_136_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg(lean_object* v_x_149_, lean_object* v_a_150_){
_start:
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg___closed__4));
lean_inc(v_x_149_);
v___x_152_ = l_Lean_Syntax_isOfKind(v_x_149_, v___x_151_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; lean_object* v___x_154_; 
lean_dec(v_x_149_);
v___x_153_ = lean_box(0);
v___x_154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v_a_150_);
return v___x_154_;
}
else
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_155_ = lean_unsigned_to_nat(1u);
v___x_156_ = l_Lean_Syntax_getArg(v_x_149_, v___x_155_);
lean_dec(v_x_149_);
v___x_157_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_156_);
v___x_158_ = l_Lean_Syntax_matchesNull(v___x_156_, v___x_157_);
if (v___x_158_ == 0)
{
lean_object* v___x_159_; lean_object* v___x_160_; 
lean_dec(v___x_156_);
v___x_159_ = lean_box(0);
v___x_160_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_159_);
lean_ctor_set(v___x_160_, 1, v_a_150_);
return v___x_160_;
}
else
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_161_ = lean_unsigned_to_nat(0u);
v___x_162_ = l_Lean_Syntax_getArg(v___x_156_, v___x_161_);
lean_dec(v___x_156_);
v___x_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v_a_150_);
return v___x_163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole(lean_object* v_x_164_, lean_object* v_a_165_, lean_object* v_a_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___redArg(v_x_164_, v_a_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole___boxed(lean_object* v_x_168_, lean_object* v_a_169_, lean_object* v_a_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_Mathlib_Tactic_TermCongr_unexpandCHole(v_x_168_, v_a_169_, v_a_170_);
lean_dec(v_a_169_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole(uint8_t v_forLhs_178_, lean_object* v_val_179_, lean_object* v_pf_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
uint8_t v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_186_ = 0;
v___x_187_ = lean_box(0);
v___x_188_ = l_Lean_Meta_mkFreshTypeMVar(v___x_186_, v___x_187_, v_a_181_, v_a_182_, v_a_183_, v_a_184_);
if (lean_obj_tag(v___x_188_) == 0)
{
lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_219_; 
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_188_);
if (v_isSharedCheck_219_ == 0)
{
lean_object* v_unused_220_; 
v_unused_220_ = lean_ctor_get(v___x_188_, 0);
lean_dec(v_unused_220_);
v___x_190_ = v___x_188_;
v_isShared_191_ = v_isSharedCheck_219_;
goto v_resetjp_189_;
}
else
{
lean_dec(v___x_188_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_219_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_192_ = lean_st_ref_get(v_a_182_);
v___x_193_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___closed__1));
v___x_194_ = lean_unsigned_to_nat(2u);
v___x_195_ = lean_mk_empty_array_with_capacity(v___x_194_);
v___x_196_ = lean_array_push(v___x_195_, v_val_179_);
v___x_197_ = lean_array_push(v___x_196_, v_pf_180_);
v___x_198_ = l_Lean_Meta_mkAppM(v___x_193_, v___x_197_, v_a_181_, v_a_182_, v_a_183_, v_a_184_);
if (lean_obj_tag(v___x_198_) == 0)
{
lean_object* v_mctx_199_; lean_object* v_a_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_218_; 
v_mctx_199_ = lean_ctor_get(v___x_192_, 0);
lean_inc_ref(v_mctx_199_);
lean_dec(v___x_192_);
v_a_200_ = lean_ctor_get(v___x_198_, 0);
v_isSharedCheck_218_ = !lean_is_exclusive(v___x_198_);
if (v_isSharedCheck_218_ == 0)
{
v___x_202_ = v___x_198_;
v_isShared_203_ = v_isSharedCheck_218_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_a_200_);
lean_dec(v___x_198_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_218_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v_mvarCounter_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_211_; 
v_mvarCounter_204_ = lean_ctor_get(v_mctx_199_, 3);
lean_inc(v_mvarCounter_204_);
lean_dec_ref(v_mctx_199_);
v___x_205_ = l_Lean_KVMap_empty;
v___x_206_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey));
v___x_207_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_207_, 0, v_forLhs_178_);
v___x_208_ = l_Lean_KVMap_insert(v___x_205_, v___x_206_, v___x_207_);
v___x_209_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex));
if (v_isShared_191_ == 0)
{
lean_ctor_set_tag(v___x_190_, 3);
lean_ctor_set(v___x_190_, 0, v_mvarCounter_204_);
v___x_211_ = v___x_190_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_mvarCounter_204_);
v___x_211_ = v_reuseFailAlloc_217_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_215_; 
v___x_212_ = l_Lean_KVMap_insert(v___x_208_, v___x_209_, v___x_211_);
v___x_213_ = l_Lean_Expr_mdata___override(v___x_212_, v_a_200_);
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 0, v___x_213_);
v___x_215_ = v___x_202_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v___x_213_);
v___x_215_ = v_reuseFailAlloc_216_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
return v___x_215_;
}
}
}
}
else
{
lean_dec(v___x_192_);
lean_del_object(v___x_190_);
return v___x_198_;
}
}
}
else
{
lean_dec_ref(v_pf_180_);
lean_dec_ref(v_val_179_);
return v___x_188_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole___boxed(lean_object* v_forLhs_221_, lean_object* v_val_222_, lean_object* v_pf_223_, lean_object* v_a_224_, lean_object* v_a_225_, lean_object* v_a_226_, lean_object* v_a_227_, lean_object* v_a_228_){
_start:
{
uint8_t v_forLhs_boxed_229_; lean_object* v_res_230_; 
v_forLhs_boxed_229_ = lean_unbox(v_forLhs_221_);
v_res_230_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole(v_forLhs_boxed_229_, v_val_222_, v_pf_223_, v_a_224_, v_a_225_, v_a_226_, v_a_227_);
lean_dec(v_a_227_);
lean_dec_ref(v_a_226_);
lean_dec(v_a_225_);
lean_dec_ref(v_a_224_);
return v_res_230_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0(void){
_start:
{
lean_object* v___x_231_; lean_object* v_dummy_232_; 
v___x_231_ = lean_box(0);
v_dummy_232_ = l_Lean_Expr_sort___override(v___x_231_);
return v_dummy_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f(lean_object* v_e_233_, lean_object* v_mvarCounterSaved_x3f_234_){
_start:
{
if (lean_obj_tag(v_e_233_) == 10)
{
lean_object* v_data_235_; lean_object* v_expr_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v_data_235_ = lean_ctor_get(v_e_233_, 0);
lean_inc(v_data_235_);
v_expr_236_ = lean_ctor_get(v_e_233_, 1);
lean_inc_ref(v_expr_236_);
lean_dec_ref_known(v_e_233_, 2);
v___x_237_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleForLhsKey));
v___x_238_ = l_Lean_KVMap_find(v_data_235_, v___x_237_);
if (lean_obj_tag(v___x_238_) == 0)
{
lean_object* v___x_239_; 
lean_dec_ref(v_expr_236_);
lean_dec(v_data_235_);
v___x_239_ = lean_box(0);
return v___x_239_;
}
else
{
lean_object* v_val_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_275_; 
v_val_240_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_275_ == 0)
{
v___x_242_ = v___x_238_;
v_isShared_243_ = v_isSharedCheck_275_;
goto v_resetjp_241_;
}
else
{
lean_inc(v_val_240_);
lean_dec(v___x_238_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_275_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
if (lean_obj_tag(v_val_240_) == 1)
{
uint8_t v_v_244_; lean_object* v___x_265_; lean_object* v___x_266_; 
v_v_244_ = lean_ctor_get_uint8(v_val_240_, 0);
lean_dec_ref_known(v_val_240_, 0);
v___x_265_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_congrHoleIndex));
v___x_266_ = l_Lean_KVMap_find(v_data_235_, v___x_265_);
lean_dec(v_data_235_);
if (lean_obj_tag(v___x_266_) == 0)
{
lean_object* v___x_267_; 
lean_del_object(v___x_242_);
lean_dec_ref(v_expr_236_);
v___x_267_ = lean_box(0);
return v___x_267_;
}
else
{
lean_object* v_val_268_; 
v_val_268_ = lean_ctor_get(v___x_266_, 0);
lean_inc(v_val_268_);
lean_dec_ref_known(v___x_266_, 1);
if (lean_obj_tag(v_val_268_) == 3)
{
if (lean_obj_tag(v_mvarCounterSaved_x3f_234_) == 1)
{
lean_object* v_v_269_; lean_object* v_val_270_; uint8_t v___x_271_; 
v_v_269_ = lean_ctor_get(v_val_268_, 0);
lean_inc(v_v_269_);
lean_dec_ref_known(v_val_268_, 1);
v_val_270_ = lean_ctor_get(v_mvarCounterSaved_x3f_234_, 0);
v___x_271_ = lean_nat_dec_le(v_val_270_, v_v_269_);
lean_dec(v_v_269_);
if (v___x_271_ == 0)
{
lean_object* v___x_272_; 
lean_del_object(v___x_242_);
lean_dec_ref(v_expr_236_);
v___x_272_ = lean_box(0);
return v___x_272_;
}
else
{
goto v___jp_245_;
}
}
else
{
lean_dec_ref_known(v_val_268_, 1);
goto v___jp_245_;
}
}
else
{
lean_object* v___x_273_; 
lean_dec(v_val_268_);
lean_del_object(v___x_242_);
lean_dec_ref(v_expr_236_);
v___x_273_ = lean_box(0);
return v___x_273_;
}
}
v___jp_245_:
{
lean_object* v_dummy_246_; lean_object* v_nargs_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; uint8_t v___x_254_; 
v_dummy_246_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0, &lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0);
v_nargs_247_ = l_Lean_Expr_getAppNumArgs(v_expr_236_);
lean_inc(v_nargs_247_);
v___x_248_ = lean_mk_array(v_nargs_247_, v_dummy_246_);
v___x_249_ = lean_unsigned_to_nat(1u);
v___x_250_ = lean_nat_sub(v_nargs_247_, v___x_249_);
lean_dec(v_nargs_247_);
v___x_251_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_expr_236_, v___x_248_, v___x_250_);
v___x_252_ = lean_array_get_size(v___x_251_);
v___x_253_ = lean_unsigned_to_nat(4u);
v___x_254_ = lean_nat_dec_eq(v___x_252_, v___x_253_);
if (v___x_254_ == 0)
{
lean_object* v___x_255_; 
lean_dec_ref(v___x_251_);
lean_del_object(v___x_242_);
v___x_255_ = lean_box(0);
return v___x_255_;
}
else
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_263_; 
v___x_256_ = lean_array_fget(v___x_251_, v___x_249_);
v___x_257_ = lean_unsigned_to_nat(3u);
v___x_258_ = lean_array_fget(v___x_251_, v___x_257_);
lean_dec_ref(v___x_251_);
v___x_259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_256_);
lean_ctor_set(v___x_259_, 1, v___x_258_);
v___x_260_ = lean_box(v_v_244_);
v___x_261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
lean_ctor_set(v___x_261_, 1, v___x_259_);
if (v_isShared_243_ == 0)
{
lean_ctor_set(v___x_242_, 0, v___x_261_);
v___x_263_ = v___x_242_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v___x_261_);
v___x_263_ = v_reuseFailAlloc_264_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
return v___x_263_;
}
}
}
}
else
{
lean_object* v___x_274_; 
lean_del_object(v___x_242_);
lean_dec(v_val_240_);
lean_dec_ref(v_expr_236_);
lean_dec(v_data_235_);
v___x_274_ = lean_box(0);
return v___x_274_;
}
}
}
}
else
{
lean_object* v___x_276_; 
lean_dec_ref(v_e_233_);
v___x_276_ = lean_box(0);
return v___x_276_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___boxed(lean_object* v_e_277_, lean_object* v_mvarCounterSaved_x3f_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f(v_e_277_, v_mvarCounterSaved_x3f_278_);
lean_dec(v_mvarCounterSaved_x3f_278_);
return v_res_279_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___lam__0(lean_object* v_mvarCounterSaved_280_, lean_object* v_e_x27_281_){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_282_, 0, v_mvarCounterSaved_280_);
v___x_283_ = lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f(v_e_x27_281_, v___x_282_);
lean_dec_ref_known(v___x_282_, 1);
if (lean_obj_tag(v___x_283_) == 0)
{
uint8_t v___x_284_; 
v___x_284_ = 0;
return v___x_284_;
}
else
{
uint8_t v___x_285_; 
lean_dec_ref_known(v___x_283_, 1);
v___x_285_ = 1;
return v___x_285_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___lam__0___boxed(lean_object* v_mvarCounterSaved_286_, lean_object* v_e_x27_287_){
_start:
{
uint8_t v_res_288_; lean_object* v_r_289_; 
v_res_288_ = lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___lam__0(v_mvarCounterSaved_286_, v_e_x27_287_);
v_r_289_ = lean_box(v_res_288_);
return v_r_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole(lean_object* v_mvarCounterSaved_290_, lean_object* v_e_291_){
_start:
{
lean_object* v___f_292_; lean_object* v___x_293_; 
v___f_292_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___lam__0___boxed), 2, 1);
lean_closure_set(v___f_292_, 0, v_mvarCounterSaved_290_);
v___x_293_ = lean_find_expr(v___f_292_, v_e_291_);
lean_dec_ref(v___f_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole___boxed(lean_object* v_mvarCounterSaved_294_, lean_object* v_e_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole(v_mvarCounterSaved_294_, v_e_295_);
lean_dec_ref(v_e_295_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___lam__0(lean_object* v_e_x27_297_){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_298_ = lean_box(0);
v___x_299_ = lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f(v_e_x27_297_, v___x_298_);
if (lean_obj_tag(v___x_299_) == 1)
{
lean_object* v_val_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_309_; 
v_val_300_ = lean_ctor_get(v___x_299_, 0);
v_isSharedCheck_309_ = !lean_is_exclusive(v___x_299_);
if (v_isSharedCheck_309_ == 0)
{
v___x_302_ = v___x_299_;
v_isShared_303_ = v_isSharedCheck_309_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_val_300_);
lean_dec(v___x_299_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_309_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v_snd_304_; lean_object* v_fst_305_; lean_object* v___x_307_; 
v_snd_304_ = lean_ctor_get(v_val_300_, 1);
lean_inc(v_snd_304_);
lean_dec(v_val_300_);
v_fst_305_ = lean_ctor_get(v_snd_304_, 0);
lean_inc(v_fst_305_);
lean_dec(v_snd_304_);
if (v_isShared_303_ == 0)
{
lean_ctor_set(v___x_302_, 0, v_fst_305_);
v___x_307_ = v___x_302_;
goto v_reusejp_306_;
}
else
{
lean_object* v_reuseFailAlloc_308_; 
v_reuseFailAlloc_308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_308_, 0, v_fst_305_);
v___x_307_ = v_reuseFailAlloc_308_;
goto v_reusejp_306_;
}
v_reusejp_306_:
{
return v___x_307_;
}
}
}
else
{
lean_dec(v___x_299_);
return v___x_298_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(lean_object* v_e_311_){
_start:
{
lean_object* v___f_312_; lean_object* v___x_313_; 
v___f_312_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___closed__0));
v___x_313_ = lean_replace_expr(v___f_312_, v_e_311_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles___boxed(lean_object* v_e_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(v_e_314_);
lean_dec_ref(v_e_314_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(lean_object* v_msgData_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v___x_322_; lean_object* v_env_323_; lean_object* v___x_324_; lean_object* v_mctx_325_; lean_object* v_lctx_326_; lean_object* v_options_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_322_ = lean_st_ref_get(v___y_320_);
v_env_323_ = lean_ctor_get(v___x_322_, 0);
lean_inc_ref(v_env_323_);
lean_dec(v___x_322_);
v___x_324_ = lean_st_ref_get(v___y_318_);
v_mctx_325_ = lean_ctor_get(v___x_324_, 0);
lean_inc_ref(v_mctx_325_);
lean_dec(v___x_324_);
v_lctx_326_ = lean_ctor_get(v___y_317_, 2);
v_options_327_ = lean_ctor_get(v___y_319_, 2);
lean_inc_ref(v_options_327_);
lean_inc_ref(v_lctx_326_);
v___x_328_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_328_, 0, v_env_323_);
lean_ctor_set(v___x_328_, 1, v_mctx_325_);
lean_ctor_set(v___x_328_, 2, v_lctx_326_);
lean_ctor_set(v___x_328_, 3, v_options_327_);
v___x_329_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
lean_ctor_set(v___x_329_, 1, v_msgData_316_);
v___x_330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0___boxed(lean_object* v_msgData_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(v_msgData_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
lean_dec(v___y_333_);
lean_dec_ref(v___y_332_);
return v_res_337_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__2(lean_object* v_opts_338_, lean_object* v_opt_339_){
_start:
{
lean_object* v_name_340_; lean_object* v_defValue_341_; lean_object* v_map_342_; lean_object* v___x_343_; 
v_name_340_ = lean_ctor_get(v_opt_339_, 0);
v_defValue_341_ = lean_ctor_get(v_opt_339_, 1);
v_map_342_ = lean_ctor_get(v_opts_338_, 0);
v___x_343_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_342_, v_name_340_);
if (lean_obj_tag(v___x_343_) == 0)
{
uint8_t v___x_344_; 
v___x_344_ = lean_unbox(v_defValue_341_);
return v___x_344_;
}
else
{
lean_object* v_val_345_; 
v_val_345_ = lean_ctor_get(v___x_343_, 0);
lean_inc(v_val_345_);
lean_dec_ref_known(v___x_343_, 1);
if (lean_obj_tag(v_val_345_) == 1)
{
uint8_t v_v_346_; 
v_v_346_ = lean_ctor_get_uint8(v_val_345_, 0);
lean_dec_ref_known(v_val_345_, 0);
return v_v_346_;
}
else
{
uint8_t v___x_347_; 
lean_dec(v_val_345_);
v___x_347_ = lean_unbox(v_defValue_341_);
return v___x_347_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__2___boxed(lean_object* v_opts_348_, lean_object* v_opt_349_){
_start:
{
uint8_t v_res_350_; lean_object* v_r_351_; 
v_res_350_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__2(v_opts_348_, v_opt_349_);
lean_dec_ref(v_opt_349_);
lean_dec_ref(v_opts_348_);
v_r_351_ = lean_box(v_res_350_);
return v_r_351_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_352_ = lean_box(1);
v___x_353_ = l_Lean_MessageData_ofFormat(v___x_352_);
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__3(void){
_start:
{
lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_357_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__2));
v___x_358_ = l_Lean_MessageData_ofFormat(v___x_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3(lean_object* v_x_359_, lean_object* v_x_360_){
_start:
{
if (lean_obj_tag(v_x_360_) == 0)
{
return v_x_359_;
}
else
{
lean_object* v_head_361_; lean_object* v_tail_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_384_; 
v_head_361_ = lean_ctor_get(v_x_360_, 0);
v_tail_362_ = lean_ctor_get(v_x_360_, 1);
v_isSharedCheck_384_ = !lean_is_exclusive(v_x_360_);
if (v_isSharedCheck_384_ == 0)
{
v___x_364_ = v_x_360_;
v_isShared_365_ = v_isSharedCheck_384_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_tail_362_);
lean_inc(v_head_361_);
lean_dec(v_x_360_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_384_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v_before_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_382_; 
v_before_366_ = lean_ctor_get(v_head_361_, 0);
v_isSharedCheck_382_ = !lean_is_exclusive(v_head_361_);
if (v_isSharedCheck_382_ == 0)
{
lean_object* v_unused_383_; 
v_unused_383_ = lean_ctor_get(v_head_361_, 1);
lean_dec(v_unused_383_);
v___x_368_ = v_head_361_;
v_isShared_369_ = v_isSharedCheck_382_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_before_366_);
lean_dec(v_head_361_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_382_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_370_; lean_object* v___x_372_; 
v___x_370_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_369_ == 0)
{
lean_ctor_set_tag(v___x_368_, 7);
lean_ctor_set(v___x_368_, 1, v___x_370_);
lean_ctor_set(v___x_368_, 0, v_x_359_);
v___x_372_ = v___x_368_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v_x_359_);
lean_ctor_set(v_reuseFailAlloc_381_, 1, v___x_370_);
v___x_372_ = v_reuseFailAlloc_381_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
lean_object* v___x_373_; lean_object* v___x_375_; 
v___x_373_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__3);
if (v_isShared_365_ == 0)
{
lean_ctor_set_tag(v___x_364_, 7);
lean_ctor_set(v___x_364_, 1, v___x_373_);
lean_ctor_set(v___x_364_, 0, v___x_372_);
v___x_375_ = v___x_364_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_380_; 
v_reuseFailAlloc_380_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_380_, 0, v___x_372_);
lean_ctor_set(v_reuseFailAlloc_380_, 1, v___x_373_);
v___x_375_ = v_reuseFailAlloc_380_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_376_ = l_Lean_MessageData_ofSyntax(v_before_366_);
v___x_377_ = l_Lean_indentD(v___x_376_);
v___x_378_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_375_);
lean_ctor_set(v___x_378_, 1, v___x_377_);
v_x_359_ = v___x_378_;
v_x_360_ = v_tail_362_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__2(void){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; 
v___x_388_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__1));
v___x_389_ = l_Lean_MessageData_ofFormat(v___x_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg(lean_object* v_msgData_390_, lean_object* v_macroStack_391_, lean_object* v___y_392_){
_start:
{
lean_object* v_options_394_; lean_object* v___x_395_; uint8_t v___x_396_; 
v_options_394_ = lean_ctor_get(v___y_392_, 2);
v___x_395_ = l_Lean_Elab_pp_macroStack;
v___x_396_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__2(v_options_394_, v___x_395_);
if (v___x_396_ == 0)
{
lean_object* v___x_397_; 
lean_dec(v_macroStack_391_);
v___x_397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_397_, 0, v_msgData_390_);
return v___x_397_;
}
else
{
if (lean_obj_tag(v_macroStack_391_) == 0)
{
lean_object* v___x_398_; 
v___x_398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_398_, 0, v_msgData_390_);
return v___x_398_;
}
else
{
lean_object* v_head_399_; lean_object* v_after_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_415_; 
v_head_399_ = lean_ctor_get(v_macroStack_391_, 0);
lean_inc(v_head_399_);
v_after_400_ = lean_ctor_get(v_head_399_, 1);
v_isSharedCheck_415_ = !lean_is_exclusive(v_head_399_);
if (v_isSharedCheck_415_ == 0)
{
lean_object* v_unused_416_; 
v_unused_416_ = lean_ctor_get(v_head_399_, 0);
lean_dec(v_unused_416_);
v___x_402_ = v_head_399_;
v_isShared_403_ = v_isSharedCheck_415_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_after_400_);
lean_dec(v_head_399_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_415_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v___x_404_; lean_object* v___x_406_; 
v___x_404_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3___closed__0);
if (v_isShared_403_ == 0)
{
lean_ctor_set_tag(v___x_402_, 7);
lean_ctor_set(v___x_402_, 1, v___x_404_);
lean_ctor_set(v___x_402_, 0, v_msgData_390_);
v___x_406_ = v___x_402_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_msgData_390_);
lean_ctor_set(v_reuseFailAlloc_414_, 1, v___x_404_);
v___x_406_ = v_reuseFailAlloc_414_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v_msgData_411_; lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_407_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___closed__2);
v___x_408_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_406_);
lean_ctor_set(v___x_408_, 1, v___x_407_);
v___x_409_ = l_Lean_MessageData_ofSyntax(v_after_400_);
v___x_410_ = l_Lean_indentD(v___x_409_);
v_msgData_411_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_411_, 0, v___x_408_);
lean_ctor_set(v_msgData_411_, 1, v___x_410_);
v___x_412_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1_spec__3(v_msgData_411_, v_macroStack_391_);
v___x_413_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_413_, 0, v___x_412_);
return v___x_413_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg___boxed(lean_object* v_msgData_417_, lean_object* v_macroStack_418_, lean_object* v___y_419_, lean_object* v___y_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg(v_msgData_417_, v_macroStack_418_, v___y_419_);
lean_dec_ref(v___y_419_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(lean_object* v_msg_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
lean_object* v_ref_430_; lean_object* v___x_431_; lean_object* v_a_432_; lean_object* v_macroStack_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_444_; 
v_ref_430_ = lean_ctor_get(v___y_427_, 5);
v___x_431_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(v_msg_422_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
v_a_432_ = lean_ctor_get(v___x_431_, 0);
lean_inc(v_a_432_);
lean_dec_ref(v___x_431_);
v_macroStack_433_ = lean_ctor_get(v___y_423_, 1);
v___x_434_ = l_Lean_Elab_getBetterRef(v_ref_430_, v_macroStack_433_);
lean_inc(v_macroStack_433_);
v___x_435_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg(v_a_432_, v_macroStack_433_, v___y_427_);
v_a_436_ = lean_ctor_get(v___x_435_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_435_);
if (v_isSharedCheck_444_ == 0)
{
v___x_438_ = v___x_435_;
v_isShared_439_ = v_isSharedCheck_444_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v___x_435_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_444_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_440_; lean_object* v___x_442_; 
v___x_440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_440_, 0, v___x_434_);
lean_ctor_set(v___x_440_, 1, v_a_436_);
if (v_isShared_439_ == 0)
{
lean_ctor_set_tag(v___x_438_, 1);
lean_ctor_set(v___x_438_, 0, v___x_440_);
v___x_442_ = v___x_438_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v___x_440_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg___boxed(lean_object* v_msg_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(v_msg_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_, v___y_450_, v___y_451_);
lean_dec(v___y_451_);
lean_dec_ref(v___y_450_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
return v_res_453_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__1(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_455_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__0));
v___x_456_ = l_Lean_stringToMessageData(v___x_455_);
return v___x_456_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__3(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; 
v___x_458_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__2));
v___x_459_ = l_Lean_stringToMessageData(v___x_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole(lean_object* v_h_460_, uint8_t v_forLhs_461_, lean_object* v_expectedType_x3f_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_){
_start:
{
lean_object* v___x_470_; uint8_t v___x_471_; lean_object* v___x_472_; 
v___x_470_ = lean_box(0);
v___x_471_ = 1;
v___x_472_ = l_Lean_Elab_Term_elabTerm(v_h_460_, v___x_470_, v___x_471_, v___x_471_, v_a_463_, v_a_464_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
if (lean_obj_tag(v___x_472_) == 0)
{
lean_object* v_a_473_; lean_object* v___y_475_; lean_object* v___y_476_; lean_object* v___y_477_; lean_object* v___y_478_; lean_object* v___y_479_; lean_object* v___x_494_; 
v_a_473_ = lean_ctor_get(v___x_472_, 0);
lean_inc_n(v_a_473_, 2);
lean_dec_ref_known(v___x_472_, 1);
lean_inc(v_a_468_);
lean_inc_ref(v_a_467_);
lean_inc(v_a_466_);
lean_inc_ref(v_a_465_);
v___x_494_ = lean_infer_type(v_a_473_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
if (lean_obj_tag(v___x_494_) == 0)
{
lean_object* v_a_495_; lean_object* v___y_497_; lean_object* v___y_498_; lean_object* v___y_499_; lean_object* v___y_500_; lean_object* v___x_514_; 
v_a_495_ = lean_ctor_get(v___x_494_, 0);
lean_inc_n(v_a_495_, 2);
lean_dec_ref_known(v___x_494_, 1);
lean_inc(v_a_468_);
lean_inc_ref(v_a_467_);
lean_inc(v_a_466_);
lean_inc_ref(v_a_465_);
v___x_514_ = lean_infer_type(v_a_495_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
if (lean_obj_tag(v___x_514_) == 0)
{
lean_object* v_a_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v_a_515_ = lean_ctor_get(v___x_514_, 0);
lean_inc(v_a_515_);
lean_dec_ref_known(v___x_514_, 1);
v___x_516_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0, &lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0);
v___x_517_ = l_Lean_Meta_isExprDefEq(v_a_515_, v___x_516_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
if (lean_obj_tag(v___x_517_) == 0)
{
lean_object* v_a_518_; uint8_t v___x_519_; 
v_a_518_ = lean_ctor_get(v___x_517_, 0);
lean_inc(v_a_518_);
lean_dec_ref_known(v___x_517_, 1);
v___x_519_ = lean_unbox(v_a_518_);
lean_dec(v_a_518_);
if (v___x_519_ == 0)
{
lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v_a_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_534_; 
lean_dec(v_a_473_);
lean_dec(v_expectedType_x3f_462_);
v___x_520_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__1);
v___x_521_ = l_Lean_MessageData_ofExpr(v_a_495_);
v___x_522_ = l_Lean_indentD(v___x_521_);
v___x_523_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_523_, 0, v___x_520_);
lean_ctor_set(v___x_523_, 1, v___x_522_);
v___x_524_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___closed__3);
v___x_525_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_523_);
lean_ctor_set(v___x_525_, 1, v___x_524_);
v___x_526_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(v___x_525_, v_a_463_, v_a_464_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
v_a_527_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_534_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_534_ == 0)
{
v___x_529_ = v___x_526_;
v_isShared_530_ = v_isSharedCheck_534_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_a_527_);
lean_dec(v___x_526_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_534_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
lean_object* v___x_532_; 
if (v_isShared_530_ == 0)
{
v___x_532_ = v___x_529_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_533_, 0, v_a_527_);
v___x_532_ = v_reuseFailAlloc_533_;
goto v_reusejp_531_;
}
v_reusejp_531_:
{
return v___x_532_;
}
}
}
else
{
v___y_497_ = v_a_465_;
v___y_498_ = v_a_466_;
v___y_499_ = v_a_467_;
v___y_500_ = v_a_468_;
goto v___jp_496_;
}
}
else
{
lean_object* v_a_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_542_; 
lean_dec(v_a_495_);
lean_dec(v_a_473_);
lean_dec(v_expectedType_x3f_462_);
v_a_535_ = lean_ctor_get(v___x_517_, 0);
v_isSharedCheck_542_ = !lean_is_exclusive(v___x_517_);
if (v_isSharedCheck_542_ == 0)
{
v___x_537_ = v___x_517_;
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_a_535_);
lean_dec(v___x_517_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_540_; 
if (v_isShared_538_ == 0)
{
v___x_540_ = v___x_537_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v_a_535_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
}
}
else
{
lean_dec(v_a_495_);
lean_dec(v_a_473_);
lean_dec(v_expectedType_x3f_462_);
return v___x_514_;
}
v___jp_496_:
{
lean_object* v___x_501_; 
lean_inc(v___y_500_);
lean_inc_ref(v___y_499_);
lean_inc(v___y_498_);
lean_inc_ref(v___y_497_);
v___x_501_ = lean_whnf(v_a_495_, v___y_497_, v___y_498_, v___y_499_, v___y_500_);
if (lean_obj_tag(v___x_501_) == 0)
{
lean_object* v_a_502_; lean_object* v___x_503_; 
v_a_502_ = lean_ctor_get(v___x_501_, 0);
lean_inc(v_a_502_);
lean_dec_ref_known(v___x_501_, 1);
v___x_503_ = lp_mathlib_Lean_Expr_sides_x3f(v_a_502_);
lean_dec(v_a_502_);
if (lean_obj_tag(v___x_503_) == 1)
{
lean_object* v_val_504_; lean_object* v_snd_505_; 
v_val_504_ = lean_ctor_get(v___x_503_, 0);
lean_inc(v_val_504_);
lean_dec_ref_known(v___x_503_, 1);
v_snd_505_ = lean_ctor_get(v_val_504_, 1);
lean_inc(v_snd_505_);
lean_dec(v_val_504_);
if (v_forLhs_461_ == 0)
{
lean_object* v_snd_506_; lean_object* v_snd_507_; 
v_snd_506_ = lean_ctor_get(v_snd_505_, 1);
lean_inc(v_snd_506_);
lean_dec(v_snd_505_);
v_snd_507_ = lean_ctor_get(v_snd_506_, 1);
lean_inc(v_snd_507_);
lean_dec(v_snd_506_);
v___y_475_ = v___y_498_;
v___y_476_ = v___y_497_;
v___y_477_ = v___y_500_;
v___y_478_ = v___y_499_;
v___y_479_ = v_snd_507_;
goto v___jp_474_;
}
else
{
lean_object* v_fst_508_; 
v_fst_508_ = lean_ctor_get(v_snd_505_, 0);
lean_inc(v_fst_508_);
lean_dec(v_snd_505_);
v___y_475_ = v___y_498_;
v___y_476_ = v___y_497_;
v___y_477_ = v___y_500_;
v___y_478_ = v___y_499_;
v___y_479_ = v_fst_508_;
goto v___jp_474_;
}
}
else
{
uint8_t v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; 
lean_dec(v___x_503_);
v___x_509_ = 0;
v___x_510_ = lean_box(0);
v___x_511_ = l_Lean_Meta_mkFreshExprMVar(v_expectedType_x3f_462_, v___x_509_, v___x_510_, v___y_497_, v___y_498_, v___y_499_, v___y_500_);
if (lean_obj_tag(v___x_511_) == 0)
{
lean_object* v_a_512_; lean_object* v___x_513_; 
v_a_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc(v_a_512_);
lean_dec_ref_known(v___x_511_, 1);
v___x_513_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole(v_forLhs_461_, v_a_512_, v_a_473_, v___y_497_, v___y_498_, v___y_499_, v___y_500_);
return v___x_513_;
}
else
{
lean_dec(v_a_473_);
return v___x_511_;
}
}
}
else
{
lean_dec(v_a_473_);
lean_dec(v_expectedType_x3f_462_);
return v___x_501_;
}
}
}
else
{
lean_dec(v_a_473_);
lean_dec(v_expectedType_x3f_462_);
return v___x_494_;
}
v___jp_474_:
{
if (lean_obj_tag(v_expectedType_x3f_462_) == 1)
{
lean_object* v_val_480_; lean_object* v___x_481_; 
v_val_480_ = lean_ctor_get(v_expectedType_x3f_462_, 0);
lean_inc(v_val_480_);
lean_dec_ref_known(v_expectedType_x3f_462_, 1);
lean_inc(v___y_477_);
lean_inc_ref(v___y_478_);
lean_inc(v___y_475_);
lean_inc_ref(v___y_476_);
lean_inc_ref(v___y_479_);
v___x_481_ = lean_infer_type(v___y_479_, v___y_476_, v___y_475_, v___y_478_, v___y_477_);
if (lean_obj_tag(v___x_481_) == 0)
{
lean_object* v_a_482_; lean_object* v___x_483_; 
v_a_482_ = lean_ctor_get(v___x_481_, 0);
lean_inc(v_a_482_);
lean_dec_ref_known(v___x_481_, 1);
v___x_483_ = l_Lean_Meta_isExprDefEq(v_val_480_, v_a_482_, v___y_476_, v___y_475_, v___y_478_, v___y_477_);
if (lean_obj_tag(v___x_483_) == 0)
{
lean_object* v___x_484_; 
lean_dec_ref_known(v___x_483_, 1);
v___x_484_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole(v_forLhs_461_, v___y_479_, v_a_473_, v___y_476_, v___y_475_, v___y_478_, v___y_477_);
return v___x_484_;
}
else
{
lean_object* v_a_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_492_; 
lean_dec_ref(v___y_479_);
lean_dec(v_a_473_);
v_a_485_ = lean_ctor_get(v___x_483_, 0);
v_isSharedCheck_492_ = !lean_is_exclusive(v___x_483_);
if (v_isSharedCheck_492_ == 0)
{
v___x_487_ = v___x_483_;
v_isShared_488_ = v_isSharedCheck_492_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_a_485_);
lean_dec(v___x_483_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_492_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_490_; 
if (v_isShared_488_ == 0)
{
v___x_490_ = v___x_487_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_491_; 
v_reuseFailAlloc_491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_491_, 0, v_a_485_);
v___x_490_ = v_reuseFailAlloc_491_;
goto v_reusejp_489_;
}
v_reusejp_489_:
{
return v___x_490_;
}
}
}
}
else
{
lean_dec(v_val_480_);
lean_dec_ref(v___y_479_);
lean_dec(v_a_473_);
return v___x_481_;
}
}
else
{
lean_object* v___x_493_; 
lean_dec(v_expectedType_x3f_462_);
v___x_493_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCHole(v_forLhs_461_, v___y_479_, v_a_473_, v___y_476_, v___y_475_, v___y_478_, v___y_477_);
return v___x_493_;
}
}
}
else
{
lean_dec(v_expectedType_x3f_462_);
return v___x_472_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole___boxed(lean_object* v_h_543_, lean_object* v_forLhs_544_, lean_object* v_expectedType_x3f_545_, lean_object* v_a_546_, lean_object* v_a_547_, lean_object* v_a_548_, lean_object* v_a_549_, lean_object* v_a_550_, lean_object* v_a_551_, lean_object* v_a_552_){
_start:
{
uint8_t v_forLhs_boxed_553_; lean_object* v_res_554_; 
v_forLhs_boxed_553_ = lean_unbox(v_forLhs_544_);
v_res_554_ = lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole(v_h_543_, v_forLhs_boxed_553_, v_expectedType_x3f_545_, v_a_546_, v_a_547_, v_a_548_, v_a_549_, v_a_550_, v_a_551_);
lean_dec(v_a_551_);
lean_dec_ref(v_a_550_);
lean_dec(v_a_549_);
lean_dec_ref(v_a_548_);
lean_dec(v_a_547_);
lean_dec_ref(v_a_546_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0(lean_object* v_00_u03b1_555_, lean_object* v_msg_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_){
_start:
{
lean_object* v___x_564_; 
v___x_564_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(v_msg_556_, v___y_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_);
return v___x_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___boxed(lean_object* v_00_u03b1_565_, lean_object* v_msg_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0(v_00_u03b1_565_, v_msg_566_, v___y_567_, v___y_568_, v___y_569_, v___y_570_, v___y_571_, v___y_572_);
lean_dec(v___y_572_);
lean_dec_ref(v___y_571_);
lean_dec(v___y_570_);
lean_dec_ref(v___y_569_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1(lean_object* v_msgData_575_, lean_object* v_macroStack_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___redArg(v_msgData_575_, v_macroStack_576_, v___y_581_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1___boxed(lean_object* v_msgData_585_, lean_object* v_macroStack_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_){
_start:
{
lean_object* v_res_594_; 
v_res_594_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__1(v_msgData_585_, v_macroStack_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_, v___y_591_, v___y_592_);
lean_dec(v___y_592_);
lean_dec_ref(v___y_591_);
lean_dec(v___y_590_);
lean_dec_ref(v___y_589_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
return v_res_594_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; 
v___x_647_ = lean_box(0);
v___x_648_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_649_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_649_, 0, v___x_648_);
lean_ctor_set(v___x_649_, 1, v___x_647_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg(){
_start:
{
lean_object* v___x_651_; lean_object* v___x_652_; 
v___x_651_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___closed__0);
v___x_652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_652_, 0, v___x_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg___boxed(lean_object* v___y_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg();
return v_res_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0(lean_object* v_00_u03b1_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg();
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___boxed(lean_object* v_00_u03b1_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0(v_00_u03b1_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
lean_dec(v___y_670_);
lean_dec_ref(v___y_669_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
lean_dec(v___y_666_);
lean_dec_ref(v___y_665_);
return v_res_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHoleExpand(lean_object* v_stx_673_, lean_object* v_expectedType_x3f_674_, lean_object* v_a_675_, lean_object* v_a_676_, lean_object* v_a_677_, lean_object* v_a_678_, lean_object* v_a_679_, lean_object* v_a_680_){
_start:
{
lean_object* v___x_682_; uint8_t v___x_683_; 
v___x_682_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1));
lean_inc(v_stx_673_);
v___x_683_ = l_Lean_Syntax_isOfKind(v_stx_673_, v___x_682_);
if (v___x_683_ == 0)
{
lean_object* v___x_684_; 
lean_dec(v_expectedType_x3f_674_);
lean_dec(v_stx_673_);
v___x_684_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg();
return v___x_684_;
}
else
{
lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; uint8_t v___x_688_; 
v___x_685_ = lean_unsigned_to_nat(1u);
v___x_686_ = l_Lean_Syntax_getArg(v_stx_673_, v___x_685_);
v___x_687_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8));
lean_inc(v___x_686_);
v___x_688_ = l_Lean_Syntax_isOfKind(v___x_686_, v___x_687_);
if (v___x_688_ == 0)
{
lean_object* v___x_689_; uint8_t v___x_690_; 
v___x_689_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12));
v___x_690_ = l_Lean_Syntax_isOfKind(v___x_686_, v___x_689_);
if (v___x_690_ == 0)
{
lean_object* v___x_691_; 
lean_dec(v_expectedType_x3f_674_);
lean_dec(v_stx_673_);
v___x_691_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg();
return v___x_691_;
}
else
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_692_ = lean_unsigned_to_nat(2u);
v___x_693_ = l_Lean_Syntax_getArg(v_stx_673_, v___x_692_);
lean_dec(v_stx_673_);
v___x_694_ = lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole(v___x_693_, v___x_688_, v_expectedType_x3f_674_, v_a_675_, v_a_676_, v_a_677_, v_a_678_, v_a_679_, v_a_680_);
return v___x_694_;
}
}
else
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
lean_dec(v___x_686_);
v___x_695_ = lean_unsigned_to_nat(2u);
v___x_696_ = l_Lean_Syntax_getArg(v_stx_673_, v___x_695_);
lean_dec(v_stx_673_);
v___x_697_ = lp_mathlib_Mathlib_Tactic_TermCongr_elabCHole(v___x_696_, v___x_688_, v_expectedType_x3f_674_, v_a_675_, v_a_676_, v_a_677_, v_a_678_, v_a_679_, v_a_680_);
return v___x_697_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabCHoleExpand___boxed(lean_object* v_stx_698_, lean_object* v_expectedType_x3f_699_, lean_object* v_a_700_, lean_object* v_a_701_, lean_object* v_a_702_, lean_object* v_a_703_, lean_object* v_a_704_, lean_object* v_a_705_, lean_object* v_a_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_mathlib_Mathlib_Tactic_TermCongr_elabCHoleExpand(v_stx_698_, v_expectedType_x3f_699_, v_a_700_, v_a_701_, v_a_702_, v_a_703_, v_a_704_, v_a_705_);
lean_dec(v_a_705_);
lean_dec_ref(v_a_704_);
lean_dec(v_a_703_);
lean_dec_ref(v_a_702_);
lean_dec(v_a_701_);
lean_dec_ref(v_a_700_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___redArg(lean_object* v_ref_708_, lean_object* v_msg_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
lean_object* v_fileName_717_; lean_object* v_fileMap_718_; lean_object* v_options_719_; lean_object* v_currRecDepth_720_; lean_object* v_maxRecDepth_721_; lean_object* v_ref_722_; lean_object* v_currNamespace_723_; lean_object* v_openDecls_724_; lean_object* v_initHeartbeats_725_; lean_object* v_maxHeartbeats_726_; lean_object* v_quotContext_727_; lean_object* v_currMacroScope_728_; uint8_t v_diag_729_; lean_object* v_cancelTk_x3f_730_; uint8_t v_suppressElabErrors_731_; lean_object* v_inheritedTraceOptions_732_; lean_object* v_ref_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v_fileName_717_ = lean_ctor_get(v___y_714_, 0);
v_fileMap_718_ = lean_ctor_get(v___y_714_, 1);
v_options_719_ = lean_ctor_get(v___y_714_, 2);
v_currRecDepth_720_ = lean_ctor_get(v___y_714_, 3);
v_maxRecDepth_721_ = lean_ctor_get(v___y_714_, 4);
v_ref_722_ = lean_ctor_get(v___y_714_, 5);
v_currNamespace_723_ = lean_ctor_get(v___y_714_, 6);
v_openDecls_724_ = lean_ctor_get(v___y_714_, 7);
v_initHeartbeats_725_ = lean_ctor_get(v___y_714_, 8);
v_maxHeartbeats_726_ = lean_ctor_get(v___y_714_, 9);
v_quotContext_727_ = lean_ctor_get(v___y_714_, 10);
v_currMacroScope_728_ = lean_ctor_get(v___y_714_, 11);
v_diag_729_ = lean_ctor_get_uint8(v___y_714_, sizeof(void*)*14);
v_cancelTk_x3f_730_ = lean_ctor_get(v___y_714_, 12);
v_suppressElabErrors_731_ = lean_ctor_get_uint8(v___y_714_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_732_ = lean_ctor_get(v___y_714_, 13);
v_ref_733_ = l_Lean_replaceRef(v_ref_708_, v_ref_722_);
lean_inc_ref(v_inheritedTraceOptions_732_);
lean_inc(v_cancelTk_x3f_730_);
lean_inc(v_currMacroScope_728_);
lean_inc(v_quotContext_727_);
lean_inc(v_maxHeartbeats_726_);
lean_inc(v_initHeartbeats_725_);
lean_inc(v_openDecls_724_);
lean_inc(v_currNamespace_723_);
lean_inc(v_maxRecDepth_721_);
lean_inc(v_currRecDepth_720_);
lean_inc_ref(v_options_719_);
lean_inc_ref(v_fileMap_718_);
lean_inc_ref(v_fileName_717_);
v___x_734_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_734_, 0, v_fileName_717_);
lean_ctor_set(v___x_734_, 1, v_fileMap_718_);
lean_ctor_set(v___x_734_, 2, v_options_719_);
lean_ctor_set(v___x_734_, 3, v_currRecDepth_720_);
lean_ctor_set(v___x_734_, 4, v_maxRecDepth_721_);
lean_ctor_set(v___x_734_, 5, v_ref_733_);
lean_ctor_set(v___x_734_, 6, v_currNamespace_723_);
lean_ctor_set(v___x_734_, 7, v_openDecls_724_);
lean_ctor_set(v___x_734_, 8, v_initHeartbeats_725_);
lean_ctor_set(v___x_734_, 9, v_maxHeartbeats_726_);
lean_ctor_set(v___x_734_, 10, v_quotContext_727_);
lean_ctor_set(v___x_734_, 11, v_currMacroScope_728_);
lean_ctor_set(v___x_734_, 12, v_cancelTk_x3f_730_);
lean_ctor_set(v___x_734_, 13, v_inheritedTraceOptions_732_);
lean_ctor_set_uint8(v___x_734_, sizeof(void*)*14, v_diag_729_);
lean_ctor_set_uint8(v___x_734_, sizeof(void*)*14 + 1, v_suppressElabErrors_731_);
v___x_735_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(v_msg_709_, v___y_710_, v___y_711_, v___y_712_, v___y_713_, v___x_734_, v___y_715_);
lean_dec_ref_known(v___x_734_, 14);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___redArg___boxed(lean_object* v_ref_736_, lean_object* v_msg_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_){
_start:
{
lean_object* v_res_745_; 
v_res_745_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___redArg(v_ref_736_, v_msg_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_);
lean_dec(v___y_743_);
lean_dec_ref(v___y_742_);
lean_dec(v___y_741_);
lean_dec_ref(v___y_740_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
lean_dec(v_ref_736_);
return v_res_745_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_any___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__0(lean_object* v_x_746_){
_start:
{
if (lean_obj_tag(v_x_746_) == 0)
{
uint8_t v___x_747_; 
v___x_747_ = 0;
return v___x_747_;
}
else
{
lean_object* v_head_748_; lean_object* v_tail_749_; lean_object* v_fst_750_; lean_object* v___x_751_; uint8_t v___x_752_; 
v_head_748_ = lean_ctor_get(v_x_746_, 0);
v_tail_749_ = lean_ctor_get(v_x_746_, 1);
v_fst_750_ = lean_ctor_get(v_head_748_, 0);
v___x_751_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__11));
v___x_752_ = lean_name_eq(v_fst_750_, v___x_751_);
if (v___x_752_ == 0)
{
v_x_746_ = v_tail_749_;
goto _start;
}
else
{
return v___x_752_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_any___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__0___boxed(lean_object* v_x_754_){
_start:
{
uint8_t v_res_755_; lean_object* v_r_756_; 
v_res_755_ = lp_mathlib_List_any___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__0(v_x_754_);
lean_dec(v_x_754_);
v_r_756_ = lean_box(v_res_755_);
return v_r_756_;
}
}
static lean_object* _init_lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__1(void){
_start:
{
lean_object* v___x_758_; lean_object* v___x_759_; 
v___x_758_ = ((lean_object*)(lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__0));
v___x_759_ = l_Lean_stringToMessageData(v___x_758_);
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0(lean_object* v_expand_760_, lean_object* v_s_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_){
_start:
{
lean_object* v___y_770_; lean_object* v___y_771_; lean_object* v___y_772_; lean_object* v___y_773_; lean_object* v___y_774_; lean_object* v___y_775_; uint8_t v___x_796_; 
lean_inc(v_s_761_);
v___x_796_ = l_Lean_Syntax_isAntiquots(v_s_761_);
if (v___x_796_ == 0)
{
lean_object* v___x_797_; lean_object* v___x_798_; 
lean_dec(v_s_761_);
lean_dec_ref(v_expand_760_);
v___x_797_ = lean_box(0);
v___x_798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_798_, 0, v___x_797_);
return v___x_798_;
}
else
{
lean_object* v_ks_799_; uint8_t v___x_800_; 
lean_inc(v_s_761_);
v_ks_799_ = l_Lean_Syntax_antiquotKinds(v_s_761_);
v___x_800_ = lp_mathlib_List_any___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__0(v_ks_799_);
lean_dec(v_ks_799_);
if (v___x_800_ == 0)
{
lean_object* v___x_801_; lean_object* v___x_802_; 
v___x_801_ = lean_obj_once(&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__1, &lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__1_once, _init_lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___closed__1);
v___x_802_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___redArg(v_s_761_, v___x_801_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_802_) == 0)
{
lean_dec_ref_known(v___x_802_, 1);
v___y_770_ = v___y_762_;
v___y_771_ = v___y_763_;
v___y_772_ = v___y_764_;
v___y_773_ = v___y_765_;
v___y_774_ = v___y_766_;
v___y_775_ = v___y_767_;
goto v___jp_769_;
}
else
{
lean_object* v_a_803_; lean_object* v___x_805_; uint8_t v_isShared_806_; uint8_t v_isSharedCheck_810_; 
lean_dec(v_s_761_);
lean_dec_ref(v_expand_760_);
v_a_803_ = lean_ctor_get(v___x_802_, 0);
v_isSharedCheck_810_ = !lean_is_exclusive(v___x_802_);
if (v_isSharedCheck_810_ == 0)
{
v___x_805_ = v___x_802_;
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
else
{
lean_inc(v_a_803_);
lean_dec(v___x_802_);
v___x_805_ = lean_box(0);
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
v_resetjp_804_:
{
lean_object* v___x_808_; 
if (v_isShared_806_ == 0)
{
v___x_808_ = v___x_805_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_a_803_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
return v___x_808_;
}
}
}
}
else
{
v___y_770_ = v___y_762_;
v___y_771_ = v___y_763_;
v___y_772_ = v___y_764_;
v___y_773_ = v___y_765_;
v___y_774_ = v___y_766_;
v___y_775_ = v___y_767_;
goto v___jp_769_;
}
}
v___jp_769_:
{
lean_object* v___x_776_; lean_object* v_h_777_; lean_object* v___x_778_; 
v___x_776_ = l_Lean_Syntax_getCanonicalAntiquot(v_s_761_);
v_h_777_ = l_Lean_Syntax_getAntiquotTerm(v___x_776_);
lean_dec(v___x_776_);
lean_inc(v___y_775_);
lean_inc_ref(v___y_774_);
lean_inc(v___y_773_);
lean_inc_ref(v___y_772_);
lean_inc(v___y_771_);
lean_inc_ref(v___y_770_);
v___x_778_ = lean_apply_8(v_expand_760_, v_h_777_, v___y_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_, lean_box(0));
if (lean_obj_tag(v___x_778_) == 0)
{
lean_object* v_a_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_787_; 
v_a_779_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_787_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_787_ == 0)
{
v___x_781_ = v___x_778_;
v_isShared_782_ = v_isSharedCheck_787_;
goto v_resetjp_780_;
}
else
{
lean_inc(v_a_779_);
lean_dec(v___x_778_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_787_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v___x_783_; lean_object* v___x_785_; 
v___x_783_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_783_, 0, v_a_779_);
if (v_isShared_782_ == 0)
{
lean_ctor_set(v___x_781_, 0, v___x_783_);
v___x_785_ = v___x_781_;
goto v_reusejp_784_;
}
else
{
lean_object* v_reuseFailAlloc_786_; 
v_reuseFailAlloc_786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_786_, 0, v___x_783_);
v___x_785_ = v_reuseFailAlloc_786_;
goto v_reusejp_784_;
}
v_reusejp_784_:
{
return v___x_785_;
}
}
}
else
{
lean_object* v_a_788_; lean_object* v___x_790_; uint8_t v_isShared_791_; uint8_t v_isSharedCheck_795_; 
v_a_788_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_795_ == 0)
{
v___x_790_ = v___x_778_;
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
else
{
lean_inc(v_a_788_);
lean_dec(v___x_778_);
v___x_790_ = lean_box(0);
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
v_resetjp_789_:
{
lean_object* v___x_793_; 
if (v_isShared_791_ == 0)
{
v___x_793_ = v___x_790_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_a_788_);
v___x_793_ = v_reuseFailAlloc_794_;
goto v_reusejp_792_;
}
v_reusejp_792_:
{
return v___x_793_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0___boxed(lean_object* v_expand_811_, lean_object* v_s_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_){
_start:
{
lean_object* v_res_820_; 
v_res_820_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0(v_expand_811_, v_s_812_, v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_, v___y_818_);
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
lean_dec(v___y_816_);
lean_dec_ref(v___y_815_);
lean_dec(v___y_814_);
lean_dec_ref(v___y_813_);
return v_res_820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2(lean_object* v_expand_821_, lean_object* v_x_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_){
_start:
{
if (lean_obj_tag(v_x_822_) == 1)
{
lean_object* v_info_830_; lean_object* v_kind_831_; lean_object* v_args_832_; lean_object* v___x_833_; 
v_info_830_ = lean_ctor_get(v_x_822_, 0);
lean_inc(v_info_830_);
v_kind_831_ = lean_ctor_get(v_x_822_, 1);
lean_inc(v_kind_831_);
v_args_832_ = lean_ctor_get(v_x_822_, 2);
lean_inc_ref(v_args_832_);
lean_inc_ref(v_expand_821_);
v___x_833_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0(v_expand_821_, v_x_822_, v___y_823_, v___y_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_);
if (lean_obj_tag(v___x_833_) == 0)
{
lean_object* v_a_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_862_; 
v_a_834_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_862_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_862_ == 0)
{
v___x_836_ = v___x_833_;
v_isShared_837_ = v_isSharedCheck_862_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_a_834_);
lean_dec(v___x_833_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_862_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
if (lean_obj_tag(v_a_834_) == 0)
{
size_t v_sz_838_; size_t v___x_839_; lean_object* v___x_840_; 
lean_del_object(v___x_836_);
v_sz_838_ = lean_array_size(v_args_832_);
v___x_839_ = ((size_t)0ULL);
v___x_840_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2_spec__2(v_expand_821_, v_sz_838_, v___x_839_, v_args_832_, v___y_823_, v___y_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_);
if (lean_obj_tag(v___x_840_) == 0)
{
lean_object* v_a_841_; lean_object* v___x_843_; uint8_t v_isShared_844_; uint8_t v_isSharedCheck_849_; 
v_a_841_ = lean_ctor_get(v___x_840_, 0);
v_isSharedCheck_849_ = !lean_is_exclusive(v___x_840_);
if (v_isSharedCheck_849_ == 0)
{
v___x_843_ = v___x_840_;
v_isShared_844_ = v_isSharedCheck_849_;
goto v_resetjp_842_;
}
else
{
lean_inc(v_a_841_);
lean_dec(v___x_840_);
v___x_843_ = lean_box(0);
v_isShared_844_ = v_isSharedCheck_849_;
goto v_resetjp_842_;
}
v_resetjp_842_:
{
lean_object* v___x_845_; lean_object* v___x_847_; 
v___x_845_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_845_, 0, v_info_830_);
lean_ctor_set(v___x_845_, 1, v_kind_831_);
lean_ctor_set(v___x_845_, 2, v_a_841_);
if (v_isShared_844_ == 0)
{
lean_ctor_set(v___x_843_, 0, v___x_845_);
v___x_847_ = v___x_843_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_848_; 
v_reuseFailAlloc_848_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_848_, 0, v___x_845_);
v___x_847_ = v_reuseFailAlloc_848_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
return v___x_847_;
}
}
}
else
{
lean_object* v_a_850_; lean_object* v___x_852_; uint8_t v_isShared_853_; uint8_t v_isSharedCheck_857_; 
lean_dec(v_kind_831_);
lean_dec(v_info_830_);
v_a_850_ = lean_ctor_get(v___x_840_, 0);
v_isSharedCheck_857_ = !lean_is_exclusive(v___x_840_);
if (v_isSharedCheck_857_ == 0)
{
v___x_852_ = v___x_840_;
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
else
{
lean_inc(v_a_850_);
lean_dec(v___x_840_);
v___x_852_ = lean_box(0);
v_isShared_853_ = v_isSharedCheck_857_;
goto v_resetjp_851_;
}
v_resetjp_851_:
{
lean_object* v___x_855_; 
if (v_isShared_853_ == 0)
{
v___x_855_ = v___x_852_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v_a_850_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
return v___x_855_;
}
}
}
}
else
{
lean_object* v_val_858_; lean_object* v___x_860_; 
lean_dec_ref(v_args_832_);
lean_dec(v_kind_831_);
lean_dec(v_info_830_);
lean_dec_ref(v_expand_821_);
v_val_858_ = lean_ctor_get(v_a_834_, 0);
lean_inc(v_val_858_);
lean_dec_ref_known(v_a_834_, 1);
if (v_isShared_837_ == 0)
{
lean_ctor_set(v___x_836_, 0, v_val_858_);
v___x_860_ = v___x_836_;
goto v_reusejp_859_;
}
else
{
lean_object* v_reuseFailAlloc_861_; 
v_reuseFailAlloc_861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_861_, 0, v_val_858_);
v___x_860_ = v_reuseFailAlloc_861_;
goto v_reusejp_859_;
}
v_reusejp_859_:
{
return v___x_860_;
}
}
}
}
else
{
lean_object* v_a_863_; lean_object* v___x_865_; uint8_t v_isShared_866_; uint8_t v_isSharedCheck_870_; 
lean_dec_ref(v_args_832_);
lean_dec(v_kind_831_);
lean_dec(v_info_830_);
lean_dec_ref(v_expand_821_);
v_a_863_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_870_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_870_ == 0)
{
v___x_865_ = v___x_833_;
v_isShared_866_ = v_isSharedCheck_870_;
goto v_resetjp_864_;
}
else
{
lean_inc(v_a_863_);
lean_dec(v___x_833_);
v___x_865_ = lean_box(0);
v_isShared_866_ = v_isSharedCheck_870_;
goto v_resetjp_864_;
}
v_resetjp_864_:
{
lean_object* v___x_868_; 
if (v_isShared_866_ == 0)
{
v___x_868_ = v___x_865_;
goto v_reusejp_867_;
}
else
{
lean_object* v_reuseFailAlloc_869_; 
v_reuseFailAlloc_869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_869_, 0, v_a_863_);
v___x_868_ = v_reuseFailAlloc_869_;
goto v_reusejp_867_;
}
v_reusejp_867_:
{
return v___x_868_;
}
}
}
}
else
{
lean_object* v___x_871_; 
lean_inc(v_x_822_);
v___x_871_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___lam__0(v_expand_821_, v_x_822_, v___y_823_, v___y_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_);
if (lean_obj_tag(v___x_871_) == 0)
{
lean_object* v_a_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_883_; 
v_a_872_ = lean_ctor_get(v___x_871_, 0);
v_isSharedCheck_883_ = !lean_is_exclusive(v___x_871_);
if (v_isSharedCheck_883_ == 0)
{
v___x_874_ = v___x_871_;
v_isShared_875_ = v_isSharedCheck_883_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_a_872_);
lean_dec(v___x_871_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_883_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
if (lean_obj_tag(v_a_872_) == 0)
{
lean_object* v___x_877_; 
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 0, v_x_822_);
v___x_877_ = v___x_874_;
goto v_reusejp_876_;
}
else
{
lean_object* v_reuseFailAlloc_878_; 
v_reuseFailAlloc_878_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_878_, 0, v_x_822_);
v___x_877_ = v_reuseFailAlloc_878_;
goto v_reusejp_876_;
}
v_reusejp_876_:
{
return v___x_877_;
}
}
else
{
lean_object* v_val_879_; lean_object* v___x_881_; 
lean_dec(v_x_822_);
v_val_879_ = lean_ctor_get(v_a_872_, 0);
lean_inc(v_val_879_);
lean_dec_ref_known(v_a_872_, 1);
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 0, v_val_879_);
v___x_881_ = v___x_874_;
goto v_reusejp_880_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_val_879_);
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
lean_dec(v_x_822_);
v_a_884_ = lean_ctor_get(v___x_871_, 0);
v_isSharedCheck_891_ = !lean_is_exclusive(v___x_871_);
if (v_isSharedCheck_891_ == 0)
{
v___x_886_ = v___x_871_;
v_isShared_887_ = v_isSharedCheck_891_;
goto v_resetjp_885_;
}
else
{
lean_inc(v_a_884_);
lean_dec(v___x_871_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2_spec__2(lean_object* v_expand_892_, size_t v_sz_893_, size_t v_i_894_, lean_object* v_bs_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_){
_start:
{
uint8_t v___x_903_; 
v___x_903_ = lean_usize_dec_lt(v_i_894_, v_sz_893_);
if (v___x_903_ == 0)
{
lean_object* v___x_904_; 
lean_dec_ref(v_expand_892_);
v___x_904_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_904_, 0, v_bs_895_);
return v___x_904_;
}
else
{
lean_object* v_v_905_; lean_object* v___x_906_; 
v_v_905_ = lean_array_uget_borrowed(v_bs_895_, v_i_894_);
lean_inc(v_v_905_);
lean_inc_ref(v_expand_892_);
v___x_906_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2(v_expand_892_, v_v_905_, v___y_896_, v___y_897_, v___y_898_, v___y_899_, v___y_900_, v___y_901_);
if (lean_obj_tag(v___x_906_) == 0)
{
lean_object* v_a_907_; lean_object* v___x_908_; lean_object* v_bs_x27_909_; size_t v___x_910_; size_t v___x_911_; lean_object* v___x_912_; 
v_a_907_ = lean_ctor_get(v___x_906_, 0);
lean_inc(v_a_907_);
lean_dec_ref_known(v___x_906_, 1);
v___x_908_ = lean_unsigned_to_nat(0u);
v_bs_x27_909_ = lean_array_uset(v_bs_895_, v_i_894_, v___x_908_);
v___x_910_ = ((size_t)1ULL);
v___x_911_ = lean_usize_add(v_i_894_, v___x_910_);
v___x_912_ = lean_array_uset(v_bs_x27_909_, v_i_894_, v_a_907_);
v_i_894_ = v___x_911_;
v_bs_895_ = v___x_912_;
goto _start;
}
else
{
lean_object* v_a_914_; lean_object* v___x_916_; uint8_t v_isShared_917_; uint8_t v_isSharedCheck_921_; 
lean_dec_ref(v_bs_895_);
lean_dec_ref(v_expand_892_);
v_a_914_ = lean_ctor_get(v___x_906_, 0);
v_isSharedCheck_921_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_921_ == 0)
{
v___x_916_ = v___x_906_;
v_isShared_917_ = v_isSharedCheck_921_;
goto v_resetjp_915_;
}
else
{
lean_inc(v_a_914_);
lean_dec(v___x_906_);
v___x_916_ = lean_box(0);
v_isShared_917_ = v_isSharedCheck_921_;
goto v_resetjp_915_;
}
v_resetjp_915_:
{
lean_object* v___x_919_; 
if (v_isShared_917_ == 0)
{
v___x_919_ = v___x_916_;
goto v_reusejp_918_;
}
else
{
lean_object* v_reuseFailAlloc_920_; 
v_reuseFailAlloc_920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_920_, 0, v_a_914_);
v___x_919_ = v_reuseFailAlloc_920_;
goto v_reusejp_918_;
}
v_reusejp_918_:
{
return v___x_919_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2_spec__2___boxed(lean_object* v_expand_922_, lean_object* v_sz_923_, lean_object* v_i_924_, lean_object* v_bs_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
size_t v_sz_boxed_933_; size_t v_i_boxed_934_; lean_object* v_res_935_; 
v_sz_boxed_933_ = lean_unbox_usize(v_sz_923_);
lean_dec(v_sz_923_);
v_i_boxed_934_ = lean_unbox_usize(v_i_924_);
lean_dec(v_i_924_);
v_res_935_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2_spec__2(v_expand_922_, v_sz_boxed_933_, v_i_boxed_934_, v_bs_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_);
lean_dec(v___y_931_);
lean_dec_ref(v___y_930_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
lean_dec(v___y_927_);
lean_dec_ref(v___y_926_);
return v_res_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2___boxed(lean_object* v_expand_936_, lean_object* v_x_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_){
_start:
{
lean_object* v_res_945_; 
v_res_945_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2(v_expand_936_, v_x_937_, v___y_938_, v___y_939_, v___y_940_, v___y_941_, v___y_942_, v___y_943_);
lean_dec(v___y_943_);
lean_dec_ref(v___y_942_);
lean_dec(v___y_941_);
lean_dec_ref(v___y_940_);
lean_dec(v___y_939_);
lean_dec_ref(v___y_938_);
return v_res_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_processAntiquot(lean_object* v_t_946_, lean_object* v_expand_947_, lean_object* v_a_948_, lean_object* v_a_949_, lean_object* v_a_950_, lean_object* v_a_951_, lean_object* v_a_952_, lean_object* v_a_953_){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__2(v_expand_947_, v_t_946_, v_a_948_, v_a_949_, v_a_950_, v_a_951_, v_a_952_, v_a_953_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_processAntiquot___boxed(lean_object* v_t_972_, lean_object* v_expand_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_, lean_object* v_a_978_, lean_object* v_a_979_, lean_object* v_a_980_){
_start:
{
lean_object* v_res_981_; 
v_res_981_ = lp_mathlib_Mathlib_Tactic_TermCongr_processAntiquot(v_t_972_, v_expand_973_, v_a_974_, v_a_975_, v_a_976_, v_a_977_, v_a_978_, v_a_979_);
lean_dec(v_a_979_);
lean_dec_ref(v_a_978_);
lean_dec(v_a_977_);
lean_dec_ref(v_a_976_);
lean_dec(v_a_975_);
lean_dec_ref(v_a_974_);
return v_res_981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1(lean_object* v_00_u03b1_982_, lean_object* v_ref_983_, lean_object* v_msg_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_){
_start:
{
lean_object* v___x_992_; 
v___x_992_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___redArg(v_ref_983_, v_msg_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_, v___y_990_);
return v___x_992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1___boxed(lean_object* v_00_u03b1_993_, lean_object* v_ref_994_, lean_object* v_msg_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_){
_start:
{
lean_object* v_res_1003_; 
v_res_1003_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_TermCongr_processAntiquot_spec__1(v_00_u03b1_993_, v_ref_994_, v_msg_995_, v___y_996_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_);
lean_dec(v___y_1001_);
lean_dec_ref(v___y_1000_);
lean_dec(v___y_999_);
lean_dec_ref(v___y_998_);
lean_dec(v___y_997_);
lean_dec_ref(v___y_996_);
lean_dec(v_ref_994_);
return v_res_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0___redArg(lean_object* v_a_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
lean_object* v___x_1012_; 
v___x_1012_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_1004_, v___y_1005_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_, v___y_1010_);
return v___x_1012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0___redArg___boxed(lean_object* v_a_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_){
_start:
{
lean_object* v_res_1021_; 
v_res_1021_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0___redArg(v_a_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
lean_dec(v___y_1019_);
lean_dec_ref(v___y_1018_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
return v_res_1021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0(lean_object* v_00_u03b1_1022_, lean_object* v_a_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_){
_start:
{
lean_object* v___x_1031_; 
v___x_1031_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_1023_, v___y_1024_, v___y_1025_, v___y_1026_, v___y_1027_, v___y_1028_, v___y_1029_);
return v___x_1031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0___boxed(lean_object* v_00_u03b1_1032_, lean_object* v_a_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_){
_start:
{
lean_object* v_res_1041_; 
v_res_1041_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_TermCongr_elaboratePattern_spec__0(v_00_u03b1_1032_, v_a_1033_, v___y_1034_, v___y_1035_, v___y_1036_, v___y_1037_, v___y_1038_, v___y_1039_);
lean_dec(v___y_1039_);
lean_dec_ref(v___y_1038_);
lean_dec(v___y_1037_);
lean_dec_ref(v___y_1036_);
lean_dec(v___y_1035_);
lean_dec_ref(v___y_1034_);
return v_res_1041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0(uint8_t v_forLhs_1043_, lean_object* v_h_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
if (v_forLhs_1043_ == 0)
{
lean_object* v_ref_1052_; lean_object* v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; 
v_ref_1052_ = lean_ctor_get(v___y_1049_, 5);
v___x_1053_ = l_Lean_SourceInfo_fromRef(v_ref_1052_, v_forLhs_1043_);
v___x_1054_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1));
v___x_1055_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___closed__0));
lean_inc_n(v___x_1053_, 3);
v___x_1056_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1053_);
lean_ctor_set(v___x_1056_, 1, v___x_1055_);
v___x_1057_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__11));
v___x_1058_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__12));
v___x_1059_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1059_, 0, v___x_1053_);
lean_ctor_set(v___x_1059_, 1, v___x_1057_);
v___x_1060_ = l_Lean_Syntax_node1(v___x_1053_, v___x_1058_, v___x_1059_);
v___x_1061_ = l_Lean_Syntax_node3(v___x_1053_, v___x_1054_, v___x_1056_, v___x_1060_, v_h_1044_);
v___x_1062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1061_);
return v___x_1062_;
}
else
{
lean_object* v_ref_1063_; uint8_t v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; 
v_ref_1063_ = lean_ctor_get(v___y_1049_, 5);
v___x_1064_ = 0;
v___x_1065_ = l_Lean_SourceInfo_fromRef(v_ref_1063_, v___x_1064_);
v___x_1066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__1));
v___x_1067_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___closed__0));
lean_inc_n(v___x_1065_, 3);
v___x_1068_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1068_, 0, v___x_1065_);
lean_ctor_set(v___x_1068_, 1, v___x_1067_);
v___x_1069_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__6));
v___x_1070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_cHoleExpand___closed__8));
v___x_1071_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1071_, 0, v___x_1065_);
lean_ctor_set(v___x_1071_, 1, v___x_1069_);
v___x_1072_ = l_Lean_Syntax_node1(v___x_1065_, v___x_1070_, v___x_1071_);
v___x_1073_ = l_Lean_Syntax_node3(v___x_1065_, v___x_1066_, v___x_1068_, v___x_1072_, v_h_1044_);
v___x_1074_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1074_, 0, v___x_1073_);
return v___x_1074_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___boxed(lean_object* v_forLhs_1075_, lean_object* v_h_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_){
_start:
{
uint8_t v_forLhs_boxed_1084_; lean_object* v_res_1085_; 
v_forLhs_boxed_1084_ = lean_unbox(v_forLhs_1075_);
v_res_1085_ = lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0(v_forLhs_boxed_1084_, v_h_1076_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_);
lean_dec(v___y_1082_);
lean_dec_ref(v___y_1081_);
lean_dec(v___y_1080_);
lean_dec_ref(v___y_1079_);
lean_dec(v___y_1078_);
lean_dec_ref(v___y_1077_);
return v_res_1085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__1(lean_object* v_t_1086_, lean_object* v___f_1087_, lean_object* v_expectedType_x3f_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_){
_start:
{
lean_object* v___x_1096_; 
v___x_1096_ = lp_mathlib_Mathlib_Tactic_TermCongr_processAntiquot(v_t_1086_, v___f_1087_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
if (lean_obj_tag(v___x_1096_) == 0)
{
lean_object* v_a_1097_; uint8_t v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; 
v_a_1097_ = lean_ctor_get(v___x_1096_, 0);
lean_inc(v_a_1097_);
lean_dec_ref_known(v___x_1096_, 1);
v___x_1098_ = 1;
v___x_1099_ = lean_box(0);
v___x_1100_ = l_Lean_Elab_Term_elabTermEnsuringType(v_a_1097_, v_expectedType_x3f_1088_, v___x_1098_, v___x_1098_, v___x_1099_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
return v___x_1100_;
}
else
{
lean_object* v_a_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1108_; 
lean_dec(v_expectedType_x3f_1088_);
v_a_1101_ = lean_ctor_get(v___x_1096_, 0);
v_isSharedCheck_1108_ = !lean_is_exclusive(v___x_1096_);
if (v_isSharedCheck_1108_ == 0)
{
v___x_1103_ = v___x_1096_;
v_isShared_1104_ = v_isSharedCheck_1108_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_a_1101_);
lean_dec(v___x_1096_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1108_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v___x_1106_; 
if (v_isShared_1104_ == 0)
{
v___x_1106_ = v___x_1103_;
goto v_reusejp_1105_;
}
else
{
lean_object* v_reuseFailAlloc_1107_; 
v_reuseFailAlloc_1107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1107_, 0, v_a_1101_);
v___x_1106_ = v_reuseFailAlloc_1107_;
goto v_reusejp_1105_;
}
v_reusejp_1105_:
{
return v___x_1106_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__1___boxed(lean_object* v_t_1109_, lean_object* v___f_1110_, lean_object* v_expectedType_x3f_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_){
_start:
{
lean_object* v_res_1119_; 
v_res_1119_ = lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__1(v_t_1109_, v___f_1110_, v_expectedType_x3f_1111_, v___y_1112_, v___y_1113_, v___y_1114_, v___y_1115_, v___y_1116_, v___y_1117_);
lean_dec(v___y_1117_);
lean_dec_ref(v___y_1116_);
lean_dec(v___y_1115_);
lean_dec_ref(v___y_1114_);
lean_dec(v___y_1113_);
lean_dec_ref(v___y_1112_);
return v_res_1119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern(lean_object* v_t_1120_, lean_object* v_expectedType_x3f_1121_, uint8_t v_forLhs_1122_, lean_object* v_a_1123_, lean_object* v_a_1124_, lean_object* v_a_1125_, lean_object* v_a_1126_, lean_object* v_a_1127_, lean_object* v_a_1128_){
_start:
{
lean_object* v___x_1130_; lean_object* v___f_1131_; lean_object* v___f_1132_; lean_object* v___x_1133_; 
v___x_1130_ = lean_box(v_forLhs_1122_);
v___f_1131_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__0___boxed), 9, 1);
lean_closure_set(v___f_1131_, 0, v___x_1130_);
v___f_1132_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___lam__1___boxed), 10, 3);
lean_closure_set(v___f_1132_, 0, v_t_1120_);
lean_closure_set(v___f_1132_, 1, v___f_1131_);
lean_closure_set(v___f_1132_, 2, v_expectedType_x3f_1121_);
v___x_1133_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___f_1132_, v_a_1123_, v_a_1124_, v_a_1125_, v_a_1126_, v_a_1127_, v_a_1128_);
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern___boxed(lean_object* v_t_1134_, lean_object* v_expectedType_x3f_1135_, lean_object* v_forLhs_1136_, lean_object* v_a_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_, lean_object* v_a_1141_, lean_object* v_a_1142_, lean_object* v_a_1143_){
_start:
{
uint8_t v_forLhs_boxed_1144_; lean_object* v_res_1145_; 
v_forLhs_boxed_1144_ = lean_unbox(v_forLhs_1136_);
v_res_1145_ = lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern(v_t_1134_, v_expectedType_x3f_1135_, v_forLhs_boxed_1144_, v_a_1137_, v_a_1138_, v_a_1139_, v_a_1140_, v_a_1141_, v_a_1142_);
lean_dec(v_a_1142_);
lean_dec_ref(v_a_1141_);
lean_dec(v_a_1140_);
lean_dec_ref(v_a_1139_);
lean_dec(v_a_1138_);
lean_dec_ref(v_a_1137_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(lean_object* v_msg_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_){
_start:
{
lean_object* v_ref_1152_; lean_object* v___x_1153_; lean_object* v_a_1154_; lean_object* v___x_1156_; uint8_t v_isShared_1157_; uint8_t v_isSharedCheck_1162_; 
v_ref_1152_ = lean_ctor_get(v___y_1149_, 5);
v___x_1153_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(v_msg_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_);
v_a_1154_ = lean_ctor_get(v___x_1153_, 0);
v_isSharedCheck_1162_ = !lean_is_exclusive(v___x_1153_);
if (v_isSharedCheck_1162_ == 0)
{
v___x_1156_ = v___x_1153_;
v_isShared_1157_ = v_isSharedCheck_1162_;
goto v_resetjp_1155_;
}
else
{
lean_inc(v_a_1154_);
lean_dec(v___x_1153_);
v___x_1156_ = lean_box(0);
v_isShared_1157_ = v_isSharedCheck_1162_;
goto v_resetjp_1155_;
}
v_resetjp_1155_:
{
lean_object* v___x_1158_; lean_object* v___x_1160_; 
lean_inc(v_ref_1152_);
v___x_1158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1158_, 0, v_ref_1152_);
lean_ctor_set(v___x_1158_, 1, v_a_1154_);
if (v_isShared_1157_ == 0)
{
lean_ctor_set_tag(v___x_1156_, 1);
lean_ctor_set(v___x_1156_, 0, v___x_1158_);
v___x_1160_ = v___x_1156_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v___x_1158_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg___boxed(lean_object* v_msg_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_){
_start:
{
lean_object* v_res_1169_; 
v_res_1169_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v_msg_1163_, v___y_1164_, v___y_1165_, v___y_1166_, v___y_1167_);
lean_dec(v___y_1167_);
lean_dec_ref(v___y_1166_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
return v_res_1169_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3(void){
_start:
{
lean_object* v___x_1174_; lean_object* v___x_1175_; 
v___x_1174_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__2));
v___x_1175_ = l_Lean_stringToMessageData(v___x_1174_);
return v___x_1175_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__5(void){
_start:
{
lean_object* v___x_1177_; lean_object* v___x_1178_; 
v___x_1177_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__4));
v___x_1178_ = l_Lean_stringToMessageData(v___x_1177_);
return v___x_1178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType(lean_object* v_expectedType_x3f_1179_, lean_object* v_a_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_, lean_object* v_a_1183_){
_start:
{
lean_object* v___x_1185_; 
v___x_1185_ = l_Lean_Meta_mkFreshLevelMVar(v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_);
if (lean_obj_tag(v___x_1185_) == 0)
{
lean_object* v_a_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; uint8_t v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; 
v_a_1186_ = lean_ctor_get(v___x_1185_, 0);
lean_inc_n(v_a_1186_, 2);
lean_dec_ref_known(v___x_1185_, 1);
v___x_1187_ = l_Lean_mkSort(v_a_1186_);
v___x_1188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1188_, 0, v___x_1187_);
v___x_1189_ = 0;
v___x_1190_ = lean_box(0);
v___x_1191_ = l_Lean_Meta_mkFreshExprMVar(v___x_1188_, v___x_1189_, v___x_1190_, v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_);
if (lean_obj_tag(v___x_1191_) == 0)
{
lean_object* v_a_1192_; lean_object* v___x_1194_; uint8_t v_isShared_1195_; uint8_t v_isSharedCheck_1249_; 
v_a_1192_ = lean_ctor_get(v___x_1191_, 0);
v_isSharedCheck_1249_ = !lean_is_exclusive(v___x_1191_);
if (v_isSharedCheck_1249_ == 0)
{
v___x_1194_ = v___x_1191_;
v_isShared_1195_ = v_isSharedCheck_1249_;
goto v_resetjp_1193_;
}
else
{
lean_inc(v_a_1192_);
lean_dec(v___x_1191_);
v___x_1194_ = lean_box(0);
v_isShared_1195_ = v_isSharedCheck_1249_;
goto v_resetjp_1193_;
}
v_resetjp_1193_:
{
lean_object* v___x_1197_; 
lean_inc(v_a_1192_);
if (v_isShared_1195_ == 0)
{
lean_ctor_set_tag(v___x_1194_, 1);
v___x_1197_ = v___x_1194_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1248_; 
v_reuseFailAlloc_1248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1248_, 0, v_a_1192_);
v___x_1197_ = v_reuseFailAlloc_1248_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
lean_object* v___x_1198_; 
lean_inc_ref(v___x_1197_);
v___x_1198_ = l_Lean_Meta_mkFreshExprMVar(v___x_1197_, v___x_1189_, v___x_1190_, v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_);
if (lean_obj_tag(v___x_1198_) == 0)
{
lean_object* v_a_1199_; lean_object* v___x_1200_; 
v_a_1199_ = lean_ctor_get(v___x_1198_, 0);
lean_inc(v_a_1199_);
lean_dec_ref_known(v___x_1198_, 1);
v___x_1200_ = l_Lean_Meta_mkFreshExprMVar(v___x_1197_, v___x_1189_, v___x_1190_, v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_);
if (lean_obj_tag(v___x_1200_) == 0)
{
lean_object* v_a_1201_; lean_object* v___x_1203_; uint8_t v_isShared_1204_; uint8_t v_isSharedCheck_1247_; 
v_a_1201_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1247_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1247_ == 0)
{
v___x_1203_ = v___x_1200_;
v_isShared_1204_ = v_isSharedCheck_1247_;
goto v_resetjp_1202_;
}
else
{
lean_inc(v_a_1201_);
lean_dec(v___x_1200_);
v___x_1203_ = lean_box(0);
v_isShared_1204_ = v_isSharedCheck_1247_;
goto v_resetjp_1202_;
}
v_resetjp_1202_:
{
lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; 
v___x_1205_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__1));
v___x_1206_ = lean_box(0);
v___x_1207_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1207_, 0, v_a_1186_);
lean_ctor_set(v___x_1207_, 1, v___x_1206_);
v___x_1208_ = l_Lean_mkConst(v___x_1205_, v___x_1207_);
v___x_1209_ = l_Lean_mkApp3(v___x_1208_, v_a_1192_, v_a_1199_, v_a_1201_);
if (lean_obj_tag(v_expectedType_x3f_1179_) == 1)
{
lean_object* v_val_1210_; lean_object* v___x_1211_; 
lean_del_object(v___x_1203_);
v_val_1210_ = lean_ctor_get(v_expectedType_x3f_1179_, 0);
lean_inc_n(v_val_1210_, 2);
lean_dec_ref_known(v_expectedType_x3f_1179_, 1);
lean_inc_ref(v___x_1209_);
v___x_1211_ = l_Lean_Meta_isExprDefEq(v_val_1210_, v___x_1209_, v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_);
if (lean_obj_tag(v___x_1211_) == 0)
{
lean_object* v_a_1212_; lean_object* v___x_1214_; uint8_t v_isShared_1215_; uint8_t v_isSharedCheck_1235_; 
v_a_1212_ = lean_ctor_get(v___x_1211_, 0);
v_isSharedCheck_1235_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1235_ == 0)
{
v___x_1214_ = v___x_1211_;
v_isShared_1215_ = v_isSharedCheck_1235_;
goto v_resetjp_1213_;
}
else
{
lean_inc(v_a_1212_);
lean_dec(v___x_1211_);
v___x_1214_ = lean_box(0);
v_isShared_1215_ = v_isSharedCheck_1235_;
goto v_resetjp_1213_;
}
v_resetjp_1213_:
{
uint8_t v___x_1216_; 
v___x_1216_ = lean_unbox(v_a_1212_);
lean_dec(v_a_1212_);
if (v___x_1216_ == 0)
{
lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v_a_1224_; lean_object* v___x_1226_; uint8_t v_isShared_1227_; uint8_t v_isSharedCheck_1231_; 
lean_del_object(v___x_1214_);
lean_dec_ref(v___x_1209_);
v___x_1217_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3);
v___x_1218_ = l_Lean_MessageData_ofExpr(v_val_1210_);
v___x_1219_ = l_Lean_indentD(v___x_1218_);
v___x_1220_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1220_, 0, v___x_1217_);
lean_ctor_set(v___x_1220_, 1, v___x_1219_);
v___x_1221_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__5, &lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__5);
v___x_1222_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1222_, 0, v___x_1220_);
lean_ctor_set(v___x_1222_, 1, v___x_1221_);
v___x_1223_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_1222_, v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_);
v_a_1224_ = lean_ctor_get(v___x_1223_, 0);
v_isSharedCheck_1231_ = !lean_is_exclusive(v___x_1223_);
if (v_isSharedCheck_1231_ == 0)
{
v___x_1226_ = v___x_1223_;
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
else
{
lean_inc(v_a_1224_);
lean_dec(v___x_1223_);
v___x_1226_ = lean_box(0);
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
v_resetjp_1225_:
{
lean_object* v___x_1229_; 
if (v_isShared_1227_ == 0)
{
v___x_1229_ = v___x_1226_;
goto v_reusejp_1228_;
}
else
{
lean_object* v_reuseFailAlloc_1230_; 
v_reuseFailAlloc_1230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1230_, 0, v_a_1224_);
v___x_1229_ = v_reuseFailAlloc_1230_;
goto v_reusejp_1228_;
}
v_reusejp_1228_:
{
return v___x_1229_;
}
}
}
else
{
lean_object* v___x_1233_; 
lean_dec(v_val_1210_);
if (v_isShared_1215_ == 0)
{
lean_ctor_set(v___x_1214_, 0, v___x_1209_);
v___x_1233_ = v___x_1214_;
goto v_reusejp_1232_;
}
else
{
lean_object* v_reuseFailAlloc_1234_; 
v_reuseFailAlloc_1234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1234_, 0, v___x_1209_);
v___x_1233_ = v_reuseFailAlloc_1234_;
goto v_reusejp_1232_;
}
v_reusejp_1232_:
{
return v___x_1233_;
}
}
}
}
else
{
lean_object* v_a_1236_; lean_object* v___x_1238_; uint8_t v_isShared_1239_; uint8_t v_isSharedCheck_1243_; 
lean_dec(v_val_1210_);
lean_dec_ref(v___x_1209_);
v_a_1236_ = lean_ctor_get(v___x_1211_, 0);
v_isSharedCheck_1243_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1243_ == 0)
{
v___x_1238_ = v___x_1211_;
v_isShared_1239_ = v_isSharedCheck_1243_;
goto v_resetjp_1237_;
}
else
{
lean_inc(v_a_1236_);
lean_dec(v___x_1211_);
v___x_1238_ = lean_box(0);
v_isShared_1239_ = v_isSharedCheck_1243_;
goto v_resetjp_1237_;
}
v_resetjp_1237_:
{
lean_object* v___x_1241_; 
if (v_isShared_1239_ == 0)
{
v___x_1241_ = v___x_1238_;
goto v_reusejp_1240_;
}
else
{
lean_object* v_reuseFailAlloc_1242_; 
v_reuseFailAlloc_1242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1242_, 0, v_a_1236_);
v___x_1241_ = v_reuseFailAlloc_1242_;
goto v_reusejp_1240_;
}
v_reusejp_1240_:
{
return v___x_1241_;
}
}
}
}
else
{
lean_object* v___x_1245_; 
lean_dec(v_expectedType_x3f_1179_);
if (v_isShared_1204_ == 0)
{
lean_ctor_set(v___x_1203_, 0, v___x_1209_);
v___x_1245_ = v___x_1203_;
goto v_reusejp_1244_;
}
else
{
lean_object* v_reuseFailAlloc_1246_; 
v_reuseFailAlloc_1246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1246_, 0, v___x_1209_);
v___x_1245_ = v_reuseFailAlloc_1246_;
goto v_reusejp_1244_;
}
v_reusejp_1244_:
{
return v___x_1245_;
}
}
}
}
else
{
lean_dec(v_a_1199_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v_expectedType_x3f_1179_);
return v___x_1200_;
}
}
else
{
lean_dec_ref(v___x_1197_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v_expectedType_x3f_1179_);
return v___x_1198_;
}
}
}
}
else
{
lean_dec(v_a_1186_);
lean_dec(v_expectedType_x3f_1179_);
return v___x_1191_;
}
}
else
{
lean_object* v_a_1250_; lean_object* v___x_1252_; uint8_t v_isShared_1253_; uint8_t v_isSharedCheck_1257_; 
lean_dec(v_expectedType_x3f_1179_);
v_a_1250_ = lean_ctor_get(v___x_1185_, 0);
v_isSharedCheck_1257_ = !lean_is_exclusive(v___x_1185_);
if (v_isSharedCheck_1257_ == 0)
{
v___x_1252_ = v___x_1185_;
v_isShared_1253_ = v_isSharedCheck_1257_;
goto v_resetjp_1251_;
}
else
{
lean_inc(v_a_1250_);
lean_dec(v___x_1185_);
v___x_1252_ = lean_box(0);
v_isShared_1253_ = v_isSharedCheck_1257_;
goto v_resetjp_1251_;
}
v_resetjp_1251_:
{
lean_object* v___x_1255_; 
if (v_isShared_1253_ == 0)
{
v___x_1255_ = v___x_1252_;
goto v_reusejp_1254_;
}
else
{
lean_object* v_reuseFailAlloc_1256_; 
v_reuseFailAlloc_1256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1256_, 0, v_a_1250_);
v___x_1255_ = v_reuseFailAlloc_1256_;
goto v_reusejp_1254_;
}
v_reusejp_1254_:
{
return v___x_1255_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___boxed(lean_object* v_expectedType_x3f_1258_, lean_object* v_a_1259_, lean_object* v_a_1260_, lean_object* v_a_1261_, lean_object* v_a_1262_, lean_object* v_a_1263_){
_start:
{
lean_object* v_res_1264_; 
v_res_1264_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType(v_expectedType_x3f_1258_, v_a_1259_, v_a_1260_, v_a_1261_, v_a_1262_);
lean_dec(v_a_1262_);
lean_dec_ref(v_a_1261_);
lean_dec(v_a_1260_);
lean_dec_ref(v_a_1259_);
return v_res_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0(lean_object* v_00_u03b1_1265_, lean_object* v_msg_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_){
_start:
{
lean_object* v___x_1272_; 
v___x_1272_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v_msg_1266_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_);
return v___x_1272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___boxed(lean_object* v_00_u03b1_1273_, lean_object* v_msg_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_){
_start:
{
lean_object* v_res_1280_; 
v_res_1280_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0(v_00_u03b1_1273_, v_msg_1274_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_);
lean_dec(v___y_1278_);
lean_dec_ref(v___y_1277_);
lean_dec(v___y_1276_);
lean_dec_ref(v___y_1275_);
return v_res_1280_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__3(void){
_start:
{
lean_object* v___x_1285_; lean_object* v___x_1286_; 
v___x_1285_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__2));
v___x_1286_ = l_Lean_stringToMessageData(v___x_1285_);
return v___x_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType(lean_object* v_expectedType_x3f_1287_, lean_object* v_a_1288_, lean_object* v_a_1289_, lean_object* v_a_1290_, lean_object* v_a_1291_){
_start:
{
lean_object* v___x_1293_; 
v___x_1293_ = l_Lean_Meta_mkFreshLevelMVar(v_a_1288_, v_a_1289_, v_a_1290_, v_a_1291_);
if (lean_obj_tag(v___x_1293_) == 0)
{
lean_object* v_a_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; uint8_t v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; 
v_a_1294_ = lean_ctor_get(v___x_1293_, 0);
lean_inc_n(v_a_1294_, 2);
lean_dec_ref_known(v___x_1293_, 1);
v___x_1295_ = l_Lean_mkSort(v_a_1294_);
v___x_1296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1296_, 0, v___x_1295_);
v___x_1297_ = 0;
v___x_1298_ = lean_box(0);
lean_inc_ref(v___x_1296_);
v___x_1299_ = l_Lean_Meta_mkFreshExprMVar(v___x_1296_, v___x_1297_, v___x_1298_, v_a_1288_, v_a_1289_, v_a_1290_, v_a_1291_);
if (lean_obj_tag(v___x_1299_) == 0)
{
lean_object* v_a_1300_; lean_object* v___x_1301_; 
v_a_1300_ = lean_ctor_get(v___x_1299_, 0);
lean_inc(v_a_1300_);
lean_dec_ref_known(v___x_1299_, 1);
v___x_1301_ = l_Lean_Meta_mkFreshExprMVar(v___x_1296_, v___x_1297_, v___x_1298_, v_a_1288_, v_a_1289_, v_a_1290_, v_a_1291_);
if (lean_obj_tag(v___x_1301_) == 0)
{
lean_object* v_a_1302_; lean_object* v___x_1304_; uint8_t v_isShared_1305_; uint8_t v_isSharedCheck_1366_; 
v_a_1302_ = lean_ctor_get(v___x_1301_, 0);
v_isSharedCheck_1366_ = !lean_is_exclusive(v___x_1301_);
if (v_isSharedCheck_1366_ == 0)
{
v___x_1304_ = v___x_1301_;
v_isShared_1305_ = v_isSharedCheck_1366_;
goto v_resetjp_1303_;
}
else
{
lean_inc(v_a_1302_);
lean_dec(v___x_1301_);
v___x_1304_ = lean_box(0);
v_isShared_1305_ = v_isSharedCheck_1366_;
goto v_resetjp_1303_;
}
v_resetjp_1303_:
{
lean_object* v___x_1307_; 
lean_inc(v_a_1300_);
if (v_isShared_1305_ == 0)
{
lean_ctor_set_tag(v___x_1304_, 1);
lean_ctor_set(v___x_1304_, 0, v_a_1300_);
v___x_1307_ = v___x_1304_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1365_; 
v_reuseFailAlloc_1365_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1365_, 0, v_a_1300_);
v___x_1307_ = v_reuseFailAlloc_1365_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
lean_object* v___x_1308_; 
v___x_1308_ = l_Lean_Meta_mkFreshExprMVar(v___x_1307_, v___x_1297_, v___x_1298_, v_a_1288_, v_a_1289_, v_a_1290_, v_a_1291_);
if (lean_obj_tag(v___x_1308_) == 0)
{
lean_object* v_a_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1364_; 
v_a_1309_ = lean_ctor_get(v___x_1308_, 0);
v_isSharedCheck_1364_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1364_ == 0)
{
v___x_1311_ = v___x_1308_;
v_isShared_1312_ = v_isSharedCheck_1364_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_a_1309_);
lean_dec(v___x_1308_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1364_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
lean_object* v___x_1314_; 
lean_inc(v_a_1302_);
if (v_isShared_1312_ == 0)
{
lean_ctor_set_tag(v___x_1311_, 1);
lean_ctor_set(v___x_1311_, 0, v_a_1302_);
v___x_1314_ = v___x_1311_;
goto v_reusejp_1313_;
}
else
{
lean_object* v_reuseFailAlloc_1363_; 
v_reuseFailAlloc_1363_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1363_, 0, v_a_1302_);
v___x_1314_ = v_reuseFailAlloc_1363_;
goto v_reusejp_1313_;
}
v_reusejp_1313_:
{
lean_object* v___x_1315_; 
v___x_1315_ = l_Lean_Meta_mkFreshExprMVar(v___x_1314_, v___x_1297_, v___x_1298_, v_a_1288_, v_a_1289_, v_a_1290_, v_a_1291_);
if (lean_obj_tag(v___x_1315_) == 0)
{
lean_object* v_a_1316_; lean_object* v___x_1318_; uint8_t v_isShared_1319_; uint8_t v_isSharedCheck_1362_; 
v_a_1316_ = lean_ctor_get(v___x_1315_, 0);
v_isSharedCheck_1362_ = !lean_is_exclusive(v___x_1315_);
if (v_isSharedCheck_1362_ == 0)
{
v___x_1318_ = v___x_1315_;
v_isShared_1319_ = v_isSharedCheck_1362_;
goto v_resetjp_1317_;
}
else
{
lean_inc(v_a_1316_);
lean_dec(v___x_1315_);
v___x_1318_ = lean_box(0);
v_isShared_1319_ = v_isSharedCheck_1362_;
goto v_resetjp_1317_;
}
v_resetjp_1317_:
{
lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; 
v___x_1320_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__1));
v___x_1321_ = lean_box(0);
v___x_1322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1322_, 0, v_a_1294_);
lean_ctor_set(v___x_1322_, 1, v___x_1321_);
v___x_1323_ = l_Lean_mkConst(v___x_1320_, v___x_1322_);
v___x_1324_ = l_Lean_mkApp4(v___x_1323_, v_a_1300_, v_a_1309_, v_a_1302_, v_a_1316_);
if (lean_obj_tag(v_expectedType_x3f_1287_) == 1)
{
lean_object* v_val_1325_; lean_object* v___x_1326_; 
lean_del_object(v___x_1318_);
v_val_1325_ = lean_ctor_get(v_expectedType_x3f_1287_, 0);
lean_inc_n(v_val_1325_, 2);
lean_dec_ref_known(v_expectedType_x3f_1287_, 1);
lean_inc_ref(v___x_1324_);
v___x_1326_ = l_Lean_Meta_isExprDefEq(v_val_1325_, v___x_1324_, v_a_1288_, v_a_1289_, v_a_1290_, v_a_1291_);
if (lean_obj_tag(v___x_1326_) == 0)
{
lean_object* v_a_1327_; lean_object* v___x_1329_; uint8_t v_isShared_1330_; uint8_t v_isSharedCheck_1350_; 
v_a_1327_ = lean_ctor_get(v___x_1326_, 0);
v_isSharedCheck_1350_ = !lean_is_exclusive(v___x_1326_);
if (v_isSharedCheck_1350_ == 0)
{
v___x_1329_ = v___x_1326_;
v_isShared_1330_ = v_isSharedCheck_1350_;
goto v_resetjp_1328_;
}
else
{
lean_inc(v_a_1327_);
lean_dec(v___x_1326_);
v___x_1329_ = lean_box(0);
v_isShared_1330_ = v_isSharedCheck_1350_;
goto v_resetjp_1328_;
}
v_resetjp_1328_:
{
uint8_t v___x_1331_; 
v___x_1331_ = lean_unbox(v_a_1327_);
lean_dec(v_a_1327_);
if (v___x_1331_ == 0)
{
lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v_a_1339_; lean_object* v___x_1341_; uint8_t v_isShared_1342_; uint8_t v_isSharedCheck_1346_; 
lean_del_object(v___x_1329_);
lean_dec_ref(v___x_1324_);
v___x_1332_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3);
v___x_1333_ = l_Lean_MessageData_ofExpr(v_val_1325_);
v___x_1334_ = l_Lean_indentD(v___x_1333_);
v___x_1335_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1335_, 0, v___x_1332_);
lean_ctor_set(v___x_1335_, 1, v___x_1334_);
v___x_1336_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__3);
v___x_1337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1337_, 0, v___x_1335_);
lean_ctor_set(v___x_1337_, 1, v___x_1336_);
v___x_1338_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_1337_, v_a_1288_, v_a_1289_, v_a_1290_, v_a_1291_);
v_a_1339_ = lean_ctor_get(v___x_1338_, 0);
v_isSharedCheck_1346_ = !lean_is_exclusive(v___x_1338_);
if (v_isSharedCheck_1346_ == 0)
{
v___x_1341_ = v___x_1338_;
v_isShared_1342_ = v_isSharedCheck_1346_;
goto v_resetjp_1340_;
}
else
{
lean_inc(v_a_1339_);
lean_dec(v___x_1338_);
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
else
{
lean_object* v___x_1348_; 
lean_dec(v_val_1325_);
if (v_isShared_1330_ == 0)
{
lean_ctor_set(v___x_1329_, 0, v___x_1324_);
v___x_1348_ = v___x_1329_;
goto v_reusejp_1347_;
}
else
{
lean_object* v_reuseFailAlloc_1349_; 
v_reuseFailAlloc_1349_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1349_, 0, v___x_1324_);
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
lean_object* v_a_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1358_; 
lean_dec(v_val_1325_);
lean_dec_ref(v___x_1324_);
v_a_1351_ = lean_ctor_get(v___x_1326_, 0);
v_isSharedCheck_1358_ = !lean_is_exclusive(v___x_1326_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1353_ = v___x_1326_;
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_a_1351_);
lean_dec(v___x_1326_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v___x_1356_; 
if (v_isShared_1354_ == 0)
{
v___x_1356_ = v___x_1353_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v_a_1351_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
return v___x_1356_;
}
}
}
}
else
{
lean_object* v___x_1360_; 
lean_dec(v_expectedType_x3f_1287_);
if (v_isShared_1319_ == 0)
{
lean_ctor_set(v___x_1318_, 0, v___x_1324_);
v___x_1360_ = v___x_1318_;
goto v_reusejp_1359_;
}
else
{
lean_object* v_reuseFailAlloc_1361_; 
v_reuseFailAlloc_1361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1361_, 0, v___x_1324_);
v___x_1360_ = v_reuseFailAlloc_1361_;
goto v_reusejp_1359_;
}
v_reusejp_1359_:
{
return v___x_1360_;
}
}
}
}
else
{
lean_dec(v_a_1309_);
lean_dec(v_a_1302_);
lean_dec(v_a_1300_);
lean_dec(v_a_1294_);
lean_dec(v_expectedType_x3f_1287_);
return v___x_1315_;
}
}
}
}
else
{
lean_dec(v_a_1302_);
lean_dec(v_a_1300_);
lean_dec(v_a_1294_);
lean_dec(v_expectedType_x3f_1287_);
return v___x_1308_;
}
}
}
}
else
{
lean_dec(v_a_1300_);
lean_dec(v_a_1294_);
lean_dec(v_expectedType_x3f_1287_);
return v___x_1301_;
}
}
else
{
lean_dec_ref_known(v___x_1296_, 1);
lean_dec(v_a_1294_);
lean_dec(v_expectedType_x3f_1287_);
return v___x_1299_;
}
}
else
{
lean_object* v_a_1367_; lean_object* v___x_1369_; uint8_t v_isShared_1370_; uint8_t v_isSharedCheck_1374_; 
lean_dec(v_expectedType_x3f_1287_);
v_a_1367_ = lean_ctor_get(v___x_1293_, 0);
v_isSharedCheck_1374_ = !lean_is_exclusive(v___x_1293_);
if (v_isSharedCheck_1374_ == 0)
{
v___x_1369_ = v___x_1293_;
v_isShared_1370_ = v_isSharedCheck_1374_;
goto v_resetjp_1368_;
}
else
{
lean_inc(v_a_1367_);
lean_dec(v___x_1293_);
v___x_1369_ = lean_box(0);
v_isShared_1370_ = v_isSharedCheck_1374_;
goto v_resetjp_1368_;
}
v_resetjp_1368_:
{
lean_object* v___x_1372_; 
if (v_isShared_1370_ == 0)
{
v___x_1372_ = v___x_1369_;
goto v_reusejp_1371_;
}
else
{
lean_object* v_reuseFailAlloc_1373_; 
v_reuseFailAlloc_1373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1373_, 0, v_a_1367_);
v___x_1372_ = v_reuseFailAlloc_1373_;
goto v_reusejp_1371_;
}
v_reusejp_1371_:
{
return v___x_1372_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___boxed(lean_object* v_expectedType_x3f_1375_, lean_object* v_a_1376_, lean_object* v_a_1377_, lean_object* v_a_1378_, lean_object* v_a_1379_, lean_object* v_a_1380_){
_start:
{
lean_object* v_res_1381_; 
v_res_1381_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType(v_expectedType_x3f_1375_, v_a_1376_, v_a_1377_, v_a_1378_, v_a_1379_);
lean_dec(v_a_1379_);
lean_dec_ref(v_a_1378_);
lean_dec(v_a_1377_);
lean_dec_ref(v_a_1376_);
return v_res_1381_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__0(void){
_start:
{
lean_object* v___x_1382_; lean_object* v___x_1383_; 
v___x_1382_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0, &lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0);
v___x_1383_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1383_, 0, v___x_1382_);
return v___x_1383_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__3(void){
_start:
{
lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1387_ = lean_box(0);
v___x_1388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__2));
v___x_1389_ = l_Lean_Expr_const___override(v___x_1388_, v___x_1387_);
return v___x_1389_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__5(void){
_start:
{
lean_object* v___x_1391_; lean_object* v___x_1392_; 
v___x_1391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__4));
v___x_1392_ = l_Lean_stringToMessageData(v___x_1391_);
return v___x_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType(lean_object* v_expectedType_x3f_1393_, lean_object* v_a_1394_, lean_object* v_a_1395_, lean_object* v_a_1396_, lean_object* v_a_1397_){
_start:
{
lean_object* v___x_1399_; uint8_t v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; 
v___x_1399_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__0, &lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__0);
v___x_1400_ = 0;
v___x_1401_ = lean_box(0);
v___x_1402_ = l_Lean_Meta_mkFreshExprMVar(v___x_1399_, v___x_1400_, v___x_1401_, v_a_1394_, v_a_1395_, v_a_1396_, v_a_1397_);
if (lean_obj_tag(v___x_1402_) == 0)
{
lean_object* v_a_1403_; lean_object* v___x_1404_; 
v_a_1403_ = lean_ctor_get(v___x_1402_, 0);
lean_inc(v_a_1403_);
lean_dec_ref_known(v___x_1402_, 1);
v___x_1404_ = l_Lean_Meta_mkFreshExprMVar(v___x_1399_, v___x_1400_, v___x_1401_, v_a_1394_, v_a_1395_, v_a_1396_, v_a_1397_);
if (lean_obj_tag(v___x_1404_) == 0)
{
lean_object* v_a_1405_; lean_object* v___x_1407_; uint8_t v_isShared_1408_; uint8_t v_isSharedCheck_1448_; 
v_a_1405_ = lean_ctor_get(v___x_1404_, 0);
v_isSharedCheck_1448_ = !lean_is_exclusive(v___x_1404_);
if (v_isSharedCheck_1448_ == 0)
{
v___x_1407_ = v___x_1404_;
v_isShared_1408_ = v_isSharedCheck_1448_;
goto v_resetjp_1406_;
}
else
{
lean_inc(v_a_1405_);
lean_dec(v___x_1404_);
v___x_1407_ = lean_box(0);
v_isShared_1408_ = v_isSharedCheck_1448_;
goto v_resetjp_1406_;
}
v_resetjp_1406_:
{
lean_object* v___x_1409_; lean_object* v___x_1410_; 
v___x_1409_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__3);
v___x_1410_ = l_Lean_mkAppB(v___x_1409_, v_a_1403_, v_a_1405_);
if (lean_obj_tag(v_expectedType_x3f_1393_) == 1)
{
lean_object* v_val_1411_; lean_object* v___x_1412_; 
lean_del_object(v___x_1407_);
v_val_1411_ = lean_ctor_get(v_expectedType_x3f_1393_, 0);
lean_inc_n(v_val_1411_, 2);
lean_dec_ref_known(v_expectedType_x3f_1393_, 1);
lean_inc_ref(v___x_1410_);
v___x_1412_ = l_Lean_Meta_isExprDefEq(v_val_1411_, v___x_1410_, v_a_1394_, v_a_1395_, v_a_1396_, v_a_1397_);
if (lean_obj_tag(v___x_1412_) == 0)
{
lean_object* v_a_1413_; lean_object* v___x_1415_; uint8_t v_isShared_1416_; uint8_t v_isSharedCheck_1436_; 
v_a_1413_ = lean_ctor_get(v___x_1412_, 0);
v_isSharedCheck_1436_ = !lean_is_exclusive(v___x_1412_);
if (v_isSharedCheck_1436_ == 0)
{
v___x_1415_ = v___x_1412_;
v_isShared_1416_ = v_isSharedCheck_1436_;
goto v_resetjp_1414_;
}
else
{
lean_inc(v_a_1413_);
lean_dec(v___x_1412_);
v___x_1415_ = lean_box(0);
v_isShared_1416_ = v_isSharedCheck_1436_;
goto v_resetjp_1414_;
}
v_resetjp_1414_:
{
uint8_t v___x_1417_; 
v___x_1417_ = lean_unbox(v_a_1413_);
lean_dec(v_a_1413_);
if (v___x_1417_ == 0)
{
lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v_a_1425_; lean_object* v___x_1427_; uint8_t v_isShared_1428_; uint8_t v_isSharedCheck_1432_; 
lean_del_object(v___x_1415_);
lean_dec_ref(v___x_1410_);
v___x_1418_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__3);
v___x_1419_ = l_Lean_MessageData_ofExpr(v_val_1411_);
v___x_1420_ = l_Lean_indentD(v___x_1419_);
v___x_1421_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1421_, 0, v___x_1418_);
lean_ctor_set(v___x_1421_, 1, v___x_1420_);
v___x_1422_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__5, &lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__5);
v___x_1423_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1423_, 0, v___x_1421_);
lean_ctor_set(v___x_1423_, 1, v___x_1422_);
v___x_1424_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_1423_, v_a_1394_, v_a_1395_, v_a_1396_, v_a_1397_);
v_a_1425_ = lean_ctor_get(v___x_1424_, 0);
v_isSharedCheck_1432_ = !lean_is_exclusive(v___x_1424_);
if (v_isSharedCheck_1432_ == 0)
{
v___x_1427_ = v___x_1424_;
v_isShared_1428_ = v_isSharedCheck_1432_;
goto v_resetjp_1426_;
}
else
{
lean_inc(v_a_1425_);
lean_dec(v___x_1424_);
v___x_1427_ = lean_box(0);
v_isShared_1428_ = v_isSharedCheck_1432_;
goto v_resetjp_1426_;
}
v_resetjp_1426_:
{
lean_object* v___x_1430_; 
if (v_isShared_1428_ == 0)
{
v___x_1430_ = v___x_1427_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1431_; 
v_reuseFailAlloc_1431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1431_, 0, v_a_1425_);
v___x_1430_ = v_reuseFailAlloc_1431_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
return v___x_1430_;
}
}
}
else
{
lean_object* v___x_1434_; 
lean_dec(v_val_1411_);
if (v_isShared_1416_ == 0)
{
lean_ctor_set(v___x_1415_, 0, v___x_1410_);
v___x_1434_ = v___x_1415_;
goto v_reusejp_1433_;
}
else
{
lean_object* v_reuseFailAlloc_1435_; 
v_reuseFailAlloc_1435_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1435_, 0, v___x_1410_);
v___x_1434_ = v_reuseFailAlloc_1435_;
goto v_reusejp_1433_;
}
v_reusejp_1433_:
{
return v___x_1434_;
}
}
}
}
else
{
lean_object* v_a_1437_; lean_object* v___x_1439_; uint8_t v_isShared_1440_; uint8_t v_isSharedCheck_1444_; 
lean_dec(v_val_1411_);
lean_dec_ref(v___x_1410_);
v_a_1437_ = lean_ctor_get(v___x_1412_, 0);
v_isSharedCheck_1444_ = !lean_is_exclusive(v___x_1412_);
if (v_isSharedCheck_1444_ == 0)
{
v___x_1439_ = v___x_1412_;
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
else
{
lean_inc(v_a_1437_);
lean_dec(v___x_1412_);
v___x_1439_ = lean_box(0);
v_isShared_1440_ = v_isSharedCheck_1444_;
goto v_resetjp_1438_;
}
v_resetjp_1438_:
{
lean_object* v___x_1442_; 
if (v_isShared_1440_ == 0)
{
v___x_1442_ = v___x_1439_;
goto v_reusejp_1441_;
}
else
{
lean_object* v_reuseFailAlloc_1443_; 
v_reuseFailAlloc_1443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1443_, 0, v_a_1437_);
v___x_1442_ = v_reuseFailAlloc_1443_;
goto v_reusejp_1441_;
}
v_reusejp_1441_:
{
return v___x_1442_;
}
}
}
}
else
{
lean_object* v___x_1446_; 
lean_dec(v_expectedType_x3f_1393_);
if (v_isShared_1408_ == 0)
{
lean_ctor_set(v___x_1407_, 0, v___x_1410_);
v___x_1446_ = v___x_1407_;
goto v_reusejp_1445_;
}
else
{
lean_object* v_reuseFailAlloc_1447_; 
v_reuseFailAlloc_1447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1447_, 0, v___x_1410_);
v___x_1446_ = v_reuseFailAlloc_1447_;
goto v_reusejp_1445_;
}
v_reusejp_1445_:
{
return v___x_1446_;
}
}
}
}
else
{
lean_dec(v_a_1403_);
lean_dec(v_expectedType_x3f_1393_);
return v___x_1404_;
}
}
else
{
lean_dec(v_expectedType_x3f_1393_);
return v___x_1402_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___boxed(lean_object* v_expectedType_x3f_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_){
_start:
{
lean_object* v_res_1455_; 
v_res_1455_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType(v_expectedType_x3f_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_);
lean_dec(v_a_1453_);
lean_dec_ref(v_a_1452_);
lean_dec(v_a_1451_);
lean_dec_ref(v_a_1450_);
return v_res_1455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_ensureIff(lean_object* v_pf_1456_, lean_object* v_a_1457_, lean_object* v_a_1458_, lean_object* v_a_1459_, lean_object* v_a_1460_){
_start:
{
lean_object* v___x_1462_; 
lean_inc(v_a_1460_);
lean_inc_ref(v_a_1459_);
lean_inc(v_a_1458_);
lean_inc_ref(v_a_1457_);
lean_inc_ref(v_pf_1456_);
v___x_1462_ = lean_infer_type(v_pf_1456_, v_a_1457_, v_a_1458_, v_a_1459_, v_a_1460_);
if (lean_obj_tag(v___x_1462_) == 0)
{
lean_object* v_a_1463_; lean_object* v___x_1465_; uint8_t v_isShared_1466_; uint8_t v_isSharedCheck_1479_; 
v_a_1463_ = lean_ctor_get(v___x_1462_, 0);
v_isSharedCheck_1479_ = !lean_is_exclusive(v___x_1462_);
if (v_isSharedCheck_1479_ == 0)
{
v___x_1465_ = v___x_1462_;
v_isShared_1466_ = v_isSharedCheck_1479_;
goto v_resetjp_1464_;
}
else
{
lean_inc(v_a_1463_);
lean_dec(v___x_1462_);
v___x_1465_ = lean_box(0);
v_isShared_1466_ = v_isSharedCheck_1479_;
goto v_resetjp_1464_;
}
v_resetjp_1464_:
{
lean_object* v___x_1468_; 
if (v_isShared_1466_ == 0)
{
lean_ctor_set_tag(v___x_1465_, 1);
v___x_1468_ = v___x_1465_;
goto v_reusejp_1467_;
}
else
{
lean_object* v_reuseFailAlloc_1478_; 
v_reuseFailAlloc_1478_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1478_, 0, v_a_1463_);
v___x_1468_ = v_reuseFailAlloc_1478_;
goto v_reusejp_1467_;
}
v_reusejp_1467_:
{
lean_object* v___x_1469_; 
v___x_1469_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType(v___x_1468_, v_a_1457_, v_a_1458_, v_a_1459_, v_a_1460_);
if (lean_obj_tag(v___x_1469_) == 0)
{
lean_object* v___x_1471_; uint8_t v_isShared_1472_; uint8_t v_isSharedCheck_1476_; 
v_isSharedCheck_1476_ = !lean_is_exclusive(v___x_1469_);
if (v_isSharedCheck_1476_ == 0)
{
lean_object* v_unused_1477_; 
v_unused_1477_ = lean_ctor_get(v___x_1469_, 0);
lean_dec(v_unused_1477_);
v___x_1471_ = v___x_1469_;
v_isShared_1472_ = v_isSharedCheck_1476_;
goto v_resetjp_1470_;
}
else
{
lean_dec(v___x_1469_);
v___x_1471_ = lean_box(0);
v_isShared_1472_ = v_isSharedCheck_1476_;
goto v_resetjp_1470_;
}
v_resetjp_1470_:
{
lean_object* v___x_1474_; 
if (v_isShared_1472_ == 0)
{
lean_ctor_set(v___x_1471_, 0, v_pf_1456_);
v___x_1474_ = v___x_1471_;
goto v_reusejp_1473_;
}
else
{
lean_object* v_reuseFailAlloc_1475_; 
v_reuseFailAlloc_1475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1475_, 0, v_pf_1456_);
v___x_1474_ = v_reuseFailAlloc_1475_;
goto v_reusejp_1473_;
}
v_reusejp_1473_:
{
return v___x_1474_;
}
}
}
else
{
lean_dec_ref(v_pf_1456_);
return v___x_1469_;
}
}
}
}
else
{
lean_dec_ref(v_pf_1456_);
return v___x_1462_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_ensureIff___boxed(lean_object* v_pf_1480_, lean_object* v_a_1481_, lean_object* v_a_1482_, lean_object* v_a_1483_, lean_object* v_a_1484_, lean_object* v_a_1485_){
_start:
{
lean_object* v_res_1486_; 
v_res_1486_ = lp_mathlib_Mathlib_Tactic_TermCongr_ensureIff(v_pf_1480_, v_a_1481_, v_a_1482_, v_a_1483_, v_a_1484_);
lean_dec(v_a_1484_);
lean_dec_ref(v_a_1483_);
lean_dec(v_a_1482_);
lean_dec_ref(v_a_1481_);
return v_res_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorIdx(uint8_t v_x_1487_){
_start:
{
if (v_x_1487_ == 0)
{
lean_object* v___x_1488_; 
v___x_1488_ = lean_unsigned_to_nat(0u);
return v___x_1488_;
}
else
{
lean_object* v___x_1489_; 
v___x_1489_ = lean_unsigned_to_nat(1u);
return v___x_1489_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorIdx___boxed(lean_object* v_x_1490_){
_start:
{
uint8_t v_x_boxed_1491_; lean_object* v_res_1492_; 
v_x_boxed_1491_ = lean_unbox(v_x_1490_);
v_res_1492_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorIdx(v_x_boxed_1491_);
return v_res_1492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim___redArg(lean_object* v_k_1493_){
_start:
{
lean_inc(v_k_1493_);
return v_k_1493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim___redArg___boxed(lean_object* v_k_1494_){
_start:
{
lean_object* v_res_1495_; 
v_res_1495_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim___redArg(v_k_1494_);
lean_dec(v_k_1494_);
return v_res_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim(lean_object* v_motive_1496_, lean_object* v_ctorIdx_1497_, uint8_t v_t_1498_, lean_object* v_h_1499_, lean_object* v_k_1500_){
_start:
{
lean_inc(v_k_1500_);
return v_k_1500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim___boxed(lean_object* v_motive_1501_, lean_object* v_ctorIdx_1502_, lean_object* v_t_1503_, lean_object* v_h_1504_, lean_object* v_k_1505_){
_start:
{
uint8_t v_t_boxed_1506_; lean_object* v_res_1507_; 
v_t_boxed_1506_ = lean_unbox(v_t_1503_);
v_res_1507_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_ctorElim(v_motive_1501_, v_ctorIdx_1502_, v_t_boxed_1506_, v_h_1504_, v_k_1505_);
lean_dec(v_k_1505_);
lean_dec(v_ctorIdx_1502_);
return v_res_1507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim___redArg(lean_object* v_eq_1508_){
_start:
{
lean_inc(v_eq_1508_);
return v_eq_1508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim___redArg___boxed(lean_object* v_eq_1509_){
_start:
{
lean_object* v_res_1510_; 
v_res_1510_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim___redArg(v_eq_1509_);
lean_dec(v_eq_1509_);
return v_res_1510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim(lean_object* v_motive_1511_, uint8_t v_t_1512_, lean_object* v_h_1513_, lean_object* v_eq_1514_){
_start:
{
lean_inc(v_eq_1514_);
return v_eq_1514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim___boxed(lean_object* v_motive_1515_, lean_object* v_t_1516_, lean_object* v_h_1517_, lean_object* v_eq_1518_){
_start:
{
uint8_t v_t_boxed_1519_; lean_object* v_res_1520_; 
v_t_boxed_1519_ = lean_unbox(v_t_1516_);
v_res_1520_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_eq_elim(v_motive_1515_, v_t_boxed_1519_, v_h_1517_, v_eq_1518_);
lean_dec(v_eq_1518_);
return v_res_1520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim___redArg(lean_object* v_heq_1521_){
_start:
{
lean_inc(v_heq_1521_);
return v_heq_1521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim___redArg___boxed(lean_object* v_heq_1522_){
_start:
{
lean_object* v_res_1523_; 
v_res_1523_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim___redArg(v_heq_1522_);
lean_dec(v_heq_1522_);
return v_res_1523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim(lean_object* v_motive_1524_, uint8_t v_t_1525_, lean_object* v_h_1526_, lean_object* v_heq_1527_){
_start:
{
lean_inc(v_heq_1527_);
return v_heq_1527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim___boxed(lean_object* v_motive_1528_, lean_object* v_t_1529_, lean_object* v_h_1530_, lean_object* v_heq_1531_){
_start:
{
uint8_t v_t_boxed_1532_; lean_object* v_res_1533_; 
v_t_boxed_1532_ = lean_unbox(v_t_1529_);
v_res_1533_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrType_heq_elim(v_motive_1528_, v_t_boxed_1532_, v_h_1530_, v_heq_1531_);
lean_dec(v_heq_1531_);
return v_res_1533_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(lean_object* v_res_1534_){
_start:
{
lean_object* v_pf_x3f_1535_; 
v_pf_x3f_1535_ = lean_ctor_get(v_res_1534_, 2);
if (lean_obj_tag(v_pf_x3f_1535_) == 0)
{
uint8_t v___x_1536_; 
v___x_1536_ = 1;
return v___x_1536_;
}
else
{
uint8_t v___x_1537_; 
v___x_1537_ = 0;
return v___x_1537_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl___boxed(lean_object* v_res_1538_){
_start:
{
uint8_t v_res_1539_; lean_object* v_r_1540_; 
v_res_1539_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_res_1538_);
lean_dec_ref(v_res_1538_);
v_r_1540_ = lean_box(v_res_1539_);
return v_r_1540_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1(void){
_start:
{
lean_object* v___x_1542_; lean_object* v___x_1543_; 
v___x_1542_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__0));
v___x_1543_ = l_Lean_stringToMessageData(v___x_1542_);
return v___x_1543_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3(void){
_start:
{
lean_object* v___x_1545_; lean_object* v___x_1546_; 
v___x_1545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__2));
v___x_1546_ = l_Lean_stringToMessageData(v___x_1545_);
return v___x_1546_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__5(void){
_start:
{
lean_object* v___x_1548_; lean_object* v___x_1549_; 
v___x_1548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__4));
v___x_1549_ = l_Lean_stringToMessageData(v___x_1548_);
return v___x_1549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(lean_object* v_res_1550_, lean_object* v_a_1551_, lean_object* v_a_1552_, lean_object* v_a_1553_, lean_object* v_a_1554_){
_start:
{
lean_object* v_lhs_1556_; lean_object* v_rhs_1557_; lean_object* v_pf_x3f_1558_; lean_object* v___y_1560_; lean_object* v___y_1561_; lean_object* v___y_1562_; lean_object* v___y_1563_; lean_object* v___x_1569_; 
v_lhs_1556_ = lean_ctor_get(v_res_1550_, 0);
lean_inc_ref_n(v_lhs_1556_, 2);
v_rhs_1557_ = lean_ctor_get(v_res_1550_, 1);
lean_inc_ref(v_rhs_1557_);
v_pf_x3f_1558_ = lean_ctor_get(v_res_1550_, 2);
lean_inc(v_pf_x3f_1558_);
lean_dec_ref(v_res_1550_);
lean_inc(v_a_1554_);
lean_inc_ref(v_a_1553_);
lean_inc(v_a_1552_);
lean_inc_ref(v_a_1551_);
v___x_1569_ = lean_infer_type(v_lhs_1556_, v_a_1551_, v_a_1552_, v_a_1553_, v_a_1554_);
if (lean_obj_tag(v___x_1569_) == 0)
{
lean_object* v_a_1570_; lean_object* v___x_1571_; 
v_a_1570_ = lean_ctor_get(v___x_1569_, 0);
lean_inc(v_a_1570_);
lean_dec_ref_known(v___x_1569_, 1);
lean_inc(v_a_1554_);
lean_inc_ref(v_a_1553_);
lean_inc(v_a_1552_);
lean_inc_ref(v_a_1551_);
lean_inc_ref(v_rhs_1557_);
v___x_1571_ = lean_infer_type(v_rhs_1557_, v_a_1551_, v_a_1552_, v_a_1553_, v_a_1554_);
if (lean_obj_tag(v___x_1571_) == 0)
{
lean_object* v_a_1572_; lean_object* v___x_1573_; 
v_a_1572_ = lean_ctor_get(v___x_1571_, 0);
lean_inc(v_a_1572_);
lean_dec_ref_known(v___x_1571_, 1);
v___x_1573_ = l_Lean_Meta_isExprDefEq(v_a_1570_, v_a_1572_, v_a_1551_, v_a_1552_, v_a_1553_, v_a_1554_);
if (lean_obj_tag(v___x_1573_) == 0)
{
lean_object* v_a_1574_; uint8_t v___x_1575_; 
v_a_1574_ = lean_ctor_get(v___x_1573_, 0);
lean_inc(v_a_1574_);
lean_dec_ref_known(v___x_1573_, 1);
v___x_1575_ = lean_unbox(v_a_1574_);
lean_dec(v_a_1574_);
if (v___x_1575_ == 0)
{
lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v_a_1588_; lean_object* v___x_1590_; uint8_t v_isShared_1591_; uint8_t v_isSharedCheck_1595_; 
lean_dec(v_pf_x3f_1558_);
v___x_1576_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1);
v___x_1577_ = l_Lean_MessageData_ofExpr(v_lhs_1556_);
v___x_1578_ = l_Lean_indentD(v___x_1577_);
v___x_1579_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1579_, 0, v___x_1576_);
lean_ctor_set(v___x_1579_, 1, v___x_1578_);
v___x_1580_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3);
v___x_1581_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1581_, 0, v___x_1579_);
lean_ctor_set(v___x_1581_, 1, v___x_1580_);
v___x_1582_ = l_Lean_MessageData_ofExpr(v_rhs_1557_);
v___x_1583_ = l_Lean_indentD(v___x_1582_);
v___x_1584_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1584_, 0, v___x_1581_);
lean_ctor_set(v___x_1584_, 1, v___x_1583_);
v___x_1585_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__5, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__5);
v___x_1586_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1586_, 0, v___x_1584_);
lean_ctor_set(v___x_1586_, 1, v___x_1585_);
v___x_1587_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_1586_, v_a_1551_, v_a_1552_, v_a_1553_, v_a_1554_);
v_a_1588_ = lean_ctor_get(v___x_1587_, 0);
v_isSharedCheck_1595_ = !lean_is_exclusive(v___x_1587_);
if (v_isSharedCheck_1595_ == 0)
{
v___x_1590_ = v___x_1587_;
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
else
{
lean_inc(v_a_1588_);
lean_dec(v___x_1587_);
v___x_1590_ = lean_box(0);
v_isShared_1591_ = v_isSharedCheck_1595_;
goto v_resetjp_1589_;
}
v_resetjp_1589_:
{
lean_object* v___x_1593_; 
if (v_isShared_1591_ == 0)
{
v___x_1593_ = v___x_1590_;
goto v_reusejp_1592_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v_a_1588_);
v___x_1593_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1592_;
}
v_reusejp_1592_:
{
return v___x_1593_;
}
}
}
else
{
lean_dec_ref(v_rhs_1557_);
v___y_1560_ = v_a_1551_;
v___y_1561_ = v_a_1552_;
v___y_1562_ = v_a_1553_;
v___y_1563_ = v_a_1554_;
goto v___jp_1559_;
}
}
else
{
lean_object* v_a_1596_; lean_object* v___x_1598_; uint8_t v_isShared_1599_; uint8_t v_isSharedCheck_1603_; 
lean_dec(v_pf_x3f_1558_);
lean_dec_ref(v_rhs_1557_);
lean_dec_ref(v_lhs_1556_);
v_a_1596_ = lean_ctor_get(v___x_1573_, 0);
v_isSharedCheck_1603_ = !lean_is_exclusive(v___x_1573_);
if (v_isSharedCheck_1603_ == 0)
{
v___x_1598_ = v___x_1573_;
v_isShared_1599_ = v_isSharedCheck_1603_;
goto v_resetjp_1597_;
}
else
{
lean_inc(v_a_1596_);
lean_dec(v___x_1573_);
v___x_1598_ = lean_box(0);
v_isShared_1599_ = v_isSharedCheck_1603_;
goto v_resetjp_1597_;
}
v_resetjp_1597_:
{
lean_object* v___x_1601_; 
if (v_isShared_1599_ == 0)
{
v___x_1601_ = v___x_1598_;
goto v_reusejp_1600_;
}
else
{
lean_object* v_reuseFailAlloc_1602_; 
v_reuseFailAlloc_1602_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1602_, 0, v_a_1596_);
v___x_1601_ = v_reuseFailAlloc_1602_;
goto v_reusejp_1600_;
}
v_reusejp_1600_:
{
return v___x_1601_;
}
}
}
}
else
{
lean_dec(v_a_1570_);
lean_dec(v_pf_x3f_1558_);
lean_dec_ref(v_rhs_1557_);
lean_dec_ref(v_lhs_1556_);
return v___x_1571_;
}
}
else
{
lean_dec(v_pf_x3f_1558_);
lean_dec_ref(v_rhs_1557_);
lean_dec_ref(v_lhs_1556_);
return v___x_1569_;
}
v___jp_1559_:
{
if (lean_obj_tag(v_pf_x3f_1558_) == 0)
{
lean_object* v___x_1564_; 
v___x_1564_ = l_Lean_Meta_mkEqRefl(v_lhs_1556_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_);
return v___x_1564_;
}
else
{
lean_object* v_val_1565_; uint8_t v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; 
lean_dec_ref(v_lhs_1556_);
v_val_1565_ = lean_ctor_get(v_pf_x3f_1558_, 0);
lean_inc(v_val_1565_);
lean_dec_ref_known(v_pf_x3f_1558_, 1);
v___x_1566_ = 0;
v___x_1567_ = lean_box(v___x_1566_);
lean_inc(v___y_1563_);
lean_inc_ref(v___y_1562_);
lean_inc(v___y_1561_);
lean_inc_ref(v___y_1560_);
v___x_1568_ = lean_apply_6(v_val_1565_, v___x_1567_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_, lean_box(0));
return v___x_1568_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___boxed(lean_object* v_res_1604_, lean_object* v_a_1605_, lean_object* v_a_1606_, lean_object* v_a_1607_, lean_object* v_a_1608_, lean_object* v_a_1609_){
_start:
{
lean_object* v_res_1610_; 
v_res_1610_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_res_1604_, v_a_1605_, v_a_1606_, v_a_1607_, v_a_1608_);
lean_dec(v_a_1608_);
lean_dec_ref(v_a_1607_);
lean_dec(v_a_1606_);
lean_dec_ref(v_a_1605_);
return v_res_1610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq(lean_object* v_res_1611_, lean_object* v_a_1612_, lean_object* v_a_1613_, lean_object* v_a_1614_, lean_object* v_a_1615_){
_start:
{
lean_object* v_pf_x3f_1617_; 
v_pf_x3f_1617_ = lean_ctor_get(v_res_1611_, 2);
if (lean_obj_tag(v_pf_x3f_1617_) == 0)
{
lean_object* v_lhs_1618_; lean_object* v___x_1619_; 
v_lhs_1618_ = lean_ctor_get(v_res_1611_, 0);
lean_inc_ref(v_lhs_1618_);
lean_dec_ref(v_res_1611_);
v___x_1619_ = l_Lean_Meta_mkHEqRefl(v_lhs_1618_, v_a_1612_, v_a_1613_, v_a_1614_, v_a_1615_);
return v___x_1619_;
}
else
{
lean_object* v_val_1620_; uint8_t v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; 
lean_inc_ref(v_pf_x3f_1617_);
lean_dec_ref(v_res_1611_);
v_val_1620_ = lean_ctor_get(v_pf_x3f_1617_, 0);
lean_inc(v_val_1620_);
lean_dec_ref_known(v_pf_x3f_1617_, 1);
v___x_1621_ = 1;
v___x_1622_ = lean_box(v___x_1621_);
lean_inc(v_a_1615_);
lean_inc_ref(v_a_1614_);
lean_inc(v_a_1613_);
lean_inc_ref(v_a_1612_);
v___x_1623_ = lean_apply_6(v_val_1620_, v___x_1622_, v_a_1612_, v_a_1613_, v_a_1614_, v_a_1615_, lean_box(0));
return v___x_1623_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq___boxed(lean_object* v_res_1624_, lean_object* v_a_1625_, lean_object* v_a_1626_, lean_object* v_a_1627_, lean_object* v_a_1628_, lean_object* v_a_1629_){
_start:
{
lean_object* v_res_1630_; 
v_res_1630_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq(v_res_1624_, v_a_1625_, v_a_1626_, v_a_1627_, v_a_1628_);
lean_dec(v_a_1628_);
lean_dec_ref(v_a_1627_);
lean_dec(v_a_1626_);
lean_dec_ref(v_a_1625_);
return v_res_1630_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__2(void){
_start:
{
lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; 
v___x_1634_ = lean_box(0);
v___x_1635_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__1));
v___x_1636_ = l_Lean_Expr_const___override(v___x_1635_, v___x_1634_);
return v___x_1636_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__4(void){
_start:
{
lean_object* v___x_1638_; lean_object* v___x_1639_; 
v___x_1638_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__3));
v___x_1639_ = l_Lean_stringToMessageData(v___x_1638_);
return v___x_1639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff(lean_object* v_res_1640_, lean_object* v_a_1641_, lean_object* v_a_1642_, lean_object* v_a_1643_, lean_object* v_a_1644_){
_start:
{
lean_object* v_lhs_1646_; lean_object* v_rhs_1647_; lean_object* v___y_1649_; lean_object* v___y_1650_; lean_object* v___y_1651_; lean_object* v___y_1652_; lean_object* v___x_1664_; 
v_lhs_1646_ = lean_ctor_get(v_res_1640_, 0);
lean_inc_ref_n(v_lhs_1646_, 2);
v_rhs_1647_ = lean_ctor_get(v_res_1640_, 1);
lean_inc_ref(v_rhs_1647_);
v___x_1664_ = l_Lean_Meta_isProp(v_lhs_1646_, v_a_1641_, v_a_1642_, v_a_1643_, v_a_1644_);
if (lean_obj_tag(v___x_1664_) == 0)
{
lean_object* v_a_1665_; uint8_t v___x_1666_; 
v_a_1665_ = lean_ctor_get(v___x_1664_, 0);
lean_inc(v_a_1665_);
lean_dec_ref_known(v___x_1664_, 1);
v___x_1666_ = lean_unbox(v_a_1665_);
lean_dec(v_a_1665_);
if (v___x_1666_ == 0)
{
lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v_a_1674_; lean_object* v___x_1676_; uint8_t v_isShared_1677_; uint8_t v_isSharedCheck_1681_; 
lean_dec_ref(v_rhs_1647_);
lean_dec_ref(v_res_1640_);
v___x_1667_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__1);
v___x_1668_ = l_Lean_MessageData_ofExpr(v_lhs_1646_);
v___x_1669_ = l_Lean_indentD(v___x_1668_);
v___x_1670_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1670_, 0, v___x_1667_);
lean_ctor_set(v___x_1670_, 1, v___x_1669_);
v___x_1671_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__4, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__4);
v___x_1672_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1672_, 0, v___x_1670_);
lean_ctor_set(v___x_1672_, 1, v___x_1671_);
v___x_1673_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_1672_, v_a_1641_, v_a_1642_, v_a_1643_, v_a_1644_);
v_a_1674_ = lean_ctor_get(v___x_1673_, 0);
v_isSharedCheck_1681_ = !lean_is_exclusive(v___x_1673_);
if (v_isSharedCheck_1681_ == 0)
{
v___x_1676_ = v___x_1673_;
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
else
{
lean_inc(v_a_1674_);
lean_dec(v___x_1673_);
v___x_1676_ = lean_box(0);
v_isShared_1677_ = v_isSharedCheck_1681_;
goto v_resetjp_1675_;
}
v_resetjp_1675_:
{
lean_object* v___x_1679_; 
if (v_isShared_1677_ == 0)
{
v___x_1679_ = v___x_1676_;
goto v_reusejp_1678_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v_a_1674_);
v___x_1679_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1678_;
}
v_reusejp_1678_:
{
return v___x_1679_;
}
}
}
else
{
v___y_1649_ = v_a_1641_;
v___y_1650_ = v_a_1642_;
v___y_1651_ = v_a_1643_;
v___y_1652_ = v_a_1644_;
goto v___jp_1648_;
}
}
else
{
lean_object* v_a_1682_; lean_object* v___x_1684_; uint8_t v_isShared_1685_; uint8_t v_isSharedCheck_1689_; 
lean_dec_ref(v_rhs_1647_);
lean_dec_ref(v_lhs_1646_);
lean_dec_ref(v_res_1640_);
v_a_1682_ = lean_ctor_get(v___x_1664_, 0);
v_isSharedCheck_1689_ = !lean_is_exclusive(v___x_1664_);
if (v_isSharedCheck_1689_ == 0)
{
v___x_1684_ = v___x_1664_;
v_isShared_1685_ = v_isSharedCheck_1689_;
goto v_resetjp_1683_;
}
else
{
lean_inc(v_a_1682_);
lean_dec(v___x_1664_);
v___x_1684_ = lean_box(0);
v_isShared_1685_ = v_isSharedCheck_1689_;
goto v_resetjp_1683_;
}
v_resetjp_1683_:
{
lean_object* v___x_1687_; 
if (v_isShared_1685_ == 0)
{
v___x_1687_ = v___x_1684_;
goto v_reusejp_1686_;
}
else
{
lean_object* v_reuseFailAlloc_1688_; 
v_reuseFailAlloc_1688_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1688_, 0, v_a_1682_);
v___x_1687_ = v_reuseFailAlloc_1688_;
goto v_reusejp_1686_;
}
v_reusejp_1686_:
{
return v___x_1687_;
}
}
}
v___jp_1648_:
{
lean_object* v___x_1653_; 
v___x_1653_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_res_1640_, v___y_1649_, v___y_1650_, v___y_1651_, v___y_1652_);
if (lean_obj_tag(v___x_1653_) == 0)
{
lean_object* v_a_1654_; lean_object* v___x_1656_; uint8_t v_isShared_1657_; uint8_t v_isSharedCheck_1663_; 
v_a_1654_ = lean_ctor_get(v___x_1653_, 0);
v_isSharedCheck_1663_ = !lean_is_exclusive(v___x_1653_);
if (v_isSharedCheck_1663_ == 0)
{
v___x_1656_ = v___x_1653_;
v_isShared_1657_ = v_isSharedCheck_1663_;
goto v_resetjp_1655_;
}
else
{
lean_inc(v_a_1654_);
lean_dec(v___x_1653_);
v___x_1656_ = lean_box(0);
v_isShared_1657_ = v_isSharedCheck_1663_;
goto v_resetjp_1655_;
}
v_resetjp_1655_:
{
lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1661_; 
v___x_1658_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__2, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___closed__2);
v___x_1659_ = l_Lean_mkApp3(v___x_1658_, v_lhs_1646_, v_rhs_1647_, v_a_1654_);
if (v_isShared_1657_ == 0)
{
lean_ctor_set(v___x_1656_, 0, v___x_1659_);
v___x_1661_ = v___x_1656_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1662_; 
v_reuseFailAlloc_1662_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1662_, 0, v___x_1659_);
v___x_1661_ = v_reuseFailAlloc_1662_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
return v___x_1661_;
}
}
}
else
{
lean_dec_ref(v_rhs_1647_);
lean_dec_ref(v_lhs_1646_);
return v___x_1653_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff___boxed(lean_object* v_res_1690_, lean_object* v_a_1691_, lean_object* v_a_1692_, lean_object* v_a_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_){
_start:
{
lean_object* v_res_1696_; 
v_res_1696_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff(v_res_1690_, v_a_1691_, v_a_1692_, v_a_1693_, v_a_1694_);
lean_dec(v_a_1694_);
lean_dec_ref(v_a_1693_);
lean_dec(v_a_1692_);
lean_dec_ref(v_a_1691_);
return v_res_1696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans___lam__0(lean_object* v_res1_1697_, lean_object* v_res2_1698_, uint8_t v_x_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_){
_start:
{
if (v_x_1699_ == 0)
{
lean_object* v___x_1705_; 
v___x_1705_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_res1_1697_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
if (lean_obj_tag(v___x_1705_) == 0)
{
lean_object* v_a_1706_; lean_object* v___x_1707_; 
v_a_1706_ = lean_ctor_get(v___x_1705_, 0);
lean_inc(v_a_1706_);
lean_dec_ref_known(v___x_1705_, 1);
v___x_1707_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_res2_1698_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
if (lean_obj_tag(v___x_1707_) == 0)
{
lean_object* v_a_1708_; lean_object* v___x_1709_; 
v_a_1708_ = lean_ctor_get(v___x_1707_, 0);
lean_inc(v_a_1708_);
lean_dec_ref_known(v___x_1707_, 1);
v___x_1709_ = l_Lean_Meta_mkEqTrans(v_a_1706_, v_a_1708_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
return v___x_1709_;
}
else
{
lean_dec(v_a_1706_);
return v___x_1707_;
}
}
else
{
lean_dec_ref(v_res2_1698_);
return v___x_1705_;
}
}
else
{
lean_object* v___x_1710_; 
v___x_1710_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq(v_res1_1697_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
if (lean_obj_tag(v___x_1710_) == 0)
{
lean_object* v_a_1711_; lean_object* v___x_1712_; 
v_a_1711_ = lean_ctor_get(v___x_1710_, 0);
lean_inc(v_a_1711_);
lean_dec_ref_known(v___x_1710_, 1);
v___x_1712_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq(v_res2_1698_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
if (lean_obj_tag(v___x_1712_) == 0)
{
lean_object* v_a_1713_; lean_object* v___x_1714_; 
v_a_1713_ = lean_ctor_get(v___x_1712_, 0);
lean_inc(v_a_1713_);
lean_dec_ref_known(v___x_1712_, 1);
v___x_1714_ = l_Lean_Meta_mkHEqTrans(v_a_1711_, v_a_1713_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
return v___x_1714_;
}
else
{
lean_dec(v_a_1711_);
return v___x_1712_;
}
}
else
{
lean_dec_ref(v_res2_1698_);
return v___x_1710_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans___lam__0___boxed(lean_object* v_res1_1715_, lean_object* v_res2_1716_, lean_object* v_x_1717_, lean_object* v___y_1718_, lean_object* v___y_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_){
_start:
{
uint8_t v_x_585__boxed_1723_; lean_object* v_res_1724_; 
v_x_585__boxed_1723_ = lean_unbox(v_x_1717_);
v_res_1724_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans___lam__0(v_res1_1715_, v_res2_1716_, v_x_585__boxed_1723_, v___y_1718_, v___y_1719_, v___y_1720_, v___y_1721_);
lean_dec(v___y_1721_);
lean_dec_ref(v___y_1720_);
lean_dec(v___y_1719_);
lean_dec_ref(v___y_1718_);
return v_res_1724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans(lean_object* v_res1_1725_, lean_object* v_res2_1726_){
_start:
{
lean_object* v_lhs_1727_; lean_object* v_pf_x3f_1728_; lean_object* v_rhs_1729_; lean_object* v_pf_x3f_1730_; uint8_t v___x_1731_; 
v_lhs_1727_ = lean_ctor_get(v_res1_1725_, 0);
lean_inc_ref(v_lhs_1727_);
v_pf_x3f_1728_ = lean_ctor_get(v_res1_1725_, 2);
v_rhs_1729_ = lean_ctor_get(v_res2_1726_, 1);
lean_inc_ref(v_rhs_1729_);
v_pf_x3f_1730_ = lean_ctor_get(v_res2_1726_, 2);
v___x_1731_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_res1_1725_);
if (v___x_1731_ == 0)
{
uint8_t v___x_1732_; 
v___x_1732_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_res2_1726_);
if (v___x_1732_ == 0)
{
lean_object* v___f_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; 
v___f_1733_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1733_, 0, v_res1_1725_);
lean_closure_set(v___f_1733_, 1, v_res2_1726_);
v___x_1734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1734_, 0, v___f_1733_);
v___x_1735_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1735_, 0, v_lhs_1727_);
lean_ctor_set(v___x_1735_, 1, v_rhs_1729_);
lean_ctor_set(v___x_1735_, 2, v___x_1734_);
return v___x_1735_;
}
else
{
lean_object* v___x_1737_; uint8_t v_isShared_1738_; uint8_t v_isSharedCheck_1742_; 
lean_inc(v_pf_x3f_1728_);
lean_dec_ref(v_res1_1725_);
v_isSharedCheck_1742_ = !lean_is_exclusive(v_res2_1726_);
if (v_isSharedCheck_1742_ == 0)
{
lean_object* v_unused_1743_; lean_object* v_unused_1744_; lean_object* v_unused_1745_; 
v_unused_1743_ = lean_ctor_get(v_res2_1726_, 2);
lean_dec(v_unused_1743_);
v_unused_1744_ = lean_ctor_get(v_res2_1726_, 1);
lean_dec(v_unused_1744_);
v_unused_1745_ = lean_ctor_get(v_res2_1726_, 0);
lean_dec(v_unused_1745_);
v___x_1737_ = v_res2_1726_;
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
else
{
lean_dec(v_res2_1726_);
v___x_1737_ = lean_box(0);
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
v_resetjp_1736_:
{
lean_object* v___x_1740_; 
if (v_isShared_1738_ == 0)
{
lean_ctor_set(v___x_1737_, 2, v_pf_x3f_1728_);
lean_ctor_set(v___x_1737_, 0, v_lhs_1727_);
v___x_1740_ = v___x_1737_;
goto v_reusejp_1739_;
}
else
{
lean_object* v_reuseFailAlloc_1741_; 
v_reuseFailAlloc_1741_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1741_, 0, v_lhs_1727_);
lean_ctor_set(v_reuseFailAlloc_1741_, 1, v_rhs_1729_);
lean_ctor_set(v_reuseFailAlloc_1741_, 2, v_pf_x3f_1728_);
v___x_1740_ = v_reuseFailAlloc_1741_;
goto v_reusejp_1739_;
}
v_reusejp_1739_:
{
return v___x_1740_;
}
}
}
}
else
{
lean_object* v___x_1747_; uint8_t v_isShared_1748_; uint8_t v_isSharedCheck_1752_; 
lean_inc(v_pf_x3f_1730_);
lean_dec_ref(v_res1_1725_);
v_isSharedCheck_1752_ = !lean_is_exclusive(v_res2_1726_);
if (v_isSharedCheck_1752_ == 0)
{
lean_object* v_unused_1753_; lean_object* v_unused_1754_; lean_object* v_unused_1755_; 
v_unused_1753_ = lean_ctor_get(v_res2_1726_, 2);
lean_dec(v_unused_1753_);
v_unused_1754_ = lean_ctor_get(v_res2_1726_, 1);
lean_dec(v_unused_1754_);
v_unused_1755_ = lean_ctor_get(v_res2_1726_, 0);
lean_dec(v_unused_1755_);
v___x_1747_ = v_res2_1726_;
v_isShared_1748_ = v_isSharedCheck_1752_;
goto v_resetjp_1746_;
}
else
{
lean_dec(v_res2_1726_);
v___x_1747_ = lean_box(0);
v_isShared_1748_ = v_isSharedCheck_1752_;
goto v_resetjp_1746_;
}
v_resetjp_1746_:
{
lean_object* v___x_1750_; 
if (v_isShared_1748_ == 0)
{
lean_ctor_set(v___x_1747_, 0, v_lhs_1727_);
v___x_1750_ = v___x_1747_;
goto v_reusejp_1749_;
}
else
{
lean_object* v_reuseFailAlloc_1751_; 
v_reuseFailAlloc_1751_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1751_, 0, v_lhs_1727_);
lean_ctor_set(v_reuseFailAlloc_1751_, 1, v_rhs_1729_);
lean_ctor_set(v_reuseFailAlloc_1751_, 2, v_pf_x3f_1730_);
v___x_1750_ = v_reuseFailAlloc_1751_;
goto v_reusejp_1749_;
}
v_reusejp_1749_:
{
return v___x_1750_;
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__3(void){
_start:
{
lean_object* v___x_1760_; lean_object* v___x_1761_; 
v___x_1760_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__2));
v___x_1761_ = l_Lean_stringToMessageData(v___x_1760_);
return v___x_1761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf(lean_object* v_lhs_1762_, lean_object* v_pf_1763_, lean_object* v_a_1764_, lean_object* v_a_1765_, lean_object* v_a_1766_, lean_object* v_a_1767_){
_start:
{
lean_object* v___y_1770_; lean_object* v___y_1771_; lean_object* v___y_1772_; lean_object* v___y_1773_; lean_object* v___x_1779_; 
lean_inc(v_a_1767_);
lean_inc_ref(v_a_1766_);
lean_inc(v_a_1765_);
lean_inc_ref(v_a_1764_);
lean_inc_ref(v_pf_1763_);
v___x_1779_ = lean_infer_type(v_pf_1763_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
if (lean_obj_tag(v___x_1779_) == 0)
{
lean_object* v_a_1780_; lean_object* v___x_1781_; 
v_a_1780_ = lean_ctor_get(v___x_1779_, 0);
lean_inc(v_a_1780_);
lean_dec_ref_known(v___x_1779_, 1);
lean_inc(v_a_1767_);
lean_inc_ref(v_a_1766_);
lean_inc(v_a_1765_);
lean_inc_ref(v_a_1764_);
v___x_1781_ = lean_whnf(v_a_1780_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
if (lean_obj_tag(v___x_1781_) == 0)
{
lean_object* v_a_1782_; lean_object* v___x_1784_; uint8_t v_isShared_1785_; uint8_t v_isSharedCheck_1860_; 
v_a_1782_ = lean_ctor_get(v___x_1781_, 0);
v_isSharedCheck_1860_ = !lean_is_exclusive(v___x_1781_);
if (v_isSharedCheck_1860_ == 0)
{
v___x_1784_ = v___x_1781_;
v_isShared_1785_ = v_isSharedCheck_1860_;
goto v_resetjp_1783_;
}
else
{
lean_inc(v_a_1782_);
lean_dec(v___x_1781_);
v___x_1784_ = lean_box(0);
v_isShared_1785_ = v_isSharedCheck_1860_;
goto v_resetjp_1783_;
}
v_resetjp_1783_:
{
lean_object* v___x_1786_; lean_object* v___x_1787_; uint8_t v___x_1788_; 
v___x_1786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__2));
v___x_1787_ = lean_unsigned_to_nat(2u);
v___x_1788_ = l_Lean_Expr_isAppOfArity(v_a_1782_, v___x_1786_, v___x_1787_);
if (v___x_1788_ == 0)
{
lean_object* v___x_1789_; lean_object* v___x_1790_; uint8_t v___x_1791_; 
v___x_1789_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__1));
v___x_1790_ = lean_unsigned_to_nat(3u);
v___x_1791_ = l_Lean_Expr_isAppOfArity(v_a_1782_, v___x_1789_, v___x_1790_);
if (v___x_1791_ == 0)
{
lean_object* v___x_1792_; lean_object* v___x_1793_; uint8_t v___x_1794_; 
lean_del_object(v___x_1784_);
v___x_1792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__1));
v___x_1793_ = lean_unsigned_to_nat(4u);
v___x_1794_ = l_Lean_Expr_isAppOfArity(v_a_1782_, v___x_1792_, v___x_1793_);
if (v___x_1794_ == 0)
{
lean_object* v___x_1795_; 
lean_dec(v_a_1782_);
v___x_1795_ = l_Lean_Meta_isProp(v_lhs_1762_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
if (lean_obj_tag(v___x_1795_) == 0)
{
lean_object* v_a_1796_; uint8_t v___x_1797_; 
v_a_1796_ = lean_ctor_get(v___x_1795_, 0);
lean_inc(v_a_1796_);
lean_dec_ref_known(v___x_1795_, 1);
v___x_1797_ = lean_unbox(v_a_1796_);
lean_dec(v_a_1796_);
if (v___x_1797_ == 0)
{
lean_object* v___x_1798_; 
lean_inc(v_a_1767_);
lean_inc_ref(v_a_1766_);
lean_inc(v_a_1765_);
lean_inc_ref(v_a_1764_);
lean_inc_ref(v_pf_1763_);
v___x_1798_ = lean_infer_type(v_pf_1763_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
if (lean_obj_tag(v___x_1798_) == 0)
{
lean_object* v_a_1799_; lean_object* v___x_1801_; uint8_t v_isShared_1802_; uint8_t v_isSharedCheck_1815_; 
v_a_1799_ = lean_ctor_get(v___x_1798_, 0);
v_isSharedCheck_1815_ = !lean_is_exclusive(v___x_1798_);
if (v_isSharedCheck_1815_ == 0)
{
v___x_1801_ = v___x_1798_;
v_isShared_1802_ = v_isSharedCheck_1815_;
goto v_resetjp_1800_;
}
else
{
lean_inc(v_a_1799_);
lean_dec(v___x_1798_);
v___x_1801_ = lean_box(0);
v_isShared_1802_ = v_isSharedCheck_1815_;
goto v_resetjp_1800_;
}
v_resetjp_1800_:
{
lean_object* v___x_1804_; 
if (v_isShared_1802_ == 0)
{
lean_ctor_set_tag(v___x_1801_, 1);
v___x_1804_ = v___x_1801_;
goto v_reusejp_1803_;
}
else
{
lean_object* v_reuseFailAlloc_1814_; 
v_reuseFailAlloc_1814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1814_, 0, v_a_1799_);
v___x_1804_ = v_reuseFailAlloc_1814_;
goto v_reusejp_1803_;
}
v_reusejp_1803_:
{
lean_object* v___x_1805_; 
v___x_1805_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType(v___x_1804_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
if (lean_obj_tag(v___x_1805_) == 0)
{
lean_object* v___x_1807_; uint8_t v_isShared_1808_; uint8_t v_isSharedCheck_1812_; 
v_isSharedCheck_1812_ = !lean_is_exclusive(v___x_1805_);
if (v_isSharedCheck_1812_ == 0)
{
lean_object* v_unused_1813_; 
v_unused_1813_ = lean_ctor_get(v___x_1805_, 0);
lean_dec(v_unused_1813_);
v___x_1807_ = v___x_1805_;
v_isShared_1808_ = v_isSharedCheck_1812_;
goto v_resetjp_1806_;
}
else
{
lean_dec(v___x_1805_);
v___x_1807_ = lean_box(0);
v_isShared_1808_ = v_isSharedCheck_1812_;
goto v_resetjp_1806_;
}
v_resetjp_1806_:
{
lean_object* v___x_1810_; 
if (v_isShared_1808_ == 0)
{
lean_ctor_set(v___x_1807_, 0, v_pf_1763_);
v___x_1810_ = v___x_1807_;
goto v_reusejp_1809_;
}
else
{
lean_object* v_reuseFailAlloc_1811_; 
v_reuseFailAlloc_1811_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1811_, 0, v_pf_1763_);
v___x_1810_ = v_reuseFailAlloc_1811_;
goto v_reusejp_1809_;
}
v_reusejp_1809_:
{
return v___x_1810_;
}
}
}
else
{
lean_dec_ref(v_pf_1763_);
return v___x_1805_;
}
}
}
}
else
{
lean_dec_ref(v_pf_1763_);
return v___x_1798_;
}
}
else
{
lean_object* v___x_1816_; 
v___x_1816_ = lp_mathlib_Mathlib_Tactic_TermCongr_ensureIff(v_pf_1763_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
if (lean_obj_tag(v___x_1816_) == 0)
{
lean_object* v_a_1817_; lean_object* v___x_1818_; 
v_a_1817_ = lean_ctor_get(v___x_1816_, 0);
lean_inc(v_a_1817_);
lean_dec_ref_known(v___x_1816_, 1);
v___x_1818_ = l_Lean_Meta_mkPropExt(v_a_1817_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
return v___x_1818_;
}
else
{
return v___x_1816_;
}
}
}
else
{
lean_object* v_a_1819_; lean_object* v___x_1821_; uint8_t v_isShared_1822_; uint8_t v_isSharedCheck_1826_; 
lean_dec_ref(v_pf_1763_);
v_a_1819_ = lean_ctor_get(v___x_1795_, 0);
v_isSharedCheck_1826_ = !lean_is_exclusive(v___x_1795_);
if (v_isSharedCheck_1826_ == 0)
{
v___x_1821_ = v___x_1795_;
v_isShared_1822_ = v_isSharedCheck_1826_;
goto v_resetjp_1820_;
}
else
{
lean_inc(v_a_1819_);
lean_dec(v___x_1795_);
v___x_1821_ = lean_box(0);
v_isShared_1822_ = v_isSharedCheck_1826_;
goto v_resetjp_1820_;
}
v_resetjp_1820_:
{
lean_object* v___x_1824_; 
if (v_isShared_1822_ == 0)
{
v___x_1824_ = v___x_1821_;
goto v_reusejp_1823_;
}
else
{
lean_object* v_reuseFailAlloc_1825_; 
v_reuseFailAlloc_1825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1825_, 0, v_a_1819_);
v___x_1824_ = v_reuseFailAlloc_1825_;
goto v_reusejp_1823_;
}
v_reusejp_1823_:
{
return v___x_1824_;
}
}
}
}
else
{
lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; 
lean_dec_ref(v_lhs_1762_);
v___x_1827_ = l_Lean_Expr_appFn_x21(v_a_1782_);
v___x_1828_ = l_Lean_Expr_appFn_x21(v___x_1827_);
v___x_1829_ = l_Lean_Expr_appFn_x21(v___x_1828_);
lean_dec_ref(v___x_1828_);
v___x_1830_ = l_Lean_Expr_appArg_x21(v___x_1829_);
lean_dec_ref(v___x_1829_);
v___x_1831_ = l_Lean_Expr_appArg_x21(v___x_1827_);
lean_dec_ref(v___x_1827_);
v___x_1832_ = l_Lean_Meta_isExprDefEq(v___x_1830_, v___x_1831_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
if (lean_obj_tag(v___x_1832_) == 0)
{
lean_object* v_a_1833_; uint8_t v___x_1834_; 
v_a_1833_ = lean_ctor_get(v___x_1832_, 0);
lean_inc(v_a_1833_);
lean_dec_ref_known(v___x_1832_, 1);
v___x_1834_ = lean_unbox(v_a_1833_);
lean_dec(v_a_1833_);
if (v___x_1834_ == 0)
{
lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v_a_1840_; lean_object* v___x_1842_; uint8_t v_isShared_1843_; uint8_t v_isSharedCheck_1847_; 
lean_dec_ref(v_pf_1763_);
v___x_1835_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__3, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__3);
v___x_1836_ = l_Lean_MessageData_ofExpr(v_a_1782_);
v___x_1837_ = l_Lean_indentD(v___x_1836_);
v___x_1838_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1838_, 0, v___x_1835_);
lean_ctor_set(v___x_1838_, 1, v___x_1837_);
v___x_1839_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_1838_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
v_a_1840_ = lean_ctor_get(v___x_1839_, 0);
v_isSharedCheck_1847_ = !lean_is_exclusive(v___x_1839_);
if (v_isSharedCheck_1847_ == 0)
{
v___x_1842_ = v___x_1839_;
v_isShared_1843_ = v_isSharedCheck_1847_;
goto v_resetjp_1841_;
}
else
{
lean_inc(v_a_1840_);
lean_dec(v___x_1839_);
v___x_1842_ = lean_box(0);
v_isShared_1843_ = v_isSharedCheck_1847_;
goto v_resetjp_1841_;
}
v_resetjp_1841_:
{
lean_object* v___x_1845_; 
if (v_isShared_1843_ == 0)
{
v___x_1845_ = v___x_1842_;
goto v_reusejp_1844_;
}
else
{
lean_object* v_reuseFailAlloc_1846_; 
v_reuseFailAlloc_1846_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1846_, 0, v_a_1840_);
v___x_1845_ = v_reuseFailAlloc_1846_;
goto v_reusejp_1844_;
}
v_reusejp_1844_:
{
return v___x_1845_;
}
}
}
else
{
lean_dec(v_a_1782_);
v___y_1770_ = v_a_1764_;
v___y_1771_ = v_a_1765_;
v___y_1772_ = v_a_1766_;
v___y_1773_ = v_a_1767_;
goto v___jp_1769_;
}
}
else
{
lean_object* v_a_1848_; lean_object* v___x_1850_; uint8_t v_isShared_1851_; uint8_t v_isSharedCheck_1855_; 
lean_dec(v_a_1782_);
lean_dec_ref(v_pf_1763_);
v_a_1848_ = lean_ctor_get(v___x_1832_, 0);
v_isSharedCheck_1855_ = !lean_is_exclusive(v___x_1832_);
if (v_isSharedCheck_1855_ == 0)
{
v___x_1850_ = v___x_1832_;
v_isShared_1851_ = v_isSharedCheck_1855_;
goto v_resetjp_1849_;
}
else
{
lean_inc(v_a_1848_);
lean_dec(v___x_1832_);
v___x_1850_ = lean_box(0);
v_isShared_1851_ = v_isSharedCheck_1855_;
goto v_resetjp_1849_;
}
v_resetjp_1849_:
{
lean_object* v___x_1853_; 
if (v_isShared_1851_ == 0)
{
v___x_1853_ = v___x_1850_;
goto v_reusejp_1852_;
}
else
{
lean_object* v_reuseFailAlloc_1854_; 
v_reuseFailAlloc_1854_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1854_, 0, v_a_1848_);
v___x_1853_ = v_reuseFailAlloc_1854_;
goto v_reusejp_1852_;
}
v_reusejp_1852_:
{
return v___x_1853_;
}
}
}
}
}
else
{
lean_object* v___x_1857_; 
lean_dec(v_a_1782_);
lean_dec_ref(v_lhs_1762_);
if (v_isShared_1785_ == 0)
{
lean_ctor_set(v___x_1784_, 0, v_pf_1763_);
v___x_1857_ = v___x_1784_;
goto v_reusejp_1856_;
}
else
{
lean_object* v_reuseFailAlloc_1858_; 
v_reuseFailAlloc_1858_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1858_, 0, v_pf_1763_);
v___x_1857_ = v_reuseFailAlloc_1858_;
goto v_reusejp_1856_;
}
v_reusejp_1856_:
{
return v___x_1857_;
}
}
}
else
{
lean_object* v___x_1859_; 
lean_del_object(v___x_1784_);
lean_dec(v_a_1782_);
lean_dec_ref(v_lhs_1762_);
v___x_1859_ = l_Lean_Meta_mkPropExt(v_pf_1763_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_);
return v___x_1859_;
}
}
}
else
{
lean_dec_ref(v_pf_1763_);
lean_dec_ref(v_lhs_1762_);
return v___x_1781_;
}
}
else
{
lean_dec_ref(v_pf_1763_);
lean_dec_ref(v_lhs_1762_);
return v___x_1779_;
}
v___jp_1769_:
{
lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; 
v___x_1774_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___closed__1));
v___x_1775_ = lean_unsigned_to_nat(1u);
v___x_1776_ = lean_mk_empty_array_with_capacity(v___x_1775_);
v___x_1777_ = lean_array_push(v___x_1776_, v_pf_1763_);
v___x_1778_ = l_Lean_Meta_mkAppM(v___x_1774_, v___x_1777_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
return v___x_1778_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf___boxed(lean_object* v_lhs_1861_, lean_object* v_pf_1862_, lean_object* v_a_1863_, lean_object* v_a_1864_, lean_object* v_a_1865_, lean_object* v_a_1866_, lean_object* v_a_1867_){
_start:
{
lean_object* v_res_1868_; 
v_res_1868_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf(v_lhs_1861_, v_pf_1862_, v_a_1863_, v_a_1864_, v_a_1865_, v_a_1866_);
lean_dec(v_a_1866_);
lean_dec_ref(v_a_1865_);
lean_dec(v_a_1864_);
lean_dec_ref(v_a_1863_);
return v_res_1868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___redArg(lean_object* v_k_1869_, uint8_t v_allowLevelAssignments_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_){
_start:
{
lean_object* v___x_1876_; 
v___x_1876_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_1870_, v_k_1869_, v___y_1871_, v___y_1872_, v___y_1873_, v___y_1874_);
if (lean_obj_tag(v___x_1876_) == 0)
{
lean_object* v_a_1877_; lean_object* v___x_1879_; uint8_t v_isShared_1880_; uint8_t v_isSharedCheck_1884_; 
v_a_1877_ = lean_ctor_get(v___x_1876_, 0);
v_isSharedCheck_1884_ = !lean_is_exclusive(v___x_1876_);
if (v_isSharedCheck_1884_ == 0)
{
v___x_1879_ = v___x_1876_;
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
else
{
lean_inc(v_a_1877_);
lean_dec(v___x_1876_);
v___x_1879_ = lean_box(0);
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
v_resetjp_1878_:
{
lean_object* v___x_1882_; 
if (v_isShared_1880_ == 0)
{
v___x_1882_ = v___x_1879_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1883_; 
v_reuseFailAlloc_1883_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1883_, 0, v_a_1877_);
v___x_1882_ = v_reuseFailAlloc_1883_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
return v___x_1882_;
}
}
}
else
{
lean_object* v_a_1885_; lean_object* v___x_1887_; uint8_t v_isShared_1888_; uint8_t v_isSharedCheck_1892_; 
v_a_1885_ = lean_ctor_get(v___x_1876_, 0);
v_isSharedCheck_1892_ = !lean_is_exclusive(v___x_1876_);
if (v_isSharedCheck_1892_ == 0)
{
v___x_1887_ = v___x_1876_;
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
else
{
lean_inc(v_a_1885_);
lean_dec(v___x_1876_);
v___x_1887_ = lean_box(0);
v_isShared_1888_ = v_isSharedCheck_1892_;
goto v_resetjp_1886_;
}
v_resetjp_1886_:
{
lean_object* v___x_1890_; 
if (v_isShared_1888_ == 0)
{
v___x_1890_ = v___x_1887_;
goto v_reusejp_1889_;
}
else
{
lean_object* v_reuseFailAlloc_1891_; 
v_reuseFailAlloc_1891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1891_, 0, v_a_1885_);
v___x_1890_ = v_reuseFailAlloc_1891_;
goto v_reusejp_1889_;
}
v_reusejp_1889_:
{
return v___x_1890_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___redArg___boxed(lean_object* v_k_1893_, lean_object* v_allowLevelAssignments_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1900_; lean_object* v_res_1901_; 
v_allowLevelAssignments_boxed_1900_ = lean_unbox(v_allowLevelAssignments_1894_);
v_res_1901_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___redArg(v_k_1893_, v_allowLevelAssignments_boxed_1900_, v___y_1895_, v___y_1896_, v___y_1897_, v___y_1898_);
lean_dec(v___y_1898_);
lean_dec_ref(v___y_1897_);
lean_dec(v___y_1896_);
lean_dec_ref(v___y_1895_);
return v_res_1901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0(lean_object* v_00_u03b1_1902_, lean_object* v_k_1903_, uint8_t v_allowLevelAssignments_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_){
_start:
{
lean_object* v___x_1910_; 
v___x_1910_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___redArg(v_k_1903_, v_allowLevelAssignments_1904_, v___y_1905_, v___y_1906_, v___y_1907_, v___y_1908_);
return v___x_1910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___boxed(lean_object* v_00_u03b1_1911_, lean_object* v_k_1912_, lean_object* v_allowLevelAssignments_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1919_; lean_object* v_res_1920_; 
v_allowLevelAssignments_boxed_1919_ = lean_unbox(v_allowLevelAssignments_1913_);
v_res_1920_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0(v_00_u03b1_1911_, v_k_1912_, v_allowLevelAssignments_boxed_1919_, v___y_1914_, v___y_1915_, v___y_1916_, v___y_1917_);
lean_dec(v___y_1917_);
lean_dec_ref(v___y_1916_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
return v_res_1920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___lam__0(lean_object* v_a_1921_, lean_object* v_a_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_){
_start:
{
lean_object* v___x_1928_; 
v___x_1928_ = l_Lean_Meta_isExprDefEq(v_a_1921_, v_a_1922_, v___y_1923_, v___y_1924_, v___y_1925_, v___y_1926_);
return v___x_1928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___lam__0___boxed(lean_object* v_a_1929_, lean_object* v_a_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_){
_start:
{
lean_object* v_res_1936_; 
v_res_1936_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___lam__0(v_a_1929_, v_a_1930_, v___y_1931_, v___y_1932_, v___y_1933_, v___y_1934_);
lean_dec(v___y_1934_);
lean_dec_ref(v___y_1933_);
lean_dec(v___y_1932_);
lean_dec_ref(v___y_1931_);
return v_res_1936_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf(lean_object* v_lhs_1940_, lean_object* v_rhs_1941_, lean_object* v_pf_1942_, lean_object* v_a_1943_, lean_object* v_a_1944_, lean_object* v_a_1945_, lean_object* v_a_1946_){
_start:
{
lean_object* v___x_1948_; 
lean_inc(v_a_1946_);
lean_inc_ref(v_a_1945_);
lean_inc(v_a_1944_);
lean_inc_ref(v_a_1943_);
lean_inc_ref(v_pf_1942_);
v___x_1948_ = lean_infer_type(v_pf_1942_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1948_) == 0)
{
lean_object* v_a_1949_; lean_object* v___x_1950_; 
v_a_1949_ = lean_ctor_get(v___x_1948_, 0);
lean_inc(v_a_1949_);
lean_dec_ref_known(v___x_1948_, 1);
lean_inc(v_a_1946_);
lean_inc_ref(v_a_1945_);
lean_inc(v_a_1944_);
lean_inc_ref(v_a_1943_);
v___x_1950_ = lean_whnf(v_a_1949_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1950_) == 0)
{
lean_object* v_a_1951_; lean_object* v___x_1953_; uint8_t v_isShared_1954_; uint8_t v_isSharedCheck_2020_; 
v_a_1951_ = lean_ctor_get(v___x_1950_, 0);
v_isSharedCheck_2020_ = !lean_is_exclusive(v___x_1950_);
if (v_isSharedCheck_2020_ == 0)
{
v___x_1953_ = v___x_1950_;
v_isShared_1954_ = v_isSharedCheck_2020_;
goto v_resetjp_1952_;
}
else
{
lean_inc(v_a_1951_);
lean_dec(v___x_1950_);
v___x_1953_ = lean_box(0);
v_isShared_1954_ = v_isSharedCheck_2020_;
goto v_resetjp_1952_;
}
v_resetjp_1952_:
{
lean_object* v___x_1955_; lean_object* v___x_1956_; uint8_t v___x_1957_; 
v___x_1955_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__2));
v___x_1956_ = lean_unsigned_to_nat(2u);
v___x_1957_ = l_Lean_Expr_isAppOfArity(v_a_1951_, v___x_1955_, v___x_1956_);
if (v___x_1957_ == 0)
{
lean_object* v___x_1958_; lean_object* v___x_1959_; uint8_t v___x_1960_; 
v___x_1958_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkEqForExpectedType___closed__1));
v___x_1959_ = lean_unsigned_to_nat(3u);
v___x_1960_ = l_Lean_Expr_isAppOfArity(v_a_1951_, v___x_1958_, v___x_1959_);
if (v___x_1960_ == 0)
{
lean_object* v___x_1961_; lean_object* v___x_1962_; uint8_t v___x_1963_; 
v___x_1961_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType___closed__1));
v___x_1962_ = lean_unsigned_to_nat(4u);
v___x_1963_ = l_Lean_Expr_isAppOfArity(v_a_1951_, v___x_1961_, v___x_1962_);
lean_dec(v_a_1951_);
if (v___x_1963_ == 0)
{
lean_object* v___x_1964_; 
lean_del_object(v___x_1953_);
lean_inc(v_a_1946_);
lean_inc_ref(v_a_1945_);
lean_inc(v_a_1944_);
lean_inc_ref(v_a_1943_);
lean_inc_ref(v_lhs_1940_);
v___x_1964_ = lean_infer_type(v_lhs_1940_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1964_) == 0)
{
lean_object* v_a_1965_; lean_object* v___x_1966_; 
v_a_1965_ = lean_ctor_get(v___x_1964_, 0);
lean_inc(v_a_1965_);
lean_dec_ref_known(v___x_1964_, 1);
lean_inc(v_a_1946_);
lean_inc_ref(v_a_1945_);
lean_inc(v_a_1944_);
lean_inc_ref(v_a_1943_);
v___x_1966_ = lean_infer_type(v_rhs_1941_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1966_) == 0)
{
lean_object* v_a_1967_; lean_object* v___f_1968_; lean_object* v___x_1969_; 
v_a_1967_ = lean_ctor_get(v___x_1966_, 0);
lean_inc(v_a_1967_);
lean_dec_ref_known(v___x_1966_, 1);
v___f_1968_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1968_, 0, v_a_1965_);
lean_closure_set(v___f_1968_, 1, v_a_1967_);
v___x_1969_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf_spec__0___redArg(v___f_1968_, v___x_1963_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1969_) == 0)
{
lean_object* v_a_1970_; uint8_t v___x_1971_; 
v_a_1970_ = lean_ctor_get(v___x_1969_, 0);
lean_inc(v_a_1970_);
lean_dec_ref_known(v___x_1969_, 1);
v___x_1971_ = lean_unbox(v_a_1970_);
lean_dec(v_a_1970_);
if (v___x_1971_ == 0)
{
lean_object* v___x_1972_; 
lean_dec_ref(v_lhs_1940_);
lean_inc(v_a_1946_);
lean_inc_ref(v_a_1945_);
lean_inc(v_a_1944_);
lean_inc_ref(v_a_1943_);
lean_inc_ref(v_pf_1942_);
v___x_1972_ = lean_infer_type(v_pf_1942_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1972_) == 0)
{
lean_object* v_a_1973_; lean_object* v___x_1975_; uint8_t v_isShared_1976_; uint8_t v_isSharedCheck_1989_; 
v_a_1973_ = lean_ctor_get(v___x_1972_, 0);
v_isSharedCheck_1989_ = !lean_is_exclusive(v___x_1972_);
if (v_isSharedCheck_1989_ == 0)
{
v___x_1975_ = v___x_1972_;
v_isShared_1976_ = v_isSharedCheck_1989_;
goto v_resetjp_1974_;
}
else
{
lean_inc(v_a_1973_);
lean_dec(v___x_1972_);
v___x_1975_ = lean_box(0);
v_isShared_1976_ = v_isSharedCheck_1989_;
goto v_resetjp_1974_;
}
v_resetjp_1974_:
{
lean_object* v___x_1978_; 
if (v_isShared_1976_ == 0)
{
lean_ctor_set_tag(v___x_1975_, 1);
v___x_1978_ = v___x_1975_;
goto v_reusejp_1977_;
}
else
{
lean_object* v_reuseFailAlloc_1988_; 
v_reuseFailAlloc_1988_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1988_, 0, v_a_1973_);
v___x_1978_ = v_reuseFailAlloc_1988_;
goto v_reusejp_1977_;
}
v_reusejp_1977_:
{
lean_object* v___x_1979_; 
v___x_1979_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkHEqForExpectedType(v___x_1978_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1979_) == 0)
{
lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_1986_; 
v_isSharedCheck_1986_ = !lean_is_exclusive(v___x_1979_);
if (v_isSharedCheck_1986_ == 0)
{
lean_object* v_unused_1987_; 
v_unused_1987_ = lean_ctor_get(v___x_1979_, 0);
lean_dec(v_unused_1987_);
v___x_1981_ = v___x_1979_;
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
else
{
lean_dec(v___x_1979_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
lean_object* v___x_1984_; 
if (v_isShared_1982_ == 0)
{
lean_ctor_set(v___x_1981_, 0, v_pf_1942_);
v___x_1984_ = v___x_1981_;
goto v_reusejp_1983_;
}
else
{
lean_object* v_reuseFailAlloc_1985_; 
v_reuseFailAlloc_1985_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1985_, 0, v_pf_1942_);
v___x_1984_ = v_reuseFailAlloc_1985_;
goto v_reusejp_1983_;
}
v_reusejp_1983_:
{
return v___x_1984_;
}
}
}
else
{
lean_dec_ref(v_pf_1942_);
return v___x_1979_;
}
}
}
}
else
{
lean_dec_ref(v_pf_1942_);
return v___x_1972_;
}
}
else
{
lean_object* v___x_1990_; 
v___x_1990_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf(v_lhs_1940_, v_pf_1942_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_1990_) == 0)
{
lean_object* v_a_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; 
v_a_1991_ = lean_ctor_get(v___x_1990_, 0);
lean_inc(v_a_1991_);
lean_dec_ref_known(v___x_1990_, 1);
v___x_1992_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__1));
v___x_1993_ = lean_unsigned_to_nat(1u);
v___x_1994_ = lean_mk_empty_array_with_capacity(v___x_1993_);
v___x_1995_ = lean_array_push(v___x_1994_, v_a_1991_);
v___x_1996_ = l_Lean_Meta_mkAppM(v___x_1992_, v___x_1995_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
return v___x_1996_;
}
else
{
return v___x_1990_;
}
}
}
else
{
lean_object* v_a_1997_; lean_object* v___x_1999_; uint8_t v_isShared_2000_; uint8_t v_isSharedCheck_2004_; 
lean_dec_ref(v_pf_1942_);
lean_dec_ref(v_lhs_1940_);
v_a_1997_ = lean_ctor_get(v___x_1969_, 0);
v_isSharedCheck_2004_ = !lean_is_exclusive(v___x_1969_);
if (v_isSharedCheck_2004_ == 0)
{
v___x_1999_ = v___x_1969_;
v_isShared_2000_ = v_isSharedCheck_2004_;
goto v_resetjp_1998_;
}
else
{
lean_inc(v_a_1997_);
lean_dec(v___x_1969_);
v___x_1999_ = lean_box(0);
v_isShared_2000_ = v_isSharedCheck_2004_;
goto v_resetjp_1998_;
}
v_resetjp_1998_:
{
lean_object* v___x_2002_; 
if (v_isShared_2000_ == 0)
{
v___x_2002_ = v___x_1999_;
goto v_reusejp_2001_;
}
else
{
lean_object* v_reuseFailAlloc_2003_; 
v_reuseFailAlloc_2003_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2003_, 0, v_a_1997_);
v___x_2002_ = v_reuseFailAlloc_2003_;
goto v_reusejp_2001_;
}
v_reusejp_2001_:
{
return v___x_2002_;
}
}
}
}
else
{
lean_dec(v_a_1965_);
lean_dec_ref(v_pf_1942_);
lean_dec_ref(v_lhs_1940_);
return v___x_1966_;
}
}
else
{
lean_dec_ref(v_pf_1942_);
lean_dec_ref(v_rhs_1941_);
lean_dec_ref(v_lhs_1940_);
return v___x_1964_;
}
}
else
{
lean_object* v___x_2006_; 
lean_dec_ref(v_rhs_1941_);
lean_dec_ref(v_lhs_1940_);
if (v_isShared_1954_ == 0)
{
lean_ctor_set(v___x_1953_, 0, v_pf_1942_);
v___x_2006_ = v___x_1953_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v_pf_1942_);
v___x_2006_ = v_reuseFailAlloc_2007_;
goto v_reusejp_2005_;
}
v_reusejp_2005_:
{
return v___x_2006_;
}
}
}
else
{
lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; lean_object* v___x_2012_; 
lean_del_object(v___x_1953_);
lean_dec(v_a_1951_);
lean_dec_ref(v_rhs_1941_);
lean_dec_ref(v_lhs_1940_);
v___x_2008_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__1));
v___x_2009_ = lean_unsigned_to_nat(1u);
v___x_2010_ = lean_mk_empty_array_with_capacity(v___x_2009_);
v___x_2011_ = lean_array_push(v___x_2010_, v_pf_1942_);
v___x_2012_ = l_Lean_Meta_mkAppM(v___x_2008_, v___x_2011_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
return v___x_2012_;
}
}
else
{
lean_object* v___x_2013_; 
lean_del_object(v___x_1953_);
lean_dec(v_a_1951_);
lean_dec_ref(v_rhs_1941_);
lean_dec_ref(v_lhs_1940_);
v___x_2013_ = l_Lean_Meta_mkPropExt(v_pf_1942_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
if (lean_obj_tag(v___x_2013_) == 0)
{
lean_object* v_a_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v___x_2019_; 
v_a_2014_ = lean_ctor_get(v___x_2013_, 0);
lean_inc(v_a_2014_);
lean_dec_ref_known(v___x_2013_, 1);
v___x_2015_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___closed__1));
v___x_2016_ = lean_unsigned_to_nat(1u);
v___x_2017_ = lean_mk_empty_array_with_capacity(v___x_2016_);
v___x_2018_ = lean_array_push(v___x_2017_, v_a_2014_);
v___x_2019_ = l_Lean_Meta_mkAppM(v___x_2015_, v___x_2018_, v_a_1943_, v_a_1944_, v_a_1945_, v_a_1946_);
return v___x_2019_;
}
else
{
return v___x_2013_;
}
}
}
}
else
{
lean_dec_ref(v_pf_1942_);
lean_dec_ref(v_rhs_1941_);
lean_dec_ref(v_lhs_1940_);
return v___x_1950_;
}
}
else
{
lean_dec_ref(v_pf_1942_);
lean_dec_ref(v_rhs_1941_);
lean_dec_ref(v_lhs_1940_);
return v___x_1948_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf___boxed(lean_object* v_lhs_2021_, lean_object* v_rhs_2022_, lean_object* v_pf_2023_, lean_object* v_a_2024_, lean_object* v_a_2025_, lean_object* v_a_2026_, lean_object* v_a_2027_, lean_object* v_a_2028_){
_start:
{
lean_object* v_res_2029_; 
v_res_2029_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf(v_lhs_2021_, v_rhs_2022_, v_pf_2023_, v_a_2024_, v_a_2025_, v_a_2026_, v_a_2027_);
lean_dec(v_a_2027_);
lean_dec_ref(v_a_2026_);
lean_dec(v_a_2025_);
lean_dec_ref(v_a_2024_);
return v_res_2029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0(lean_object* v_msg_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_, lean_object* v___y_2034_, lean_object* v___y_2035_){
_start:
{
lean_object* v___f_2037_; lean_object* v___x_1513__overap_2038_; lean_object* v___x_2039_; 
v___f_2037_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0___closed__0));
v___x_1513__overap_2038_ = lean_panic_fn_borrowed(v___f_2037_, v_msg_2031_);
lean_inc(v___y_2035_);
lean_inc_ref(v___y_2034_);
lean_inc(v___y_2033_);
lean_inc_ref(v___y_2032_);
v___x_2039_ = lean_apply_5(v___x_1513__overap_2038_, v___y_2032_, v___y_2033_, v___y_2034_, v___y_2035_, lean_box(0));
return v___x_2039_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0___boxed(lean_object* v_msg_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_){
_start:
{
lean_object* v_res_2046_; 
v_res_2046_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0(v_msg_2040_, v___y_2041_, v___y_2042_, v___y_2043_, v___y_2044_);
lean_dec(v___y_2044_);
lean_dec_ref(v___y_2043_);
lean_dec(v___y_2042_);
lean_dec_ref(v___y_2041_);
return v_res_2046_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1(void){
_start:
{
lean_object* v___x_2048_; lean_object* v___x_2049_; 
v___x_2048_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__0));
v___x_2049_ = l_Lean_stringToMessageData(v___x_2048_);
return v___x_2049_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__3(void){
_start:
{
lean_object* v___x_2051_; lean_object* v___x_2052_; 
v___x_2051_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__2));
v___x_2052_ = l_Lean_stringToMessageData(v___x_2051_);
return v___x_2052_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__5(void){
_start:
{
lean_object* v___x_2054_; lean_object* v___x_2055_; 
v___x_2054_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__4));
v___x_2055_ = l_Lean_stringToMessageData(v___x_2054_);
return v___x_2055_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__9(void){
_start:
{
lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; 
v___x_2059_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__8));
v___x_2060_ = lean_unsigned_to_nat(8u);
v___x_2061_ = lean_unsigned_to_nat(377u);
v___x_2062_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__7));
v___x_2063_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__6));
v___x_2064_ = l_mkPanicMessageWithDecl(v___x_2063_, v___x_2062_, v___x_2061_, v___x_2060_, v___x_2059_);
return v___x_2064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq(lean_object* v_lhs_2065_, lean_object* v_rhs_2066_, lean_object* v_pf_2067_, lean_object* v_a_2068_, lean_object* v_a_2069_, lean_object* v_a_2070_, lean_object* v_a_2071_){
_start:
{
lean_object* v___x_2073_; 
lean_inc(v_a_2071_);
lean_inc_ref(v_a_2070_);
lean_inc(v_a_2069_);
lean_inc_ref(v_a_2068_);
lean_inc_ref(v_pf_2067_);
v___x_2073_ = lean_infer_type(v_pf_2067_, v_a_2068_, v_a_2069_, v_a_2070_, v_a_2071_);
if (lean_obj_tag(v___x_2073_) == 0)
{
lean_object* v_a_2074_; lean_object* v___x_2075_; 
v_a_2074_ = lean_ctor_get(v___x_2073_, 0);
lean_inc_n(v_a_2074_, 2);
lean_dec_ref_known(v___x_2073_, 1);
lean_inc(v_a_2071_);
lean_inc_ref(v_a_2070_);
lean_inc(v_a_2069_);
lean_inc_ref(v_a_2068_);
v___x_2075_ = lean_whnf(v_a_2074_, v_a_2068_, v_a_2069_, v_a_2070_, v_a_2071_);
if (lean_obj_tag(v___x_2075_) == 0)
{
lean_object* v_a_2076_; lean_object* v___x_2077_; 
v_a_2076_ = lean_ctor_get(v___x_2075_, 0);
lean_inc(v_a_2076_);
lean_dec_ref_known(v___x_2075_, 1);
v___x_2077_ = lp_mathlib_Lean_Expr_sides_x3f(v_a_2076_);
lean_dec(v_a_2076_);
if (lean_obj_tag(v___x_2077_) == 1)
{
lean_object* v_val_2078_; lean_object* v_snd_2079_; lean_object* v___x_2081_; uint8_t v_isShared_2082_; uint8_t v_isSharedCheck_2171_; 
v_val_2078_ = lean_ctor_get(v___x_2077_, 0);
lean_inc(v_val_2078_);
lean_dec_ref_known(v___x_2077_, 1);
v_snd_2079_ = lean_ctor_get(v_val_2078_, 1);
v_isSharedCheck_2171_ = !lean_is_exclusive(v_val_2078_);
if (v_isSharedCheck_2171_ == 0)
{
lean_object* v_unused_2172_; 
v_unused_2172_ = lean_ctor_get(v_val_2078_, 0);
lean_dec(v_unused_2172_);
v___x_2081_ = v_val_2078_;
v_isShared_2082_ = v_isSharedCheck_2171_;
goto v_resetjp_2080_;
}
else
{
lean_inc(v_snd_2079_);
lean_dec(v_val_2078_);
v___x_2081_ = lean_box(0);
v_isShared_2082_ = v_isSharedCheck_2171_;
goto v_resetjp_2080_;
}
v_resetjp_2080_:
{
lean_object* v_snd_2083_; lean_object* v_fst_2084_; lean_object* v___x_2086_; uint8_t v_isShared_2087_; uint8_t v_isSharedCheck_2170_; 
v_snd_2083_ = lean_ctor_get(v_snd_2079_, 1);
v_fst_2084_ = lean_ctor_get(v_snd_2079_, 0);
v_isSharedCheck_2170_ = !lean_is_exclusive(v_snd_2079_);
if (v_isSharedCheck_2170_ == 0)
{
v___x_2086_ = v_snd_2079_;
v_isShared_2087_ = v_isSharedCheck_2170_;
goto v_resetjp_2085_;
}
else
{
lean_inc(v_snd_2083_);
lean_inc(v_fst_2084_);
lean_dec(v_snd_2079_);
v___x_2086_ = lean_box(0);
v_isShared_2087_ = v_isSharedCheck_2170_;
goto v_resetjp_2085_;
}
v_resetjp_2085_:
{
lean_object* v_snd_2088_; lean_object* v___x_2090_; uint8_t v_isShared_2091_; uint8_t v_isSharedCheck_2168_; 
v_snd_2088_ = lean_ctor_get(v_snd_2083_, 1);
v_isSharedCheck_2168_ = !lean_is_exclusive(v_snd_2083_);
if (v_isSharedCheck_2168_ == 0)
{
lean_object* v_unused_2169_; 
v_unused_2169_ = lean_ctor_get(v_snd_2083_, 0);
lean_dec(v_unused_2169_);
v___x_2090_ = v_snd_2083_;
v_isShared_2091_ = v_isSharedCheck_2168_;
goto v_resetjp_2089_;
}
else
{
lean_inc(v_snd_2088_);
lean_dec(v_snd_2083_);
v___x_2090_ = lean_box(0);
v_isShared_2091_ = v_isSharedCheck_2168_;
goto v_resetjp_2089_;
}
v_resetjp_2089_:
{
lean_object* v___y_2093_; lean_object* v___y_2094_; lean_object* v___y_2095_; lean_object* v___y_2096_; lean_object* v___x_2139_; 
lean_inc_ref(v_lhs_2065_);
v___x_2139_ = l_Lean_Meta_isExprDefEq(v_lhs_2065_, v_fst_2084_, v_a_2068_, v_a_2069_, v_a_2070_, v_a_2071_);
if (lean_obj_tag(v___x_2139_) == 0)
{
lean_object* v_a_2140_; uint8_t v___x_2141_; 
v_a_2140_ = lean_ctor_get(v___x_2139_, 0);
lean_inc(v_a_2140_);
lean_dec_ref_known(v___x_2139_, 1);
v___x_2141_ = lean_unbox(v_a_2140_);
lean_dec(v_a_2140_);
if (v___x_2141_ == 0)
{
lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v_a_2152_; lean_object* v___x_2154_; uint8_t v_isShared_2155_; uint8_t v_isSharedCheck_2159_; 
lean_del_object(v___x_2090_);
lean_dec(v_snd_2088_);
lean_del_object(v___x_2086_);
lean_del_object(v___x_2081_);
lean_dec_ref(v_pf_2067_);
lean_dec_ref(v_rhs_2066_);
v___x_2142_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1);
v___x_2143_ = l_Lean_MessageData_ofExpr(v_a_2074_);
v___x_2144_ = l_Lean_indentD(v___x_2143_);
v___x_2145_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2145_, 0, v___x_2142_);
lean_ctor_set(v___x_2145_, 1, v___x_2144_);
v___x_2146_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__5, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__5);
v___x_2147_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2147_, 0, v___x_2145_);
lean_ctor_set(v___x_2147_, 1, v___x_2146_);
v___x_2148_ = l_Lean_MessageData_ofExpr(v_lhs_2065_);
v___x_2149_ = l_Lean_indentD(v___x_2148_);
v___x_2150_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2150_, 0, v___x_2147_);
lean_ctor_set(v___x_2150_, 1, v___x_2149_);
v___x_2151_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2150_, v_a_2068_, v_a_2069_, v_a_2070_, v_a_2071_);
v_a_2152_ = lean_ctor_get(v___x_2151_, 0);
v_isSharedCheck_2159_ = !lean_is_exclusive(v___x_2151_);
if (v_isSharedCheck_2159_ == 0)
{
v___x_2154_ = v___x_2151_;
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
else
{
lean_inc(v_a_2152_);
lean_dec(v___x_2151_);
v___x_2154_ = lean_box(0);
v_isShared_2155_ = v_isSharedCheck_2159_;
goto v_resetjp_2153_;
}
v_resetjp_2153_:
{
lean_object* v___x_2157_; 
if (v_isShared_2155_ == 0)
{
v___x_2157_ = v___x_2154_;
goto v_reusejp_2156_;
}
else
{
lean_object* v_reuseFailAlloc_2158_; 
v_reuseFailAlloc_2158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2158_, 0, v_a_2152_);
v___x_2157_ = v_reuseFailAlloc_2158_;
goto v_reusejp_2156_;
}
v_reusejp_2156_:
{
return v___x_2157_;
}
}
}
else
{
lean_dec_ref(v_lhs_2065_);
v___y_2093_ = v_a_2068_;
v___y_2094_ = v_a_2069_;
v___y_2095_ = v_a_2070_;
v___y_2096_ = v_a_2071_;
goto v___jp_2092_;
}
}
else
{
lean_object* v_a_2160_; lean_object* v___x_2162_; uint8_t v_isShared_2163_; uint8_t v_isSharedCheck_2167_; 
lean_del_object(v___x_2090_);
lean_dec(v_snd_2088_);
lean_del_object(v___x_2086_);
lean_del_object(v___x_2081_);
lean_dec(v_a_2074_);
lean_dec_ref(v_pf_2067_);
lean_dec_ref(v_rhs_2066_);
lean_dec_ref(v_lhs_2065_);
v_a_2160_ = lean_ctor_get(v___x_2139_, 0);
v_isSharedCheck_2167_ = !lean_is_exclusive(v___x_2139_);
if (v_isSharedCheck_2167_ == 0)
{
v___x_2162_ = v___x_2139_;
v_isShared_2163_ = v_isSharedCheck_2167_;
goto v_resetjp_2161_;
}
else
{
lean_inc(v_a_2160_);
lean_dec(v___x_2139_);
v___x_2162_ = lean_box(0);
v_isShared_2163_ = v_isSharedCheck_2167_;
goto v_resetjp_2161_;
}
v_resetjp_2161_:
{
lean_object* v___x_2165_; 
if (v_isShared_2163_ == 0)
{
v___x_2165_ = v___x_2162_;
goto v_reusejp_2164_;
}
else
{
lean_object* v_reuseFailAlloc_2166_; 
v_reuseFailAlloc_2166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2166_, 0, v_a_2160_);
v___x_2165_ = v_reuseFailAlloc_2166_;
goto v_reusejp_2164_;
}
v_reusejp_2164_:
{
return v___x_2165_;
}
}
}
v___jp_2092_:
{
lean_object* v___x_2097_; 
lean_inc_ref(v_rhs_2066_);
v___x_2097_ = l_Lean_Meta_isExprDefEq(v_rhs_2066_, v_snd_2088_, v___y_2093_, v___y_2094_, v___y_2095_, v___y_2096_);
if (lean_obj_tag(v___x_2097_) == 0)
{
lean_object* v_a_2098_; lean_object* v___x_2100_; uint8_t v_isShared_2101_; uint8_t v_isSharedCheck_2130_; 
v_a_2098_ = lean_ctor_get(v___x_2097_, 0);
v_isSharedCheck_2130_ = !lean_is_exclusive(v___x_2097_);
if (v_isSharedCheck_2130_ == 0)
{
v___x_2100_ = v___x_2097_;
v_isShared_2101_ = v_isSharedCheck_2130_;
goto v_resetjp_2099_;
}
else
{
lean_inc(v_a_2098_);
lean_dec(v___x_2097_);
v___x_2100_ = lean_box(0);
v_isShared_2101_ = v_isSharedCheck_2130_;
goto v_resetjp_2099_;
}
v_resetjp_2099_:
{
uint8_t v___x_2102_; 
v___x_2102_ = lean_unbox(v_a_2098_);
lean_dec(v_a_2098_);
if (v___x_2102_ == 0)
{
lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2107_; 
lean_del_object(v___x_2100_);
lean_dec_ref(v_pf_2067_);
v___x_2103_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__1);
v___x_2104_ = l_Lean_MessageData_ofExpr(v_a_2074_);
v___x_2105_ = l_Lean_indentD(v___x_2104_);
if (v_isShared_2091_ == 0)
{
lean_ctor_set_tag(v___x_2090_, 7);
lean_ctor_set(v___x_2090_, 1, v___x_2105_);
lean_ctor_set(v___x_2090_, 0, v___x_2103_);
v___x_2107_ = v___x_2090_;
goto v_reusejp_2106_;
}
else
{
lean_object* v_reuseFailAlloc_2126_; 
v_reuseFailAlloc_2126_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2126_, 0, v___x_2103_);
lean_ctor_set(v_reuseFailAlloc_2126_, 1, v___x_2105_);
v___x_2107_ = v_reuseFailAlloc_2126_;
goto v_reusejp_2106_;
}
v_reusejp_2106_:
{
lean_object* v___x_2108_; lean_object* v___x_2110_; 
v___x_2108_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__3, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__3);
if (v_isShared_2087_ == 0)
{
lean_ctor_set_tag(v___x_2086_, 7);
lean_ctor_set(v___x_2086_, 1, v___x_2108_);
lean_ctor_set(v___x_2086_, 0, v___x_2107_);
v___x_2110_ = v___x_2086_;
goto v_reusejp_2109_;
}
else
{
lean_object* v_reuseFailAlloc_2125_; 
v_reuseFailAlloc_2125_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2125_, 0, v___x_2107_);
lean_ctor_set(v_reuseFailAlloc_2125_, 1, v___x_2108_);
v___x_2110_ = v_reuseFailAlloc_2125_;
goto v_reusejp_2109_;
}
v_reusejp_2109_:
{
lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2114_; 
v___x_2111_ = l_Lean_MessageData_ofExpr(v_rhs_2066_);
v___x_2112_ = l_Lean_indentD(v___x_2111_);
if (v_isShared_2082_ == 0)
{
lean_ctor_set_tag(v___x_2081_, 7);
lean_ctor_set(v___x_2081_, 1, v___x_2112_);
lean_ctor_set(v___x_2081_, 0, v___x_2110_);
v___x_2114_ = v___x_2081_;
goto v_reusejp_2113_;
}
else
{
lean_object* v_reuseFailAlloc_2124_; 
v_reuseFailAlloc_2124_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2124_, 0, v___x_2110_);
lean_ctor_set(v_reuseFailAlloc_2124_, 1, v___x_2112_);
v___x_2114_ = v_reuseFailAlloc_2124_;
goto v_reusejp_2113_;
}
v_reusejp_2113_:
{
lean_object* v___x_2115_; lean_object* v_a_2116_; lean_object* v___x_2118_; uint8_t v_isShared_2119_; uint8_t v_isSharedCheck_2123_; 
v___x_2115_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2114_, v___y_2093_, v___y_2094_, v___y_2095_, v___y_2096_);
v_a_2116_ = lean_ctor_get(v___x_2115_, 0);
v_isSharedCheck_2123_ = !lean_is_exclusive(v___x_2115_);
if (v_isSharedCheck_2123_ == 0)
{
v___x_2118_ = v___x_2115_;
v_isShared_2119_ = v_isSharedCheck_2123_;
goto v_resetjp_2117_;
}
else
{
lean_inc(v_a_2116_);
lean_dec(v___x_2115_);
v___x_2118_ = lean_box(0);
v_isShared_2119_ = v_isSharedCheck_2123_;
goto v_resetjp_2117_;
}
v_resetjp_2117_:
{
lean_object* v___x_2121_; 
if (v_isShared_2119_ == 0)
{
v___x_2121_ = v___x_2118_;
goto v_reusejp_2120_;
}
else
{
lean_object* v_reuseFailAlloc_2122_; 
v_reuseFailAlloc_2122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2122_, 0, v_a_2116_);
v___x_2121_ = v_reuseFailAlloc_2122_;
goto v_reusejp_2120_;
}
v_reusejp_2120_:
{
return v___x_2121_;
}
}
}
}
}
}
else
{
lean_object* v___x_2128_; 
lean_del_object(v___x_2090_);
lean_del_object(v___x_2086_);
lean_del_object(v___x_2081_);
lean_dec(v_a_2074_);
lean_dec_ref(v_rhs_2066_);
if (v_isShared_2101_ == 0)
{
lean_ctor_set(v___x_2100_, 0, v_pf_2067_);
v___x_2128_ = v___x_2100_;
goto v_reusejp_2127_;
}
else
{
lean_object* v_reuseFailAlloc_2129_; 
v_reuseFailAlloc_2129_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2129_, 0, v_pf_2067_);
v___x_2128_ = v_reuseFailAlloc_2129_;
goto v_reusejp_2127_;
}
v_reusejp_2127_:
{
return v___x_2128_;
}
}
}
}
else
{
lean_object* v_a_2131_; lean_object* v___x_2133_; uint8_t v_isShared_2134_; uint8_t v_isSharedCheck_2138_; 
lean_del_object(v___x_2090_);
lean_del_object(v___x_2086_);
lean_del_object(v___x_2081_);
lean_dec(v_a_2074_);
lean_dec_ref(v_pf_2067_);
lean_dec_ref(v_rhs_2066_);
v_a_2131_ = lean_ctor_get(v___x_2097_, 0);
v_isSharedCheck_2138_ = !lean_is_exclusive(v___x_2097_);
if (v_isSharedCheck_2138_ == 0)
{
v___x_2133_ = v___x_2097_;
v_isShared_2134_ = v_isSharedCheck_2138_;
goto v_resetjp_2132_;
}
else
{
lean_inc(v_a_2131_);
lean_dec(v___x_2097_);
v___x_2133_ = lean_box(0);
v_isShared_2134_ = v_isSharedCheck_2138_;
goto v_resetjp_2132_;
}
v_resetjp_2132_:
{
lean_object* v___x_2136_; 
if (v_isShared_2134_ == 0)
{
v___x_2136_ = v___x_2133_;
goto v_reusejp_2135_;
}
else
{
lean_object* v_reuseFailAlloc_2137_; 
v_reuseFailAlloc_2137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2137_, 0, v_a_2131_);
v___x_2136_ = v_reuseFailAlloc_2137_;
goto v_reusejp_2135_;
}
v_reusejp_2135_:
{
return v___x_2136_;
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
lean_object* v___x_2173_; lean_object* v___x_2174_; 
lean_dec(v___x_2077_);
lean_dec(v_a_2074_);
lean_dec_ref(v_pf_2067_);
lean_dec_ref(v_rhs_2066_);
lean_dec_ref(v_lhs_2065_);
v___x_2173_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__9, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__9);
v___x_2174_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq_spec__0(v___x_2173_, v_a_2068_, v_a_2069_, v_a_2070_, v_a_2071_);
return v___x_2174_;
}
}
else
{
lean_dec(v_a_2074_);
lean_dec_ref(v_pf_2067_);
lean_dec_ref(v_rhs_2066_);
lean_dec_ref(v_lhs_2065_);
return v___x_2075_;
}
}
else
{
lean_dec_ref(v_pf_2067_);
lean_dec_ref(v_rhs_2066_);
lean_dec_ref(v_lhs_2065_);
return v___x_2073_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___boxed(lean_object* v_lhs_2175_, lean_object* v_rhs_2176_, lean_object* v_pf_2177_, lean_object* v_a_2178_, lean_object* v_a_2179_, lean_object* v_a_2180_, lean_object* v_a_2181_, lean_object* v_a_2182_){
_start:
{
lean_object* v_res_2183_; 
v_res_2183_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq(v_lhs_2175_, v_rhs_2176_, v_pf_2177_, v_a_2178_, v_a_2179_, v_a_2180_, v_a_2181_);
lean_dec(v_a_2181_);
lean_dec_ref(v_a_2180_);
lean_dec(v_a_2179_);
lean_dec_ref(v_a_2178_);
return v_res_2183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27___lam__0(lean_object* v_lhs_2184_, lean_object* v_pf_2185_, lean_object* v_rhs_2186_, uint8_t v_x_2187_, lean_object* v___y_2188_, lean_object* v___y_2189_, lean_object* v___y_2190_, lean_object* v___y_2191_){
_start:
{
if (v_x_2187_ == 0)
{
lean_object* v___x_2193_; 
lean_inc_ref(v_lhs_2184_);
v___x_2193_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toEqPf(v_lhs_2184_, v_pf_2185_, v___y_2188_, v___y_2189_, v___y_2190_, v___y_2191_);
if (lean_obj_tag(v___x_2193_) == 0)
{
lean_object* v_a_2194_; lean_object* v___x_2195_; 
v_a_2194_ = lean_ctor_get(v___x_2193_, 0);
lean_inc(v_a_2194_);
lean_dec_ref_known(v___x_2193_, 1);
v___x_2195_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq(v_lhs_2184_, v_rhs_2186_, v_a_2194_, v___y_2188_, v___y_2189_, v___y_2190_, v___y_2191_);
return v___x_2195_;
}
else
{
lean_dec_ref(v_rhs_2186_);
lean_dec_ref(v_lhs_2184_);
return v___x_2193_;
}
}
else
{
lean_object* v___x_2196_; 
lean_inc_ref(v_rhs_2186_);
lean_inc_ref(v_lhs_2184_);
v___x_2196_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_toHEqPf(v_lhs_2184_, v_rhs_2186_, v_pf_2185_, v___y_2188_, v___y_2189_, v___y_2190_, v___y_2191_);
if (lean_obj_tag(v___x_2196_) == 0)
{
lean_object* v_a_2197_; lean_object* v___x_2198_; 
v_a_2197_ = lean_ctor_get(v___x_2196_, 0);
lean_inc(v_a_2197_);
lean_dec_ref_known(v___x_2196_, 1);
v___x_2198_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq(v_lhs_2184_, v_rhs_2186_, v_a_2197_, v___y_2188_, v___y_2189_, v___y_2190_, v___y_2191_);
return v___x_2198_;
}
else
{
lean_dec_ref(v_rhs_2186_);
lean_dec_ref(v_lhs_2184_);
return v___x_2196_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27___lam__0___boxed(lean_object* v_lhs_2199_, lean_object* v_pf_2200_, lean_object* v_rhs_2201_, lean_object* v_x_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_, lean_object* v___y_2206_, lean_object* v___y_2207_){
_start:
{
uint8_t v_x_372__boxed_2208_; lean_object* v_res_2209_; 
v_x_372__boxed_2208_ = lean_unbox(v_x_2202_);
v_res_2209_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27___lam__0(v_lhs_2199_, v_pf_2200_, v_rhs_2201_, v_x_372__boxed_2208_, v___y_2203_, v___y_2204_, v___y_2205_, v___y_2206_);
lean_dec(v___y_2206_);
lean_dec_ref(v___y_2205_);
lean_dec(v___y_2204_);
lean_dec_ref(v___y_2203_);
return v_res_2209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(lean_object* v_lhs_2210_, lean_object* v_rhs_2211_, lean_object* v_pf_2212_){
_start:
{
lean_object* v___x_2213_; 
v___x_2213_ = l_Lean_Meta_isRefl_x3f(v_pf_2212_);
if (lean_obj_tag(v___x_2213_) == 0)
{
lean_object* v___f_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; 
lean_inc_ref(v_rhs_2211_);
lean_inc_ref(v_lhs_2210_);
v___f_2214_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27___lam__0___boxed), 9, 3);
lean_closure_set(v___f_2214_, 0, v_lhs_2210_);
lean_closure_set(v___f_2214_, 1, v_pf_2212_);
lean_closure_set(v___f_2214_, 2, v_rhs_2211_);
v___x_2215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2215_, 0, v___f_2214_);
v___x_2216_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2216_, 0, v_lhs_2210_);
lean_ctor_set(v___x_2216_, 1, v_rhs_2211_);
lean_ctor_set(v___x_2216_, 2, v___x_2215_);
return v___x_2216_;
}
else
{
lean_object* v___x_2217_; lean_object* v___x_2218_; 
lean_dec_ref_known(v___x_2213_, 1);
lean_dec_ref(v_pf_2212_);
v___x_2217_ = lean_box(0);
v___x_2218_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2218_, 0, v_lhs_2210_);
lean_ctor_set(v___x_2218_, 1, v_rhs_2211_);
lean_ctor_set(v___x_2218_, 2, v___x_2217_);
return v___x_2218_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__1(void){
_start:
{
lean_object* v___x_2220_; lean_object* v___x_2221_; 
v___x_2220_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__0));
v___x_2221_ = l_Lean_stringToMessageData(v___x_2220_);
return v___x_2221_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__3(void){
_start:
{
lean_object* v___x_2223_; lean_object* v___x_2224_; 
v___x_2223_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__2));
v___x_2224_ = l_Lean_stringToMessageData(v___x_2223_);
return v___x_2224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq(lean_object* v_res_2225_, lean_object* v_a_2226_, lean_object* v_a_2227_, lean_object* v_a_2228_, lean_object* v_a_2229_){
_start:
{
uint8_t v___x_2231_; 
v___x_2231_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_res_2225_);
if (v___x_2231_ == 0)
{
lean_object* v_lhs_2232_; lean_object* v_rhs_2233_; lean_object* v___y_2235_; lean_object* v___y_2236_; lean_object* v___y_2237_; lean_object* v___y_2238_; lean_object* v___x_2258_; 
v_lhs_2232_ = lean_ctor_get(v_res_2225_, 0);
lean_inc_ref_n(v_lhs_2232_, 2);
v_rhs_2233_ = lean_ctor_get(v_res_2225_, 1);
lean_inc_ref_n(v_rhs_2233_, 2);
v___x_2258_ = l_Lean_Meta_isExprDefEq(v_lhs_2232_, v_rhs_2233_, v_a_2226_, v_a_2227_, v_a_2228_, v_a_2229_);
if (lean_obj_tag(v___x_2258_) == 0)
{
lean_object* v_a_2259_; uint8_t v___x_2260_; 
v_a_2259_ = lean_ctor_get(v___x_2258_, 0);
lean_inc(v_a_2259_);
lean_dec_ref_known(v___x_2258_, 1);
v___x_2260_ = lean_unbox(v_a_2259_);
lean_dec(v_a_2259_);
if (v___x_2260_ == 0)
{
lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v_a_2271_; lean_object* v___x_2273_; uint8_t v_isShared_2274_; uint8_t v_isSharedCheck_2278_; 
lean_dec_ref(v_res_2225_);
v___x_2261_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__1);
v___x_2262_ = l_Lean_MessageData_ofExpr(v_lhs_2232_);
v___x_2263_ = l_Lean_indentD(v___x_2262_);
v___x_2264_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2264_, 0, v___x_2261_);
lean_ctor_set(v___x_2264_, 1, v___x_2263_);
v___x_2265_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___closed__3);
v___x_2266_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2266_, 0, v___x_2264_);
lean_ctor_set(v___x_2266_, 1, v___x_2265_);
v___x_2267_ = l_Lean_MessageData_ofExpr(v_rhs_2233_);
v___x_2268_ = l_Lean_indentD(v___x_2267_);
v___x_2269_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2269_, 0, v___x_2266_);
lean_ctor_set(v___x_2269_, 1, v___x_2268_);
v___x_2270_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2269_, v_a_2226_, v_a_2227_, v_a_2228_, v_a_2229_);
v_a_2271_ = lean_ctor_get(v___x_2270_, 0);
v_isSharedCheck_2278_ = !lean_is_exclusive(v___x_2270_);
if (v_isSharedCheck_2278_ == 0)
{
v___x_2273_ = v___x_2270_;
v_isShared_2274_ = v_isSharedCheck_2278_;
goto v_resetjp_2272_;
}
else
{
lean_inc(v_a_2271_);
lean_dec(v___x_2270_);
v___x_2273_ = lean_box(0);
v_isShared_2274_ = v_isSharedCheck_2278_;
goto v_resetjp_2272_;
}
v_resetjp_2272_:
{
lean_object* v___x_2276_; 
if (v_isShared_2274_ == 0)
{
v___x_2276_ = v___x_2273_;
goto v_reusejp_2275_;
}
else
{
lean_object* v_reuseFailAlloc_2277_; 
v_reuseFailAlloc_2277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2277_, 0, v_a_2271_);
v___x_2276_ = v_reuseFailAlloc_2277_;
goto v_reusejp_2275_;
}
v_reusejp_2275_:
{
return v___x_2276_;
}
}
}
else
{
v___y_2235_ = v_a_2226_;
v___y_2236_ = v_a_2227_;
v___y_2237_ = v_a_2228_;
v___y_2238_ = v_a_2229_;
goto v___jp_2234_;
}
}
else
{
lean_object* v_a_2279_; lean_object* v___x_2281_; uint8_t v_isShared_2282_; uint8_t v_isSharedCheck_2286_; 
lean_dec_ref(v_rhs_2233_);
lean_dec_ref(v_lhs_2232_);
lean_dec_ref(v_res_2225_);
v_a_2279_ = lean_ctor_get(v___x_2258_, 0);
v_isSharedCheck_2286_ = !lean_is_exclusive(v___x_2258_);
if (v_isSharedCheck_2286_ == 0)
{
v___x_2281_ = v___x_2258_;
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
else
{
lean_inc(v_a_2279_);
lean_dec(v___x_2258_);
v___x_2281_ = lean_box(0);
v_isShared_2282_ = v_isSharedCheck_2286_;
goto v_resetjp_2280_;
}
v_resetjp_2280_:
{
lean_object* v___x_2284_; 
if (v_isShared_2282_ == 0)
{
v___x_2284_ = v___x_2281_;
goto v_reusejp_2283_;
}
else
{
lean_object* v_reuseFailAlloc_2285_; 
v_reuseFailAlloc_2285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2285_, 0, v_a_2279_);
v___x_2284_ = v_reuseFailAlloc_2285_;
goto v_reusejp_2283_;
}
v_reusejp_2283_:
{
return v___x_2284_;
}
}
}
v___jp_2234_:
{
lean_object* v___x_2239_; 
v___x_2239_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_res_2225_, v___y_2235_, v___y_2236_, v___y_2237_, v___y_2238_);
if (lean_obj_tag(v___x_2239_) == 0)
{
lean_object* v___x_2241_; uint8_t v_isShared_2242_; uint8_t v_isSharedCheck_2248_; 
v_isSharedCheck_2248_ = !lean_is_exclusive(v___x_2239_);
if (v_isSharedCheck_2248_ == 0)
{
lean_object* v_unused_2249_; 
v_unused_2249_ = lean_ctor_get(v___x_2239_, 0);
lean_dec(v_unused_2249_);
v___x_2241_ = v___x_2239_;
v_isShared_2242_ = v_isSharedCheck_2248_;
goto v_resetjp_2240_;
}
else
{
lean_dec(v___x_2239_);
v___x_2241_ = lean_box(0);
v_isShared_2242_ = v_isSharedCheck_2248_;
goto v_resetjp_2240_;
}
v_resetjp_2240_:
{
lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2246_; 
v___x_2243_ = lean_box(0);
v___x_2244_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2244_, 0, v_lhs_2232_);
lean_ctor_set(v___x_2244_, 1, v_rhs_2233_);
lean_ctor_set(v___x_2244_, 2, v___x_2243_);
if (v_isShared_2242_ == 0)
{
lean_ctor_set(v___x_2241_, 0, v___x_2244_);
v___x_2246_ = v___x_2241_;
goto v_reusejp_2245_;
}
else
{
lean_object* v_reuseFailAlloc_2247_; 
v_reuseFailAlloc_2247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2247_, 0, v___x_2244_);
v___x_2246_ = v_reuseFailAlloc_2247_;
goto v_reusejp_2245_;
}
v_reusejp_2245_:
{
return v___x_2246_;
}
}
}
else
{
lean_object* v_a_2250_; lean_object* v___x_2252_; uint8_t v_isShared_2253_; uint8_t v_isSharedCheck_2257_; 
lean_dec_ref(v_rhs_2233_);
lean_dec_ref(v_lhs_2232_);
v_a_2250_ = lean_ctor_get(v___x_2239_, 0);
v_isSharedCheck_2257_ = !lean_is_exclusive(v___x_2239_);
if (v_isSharedCheck_2257_ == 0)
{
v___x_2252_ = v___x_2239_;
v_isShared_2253_ = v_isSharedCheck_2257_;
goto v_resetjp_2251_;
}
else
{
lean_inc(v_a_2250_);
lean_dec(v___x_2239_);
v___x_2252_ = lean_box(0);
v_isShared_2253_ = v_isSharedCheck_2257_;
goto v_resetjp_2251_;
}
v_resetjp_2251_:
{
lean_object* v___x_2255_; 
if (v_isShared_2253_ == 0)
{
v___x_2255_ = v___x_2252_;
goto v_reusejp_2254_;
}
else
{
lean_object* v_reuseFailAlloc_2256_; 
v_reuseFailAlloc_2256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2256_, 0, v_a_2250_);
v___x_2255_ = v_reuseFailAlloc_2256_;
goto v_reusejp_2254_;
}
v_reusejp_2254_:
{
return v___x_2255_;
}
}
}
}
}
else
{
lean_object* v___x_2287_; 
v___x_2287_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2287_, 0, v_res_2225_);
return v___x_2287_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq___boxed(lean_object* v_res_2288_, lean_object* v_a_2289_, lean_object* v_a_2290_, lean_object* v_a_2291_, lean_object* v_a_2292_, lean_object* v_a_2293_){
_start:
{
lean_object* v_res_2294_; 
v_res_2294_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq(v_res_2288_, v_a_2289_, v_a_2290_, v_a_2291_, v_a_2292_);
lean_dec(v_a_2292_);
lean_dec_ref(v_a_2291_);
lean_dec(v_a_2290_);
lean_dec_ref(v_a_2289_);
return v_res_2294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg(lean_object* v_x_2295_, lean_object* v___y_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_){
_start:
{
lean_object* v___x_2301_; 
v___x_2301_ = l_Lean_Meta_saveState___redArg(v___y_2297_, v___y_2299_);
if (lean_obj_tag(v___x_2301_) == 0)
{
lean_object* v_a_2302_; lean_object* v___x_2303_; 
v_a_2302_ = lean_ctor_get(v___x_2301_, 0);
lean_inc(v_a_2302_);
lean_dec_ref_known(v___x_2301_, 1);
lean_inc(v___y_2299_);
lean_inc_ref(v___y_2298_);
lean_inc(v___y_2297_);
lean_inc_ref(v___y_2296_);
v___x_2303_ = lean_apply_5(v_x_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_, lean_box(0));
if (lean_obj_tag(v___x_2303_) == 0)
{
lean_object* v_a_2304_; lean_object* v___x_2306_; uint8_t v_isShared_2307_; uint8_t v_isSharedCheck_2312_; 
lean_dec(v_a_2302_);
v_a_2304_ = lean_ctor_get(v___x_2303_, 0);
v_isSharedCheck_2312_ = !lean_is_exclusive(v___x_2303_);
if (v_isSharedCheck_2312_ == 0)
{
v___x_2306_ = v___x_2303_;
v_isShared_2307_ = v_isSharedCheck_2312_;
goto v_resetjp_2305_;
}
else
{
lean_inc(v_a_2304_);
lean_dec(v___x_2303_);
v___x_2306_ = lean_box(0);
v_isShared_2307_ = v_isSharedCheck_2312_;
goto v_resetjp_2305_;
}
v_resetjp_2305_:
{
lean_object* v___x_2308_; lean_object* v___x_2310_; 
v___x_2308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2308_, 0, v_a_2304_);
if (v_isShared_2307_ == 0)
{
lean_ctor_set(v___x_2306_, 0, v___x_2308_);
v___x_2310_ = v___x_2306_;
goto v_reusejp_2309_;
}
else
{
lean_object* v_reuseFailAlloc_2311_; 
v_reuseFailAlloc_2311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2311_, 0, v___x_2308_);
v___x_2310_ = v_reuseFailAlloc_2311_;
goto v_reusejp_2309_;
}
v_reusejp_2309_:
{
return v___x_2310_;
}
}
}
else
{
lean_object* v_a_2313_; lean_object* v___x_2315_; uint8_t v_isShared_2316_; uint8_t v_isSharedCheck_2342_; 
v_a_2313_ = lean_ctor_get(v___x_2303_, 0);
v_isSharedCheck_2342_ = !lean_is_exclusive(v___x_2303_);
if (v_isSharedCheck_2342_ == 0)
{
v___x_2315_ = v___x_2303_;
v_isShared_2316_ = v_isSharedCheck_2342_;
goto v_resetjp_2314_;
}
else
{
lean_inc(v_a_2313_);
lean_dec(v___x_2303_);
v___x_2315_ = lean_box(0);
v_isShared_2316_ = v_isSharedCheck_2342_;
goto v_resetjp_2314_;
}
v_resetjp_2314_:
{
uint8_t v___y_2318_; uint8_t v___x_2340_; 
v___x_2340_ = l_Lean_Exception_isInterrupt(v_a_2313_);
if (v___x_2340_ == 0)
{
uint8_t v___x_2341_; 
lean_inc(v_a_2313_);
v___x_2341_ = l_Lean_Exception_isRuntime(v_a_2313_);
v___y_2318_ = v___x_2341_;
goto v___jp_2317_;
}
else
{
v___y_2318_ = v___x_2340_;
goto v___jp_2317_;
}
v___jp_2317_:
{
if (v___y_2318_ == 0)
{
lean_object* v___x_2319_; 
lean_del_object(v___x_2315_);
lean_dec(v_a_2313_);
v___x_2319_ = l_Lean_Meta_SavedState_restore___redArg(v_a_2302_, v___y_2297_, v___y_2299_);
lean_dec(v_a_2302_);
if (lean_obj_tag(v___x_2319_) == 0)
{
lean_object* v___x_2321_; uint8_t v_isShared_2322_; uint8_t v_isSharedCheck_2327_; 
v_isSharedCheck_2327_ = !lean_is_exclusive(v___x_2319_);
if (v_isSharedCheck_2327_ == 0)
{
lean_object* v_unused_2328_; 
v_unused_2328_ = lean_ctor_get(v___x_2319_, 0);
lean_dec(v_unused_2328_);
v___x_2321_ = v___x_2319_;
v_isShared_2322_ = v_isSharedCheck_2327_;
goto v_resetjp_2320_;
}
else
{
lean_dec(v___x_2319_);
v___x_2321_ = lean_box(0);
v_isShared_2322_ = v_isSharedCheck_2327_;
goto v_resetjp_2320_;
}
v_resetjp_2320_:
{
lean_object* v___x_2323_; lean_object* v___x_2325_; 
v___x_2323_ = lean_box(0);
if (v_isShared_2322_ == 0)
{
lean_ctor_set(v___x_2321_, 0, v___x_2323_);
v___x_2325_ = v___x_2321_;
goto v_reusejp_2324_;
}
else
{
lean_object* v_reuseFailAlloc_2326_; 
v_reuseFailAlloc_2326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2326_, 0, v___x_2323_);
v___x_2325_ = v_reuseFailAlloc_2326_;
goto v_reusejp_2324_;
}
v_reusejp_2324_:
{
return v___x_2325_;
}
}
}
else
{
lean_object* v_a_2329_; lean_object* v___x_2331_; uint8_t v_isShared_2332_; uint8_t v_isSharedCheck_2336_; 
v_a_2329_ = lean_ctor_get(v___x_2319_, 0);
v_isSharedCheck_2336_ = !lean_is_exclusive(v___x_2319_);
if (v_isSharedCheck_2336_ == 0)
{
v___x_2331_ = v___x_2319_;
v_isShared_2332_ = v_isSharedCheck_2336_;
goto v_resetjp_2330_;
}
else
{
lean_inc(v_a_2329_);
lean_dec(v___x_2319_);
v___x_2331_ = lean_box(0);
v_isShared_2332_ = v_isSharedCheck_2336_;
goto v_resetjp_2330_;
}
v_resetjp_2330_:
{
lean_object* v___x_2334_; 
if (v_isShared_2332_ == 0)
{
v___x_2334_ = v___x_2331_;
goto v_reusejp_2333_;
}
else
{
lean_object* v_reuseFailAlloc_2335_; 
v_reuseFailAlloc_2335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2335_, 0, v_a_2329_);
v___x_2334_ = v_reuseFailAlloc_2335_;
goto v_reusejp_2333_;
}
v_reusejp_2333_:
{
return v___x_2334_;
}
}
}
}
else
{
lean_object* v___x_2338_; 
lean_dec(v_a_2302_);
if (v_isShared_2316_ == 0)
{
v___x_2338_ = v___x_2315_;
goto v_reusejp_2337_;
}
else
{
lean_object* v_reuseFailAlloc_2339_; 
v_reuseFailAlloc_2339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2339_, 0, v_a_2313_);
v___x_2338_ = v_reuseFailAlloc_2339_;
goto v_reusejp_2337_;
}
v_reusejp_2337_:
{
return v___x_2338_;
}
}
}
}
}
}
else
{
lean_object* v_a_2343_; lean_object* v___x_2345_; uint8_t v_isShared_2346_; uint8_t v_isSharedCheck_2350_; 
lean_dec_ref(v_x_2295_);
v_a_2343_ = lean_ctor_get(v___x_2301_, 0);
v_isSharedCheck_2350_ = !lean_is_exclusive(v___x_2301_);
if (v_isSharedCheck_2350_ == 0)
{
v___x_2345_ = v___x_2301_;
v_isShared_2346_ = v_isSharedCheck_2350_;
goto v_resetjp_2344_;
}
else
{
lean_inc(v_a_2343_);
lean_dec(v___x_2301_);
v___x_2345_ = lean_box(0);
v_isShared_2346_ = v_isSharedCheck_2350_;
goto v_resetjp_2344_;
}
v_resetjp_2344_:
{
lean_object* v___x_2348_; 
if (v_isShared_2346_ == 0)
{
v___x_2348_ = v___x_2345_;
goto v_reusejp_2347_;
}
else
{
lean_object* v_reuseFailAlloc_2349_; 
v_reuseFailAlloc_2349_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2349_, 0, v_a_2343_);
v___x_2348_ = v_reuseFailAlloc_2349_;
goto v_reusejp_2347_;
}
v_reusejp_2347_:
{
return v___x_2348_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg___boxed(lean_object* v_x_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_){
_start:
{
lean_object* v_res_2357_; 
v_res_2357_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg(v_x_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_);
lean_dec(v___y_2355_);
lean_dec_ref(v___y_2354_);
lean_dec(v___y_2353_);
lean_dec_ref(v___y_2352_);
return v_res_2357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0(lean_object* v_00_u03b1_2358_, lean_object* v_x_2359_, lean_object* v___y_2360_, lean_object* v___y_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_){
_start:
{
lean_object* v___x_2365_; 
v___x_2365_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg(v_x_2359_, v___y_2360_, v___y_2361_, v___y_2362_, v___y_2363_);
return v___x_2365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___boxed(lean_object* v_00_u03b1_2366_, lean_object* v_x_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_, lean_object* v___y_2370_, lean_object* v___y_2371_, lean_object* v___y_2372_){
_start:
{
lean_object* v_res_2373_; 
v_res_2373_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0(v_00_u03b1_2366_, v_x_2367_, v___y_2368_, v___y_2369_, v___y_2370_, v___y_2371_);
lean_dec(v___y_2371_);
lean_dec_ref(v___y_2370_);
lean_dec(v___y_2369_);
lean_dec_ref(v___y_2368_);
return v_res_2373_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__6(void){
_start:
{
lean_object* v___x_2383_; lean_object* v___x_2384_; 
v___x_2383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__5));
v___x_2384_ = l_Lean_stringToMessageData(v___x_2383_);
return v___x_2384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault(lean_object* v_lhs_2385_, lean_object* v_rhs_2386_, lean_object* v_a_2387_, lean_object* v_a_2388_, lean_object* v_a_2389_, lean_object* v_a_2390_){
_start:
{
lean_object* v_pf_2393_; lean_object* v___x_2396_; 
lean_inc_ref(v_rhs_2386_);
lean_inc_ref(v_lhs_2385_);
v___x_2396_ = l_Lean_Meta_isExprDefEq(v_lhs_2385_, v_rhs_2386_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_);
if (lean_obj_tag(v___x_2396_) == 0)
{
lean_object* v_a_2397_; lean_object* v___x_2399_; uint8_t v_isShared_2400_; uint8_t v_isSharedCheck_2447_; 
v_a_2397_ = lean_ctor_get(v___x_2396_, 0);
v_isSharedCheck_2447_ = !lean_is_exclusive(v___x_2396_);
if (v_isSharedCheck_2447_ == 0)
{
v___x_2399_ = v___x_2396_;
v_isShared_2400_ = v_isSharedCheck_2447_;
goto v_resetjp_2398_;
}
else
{
lean_inc(v_a_2397_);
lean_dec(v___x_2396_);
v___x_2399_ = lean_box(0);
v_isShared_2400_ = v_isSharedCheck_2447_;
goto v_resetjp_2398_;
}
v_resetjp_2398_:
{
uint8_t v___x_2401_; 
v___x_2401_ = lean_unbox(v_a_2397_);
lean_dec(v_a_2397_);
if (v___x_2401_ == 0)
{
lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; lean_object* v___x_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; 
lean_del_object(v___x_2399_);
v___x_2402_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__2));
v___x_2403_ = lean_unsigned_to_nat(2u);
v___x_2404_ = lean_mk_empty_array_with_capacity(v___x_2403_);
lean_inc_ref(v_lhs_2385_);
v___x_2405_ = lean_array_push(v___x_2404_, v_lhs_2385_);
lean_inc_ref(v_rhs_2386_);
v___x_2406_ = lean_array_push(v___x_2405_, v_rhs_2386_);
lean_inc_ref(v___x_2406_);
v___x_2407_ = lean_alloc_closure((void*)(l_Lean_Meta_mkAppM___boxed), 7, 2);
lean_closure_set(v___x_2407_, 0, v___x_2402_);
lean_closure_set(v___x_2407_, 1, v___x_2406_);
v___x_2408_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg(v___x_2407_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_);
if (lean_obj_tag(v___x_2408_) == 0)
{
lean_object* v_a_2409_; 
v_a_2409_ = lean_ctor_get(v___x_2408_, 0);
lean_inc(v_a_2409_);
lean_dec_ref_known(v___x_2408_, 1);
if (lean_obj_tag(v_a_2409_) == 1)
{
lean_object* v_val_2410_; 
lean_dec_ref(v___x_2406_);
v_val_2410_ = lean_ctor_get(v_a_2409_, 0);
lean_inc(v_val_2410_);
lean_dec_ref_known(v_a_2409_, 1);
v_pf_2393_ = v_val_2410_;
goto v___jp_2392_;
}
else
{
lean_object* v___x_2411_; lean_object* v___x_2412_; lean_object* v___x_2413_; 
lean_dec(v_a_2409_);
v___x_2411_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__4));
v___x_2412_ = lean_alloc_closure((void*)(l_Lean_Meta_mkAppM___boxed), 7, 2);
lean_closure_set(v___x_2412_, 0, v___x_2411_);
lean_closure_set(v___x_2412_, 1, v___x_2406_);
v___x_2413_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_TermCongr_CongrResult_mkDefault_spec__0___redArg(v___x_2412_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_);
if (lean_obj_tag(v___x_2413_) == 0)
{
lean_object* v_a_2414_; 
v_a_2414_ = lean_ctor_get(v___x_2413_, 0);
lean_inc(v_a_2414_);
lean_dec_ref_known(v___x_2413_, 1);
if (lean_obj_tag(v_a_2414_) == 1)
{
lean_object* v_val_2415_; 
v_val_2415_ = lean_ctor_get(v_a_2414_, 0);
lean_inc(v_val_2415_);
lean_dec_ref_known(v_a_2414_, 1);
v_pf_2393_ = v_val_2415_;
goto v___jp_2392_;
}
else
{
lean_object* v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; 
lean_dec(v_a_2414_);
v___x_2416_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__6, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___closed__6);
v___x_2417_ = l_Lean_MessageData_ofExpr(v_lhs_2385_);
v___x_2418_ = l_Lean_indentD(v___x_2417_);
v___x_2419_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2419_, 0, v___x_2416_);
lean_ctor_set(v___x_2419_, 1, v___x_2418_);
v___x_2420_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq___closed__3);
v___x_2421_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2421_, 0, v___x_2419_);
lean_ctor_set(v___x_2421_, 1, v___x_2420_);
v___x_2422_ = l_Lean_MessageData_ofExpr(v_rhs_2386_);
v___x_2423_ = l_Lean_indentD(v___x_2422_);
v___x_2424_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2424_, 0, v___x_2421_);
lean_ctor_set(v___x_2424_, 1, v___x_2423_);
v___x_2425_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2424_, v_a_2387_, v_a_2388_, v_a_2389_, v_a_2390_);
return v___x_2425_;
}
}
else
{
lean_object* v_a_2426_; lean_object* v___x_2428_; uint8_t v_isShared_2429_; uint8_t v_isSharedCheck_2433_; 
lean_dec_ref(v_rhs_2386_);
lean_dec_ref(v_lhs_2385_);
v_a_2426_ = lean_ctor_get(v___x_2413_, 0);
v_isSharedCheck_2433_ = !lean_is_exclusive(v___x_2413_);
if (v_isSharedCheck_2433_ == 0)
{
v___x_2428_ = v___x_2413_;
v_isShared_2429_ = v_isSharedCheck_2433_;
goto v_resetjp_2427_;
}
else
{
lean_inc(v_a_2426_);
lean_dec(v___x_2413_);
v___x_2428_ = lean_box(0);
v_isShared_2429_ = v_isSharedCheck_2433_;
goto v_resetjp_2427_;
}
v_resetjp_2427_:
{
lean_object* v___x_2431_; 
if (v_isShared_2429_ == 0)
{
v___x_2431_ = v___x_2428_;
goto v_reusejp_2430_;
}
else
{
lean_object* v_reuseFailAlloc_2432_; 
v_reuseFailAlloc_2432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2432_, 0, v_a_2426_);
v___x_2431_ = v_reuseFailAlloc_2432_;
goto v_reusejp_2430_;
}
v_reusejp_2430_:
{
return v___x_2431_;
}
}
}
}
}
else
{
lean_object* v_a_2434_; lean_object* v___x_2436_; uint8_t v_isShared_2437_; uint8_t v_isSharedCheck_2441_; 
lean_dec_ref(v___x_2406_);
lean_dec_ref(v_rhs_2386_);
lean_dec_ref(v_lhs_2385_);
v_a_2434_ = lean_ctor_get(v___x_2408_, 0);
v_isSharedCheck_2441_ = !lean_is_exclusive(v___x_2408_);
if (v_isSharedCheck_2441_ == 0)
{
v___x_2436_ = v___x_2408_;
v_isShared_2437_ = v_isSharedCheck_2441_;
goto v_resetjp_2435_;
}
else
{
lean_inc(v_a_2434_);
lean_dec(v___x_2408_);
v___x_2436_ = lean_box(0);
v_isShared_2437_ = v_isSharedCheck_2441_;
goto v_resetjp_2435_;
}
v_resetjp_2435_:
{
lean_object* v___x_2439_; 
if (v_isShared_2437_ == 0)
{
v___x_2439_ = v___x_2436_;
goto v_reusejp_2438_;
}
else
{
lean_object* v_reuseFailAlloc_2440_; 
v_reuseFailAlloc_2440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2440_, 0, v_a_2434_);
v___x_2439_ = v_reuseFailAlloc_2440_;
goto v_reusejp_2438_;
}
v_reusejp_2438_:
{
return v___x_2439_;
}
}
}
}
else
{
lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2445_; 
v___x_2442_ = lean_box(0);
v___x_2443_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2443_, 0, v_lhs_2385_);
lean_ctor_set(v___x_2443_, 1, v_rhs_2386_);
lean_ctor_set(v___x_2443_, 2, v___x_2442_);
if (v_isShared_2400_ == 0)
{
lean_ctor_set(v___x_2399_, 0, v___x_2443_);
v___x_2445_ = v___x_2399_;
goto v_reusejp_2444_;
}
else
{
lean_object* v_reuseFailAlloc_2446_; 
v_reuseFailAlloc_2446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2446_, 0, v___x_2443_);
v___x_2445_ = v_reuseFailAlloc_2446_;
goto v_reusejp_2444_;
}
v_reusejp_2444_:
{
return v___x_2445_;
}
}
}
}
else
{
lean_object* v_a_2448_; lean_object* v___x_2450_; uint8_t v_isShared_2451_; uint8_t v_isSharedCheck_2455_; 
lean_dec_ref(v_rhs_2386_);
lean_dec_ref(v_lhs_2385_);
v_a_2448_ = lean_ctor_get(v___x_2396_, 0);
v_isSharedCheck_2455_ = !lean_is_exclusive(v___x_2396_);
if (v_isSharedCheck_2455_ == 0)
{
v___x_2450_ = v___x_2396_;
v_isShared_2451_ = v_isSharedCheck_2455_;
goto v_resetjp_2449_;
}
else
{
lean_inc(v_a_2448_);
lean_dec(v___x_2396_);
v___x_2450_ = lean_box(0);
v_isShared_2451_ = v_isSharedCheck_2455_;
goto v_resetjp_2449_;
}
v_resetjp_2449_:
{
lean_object* v___x_2453_; 
if (v_isShared_2451_ == 0)
{
v___x_2453_ = v___x_2450_;
goto v_reusejp_2452_;
}
else
{
lean_object* v_reuseFailAlloc_2454_; 
v_reuseFailAlloc_2454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2454_, 0, v_a_2448_);
v___x_2453_ = v_reuseFailAlloc_2454_;
goto v_reusejp_2452_;
}
v_reusejp_2452_:
{
return v___x_2453_;
}
}
}
v___jp_2392_:
{
lean_object* v___x_2394_; lean_object* v___x_2395_; 
v___x_2394_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v_lhs_2385_, v_rhs_2386_, v_pf_2393_);
v___x_2395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2395_, 0, v___x_2394_);
return v___x_2395_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault___boxed(lean_object* v_lhs_2456_, lean_object* v_rhs_2457_, lean_object* v_a_2458_, lean_object* v_a_2459_, lean_object* v_a_2460_, lean_object* v_a_2461_, lean_object* v_a_2462_){
_start:
{
lean_object* v_res_2463_; 
v_res_2463_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault(v_lhs_2456_, v_rhs_2457_, v_a_2458_, v_a_2459_, v_a_2460_, v_a_2461_);
lean_dec(v_a_2461_);
lean_dec_ref(v_a_2460_);
lean_dec(v_a_2459_);
lean_dec_ref(v_a_2458_);
return v_res_2463_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__1(void){
_start:
{
lean_object* v___x_2465_; lean_object* v___x_2466_; 
v___x_2465_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__0));
v___x_2466_ = l_Lean_stringToMessageData(v___x_2465_);
return v___x_2466_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3(void){
_start:
{
lean_object* v___x_2468_; lean_object* v___x_2469_; 
v___x_2468_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__2));
v___x_2469_ = l_Lean_stringToMessageData(v___x_2468_);
return v___x_2469_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__5(void){
_start:
{
lean_object* v___x_2471_; lean_object* v___x_2472_; 
v___x_2471_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__4));
v___x_2472_ = l_Lean_stringToMessageData(v___x_2471_);
return v___x_2472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(lean_object* v_mvarCounterSaved_2473_, lean_object* v_lhs_2474_, lean_object* v_rhs_2475_, lean_object* v_a_2476_, lean_object* v_a_2477_, lean_object* v_a_2478_, lean_object* v_a_2479_){
_start:
{
lean_object* v___y_2482_; lean_object* v___y_2483_; lean_object* v___y_2484_; lean_object* v___y_2485_; lean_object* v___x_2507_; 
lean_inc(v_mvarCounterSaved_2473_);
v___x_2507_ = lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole(v_mvarCounterSaved_2473_, v_lhs_2474_);
if (lean_obj_tag(v___x_2507_) == 1)
{
lean_object* v_val_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; lean_object* v___x_2518_; lean_object* v_a_2519_; lean_object* v___x_2521_; uint8_t v_isShared_2522_; uint8_t v_isSharedCheck_2526_; 
lean_dec_ref(v_rhs_2475_);
lean_dec(v_mvarCounterSaved_2473_);
v_val_2508_ = lean_ctor_get(v___x_2507_, 0);
lean_inc(v_val_2508_);
lean_dec_ref_known(v___x_2507_, 1);
v___x_2509_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__5, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__5);
v___x_2510_ = l_Lean_MessageData_ofExpr(v_lhs_2474_);
v___x_2511_ = l_Lean_indentD(v___x_2510_);
v___x_2512_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2512_, 0, v___x_2509_);
lean_ctor_set(v___x_2512_, 1, v___x_2511_);
v___x_2513_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3);
v___x_2514_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2514_, 0, v___x_2512_);
lean_ctor_set(v___x_2514_, 1, v___x_2513_);
v___x_2515_ = l_Lean_MessageData_ofExpr(v_val_2508_);
v___x_2516_ = l_Lean_indentD(v___x_2515_);
v___x_2517_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2517_, 0, v___x_2514_);
lean_ctor_set(v___x_2517_, 1, v___x_2516_);
v___x_2518_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2517_, v_a_2476_, v_a_2477_, v_a_2478_, v_a_2479_);
v_a_2519_ = lean_ctor_get(v___x_2518_, 0);
v_isSharedCheck_2526_ = !lean_is_exclusive(v___x_2518_);
if (v_isSharedCheck_2526_ == 0)
{
v___x_2521_ = v___x_2518_;
v_isShared_2522_ = v_isSharedCheck_2526_;
goto v_resetjp_2520_;
}
else
{
lean_inc(v_a_2519_);
lean_dec(v___x_2518_);
v___x_2521_ = lean_box(0);
v_isShared_2522_ = v_isSharedCheck_2526_;
goto v_resetjp_2520_;
}
v_resetjp_2520_:
{
lean_object* v___x_2524_; 
if (v_isShared_2522_ == 0)
{
v___x_2524_ = v___x_2521_;
goto v_reusejp_2523_;
}
else
{
lean_object* v_reuseFailAlloc_2525_; 
v_reuseFailAlloc_2525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2525_, 0, v_a_2519_);
v___x_2524_ = v_reuseFailAlloc_2525_;
goto v_reusejp_2523_;
}
v_reusejp_2523_:
{
return v___x_2524_;
}
}
}
else
{
lean_dec(v___x_2507_);
v___y_2482_ = v_a_2476_;
v___y_2483_ = v_a_2477_;
v___y_2484_ = v_a_2478_;
v___y_2485_ = v_a_2479_;
goto v___jp_2481_;
}
v___jp_2481_:
{
lean_object* v___x_2486_; 
v___x_2486_ = lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole(v_mvarCounterSaved_2473_, v_rhs_2475_);
if (lean_obj_tag(v___x_2486_) == 1)
{
lean_object* v_val_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; lean_object* v_a_2498_; lean_object* v___x_2500_; uint8_t v_isShared_2501_; uint8_t v_isSharedCheck_2505_; 
lean_dec_ref(v_lhs_2474_);
v_val_2487_ = lean_ctor_get(v___x_2486_, 0);
lean_inc(v_val_2487_);
lean_dec_ref_known(v___x_2486_, 1);
v___x_2488_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__1);
v___x_2489_ = l_Lean_MessageData_ofExpr(v_rhs_2475_);
v___x_2490_ = l_Lean_indentD(v___x_2489_);
v___x_2491_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2491_, 0, v___x_2488_);
lean_ctor_set(v___x_2491_, 1, v___x_2490_);
v___x_2492_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___closed__3);
v___x_2493_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2493_, 0, v___x_2491_);
lean_ctor_set(v___x_2493_, 1, v___x_2492_);
v___x_2494_ = l_Lean_MessageData_ofExpr(v_val_2487_);
v___x_2495_ = l_Lean_indentD(v___x_2494_);
v___x_2496_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2496_, 0, v___x_2493_);
lean_ctor_set(v___x_2496_, 1, v___x_2495_);
v___x_2497_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2496_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_);
v_a_2498_ = lean_ctor_get(v___x_2497_, 0);
v_isSharedCheck_2505_ = !lean_is_exclusive(v___x_2497_);
if (v_isSharedCheck_2505_ == 0)
{
v___x_2500_ = v___x_2497_;
v_isShared_2501_ = v_isSharedCheck_2505_;
goto v_resetjp_2499_;
}
else
{
lean_inc(v_a_2498_);
lean_dec(v___x_2497_);
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
else
{
lean_object* v___x_2506_; 
lean_dec(v___x_2486_);
v___x_2506_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault(v_lhs_2474_, v_rhs_2475_, v___y_2482_, v___y_2483_, v___y_2484_, v___y_2485_);
return v___x_2506_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27___boxed(lean_object* v_mvarCounterSaved_2527_, lean_object* v_lhs_2528_, lean_object* v_rhs_2529_, lean_object* v_a_2530_, lean_object* v_a_2531_, lean_object* v_a_2532_, lean_object* v_a_2533_, lean_object* v_a_2534_){
_start:
{
lean_object* v_res_2535_; 
v_res_2535_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_2527_, v_lhs_2528_, v_rhs_2529_, v_a_2530_, v_a_2531_, v_a_2532_, v_a_2533_);
lean_dec(v_a_2533_);
lean_dec_ref(v_a_2532_);
lean_dec(v_a_2531_);
lean_dec_ref(v_a_2530_);
return v_res_2535_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__1(void){
_start:
{
lean_object* v___x_2537_; lean_object* v___x_2538_; 
v___x_2537_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__0));
v___x_2538_ = l_Lean_stringToMessageData(v___x_2537_);
return v___x_2538_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__3(void){
_start:
{
lean_object* v___x_2540_; lean_object* v___x_2541_; 
v___x_2540_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__2));
v___x_2541_ = l_Lean_stringToMessageData(v___x_2540_);
return v___x_2541_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__5(void){
_start:
{
lean_object* v___x_2543_; lean_object* v___x_2544_; 
v___x_2543_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__4));
v___x_2544_ = l_Lean_stringToMessageData(v___x_2543_);
return v___x_2544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(lean_object* v_lhs_2545_, lean_object* v_rhs_2546_, lean_object* v_msg_2547_, lean_object* v_a_2548_, lean_object* v_a_2549_, lean_object* v_a_2550_, lean_object* v_a_2551_){
_start:
{
lean_object* v___x_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; 
v___x_2553_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__1);
v___x_2554_ = l_Lean_MessageData_ofExpr(v_lhs_2545_);
v___x_2555_ = l_Lean_indentD(v___x_2554_);
v___x_2556_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2556_, 0, v___x_2553_);
lean_ctor_set(v___x_2556_, 1, v___x_2555_);
v___x_2557_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__3);
v___x_2558_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2558_, 0, v___x_2556_);
lean_ctor_set(v___x_2558_, 1, v___x_2557_);
v___x_2559_ = l_Lean_MessageData_ofExpr(v_rhs_2546_);
v___x_2560_ = l_Lean_indentD(v___x_2559_);
v___x_2561_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2561_, 0, v___x_2558_);
lean_ctor_set(v___x_2561_, 1, v___x_2560_);
v___x_2562_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__5, &lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___closed__5);
v___x_2563_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2563_, 0, v___x_2561_);
lean_ctor_set(v___x_2563_, 1, v___x_2562_);
v___x_2564_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2564_, 0, v___x_2563_);
lean_ctor_set(v___x_2564_, 1, v_msg_2547_);
v___x_2565_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2564_, v_a_2548_, v_a_2549_, v_a_2550_, v_a_2551_);
return v___x_2565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg___boxed(lean_object* v_lhs_2566_, lean_object* v_rhs_2567_, lean_object* v_msg_2568_, lean_object* v_a_2569_, lean_object* v_a_2570_, lean_object* v_a_2571_, lean_object* v_a_2572_, lean_object* v_a_2573_){
_start:
{
lean_object* v_res_2574_; 
v_res_2574_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_lhs_2566_, v_rhs_2567_, v_msg_2568_, v_a_2569_, v_a_2570_, v_a_2571_, v_a_2572_);
lean_dec(v_a_2572_);
lean_dec_ref(v_a_2571_);
lean_dec(v_a_2570_);
lean_dec_ref(v_a_2569_);
return v_res_2574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx(lean_object* v_00_u03b1_2575_, lean_object* v_lhs_2576_, lean_object* v_rhs_2577_, lean_object* v_msg_2578_, lean_object* v_a_2579_, lean_object* v_a_2580_, lean_object* v_a_2581_, lean_object* v_a_2582_){
_start:
{
lean_object* v___x_2584_; 
v___x_2584_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_lhs_2576_, v_rhs_2577_, v_msg_2578_, v_a_2579_, v_a_2580_, v_a_2581_, v_a_2582_);
return v___x_2584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___boxed(lean_object* v_00_u03b1_2585_, lean_object* v_lhs_2586_, lean_object* v_rhs_2587_, lean_object* v_msg_2588_, lean_object* v_a_2589_, lean_object* v_a_2590_, lean_object* v_a_2591_, lean_object* v_a_2592_, lean_object* v_a_2593_){
_start:
{
lean_object* v_res_2594_; 
v_res_2594_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx(v_00_u03b1_2585_, v_lhs_2586_, v_rhs_2587_, v_msg_2588_, v_a_2589_, v_a_2590_, v_a_2591_, v_a_2592_);
lean_dec(v_a_2592_);
lean_dec_ref(v_a_2591_);
lean_dec(v_a_2590_);
lean_dec_ref(v_a_2589_);
return v_res_2594_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0(void){
_start:
{
lean_object* v___x_2595_; double v___x_2596_; 
v___x_2595_ = lean_unsigned_to_nat(0u);
v___x_2596_ = lean_float_of_nat(v___x_2595_);
return v___x_2596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0(lean_object* v_cls_2600_, lean_object* v_msg_2601_, lean_object* v___y_2602_, lean_object* v___y_2603_, lean_object* v___y_2604_, lean_object* v___y_2605_){
_start:
{
lean_object* v_ref_2607_; lean_object* v___x_2608_; lean_object* v_a_2609_; lean_object* v___x_2611_; uint8_t v_isShared_2612_; uint8_t v_isSharedCheck_2653_; 
v_ref_2607_ = lean_ctor_get(v___y_2604_, 5);
v___x_2608_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(v_msg_2601_, v___y_2602_, v___y_2603_, v___y_2604_, v___y_2605_);
v_a_2609_ = lean_ctor_get(v___x_2608_, 0);
v_isSharedCheck_2653_ = !lean_is_exclusive(v___x_2608_);
if (v_isSharedCheck_2653_ == 0)
{
v___x_2611_ = v___x_2608_;
v_isShared_2612_ = v_isSharedCheck_2653_;
goto v_resetjp_2610_;
}
else
{
lean_inc(v_a_2609_);
lean_dec(v___x_2608_);
v___x_2611_ = lean_box(0);
v_isShared_2612_ = v_isSharedCheck_2653_;
goto v_resetjp_2610_;
}
v_resetjp_2610_:
{
lean_object* v___x_2613_; lean_object* v_traceState_2614_; lean_object* v_env_2615_; lean_object* v_nextMacroScope_2616_; lean_object* v_ngen_2617_; lean_object* v_auxDeclNGen_2618_; lean_object* v_cache_2619_; lean_object* v_messages_2620_; lean_object* v_infoState_2621_; lean_object* v_snapshotTasks_2622_; lean_object* v___x_2624_; uint8_t v_isShared_2625_; uint8_t v_isSharedCheck_2652_; 
v___x_2613_ = lean_st_ref_take(v___y_2605_);
v_traceState_2614_ = lean_ctor_get(v___x_2613_, 4);
v_env_2615_ = lean_ctor_get(v___x_2613_, 0);
v_nextMacroScope_2616_ = lean_ctor_get(v___x_2613_, 1);
v_ngen_2617_ = lean_ctor_get(v___x_2613_, 2);
v_auxDeclNGen_2618_ = lean_ctor_get(v___x_2613_, 3);
v_cache_2619_ = lean_ctor_get(v___x_2613_, 5);
v_messages_2620_ = lean_ctor_get(v___x_2613_, 6);
v_infoState_2621_ = lean_ctor_get(v___x_2613_, 7);
v_snapshotTasks_2622_ = lean_ctor_get(v___x_2613_, 8);
v_isSharedCheck_2652_ = !lean_is_exclusive(v___x_2613_);
if (v_isSharedCheck_2652_ == 0)
{
v___x_2624_ = v___x_2613_;
v_isShared_2625_ = v_isSharedCheck_2652_;
goto v_resetjp_2623_;
}
else
{
lean_inc(v_snapshotTasks_2622_);
lean_inc(v_infoState_2621_);
lean_inc(v_messages_2620_);
lean_inc(v_cache_2619_);
lean_inc(v_traceState_2614_);
lean_inc(v_auxDeclNGen_2618_);
lean_inc(v_ngen_2617_);
lean_inc(v_nextMacroScope_2616_);
lean_inc(v_env_2615_);
lean_dec(v___x_2613_);
v___x_2624_ = lean_box(0);
v_isShared_2625_ = v_isSharedCheck_2652_;
goto v_resetjp_2623_;
}
v_resetjp_2623_:
{
uint64_t v_tid_2626_; lean_object* v_traces_2627_; lean_object* v___x_2629_; uint8_t v_isShared_2630_; uint8_t v_isSharedCheck_2651_; 
v_tid_2626_ = lean_ctor_get_uint64(v_traceState_2614_, sizeof(void*)*1);
v_traces_2627_ = lean_ctor_get(v_traceState_2614_, 0);
v_isSharedCheck_2651_ = !lean_is_exclusive(v_traceState_2614_);
if (v_isSharedCheck_2651_ == 0)
{
v___x_2629_ = v_traceState_2614_;
v_isShared_2630_ = v_isSharedCheck_2651_;
goto v_resetjp_2628_;
}
else
{
lean_inc(v_traces_2627_);
lean_dec(v_traceState_2614_);
v___x_2629_ = lean_box(0);
v_isShared_2630_ = v_isSharedCheck_2651_;
goto v_resetjp_2628_;
}
v_resetjp_2628_:
{
lean_object* v___x_2631_; double v___x_2632_; uint8_t v___x_2633_; lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2641_; 
v___x_2631_ = lean_box(0);
v___x_2632_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0);
v___x_2633_ = 0;
v___x_2634_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__1));
v___x_2635_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2635_, 0, v_cls_2600_);
lean_ctor_set(v___x_2635_, 1, v___x_2631_);
lean_ctor_set(v___x_2635_, 2, v___x_2634_);
lean_ctor_set_float(v___x_2635_, sizeof(void*)*3, v___x_2632_);
lean_ctor_set_float(v___x_2635_, sizeof(void*)*3 + 8, v___x_2632_);
lean_ctor_set_uint8(v___x_2635_, sizeof(void*)*3 + 16, v___x_2633_);
v___x_2636_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__2));
v___x_2637_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2637_, 0, v___x_2635_);
lean_ctor_set(v___x_2637_, 1, v_a_2609_);
lean_ctor_set(v___x_2637_, 2, v___x_2636_);
lean_inc(v_ref_2607_);
v___x_2638_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2638_, 0, v_ref_2607_);
lean_ctor_set(v___x_2638_, 1, v___x_2637_);
v___x_2639_ = l_Lean_PersistentArray_push___redArg(v_traces_2627_, v___x_2638_);
if (v_isShared_2630_ == 0)
{
lean_ctor_set(v___x_2629_, 0, v___x_2639_);
v___x_2641_ = v___x_2629_;
goto v_reusejp_2640_;
}
else
{
lean_object* v_reuseFailAlloc_2650_; 
v_reuseFailAlloc_2650_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2650_, 0, v___x_2639_);
lean_ctor_set_uint64(v_reuseFailAlloc_2650_, sizeof(void*)*1, v_tid_2626_);
v___x_2641_ = v_reuseFailAlloc_2650_;
goto v_reusejp_2640_;
}
v_reusejp_2640_:
{
lean_object* v___x_2643_; 
if (v_isShared_2625_ == 0)
{
lean_ctor_set(v___x_2624_, 4, v___x_2641_);
v___x_2643_ = v___x_2624_;
goto v_reusejp_2642_;
}
else
{
lean_object* v_reuseFailAlloc_2649_; 
v_reuseFailAlloc_2649_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2649_, 0, v_env_2615_);
lean_ctor_set(v_reuseFailAlloc_2649_, 1, v_nextMacroScope_2616_);
lean_ctor_set(v_reuseFailAlloc_2649_, 2, v_ngen_2617_);
lean_ctor_set(v_reuseFailAlloc_2649_, 3, v_auxDeclNGen_2618_);
lean_ctor_set(v_reuseFailAlloc_2649_, 4, v___x_2641_);
lean_ctor_set(v_reuseFailAlloc_2649_, 5, v_cache_2619_);
lean_ctor_set(v_reuseFailAlloc_2649_, 6, v_messages_2620_);
lean_ctor_set(v_reuseFailAlloc_2649_, 7, v_infoState_2621_);
lean_ctor_set(v_reuseFailAlloc_2649_, 8, v_snapshotTasks_2622_);
v___x_2643_ = v_reuseFailAlloc_2649_;
goto v_reusejp_2642_;
}
v_reusejp_2642_:
{
lean_object* v___x_2644_; lean_object* v___x_2645_; lean_object* v___x_2647_; 
v___x_2644_ = lean_st_ref_set(v___y_2605_, v___x_2643_);
v___x_2645_ = lean_box(0);
if (v_isShared_2612_ == 0)
{
lean_ctor_set(v___x_2611_, 0, v___x_2645_);
v___x_2647_ = v___x_2611_;
goto v_reusejp_2646_;
}
else
{
lean_object* v_reuseFailAlloc_2648_; 
v_reuseFailAlloc_2648_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2648_, 0, v___x_2645_);
v___x_2647_ = v_reuseFailAlloc_2648_;
goto v_reusejp_2646_;
}
v_reusejp_2646_:
{
return v___x_2647_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___boxed(lean_object* v_cls_2654_, lean_object* v_msg_2655_, lean_object* v___y_2656_, lean_object* v___y_2657_, lean_object* v___y_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_){
_start:
{
lean_object* v_res_2661_; 
v_res_2661_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0(v_cls_2654_, v_msg_2655_, v___y_2656_, v___y_2657_, v___y_2658_, v___y_2659_);
lean_dec(v___y_2659_);
lean_dec_ref(v___y_2658_);
lean_dec(v___y_2657_);
lean_dec_ref(v___y_2656_);
return v_res_2661_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__2(void){
_start:
{
lean_object* v___x_2665_; lean_object* v___x_2666_; 
v___x_2665_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__1));
v___x_2666_ = l_Lean_MessageData_ofFormat(v___x_2665_);
return v___x_2666_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__5(void){
_start:
{
lean_object* v___x_2670_; lean_object* v___x_2671_; 
v___x_2670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__4));
v___x_2671_ = l_Lean_MessageData_ofFormat(v___x_2670_);
return v___x_2671_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__7(void){
_start:
{
lean_object* v___x_2673_; lean_object* v___x_2674_; 
v___x_2673_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__6));
v___x_2674_ = l_Lean_stringToMessageData(v___x_2673_);
return v___x_2674_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9(void){
_start:
{
lean_object* v___x_2676_; lean_object* v___x_2677_; 
v___x_2676_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__8));
v___x_2677_ = l_Lean_stringToMessageData(v___x_2676_);
return v___x_2677_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__11(void){
_start:
{
lean_object* v___x_2679_; lean_object* v___x_2680_; 
v___x_2679_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__10));
v___x_2680_ = l_Lean_stringToMessageData(v___x_2679_);
return v___x_2680_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__14(void){
_start:
{
lean_object* v___x_2684_; lean_object* v___x_2685_; 
v___x_2684_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__13));
v___x_2685_ = l_Lean_MessageData_ofFormat(v___x_2684_);
return v___x_2685_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__17(void){
_start:
{
lean_object* v___x_2689_; lean_object* v___x_2690_; 
v___x_2689_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__16));
v___x_2690_ = l_Lean_MessageData_ofFormat(v___x_2689_);
return v___x_2690_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__20(void){
_start:
{
lean_object* v___x_2694_; lean_object* v___x_2695_; 
v___x_2694_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__19));
v___x_2695_ = l_Lean_MessageData_ofFormat(v___x_2694_);
return v___x_2695_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23(void){
_start:
{
lean_object* v_cls_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; 
v_cls_2699_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_2700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__22));
v___x_2701_ = l_Lean_Name_append(v___x_2700_, v_cls_2699_);
return v___x_2701_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__25(void){
_start:
{
lean_object* v___x_2703_; lean_object* v___x_2704_; 
v___x_2703_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__24));
v___x_2704_ = l_Lean_stringToMessageData(v___x_2703_);
return v___x_2704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f(lean_object* v_mvarCounterSaved_2705_, lean_object* v_lhs_2706_, lean_object* v_rhs_2707_, lean_object* v_a_2708_, lean_object* v_a_2709_, lean_object* v_a_2710_, lean_object* v_a_2711_){
_start:
{
lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; 
v___x_2713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2713_, 0, v_mvarCounterSaved_2705_);
lean_inc_ref(v_lhs_2706_);
v___x_2714_ = lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f(v_lhs_2706_, v___x_2713_);
lean_inc_ref(v_rhs_2707_);
v___x_2715_ = lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f(v_rhs_2707_, v___x_2713_);
lean_dec_ref_known(v___x_2713_, 1);
if (lean_obj_tag(v___x_2714_) == 0)
{
if (lean_obj_tag(v___x_2715_) == 0)
{
lean_object* v___x_2716_; lean_object* v___x_2717_; 
lean_dec_ref(v_rhs_2707_);
lean_dec_ref(v_lhs_2706_);
v___x_2716_ = lean_box(0);
v___x_2717_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2717_, 0, v___x_2716_);
return v___x_2717_;
}
else
{
lean_object* v___x_2718_; lean_object* v___x_2719_; 
lean_dec_ref_known(v___x_2715_, 1);
v___x_2718_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__2, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__2);
v___x_2719_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_lhs_2706_, v_rhs_2707_, v___x_2718_, v_a_2708_, v_a_2709_, v_a_2710_, v_a_2711_);
return v___x_2719_;
}
}
else
{
lean_object* v_val_2720_; lean_object* v___x_2722_; uint8_t v_isShared_2723_; uint8_t v_isSharedCheck_2975_; 
v_val_2720_ = lean_ctor_get(v___x_2714_, 0);
v_isSharedCheck_2975_ = !lean_is_exclusive(v___x_2714_);
if (v_isSharedCheck_2975_ == 0)
{
v___x_2722_ = v___x_2714_;
v_isShared_2723_ = v_isSharedCheck_2975_;
goto v_resetjp_2721_;
}
else
{
lean_inc(v_val_2720_);
lean_dec(v___x_2714_);
v___x_2722_ = lean_box(0);
v_isShared_2723_ = v_isSharedCheck_2975_;
goto v_resetjp_2721_;
}
v_resetjp_2721_:
{
lean_object* v_snd_2724_; 
v_snd_2724_ = lean_ctor_get(v_val_2720_, 1);
lean_inc(v_snd_2724_);
if (lean_obj_tag(v___x_2715_) == 0)
{
lean_object* v___x_2725_; lean_object* v___x_2726_; 
lean_dec(v_snd_2724_);
lean_del_object(v___x_2722_);
lean_dec(v_val_2720_);
v___x_2725_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__5, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__5);
v___x_2726_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_lhs_2706_, v_rhs_2707_, v___x_2725_, v_a_2708_, v_a_2709_, v_a_2710_, v_a_2711_);
return v___x_2726_;
}
else
{
lean_object* v_val_2727_; lean_object* v___x_2729_; uint8_t v_isShared_2730_; uint8_t v_isSharedCheck_2974_; 
v_val_2727_ = lean_ctor_get(v___x_2715_, 0);
v_isSharedCheck_2974_ = !lean_is_exclusive(v___x_2715_);
if (v_isSharedCheck_2974_ == 0)
{
v___x_2729_ = v___x_2715_;
v_isShared_2730_ = v_isSharedCheck_2974_;
goto v_resetjp_2728_;
}
else
{
lean_inc(v_val_2727_);
lean_dec(v___x_2715_);
v___x_2729_ = lean_box(0);
v_isShared_2730_ = v_isSharedCheck_2974_;
goto v_resetjp_2728_;
}
v_resetjp_2728_:
{
lean_object* v_snd_2731_; lean_object* v_fst_2732_; lean_object* v_fst_2733_; lean_object* v_snd_2734_; lean_object* v___x_2736_; uint8_t v_isShared_2737_; uint8_t v_isSharedCheck_2973_; 
v_snd_2731_ = lean_ctor_get(v_val_2727_, 1);
lean_inc(v_snd_2731_);
v_fst_2732_ = lean_ctor_get(v_val_2720_, 0);
lean_inc(v_fst_2732_);
lean_dec(v_val_2720_);
v_fst_2733_ = lean_ctor_get(v_snd_2724_, 0);
v_snd_2734_ = lean_ctor_get(v_snd_2724_, 1);
v_isSharedCheck_2973_ = !lean_is_exclusive(v_snd_2724_);
if (v_isSharedCheck_2973_ == 0)
{
v___x_2736_ = v_snd_2724_;
v_isShared_2737_ = v_isSharedCheck_2973_;
goto v_resetjp_2735_;
}
else
{
lean_inc(v_snd_2734_);
lean_inc(v_fst_2733_);
lean_dec(v_snd_2724_);
v___x_2736_ = lean_box(0);
v_isShared_2737_ = v_isSharedCheck_2973_;
goto v_resetjp_2735_;
}
v_resetjp_2735_:
{
lean_object* v_fst_2738_; lean_object* v___x_2740_; uint8_t v_isShared_2741_; uint8_t v_isSharedCheck_2971_; 
v_fst_2738_ = lean_ctor_get(v_val_2727_, 0);
v_isSharedCheck_2971_ = !lean_is_exclusive(v_val_2727_);
if (v_isSharedCheck_2971_ == 0)
{
lean_object* v_unused_2972_; 
v_unused_2972_ = lean_ctor_get(v_val_2727_, 1);
lean_dec(v_unused_2972_);
v___x_2740_ = v_val_2727_;
v_isShared_2741_ = v_isSharedCheck_2971_;
goto v_resetjp_2739_;
}
else
{
lean_inc(v_fst_2738_);
lean_dec(v_val_2727_);
v___x_2740_ = lean_box(0);
v_isShared_2741_ = v_isSharedCheck_2971_;
goto v_resetjp_2739_;
}
v_resetjp_2739_:
{
lean_object* v_fst_2742_; lean_object* v_snd_2743_; lean_object* v___x_2745_; uint8_t v_isShared_2746_; uint8_t v_isSharedCheck_2970_; 
v_fst_2742_ = lean_ctor_get(v_snd_2731_, 0);
v_snd_2743_ = lean_ctor_get(v_snd_2731_, 1);
v_isSharedCheck_2970_ = !lean_is_exclusive(v_snd_2731_);
if (v_isSharedCheck_2970_ == 0)
{
v___x_2745_ = v_snd_2731_;
v_isShared_2746_ = v_isSharedCheck_2970_;
goto v_resetjp_2744_;
}
else
{
lean_inc(v_snd_2743_);
lean_inc(v_fst_2742_);
lean_dec(v_snd_2731_);
v___x_2745_ = lean_box(0);
v_isShared_2746_ = v_isSharedCheck_2970_;
goto v_resetjp_2744_;
}
v_resetjp_2744_:
{
lean_object* v___y_2756_; lean_object* v___y_2757_; lean_object* v___y_2758_; lean_object* v___y_2759_; lean_object* v___y_2760_; lean_object* v___y_2797_; lean_object* v___y_2798_; lean_object* v___y_2799_; lean_object* v___y_2800_; lean_object* v___y_2877_; lean_object* v___y_2878_; lean_object* v___y_2879_; lean_object* v___y_2880_; lean_object* v___y_2923_; lean_object* v___y_2924_; lean_object* v___y_2925_; lean_object* v___y_2926_; lean_object* v___y_2939_; lean_object* v___y_2940_; lean_object* v___y_2941_; lean_object* v___y_2942_; lean_object* v_options_2954_; uint8_t v_hasTrace_2955_; 
v_options_2954_ = lean_ctor_get(v_a_2710_, 2);
v_hasTrace_2955_ = lean_ctor_get_uint8(v_options_2954_, sizeof(void*)*1);
if (v_hasTrace_2955_ == 0)
{
v___y_2939_ = v_a_2708_;
v___y_2940_ = v_a_2709_;
v___y_2941_ = v_a_2710_;
v___y_2942_ = v_a_2711_;
goto v___jp_2938_;
}
else
{
lean_object* v_inheritedTraceOptions_2956_; lean_object* v_cls_2957_; lean_object* v___x_2958_; uint8_t v___x_2959_; 
v_inheritedTraceOptions_2956_ = lean_ctor_get(v_a_2710_, 13);
v_cls_2957_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_2958_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23);
v___x_2959_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2956_, v_options_2954_, v___x_2958_);
if (v___x_2959_ == 0)
{
v___y_2939_ = v_a_2708_;
v___y_2940_ = v_a_2709_;
v___y_2941_ = v_a_2710_;
v___y_2942_ = v_a_2711_;
goto v___jp_2938_;
}
else
{
lean_object* v___x_2960_; lean_object* v___x_2961_; 
v___x_2960_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__25, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__25_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__25);
v___x_2961_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0(v_cls_2957_, v___x_2960_, v_a_2708_, v_a_2709_, v_a_2710_, v_a_2711_);
if (lean_obj_tag(v___x_2961_) == 0)
{
lean_dec_ref_known(v___x_2961_, 1);
v___y_2939_ = v_a_2708_;
v___y_2940_ = v_a_2709_;
v___y_2941_ = v_a_2710_;
v___y_2942_ = v_a_2711_;
goto v___jp_2938_;
}
else
{
lean_object* v_a_2962_; lean_object* v___x_2964_; uint8_t v_isShared_2965_; uint8_t v_isSharedCheck_2969_; 
lean_del_object(v___x_2745_);
lean_dec(v_snd_2743_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_dec(v_fst_2738_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_dec(v_fst_2732_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
lean_dec_ref(v_rhs_2707_);
lean_dec_ref(v_lhs_2706_);
v_a_2962_ = lean_ctor_get(v___x_2961_, 0);
v_isSharedCheck_2969_ = !lean_is_exclusive(v___x_2961_);
if (v_isSharedCheck_2969_ == 0)
{
v___x_2964_ = v___x_2961_;
v_isShared_2965_ = v_isSharedCheck_2969_;
goto v_resetjp_2963_;
}
else
{
lean_inc(v_a_2962_);
lean_dec(v___x_2961_);
v___x_2964_ = lean_box(0);
v_isShared_2965_ = v_isSharedCheck_2969_;
goto v_resetjp_2963_;
}
v_resetjp_2963_:
{
lean_object* v___x_2967_; 
if (v_isShared_2965_ == 0)
{
v___x_2967_ = v___x_2964_;
goto v_reusejp_2966_;
}
else
{
lean_object* v_reuseFailAlloc_2968_; 
v_reuseFailAlloc_2968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2968_, 0, v_a_2962_);
v___x_2967_ = v_reuseFailAlloc_2968_;
goto v_reusejp_2966_;
}
v_reusejp_2966_:
{
return v___x_2967_;
}
}
}
}
}
v___jp_2747_:
{
lean_object* v___x_2748_; lean_object* v___x_2750_; 
v___x_2748_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v_fst_2733_, v_fst_2742_, v_snd_2734_);
if (v_isShared_2730_ == 0)
{
lean_ctor_set(v___x_2729_, 0, v___x_2748_);
v___x_2750_ = v___x_2729_;
goto v_reusejp_2749_;
}
else
{
lean_object* v_reuseFailAlloc_2754_; 
v_reuseFailAlloc_2754_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2754_, 0, v___x_2748_);
v___x_2750_ = v_reuseFailAlloc_2754_;
goto v_reusejp_2749_;
}
v_reusejp_2749_:
{
lean_object* v___x_2752_; 
if (v_isShared_2723_ == 0)
{
lean_ctor_set_tag(v___x_2722_, 0);
lean_ctor_set(v___x_2722_, 0, v___x_2750_);
v___x_2752_ = v___x_2722_;
goto v_reusejp_2751_;
}
else
{
lean_object* v_reuseFailAlloc_2753_; 
v_reuseFailAlloc_2753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2753_, 0, v___x_2750_);
v___x_2752_ = v_reuseFailAlloc_2753_;
goto v_reusejp_2751_;
}
v_reusejp_2751_:
{
return v___x_2752_;
}
}
}
v___jp_2755_:
{
lean_object* v___x_2761_; 
lean_inc_ref(v___y_2756_);
lean_inc(v_fst_2742_);
v___x_2761_ = l_Lean_Meta_isExprDefEq(v_fst_2742_, v___y_2756_, v___y_2757_, v___y_2758_, v___y_2759_, v___y_2760_);
if (lean_obj_tag(v___x_2761_) == 0)
{
lean_object* v_a_2762_; uint8_t v___x_2763_; 
v_a_2762_ = lean_ctor_get(v___x_2761_, 0);
lean_inc(v_a_2762_);
lean_dec_ref_known(v___x_2761_, 1);
v___x_2763_ = lean_unbox(v_a_2762_);
lean_dec(v_a_2762_);
if (v___x_2763_ == 0)
{
lean_object* v___x_2764_; lean_object* v___x_2765_; lean_object* v___x_2766_; lean_object* v___x_2768_; 
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v___x_2764_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__7, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__7);
v___x_2765_ = l_Lean_MessageData_ofExpr(v___y_2756_);
v___x_2766_ = l_Lean_indentD(v___x_2765_);
if (v_isShared_2746_ == 0)
{
lean_ctor_set_tag(v___x_2745_, 7);
lean_ctor_set(v___x_2745_, 1, v___x_2766_);
lean_ctor_set(v___x_2745_, 0, v___x_2764_);
v___x_2768_ = v___x_2745_;
goto v_reusejp_2767_;
}
else
{
lean_object* v_reuseFailAlloc_2787_; 
v_reuseFailAlloc_2787_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2787_, 0, v___x_2764_);
lean_ctor_set(v_reuseFailAlloc_2787_, 1, v___x_2766_);
v___x_2768_ = v_reuseFailAlloc_2787_;
goto v_reusejp_2767_;
}
v_reusejp_2767_:
{
lean_object* v___x_2769_; lean_object* v___x_2771_; 
v___x_2769_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9);
if (v_isShared_2741_ == 0)
{
lean_ctor_set_tag(v___x_2740_, 7);
lean_ctor_set(v___x_2740_, 1, v___x_2769_);
lean_ctor_set(v___x_2740_, 0, v___x_2768_);
v___x_2771_ = v___x_2740_;
goto v_reusejp_2770_;
}
else
{
lean_object* v_reuseFailAlloc_2786_; 
v_reuseFailAlloc_2786_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2786_, 0, v___x_2768_);
lean_ctor_set(v_reuseFailAlloc_2786_, 1, v___x_2769_);
v___x_2771_ = v_reuseFailAlloc_2786_;
goto v_reusejp_2770_;
}
v_reusejp_2770_:
{
lean_object* v___x_2772_; lean_object* v___x_2773_; lean_object* v___x_2775_; 
v___x_2772_ = l_Lean_MessageData_ofExpr(v_fst_2742_);
v___x_2773_ = l_Lean_indentD(v___x_2772_);
if (v_isShared_2737_ == 0)
{
lean_ctor_set_tag(v___x_2736_, 7);
lean_ctor_set(v___x_2736_, 1, v___x_2773_);
lean_ctor_set(v___x_2736_, 0, v___x_2771_);
v___x_2775_ = v___x_2736_;
goto v_reusejp_2774_;
}
else
{
lean_object* v_reuseFailAlloc_2785_; 
v_reuseFailAlloc_2785_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2785_, 0, v___x_2771_);
lean_ctor_set(v_reuseFailAlloc_2785_, 1, v___x_2773_);
v___x_2775_ = v_reuseFailAlloc_2785_;
goto v_reusejp_2774_;
}
v_reusejp_2774_:
{
lean_object* v___x_2776_; lean_object* v_a_2777_; lean_object* v___x_2779_; uint8_t v_isShared_2780_; uint8_t v_isSharedCheck_2784_; 
v___x_2776_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2775_, v___y_2757_, v___y_2758_, v___y_2759_, v___y_2760_);
v_a_2777_ = lean_ctor_get(v___x_2776_, 0);
v_isSharedCheck_2784_ = !lean_is_exclusive(v___x_2776_);
if (v_isSharedCheck_2784_ == 0)
{
v___x_2779_ = v___x_2776_;
v_isShared_2780_ = v_isSharedCheck_2784_;
goto v_resetjp_2778_;
}
else
{
lean_inc(v_a_2777_);
lean_dec(v___x_2776_);
v___x_2779_ = lean_box(0);
v_isShared_2780_ = v_isSharedCheck_2784_;
goto v_resetjp_2778_;
}
v_resetjp_2778_:
{
lean_object* v___x_2782_; 
if (v_isShared_2780_ == 0)
{
v___x_2782_ = v___x_2779_;
goto v_reusejp_2781_;
}
else
{
lean_object* v_reuseFailAlloc_2783_; 
v_reuseFailAlloc_2783_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2783_, 0, v_a_2777_);
v___x_2782_ = v_reuseFailAlloc_2783_;
goto v_reusejp_2781_;
}
v_reusejp_2781_:
{
return v___x_2782_;
}
}
}
}
}
}
else
{
lean_dec_ref(v___y_2756_);
lean_del_object(v___x_2745_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
goto v___jp_2747_;
}
}
else
{
lean_object* v_a_2788_; lean_object* v___x_2790_; uint8_t v_isShared_2791_; uint8_t v_isSharedCheck_2795_; 
lean_dec_ref(v___y_2756_);
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v_a_2788_ = lean_ctor_get(v___x_2761_, 0);
v_isSharedCheck_2795_ = !lean_is_exclusive(v___x_2761_);
if (v_isSharedCheck_2795_ == 0)
{
v___x_2790_ = v___x_2761_;
v_isShared_2791_ = v_isSharedCheck_2795_;
goto v_resetjp_2789_;
}
else
{
lean_inc(v_a_2788_);
lean_dec(v___x_2761_);
v___x_2790_ = lean_box(0);
v_isShared_2791_ = v_isSharedCheck_2795_;
goto v_resetjp_2789_;
}
v_resetjp_2789_:
{
lean_object* v___x_2793_; 
if (v_isShared_2791_ == 0)
{
v___x_2793_ = v___x_2790_;
goto v_reusejp_2792_;
}
else
{
lean_object* v_reuseFailAlloc_2794_; 
v_reuseFailAlloc_2794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2794_, 0, v_a_2788_);
v___x_2793_ = v_reuseFailAlloc_2794_;
goto v_reusejp_2792_;
}
v_reusejp_2792_:
{
return v___x_2793_;
}
}
}
}
v___jp_2796_:
{
lean_object* v___x_2801_; 
lean_inc(v___y_2800_);
lean_inc_ref(v___y_2799_);
lean_inc(v___y_2798_);
lean_inc_ref(v___y_2797_);
lean_inc(v_snd_2734_);
v___x_2801_ = lean_infer_type(v_snd_2734_, v___y_2797_, v___y_2798_, v___y_2799_, v___y_2800_);
if (lean_obj_tag(v___x_2801_) == 0)
{
lean_object* v_a_2802_; lean_object* v___x_2803_; 
v_a_2802_ = lean_ctor_get(v___x_2801_, 0);
lean_inc(v_a_2802_);
lean_dec_ref_known(v___x_2801_, 1);
lean_inc(v___y_2800_);
lean_inc_ref(v___y_2799_);
lean_inc(v___y_2798_);
lean_inc_ref(v___y_2797_);
v___x_2803_ = lean_whnf(v_a_2802_, v___y_2797_, v___y_2798_, v___y_2799_, v___y_2800_);
if (lean_obj_tag(v___x_2803_) == 0)
{
lean_object* v_a_2804_; lean_object* v___x_2805_; 
v_a_2804_ = lean_ctor_get(v___x_2803_, 0);
lean_inc(v_a_2804_);
lean_dec_ref_known(v___x_2803_, 1);
v___x_2805_ = lp_mathlib_Lean_Expr_sides_x3f(v_a_2804_);
lean_dec(v_a_2804_);
if (lean_obj_tag(v___x_2805_) == 1)
{
lean_object* v_val_2806_; lean_object* v_snd_2807_; lean_object* v___x_2809_; uint8_t v_isShared_2810_; uint8_t v_isSharedCheck_2858_; 
v_val_2806_ = lean_ctor_get(v___x_2805_, 0);
lean_inc(v_val_2806_);
lean_dec_ref_known(v___x_2805_, 1);
v_snd_2807_ = lean_ctor_get(v_val_2806_, 1);
v_isSharedCheck_2858_ = !lean_is_exclusive(v_val_2806_);
if (v_isSharedCheck_2858_ == 0)
{
lean_object* v_unused_2859_; 
v_unused_2859_ = lean_ctor_get(v_val_2806_, 0);
lean_dec(v_unused_2859_);
v___x_2809_ = v_val_2806_;
v_isShared_2810_ = v_isSharedCheck_2858_;
goto v_resetjp_2808_;
}
else
{
lean_inc(v_snd_2807_);
lean_dec(v_val_2806_);
v___x_2809_ = lean_box(0);
v_isShared_2810_ = v_isSharedCheck_2858_;
goto v_resetjp_2808_;
}
v_resetjp_2808_:
{
lean_object* v_snd_2811_; lean_object* v_fst_2812_; lean_object* v___x_2814_; uint8_t v_isShared_2815_; uint8_t v_isSharedCheck_2857_; 
v_snd_2811_ = lean_ctor_get(v_snd_2807_, 1);
v_fst_2812_ = lean_ctor_get(v_snd_2807_, 0);
v_isSharedCheck_2857_ = !lean_is_exclusive(v_snd_2807_);
if (v_isSharedCheck_2857_ == 0)
{
v___x_2814_ = v_snd_2807_;
v_isShared_2815_ = v_isSharedCheck_2857_;
goto v_resetjp_2813_;
}
else
{
lean_inc(v_snd_2811_);
lean_inc(v_fst_2812_);
lean_dec(v_snd_2807_);
v___x_2814_ = lean_box(0);
v_isShared_2815_ = v_isSharedCheck_2857_;
goto v_resetjp_2813_;
}
v_resetjp_2813_:
{
lean_object* v_snd_2816_; lean_object* v___x_2818_; uint8_t v_isShared_2819_; uint8_t v_isSharedCheck_2855_; 
v_snd_2816_ = lean_ctor_get(v_snd_2811_, 1);
v_isSharedCheck_2855_ = !lean_is_exclusive(v_snd_2811_);
if (v_isSharedCheck_2855_ == 0)
{
lean_object* v_unused_2856_; 
v_unused_2856_ = lean_ctor_get(v_snd_2811_, 0);
lean_dec(v_unused_2856_);
v___x_2818_ = v_snd_2811_;
v_isShared_2819_ = v_isSharedCheck_2855_;
goto v_resetjp_2817_;
}
else
{
lean_inc(v_snd_2816_);
lean_dec(v_snd_2811_);
v___x_2818_ = lean_box(0);
v_isShared_2819_ = v_isSharedCheck_2855_;
goto v_resetjp_2817_;
}
v_resetjp_2817_:
{
lean_object* v___x_2820_; 
lean_inc(v_fst_2812_);
lean_inc(v_fst_2733_);
v___x_2820_ = l_Lean_Meta_isExprDefEq(v_fst_2733_, v_fst_2812_, v___y_2797_, v___y_2798_, v___y_2799_, v___y_2800_);
if (lean_obj_tag(v___x_2820_) == 0)
{
lean_object* v_a_2821_; uint8_t v___x_2822_; 
v_a_2821_ = lean_ctor_get(v___x_2820_, 0);
lean_inc(v_a_2821_);
lean_dec_ref_known(v___x_2820_, 1);
v___x_2822_ = lean_unbox(v_a_2821_);
lean_dec(v_a_2821_);
if (v___x_2822_ == 0)
{
lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___x_2825_; lean_object* v___x_2827_; 
lean_dec(v_snd_2816_);
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v___x_2823_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__11, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__11);
v___x_2824_ = l_Lean_MessageData_ofExpr(v_fst_2812_);
v___x_2825_ = l_Lean_indentD(v___x_2824_);
if (v_isShared_2819_ == 0)
{
lean_ctor_set_tag(v___x_2818_, 7);
lean_ctor_set(v___x_2818_, 1, v___x_2825_);
lean_ctor_set(v___x_2818_, 0, v___x_2823_);
v___x_2827_ = v___x_2818_;
goto v_reusejp_2826_;
}
else
{
lean_object* v_reuseFailAlloc_2846_; 
v_reuseFailAlloc_2846_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2846_, 0, v___x_2823_);
lean_ctor_set(v_reuseFailAlloc_2846_, 1, v___x_2825_);
v___x_2827_ = v_reuseFailAlloc_2846_;
goto v_reusejp_2826_;
}
v_reusejp_2826_:
{
lean_object* v___x_2828_; lean_object* v___x_2830_; 
v___x_2828_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__9);
if (v_isShared_2815_ == 0)
{
lean_ctor_set_tag(v___x_2814_, 7);
lean_ctor_set(v___x_2814_, 1, v___x_2828_);
lean_ctor_set(v___x_2814_, 0, v___x_2827_);
v___x_2830_ = v___x_2814_;
goto v_reusejp_2829_;
}
else
{
lean_object* v_reuseFailAlloc_2845_; 
v_reuseFailAlloc_2845_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2845_, 0, v___x_2827_);
lean_ctor_set(v_reuseFailAlloc_2845_, 1, v___x_2828_);
v___x_2830_ = v_reuseFailAlloc_2845_;
goto v_reusejp_2829_;
}
v_reusejp_2829_:
{
lean_object* v___x_2831_; lean_object* v___x_2832_; lean_object* v___x_2834_; 
v___x_2831_ = l_Lean_MessageData_ofExpr(v_fst_2733_);
v___x_2832_ = l_Lean_indentD(v___x_2831_);
if (v_isShared_2810_ == 0)
{
lean_ctor_set_tag(v___x_2809_, 7);
lean_ctor_set(v___x_2809_, 1, v___x_2832_);
lean_ctor_set(v___x_2809_, 0, v___x_2830_);
v___x_2834_ = v___x_2809_;
goto v_reusejp_2833_;
}
else
{
lean_object* v_reuseFailAlloc_2844_; 
v_reuseFailAlloc_2844_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2844_, 0, v___x_2830_);
lean_ctor_set(v_reuseFailAlloc_2844_, 1, v___x_2832_);
v___x_2834_ = v_reuseFailAlloc_2844_;
goto v_reusejp_2833_;
}
v_reusejp_2833_:
{
lean_object* v___x_2835_; lean_object* v_a_2836_; lean_object* v___x_2838_; uint8_t v_isShared_2839_; uint8_t v_isSharedCheck_2843_; 
v___x_2835_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkEqForExpectedType_spec__0___redArg(v___x_2834_, v___y_2797_, v___y_2798_, v___y_2799_, v___y_2800_);
v_a_2836_ = lean_ctor_get(v___x_2835_, 0);
v_isSharedCheck_2843_ = !lean_is_exclusive(v___x_2835_);
if (v_isSharedCheck_2843_ == 0)
{
v___x_2838_ = v___x_2835_;
v_isShared_2839_ = v_isSharedCheck_2843_;
goto v_resetjp_2837_;
}
else
{
lean_inc(v_a_2836_);
lean_dec(v___x_2835_);
v___x_2838_ = lean_box(0);
v_isShared_2839_ = v_isSharedCheck_2843_;
goto v_resetjp_2837_;
}
v_resetjp_2837_:
{
lean_object* v___x_2841_; 
if (v_isShared_2839_ == 0)
{
v___x_2841_ = v___x_2838_;
goto v_reusejp_2840_;
}
else
{
lean_object* v_reuseFailAlloc_2842_; 
v_reuseFailAlloc_2842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2842_, 0, v_a_2836_);
v___x_2841_ = v_reuseFailAlloc_2842_;
goto v_reusejp_2840_;
}
v_reusejp_2840_:
{
return v___x_2841_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_2818_);
lean_del_object(v___x_2814_);
lean_dec(v_fst_2812_);
lean_del_object(v___x_2809_);
v___y_2756_ = v_snd_2816_;
v___y_2757_ = v___y_2797_;
v___y_2758_ = v___y_2798_;
v___y_2759_ = v___y_2799_;
v___y_2760_ = v___y_2800_;
goto v___jp_2755_;
}
}
else
{
lean_object* v_a_2847_; lean_object* v___x_2849_; uint8_t v_isShared_2850_; uint8_t v_isSharedCheck_2854_; 
lean_del_object(v___x_2818_);
lean_dec(v_snd_2816_);
lean_del_object(v___x_2814_);
lean_dec(v_fst_2812_);
lean_del_object(v___x_2809_);
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v_a_2847_ = lean_ctor_get(v___x_2820_, 0);
v_isSharedCheck_2854_ = !lean_is_exclusive(v___x_2820_);
if (v_isSharedCheck_2854_ == 0)
{
v___x_2849_ = v___x_2820_;
v_isShared_2850_ = v_isSharedCheck_2854_;
goto v_resetjp_2848_;
}
else
{
lean_inc(v_a_2847_);
lean_dec(v___x_2820_);
v___x_2849_ = lean_box(0);
v_isShared_2850_ = v_isSharedCheck_2854_;
goto v_resetjp_2848_;
}
v_resetjp_2848_:
{
lean_object* v___x_2852_; 
if (v_isShared_2850_ == 0)
{
v___x_2852_ = v___x_2849_;
goto v_reusejp_2851_;
}
else
{
lean_object* v_reuseFailAlloc_2853_; 
v_reuseFailAlloc_2853_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2853_, 0, v_a_2847_);
v___x_2852_ = v_reuseFailAlloc_2853_;
goto v_reusejp_2851_;
}
v_reusejp_2851_:
{
return v___x_2852_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_2805_);
lean_del_object(v___x_2745_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
goto v___jp_2747_;
}
}
else
{
lean_object* v_a_2860_; lean_object* v___x_2862_; uint8_t v_isShared_2863_; uint8_t v_isSharedCheck_2867_; 
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v_a_2860_ = lean_ctor_get(v___x_2803_, 0);
v_isSharedCheck_2867_ = !lean_is_exclusive(v___x_2803_);
if (v_isSharedCheck_2867_ == 0)
{
v___x_2862_ = v___x_2803_;
v_isShared_2863_ = v_isSharedCheck_2867_;
goto v_resetjp_2861_;
}
else
{
lean_inc(v_a_2860_);
lean_dec(v___x_2803_);
v___x_2862_ = lean_box(0);
v_isShared_2863_ = v_isSharedCheck_2867_;
goto v_resetjp_2861_;
}
v_resetjp_2861_:
{
lean_object* v___x_2865_; 
if (v_isShared_2863_ == 0)
{
v___x_2865_ = v___x_2862_;
goto v_reusejp_2864_;
}
else
{
lean_object* v_reuseFailAlloc_2866_; 
v_reuseFailAlloc_2866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2866_, 0, v_a_2860_);
v___x_2865_ = v_reuseFailAlloc_2866_;
goto v_reusejp_2864_;
}
v_reusejp_2864_:
{
return v___x_2865_;
}
}
}
}
else
{
lean_object* v_a_2868_; lean_object* v___x_2870_; uint8_t v_isShared_2871_; uint8_t v_isSharedCheck_2875_; 
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v_a_2868_ = lean_ctor_get(v___x_2801_, 0);
v_isSharedCheck_2875_ = !lean_is_exclusive(v___x_2801_);
if (v_isSharedCheck_2875_ == 0)
{
v___x_2870_ = v___x_2801_;
v_isShared_2871_ = v_isSharedCheck_2875_;
goto v_resetjp_2869_;
}
else
{
lean_inc(v_a_2868_);
lean_dec(v___x_2801_);
v___x_2870_ = lean_box(0);
v_isShared_2871_ = v_isSharedCheck_2875_;
goto v_resetjp_2869_;
}
v_resetjp_2869_:
{
lean_object* v___x_2873_; 
if (v_isShared_2871_ == 0)
{
v___x_2873_ = v___x_2870_;
goto v_reusejp_2872_;
}
else
{
lean_object* v_reuseFailAlloc_2874_; 
v_reuseFailAlloc_2874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2874_, 0, v_a_2868_);
v___x_2873_ = v_reuseFailAlloc_2874_;
goto v_reusejp_2872_;
}
v_reusejp_2872_:
{
return v___x_2873_;
}
}
}
}
v___jp_2876_:
{
lean_object* v___x_2881_; 
lean_inc(v___y_2880_);
lean_inc_ref(v___y_2879_);
lean_inc(v___y_2878_);
lean_inc_ref(v___y_2877_);
lean_inc(v_snd_2734_);
v___x_2881_ = lean_infer_type(v_snd_2734_, v___y_2877_, v___y_2878_, v___y_2879_, v___y_2880_);
if (lean_obj_tag(v___x_2881_) == 0)
{
lean_object* v_a_2882_; lean_object* v___x_2883_; 
v_a_2882_ = lean_ctor_get(v___x_2881_, 0);
lean_inc(v_a_2882_);
lean_dec_ref_known(v___x_2881_, 1);
lean_inc(v___y_2880_);
lean_inc_ref(v___y_2879_);
lean_inc(v___y_2878_);
lean_inc_ref(v___y_2877_);
v___x_2883_ = lean_infer_type(v_snd_2743_, v___y_2877_, v___y_2878_, v___y_2879_, v___y_2880_);
if (lean_obj_tag(v___x_2883_) == 0)
{
lean_object* v_a_2884_; lean_object* v___x_2885_; 
v_a_2884_ = lean_ctor_get(v___x_2883_, 0);
lean_inc(v_a_2884_);
lean_dec_ref_known(v___x_2883_, 1);
v___x_2885_ = l_Lean_Meta_isExprDefEq(v_a_2882_, v_a_2884_, v___y_2877_, v___y_2878_, v___y_2879_, v___y_2880_);
if (lean_obj_tag(v___x_2885_) == 0)
{
lean_object* v_a_2886_; uint8_t v___x_2887_; 
v_a_2886_ = lean_ctor_get(v___x_2885_, 0);
lean_inc(v_a_2886_);
lean_dec_ref_known(v___x_2885_, 1);
v___x_2887_ = lean_unbox(v_a_2886_);
lean_dec(v_a_2886_);
if (v___x_2887_ == 0)
{
lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v_a_2890_; lean_object* v___x_2892_; uint8_t v_isShared_2893_; uint8_t v_isSharedCheck_2897_; 
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v___x_2888_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__14, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__14);
v___x_2889_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_lhs_2706_, v_rhs_2707_, v___x_2888_, v___y_2877_, v___y_2878_, v___y_2879_, v___y_2880_);
v_a_2890_ = lean_ctor_get(v___x_2889_, 0);
v_isSharedCheck_2897_ = !lean_is_exclusive(v___x_2889_);
if (v_isSharedCheck_2897_ == 0)
{
v___x_2892_ = v___x_2889_;
v_isShared_2893_ = v_isSharedCheck_2897_;
goto v_resetjp_2891_;
}
else
{
lean_inc(v_a_2890_);
lean_dec(v___x_2889_);
v___x_2892_ = lean_box(0);
v_isShared_2893_ = v_isSharedCheck_2897_;
goto v_resetjp_2891_;
}
v_resetjp_2891_:
{
lean_object* v___x_2895_; 
if (v_isShared_2893_ == 0)
{
v___x_2895_ = v___x_2892_;
goto v_reusejp_2894_;
}
else
{
lean_object* v_reuseFailAlloc_2896_; 
v_reuseFailAlloc_2896_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2896_, 0, v_a_2890_);
v___x_2895_ = v_reuseFailAlloc_2896_;
goto v_reusejp_2894_;
}
v_reusejp_2894_:
{
return v___x_2895_;
}
}
}
else
{
lean_dec_ref(v_rhs_2707_);
lean_dec_ref(v_lhs_2706_);
v___y_2797_ = v___y_2877_;
v___y_2798_ = v___y_2878_;
v___y_2799_ = v___y_2879_;
v___y_2800_ = v___y_2880_;
goto v___jp_2796_;
}
}
else
{
lean_object* v_a_2898_; lean_object* v___x_2900_; uint8_t v_isShared_2901_; uint8_t v_isSharedCheck_2905_; 
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
lean_dec_ref(v_rhs_2707_);
lean_dec_ref(v_lhs_2706_);
v_a_2898_ = lean_ctor_get(v___x_2885_, 0);
v_isSharedCheck_2905_ = !lean_is_exclusive(v___x_2885_);
if (v_isSharedCheck_2905_ == 0)
{
v___x_2900_ = v___x_2885_;
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
else
{
lean_inc(v_a_2898_);
lean_dec(v___x_2885_);
v___x_2900_ = lean_box(0);
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
v_resetjp_2899_:
{
lean_object* v___x_2903_; 
if (v_isShared_2901_ == 0)
{
v___x_2903_ = v___x_2900_;
goto v_reusejp_2902_;
}
else
{
lean_object* v_reuseFailAlloc_2904_; 
v_reuseFailAlloc_2904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2904_, 0, v_a_2898_);
v___x_2903_ = v_reuseFailAlloc_2904_;
goto v_reusejp_2902_;
}
v_reusejp_2902_:
{
return v___x_2903_;
}
}
}
}
else
{
lean_object* v_a_2906_; lean_object* v___x_2908_; uint8_t v_isShared_2909_; uint8_t v_isSharedCheck_2913_; 
lean_dec(v_a_2882_);
lean_del_object(v___x_2745_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
lean_dec_ref(v_rhs_2707_);
lean_dec_ref(v_lhs_2706_);
v_a_2906_ = lean_ctor_get(v___x_2883_, 0);
v_isSharedCheck_2913_ = !lean_is_exclusive(v___x_2883_);
if (v_isSharedCheck_2913_ == 0)
{
v___x_2908_ = v___x_2883_;
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
else
{
lean_inc(v_a_2906_);
lean_dec(v___x_2883_);
v___x_2908_ = lean_box(0);
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
v_resetjp_2907_:
{
lean_object* v___x_2911_; 
if (v_isShared_2909_ == 0)
{
v___x_2911_ = v___x_2908_;
goto v_reusejp_2910_;
}
else
{
lean_object* v_reuseFailAlloc_2912_; 
v_reuseFailAlloc_2912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2912_, 0, v_a_2906_);
v___x_2911_ = v_reuseFailAlloc_2912_;
goto v_reusejp_2910_;
}
v_reusejp_2910_:
{
return v___x_2911_;
}
}
}
}
else
{
lean_object* v_a_2914_; lean_object* v___x_2916_; uint8_t v_isShared_2917_; uint8_t v_isSharedCheck_2921_; 
lean_del_object(v___x_2745_);
lean_dec(v_snd_2743_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
lean_dec_ref(v_rhs_2707_);
lean_dec_ref(v_lhs_2706_);
v_a_2914_ = lean_ctor_get(v___x_2881_, 0);
v_isSharedCheck_2921_ = !lean_is_exclusive(v___x_2881_);
if (v_isSharedCheck_2921_ == 0)
{
v___x_2916_ = v___x_2881_;
v_isShared_2917_ = v_isSharedCheck_2921_;
goto v_resetjp_2915_;
}
else
{
lean_inc(v_a_2914_);
lean_dec(v___x_2881_);
v___x_2916_ = lean_box(0);
v_isShared_2917_ = v_isSharedCheck_2921_;
goto v_resetjp_2915_;
}
v_resetjp_2915_:
{
lean_object* v___x_2919_; 
if (v_isShared_2917_ == 0)
{
v___x_2919_ = v___x_2916_;
goto v_reusejp_2918_;
}
else
{
lean_object* v_reuseFailAlloc_2920_; 
v_reuseFailAlloc_2920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2920_, 0, v_a_2914_);
v___x_2919_ = v_reuseFailAlloc_2920_;
goto v_reusejp_2918_;
}
v_reusejp_2918_:
{
return v___x_2919_;
}
}
}
}
v___jp_2922_:
{
uint8_t v___x_2927_; 
v___x_2927_ = lean_unbox(v_fst_2738_);
lean_dec(v_fst_2738_);
if (v___x_2927_ == 0)
{
v___y_2877_ = v___y_2923_;
v___y_2878_ = v___y_2924_;
v___y_2879_ = v___y_2925_;
v___y_2880_ = v___y_2926_;
goto v___jp_2876_;
}
else
{
lean_object* v___x_2928_; lean_object* v___x_2929_; lean_object* v_a_2930_; lean_object* v___x_2932_; uint8_t v_isShared_2933_; uint8_t v_isSharedCheck_2937_; 
lean_del_object(v___x_2745_);
lean_dec(v_snd_2743_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v___x_2928_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__17, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__17);
v___x_2929_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_lhs_2706_, v_rhs_2707_, v___x_2928_, v___y_2923_, v___y_2924_, v___y_2925_, v___y_2926_);
v_a_2930_ = lean_ctor_get(v___x_2929_, 0);
v_isSharedCheck_2937_ = !lean_is_exclusive(v___x_2929_);
if (v_isSharedCheck_2937_ == 0)
{
v___x_2932_ = v___x_2929_;
v_isShared_2933_ = v_isSharedCheck_2937_;
goto v_resetjp_2931_;
}
else
{
lean_inc(v_a_2930_);
lean_dec(v___x_2929_);
v___x_2932_ = lean_box(0);
v_isShared_2933_ = v_isSharedCheck_2937_;
goto v_resetjp_2931_;
}
v_resetjp_2931_:
{
lean_object* v___x_2935_; 
if (v_isShared_2933_ == 0)
{
v___x_2935_ = v___x_2932_;
goto v_reusejp_2934_;
}
else
{
lean_object* v_reuseFailAlloc_2936_; 
v_reuseFailAlloc_2936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2936_, 0, v_a_2930_);
v___x_2935_ = v_reuseFailAlloc_2936_;
goto v_reusejp_2934_;
}
v_reusejp_2934_:
{
return v___x_2935_;
}
}
}
}
v___jp_2938_:
{
uint8_t v___x_2943_; 
v___x_2943_ = lean_unbox(v_fst_2732_);
lean_dec(v_fst_2732_);
if (v___x_2943_ == 0)
{
lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v_a_2946_; lean_object* v___x_2948_; uint8_t v_isShared_2949_; uint8_t v_isSharedCheck_2953_; 
lean_del_object(v___x_2745_);
lean_dec(v_snd_2743_);
lean_dec(v_fst_2742_);
lean_del_object(v___x_2740_);
lean_dec(v_fst_2738_);
lean_del_object(v___x_2736_);
lean_dec(v_snd_2734_);
lean_dec(v_fst_2733_);
lean_del_object(v___x_2729_);
lean_del_object(v___x_2722_);
v___x_2944_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__20, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__20);
v___x_2945_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_lhs_2706_, v_rhs_2707_, v___x_2944_, v___y_2939_, v___y_2940_, v___y_2941_, v___y_2942_);
v_a_2946_ = lean_ctor_get(v___x_2945_, 0);
v_isSharedCheck_2953_ = !lean_is_exclusive(v___x_2945_);
if (v_isSharedCheck_2953_ == 0)
{
v___x_2948_ = v___x_2945_;
v_isShared_2949_ = v_isSharedCheck_2953_;
goto v_resetjp_2947_;
}
else
{
lean_inc(v_a_2946_);
lean_dec(v___x_2945_);
v___x_2948_ = lean_box(0);
v_isShared_2949_ = v_isSharedCheck_2953_;
goto v_resetjp_2947_;
}
v_resetjp_2947_:
{
lean_object* v___x_2951_; 
if (v_isShared_2949_ == 0)
{
v___x_2951_ = v___x_2948_;
goto v_reusejp_2950_;
}
else
{
lean_object* v_reuseFailAlloc_2952_; 
v_reuseFailAlloc_2952_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2952_, 0, v_a_2946_);
v___x_2951_ = v_reuseFailAlloc_2952_;
goto v_reusejp_2950_;
}
v_reusejp_2950_:
{
return v___x_2951_;
}
}
}
else
{
v___y_2923_ = v___y_2939_;
v___y_2924_ = v___y_2940_;
v___y_2925_ = v___y_2941_;
v___y_2926_ = v___y_2942_;
goto v___jp_2922_;
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___boxed(lean_object* v_mvarCounterSaved_2976_, lean_object* v_lhs_2977_, lean_object* v_rhs_2978_, lean_object* v_a_2979_, lean_object* v_a_2980_, lean_object* v_a_2981_, lean_object* v_a_2982_, lean_object* v_a_2983_){
_start:
{
lean_object* v_res_2984_; 
v_res_2984_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f(v_mvarCounterSaved_2976_, v_lhs_2977_, v_rhs_2978_, v_a_2979_, v_a_2980_, v_a_2981_, v_a_2982_);
lean_dec(v_a_2982_);
lean_dec_ref(v_a_2981_);
lean_dec(v_a_2980_);
lean_dec_ref(v_a_2979_);
return v_res_2984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_getJointAppFns(lean_object* v_e_2985_, lean_object* v_e_x27_2986_){
_start:
{
uint8_t v___x_2987_; 
v___x_2987_ = lean_expr_eqv(v_e_2985_, v_e_x27_2986_);
if (v___x_2987_ == 0)
{
if (lean_obj_tag(v_e_2985_) == 5)
{
if (lean_obj_tag(v_e_x27_2986_) == 5)
{
lean_object* v_fn_2988_; lean_object* v_fn_2989_; 
v_fn_2988_ = lean_ctor_get(v_e_2985_, 0);
lean_inc_ref(v_fn_2988_);
lean_dec_ref_known(v_e_2985_, 2);
v_fn_2989_ = lean_ctor_get(v_e_x27_2986_, 0);
lean_inc_ref(v_fn_2989_);
lean_dec_ref_known(v_e_x27_2986_, 2);
v_e_2985_ = v_fn_2988_;
v_e_x27_2986_ = v_fn_2989_;
goto _start;
}
else
{
lean_object* v___x_2991_; 
v___x_2991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2991_, 0, v_e_2985_);
lean_ctor_set(v___x_2991_, 1, v_e_x27_2986_);
return v___x_2991_;
}
}
else
{
lean_object* v___x_2992_; 
v___x_2992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2992_, 0, v_e_2985_);
lean_ctor_set(v___x_2992_, 1, v_e_x27_2986_);
return v___x_2992_;
}
}
else
{
lean_object* v___x_2993_; 
lean_dec_ref(v_e_x27_2986_);
lean_inc_ref(v_e_2985_);
v___x_2993_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2993_, 0, v_e_2985_);
lean_ctor_set(v___x_2993_, 1, v_e_2985_);
return v___x_2993_;
}
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__2(void){
_start:
{
lean_object* v___x_2996_; 
v___x_2996_ = l_instMonadEIO(lean_box(0));
return v___x_2996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2(lean_object* v_msg_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_){
_start:
{
lean_object* v___x_3008_; lean_object* v___x_3009_; lean_object* v___f_3010_; lean_object* v___x_3011_; lean_object* v___f_3012_; lean_object* v___x_3013_; lean_object* v___x_3014_; lean_object* v_toApplicative_3015_; lean_object* v___x_3017_; uint8_t v_isShared_3018_; uint8_t v_isSharedCheck_3077_; 
v___x_3008_ = lean_box(0);
v___x_3009_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__0));
v___f_3010_ = lean_alloc_closure((void*)(l_instBEqProd___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_3010_, 0, v___x_3009_);
lean_closure_set(v___f_3010_, 1, v___x_3009_);
v___x_3011_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__1));
v___f_3012_ = lean_alloc_closure((void*)(l_instHashableProd___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3012_, 0, v___x_3011_);
lean_closure_set(v___f_3012_, 1, v___x_3011_);
v___x_3013_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__2, &lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__2_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__2);
v___x_3014_ = l_StateRefT_x27_instMonad___redArg(v___x_3013_);
v_toApplicative_3015_ = lean_ctor_get(v___x_3014_, 0);
v_isSharedCheck_3077_ = !lean_is_exclusive(v___x_3014_);
if (v_isSharedCheck_3077_ == 0)
{
lean_object* v_unused_3078_; 
v_unused_3078_ = lean_ctor_get(v___x_3014_, 1);
lean_dec(v_unused_3078_);
v___x_3017_ = v___x_3014_;
v_isShared_3018_ = v_isSharedCheck_3077_;
goto v_resetjp_3016_;
}
else
{
lean_inc(v_toApplicative_3015_);
lean_dec(v___x_3014_);
v___x_3017_ = lean_box(0);
v_isShared_3018_ = v_isSharedCheck_3077_;
goto v_resetjp_3016_;
}
v_resetjp_3016_:
{
lean_object* v_toFunctor_3019_; lean_object* v_toSeq_3020_; lean_object* v_toSeqLeft_3021_; lean_object* v_toSeqRight_3022_; lean_object* v___x_3024_; uint8_t v_isShared_3025_; uint8_t v_isSharedCheck_3075_; 
v_toFunctor_3019_ = lean_ctor_get(v_toApplicative_3015_, 0);
v_toSeq_3020_ = lean_ctor_get(v_toApplicative_3015_, 2);
v_toSeqLeft_3021_ = lean_ctor_get(v_toApplicative_3015_, 3);
v_toSeqRight_3022_ = lean_ctor_get(v_toApplicative_3015_, 4);
v_isSharedCheck_3075_ = !lean_is_exclusive(v_toApplicative_3015_);
if (v_isSharedCheck_3075_ == 0)
{
lean_object* v_unused_3076_; 
v_unused_3076_ = lean_ctor_get(v_toApplicative_3015_, 1);
lean_dec(v_unused_3076_);
v___x_3024_ = v_toApplicative_3015_;
v_isShared_3025_ = v_isSharedCheck_3075_;
goto v_resetjp_3023_;
}
else
{
lean_inc(v_toSeqRight_3022_);
lean_inc(v_toSeqLeft_3021_);
lean_inc(v_toSeq_3020_);
lean_inc(v_toFunctor_3019_);
lean_dec(v_toApplicative_3015_);
v___x_3024_ = lean_box(0);
v_isShared_3025_ = v_isSharedCheck_3075_;
goto v_resetjp_3023_;
}
v_resetjp_3023_:
{
lean_object* v___f_3026_; lean_object* v___f_3027_; lean_object* v___f_3028_; lean_object* v___f_3029_; lean_object* v___x_3030_; lean_object* v___f_3031_; lean_object* v___f_3032_; lean_object* v___f_3033_; lean_object* v___x_3035_; 
v___f_3026_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__3));
v___f_3027_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__4));
lean_inc_ref(v_toFunctor_3019_);
v___f_3028_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3028_, 0, v_toFunctor_3019_);
v___f_3029_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3029_, 0, v_toFunctor_3019_);
v___x_3030_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3030_, 0, v___f_3028_);
lean_ctor_set(v___x_3030_, 1, v___f_3029_);
v___f_3031_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3031_, 0, v_toSeqRight_3022_);
v___f_3032_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3032_, 0, v_toSeqLeft_3021_);
v___f_3033_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3033_, 0, v_toSeq_3020_);
if (v_isShared_3025_ == 0)
{
lean_ctor_set(v___x_3024_, 4, v___f_3031_);
lean_ctor_set(v___x_3024_, 3, v___f_3032_);
lean_ctor_set(v___x_3024_, 2, v___f_3033_);
lean_ctor_set(v___x_3024_, 1, v___f_3026_);
lean_ctor_set(v___x_3024_, 0, v___x_3030_);
v___x_3035_ = v___x_3024_;
goto v_reusejp_3034_;
}
else
{
lean_object* v_reuseFailAlloc_3074_; 
v_reuseFailAlloc_3074_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3074_, 0, v___x_3030_);
lean_ctor_set(v_reuseFailAlloc_3074_, 1, v___f_3026_);
lean_ctor_set(v_reuseFailAlloc_3074_, 2, v___f_3033_);
lean_ctor_set(v_reuseFailAlloc_3074_, 3, v___f_3032_);
lean_ctor_set(v_reuseFailAlloc_3074_, 4, v___f_3031_);
v___x_3035_ = v_reuseFailAlloc_3074_;
goto v_reusejp_3034_;
}
v_reusejp_3034_:
{
lean_object* v___x_3037_; 
if (v_isShared_3018_ == 0)
{
lean_ctor_set(v___x_3017_, 1, v___f_3027_);
lean_ctor_set(v___x_3017_, 0, v___x_3035_);
v___x_3037_ = v___x_3017_;
goto v_reusejp_3036_;
}
else
{
lean_object* v_reuseFailAlloc_3073_; 
v_reuseFailAlloc_3073_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3073_, 0, v___x_3035_);
lean_ctor_set(v_reuseFailAlloc_3073_, 1, v___f_3027_);
v___x_3037_ = v_reuseFailAlloc_3073_;
goto v_reusejp_3036_;
}
v_reusejp_3036_:
{
lean_object* v___x_3038_; lean_object* v_toApplicative_3039_; lean_object* v___x_3041_; uint8_t v_isShared_3042_; uint8_t v_isSharedCheck_3071_; 
v___x_3038_ = l_StateRefT_x27_instMonad___redArg(v___x_3037_);
v_toApplicative_3039_ = lean_ctor_get(v___x_3038_, 0);
v_isSharedCheck_3071_ = !lean_is_exclusive(v___x_3038_);
if (v_isSharedCheck_3071_ == 0)
{
lean_object* v_unused_3072_; 
v_unused_3072_ = lean_ctor_get(v___x_3038_, 1);
lean_dec(v_unused_3072_);
v___x_3041_ = v___x_3038_;
v_isShared_3042_ = v_isSharedCheck_3071_;
goto v_resetjp_3040_;
}
else
{
lean_inc(v_toApplicative_3039_);
lean_dec(v___x_3038_);
v___x_3041_ = lean_box(0);
v_isShared_3042_ = v_isSharedCheck_3071_;
goto v_resetjp_3040_;
}
v_resetjp_3040_:
{
lean_object* v_toFunctor_3043_; lean_object* v_toSeq_3044_; lean_object* v_toSeqLeft_3045_; lean_object* v_toSeqRight_3046_; lean_object* v___x_3048_; uint8_t v_isShared_3049_; uint8_t v_isSharedCheck_3069_; 
v_toFunctor_3043_ = lean_ctor_get(v_toApplicative_3039_, 0);
v_toSeq_3044_ = lean_ctor_get(v_toApplicative_3039_, 2);
v_toSeqLeft_3045_ = lean_ctor_get(v_toApplicative_3039_, 3);
v_toSeqRight_3046_ = lean_ctor_get(v_toApplicative_3039_, 4);
v_isSharedCheck_3069_ = !lean_is_exclusive(v_toApplicative_3039_);
if (v_isSharedCheck_3069_ == 0)
{
lean_object* v_unused_3070_; 
v_unused_3070_ = lean_ctor_get(v_toApplicative_3039_, 1);
lean_dec(v_unused_3070_);
v___x_3048_ = v_toApplicative_3039_;
v_isShared_3049_ = v_isSharedCheck_3069_;
goto v_resetjp_3047_;
}
else
{
lean_inc(v_toSeqRight_3046_);
lean_inc(v_toSeqLeft_3045_);
lean_inc(v_toSeq_3044_);
lean_inc(v_toFunctor_3043_);
lean_dec(v_toApplicative_3039_);
v___x_3048_ = lean_box(0);
v_isShared_3049_ = v_isSharedCheck_3069_;
goto v_resetjp_3047_;
}
v_resetjp_3047_:
{
lean_object* v___f_3050_; lean_object* v___f_3051_; lean_object* v___f_3052_; lean_object* v___f_3053_; lean_object* v___x_3054_; lean_object* v___f_3055_; lean_object* v___f_3056_; lean_object* v___f_3057_; lean_object* v___x_3059_; 
v___f_3050_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__5));
v___f_3051_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___closed__6));
lean_inc_ref(v_toFunctor_3043_);
v___f_3052_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3052_, 0, v_toFunctor_3043_);
v___f_3053_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3053_, 0, v_toFunctor_3043_);
v___x_3054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3054_, 0, v___f_3052_);
lean_ctor_set(v___x_3054_, 1, v___f_3053_);
v___f_3055_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3055_, 0, v_toSeqRight_3046_);
v___f_3056_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3056_, 0, v_toSeqLeft_3045_);
v___f_3057_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3057_, 0, v_toSeq_3044_);
if (v_isShared_3049_ == 0)
{
lean_ctor_set(v___x_3048_, 4, v___f_3055_);
lean_ctor_set(v___x_3048_, 3, v___f_3056_);
lean_ctor_set(v___x_3048_, 2, v___f_3057_);
lean_ctor_set(v___x_3048_, 1, v___f_3050_);
lean_ctor_set(v___x_3048_, 0, v___x_3054_);
v___x_3059_ = v___x_3048_;
goto v_reusejp_3058_;
}
else
{
lean_object* v_reuseFailAlloc_3068_; 
v_reuseFailAlloc_3068_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3068_, 0, v___x_3054_);
lean_ctor_set(v_reuseFailAlloc_3068_, 1, v___f_3050_);
lean_ctor_set(v_reuseFailAlloc_3068_, 2, v___f_3057_);
lean_ctor_set(v_reuseFailAlloc_3068_, 3, v___f_3056_);
lean_ctor_set(v_reuseFailAlloc_3068_, 4, v___f_3055_);
v___x_3059_ = v_reuseFailAlloc_3068_;
goto v_reusejp_3058_;
}
v_reusejp_3058_:
{
lean_object* v___x_3061_; 
if (v_isShared_3042_ == 0)
{
lean_ctor_set(v___x_3041_, 1, v___f_3051_);
lean_ctor_set(v___x_3041_, 0, v___x_3059_);
v___x_3061_ = v___x_3041_;
goto v_reusejp_3060_;
}
else
{
lean_object* v_reuseFailAlloc_3067_; 
v_reuseFailAlloc_3067_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3067_, 0, v___x_3059_);
lean_ctor_set(v_reuseFailAlloc_3067_, 1, v___f_3051_);
v___x_3061_ = v_reuseFailAlloc_3067_;
goto v_reusejp_3060_;
}
v_reusejp_3060_:
{
lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; lean_object* v___x_76828__overap_3065_; lean_object* v___x_3066_; 
v___x_3062_ = l_Lean_MonadCacheT_instMonad___redArg(v___x_3008_, v___f_3010_, v___f_3012_, v___x_3061_);
v___x_3063_ = lean_box(0);
v___x_3064_ = l_instInhabitedOfMonad___redArg(v___x_3062_, v___x_3063_);
v___x_76828__overap_3065_ = lean_panic_fn_borrowed(v___x_3064_, v_msg_3001_);
lean_dec(v___x_3064_);
lean_inc(v___y_3006_);
lean_inc_ref(v___y_3005_);
lean_inc(v___y_3004_);
lean_inc_ref(v___y_3003_);
lean_inc(v___y_3002_);
v___x_3066_ = lean_apply_6(v___x_76828__overap_3065_, v___y_3002_, v___y_3003_, v___y_3004_, v___y_3005_, v___y_3006_, lean_box(0));
return v___x_3066_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2___boxed(lean_object* v_msg_3079_, lean_object* v___y_3080_, lean_object* v___y_3081_, lean_object* v___y_3082_, lean_object* v___y_3083_, lean_object* v___y_3084_, lean_object* v___y_3085_){
_start:
{
lean_object* v_res_3086_; 
v_res_3086_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2(v_msg_3079_, v___y_3080_, v___y_3081_, v___y_3082_, v___y_3083_, v___y_3084_);
lean_dec(v___y_3084_);
lean_dec_ref(v___y_3083_);
lean_dec(v___y_3082_);
lean_dec_ref(v___y_3081_);
lean_dec(v___y_3080_);
return v_res_3086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___lam__0(lean_object* v_k_3087_, lean_object* v___y_3088_, lean_object* v_b_3089_, lean_object* v_c_3090_, lean_object* v___y_3091_, lean_object* v___y_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_){
_start:
{
lean_object* v___x_3096_; 
lean_inc(v___y_3094_);
lean_inc_ref(v___y_3093_);
lean_inc(v___y_3092_);
lean_inc_ref(v___y_3091_);
lean_inc(v___y_3088_);
v___x_3096_ = lean_apply_8(v_k_3087_, v_b_3089_, v_c_3090_, v___y_3088_, v___y_3091_, v___y_3092_, v___y_3093_, v___y_3094_, lean_box(0));
return v___x_3096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___lam__0___boxed(lean_object* v_k_3097_, lean_object* v___y_3098_, lean_object* v_b_3099_, lean_object* v_c_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_){
_start:
{
lean_object* v_res_3106_; 
v_res_3106_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___lam__0(v_k_3097_, v___y_3098_, v_b_3099_, v_c_3100_, v___y_3101_, v___y_3102_, v___y_3103_, v___y_3104_);
lean_dec(v___y_3104_);
lean_dec_ref(v___y_3103_);
lean_dec(v___y_3102_);
lean_dec_ref(v___y_3101_);
lean_dec(v___y_3098_);
return v_res_3106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg(lean_object* v_type_3107_, lean_object* v_maxFVars_x3f_3108_, lean_object* v_k_3109_, uint8_t v_cleanupAnnotations_3110_, uint8_t v_whnfType_3111_, lean_object* v___y_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_){
_start:
{
lean_object* v___f_3118_; lean_object* v___x_3119_; 
lean_inc(v___y_3112_);
v___f_3118_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___lam__0___boxed), 9, 2);
lean_closure_set(v___f_3118_, 0, v_k_3109_);
lean_closure_set(v___f_3118_, 1, v___y_3112_);
v___x_3119_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_3107_, v_maxFVars_x3f_3108_, v___f_3118_, v_cleanupAnnotations_3110_, v_whnfType_3111_, v___y_3113_, v___y_3114_, v___y_3115_, v___y_3116_);
if (lean_obj_tag(v___x_3119_) == 0)
{
return v___x_3119_;
}
else
{
lean_object* v_a_3120_; lean_object* v___x_3122_; uint8_t v_isShared_3123_; uint8_t v_isSharedCheck_3127_; 
v_a_3120_ = lean_ctor_get(v___x_3119_, 0);
v_isSharedCheck_3127_ = !lean_is_exclusive(v___x_3119_);
if (v_isSharedCheck_3127_ == 0)
{
v___x_3122_ = v___x_3119_;
v_isShared_3123_ = v_isSharedCheck_3127_;
goto v_resetjp_3121_;
}
else
{
lean_inc(v_a_3120_);
lean_dec(v___x_3119_);
v___x_3122_ = lean_box(0);
v_isShared_3123_ = v_isSharedCheck_3127_;
goto v_resetjp_3121_;
}
v_resetjp_3121_:
{
lean_object* v___x_3125_; 
if (v_isShared_3123_ == 0)
{
v___x_3125_ = v___x_3122_;
goto v_reusejp_3124_;
}
else
{
lean_object* v_reuseFailAlloc_3126_; 
v_reuseFailAlloc_3126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3126_, 0, v_a_3120_);
v___x_3125_ = v_reuseFailAlloc_3126_;
goto v_reusejp_3124_;
}
v_reusejp_3124_:
{
return v___x_3125_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg___boxed(lean_object* v_type_3128_, lean_object* v_maxFVars_x3f_3129_, lean_object* v_k_3130_, lean_object* v_cleanupAnnotations_3131_, lean_object* v_whnfType_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_, lean_object* v___y_3138_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_3139_; uint8_t v_whnfType_boxed_3140_; lean_object* v_res_3141_; 
v_cleanupAnnotations_boxed_3139_ = lean_unbox(v_cleanupAnnotations_3131_);
v_whnfType_boxed_3140_ = lean_unbox(v_whnfType_3132_);
v_res_3141_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg(v_type_3128_, v_maxFVars_x3f_3129_, v_k_3130_, v_cleanupAnnotations_boxed_3139_, v_whnfType_boxed_3140_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_);
lean_dec(v___y_3137_);
lean_dec_ref(v___y_3136_);
lean_dec(v___y_3135_);
lean_dec_ref(v___y_3134_);
lean_dec(v___y_3133_);
return v_res_3141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5(lean_object* v_00_u03b1_3142_, lean_object* v_type_3143_, lean_object* v_maxFVars_x3f_3144_, lean_object* v_k_3145_, uint8_t v_cleanupAnnotations_3146_, uint8_t v_whnfType_3147_, lean_object* v___y_3148_, lean_object* v___y_3149_, lean_object* v___y_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_){
_start:
{
lean_object* v___x_3154_; 
v___x_3154_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg(v_type_3143_, v_maxFVars_x3f_3144_, v_k_3145_, v_cleanupAnnotations_3146_, v_whnfType_3147_, v___y_3148_, v___y_3149_, v___y_3150_, v___y_3151_, v___y_3152_);
return v___x_3154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___boxed(lean_object* v_00_u03b1_3155_, lean_object* v_type_3156_, lean_object* v_maxFVars_x3f_3157_, lean_object* v_k_3158_, lean_object* v_cleanupAnnotations_3159_, lean_object* v_whnfType_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_, lean_object* v___y_3163_, lean_object* v___y_3164_, lean_object* v___y_3165_, lean_object* v___y_3166_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_3167_; uint8_t v_whnfType_boxed_3168_; lean_object* v_res_3169_; 
v_cleanupAnnotations_boxed_3167_ = lean_unbox(v_cleanupAnnotations_3159_);
v_whnfType_boxed_3168_ = lean_unbox(v_whnfType_3160_);
v_res_3169_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5(v_00_u03b1_3155_, v_type_3156_, v_maxFVars_x3f_3157_, v_k_3158_, v_cleanupAnnotations_boxed_3167_, v_whnfType_boxed_3168_, v___y_3161_, v___y_3162_, v___y_3163_, v___y_3164_, v___y_3165_);
lean_dec(v___y_3165_);
lean_dec_ref(v___y_3164_);
lean_dec(v___y_3163_);
lean_dec_ref(v___y_3162_);
lean_dec(v___y_3161_);
return v_res_3169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___lam__0(lean_object* v_xs_3170_, lean_object* v_x_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_, lean_object* v___y_3175_, lean_object* v___y_3176_){
_start:
{
lean_object* v___x_3178_; lean_object* v___x_3179_; 
v___x_3178_ = lean_array_get_size(v_xs_3170_);
v___x_3179_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3179_, 0, v___x_3178_);
return v___x_3179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___lam__0___boxed(lean_object* v_xs_3180_, lean_object* v_x_3181_, lean_object* v___y_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_){
_start:
{
lean_object* v_res_3188_; 
v_res_3188_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___lam__0(v_xs_3180_, v_x_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_, v___y_3186_);
lean_dec(v___y_3186_);
lean_dec_ref(v___y_3185_);
lean_dec(v___y_3184_);
lean_dec_ref(v___y_3183_);
lean_dec(v___y_3182_);
lean_dec_ref(v_x_3181_);
lean_dec_ref(v_xs_3180_);
return v_res_3188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg(lean_object* v_e_3189_, lean_object* v___y_3190_){
_start:
{
uint8_t v___x_3192_; 
v___x_3192_ = l_Lean_Expr_hasMVar(v_e_3189_);
if (v___x_3192_ == 0)
{
lean_object* v___x_3193_; 
v___x_3193_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3193_, 0, v_e_3189_);
return v___x_3193_;
}
else
{
lean_object* v___x_3194_; lean_object* v_mctx_3195_; lean_object* v___x_3196_; lean_object* v_fst_3197_; lean_object* v_snd_3198_; lean_object* v___x_3199_; lean_object* v_cache_3200_; lean_object* v_zetaDeltaFVarIds_3201_; lean_object* v_postponed_3202_; lean_object* v_diag_3203_; lean_object* v___x_3205_; uint8_t v_isShared_3206_; uint8_t v_isSharedCheck_3212_; 
v___x_3194_ = lean_st_ref_get(v___y_3190_);
v_mctx_3195_ = lean_ctor_get(v___x_3194_, 0);
lean_inc_ref(v_mctx_3195_);
lean_dec(v___x_3194_);
v___x_3196_ = l_Lean_instantiateMVarsCore(v_mctx_3195_, v_e_3189_);
v_fst_3197_ = lean_ctor_get(v___x_3196_, 0);
lean_inc(v_fst_3197_);
v_snd_3198_ = lean_ctor_get(v___x_3196_, 1);
lean_inc(v_snd_3198_);
lean_dec_ref(v___x_3196_);
v___x_3199_ = lean_st_ref_take(v___y_3190_);
v_cache_3200_ = lean_ctor_get(v___x_3199_, 1);
v_zetaDeltaFVarIds_3201_ = lean_ctor_get(v___x_3199_, 2);
v_postponed_3202_ = lean_ctor_get(v___x_3199_, 3);
v_diag_3203_ = lean_ctor_get(v___x_3199_, 4);
v_isSharedCheck_3212_ = !lean_is_exclusive(v___x_3199_);
if (v_isSharedCheck_3212_ == 0)
{
lean_object* v_unused_3213_; 
v_unused_3213_ = lean_ctor_get(v___x_3199_, 0);
lean_dec(v_unused_3213_);
v___x_3205_ = v___x_3199_;
v_isShared_3206_ = v_isSharedCheck_3212_;
goto v_resetjp_3204_;
}
else
{
lean_inc(v_diag_3203_);
lean_inc(v_postponed_3202_);
lean_inc(v_zetaDeltaFVarIds_3201_);
lean_inc(v_cache_3200_);
lean_dec(v___x_3199_);
v___x_3205_ = lean_box(0);
v_isShared_3206_ = v_isSharedCheck_3212_;
goto v_resetjp_3204_;
}
v_resetjp_3204_:
{
lean_object* v___x_3208_; 
if (v_isShared_3206_ == 0)
{
lean_ctor_set(v___x_3205_, 0, v_snd_3198_);
v___x_3208_ = v___x_3205_;
goto v_reusejp_3207_;
}
else
{
lean_object* v_reuseFailAlloc_3211_; 
v_reuseFailAlloc_3211_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3211_, 0, v_snd_3198_);
lean_ctor_set(v_reuseFailAlloc_3211_, 1, v_cache_3200_);
lean_ctor_set(v_reuseFailAlloc_3211_, 2, v_zetaDeltaFVarIds_3201_);
lean_ctor_set(v_reuseFailAlloc_3211_, 3, v_postponed_3202_);
lean_ctor_set(v_reuseFailAlloc_3211_, 4, v_diag_3203_);
v___x_3208_ = v_reuseFailAlloc_3211_;
goto v_reusejp_3207_;
}
v_reusejp_3207_:
{
lean_object* v___x_3209_; lean_object* v___x_3210_; 
v___x_3209_ = lean_st_ref_set(v___y_3190_, v___x_3208_);
v___x_3210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3210_, 0, v_fst_3197_);
return v___x_3210_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg___boxed(lean_object* v_e_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_){
_start:
{
lean_object* v_res_3217_; 
v_res_3217_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg(v_e_3214_, v___y_3215_);
lean_dec(v___y_3215_);
return v_res_3217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___redArg(lean_object* v_a_3218_, lean_object* v_x_3219_){
_start:
{
if (lean_obj_tag(v_x_3219_) == 0)
{
lean_object* v___x_3220_; 
v___x_3220_ = lean_box(0);
return v___x_3220_;
}
else
{
lean_object* v_key_3221_; lean_object* v_value_3222_; lean_object* v_tail_3223_; uint8_t v___y_3225_; lean_object* v_fst_3228_; lean_object* v_snd_3229_; lean_object* v_fst_3230_; lean_object* v_snd_3231_; uint8_t v___x_3232_; 
v_key_3221_ = lean_ctor_get(v_x_3219_, 0);
v_value_3222_ = lean_ctor_get(v_x_3219_, 1);
v_tail_3223_ = lean_ctor_get(v_x_3219_, 2);
v_fst_3228_ = lean_ctor_get(v_key_3221_, 0);
v_snd_3229_ = lean_ctor_get(v_key_3221_, 1);
v_fst_3230_ = lean_ctor_get(v_a_3218_, 0);
v_snd_3231_ = lean_ctor_get(v_a_3218_, 1);
v___x_3232_ = lean_expr_eqv(v_fst_3228_, v_fst_3230_);
if (v___x_3232_ == 0)
{
v___y_3225_ = v___x_3232_;
goto v___jp_3224_;
}
else
{
uint8_t v___x_3233_; 
v___x_3233_ = lean_expr_eqv(v_snd_3229_, v_snd_3231_);
v___y_3225_ = v___x_3233_;
goto v___jp_3224_;
}
v___jp_3224_:
{
if (v___y_3225_ == 0)
{
v_x_3219_ = v_tail_3223_;
goto _start;
}
else
{
lean_object* v___x_3227_; 
lean_inc(v_value_3222_);
v___x_3227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3227_, 0, v_value_3222_);
return v___x_3227_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___redArg___boxed(lean_object* v_a_3234_, lean_object* v_x_3235_){
_start:
{
lean_object* v_res_3236_; 
v_res_3236_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___redArg(v_a_3234_, v_x_3235_);
lean_dec(v_x_3235_);
lean_dec_ref(v_a_3234_);
return v_res_3236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___redArg(lean_object* v_m_3237_, lean_object* v_a_3238_){
_start:
{
lean_object* v_buckets_3239_; lean_object* v_fst_3240_; lean_object* v_snd_3241_; lean_object* v___x_3242_; uint64_t v___x_3243_; uint64_t v___x_3244_; uint64_t v___x_3245_; uint64_t v___x_3246_; uint64_t v___x_3247_; uint64_t v_fold_3248_; uint64_t v___x_3249_; uint64_t v___x_3250_; uint64_t v___x_3251_; size_t v___x_3252_; size_t v___x_3253_; size_t v___x_3254_; size_t v___x_3255_; size_t v___x_3256_; lean_object* v___x_3257_; lean_object* v___x_3258_; 
v_buckets_3239_ = lean_ctor_get(v_m_3237_, 1);
v_fst_3240_ = lean_ctor_get(v_a_3238_, 0);
v_snd_3241_ = lean_ctor_get(v_a_3238_, 1);
v___x_3242_ = lean_array_get_size(v_buckets_3239_);
v___x_3243_ = l_Lean_Expr_hash(v_fst_3240_);
v___x_3244_ = l_Lean_Expr_hash(v_snd_3241_);
v___x_3245_ = lean_uint64_mix_hash(v___x_3243_, v___x_3244_);
v___x_3246_ = 32ULL;
v___x_3247_ = lean_uint64_shift_right(v___x_3245_, v___x_3246_);
v_fold_3248_ = lean_uint64_xor(v___x_3245_, v___x_3247_);
v___x_3249_ = 16ULL;
v___x_3250_ = lean_uint64_shift_right(v_fold_3248_, v___x_3249_);
v___x_3251_ = lean_uint64_xor(v_fold_3248_, v___x_3250_);
v___x_3252_ = lean_uint64_to_usize(v___x_3251_);
v___x_3253_ = lean_usize_of_nat(v___x_3242_);
v___x_3254_ = ((size_t)1ULL);
v___x_3255_ = lean_usize_sub(v___x_3253_, v___x_3254_);
v___x_3256_ = lean_usize_land(v___x_3252_, v___x_3255_);
v___x_3257_ = lean_array_uget_borrowed(v_buckets_3239_, v___x_3256_);
v___x_3258_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___redArg(v_a_3238_, v___x_3257_);
return v___x_3258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___redArg___boxed(lean_object* v_m_3259_, lean_object* v_a_3260_){
_start:
{
lean_object* v_res_3261_; 
v_res_3261_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___redArg(v_m_3259_, v_a_3260_);
lean_dec_ref(v_a_3260_);
lean_dec_ref(v_m_3259_);
return v_res_3261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12_spec__15___redArg(lean_object* v_x_3262_, lean_object* v_x_3263_){
_start:
{
if (lean_obj_tag(v_x_3263_) == 0)
{
return v_x_3262_;
}
else
{
lean_object* v_key_3264_; lean_object* v_value_3265_; lean_object* v_tail_3266_; lean_object* v___x_3268_; uint8_t v_isShared_3269_; uint8_t v_isSharedCheck_3293_; 
v_key_3264_ = lean_ctor_get(v_x_3263_, 0);
v_value_3265_ = lean_ctor_get(v_x_3263_, 1);
v_tail_3266_ = lean_ctor_get(v_x_3263_, 2);
v_isSharedCheck_3293_ = !lean_is_exclusive(v_x_3263_);
if (v_isSharedCheck_3293_ == 0)
{
v___x_3268_ = v_x_3263_;
v_isShared_3269_ = v_isSharedCheck_3293_;
goto v_resetjp_3267_;
}
else
{
lean_inc(v_tail_3266_);
lean_inc(v_value_3265_);
lean_inc(v_key_3264_);
lean_dec(v_x_3263_);
v___x_3268_ = lean_box(0);
v_isShared_3269_ = v_isSharedCheck_3293_;
goto v_resetjp_3267_;
}
v_resetjp_3267_:
{
lean_object* v_fst_3270_; lean_object* v_snd_3271_; lean_object* v___x_3272_; uint64_t v___x_3273_; uint64_t v___x_3274_; uint64_t v___x_3275_; uint64_t v___x_3276_; uint64_t v___x_3277_; uint64_t v_fold_3278_; uint64_t v___x_3279_; uint64_t v___x_3280_; uint64_t v___x_3281_; size_t v___x_3282_; size_t v___x_3283_; size_t v___x_3284_; size_t v___x_3285_; size_t v___x_3286_; lean_object* v___x_3287_; lean_object* v___x_3289_; 
v_fst_3270_ = lean_ctor_get(v_key_3264_, 0);
v_snd_3271_ = lean_ctor_get(v_key_3264_, 1);
v___x_3272_ = lean_array_get_size(v_x_3262_);
v___x_3273_ = l_Lean_Expr_hash(v_fst_3270_);
v___x_3274_ = l_Lean_Expr_hash(v_snd_3271_);
v___x_3275_ = lean_uint64_mix_hash(v___x_3273_, v___x_3274_);
v___x_3276_ = 32ULL;
v___x_3277_ = lean_uint64_shift_right(v___x_3275_, v___x_3276_);
v_fold_3278_ = lean_uint64_xor(v___x_3275_, v___x_3277_);
v___x_3279_ = 16ULL;
v___x_3280_ = lean_uint64_shift_right(v_fold_3278_, v___x_3279_);
v___x_3281_ = lean_uint64_xor(v_fold_3278_, v___x_3280_);
v___x_3282_ = lean_uint64_to_usize(v___x_3281_);
v___x_3283_ = lean_usize_of_nat(v___x_3272_);
v___x_3284_ = ((size_t)1ULL);
v___x_3285_ = lean_usize_sub(v___x_3283_, v___x_3284_);
v___x_3286_ = lean_usize_land(v___x_3282_, v___x_3285_);
v___x_3287_ = lean_array_uget_borrowed(v_x_3262_, v___x_3286_);
lean_inc(v___x_3287_);
if (v_isShared_3269_ == 0)
{
lean_ctor_set(v___x_3268_, 2, v___x_3287_);
v___x_3289_ = v___x_3268_;
goto v_reusejp_3288_;
}
else
{
lean_object* v_reuseFailAlloc_3292_; 
v_reuseFailAlloc_3292_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3292_, 0, v_key_3264_);
lean_ctor_set(v_reuseFailAlloc_3292_, 1, v_value_3265_);
lean_ctor_set(v_reuseFailAlloc_3292_, 2, v___x_3287_);
v___x_3289_ = v_reuseFailAlloc_3292_;
goto v_reusejp_3288_;
}
v_reusejp_3288_:
{
lean_object* v___x_3290_; 
v___x_3290_ = lean_array_uset(v_x_3262_, v___x_3286_, v___x_3289_);
v_x_3262_ = v___x_3290_;
v_x_3263_ = v_tail_3266_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12___redArg(lean_object* v_i_3294_, lean_object* v_source_3295_, lean_object* v_target_3296_){
_start:
{
lean_object* v___x_3297_; uint8_t v___x_3298_; 
v___x_3297_ = lean_array_get_size(v_source_3295_);
v___x_3298_ = lean_nat_dec_lt(v_i_3294_, v___x_3297_);
if (v___x_3298_ == 0)
{
lean_dec_ref(v_source_3295_);
lean_dec(v_i_3294_);
return v_target_3296_;
}
else
{
lean_object* v_es_3299_; lean_object* v___x_3300_; lean_object* v_source_3301_; lean_object* v_target_3302_; lean_object* v___x_3303_; lean_object* v___x_3304_; 
v_es_3299_ = lean_array_fget(v_source_3295_, v_i_3294_);
v___x_3300_ = lean_box(0);
v_source_3301_ = lean_array_fset(v_source_3295_, v_i_3294_, v___x_3300_);
v_target_3302_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12_spec__15___redArg(v_target_3296_, v_es_3299_);
v___x_3303_ = lean_unsigned_to_nat(1u);
v___x_3304_ = lean_nat_add(v_i_3294_, v___x_3303_);
lean_dec(v_i_3294_);
v_i_3294_ = v___x_3304_;
v_source_3295_ = v_source_3301_;
v_target_3296_ = v_target_3302_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9___redArg(lean_object* v_data_3306_){
_start:
{
lean_object* v___x_3307_; lean_object* v___x_3308_; lean_object* v_nbuckets_3309_; lean_object* v___x_3310_; lean_object* v___x_3311_; lean_object* v___x_3312_; lean_object* v___x_3313_; 
v___x_3307_ = lean_array_get_size(v_data_3306_);
v___x_3308_ = lean_unsigned_to_nat(2u);
v_nbuckets_3309_ = lean_nat_mul(v___x_3307_, v___x_3308_);
v___x_3310_ = lean_unsigned_to_nat(0u);
v___x_3311_ = lean_box(0);
v___x_3312_ = lean_mk_array(v_nbuckets_3309_, v___x_3311_);
v___x_3313_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12___redArg(v___x_3310_, v_data_3306_, v___x_3312_);
return v___x_3313_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___redArg(lean_object* v_a_3314_, lean_object* v_x_3315_){
_start:
{
if (lean_obj_tag(v_x_3315_) == 0)
{
uint8_t v___x_3316_; 
v___x_3316_ = 0;
return v___x_3316_;
}
else
{
lean_object* v_key_3317_; lean_object* v_tail_3318_; uint8_t v___y_3320_; lean_object* v_fst_3322_; lean_object* v_snd_3323_; lean_object* v_fst_3324_; lean_object* v_snd_3325_; uint8_t v___x_3326_; 
v_key_3317_ = lean_ctor_get(v_x_3315_, 0);
v_tail_3318_ = lean_ctor_get(v_x_3315_, 2);
v_fst_3322_ = lean_ctor_get(v_key_3317_, 0);
v_snd_3323_ = lean_ctor_get(v_key_3317_, 1);
v_fst_3324_ = lean_ctor_get(v_a_3314_, 0);
v_snd_3325_ = lean_ctor_get(v_a_3314_, 1);
v___x_3326_ = lean_expr_eqv(v_fst_3322_, v_fst_3324_);
if (v___x_3326_ == 0)
{
v___y_3320_ = v___x_3326_;
goto v___jp_3319_;
}
else
{
uint8_t v___x_3327_; 
v___x_3327_ = lean_expr_eqv(v_snd_3323_, v_snd_3325_);
v___y_3320_ = v___x_3327_;
goto v___jp_3319_;
}
v___jp_3319_:
{
if (v___y_3320_ == 0)
{
v_x_3315_ = v_tail_3318_;
goto _start;
}
else
{
return v___y_3320_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___redArg___boxed(lean_object* v_a_3328_, lean_object* v_x_3329_){
_start:
{
uint8_t v_res_3330_; lean_object* v_r_3331_; 
v_res_3330_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___redArg(v_a_3328_, v_x_3329_);
lean_dec(v_x_3329_);
lean_dec_ref(v_a_3328_);
v_r_3331_ = lean_box(v_res_3330_);
return v_r_3331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__10___redArg(lean_object* v_a_3332_, lean_object* v_b_3333_, lean_object* v_x_3334_){
_start:
{
if (lean_obj_tag(v_x_3334_) == 0)
{
lean_dec(v_b_3333_);
lean_dec_ref(v_a_3332_);
return v_x_3334_;
}
else
{
lean_object* v_key_3335_; lean_object* v_value_3336_; lean_object* v_tail_3337_; lean_object* v___x_3339_; uint8_t v_isShared_3340_; uint8_t v_isSharedCheck_3356_; 
v_key_3335_ = lean_ctor_get(v_x_3334_, 0);
v_value_3336_ = lean_ctor_get(v_x_3334_, 1);
v_tail_3337_ = lean_ctor_get(v_x_3334_, 2);
v_isSharedCheck_3356_ = !lean_is_exclusive(v_x_3334_);
if (v_isSharedCheck_3356_ == 0)
{
v___x_3339_ = v_x_3334_;
v_isShared_3340_ = v_isSharedCheck_3356_;
goto v_resetjp_3338_;
}
else
{
lean_inc(v_tail_3337_);
lean_inc(v_value_3336_);
lean_inc(v_key_3335_);
lean_dec(v_x_3334_);
v___x_3339_ = lean_box(0);
v_isShared_3340_ = v_isSharedCheck_3356_;
goto v_resetjp_3338_;
}
v_resetjp_3338_:
{
uint8_t v___y_3342_; lean_object* v_fst_3350_; lean_object* v_snd_3351_; lean_object* v_fst_3352_; lean_object* v_snd_3353_; uint8_t v___x_3354_; 
v_fst_3350_ = lean_ctor_get(v_key_3335_, 0);
v_snd_3351_ = lean_ctor_get(v_key_3335_, 1);
v_fst_3352_ = lean_ctor_get(v_a_3332_, 0);
v_snd_3353_ = lean_ctor_get(v_a_3332_, 1);
v___x_3354_ = lean_expr_eqv(v_fst_3350_, v_fst_3352_);
if (v___x_3354_ == 0)
{
v___y_3342_ = v___x_3354_;
goto v___jp_3341_;
}
else
{
uint8_t v___x_3355_; 
v___x_3355_ = lean_expr_eqv(v_snd_3351_, v_snd_3353_);
v___y_3342_ = v___x_3355_;
goto v___jp_3341_;
}
v___jp_3341_:
{
if (v___y_3342_ == 0)
{
lean_object* v___x_3343_; lean_object* v___x_3345_; 
v___x_3343_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__10___redArg(v_a_3332_, v_b_3333_, v_tail_3337_);
if (v_isShared_3340_ == 0)
{
lean_ctor_set(v___x_3339_, 2, v___x_3343_);
v___x_3345_ = v___x_3339_;
goto v_reusejp_3344_;
}
else
{
lean_object* v_reuseFailAlloc_3346_; 
v_reuseFailAlloc_3346_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3346_, 0, v_key_3335_);
lean_ctor_set(v_reuseFailAlloc_3346_, 1, v_value_3336_);
lean_ctor_set(v_reuseFailAlloc_3346_, 2, v___x_3343_);
v___x_3345_ = v_reuseFailAlloc_3346_;
goto v_reusejp_3344_;
}
v_reusejp_3344_:
{
return v___x_3345_;
}
}
else
{
lean_object* v___x_3348_; 
lean_dec(v_value_3336_);
lean_dec(v_key_3335_);
if (v_isShared_3340_ == 0)
{
lean_ctor_set(v___x_3339_, 1, v_b_3333_);
lean_ctor_set(v___x_3339_, 0, v_a_3332_);
v___x_3348_ = v___x_3339_;
goto v_reusejp_3347_;
}
else
{
lean_object* v_reuseFailAlloc_3349_; 
v_reuseFailAlloc_3349_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3349_, 0, v_a_3332_);
lean_ctor_set(v_reuseFailAlloc_3349_, 1, v_b_3333_);
lean_ctor_set(v_reuseFailAlloc_3349_, 2, v_tail_3337_);
v___x_3348_ = v_reuseFailAlloc_3349_;
goto v_reusejp_3347_;
}
v_reusejp_3347_:
{
return v___x_3348_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8___redArg(lean_object* v_m_3357_, lean_object* v_a_3358_, lean_object* v_b_3359_){
_start:
{
lean_object* v_size_3360_; lean_object* v_buckets_3361_; lean_object* v___x_3363_; uint8_t v_isShared_3364_; uint8_t v_isSharedCheck_3408_; 
v_size_3360_ = lean_ctor_get(v_m_3357_, 0);
v_buckets_3361_ = lean_ctor_get(v_m_3357_, 1);
v_isSharedCheck_3408_ = !lean_is_exclusive(v_m_3357_);
if (v_isSharedCheck_3408_ == 0)
{
v___x_3363_ = v_m_3357_;
v_isShared_3364_ = v_isSharedCheck_3408_;
goto v_resetjp_3362_;
}
else
{
lean_inc(v_buckets_3361_);
lean_inc(v_size_3360_);
lean_dec(v_m_3357_);
v___x_3363_ = lean_box(0);
v_isShared_3364_ = v_isSharedCheck_3408_;
goto v_resetjp_3362_;
}
v_resetjp_3362_:
{
lean_object* v_fst_3365_; lean_object* v_snd_3366_; lean_object* v___x_3367_; uint64_t v___x_3368_; uint64_t v___x_3369_; uint64_t v___x_3370_; uint64_t v___x_3371_; uint64_t v___x_3372_; uint64_t v_fold_3373_; uint64_t v___x_3374_; uint64_t v___x_3375_; uint64_t v___x_3376_; size_t v___x_3377_; size_t v___x_3378_; size_t v___x_3379_; size_t v___x_3380_; size_t v___x_3381_; lean_object* v_bkt_3382_; uint8_t v___x_3383_; 
v_fst_3365_ = lean_ctor_get(v_a_3358_, 0);
v_snd_3366_ = lean_ctor_get(v_a_3358_, 1);
v___x_3367_ = lean_array_get_size(v_buckets_3361_);
v___x_3368_ = l_Lean_Expr_hash(v_fst_3365_);
v___x_3369_ = l_Lean_Expr_hash(v_snd_3366_);
v___x_3370_ = lean_uint64_mix_hash(v___x_3368_, v___x_3369_);
v___x_3371_ = 32ULL;
v___x_3372_ = lean_uint64_shift_right(v___x_3370_, v___x_3371_);
v_fold_3373_ = lean_uint64_xor(v___x_3370_, v___x_3372_);
v___x_3374_ = 16ULL;
v___x_3375_ = lean_uint64_shift_right(v_fold_3373_, v___x_3374_);
v___x_3376_ = lean_uint64_xor(v_fold_3373_, v___x_3375_);
v___x_3377_ = lean_uint64_to_usize(v___x_3376_);
v___x_3378_ = lean_usize_of_nat(v___x_3367_);
v___x_3379_ = ((size_t)1ULL);
v___x_3380_ = lean_usize_sub(v___x_3378_, v___x_3379_);
v___x_3381_ = lean_usize_land(v___x_3377_, v___x_3380_);
v_bkt_3382_ = lean_array_uget_borrowed(v_buckets_3361_, v___x_3381_);
v___x_3383_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___redArg(v_a_3358_, v_bkt_3382_);
if (v___x_3383_ == 0)
{
lean_object* v___x_3384_; lean_object* v_size_x27_3385_; lean_object* v___x_3386_; lean_object* v_buckets_x27_3387_; lean_object* v___x_3388_; lean_object* v___x_3389_; lean_object* v___x_3390_; lean_object* v___x_3391_; lean_object* v___x_3392_; uint8_t v___x_3393_; 
v___x_3384_ = lean_unsigned_to_nat(1u);
v_size_x27_3385_ = lean_nat_add(v_size_3360_, v___x_3384_);
lean_dec(v_size_3360_);
lean_inc(v_bkt_3382_);
v___x_3386_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3386_, 0, v_a_3358_);
lean_ctor_set(v___x_3386_, 1, v_b_3359_);
lean_ctor_set(v___x_3386_, 2, v_bkt_3382_);
v_buckets_x27_3387_ = lean_array_uset(v_buckets_3361_, v___x_3381_, v___x_3386_);
v___x_3388_ = lean_unsigned_to_nat(4u);
v___x_3389_ = lean_nat_mul(v_size_x27_3385_, v___x_3388_);
v___x_3390_ = lean_unsigned_to_nat(3u);
v___x_3391_ = lean_nat_div(v___x_3389_, v___x_3390_);
lean_dec(v___x_3389_);
v___x_3392_ = lean_array_get_size(v_buckets_x27_3387_);
v___x_3393_ = lean_nat_dec_le(v___x_3391_, v___x_3392_);
lean_dec(v___x_3391_);
if (v___x_3393_ == 0)
{
lean_object* v_val_3394_; lean_object* v___x_3396_; 
v_val_3394_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9___redArg(v_buckets_x27_3387_);
if (v_isShared_3364_ == 0)
{
lean_ctor_set(v___x_3363_, 1, v_val_3394_);
lean_ctor_set(v___x_3363_, 0, v_size_x27_3385_);
v___x_3396_ = v___x_3363_;
goto v_reusejp_3395_;
}
else
{
lean_object* v_reuseFailAlloc_3397_; 
v_reuseFailAlloc_3397_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3397_, 0, v_size_x27_3385_);
lean_ctor_set(v_reuseFailAlloc_3397_, 1, v_val_3394_);
v___x_3396_ = v_reuseFailAlloc_3397_;
goto v_reusejp_3395_;
}
v_reusejp_3395_:
{
return v___x_3396_;
}
}
else
{
lean_object* v___x_3399_; 
if (v_isShared_3364_ == 0)
{
lean_ctor_set(v___x_3363_, 1, v_buckets_x27_3387_);
lean_ctor_set(v___x_3363_, 0, v_size_x27_3385_);
v___x_3399_ = v___x_3363_;
goto v_reusejp_3398_;
}
else
{
lean_object* v_reuseFailAlloc_3400_; 
v_reuseFailAlloc_3400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3400_, 0, v_size_x27_3385_);
lean_ctor_set(v_reuseFailAlloc_3400_, 1, v_buckets_x27_3387_);
v___x_3399_ = v_reuseFailAlloc_3400_;
goto v_reusejp_3398_;
}
v_reusejp_3398_:
{
return v___x_3399_;
}
}
}
else
{
lean_object* v___x_3401_; lean_object* v_buckets_x27_3402_; lean_object* v___x_3403_; lean_object* v___x_3404_; lean_object* v___x_3406_; 
lean_inc(v_bkt_3382_);
v___x_3401_ = lean_box(0);
v_buckets_x27_3402_ = lean_array_uset(v_buckets_3361_, v___x_3381_, v___x_3401_);
v___x_3403_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__10___redArg(v_a_3358_, v_b_3359_, v_bkt_3382_);
v___x_3404_ = lean_array_uset(v_buckets_x27_3402_, v___x_3381_, v___x_3403_);
if (v_isShared_3364_ == 0)
{
lean_ctor_set(v___x_3363_, 1, v___x_3404_);
v___x_3406_ = v___x_3363_;
goto v_reusejp_3405_;
}
else
{
lean_object* v_reuseFailAlloc_3407_; 
v_reuseFailAlloc_3407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3407_, 0, v_size_3360_);
lean_ctor_set(v_reuseFailAlloc_3407_, 1, v___x_3404_);
v___x_3406_ = v_reuseFailAlloc_3407_;
goto v_reusejp_3405_;
}
v_reusejp_3405_:
{
return v___x_3406_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___redArg(lean_object* v_msg_3409_, lean_object* v___y_3410_, lean_object* v___y_3411_, lean_object* v___y_3412_, lean_object* v___y_3413_){
_start:
{
lean_object* v_ref_3415_; lean_object* v___x_3416_; lean_object* v_a_3417_; lean_object* v___x_3419_; uint8_t v_isShared_3420_; uint8_t v_isSharedCheck_3425_; 
v_ref_3415_ = lean_ctor_get(v___y_3412_, 5);
v___x_3416_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(v_msg_3409_, v___y_3410_, v___y_3411_, v___y_3412_, v___y_3413_);
v_a_3417_ = lean_ctor_get(v___x_3416_, 0);
v_isSharedCheck_3425_ = !lean_is_exclusive(v___x_3416_);
if (v_isSharedCheck_3425_ == 0)
{
v___x_3419_ = v___x_3416_;
v_isShared_3420_ = v_isSharedCheck_3425_;
goto v_resetjp_3418_;
}
else
{
lean_inc(v_a_3417_);
lean_dec(v___x_3416_);
v___x_3419_ = lean_box(0);
v_isShared_3420_ = v_isSharedCheck_3425_;
goto v_resetjp_3418_;
}
v_resetjp_3418_:
{
lean_object* v___x_3421_; lean_object* v___x_3423_; 
lean_inc(v_ref_3415_);
v___x_3421_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3421_, 0, v_ref_3415_);
lean_ctor_set(v___x_3421_, 1, v_a_3417_);
if (v_isShared_3420_ == 0)
{
lean_ctor_set_tag(v___x_3419_, 1);
lean_ctor_set(v___x_3419_, 0, v___x_3421_);
v___x_3423_ = v___x_3419_;
goto v_reusejp_3422_;
}
else
{
lean_object* v_reuseFailAlloc_3424_; 
v_reuseFailAlloc_3424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3424_, 0, v___x_3421_);
v___x_3423_ = v_reuseFailAlloc_3424_;
goto v_reusejp_3422_;
}
v_reusejp_3422_:
{
return v___x_3423_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___redArg___boxed(lean_object* v_msg_3426_, lean_object* v___y_3427_, lean_object* v___y_3428_, lean_object* v___y_3429_, lean_object* v___y_3430_, lean_object* v___y_3431_){
_start:
{
lean_object* v_res_3432_; 
v_res_3432_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___redArg(v_msg_3426_, v___y_3427_, v___y_3428_, v___y_3429_, v___y_3430_);
lean_dec(v___y_3430_);
lean_dec_ref(v___y_3429_);
lean_dec(v___y_3428_);
lean_dec_ref(v___y_3427_);
return v_res_3432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(lean_object* v_cls_3433_, lean_object* v_msg_3434_, lean_object* v___y_3435_, lean_object* v___y_3436_, lean_object* v___y_3437_, lean_object* v___y_3438_){
_start:
{
lean_object* v_ref_3440_; lean_object* v___x_3441_; lean_object* v_a_3442_; lean_object* v___x_3444_; uint8_t v_isShared_3445_; uint8_t v_isSharedCheck_3486_; 
v_ref_3440_ = lean_ctor_get(v___y_3437_, 5);
v___x_3441_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0_spec__0(v_msg_3434_, v___y_3435_, v___y_3436_, v___y_3437_, v___y_3438_);
v_a_3442_ = lean_ctor_get(v___x_3441_, 0);
v_isSharedCheck_3486_ = !lean_is_exclusive(v___x_3441_);
if (v_isSharedCheck_3486_ == 0)
{
v___x_3444_ = v___x_3441_;
v_isShared_3445_ = v_isSharedCheck_3486_;
goto v_resetjp_3443_;
}
else
{
lean_inc(v_a_3442_);
lean_dec(v___x_3441_);
v___x_3444_ = lean_box(0);
v_isShared_3445_ = v_isSharedCheck_3486_;
goto v_resetjp_3443_;
}
v_resetjp_3443_:
{
lean_object* v___x_3446_; lean_object* v_traceState_3447_; lean_object* v_env_3448_; lean_object* v_nextMacroScope_3449_; lean_object* v_ngen_3450_; lean_object* v_auxDeclNGen_3451_; lean_object* v_cache_3452_; lean_object* v_messages_3453_; lean_object* v_infoState_3454_; lean_object* v_snapshotTasks_3455_; lean_object* v___x_3457_; uint8_t v_isShared_3458_; uint8_t v_isSharedCheck_3485_; 
v___x_3446_ = lean_st_ref_take(v___y_3438_);
v_traceState_3447_ = lean_ctor_get(v___x_3446_, 4);
v_env_3448_ = lean_ctor_get(v___x_3446_, 0);
v_nextMacroScope_3449_ = lean_ctor_get(v___x_3446_, 1);
v_ngen_3450_ = lean_ctor_get(v___x_3446_, 2);
v_auxDeclNGen_3451_ = lean_ctor_get(v___x_3446_, 3);
v_cache_3452_ = lean_ctor_get(v___x_3446_, 5);
v_messages_3453_ = lean_ctor_get(v___x_3446_, 6);
v_infoState_3454_ = lean_ctor_get(v___x_3446_, 7);
v_snapshotTasks_3455_ = lean_ctor_get(v___x_3446_, 8);
v_isSharedCheck_3485_ = !lean_is_exclusive(v___x_3446_);
if (v_isSharedCheck_3485_ == 0)
{
v___x_3457_ = v___x_3446_;
v_isShared_3458_ = v_isSharedCheck_3485_;
goto v_resetjp_3456_;
}
else
{
lean_inc(v_snapshotTasks_3455_);
lean_inc(v_infoState_3454_);
lean_inc(v_messages_3453_);
lean_inc(v_cache_3452_);
lean_inc(v_traceState_3447_);
lean_inc(v_auxDeclNGen_3451_);
lean_inc(v_ngen_3450_);
lean_inc(v_nextMacroScope_3449_);
lean_inc(v_env_3448_);
lean_dec(v___x_3446_);
v___x_3457_ = lean_box(0);
v_isShared_3458_ = v_isSharedCheck_3485_;
goto v_resetjp_3456_;
}
v_resetjp_3456_:
{
uint64_t v_tid_3459_; lean_object* v_traces_3460_; lean_object* v___x_3462_; uint8_t v_isShared_3463_; uint8_t v_isSharedCheck_3484_; 
v_tid_3459_ = lean_ctor_get_uint64(v_traceState_3447_, sizeof(void*)*1);
v_traces_3460_ = lean_ctor_get(v_traceState_3447_, 0);
v_isSharedCheck_3484_ = !lean_is_exclusive(v_traceState_3447_);
if (v_isSharedCheck_3484_ == 0)
{
v___x_3462_ = v_traceState_3447_;
v_isShared_3463_ = v_isSharedCheck_3484_;
goto v_resetjp_3461_;
}
else
{
lean_inc(v_traces_3460_);
lean_dec(v_traceState_3447_);
v___x_3462_ = lean_box(0);
v_isShared_3463_ = v_isSharedCheck_3484_;
goto v_resetjp_3461_;
}
v_resetjp_3461_:
{
lean_object* v___x_3464_; double v___x_3465_; uint8_t v___x_3466_; lean_object* v___x_3467_; lean_object* v___x_3468_; lean_object* v___x_3469_; lean_object* v___x_3470_; lean_object* v___x_3471_; lean_object* v___x_3472_; lean_object* v___x_3474_; 
v___x_3464_ = lean_box(0);
v___x_3465_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__0);
v___x_3466_ = 0;
v___x_3467_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__1));
v___x_3468_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_3468_, 0, v_cls_3433_);
lean_ctor_set(v___x_3468_, 1, v___x_3464_);
lean_ctor_set(v___x_3468_, 2, v___x_3467_);
lean_ctor_set_float(v___x_3468_, sizeof(void*)*3, v___x_3465_);
lean_ctor_set_float(v___x_3468_, sizeof(void*)*3 + 8, v___x_3465_);
lean_ctor_set_uint8(v___x_3468_, sizeof(void*)*3 + 16, v___x_3466_);
v___x_3469_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f_spec__0___closed__2));
v___x_3470_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_3470_, 0, v___x_3468_);
lean_ctor_set(v___x_3470_, 1, v_a_3442_);
lean_ctor_set(v___x_3470_, 2, v___x_3469_);
lean_inc(v_ref_3440_);
v___x_3471_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3471_, 0, v_ref_3440_);
lean_ctor_set(v___x_3471_, 1, v___x_3470_);
v___x_3472_ = l_Lean_PersistentArray_push___redArg(v_traces_3460_, v___x_3471_);
if (v_isShared_3463_ == 0)
{
lean_ctor_set(v___x_3462_, 0, v___x_3472_);
v___x_3474_ = v___x_3462_;
goto v_reusejp_3473_;
}
else
{
lean_object* v_reuseFailAlloc_3483_; 
v_reuseFailAlloc_3483_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3483_, 0, v___x_3472_);
lean_ctor_set_uint64(v_reuseFailAlloc_3483_, sizeof(void*)*1, v_tid_3459_);
v___x_3474_ = v_reuseFailAlloc_3483_;
goto v_reusejp_3473_;
}
v_reusejp_3473_:
{
lean_object* v___x_3476_; 
if (v_isShared_3458_ == 0)
{
lean_ctor_set(v___x_3457_, 4, v___x_3474_);
v___x_3476_ = v___x_3457_;
goto v_reusejp_3475_;
}
else
{
lean_object* v_reuseFailAlloc_3482_; 
v_reuseFailAlloc_3482_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3482_, 0, v_env_3448_);
lean_ctor_set(v_reuseFailAlloc_3482_, 1, v_nextMacroScope_3449_);
lean_ctor_set(v_reuseFailAlloc_3482_, 2, v_ngen_3450_);
lean_ctor_set(v_reuseFailAlloc_3482_, 3, v_auxDeclNGen_3451_);
lean_ctor_set(v_reuseFailAlloc_3482_, 4, v___x_3474_);
lean_ctor_set(v_reuseFailAlloc_3482_, 5, v_cache_3452_);
lean_ctor_set(v_reuseFailAlloc_3482_, 6, v_messages_3453_);
lean_ctor_set(v_reuseFailAlloc_3482_, 7, v_infoState_3454_);
lean_ctor_set(v_reuseFailAlloc_3482_, 8, v_snapshotTasks_3455_);
v___x_3476_ = v_reuseFailAlloc_3482_;
goto v_reusejp_3475_;
}
v_reusejp_3475_:
{
lean_object* v___x_3477_; lean_object* v___x_3478_; lean_object* v___x_3480_; 
v___x_3477_ = lean_st_ref_set(v___y_3438_, v___x_3476_);
v___x_3478_ = lean_box(0);
if (v_isShared_3445_ == 0)
{
lean_ctor_set(v___x_3444_, 0, v___x_3478_);
v___x_3480_ = v___x_3444_;
goto v_reusejp_3479_;
}
else
{
lean_object* v_reuseFailAlloc_3481_; 
v_reuseFailAlloc_3481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3481_, 0, v___x_3478_);
v___x_3480_ = v_reuseFailAlloc_3481_;
goto v_reusejp_3479_;
}
v_reusejp_3479_:
{
return v___x_3480_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg___boxed(lean_object* v_cls_3487_, lean_object* v_msg_3488_, lean_object* v___y_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_, lean_object* v___y_3493_){
_start:
{
lean_object* v_res_3494_; 
v_res_3494_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_3487_, v_msg_3488_, v___y_3489_, v___y_3490_, v___y_3491_, v___y_3492_);
lean_dec(v___y_3492_);
lean_dec_ref(v___y_3491_);
lean_dec(v___y_3490_);
lean_dec_ref(v___y_3489_);
return v_res_3494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0(lean_object* v_cls_3495_, lean_object* v_____do__lift_3496_, lean_object* v___y_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_){
_start:
{
lean_object* v_options_3503_; uint8_t v_hasTrace_3504_; 
v_options_3503_ = lean_ctor_get(v___y_3500_, 2);
v_hasTrace_3504_ = lean_ctor_get_uint8(v_options_3503_, sizeof(void*)*1);
if (v_hasTrace_3504_ == 0)
{
lean_object* v___x_3505_; lean_object* v___x_3506_; 
lean_dec(v_cls_3495_);
v___x_3505_ = lean_box(v_hasTrace_3504_);
v___x_3506_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3506_, 0, v___x_3505_);
return v___x_3506_;
}
else
{
lean_object* v___x_3507_; lean_object* v___x_3508_; uint8_t v___x_3509_; lean_object* v___x_3510_; lean_object* v___x_3511_; 
v___x_3507_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__22));
v___x_3508_ = l_Lean_Name_append(v___x_3507_, v_cls_3495_);
v___x_3509_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_____do__lift_3496_, v_options_3503_, v___x_3508_);
lean_dec(v___x_3508_);
v___x_3510_ = lean_box(v___x_3509_);
v___x_3511_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3511_, 0, v___x_3510_);
return v___x_3511_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0___boxed(lean_object* v_cls_3512_, lean_object* v_____do__lift_3513_, lean_object* v___y_3514_, lean_object* v___y_3515_, lean_object* v___y_3516_, lean_object* v___y_3517_, lean_object* v___y_3518_, lean_object* v___y_3519_){
_start:
{
lean_object* v_res_3520_; 
v_res_3520_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0(v_cls_3512_, v_____do__lift_3513_, v___y_3514_, v___y_3515_, v___y_3516_, v___y_3517_, v___y_3518_);
lean_dec(v___y_3518_);
lean_dec_ref(v___y_3517_);
lean_dec(v___y_3516_);
lean_dec_ref(v___y_3515_);
lean_dec(v___y_3514_);
lean_dec_ref(v_____do__lift_3513_);
return v_res_3520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___lam__0(lean_object* v_k_3521_, lean_object* v___y_3522_, lean_object* v_b_3523_, lean_object* v___y_3524_, lean_object* v___y_3525_, lean_object* v___y_3526_, lean_object* v___y_3527_){
_start:
{
lean_object* v___x_3529_; 
lean_inc(v___y_3527_);
lean_inc_ref(v___y_3526_);
lean_inc(v___y_3525_);
lean_inc_ref(v___y_3524_);
lean_inc(v___y_3522_);
v___x_3529_ = lean_apply_7(v_k_3521_, v_b_3523_, v___y_3522_, v___y_3524_, v___y_3525_, v___y_3526_, v___y_3527_, lean_box(0));
return v___x_3529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___lam__0___boxed(lean_object* v_k_3530_, lean_object* v___y_3531_, lean_object* v_b_3532_, lean_object* v___y_3533_, lean_object* v___y_3534_, lean_object* v___y_3535_, lean_object* v___y_3536_, lean_object* v___y_3537_){
_start:
{
lean_object* v_res_3538_; 
v_res_3538_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___lam__0(v_k_3530_, v___y_3531_, v_b_3532_, v___y_3533_, v___y_3534_, v___y_3535_, v___y_3536_);
lean_dec(v___y_3536_);
lean_dec_ref(v___y_3535_);
lean_dec(v___y_3534_);
lean_dec_ref(v___y_3533_);
lean_dec(v___y_3531_);
return v_res_3538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg(lean_object* v_name_3539_, uint8_t v_bi_3540_, lean_object* v_type_3541_, lean_object* v_k_3542_, uint8_t v_kind_3543_, lean_object* v___y_3544_, lean_object* v___y_3545_, lean_object* v___y_3546_, lean_object* v___y_3547_, lean_object* v___y_3548_){
_start:
{
lean_object* v___f_3550_; lean_object* v___x_3551_; 
lean_inc(v___y_3544_);
v___f_3550_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_3550_, 0, v_k_3542_);
lean_closure_set(v___f_3550_, 1, v___y_3544_);
v___x_3551_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_3539_, v_bi_3540_, v_type_3541_, v___f_3550_, v_kind_3543_, v___y_3545_, v___y_3546_, v___y_3547_, v___y_3548_);
if (lean_obj_tag(v___x_3551_) == 0)
{
return v___x_3551_;
}
else
{
lean_object* v_a_3552_; lean_object* v___x_3554_; uint8_t v_isShared_3555_; uint8_t v_isSharedCheck_3559_; 
v_a_3552_ = lean_ctor_get(v___x_3551_, 0);
v_isSharedCheck_3559_ = !lean_is_exclusive(v___x_3551_);
if (v_isSharedCheck_3559_ == 0)
{
v___x_3554_ = v___x_3551_;
v_isShared_3555_ = v_isSharedCheck_3559_;
goto v_resetjp_3553_;
}
else
{
lean_inc(v_a_3552_);
lean_dec(v___x_3551_);
v___x_3554_ = lean_box(0);
v_isShared_3555_ = v_isSharedCheck_3559_;
goto v_resetjp_3553_;
}
v_resetjp_3553_:
{
lean_object* v___x_3557_; 
if (v_isShared_3555_ == 0)
{
v___x_3557_ = v___x_3554_;
goto v_reusejp_3556_;
}
else
{
lean_object* v_reuseFailAlloc_3558_; 
v_reuseFailAlloc_3558_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3558_, 0, v_a_3552_);
v___x_3557_ = v_reuseFailAlloc_3558_;
goto v_reusejp_3556_;
}
v_reusejp_3556_:
{
return v___x_3557_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg___boxed(lean_object* v_name_3560_, lean_object* v_bi_3561_, lean_object* v_type_3562_, lean_object* v_k_3563_, lean_object* v_kind_3564_, lean_object* v___y_3565_, lean_object* v___y_3566_, lean_object* v___y_3567_, lean_object* v___y_3568_, lean_object* v___y_3569_, lean_object* v___y_3570_){
_start:
{
uint8_t v_bi_boxed_3571_; uint8_t v_kind_boxed_3572_; lean_object* v_res_3573_; 
v_bi_boxed_3571_ = lean_unbox(v_bi_3561_);
v_kind_boxed_3572_ = lean_unbox(v_kind_3564_);
v_res_3573_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg(v_name_3560_, v_bi_boxed_3571_, v_type_3562_, v_k_3563_, v_kind_boxed_3572_, v___y_3565_, v___y_3566_, v___y_3567_, v___y_3568_, v___y_3569_);
lean_dec(v___y_3569_);
lean_dec_ref(v___y_3568_);
lean_dec(v___y_3567_);
lean_dec_ref(v___y_3566_);
lean_dec(v___y_3565_);
return v_res_3573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__0(size_t v_sz_3574_, size_t v_i_3575_, lean_object* v_bs_3576_){
_start:
{
uint8_t v___x_3577_; 
v___x_3577_ = lean_usize_dec_lt(v_i_3575_, v_sz_3574_);
if (v___x_3577_ == 0)
{
return v_bs_3576_;
}
else
{
lean_object* v_v_3578_; lean_object* v___x_3579_; lean_object* v_bs_x27_3580_; lean_object* v___x_3581_; size_t v___x_3582_; size_t v___x_3583_; lean_object* v___x_3584_; 
v_v_3578_ = lean_array_uget(v_bs_3576_, v_i_3575_);
v___x_3579_ = lean_unsigned_to_nat(0u);
v_bs_x27_3580_ = lean_array_uset(v_bs_3576_, v_i_3575_, v___x_3579_);
v___x_3581_ = lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(v_v_3578_);
lean_dec(v_v_3578_);
v___x_3582_ = ((size_t)1ULL);
v___x_3583_ = lean_usize_add(v_i_3575_, v___x_3582_);
v___x_3584_ = lean_array_uset(v_bs_x27_3580_, v_i_3575_, v___x_3581_);
v_i_3575_ = v___x_3583_;
v_bs_3576_ = v___x_3584_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__0___boxed(lean_object* v_sz_3586_, lean_object* v_i_3587_, lean_object* v_bs_3588_){
_start:
{
size_t v_sz_boxed_3589_; size_t v_i_boxed_3590_; lean_object* v_res_3591_; 
v_sz_boxed_3589_ = lean_unbox_usize(v_sz_3586_);
lean_dec(v_sz_3586_);
v_i_boxed_3590_ = lean_unbox_usize(v_i_3587_);
lean_dec(v_i_3587_);
v_res_3591_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__0(v_sz_boxed_3589_, v_i_boxed_3590_, v_bs_3588_);
return v_res_3591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___redArg(lean_object* v_as_3592_, size_t v_sz_3593_, size_t v_i_3594_, lean_object* v_b_3595_, lean_object* v___y_3596_, lean_object* v___y_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_){
_start:
{
uint8_t v___x_3601_; 
v___x_3601_ = lean_usize_dec_lt(v_i_3594_, v_sz_3593_);
if (v___x_3601_ == 0)
{
lean_object* v___x_3602_; 
v___x_3602_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3602_, 0, v_b_3595_);
return v___x_3602_;
}
else
{
lean_object* v_a_3603_; lean_object* v___x_3604_; 
v_a_3603_ = lean_array_uget_borrowed(v_as_3592_, v_i_3594_);
lean_inc(v_a_3603_);
v___x_3604_ = l_Lean_Meta_mkCongrFun(v_b_3595_, v_a_3603_, v___y_3596_, v___y_3597_, v___y_3598_, v___y_3599_);
if (lean_obj_tag(v___x_3604_) == 0)
{
lean_object* v_a_3605_; size_t v___x_3606_; size_t v___x_3607_; 
v_a_3605_ = lean_ctor_get(v___x_3604_, 0);
lean_inc(v_a_3605_);
lean_dec_ref_known(v___x_3604_, 1);
v___x_3606_ = ((size_t)1ULL);
v___x_3607_ = lean_usize_add(v_i_3594_, v___x_3606_);
v_i_3594_ = v___x_3607_;
v_b_3595_ = v_a_3605_;
goto _start;
}
else
{
return v___x_3604_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___redArg___boxed(lean_object* v_as_3609_, lean_object* v_sz_3610_, lean_object* v_i_3611_, lean_object* v_b_3612_, lean_object* v___y_3613_, lean_object* v___y_3614_, lean_object* v___y_3615_, lean_object* v___y_3616_, lean_object* v___y_3617_){
_start:
{
size_t v_sz_boxed_3618_; size_t v_i_boxed_3619_; lean_object* v_res_3620_; 
v_sz_boxed_3618_ = lean_unbox_usize(v_sz_3610_);
lean_dec(v_sz_3610_);
v_i_boxed_3619_ = lean_unbox_usize(v_i_3611_);
lean_dec(v_i_3611_);
v_res_3620_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___redArg(v_as_3609_, v_sz_boxed_3618_, v_i_boxed_3619_, v_b_3612_, v___y_3613_, v___y_3614_, v___y_3615_, v___y_3616_);
lean_dec(v___y_3616_);
lean_dec_ref(v___y_3615_);
lean_dec(v___y_3614_);
lean_dec_ref(v___y_3613_);
lean_dec_ref(v_as_3609_);
return v_res_3620_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__2(void){
_start:
{
lean_object* v___x_3627_; lean_object* v___x_3628_; 
v___x_3627_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__1));
v___x_3628_ = l_Lean_stringToMessageData(v___x_3627_);
return v___x_3628_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__2(void){
_start:
{
lean_object* v___x_3632_; lean_object* v___x_3633_; 
v___x_3632_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__1));
v___x_3633_ = l_Lean_MessageData_ofFormat(v___x_3632_);
return v___x_3633_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__4(void){
_start:
{
lean_object* v___x_3635_; lean_object* v___x_3636_; 
v___x_3635_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__3));
v___x_3636_ = l_Lean_stringToMessageData(v___x_3635_);
return v___x_3636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___boxed(lean_object* v_a_3637_, lean_object* v_a_3638_, lean_object* v___x_3639_, lean_object* v_mvarCounterSaved_3640_, lean_object* v___x_3641_, lean_object* v___y_3642_, lean_object* v___x_3643_, lean_object* v_x_3644_, lean_object* v___y_3645_, lean_object* v___y_3646_, lean_object* v___y_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_){
_start:
{
uint8_t v___y_83414__boxed_3651_; uint8_t v___x_83415__boxed_3652_; lean_object* v_res_3653_; 
v___y_83414__boxed_3651_ = lean_unbox(v___y_3642_);
v___x_83415__boxed_3652_ = lean_unbox(v___x_3643_);
v_res_3653_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1(v_a_3637_, v_a_3638_, v___x_3639_, v_mvarCounterSaved_3640_, v___x_3641_, v___y_83414__boxed_3651_, v___x_83415__boxed_3652_, v_x_3644_, v___y_3645_, v___y_3646_, v___y_3647_, v___y_3648_, v___y_3649_);
lean_dec(v___y_3649_);
lean_dec_ref(v___y_3648_);
lean_dec(v___y_3647_);
lean_dec_ref(v___y_3646_);
lean_dec(v___y_3645_);
lean_dec(v___x_3641_);
lean_dec_ref(v_a_3638_);
lean_dec_ref(v_a_3637_);
return v_res_3653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2(lean_object* v_a_3657_, lean_object* v_a_3658_, lean_object* v___x_3659_, lean_object* v_mvarCounterSaved_3660_, lean_object* v___x_3661_, uint8_t v_a_3662_, uint8_t v___x_3663_, lean_object* v_x_3664_, lean_object* v___y_3665_, lean_object* v___y_3666_, lean_object* v___y_3667_, lean_object* v___y_3668_, lean_object* v___y_3669_){
_start:
{
lean_object* v___x_3671_; lean_object* v___x_3672_; lean_object* v___x_3673_; lean_object* v___x_3674_; lean_object* v___x_3675_; 
v___x_3671_ = l_Lean_Expr_bindingBody_x21(v_a_3657_);
v___x_3672_ = lean_expr_instantiate1(v___x_3671_, v_x_3664_);
lean_dec_ref(v___x_3671_);
v___x_3673_ = l_Lean_Expr_bindingBody_x21(v_a_3658_);
v___x_3674_ = lean_expr_instantiate1(v___x_3673_, v_x_3664_);
lean_dec_ref(v___x_3673_);
v___x_3675_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_3659_, v_mvarCounterSaved_3660_, v___x_3672_, v___x_3674_, v___y_3665_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
if (lean_obj_tag(v___x_3675_) == 0)
{
lean_object* v_a_3676_; lean_object* v_lhs_3677_; lean_object* v_rhs_3678_; lean_object* v___x_3679_; lean_object* v___x_3680_; uint8_t v___x_3681_; lean_object* v___x_3682_; 
v_a_3676_ = lean_ctor_get(v___x_3675_, 0);
lean_inc(v_a_3676_);
lean_dec_ref_known(v___x_3675_, 1);
v_lhs_3677_ = lean_ctor_get(v_a_3676_, 0);
v_rhs_3678_ = lean_ctor_get(v_a_3676_, 1);
v___x_3679_ = lean_mk_empty_array_with_capacity(v___x_3661_);
lean_inc_ref(v___x_3679_);
v___x_3680_ = lean_array_push(v___x_3679_, v_x_3664_);
v___x_3681_ = 1;
lean_inc_ref(v_lhs_3677_);
v___x_3682_ = l_Lean_Meta_mkLambdaFVars(v___x_3680_, v_lhs_3677_, v_a_3662_, v___x_3663_, v_a_3662_, v___x_3663_, v___x_3681_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
if (lean_obj_tag(v___x_3682_) == 0)
{
lean_object* v_a_3683_; lean_object* v___x_3684_; 
v_a_3683_ = lean_ctor_get(v___x_3682_, 0);
lean_inc(v_a_3683_);
lean_dec_ref_known(v___x_3682_, 1);
lean_inc_ref(v_rhs_3678_);
v___x_3684_ = l_Lean_Meta_mkLambdaFVars(v___x_3680_, v_rhs_3678_, v_a_3662_, v___x_3663_, v_a_3662_, v___x_3663_, v___x_3681_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
if (lean_obj_tag(v___x_3684_) == 0)
{
lean_object* v_a_3685_; lean_object* v___x_3687_; uint8_t v_isShared_3688_; uint8_t v_isSharedCheck_3744_; 
v_a_3685_ = lean_ctor_get(v___x_3684_, 0);
v_isSharedCheck_3744_ = !lean_is_exclusive(v___x_3684_);
if (v_isSharedCheck_3744_ == 0)
{
v___x_3687_ = v___x_3684_;
v_isShared_3688_ = v_isSharedCheck_3744_;
goto v_resetjp_3686_;
}
else
{
lean_inc(v_a_3685_);
lean_dec(v___x_3684_);
v___x_3687_ = lean_box(0);
v_isShared_3688_ = v_isSharedCheck_3744_;
goto v_resetjp_3686_;
}
v_resetjp_3686_:
{
uint8_t v___x_3689_; 
v___x_3689_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_a_3676_);
if (v___x_3689_ == 0)
{
lean_object* v___x_3690_; 
lean_del_object(v___x_3687_);
v___x_3690_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_a_3676_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
if (lean_obj_tag(v___x_3690_) == 0)
{
lean_object* v_a_3691_; lean_object* v___x_3692_; 
v_a_3691_ = lean_ctor_get(v___x_3690_, 0);
lean_inc(v_a_3691_);
lean_dec_ref_known(v___x_3690_, 1);
v___x_3692_ = l_Lean_Meta_mkLambdaFVars(v___x_3680_, v_a_3691_, v_a_3662_, v___x_3663_, v_a_3662_, v___x_3663_, v___x_3681_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
lean_dec_ref(v___x_3680_);
if (lean_obj_tag(v___x_3692_) == 0)
{
lean_object* v_a_3693_; lean_object* v___x_3694_; lean_object* v___x_3695_; lean_object* v___x_3696_; 
v_a_3693_ = lean_ctor_get(v___x_3692_, 0);
lean_inc(v_a_3693_);
lean_dec_ref_known(v___x_3692_, 1);
v___x_3694_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___closed__1));
v___x_3695_ = lean_array_push(v___x_3679_, v_a_3693_);
v___x_3696_ = l_Lean_Meta_mkAppM(v___x_3694_, v___x_3695_, v___y_3666_, v___y_3667_, v___y_3668_, v___y_3669_);
if (lean_obj_tag(v___x_3696_) == 0)
{
lean_object* v_a_3697_; lean_object* v___x_3699_; uint8_t v_isShared_3700_; uint8_t v_isSharedCheck_3705_; 
v_a_3697_ = lean_ctor_get(v___x_3696_, 0);
v_isSharedCheck_3705_ = !lean_is_exclusive(v___x_3696_);
if (v_isSharedCheck_3705_ == 0)
{
v___x_3699_ = v___x_3696_;
v_isShared_3700_ = v_isSharedCheck_3705_;
goto v_resetjp_3698_;
}
else
{
lean_inc(v_a_3697_);
lean_dec(v___x_3696_);
v___x_3699_ = lean_box(0);
v_isShared_3700_ = v_isSharedCheck_3705_;
goto v_resetjp_3698_;
}
v_resetjp_3698_:
{
lean_object* v___x_3701_; lean_object* v___x_3703_; 
v___x_3701_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v_a_3683_, v_a_3685_, v_a_3697_);
if (v_isShared_3700_ == 0)
{
lean_ctor_set(v___x_3699_, 0, v___x_3701_);
v___x_3703_ = v___x_3699_;
goto v_reusejp_3702_;
}
else
{
lean_object* v_reuseFailAlloc_3704_; 
v_reuseFailAlloc_3704_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3704_, 0, v___x_3701_);
v___x_3703_ = v_reuseFailAlloc_3704_;
goto v_reusejp_3702_;
}
v_reusejp_3702_:
{
return v___x_3703_;
}
}
}
else
{
lean_object* v_a_3706_; lean_object* v___x_3708_; uint8_t v_isShared_3709_; uint8_t v_isSharedCheck_3713_; 
lean_dec(v_a_3685_);
lean_dec(v_a_3683_);
v_a_3706_ = lean_ctor_get(v___x_3696_, 0);
v_isSharedCheck_3713_ = !lean_is_exclusive(v___x_3696_);
if (v_isSharedCheck_3713_ == 0)
{
v___x_3708_ = v___x_3696_;
v_isShared_3709_ = v_isSharedCheck_3713_;
goto v_resetjp_3707_;
}
else
{
lean_inc(v_a_3706_);
lean_dec(v___x_3696_);
v___x_3708_ = lean_box(0);
v_isShared_3709_ = v_isSharedCheck_3713_;
goto v_resetjp_3707_;
}
v_resetjp_3707_:
{
lean_object* v___x_3711_; 
if (v_isShared_3709_ == 0)
{
v___x_3711_ = v___x_3708_;
goto v_reusejp_3710_;
}
else
{
lean_object* v_reuseFailAlloc_3712_; 
v_reuseFailAlloc_3712_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3712_, 0, v_a_3706_);
v___x_3711_ = v_reuseFailAlloc_3712_;
goto v_reusejp_3710_;
}
v_reusejp_3710_:
{
return v___x_3711_;
}
}
}
}
else
{
lean_object* v_a_3714_; lean_object* v___x_3716_; uint8_t v_isShared_3717_; uint8_t v_isSharedCheck_3721_; 
lean_dec(v_a_3685_);
lean_dec(v_a_3683_);
lean_dec_ref(v___x_3679_);
v_a_3714_ = lean_ctor_get(v___x_3692_, 0);
v_isSharedCheck_3721_ = !lean_is_exclusive(v___x_3692_);
if (v_isSharedCheck_3721_ == 0)
{
v___x_3716_ = v___x_3692_;
v_isShared_3717_ = v_isSharedCheck_3721_;
goto v_resetjp_3715_;
}
else
{
lean_inc(v_a_3714_);
lean_dec(v___x_3692_);
v___x_3716_ = lean_box(0);
v_isShared_3717_ = v_isSharedCheck_3721_;
goto v_resetjp_3715_;
}
v_resetjp_3715_:
{
lean_object* v___x_3719_; 
if (v_isShared_3717_ == 0)
{
v___x_3719_ = v___x_3716_;
goto v_reusejp_3718_;
}
else
{
lean_object* v_reuseFailAlloc_3720_; 
v_reuseFailAlloc_3720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3720_, 0, v_a_3714_);
v___x_3719_ = v_reuseFailAlloc_3720_;
goto v_reusejp_3718_;
}
v_reusejp_3718_:
{
return v___x_3719_;
}
}
}
}
else
{
lean_object* v_a_3722_; lean_object* v___x_3724_; uint8_t v_isShared_3725_; uint8_t v_isSharedCheck_3729_; 
lean_dec(v_a_3685_);
lean_dec(v_a_3683_);
lean_dec_ref(v___x_3680_);
lean_dec_ref(v___x_3679_);
v_a_3722_ = lean_ctor_get(v___x_3690_, 0);
v_isSharedCheck_3729_ = !lean_is_exclusive(v___x_3690_);
if (v_isSharedCheck_3729_ == 0)
{
v___x_3724_ = v___x_3690_;
v_isShared_3725_ = v_isSharedCheck_3729_;
goto v_resetjp_3723_;
}
else
{
lean_inc(v_a_3722_);
lean_dec(v___x_3690_);
v___x_3724_ = lean_box(0);
v_isShared_3725_ = v_isSharedCheck_3729_;
goto v_resetjp_3723_;
}
v_resetjp_3723_:
{
lean_object* v___x_3727_; 
if (v_isShared_3725_ == 0)
{
v___x_3727_ = v___x_3724_;
goto v_reusejp_3726_;
}
else
{
lean_object* v_reuseFailAlloc_3728_; 
v_reuseFailAlloc_3728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3728_, 0, v_a_3722_);
v___x_3727_ = v_reuseFailAlloc_3728_;
goto v_reusejp_3726_;
}
v_reusejp_3726_:
{
return v___x_3727_;
}
}
}
}
else
{
lean_object* v___x_3731_; uint8_t v_isShared_3732_; uint8_t v_isSharedCheck_3740_; 
lean_dec_ref(v___x_3680_);
lean_dec_ref(v___x_3679_);
v_isSharedCheck_3740_ = !lean_is_exclusive(v_a_3676_);
if (v_isSharedCheck_3740_ == 0)
{
lean_object* v_unused_3741_; lean_object* v_unused_3742_; lean_object* v_unused_3743_; 
v_unused_3741_ = lean_ctor_get(v_a_3676_, 2);
lean_dec(v_unused_3741_);
v_unused_3742_ = lean_ctor_get(v_a_3676_, 1);
lean_dec(v_unused_3742_);
v_unused_3743_ = lean_ctor_get(v_a_3676_, 0);
lean_dec(v_unused_3743_);
v___x_3731_ = v_a_3676_;
v_isShared_3732_ = v_isSharedCheck_3740_;
goto v_resetjp_3730_;
}
else
{
lean_dec(v_a_3676_);
v___x_3731_ = lean_box(0);
v_isShared_3732_ = v_isSharedCheck_3740_;
goto v_resetjp_3730_;
}
v_resetjp_3730_:
{
lean_object* v___x_3733_; lean_object* v___x_3735_; 
v___x_3733_ = lean_box(0);
if (v_isShared_3732_ == 0)
{
lean_ctor_set(v___x_3731_, 2, v___x_3733_);
lean_ctor_set(v___x_3731_, 1, v_a_3685_);
lean_ctor_set(v___x_3731_, 0, v_a_3683_);
v___x_3735_ = v___x_3731_;
goto v_reusejp_3734_;
}
else
{
lean_object* v_reuseFailAlloc_3739_; 
v_reuseFailAlloc_3739_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3739_, 0, v_a_3683_);
lean_ctor_set(v_reuseFailAlloc_3739_, 1, v_a_3685_);
lean_ctor_set(v_reuseFailAlloc_3739_, 2, v___x_3733_);
v___x_3735_ = v_reuseFailAlloc_3739_;
goto v_reusejp_3734_;
}
v_reusejp_3734_:
{
lean_object* v___x_3737_; 
if (v_isShared_3688_ == 0)
{
lean_ctor_set(v___x_3687_, 0, v___x_3735_);
v___x_3737_ = v___x_3687_;
goto v_reusejp_3736_;
}
else
{
lean_object* v_reuseFailAlloc_3738_; 
v_reuseFailAlloc_3738_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3738_, 0, v___x_3735_);
v___x_3737_ = v_reuseFailAlloc_3738_;
goto v_reusejp_3736_;
}
v_reusejp_3736_:
{
return v___x_3737_;
}
}
}
}
}
}
else
{
lean_object* v_a_3745_; lean_object* v___x_3747_; uint8_t v_isShared_3748_; uint8_t v_isSharedCheck_3752_; 
lean_dec(v_a_3683_);
lean_dec_ref(v___x_3680_);
lean_dec_ref(v___x_3679_);
lean_dec(v_a_3676_);
v_a_3745_ = lean_ctor_get(v___x_3684_, 0);
v_isSharedCheck_3752_ = !lean_is_exclusive(v___x_3684_);
if (v_isSharedCheck_3752_ == 0)
{
v___x_3747_ = v___x_3684_;
v_isShared_3748_ = v_isSharedCheck_3752_;
goto v_resetjp_3746_;
}
else
{
lean_inc(v_a_3745_);
lean_dec(v___x_3684_);
v___x_3747_ = lean_box(0);
v_isShared_3748_ = v_isSharedCheck_3752_;
goto v_resetjp_3746_;
}
v_resetjp_3746_:
{
lean_object* v___x_3750_; 
if (v_isShared_3748_ == 0)
{
v___x_3750_ = v___x_3747_;
goto v_reusejp_3749_;
}
else
{
lean_object* v_reuseFailAlloc_3751_; 
v_reuseFailAlloc_3751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3751_, 0, v_a_3745_);
v___x_3750_ = v_reuseFailAlloc_3751_;
goto v_reusejp_3749_;
}
v_reusejp_3749_:
{
return v___x_3750_;
}
}
}
}
else
{
lean_object* v_a_3753_; lean_object* v___x_3755_; uint8_t v_isShared_3756_; uint8_t v_isSharedCheck_3760_; 
lean_dec_ref(v___x_3680_);
lean_dec_ref(v___x_3679_);
lean_dec(v_a_3676_);
v_a_3753_ = lean_ctor_get(v___x_3682_, 0);
v_isSharedCheck_3760_ = !lean_is_exclusive(v___x_3682_);
if (v_isSharedCheck_3760_ == 0)
{
v___x_3755_ = v___x_3682_;
v_isShared_3756_ = v_isSharedCheck_3760_;
goto v_resetjp_3754_;
}
else
{
lean_inc(v_a_3753_);
lean_dec(v___x_3682_);
v___x_3755_ = lean_box(0);
v_isShared_3756_ = v_isSharedCheck_3760_;
goto v_resetjp_3754_;
}
v_resetjp_3754_:
{
lean_object* v___x_3758_; 
if (v_isShared_3756_ == 0)
{
v___x_3758_ = v___x_3755_;
goto v_reusejp_3757_;
}
else
{
lean_object* v_reuseFailAlloc_3759_; 
v_reuseFailAlloc_3759_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3759_, 0, v_a_3753_);
v___x_3758_ = v_reuseFailAlloc_3759_;
goto v_reusejp_3757_;
}
v_reusejp_3757_:
{
return v___x_3758_;
}
}
}
}
else
{
lean_dec_ref(v_x_3664_);
return v___x_3675_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___boxed(lean_object* v_a_3761_, lean_object* v_a_3762_, lean_object* v___x_3763_, lean_object* v_mvarCounterSaved_3764_, lean_object* v___x_3765_, lean_object* v_a_3766_, lean_object* v___x_3767_, lean_object* v_x_3768_, lean_object* v___y_3769_, lean_object* v___y_3770_, lean_object* v___y_3771_, lean_object* v___y_3772_, lean_object* v___y_3773_, lean_object* v___y_3774_){
_start:
{
uint8_t v_a_83467__boxed_3775_; uint8_t v___x_83468__boxed_3776_; lean_object* v_res_3777_; 
v_a_83467__boxed_3775_ = lean_unbox(v_a_3766_);
v___x_83468__boxed_3776_ = lean_unbox(v___x_3767_);
v_res_3777_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2(v_a_3761_, v_a_3762_, v___x_3763_, v_mvarCounterSaved_3764_, v___x_3765_, v_a_83467__boxed_3775_, v___x_83468__boxed_3776_, v_x_3768_, v___y_3769_, v___y_3770_, v___y_3771_, v___y_3772_, v___y_3773_);
lean_dec(v___y_3773_);
lean_dec_ref(v___y_3772_);
lean_dec(v___y_3771_);
lean_dec_ref(v___y_3770_);
lean_dec(v___y_3769_);
lean_dec(v___x_3765_);
lean_dec_ref(v_a_3762_);
lean_dec_ref(v_a_3761_);
return v_res_3777_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_3782_; lean_object* v___x_3783_; lean_object* v___x_3784_; lean_object* v___x_3785_; lean_object* v___x_3786_; lean_object* v___x_3787_; 
v___x_3782_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__1));
v___x_3783_ = lean_unsigned_to_nat(21u);
v___x_3784_ = lean_unsigned_to_nat(673u);
v___x_3785_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__0));
v___x_3786_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__6));
v___x_3787_ = l_mkPanicMessageWithDecl(v___x_3786_, v___x_3785_, v___x_3784_, v___x_3783_, v___x_3782_);
return v___x_3787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg(lean_object* v___x_3788_, lean_object* v_mvarCounterSaved_3789_, lean_object* v_a_3790_, lean_object* v_b_3791_, lean_object* v___y_3792_, lean_object* v___y_3793_, lean_object* v___y_3794_, lean_object* v___y_3795_, lean_object* v___y_3796_){
_start:
{
lean_object* v_array_3798_; lean_object* v_start_3799_; lean_object* v_stop_3800_; lean_object* v___x_3802_; uint8_t v_isShared_3803_; uint8_t v_isSharedCheck_4021_; 
v_array_3798_ = lean_ctor_get(v_a_3790_, 0);
v_start_3799_ = lean_ctor_get(v_a_3790_, 1);
v_stop_3800_ = lean_ctor_get(v_a_3790_, 2);
v_isSharedCheck_4021_ = !lean_is_exclusive(v_a_3790_);
if (v_isSharedCheck_4021_ == 0)
{
v___x_3802_ = v_a_3790_;
v_isShared_3803_ = v_isSharedCheck_4021_;
goto v_resetjp_3801_;
}
else
{
lean_inc(v_stop_3800_);
lean_inc(v_start_3799_);
lean_inc(v_array_3798_);
lean_dec(v_a_3790_);
v___x_3802_ = lean_box(0);
v_isShared_3803_ = v_isSharedCheck_4021_;
goto v_resetjp_3801_;
}
v_resetjp_3801_:
{
uint8_t v___x_3804_; 
v___x_3804_ = lean_nat_dec_lt(v_start_3799_, v_stop_3800_);
if (v___x_3804_ == 0)
{
lean_object* v___x_3805_; 
lean_del_object(v___x_3802_);
lean_dec(v_stop_3800_);
lean_dec(v_start_3799_);
lean_dec_ref(v_array_3798_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
v___x_3805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3805_, 0, v_b_3791_);
return v___x_3805_;
}
else
{
lean_object* v_snd_3806_; lean_object* v_snd_3807_; lean_object* v_snd_3808_; lean_object* v_snd_3809_; lean_object* v_fst_3810_; lean_object* v___x_3812_; uint8_t v_isShared_3813_; uint8_t v_isSharedCheck_4019_; 
v_snd_3806_ = lean_ctor_get(v_b_3791_, 1);
lean_inc(v_snd_3806_);
v_snd_3807_ = lean_ctor_get(v_snd_3806_, 1);
lean_inc(v_snd_3807_);
v_snd_3808_ = lean_ctor_get(v_snd_3807_, 1);
lean_inc(v_snd_3808_);
v_snd_3809_ = lean_ctor_get(v_snd_3808_, 1);
lean_inc(v_snd_3809_);
v_fst_3810_ = lean_ctor_get(v_b_3791_, 0);
v_isSharedCheck_4019_ = !lean_is_exclusive(v_b_3791_);
if (v_isSharedCheck_4019_ == 0)
{
lean_object* v_unused_4020_; 
v_unused_4020_ = lean_ctor_get(v_b_3791_, 1);
lean_dec(v_unused_4020_);
v___x_3812_ = v_b_3791_;
v_isShared_3813_ = v_isSharedCheck_4019_;
goto v_resetjp_3811_;
}
else
{
lean_inc(v_fst_3810_);
lean_dec(v_b_3791_);
v___x_3812_ = lean_box(0);
v_isShared_3813_ = v_isSharedCheck_4019_;
goto v_resetjp_3811_;
}
v_resetjp_3811_:
{
lean_object* v_fst_3814_; lean_object* v___x_3816_; uint8_t v_isShared_3817_; uint8_t v_isSharedCheck_4017_; 
v_fst_3814_ = lean_ctor_get(v_snd_3806_, 0);
v_isSharedCheck_4017_ = !lean_is_exclusive(v_snd_3806_);
if (v_isSharedCheck_4017_ == 0)
{
lean_object* v_unused_4018_; 
v_unused_4018_ = lean_ctor_get(v_snd_3806_, 1);
lean_dec(v_unused_4018_);
v___x_3816_ = v_snd_3806_;
v_isShared_3817_ = v_isSharedCheck_4017_;
goto v_resetjp_3815_;
}
else
{
lean_inc(v_fst_3814_);
lean_dec(v_snd_3806_);
v___x_3816_ = lean_box(0);
v_isShared_3817_ = v_isSharedCheck_4017_;
goto v_resetjp_3815_;
}
v_resetjp_3815_:
{
lean_object* v_fst_3818_; lean_object* v___x_3820_; uint8_t v_isShared_3821_; uint8_t v_isSharedCheck_4015_; 
v_fst_3818_ = lean_ctor_get(v_snd_3807_, 0);
v_isSharedCheck_4015_ = !lean_is_exclusive(v_snd_3807_);
if (v_isSharedCheck_4015_ == 0)
{
lean_object* v_unused_4016_; 
v_unused_4016_ = lean_ctor_get(v_snd_3807_, 1);
lean_dec(v_unused_4016_);
v___x_3820_ = v_snd_3807_;
v_isShared_3821_ = v_isSharedCheck_4015_;
goto v_resetjp_3819_;
}
else
{
lean_inc(v_fst_3818_);
lean_dec(v_snd_3807_);
v___x_3820_ = lean_box(0);
v_isShared_3821_ = v_isSharedCheck_4015_;
goto v_resetjp_3819_;
}
v_resetjp_3819_:
{
lean_object* v_fst_3822_; lean_object* v___x_3824_; uint8_t v_isShared_3825_; uint8_t v_isSharedCheck_4013_; 
v_fst_3822_ = lean_ctor_get(v_snd_3808_, 0);
v_isSharedCheck_4013_ = !lean_is_exclusive(v_snd_3808_);
if (v_isSharedCheck_4013_ == 0)
{
lean_object* v_unused_4014_; 
v_unused_4014_ = lean_ctor_get(v_snd_3808_, 1);
lean_dec(v_unused_4014_);
v___x_3824_ = v_snd_3808_;
v_isShared_3825_ = v_isSharedCheck_4013_;
goto v_resetjp_3823_;
}
else
{
lean_inc(v_fst_3822_);
lean_dec(v_snd_3808_);
v___x_3824_ = lean_box(0);
v_isShared_3825_ = v_isSharedCheck_4013_;
goto v_resetjp_3823_;
}
v_resetjp_3823_:
{
lean_object* v_array_3826_; lean_object* v_start_3827_; lean_object* v_stop_3828_; uint8_t v___x_3829_; 
v_array_3826_ = lean_ctor_get(v_snd_3809_, 0);
v_start_3827_ = lean_ctor_get(v_snd_3809_, 1);
v_stop_3828_ = lean_ctor_get(v_snd_3809_, 2);
v___x_3829_ = lean_nat_dec_lt(v_start_3827_, v_stop_3828_);
if (v___x_3829_ == 0)
{
lean_object* v___x_3831_; 
lean_del_object(v___x_3802_);
lean_dec(v_stop_3800_);
lean_dec(v_start_3799_);
lean_dec_ref(v_array_3798_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
if (v_isShared_3825_ == 0)
{
v___x_3831_ = v___x_3824_;
goto v_reusejp_3830_;
}
else
{
lean_object* v_reuseFailAlloc_3842_; 
v_reuseFailAlloc_3842_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3842_, 0, v_fst_3822_);
lean_ctor_set(v_reuseFailAlloc_3842_, 1, v_snd_3809_);
v___x_3831_ = v_reuseFailAlloc_3842_;
goto v_reusejp_3830_;
}
v_reusejp_3830_:
{
lean_object* v___x_3833_; 
if (v_isShared_3821_ == 0)
{
lean_ctor_set(v___x_3820_, 1, v___x_3831_);
v___x_3833_ = v___x_3820_;
goto v_reusejp_3832_;
}
else
{
lean_object* v_reuseFailAlloc_3841_; 
v_reuseFailAlloc_3841_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3841_, 0, v_fst_3818_);
lean_ctor_set(v_reuseFailAlloc_3841_, 1, v___x_3831_);
v___x_3833_ = v_reuseFailAlloc_3841_;
goto v_reusejp_3832_;
}
v_reusejp_3832_:
{
lean_object* v___x_3835_; 
if (v_isShared_3817_ == 0)
{
lean_ctor_set(v___x_3816_, 1, v___x_3833_);
v___x_3835_ = v___x_3816_;
goto v_reusejp_3834_;
}
else
{
lean_object* v_reuseFailAlloc_3840_; 
v_reuseFailAlloc_3840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3840_, 0, v_fst_3814_);
lean_ctor_set(v_reuseFailAlloc_3840_, 1, v___x_3833_);
v___x_3835_ = v_reuseFailAlloc_3840_;
goto v_reusejp_3834_;
}
v_reusejp_3834_:
{
lean_object* v___x_3837_; 
if (v_isShared_3813_ == 0)
{
lean_ctor_set(v___x_3812_, 1, v___x_3835_);
v___x_3837_ = v___x_3812_;
goto v_reusejp_3836_;
}
else
{
lean_object* v_reuseFailAlloc_3839_; 
v_reuseFailAlloc_3839_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3839_, 0, v_fst_3810_);
lean_ctor_set(v_reuseFailAlloc_3839_, 1, v___x_3835_);
v___x_3837_ = v_reuseFailAlloc_3839_;
goto v_reusejp_3836_;
}
v_reusejp_3836_:
{
lean_object* v___x_3838_; 
v___x_3838_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3838_, 0, v___x_3837_);
return v___x_3838_;
}
}
}
}
}
else
{
lean_object* v___x_3844_; uint8_t v_isShared_3845_; uint8_t v_isSharedCheck_4009_; 
lean_inc(v_stop_3828_);
lean_inc(v_start_3827_);
lean_inc_ref(v_array_3826_);
v_isSharedCheck_4009_ = !lean_is_exclusive(v_snd_3809_);
if (v_isSharedCheck_4009_ == 0)
{
lean_object* v_unused_4010_; lean_object* v_unused_4011_; lean_object* v_unused_4012_; 
v_unused_4010_ = lean_ctor_get(v_snd_3809_, 2);
lean_dec(v_unused_4010_);
v_unused_4011_ = lean_ctor_get(v_snd_3809_, 1);
lean_dec(v_unused_4011_);
v_unused_4012_ = lean_ctor_get(v_snd_3809_, 0);
lean_dec(v_unused_4012_);
v___x_3844_ = v_snd_3809_;
v_isShared_3845_ = v_isSharedCheck_4009_;
goto v_resetjp_3843_;
}
else
{
lean_dec(v_snd_3809_);
v___x_3844_ = lean_box(0);
v_isShared_3845_ = v_isSharedCheck_4009_;
goto v_resetjp_3843_;
}
v_resetjp_3843_:
{
lean_object* v_array_3846_; lean_object* v_start_3847_; lean_object* v_stop_3848_; lean_object* v___x_3849_; lean_object* v___x_3850_; lean_object* v___x_3851_; lean_object* v___x_3853_; 
v_array_3846_ = lean_ctor_get(v_fst_3822_, 0);
v_start_3847_ = lean_ctor_get(v_fst_3822_, 1);
v_stop_3848_ = lean_ctor_get(v_fst_3822_, 2);
v___x_3849_ = lean_array_fget(v_array_3826_, v_start_3827_);
v___x_3850_ = lean_unsigned_to_nat(1u);
v___x_3851_ = lean_nat_add(v_start_3827_, v___x_3850_);
lean_dec(v_start_3827_);
if (v_isShared_3845_ == 0)
{
lean_ctor_set(v___x_3844_, 1, v___x_3851_);
v___x_3853_ = v___x_3844_;
goto v_reusejp_3852_;
}
else
{
lean_object* v_reuseFailAlloc_4008_; 
v_reuseFailAlloc_4008_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4008_, 0, v_array_3826_);
lean_ctor_set(v_reuseFailAlloc_4008_, 1, v___x_3851_);
lean_ctor_set(v_reuseFailAlloc_4008_, 2, v_stop_3828_);
v___x_3853_ = v_reuseFailAlloc_4008_;
goto v_reusejp_3852_;
}
v_reusejp_3852_:
{
uint8_t v___x_3854_; 
v___x_3854_ = lean_nat_dec_lt(v_start_3847_, v_stop_3848_);
if (v___x_3854_ == 0)
{
lean_object* v___x_3856_; 
lean_dec(v___x_3849_);
lean_del_object(v___x_3802_);
lean_dec(v_stop_3800_);
lean_dec(v_start_3799_);
lean_dec_ref(v_array_3798_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
if (v_isShared_3825_ == 0)
{
lean_ctor_set(v___x_3824_, 1, v___x_3853_);
v___x_3856_ = v___x_3824_;
goto v_reusejp_3855_;
}
else
{
lean_object* v_reuseFailAlloc_3867_; 
v_reuseFailAlloc_3867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3867_, 0, v_fst_3822_);
lean_ctor_set(v_reuseFailAlloc_3867_, 1, v___x_3853_);
v___x_3856_ = v_reuseFailAlloc_3867_;
goto v_reusejp_3855_;
}
v_reusejp_3855_:
{
lean_object* v___x_3858_; 
if (v_isShared_3821_ == 0)
{
lean_ctor_set(v___x_3820_, 1, v___x_3856_);
v___x_3858_ = v___x_3820_;
goto v_reusejp_3857_;
}
else
{
lean_object* v_reuseFailAlloc_3866_; 
v_reuseFailAlloc_3866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3866_, 0, v_fst_3818_);
lean_ctor_set(v_reuseFailAlloc_3866_, 1, v___x_3856_);
v___x_3858_ = v_reuseFailAlloc_3866_;
goto v_reusejp_3857_;
}
v_reusejp_3857_:
{
lean_object* v___x_3860_; 
if (v_isShared_3817_ == 0)
{
lean_ctor_set(v___x_3816_, 1, v___x_3858_);
v___x_3860_ = v___x_3816_;
goto v_reusejp_3859_;
}
else
{
lean_object* v_reuseFailAlloc_3865_; 
v_reuseFailAlloc_3865_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3865_, 0, v_fst_3814_);
lean_ctor_set(v_reuseFailAlloc_3865_, 1, v___x_3858_);
v___x_3860_ = v_reuseFailAlloc_3865_;
goto v_reusejp_3859_;
}
v_reusejp_3859_:
{
lean_object* v___x_3862_; 
if (v_isShared_3813_ == 0)
{
lean_ctor_set(v___x_3812_, 1, v___x_3860_);
v___x_3862_ = v___x_3812_;
goto v_reusejp_3861_;
}
else
{
lean_object* v_reuseFailAlloc_3864_; 
v_reuseFailAlloc_3864_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3864_, 0, v_fst_3810_);
lean_ctor_set(v_reuseFailAlloc_3864_, 1, v___x_3860_);
v___x_3862_ = v_reuseFailAlloc_3864_;
goto v_reusejp_3861_;
}
v_reusejp_3861_:
{
lean_object* v___x_3863_; 
v___x_3863_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3863_, 0, v___x_3862_);
return v___x_3863_;
}
}
}
}
}
else
{
lean_object* v___x_3869_; uint8_t v_isShared_3870_; uint8_t v_isSharedCheck_4004_; 
lean_inc(v_stop_3848_);
lean_inc(v_start_3847_);
lean_inc_ref(v_array_3846_);
v_isSharedCheck_4004_ = !lean_is_exclusive(v_fst_3822_);
if (v_isSharedCheck_4004_ == 0)
{
lean_object* v_unused_4005_; lean_object* v_unused_4006_; lean_object* v_unused_4007_; 
v_unused_4005_ = lean_ctor_get(v_fst_3822_, 2);
lean_dec(v_unused_4005_);
v_unused_4006_ = lean_ctor_get(v_fst_3822_, 1);
lean_dec(v_unused_4006_);
v_unused_4007_ = lean_ctor_get(v_fst_3822_, 0);
lean_dec(v_unused_4007_);
v___x_3869_ = v_fst_3822_;
v_isShared_3870_ = v_isSharedCheck_4004_;
goto v_resetjp_3868_;
}
else
{
lean_dec(v_fst_3822_);
v___x_3869_ = lean_box(0);
v_isShared_3870_ = v_isSharedCheck_4004_;
goto v_resetjp_3868_;
}
v_resetjp_3868_:
{
lean_object* v___x_3871_; lean_object* v___x_3873_; 
v___x_3871_ = lean_nat_add(v_start_3799_, v___x_3850_);
lean_inc_ref(v_array_3798_);
if (v_isShared_3870_ == 0)
{
lean_ctor_set(v___x_3869_, 2, v_stop_3800_);
lean_ctor_set(v___x_3869_, 1, v___x_3871_);
lean_ctor_set(v___x_3869_, 0, v_array_3798_);
v___x_3873_ = v___x_3869_;
goto v_reusejp_3872_;
}
else
{
lean_object* v_reuseFailAlloc_4003_; 
v_reuseFailAlloc_4003_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4003_, 0, v_array_3798_);
lean_ctor_set(v_reuseFailAlloc_4003_, 1, v___x_3871_);
lean_ctor_set(v_reuseFailAlloc_4003_, 2, v_stop_3800_);
v___x_3873_ = v_reuseFailAlloc_4003_;
goto v_reusejp_3872_;
}
v_reusejp_3872_:
{
lean_object* v___x_3874_; lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v___x_3878_; 
v___x_3874_ = lean_array_fget(v_array_3798_, v_start_3799_);
lean_dec(v_start_3799_);
lean_dec_ref(v_array_3798_);
v___x_3875_ = lean_array_fget(v_array_3846_, v_start_3847_);
v___x_3876_ = lean_nat_add(v_start_3847_, v___x_3850_);
lean_dec(v_start_3847_);
if (v_isShared_3803_ == 0)
{
lean_ctor_set(v___x_3802_, 2, v_stop_3848_);
lean_ctor_set(v___x_3802_, 1, v___x_3876_);
lean_ctor_set(v___x_3802_, 0, v_array_3846_);
v___x_3878_ = v___x_3802_;
goto v_reusejp_3877_;
}
else
{
lean_object* v_reuseFailAlloc_4002_; 
v_reuseFailAlloc_4002_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4002_, 0, v_array_3846_);
lean_ctor_set(v_reuseFailAlloc_4002_, 1, v___x_3876_);
lean_ctor_set(v_reuseFailAlloc_4002_, 2, v_stop_3848_);
v___x_3878_ = v_reuseFailAlloc_4002_;
goto v_reusejp_3877_;
}
v_reusejp_3877_:
{
uint8_t v___x_3879_; 
v___x_3879_ = lean_unbox(v___x_3849_);
lean_dec(v___x_3849_);
switch(v___x_3879_)
{
case 2:
{
lean_object* v___x_3880_; 
lean_inc(v_mvarCounterSaved_3789_);
lean_inc(v___x_3788_);
v___x_3880_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_3788_, v_mvarCounterSaved_3789_, v___x_3874_, v___x_3875_, v___y_3792_, v___y_3793_, v___y_3794_, v___y_3795_, v___y_3796_);
if (lean_obj_tag(v___x_3880_) == 0)
{
lean_object* v_a_3881_; lean_object* v___x_3882_; 
v_a_3881_ = lean_ctor_get(v___x_3880_, 0);
lean_inc_n(v_a_3881_, 2);
lean_dec_ref_known(v___x_3880_, 1);
v___x_3882_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_a_3881_, v___y_3793_, v___y_3794_, v___y_3795_, v___y_3796_);
if (lean_obj_tag(v___x_3882_) == 0)
{
lean_object* v_a_3883_; lean_object* v_lhs_3884_; lean_object* v_rhs_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; lean_object* v___x_3892_; 
v_a_3883_ = lean_ctor_get(v___x_3882_, 0);
lean_inc(v_a_3883_);
lean_dec_ref_known(v___x_3882_, 1);
v_lhs_3884_ = lean_ctor_get(v_a_3881_, 0);
lean_inc_ref_n(v_lhs_3884_, 2);
v_rhs_3885_ = lean_ctor_get(v_a_3881_, 1);
lean_inc_ref_n(v_rhs_3885_, 2);
lean_dec(v_a_3881_);
v___x_3886_ = lean_array_push(v_fst_3810_, v_lhs_3884_);
v___x_3887_ = lean_array_push(v___x_3886_, v_rhs_3885_);
v___x_3888_ = lean_array_push(v___x_3887_, v_a_3883_);
v___x_3889_ = lean_array_push(v_fst_3814_, v_lhs_3884_);
v___x_3890_ = lean_array_push(v_fst_3818_, v_rhs_3885_);
if (v_isShared_3825_ == 0)
{
lean_ctor_set(v___x_3824_, 1, v___x_3853_);
lean_ctor_set(v___x_3824_, 0, v___x_3878_);
v___x_3892_ = v___x_3824_;
goto v_reusejp_3891_;
}
else
{
lean_object* v_reuseFailAlloc_3903_; 
v_reuseFailAlloc_3903_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3903_, 0, v___x_3878_);
lean_ctor_set(v_reuseFailAlloc_3903_, 1, v___x_3853_);
v___x_3892_ = v_reuseFailAlloc_3903_;
goto v_reusejp_3891_;
}
v_reusejp_3891_:
{
lean_object* v___x_3894_; 
if (v_isShared_3821_ == 0)
{
lean_ctor_set(v___x_3820_, 1, v___x_3892_);
lean_ctor_set(v___x_3820_, 0, v___x_3890_);
v___x_3894_ = v___x_3820_;
goto v_reusejp_3893_;
}
else
{
lean_object* v_reuseFailAlloc_3902_; 
v_reuseFailAlloc_3902_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3902_, 0, v___x_3890_);
lean_ctor_set(v_reuseFailAlloc_3902_, 1, v___x_3892_);
v___x_3894_ = v_reuseFailAlloc_3902_;
goto v_reusejp_3893_;
}
v_reusejp_3893_:
{
lean_object* v___x_3896_; 
if (v_isShared_3817_ == 0)
{
lean_ctor_set(v___x_3816_, 1, v___x_3894_);
lean_ctor_set(v___x_3816_, 0, v___x_3889_);
v___x_3896_ = v___x_3816_;
goto v_reusejp_3895_;
}
else
{
lean_object* v_reuseFailAlloc_3901_; 
v_reuseFailAlloc_3901_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3901_, 0, v___x_3889_);
lean_ctor_set(v_reuseFailAlloc_3901_, 1, v___x_3894_);
v___x_3896_ = v_reuseFailAlloc_3901_;
goto v_reusejp_3895_;
}
v_reusejp_3895_:
{
lean_object* v___x_3898_; 
if (v_isShared_3813_ == 0)
{
lean_ctor_set(v___x_3812_, 1, v___x_3896_);
lean_ctor_set(v___x_3812_, 0, v___x_3888_);
v___x_3898_ = v___x_3812_;
goto v_reusejp_3897_;
}
else
{
lean_object* v_reuseFailAlloc_3900_; 
v_reuseFailAlloc_3900_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3900_, 0, v___x_3888_);
lean_ctor_set(v_reuseFailAlloc_3900_, 1, v___x_3896_);
v___x_3898_ = v_reuseFailAlloc_3900_;
goto v_reusejp_3897_;
}
v_reusejp_3897_:
{
v_a_3790_ = v___x_3873_;
v_b_3791_ = v___x_3898_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_3904_; lean_object* v___x_3906_; uint8_t v_isShared_3907_; uint8_t v_isSharedCheck_3911_; 
lean_dec(v_a_3881_);
lean_dec_ref(v___x_3878_);
lean_dec_ref(v___x_3873_);
lean_dec_ref(v___x_3853_);
lean_del_object(v___x_3824_);
lean_del_object(v___x_3820_);
lean_dec(v_fst_3818_);
lean_del_object(v___x_3816_);
lean_dec(v_fst_3814_);
lean_del_object(v___x_3812_);
lean_dec(v_fst_3810_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
v_a_3904_ = lean_ctor_get(v___x_3882_, 0);
v_isSharedCheck_3911_ = !lean_is_exclusive(v___x_3882_);
if (v_isSharedCheck_3911_ == 0)
{
v___x_3906_ = v___x_3882_;
v_isShared_3907_ = v_isSharedCheck_3911_;
goto v_resetjp_3905_;
}
else
{
lean_inc(v_a_3904_);
lean_dec(v___x_3882_);
v___x_3906_ = lean_box(0);
v_isShared_3907_ = v_isSharedCheck_3911_;
goto v_resetjp_3905_;
}
v_resetjp_3905_:
{
lean_object* v___x_3909_; 
if (v_isShared_3907_ == 0)
{
v___x_3909_ = v___x_3906_;
goto v_reusejp_3908_;
}
else
{
lean_object* v_reuseFailAlloc_3910_; 
v_reuseFailAlloc_3910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3910_, 0, v_a_3904_);
v___x_3909_ = v_reuseFailAlloc_3910_;
goto v_reusejp_3908_;
}
v_reusejp_3908_:
{
return v___x_3909_;
}
}
}
}
else
{
lean_object* v_a_3912_; lean_object* v___x_3914_; uint8_t v_isShared_3915_; uint8_t v_isSharedCheck_3919_; 
lean_dec_ref(v___x_3878_);
lean_dec_ref(v___x_3873_);
lean_dec_ref(v___x_3853_);
lean_del_object(v___x_3824_);
lean_del_object(v___x_3820_);
lean_dec(v_fst_3818_);
lean_del_object(v___x_3816_);
lean_dec(v_fst_3814_);
lean_del_object(v___x_3812_);
lean_dec(v_fst_3810_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
v_a_3912_ = lean_ctor_get(v___x_3880_, 0);
v_isSharedCheck_3919_ = !lean_is_exclusive(v___x_3880_);
if (v_isSharedCheck_3919_ == 0)
{
v___x_3914_ = v___x_3880_;
v_isShared_3915_ = v_isSharedCheck_3919_;
goto v_resetjp_3913_;
}
else
{
lean_inc(v_a_3912_);
lean_dec(v___x_3880_);
v___x_3914_ = lean_box(0);
v_isShared_3915_ = v_isSharedCheck_3919_;
goto v_resetjp_3913_;
}
v_resetjp_3913_:
{
lean_object* v___x_3917_; 
if (v_isShared_3915_ == 0)
{
v___x_3917_ = v___x_3914_;
goto v_reusejp_3916_;
}
else
{
lean_object* v_reuseFailAlloc_3918_; 
v_reuseFailAlloc_3918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3918_, 0, v_a_3912_);
v___x_3917_ = v_reuseFailAlloc_3918_;
goto v_reusejp_3916_;
}
v_reusejp_3916_:
{
return v___x_3917_;
}
}
}
}
case 4:
{
lean_object* v___x_3920_; 
lean_inc(v_mvarCounterSaved_3789_);
lean_inc(v___x_3788_);
v___x_3920_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_3788_, v_mvarCounterSaved_3789_, v___x_3874_, v___x_3875_, v___y_3792_, v___y_3793_, v___y_3794_, v___y_3795_, v___y_3796_);
if (lean_obj_tag(v___x_3920_) == 0)
{
lean_object* v_a_3921_; lean_object* v___x_3922_; 
v_a_3921_ = lean_ctor_get(v___x_3920_, 0);
lean_inc_n(v_a_3921_, 2);
lean_dec_ref_known(v___x_3920_, 1);
v___x_3922_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq(v_a_3921_, v___y_3793_, v___y_3794_, v___y_3795_, v___y_3796_);
if (lean_obj_tag(v___x_3922_) == 0)
{
lean_object* v_a_3923_; lean_object* v_lhs_3924_; lean_object* v_rhs_3925_; lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; lean_object* v___x_3932_; 
v_a_3923_ = lean_ctor_get(v___x_3922_, 0);
lean_inc(v_a_3923_);
lean_dec_ref_known(v___x_3922_, 1);
v_lhs_3924_ = lean_ctor_get(v_a_3921_, 0);
lean_inc_ref_n(v_lhs_3924_, 2);
v_rhs_3925_ = lean_ctor_get(v_a_3921_, 1);
lean_inc_ref_n(v_rhs_3925_, 2);
lean_dec(v_a_3921_);
v___x_3926_ = lean_array_push(v_fst_3810_, v_lhs_3924_);
v___x_3927_ = lean_array_push(v___x_3926_, v_rhs_3925_);
v___x_3928_ = lean_array_push(v___x_3927_, v_a_3923_);
v___x_3929_ = lean_array_push(v_fst_3814_, v_lhs_3924_);
v___x_3930_ = lean_array_push(v_fst_3818_, v_rhs_3925_);
if (v_isShared_3825_ == 0)
{
lean_ctor_set(v___x_3824_, 1, v___x_3853_);
lean_ctor_set(v___x_3824_, 0, v___x_3878_);
v___x_3932_ = v___x_3824_;
goto v_reusejp_3931_;
}
else
{
lean_object* v_reuseFailAlloc_3943_; 
v_reuseFailAlloc_3943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3943_, 0, v___x_3878_);
lean_ctor_set(v_reuseFailAlloc_3943_, 1, v___x_3853_);
v___x_3932_ = v_reuseFailAlloc_3943_;
goto v_reusejp_3931_;
}
v_reusejp_3931_:
{
lean_object* v___x_3934_; 
if (v_isShared_3821_ == 0)
{
lean_ctor_set(v___x_3820_, 1, v___x_3932_);
lean_ctor_set(v___x_3820_, 0, v___x_3930_);
v___x_3934_ = v___x_3820_;
goto v_reusejp_3933_;
}
else
{
lean_object* v_reuseFailAlloc_3942_; 
v_reuseFailAlloc_3942_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3942_, 0, v___x_3930_);
lean_ctor_set(v_reuseFailAlloc_3942_, 1, v___x_3932_);
v___x_3934_ = v_reuseFailAlloc_3942_;
goto v_reusejp_3933_;
}
v_reusejp_3933_:
{
lean_object* v___x_3936_; 
if (v_isShared_3817_ == 0)
{
lean_ctor_set(v___x_3816_, 1, v___x_3934_);
lean_ctor_set(v___x_3816_, 0, v___x_3929_);
v___x_3936_ = v___x_3816_;
goto v_reusejp_3935_;
}
else
{
lean_object* v_reuseFailAlloc_3941_; 
v_reuseFailAlloc_3941_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3941_, 0, v___x_3929_);
lean_ctor_set(v_reuseFailAlloc_3941_, 1, v___x_3934_);
v___x_3936_ = v_reuseFailAlloc_3941_;
goto v_reusejp_3935_;
}
v_reusejp_3935_:
{
lean_object* v___x_3938_; 
if (v_isShared_3813_ == 0)
{
lean_ctor_set(v___x_3812_, 1, v___x_3936_);
lean_ctor_set(v___x_3812_, 0, v___x_3928_);
v___x_3938_ = v___x_3812_;
goto v_reusejp_3937_;
}
else
{
lean_object* v_reuseFailAlloc_3940_; 
v_reuseFailAlloc_3940_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3940_, 0, v___x_3928_);
lean_ctor_set(v_reuseFailAlloc_3940_, 1, v___x_3936_);
v___x_3938_ = v_reuseFailAlloc_3940_;
goto v_reusejp_3937_;
}
v_reusejp_3937_:
{
v_a_3790_ = v___x_3873_;
v_b_3791_ = v___x_3938_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_3944_; lean_object* v___x_3946_; uint8_t v_isShared_3947_; uint8_t v_isSharedCheck_3951_; 
lean_dec(v_a_3921_);
lean_dec_ref(v___x_3878_);
lean_dec_ref(v___x_3873_);
lean_dec_ref(v___x_3853_);
lean_del_object(v___x_3824_);
lean_del_object(v___x_3820_);
lean_dec(v_fst_3818_);
lean_del_object(v___x_3816_);
lean_dec(v_fst_3814_);
lean_del_object(v___x_3812_);
lean_dec(v_fst_3810_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
v_a_3944_ = lean_ctor_get(v___x_3922_, 0);
v_isSharedCheck_3951_ = !lean_is_exclusive(v___x_3922_);
if (v_isSharedCheck_3951_ == 0)
{
v___x_3946_ = v___x_3922_;
v_isShared_3947_ = v_isSharedCheck_3951_;
goto v_resetjp_3945_;
}
else
{
lean_inc(v_a_3944_);
lean_dec(v___x_3922_);
v___x_3946_ = lean_box(0);
v_isShared_3947_ = v_isSharedCheck_3951_;
goto v_resetjp_3945_;
}
v_resetjp_3945_:
{
lean_object* v___x_3949_; 
if (v_isShared_3947_ == 0)
{
v___x_3949_ = v___x_3946_;
goto v_reusejp_3948_;
}
else
{
lean_object* v_reuseFailAlloc_3950_; 
v_reuseFailAlloc_3950_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3950_, 0, v_a_3944_);
v___x_3949_ = v_reuseFailAlloc_3950_;
goto v_reusejp_3948_;
}
v_reusejp_3948_:
{
return v___x_3949_;
}
}
}
}
else
{
lean_object* v_a_3952_; lean_object* v___x_3954_; uint8_t v_isShared_3955_; uint8_t v_isSharedCheck_3959_; 
lean_dec_ref(v___x_3878_);
lean_dec_ref(v___x_3873_);
lean_dec_ref(v___x_3853_);
lean_del_object(v___x_3824_);
lean_del_object(v___x_3820_);
lean_dec(v_fst_3818_);
lean_del_object(v___x_3816_);
lean_dec(v_fst_3814_);
lean_del_object(v___x_3812_);
lean_dec(v_fst_3810_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
v_a_3952_ = lean_ctor_get(v___x_3920_, 0);
v_isSharedCheck_3959_ = !lean_is_exclusive(v___x_3920_);
if (v_isSharedCheck_3959_ == 0)
{
v___x_3954_ = v___x_3920_;
v_isShared_3955_ = v_isSharedCheck_3959_;
goto v_resetjp_3953_;
}
else
{
lean_inc(v_a_3952_);
lean_dec(v___x_3920_);
v___x_3954_ = lean_box(0);
v_isShared_3955_ = v_isSharedCheck_3959_;
goto v_resetjp_3953_;
}
v_resetjp_3953_:
{
lean_object* v___x_3957_; 
if (v_isShared_3955_ == 0)
{
v___x_3957_ = v___x_3954_;
goto v_reusejp_3956_;
}
else
{
lean_object* v_reuseFailAlloc_3958_; 
v_reuseFailAlloc_3958_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3958_, 0, v_a_3952_);
v___x_3957_ = v_reuseFailAlloc_3958_;
goto v_reusejp_3956_;
}
v_reusejp_3956_:
{
return v___x_3957_;
}
}
}
}
case 5:
{
lean_object* v___x_3960_; lean_object* v___x_3961_; lean_object* v___x_3962_; lean_object* v___x_3963_; lean_object* v___x_3964_; lean_object* v___x_3965_; lean_object* v___x_3967_; 
v___x_3960_ = lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(v___x_3874_);
lean_dec(v___x_3874_);
v___x_3961_ = lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(v___x_3875_);
lean_dec(v___x_3875_);
lean_inc_ref(v___x_3960_);
v___x_3962_ = lean_array_push(v_fst_3810_, v___x_3960_);
lean_inc_ref(v___x_3961_);
v___x_3963_ = lean_array_push(v___x_3962_, v___x_3961_);
v___x_3964_ = lean_array_push(v_fst_3814_, v___x_3960_);
v___x_3965_ = lean_array_push(v_fst_3818_, v___x_3961_);
if (v_isShared_3825_ == 0)
{
lean_ctor_set(v___x_3824_, 1, v___x_3853_);
lean_ctor_set(v___x_3824_, 0, v___x_3878_);
v___x_3967_ = v___x_3824_;
goto v_reusejp_3966_;
}
else
{
lean_object* v_reuseFailAlloc_3978_; 
v_reuseFailAlloc_3978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3978_, 0, v___x_3878_);
lean_ctor_set(v_reuseFailAlloc_3978_, 1, v___x_3853_);
v___x_3967_ = v_reuseFailAlloc_3978_;
goto v_reusejp_3966_;
}
v_reusejp_3966_:
{
lean_object* v___x_3969_; 
if (v_isShared_3821_ == 0)
{
lean_ctor_set(v___x_3820_, 1, v___x_3967_);
lean_ctor_set(v___x_3820_, 0, v___x_3965_);
v___x_3969_ = v___x_3820_;
goto v_reusejp_3968_;
}
else
{
lean_object* v_reuseFailAlloc_3977_; 
v_reuseFailAlloc_3977_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3977_, 0, v___x_3965_);
lean_ctor_set(v_reuseFailAlloc_3977_, 1, v___x_3967_);
v___x_3969_ = v_reuseFailAlloc_3977_;
goto v_reusejp_3968_;
}
v_reusejp_3968_:
{
lean_object* v___x_3971_; 
if (v_isShared_3817_ == 0)
{
lean_ctor_set(v___x_3816_, 1, v___x_3969_);
lean_ctor_set(v___x_3816_, 0, v___x_3964_);
v___x_3971_ = v___x_3816_;
goto v_reusejp_3970_;
}
else
{
lean_object* v_reuseFailAlloc_3976_; 
v_reuseFailAlloc_3976_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3976_, 0, v___x_3964_);
lean_ctor_set(v_reuseFailAlloc_3976_, 1, v___x_3969_);
v___x_3971_ = v_reuseFailAlloc_3976_;
goto v_reusejp_3970_;
}
v_reusejp_3970_:
{
lean_object* v___x_3973_; 
if (v_isShared_3813_ == 0)
{
lean_ctor_set(v___x_3812_, 1, v___x_3971_);
lean_ctor_set(v___x_3812_, 0, v___x_3963_);
v___x_3973_ = v___x_3812_;
goto v_reusejp_3972_;
}
else
{
lean_object* v_reuseFailAlloc_3975_; 
v_reuseFailAlloc_3975_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3975_, 0, v___x_3963_);
lean_ctor_set(v_reuseFailAlloc_3975_, 1, v___x_3971_);
v___x_3973_ = v_reuseFailAlloc_3975_;
goto v_reusejp_3972_;
}
v_reusejp_3972_:
{
v_a_3790_ = v___x_3873_;
v_b_3791_ = v___x_3973_;
goto _start;
}
}
}
}
}
default: 
{
lean_object* v___x_3979_; lean_object* v___x_3980_; 
lean_dec(v___x_3875_);
lean_dec(v___x_3874_);
v___x_3979_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__2, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__2_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___closed__2);
v___x_3980_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__2(v___x_3979_, v___y_3792_, v___y_3793_, v___y_3794_, v___y_3795_, v___y_3796_);
if (lean_obj_tag(v___x_3980_) == 0)
{
lean_object* v___x_3982_; 
lean_dec_ref_known(v___x_3980_, 1);
if (v_isShared_3825_ == 0)
{
lean_ctor_set(v___x_3824_, 1, v___x_3853_);
lean_ctor_set(v___x_3824_, 0, v___x_3878_);
v___x_3982_ = v___x_3824_;
goto v_reusejp_3981_;
}
else
{
lean_object* v_reuseFailAlloc_3993_; 
v_reuseFailAlloc_3993_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3993_, 0, v___x_3878_);
lean_ctor_set(v_reuseFailAlloc_3993_, 1, v___x_3853_);
v___x_3982_ = v_reuseFailAlloc_3993_;
goto v_reusejp_3981_;
}
v_reusejp_3981_:
{
lean_object* v___x_3984_; 
if (v_isShared_3821_ == 0)
{
lean_ctor_set(v___x_3820_, 1, v___x_3982_);
v___x_3984_ = v___x_3820_;
goto v_reusejp_3983_;
}
else
{
lean_object* v_reuseFailAlloc_3992_; 
v_reuseFailAlloc_3992_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3992_, 0, v_fst_3818_);
lean_ctor_set(v_reuseFailAlloc_3992_, 1, v___x_3982_);
v___x_3984_ = v_reuseFailAlloc_3992_;
goto v_reusejp_3983_;
}
v_reusejp_3983_:
{
lean_object* v___x_3986_; 
if (v_isShared_3817_ == 0)
{
lean_ctor_set(v___x_3816_, 1, v___x_3984_);
v___x_3986_ = v___x_3816_;
goto v_reusejp_3985_;
}
else
{
lean_object* v_reuseFailAlloc_3991_; 
v_reuseFailAlloc_3991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3991_, 0, v_fst_3814_);
lean_ctor_set(v_reuseFailAlloc_3991_, 1, v___x_3984_);
v___x_3986_ = v_reuseFailAlloc_3991_;
goto v_reusejp_3985_;
}
v_reusejp_3985_:
{
lean_object* v___x_3988_; 
if (v_isShared_3813_ == 0)
{
lean_ctor_set(v___x_3812_, 1, v___x_3986_);
v___x_3988_ = v___x_3812_;
goto v_reusejp_3987_;
}
else
{
lean_object* v_reuseFailAlloc_3990_; 
v_reuseFailAlloc_3990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3990_, 0, v_fst_3810_);
lean_ctor_set(v_reuseFailAlloc_3990_, 1, v___x_3986_);
v___x_3988_ = v_reuseFailAlloc_3990_;
goto v_reusejp_3987_;
}
v_reusejp_3987_:
{
v_a_3790_ = v___x_3873_;
v_b_3791_ = v___x_3988_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_3994_; lean_object* v___x_3996_; uint8_t v_isShared_3997_; uint8_t v_isSharedCheck_4001_; 
lean_dec_ref(v___x_3878_);
lean_dec_ref(v___x_3873_);
lean_dec_ref(v___x_3853_);
lean_del_object(v___x_3824_);
lean_del_object(v___x_3820_);
lean_dec(v_fst_3818_);
lean_del_object(v___x_3816_);
lean_dec(v_fst_3814_);
lean_del_object(v___x_3812_);
lean_dec(v_fst_3810_);
lean_dec(v_mvarCounterSaved_3789_);
lean_dec(v___x_3788_);
v_a_3994_ = lean_ctor_get(v___x_3980_, 0);
v_isSharedCheck_4001_ = !lean_is_exclusive(v___x_3980_);
if (v_isSharedCheck_4001_ == 0)
{
v___x_3996_ = v___x_3980_;
v_isShared_3997_ = v_isSharedCheck_4001_;
goto v_resetjp_3995_;
}
else
{
lean_inc(v_a_3994_);
lean_dec(v___x_3980_);
v___x_3996_ = lean_box(0);
v_isShared_3997_ = v_isSharedCheck_4001_;
goto v_resetjp_3995_;
}
v_resetjp_3995_:
{
lean_object* v___x_3999_; 
if (v_isShared_3997_ == 0)
{
v___x_3999_ = v___x_3996_;
goto v_reusejp_3998_;
}
else
{
lean_object* v_reuseFailAlloc_4000_; 
v_reuseFailAlloc_4000_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4000_, 0, v_a_3994_);
v___x_3999_ = v_reuseFailAlloc_4000_;
goto v_reusejp_3998_;
}
v_reusejp_3998_:
{
return v___x_3999_;
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
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__2(void){
_start:
{
lean_object* v___x_4023_; lean_object* v___x_4024_; 
v___x_4023_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__1));
v___x_4024_ = l_Lean_stringToMessageData(v___x_4023_);
return v___x_4024_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__4(void){
_start:
{
lean_object* v___x_4026_; lean_object* v___x_4027_; 
v___x_4026_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__3));
v___x_4027_ = l_Lean_stringToMessageData(v___x_4026_);
return v___x_4027_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__6(void){
_start:
{
lean_object* v___x_4029_; lean_object* v___x_4030_; 
v___x_4029_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__5));
v___x_4030_ = l_Lean_stringToMessageData(v___x_4029_);
return v___x_4030_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__8(void){
_start:
{
lean_object* v___x_4032_; lean_object* v___x_4033_; 
v___x_4032_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__7));
v___x_4033_ = l_Lean_stringToMessageData(v___x_4032_);
return v___x_4033_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10(void){
_start:
{
lean_object* v___x_4035_; lean_object* v___x_4036_; 
v___x_4035_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__9));
v___x_4036_ = l_Lean_stringToMessageData(v___x_4035_);
return v___x_4036_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__12(void){
_start:
{
lean_object* v___x_4038_; lean_object* v___x_4039_; 
v___x_4038_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__11));
v___x_4039_ = l_Lean_stringToMessageData(v___x_4038_);
return v___x_4039_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__15(void){
_start:
{
lean_object* v___x_4042_; lean_object* v___x_4043_; 
v___x_4042_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__14));
v___x_4043_ = l_Lean_stringToMessageData(v___x_4042_);
return v___x_4043_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__17(void){
_start:
{
lean_object* v___x_4045_; lean_object* v___x_4046_; 
v___x_4045_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__16));
v___x_4046_ = l_Lean_stringToMessageData(v___x_4045_);
return v___x_4046_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go(lean_object* v_depth_4047_, lean_object* v_mvarCounterSaved_4048_, lean_object* v_arity_4049_, lean_object* v_lhsArgs_4050_, lean_object* v_rhsArgs_4051_, lean_object* v_i_4052_, lean_object* v_finfo_4053_, lean_object* v_finfoIdx_4054_, lean_object* v_f_4055_, lean_object* v_f_x27_4056_, lean_object* v_pf_4057_, lean_object* v_a_4058_, lean_object* v_a_4059_, lean_object* v_a_4060_, lean_object* v_a_4061_, lean_object* v_a_4062_){
_start:
{
lean_object* v___y_4065_; lean_object* v___y_4066_; lean_object* v___y_4067_; lean_object* v___y_4068_; lean_object* v___y_4069_; lean_object* v___y_4070_; lean_object* v___y_4071_; lean_object* v___y_4112_; lean_object* v___y_4113_; lean_object* v___y_4114_; lean_object* v___y_4115_; lean_object* v___y_4116_; lean_object* v___y_4117_; lean_object* v___y_4118_; lean_object* v___y_4119_; lean_object* v___y_4120_; lean_object* v___y_4185_; lean_object* v___y_4186_; lean_object* v___y_4187_; lean_object* v___y_4188_; lean_object* v_fArity_4189_; lean_object* v___y_4190_; lean_object* v___y_4191_; lean_object* v___y_4192_; lean_object* v___y_4193_; lean_object* v___y_4194_; lean_object* v___y_4232_; lean_object* v___y_4233_; lean_object* v___y_4234_; lean_object* v___y_4235_; lean_object* v___y_4236_; lean_object* v___y_4237_; lean_object* v___y_4238_; lean_object* v___y_4239_; lean_object* v___y_4240_; lean_object* v___y_4268_; lean_object* v___y_4269_; lean_object* v___y_4270_; lean_object* v___y_4271_; lean_object* v___y_4272_; lean_object* v___y_4273_; lean_object* v___y_4274_; lean_object* v___y_4275_; lean_object* v___y_4276_; lean_object* v___y_4300_; lean_object* v___y_4301_; lean_object* v___y_4302_; lean_object* v___y_4303_; lean_object* v___y_4304_; lean_object* v___y_4305_; lean_object* v___y_4306_; lean_object* v___y_4307_; lean_object* v___y_4308_; uint8_t v___x_4325_; 
v___x_4325_ = lean_nat_dec_le(v_arity_4049_, v_i_4052_);
if (v___x_4325_ == 0)
{
lean_object* v___f_4326_; uint8_t v___y_4328_; lean_object* v___y_4329_; lean_object* v___y_4330_; lean_object* v___y_4331_; lean_object* v___y_4332_; lean_object* v___y_4333_; lean_object* v___y_4334_; lean_object* v___y_4335_; lean_object* v___y_4336_; lean_object* v___y_4337_; uint8_t v___y_4338_; lean_object* v_finfo_4375_; lean_object* v_finfoIdx_4376_; lean_object* v___y_4377_; lean_object* v___y_4378_; lean_object* v___y_4379_; lean_object* v___y_4380_; lean_object* v___y_4381_; lean_object* v___x_4446_; lean_object* v___x_4447_; uint8_t v___x_4448_; 
v___f_4326_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__13));
v___x_4446_ = lean_nat_sub(v_i_4052_, v_finfoIdx_4054_);
v___x_4447_ = l_Lean_Meta_FunInfo_getArity(v_finfo_4053_);
v___x_4448_ = lean_nat_dec_lt(v___x_4446_, v___x_4447_);
lean_dec(v___x_4447_);
lean_dec(v___x_4446_);
if (v___x_4448_ == 0)
{
lean_object* v___x_4449_; lean_object* v___x_4450_; 
lean_dec_ref(v_finfo_4053_);
v___x_4449_ = lean_nat_sub(v_arity_4049_, v_finfoIdx_4054_);
lean_dec(v_finfoIdx_4054_);
lean_inc_ref(v_f_4055_);
v___x_4450_ = l_Lean_Meta_getFunInfoNArgs(v_f_4055_, v___x_4449_, v_a_4059_, v_a_4060_, v_a_4061_, v_a_4062_);
if (lean_obj_tag(v___x_4450_) == 0)
{
lean_object* v_a_4451_; 
v_a_4451_ = lean_ctor_get(v___x_4450_, 0);
lean_inc(v_a_4451_);
lean_dec_ref_known(v___x_4450_, 1);
lean_inc(v_i_4052_);
v_finfo_4375_ = v_a_4451_;
v_finfoIdx_4376_ = v_i_4052_;
v___y_4377_ = v_a_4058_;
v___y_4378_ = v_a_4059_;
v___y_4379_ = v_a_4060_;
v___y_4380_ = v_a_4061_;
v___y_4381_ = v_a_4062_;
goto v___jp_4374_;
}
else
{
lean_object* v_a_4452_; lean_object* v___x_4454_; uint8_t v_isShared_4455_; uint8_t v_isSharedCheck_4459_; 
lean_dec_ref(v_pf_4057_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4452_ = lean_ctor_get(v___x_4450_, 0);
v_isSharedCheck_4459_ = !lean_is_exclusive(v___x_4450_);
if (v_isSharedCheck_4459_ == 0)
{
v___x_4454_ = v___x_4450_;
v_isShared_4455_ = v_isSharedCheck_4459_;
goto v_resetjp_4453_;
}
else
{
lean_inc(v_a_4452_);
lean_dec(v___x_4450_);
v___x_4454_ = lean_box(0);
v_isShared_4455_ = v_isSharedCheck_4459_;
goto v_resetjp_4453_;
}
v_resetjp_4453_:
{
lean_object* v___x_4457_; 
if (v_isShared_4455_ == 0)
{
v___x_4457_ = v___x_4454_;
goto v_reusejp_4456_;
}
else
{
lean_object* v_reuseFailAlloc_4458_; 
v_reuseFailAlloc_4458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4458_, 0, v_a_4452_);
v___x_4457_ = v_reuseFailAlloc_4458_;
goto v_reusejp_4456_;
}
v_reusejp_4456_:
{
return v___x_4457_;
}
}
}
}
else
{
v_finfo_4375_ = v_finfo_4053_;
v_finfoIdx_4376_ = v_finfoIdx_4054_;
v___y_4377_ = v_a_4058_;
v___y_4378_ = v_a_4059_;
v___y_4379_ = v_a_4060_;
v___y_4380_ = v_a_4061_;
v___y_4381_ = v_a_4062_;
goto v___jp_4374_;
}
v___jp_4327_:
{
lean_object* v_keyedConfig_4339_; uint8_t v_trackZetaDelta_4340_; lean_object* v_zetaDeltaSet_4341_; lean_object* v_lctx_4342_; lean_object* v_localInstances_4343_; lean_object* v_defEqCtx_x3f_4344_; lean_object* v_synthPendingDepth_4345_; lean_object* v_customCanUnfoldPredicate_x3f_4346_; uint8_t v_univApprox_4347_; uint8_t v_inTypeClassResolution_4348_; uint8_t v_cacheInferType_4349_; lean_object* v___x_4350_; lean_object* v___x_4351_; lean_object* v___x_4352_; 
v_keyedConfig_4339_ = lean_ctor_get(v___y_4334_, 0);
v_trackZetaDelta_4340_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*7);
v_zetaDeltaSet_4341_ = lean_ctor_get(v___y_4334_, 1);
v_lctx_4342_ = lean_ctor_get(v___y_4334_, 2);
v_localInstances_4343_ = lean_ctor_get(v___y_4334_, 3);
v_defEqCtx_x3f_4344_ = lean_ctor_get(v___y_4334_, 4);
v_synthPendingDepth_4345_ = lean_ctor_get(v___y_4334_, 5);
v_customCanUnfoldPredicate_x3f_4346_ = lean_ctor_get(v___y_4334_, 6);
v_univApprox_4347_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_4348_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*7 + 2);
v_cacheInferType_4349_ = lean_ctor_get_uint8(v___y_4334_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_4339_);
v___x_4350_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___y_4338_, v_keyedConfig_4339_);
lean_inc(v_customCanUnfoldPredicate_x3f_4346_);
lean_inc(v_synthPendingDepth_4345_);
lean_inc(v_defEqCtx_x3f_4344_);
lean_inc_ref(v_localInstances_4343_);
lean_inc_ref(v_lctx_4342_);
lean_inc(v_zetaDeltaSet_4341_);
v___x_4351_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_4351_, 0, v___x_4350_);
lean_ctor_set(v___x_4351_, 1, v_zetaDeltaSet_4341_);
lean_ctor_set(v___x_4351_, 2, v_lctx_4342_);
lean_ctor_set(v___x_4351_, 3, v_localInstances_4343_);
lean_ctor_set(v___x_4351_, 4, v_defEqCtx_x3f_4344_);
lean_ctor_set(v___x_4351_, 5, v_synthPendingDepth_4345_);
lean_ctor_set(v___x_4351_, 6, v_customCanUnfoldPredicate_x3f_4346_);
lean_ctor_set_uint8(v___x_4351_, sizeof(void*)*7, v_trackZetaDelta_4340_);
lean_ctor_set_uint8(v___x_4351_, sizeof(void*)*7 + 1, v_univApprox_4347_);
lean_ctor_set_uint8(v___x_4351_, sizeof(void*)*7 + 2, v_inTypeClassResolution_4348_);
lean_ctor_set_uint8(v___x_4351_, sizeof(void*)*7 + 3, v_cacheInferType_4349_);
lean_inc(v___y_4332_);
lean_inc_ref(v___y_4336_);
lean_inc(v___y_4329_);
lean_inc_ref(v___x_4351_);
lean_inc_ref(v_f_4055_);
v___x_4352_ = lean_infer_type(v_f_4055_, v___x_4351_, v___y_4329_, v___y_4336_, v___y_4332_);
if (lean_obj_tag(v___x_4352_) == 0)
{
lean_object* v_a_4353_; lean_object* v___x_4354_; lean_object* v___x_4355_; lean_object* v___x_4356_; 
v_a_4353_ = lean_ctor_get(v___x_4352_, 0);
lean_inc(v_a_4353_);
lean_dec_ref_known(v___x_4352_, 1);
v___x_4354_ = lean_nat_sub(v_arity_4049_, v_i_4052_);
v___x_4355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4355_, 0, v___x_4354_);
v___x_4356_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__5___redArg(v_a_4353_, v___x_4355_, v___f_4326_, v___y_4328_, v___y_4328_, v___y_4333_, v___x_4351_, v___y_4329_, v___y_4336_, v___y_4332_);
lean_dec_ref_known(v___x_4351_, 7);
if (lean_obj_tag(v___x_4356_) == 0)
{
lean_object* v_a_4357_; 
v_a_4357_ = lean_ctor_get(v___x_4356_, 0);
lean_inc(v_a_4357_);
lean_dec_ref_known(v___x_4356_, 1);
v___y_4185_ = v___y_4330_;
v___y_4186_ = v___y_4331_;
v___y_4187_ = v___y_4335_;
v___y_4188_ = v___y_4337_;
v_fArity_4189_ = v_a_4357_;
v___y_4190_ = v___y_4333_;
v___y_4191_ = v___y_4334_;
v___y_4192_ = v___y_4329_;
v___y_4193_ = v___y_4336_;
v___y_4194_ = v___y_4332_;
goto v___jp_4184_;
}
else
{
lean_object* v_a_4358_; lean_object* v___x_4360_; uint8_t v_isShared_4361_; uint8_t v_isSharedCheck_4365_; 
lean_dec_ref(v___y_4337_);
lean_dec(v___y_4331_);
lean_dec(v___y_4330_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4358_ = lean_ctor_get(v___x_4356_, 0);
v_isSharedCheck_4365_ = !lean_is_exclusive(v___x_4356_);
if (v_isSharedCheck_4365_ == 0)
{
v___x_4360_ = v___x_4356_;
v_isShared_4361_ = v_isSharedCheck_4365_;
goto v_resetjp_4359_;
}
else
{
lean_inc(v_a_4358_);
lean_dec(v___x_4356_);
v___x_4360_ = lean_box(0);
v_isShared_4361_ = v_isSharedCheck_4365_;
goto v_resetjp_4359_;
}
v_resetjp_4359_:
{
lean_object* v___x_4363_; 
if (v_isShared_4361_ == 0)
{
v___x_4363_ = v___x_4360_;
goto v_reusejp_4362_;
}
else
{
lean_object* v_reuseFailAlloc_4364_; 
v_reuseFailAlloc_4364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4364_, 0, v_a_4358_);
v___x_4363_ = v_reuseFailAlloc_4364_;
goto v_reusejp_4362_;
}
v_reusejp_4362_:
{
return v___x_4363_;
}
}
}
}
else
{
lean_object* v_a_4366_; lean_object* v___x_4368_; uint8_t v_isShared_4369_; uint8_t v_isSharedCheck_4373_; 
lean_dec_ref_known(v___x_4351_, 7);
lean_dec_ref(v___y_4337_);
lean_dec(v___y_4331_);
lean_dec(v___y_4330_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4366_ = lean_ctor_get(v___x_4352_, 0);
v_isSharedCheck_4373_ = !lean_is_exclusive(v___x_4352_);
if (v_isSharedCheck_4373_ == 0)
{
v___x_4368_ = v___x_4352_;
v_isShared_4369_ = v_isSharedCheck_4373_;
goto v_resetjp_4367_;
}
else
{
lean_inc(v_a_4366_);
lean_dec(v___x_4352_);
v___x_4368_ = lean_box(0);
v_isShared_4369_ = v_isSharedCheck_4373_;
goto v_resetjp_4367_;
}
v_resetjp_4367_:
{
lean_object* v___x_4371_; 
if (v_isShared_4369_ == 0)
{
v___x_4371_ = v___x_4368_;
goto v_reusejp_4370_;
}
else
{
lean_object* v_reuseFailAlloc_4372_; 
v_reuseFailAlloc_4372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4372_, 0, v_a_4366_);
v___x_4371_ = v_reuseFailAlloc_4372_;
goto v_reusejp_4370_;
}
v_reusejp_4370_:
{
return v___x_4371_;
}
}
}
}
v___jp_4374_:
{
lean_object* v___x_4382_; lean_object* v_a_4383_; lean_object* v_a_x27_4384_; lean_object* v___x_4385_; lean_object* v___x_4386_; lean_object* v___x_4387_; 
v___x_4382_ = l_Lean_instInhabitedExpr;
v_a_4383_ = lean_array_get_borrowed(v___x_4382_, v_lhsArgs_4050_, v_i_4052_);
v_a_x27_4384_ = lean_array_get_borrowed(v___x_4382_, v_rhsArgs_4051_, v_i_4052_);
v___x_4385_ = lean_unsigned_to_nat(1u);
v___x_4386_ = lean_nat_add(v_depth_4047_, v___x_4385_);
lean_inc(v_a_x27_4384_);
lean_inc(v_a_4383_);
lean_inc(v_mvarCounterSaved_4048_);
lean_inc(v___x_4386_);
v___x_4387_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_4386_, v_mvarCounterSaved_4048_, v_a_4383_, v_a_x27_4384_, v___y_4377_, v___y_4378_, v___y_4379_, v___y_4380_, v___y_4381_);
if (lean_obj_tag(v___x_4387_) == 0)
{
lean_object* v_a_4388_; lean_object* v___x_4390_; uint8_t v_isShared_4391_; uint8_t v_isSharedCheck_4445_; 
v_a_4388_ = lean_ctor_get(v___x_4387_, 0);
v_isSharedCheck_4445_ = !lean_is_exclusive(v___x_4387_);
if (v_isSharedCheck_4445_ == 0)
{
v___x_4390_ = v___x_4387_;
v_isShared_4391_ = v_isSharedCheck_4445_;
goto v_resetjp_4389_;
}
else
{
lean_inc(v_a_4388_);
lean_dec(v___x_4387_);
v___x_4390_ = lean_box(0);
v_isShared_4391_ = v_isSharedCheck_4445_;
goto v_resetjp_4389_;
}
v_resetjp_4389_:
{
uint8_t v___x_4392_; 
v___x_4392_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_a_4388_);
if (v___x_4392_ == 0)
{
lean_object* v_paramInfo_4393_; lean_object* v___x_4394_; lean_object* v___x_4395_; lean_object* v_info_4396_; uint8_t v_hasFwdDeps_4397_; 
lean_del_object(v___x_4390_);
v_paramInfo_4393_ = lean_ctor_get(v_finfo_4375_, 0);
v___x_4394_ = l_Lean_Meta_instInhabitedParamInfo_default;
v___x_4395_ = lean_nat_sub(v_i_4052_, v_finfoIdx_4376_);
v_info_4396_ = lean_array_get_borrowed(v___x_4394_, v_paramInfo_4393_, v___x_4395_);
lean_dec(v___x_4395_);
v_hasFwdDeps_4397_ = lean_ctor_get_uint8(v_info_4396_, sizeof(void*)*1 + 1);
if (v_hasFwdDeps_4397_ == 0)
{
lean_dec(v___x_4386_);
v___y_4268_ = v___y_4379_;
v___y_4269_ = v_finfoIdx_4376_;
v___y_4270_ = v___y_4381_;
v___y_4271_ = v___y_4378_;
v___y_4272_ = v___y_4377_;
v___y_4273_ = v___x_4385_;
v___y_4274_ = v_a_4388_;
v___y_4275_ = v___y_4380_;
v___y_4276_ = v_finfo_4375_;
goto v___jp_4267_;
}
else
{
if (v___x_4392_ == 0)
{
lean_object* v___x_4398_; 
lean_dec(v_a_4388_);
v___x_4398_ = l_Lean_Meta_isRefl_x3f(v_pf_4057_);
if (lean_obj_tag(v___x_4398_) == 0)
{
lean_object* v_options_4399_; uint8_t v_hasTrace_4400_; 
lean_dec(v___x_4386_);
v_options_4399_ = lean_ctor_get(v___y_4380_, 2);
v_hasTrace_4400_ = lean_ctor_get_uint8(v_options_4399_, sizeof(void*)*1);
if (v_hasTrace_4400_ == 0)
{
v___y_4065_ = v_finfoIdx_4376_;
v___y_4066_ = v_finfo_4375_;
v___y_4067_ = v___y_4377_;
v___y_4068_ = v___y_4378_;
v___y_4069_ = v___y_4379_;
v___y_4070_ = v___y_4380_;
v___y_4071_ = v___y_4381_;
goto v___jp_4064_;
}
else
{
lean_object* v_inheritedTraceOptions_4401_; lean_object* v___x_4402_; lean_object* v___x_4403_; uint8_t v___x_4404_; 
v_inheritedTraceOptions_4401_ = lean_ctor_get(v___y_4380_, 13);
v___x_4402_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_4403_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23);
v___x_4404_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4401_, v_options_4399_, v___x_4403_);
if (v___x_4404_ == 0)
{
v___y_4065_ = v_finfoIdx_4376_;
v___y_4066_ = v_finfo_4375_;
v___y_4067_ = v___y_4377_;
v___y_4068_ = v___y_4378_;
v___y_4069_ = v___y_4379_;
v___y_4070_ = v___y_4380_;
v___y_4071_ = v___y_4381_;
goto v___jp_4064_;
}
else
{
lean_object* v___x_4405_; lean_object* v___x_4406_; 
v___x_4405_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__15, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__15);
v___x_4406_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v___x_4402_, v___x_4405_, v___y_4378_, v___y_4379_, v___y_4380_, v___y_4381_);
if (lean_obj_tag(v___x_4406_) == 0)
{
lean_dec_ref_known(v___x_4406_, 1);
v___y_4065_ = v_finfoIdx_4376_;
v___y_4066_ = v_finfo_4375_;
v___y_4067_ = v___y_4377_;
v___y_4068_ = v___y_4378_;
v___y_4069_ = v___y_4379_;
v___y_4070_ = v___y_4380_;
v___y_4071_ = v___y_4381_;
goto v___jp_4064_;
}
else
{
lean_object* v_a_4407_; lean_object* v___x_4409_; uint8_t v_isShared_4410_; uint8_t v_isSharedCheck_4414_; 
lean_dec(v_finfoIdx_4376_);
lean_dec_ref(v_finfo_4375_);
lean_dec_ref(v_pf_4057_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4407_ = lean_ctor_get(v___x_4406_, 0);
v_isSharedCheck_4414_ = !lean_is_exclusive(v___x_4406_);
if (v_isSharedCheck_4414_ == 0)
{
v___x_4409_ = v___x_4406_;
v_isShared_4410_ = v_isSharedCheck_4414_;
goto v_resetjp_4408_;
}
else
{
lean_inc(v_a_4407_);
lean_dec(v___x_4406_);
v___x_4409_ = lean_box(0);
v_isShared_4410_ = v_isSharedCheck_4414_;
goto v_resetjp_4408_;
}
v_resetjp_4408_:
{
lean_object* v___x_4412_; 
if (v_isShared_4410_ == 0)
{
v___x_4412_ = v___x_4409_;
goto v_reusejp_4411_;
}
else
{
lean_object* v_reuseFailAlloc_4413_; 
v_reuseFailAlloc_4413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4413_, 0, v_a_4407_);
v___x_4412_ = v_reuseFailAlloc_4413_;
goto v_reusejp_4411_;
}
v_reusejp_4411_:
{
return v___x_4412_;
}
}
}
}
}
}
else
{
uint8_t v___x_4415_; 
lean_dec_ref_known(v___x_4398_, 1);
lean_dec_ref(v_pf_4057_);
v___x_4415_ = lean_nat_dec_eq(v_finfoIdx_4376_, v_i_4052_);
if (v___x_4415_ == 0)
{
lean_object* v___x_4416_; uint8_t v_transparency_4417_; uint8_t v___x_4418_; uint8_t v___x_4419_; 
v___x_4416_ = l_Lean_Meta_Context_config(v___y_4378_);
v_transparency_4417_ = lean_ctor_get_uint8(v___x_4416_, 9);
lean_dec_ref(v___x_4416_);
v___x_4418_ = 1;
v___x_4419_ = l_Lean_Meta_TransparencyMode_lt(v_transparency_4417_, v___x_4418_);
if (v___x_4419_ == 0)
{
v___y_4328_ = v___x_4415_;
v___y_4329_ = v___y_4379_;
v___y_4330_ = v_finfoIdx_4376_;
v___y_4331_ = v___x_4386_;
v___y_4332_ = v___y_4381_;
v___y_4333_ = v___y_4377_;
v___y_4334_ = v___y_4378_;
v___y_4335_ = v___x_4385_;
v___y_4336_ = v___y_4380_;
v___y_4337_ = v_finfo_4375_;
v___y_4338_ = v_transparency_4417_;
goto v___jp_4327_;
}
else
{
v___y_4328_ = v___x_4415_;
v___y_4329_ = v___y_4379_;
v___y_4330_ = v_finfoIdx_4376_;
v___y_4331_ = v___x_4386_;
v___y_4332_ = v___y_4381_;
v___y_4333_ = v___y_4377_;
v___y_4334_ = v___y_4378_;
v___y_4335_ = v___x_4385_;
v___y_4336_ = v___y_4380_;
v___y_4337_ = v_finfo_4375_;
v___y_4338_ = v___x_4418_;
goto v___jp_4327_;
}
}
else
{
lean_object* v___x_4420_; 
v___x_4420_ = l_Lean_Meta_FunInfo_getArity(v_finfo_4375_);
v___y_4185_ = v_finfoIdx_4376_;
v___y_4186_ = v___x_4386_;
v___y_4187_ = v___x_4385_;
v___y_4188_ = v_finfo_4375_;
v_fArity_4189_ = v___x_4420_;
v___y_4190_ = v___y_4377_;
v___y_4191_ = v___y_4378_;
v___y_4192_ = v___y_4379_;
v___y_4193_ = v___y_4380_;
v___y_4194_ = v___y_4381_;
goto v___jp_4184_;
}
}
}
else
{
lean_dec(v___x_4386_);
v___y_4268_ = v___y_4379_;
v___y_4269_ = v_finfoIdx_4376_;
v___y_4270_ = v___y_4381_;
v___y_4271_ = v___y_4378_;
v___y_4272_ = v___y_4377_;
v___y_4273_ = v___x_4385_;
v___y_4274_ = v_a_4388_;
v___y_4275_ = v___y_4380_;
v___y_4276_ = v_finfo_4375_;
goto v___jp_4267_;
}
}
}
else
{
lean_object* v_options_4421_; uint8_t v_hasTrace_4422_; 
lean_dec(v___x_4386_);
v_options_4421_ = lean_ctor_get(v___y_4380_, 2);
v_hasTrace_4422_ = lean_ctor_get_uint8(v_options_4421_, sizeof(void*)*1);
if (v_hasTrace_4422_ == 0)
{
lean_del_object(v___x_4390_);
v___y_4300_ = v_finfoIdx_4376_;
v___y_4301_ = v___x_4385_;
v___y_4302_ = v_a_4388_;
v___y_4303_ = v_finfo_4375_;
v___y_4304_ = v___y_4377_;
v___y_4305_ = v___y_4378_;
v___y_4306_ = v___y_4379_;
v___y_4307_ = v___y_4380_;
v___y_4308_ = v___y_4381_;
goto v___jp_4299_;
}
else
{
lean_object* v_inheritedTraceOptions_4423_; lean_object* v___x_4424_; lean_object* v___x_4425_; uint8_t v___x_4426_; 
v_inheritedTraceOptions_4423_ = lean_ctor_get(v___y_4380_, 13);
v___x_4424_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_4425_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23);
v___x_4426_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4423_, v_options_4421_, v___x_4425_);
if (v___x_4426_ == 0)
{
lean_del_object(v___x_4390_);
v___y_4300_ = v_finfoIdx_4376_;
v___y_4301_ = v___x_4385_;
v___y_4302_ = v_a_4388_;
v___y_4303_ = v_finfo_4375_;
v___y_4304_ = v___y_4377_;
v___y_4305_ = v___y_4378_;
v___y_4306_ = v___y_4379_;
v___y_4307_ = v___y_4380_;
v___y_4308_ = v___y_4381_;
goto v___jp_4299_;
}
else
{
lean_object* v___x_4427_; lean_object* v___x_4428_; lean_object* v___x_4430_; 
v___x_4427_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10);
lean_inc(v_i_4052_);
v___x_4428_ = l_Nat_reprFast(v_i_4052_);
if (v_isShared_4391_ == 0)
{
lean_ctor_set_tag(v___x_4390_, 3);
lean_ctor_set(v___x_4390_, 0, v___x_4428_);
v___x_4430_ = v___x_4390_;
goto v_reusejp_4429_;
}
else
{
lean_object* v_reuseFailAlloc_4444_; 
v_reuseFailAlloc_4444_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4444_, 0, v___x_4428_);
v___x_4430_ = v_reuseFailAlloc_4444_;
goto v_reusejp_4429_;
}
v_reusejp_4429_:
{
lean_object* v___x_4431_; lean_object* v___x_4432_; lean_object* v___x_4433_; lean_object* v___x_4434_; lean_object* v___x_4435_; 
v___x_4431_ = l_Lean_MessageData_ofFormat(v___x_4430_);
v___x_4432_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4432_, 0, v___x_4427_);
lean_ctor_set(v___x_4432_, 1, v___x_4431_);
v___x_4433_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__17, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__17_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__17);
v___x_4434_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4434_, 0, v___x_4432_);
lean_ctor_set(v___x_4434_, 1, v___x_4433_);
v___x_4435_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v___x_4424_, v___x_4434_, v___y_4378_, v___y_4379_, v___y_4380_, v___y_4381_);
if (lean_obj_tag(v___x_4435_) == 0)
{
lean_dec_ref_known(v___x_4435_, 1);
v___y_4300_ = v_finfoIdx_4376_;
v___y_4301_ = v___x_4385_;
v___y_4302_ = v_a_4388_;
v___y_4303_ = v_finfo_4375_;
v___y_4304_ = v___y_4377_;
v___y_4305_ = v___y_4378_;
v___y_4306_ = v___y_4379_;
v___y_4307_ = v___y_4380_;
v___y_4308_ = v___y_4381_;
goto v___jp_4299_;
}
else
{
lean_object* v_a_4436_; lean_object* v___x_4438_; uint8_t v_isShared_4439_; uint8_t v_isSharedCheck_4443_; 
lean_dec(v_a_4388_);
lean_dec(v_finfoIdx_4376_);
lean_dec_ref(v_finfo_4375_);
lean_dec_ref(v_pf_4057_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4436_ = lean_ctor_get(v___x_4435_, 0);
v_isSharedCheck_4443_ = !lean_is_exclusive(v___x_4435_);
if (v_isSharedCheck_4443_ == 0)
{
v___x_4438_ = v___x_4435_;
v_isShared_4439_ = v_isSharedCheck_4443_;
goto v_resetjp_4437_;
}
else
{
lean_inc(v_a_4436_);
lean_dec(v___x_4435_);
v___x_4438_ = lean_box(0);
v_isShared_4439_ = v_isSharedCheck_4443_;
goto v_resetjp_4437_;
}
v_resetjp_4437_:
{
lean_object* v___x_4441_; 
if (v_isShared_4439_ == 0)
{
v___x_4441_ = v___x_4438_;
goto v_reusejp_4440_;
}
else
{
lean_object* v_reuseFailAlloc_4442_; 
v_reuseFailAlloc_4442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4442_, 0, v_a_4436_);
v___x_4441_ = v_reuseFailAlloc_4442_;
goto v_reusejp_4440_;
}
v_reusejp_4440_:
{
return v___x_4441_;
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
lean_dec(v___x_4386_);
lean_dec(v_finfoIdx_4376_);
lean_dec_ref(v_finfo_4375_);
lean_dec_ref(v_pf_4057_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
return v___x_4387_;
}
}
}
else
{
lean_object* v___x_4460_; lean_object* v___x_4461_; 
lean_dec(v_finfoIdx_4054_);
lean_dec_ref(v_finfo_4053_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v___x_4460_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v_f_4055_, v_f_x27_4056_, v_pf_4057_);
v___x_4461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4461_, 0, v___x_4460_);
return v___x_4461_;
}
v___jp_4064_:
{
lean_object* v___x_4072_; lean_object* v___x_4073_; size_t v_sz_4074_; size_t v___x_4075_; lean_object* v___x_4076_; lean_object* v___x_4077_; lean_object* v___x_4078_; size_t v_sz_4079_; lean_object* v___x_4080_; 
v___x_4072_ = lean_array_get_size(v_lhsArgs_4050_);
lean_inc(v_i_4052_);
v___x_4073_ = l_Array_extract___redArg(v_lhsArgs_4050_, v_i_4052_, v___x_4072_);
v_sz_4074_ = lean_array_size(v___x_4073_);
v___x_4075_ = ((size_t)0ULL);
v___x_4076_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__0(v_sz_4074_, v___x_4075_, v___x_4073_);
v___x_4077_ = l_Lean_mkAppN(v_f_4055_, v___x_4076_);
lean_inc_ref(v_f_x27_4056_);
v___x_4078_ = l_Lean_mkAppN(v_f_x27_4056_, v___x_4076_);
v_sz_4079_ = lean_array_size(v___x_4076_);
v___x_4080_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___redArg(v___x_4076_, v_sz_4079_, v___x_4075_, v_pf_4057_, v___y_4068_, v___y_4069_, v___y_4070_, v___y_4071_);
lean_dec_ref(v___x_4076_);
if (lean_obj_tag(v___x_4080_) == 0)
{
lean_object* v_a_4081_; lean_object* v___x_4082_; 
v_a_4081_ = lean_ctor_get(v___x_4080_, 0);
lean_inc(v_a_4081_);
lean_dec_ref_known(v___x_4080_, 1);
lean_inc_ref(v_f_x27_4056_);
v___x_4082_ = l_Lean_Meta_mkEqRefl(v_f_x27_4056_, v___y_4068_, v___y_4069_, v___y_4070_, v___y_4071_);
if (lean_obj_tag(v___x_4082_) == 0)
{
lean_object* v_a_4083_; lean_object* v___x_4084_; 
v_a_4083_ = lean_ctor_get(v___x_4082_, 0);
lean_inc(v_a_4083_);
lean_dec_ref_known(v___x_4082_, 1);
lean_inc_ref(v_f_x27_4056_);
v___x_4084_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go(v_depth_4047_, v_mvarCounterSaved_4048_, v_arity_4049_, v_lhsArgs_4050_, v_rhsArgs_4051_, v_i_4052_, v___y_4066_, v___y_4065_, v_f_x27_4056_, v_f_x27_4056_, v_a_4083_, v___y_4067_, v___y_4068_, v___y_4069_, v___y_4070_, v___y_4071_);
if (lean_obj_tag(v___x_4084_) == 0)
{
lean_object* v_a_4085_; lean_object* v___x_4087_; uint8_t v_isShared_4088_; uint8_t v_isSharedCheck_4094_; 
v_a_4085_ = lean_ctor_get(v___x_4084_, 0);
v_isSharedCheck_4094_ = !lean_is_exclusive(v___x_4084_);
if (v_isSharedCheck_4094_ == 0)
{
v___x_4087_ = v___x_4084_;
v_isShared_4088_ = v_isSharedCheck_4094_;
goto v_resetjp_4086_;
}
else
{
lean_inc(v_a_4085_);
lean_dec(v___x_4084_);
v___x_4087_ = lean_box(0);
v_isShared_4088_ = v_isSharedCheck_4094_;
goto v_resetjp_4086_;
}
v_resetjp_4086_:
{
lean_object* v___x_4089_; lean_object* v___x_4090_; lean_object* v___x_4092_; 
v___x_4089_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v___x_4077_, v___x_4078_, v_a_4081_);
v___x_4090_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_trans(v___x_4089_, v_a_4085_);
if (v_isShared_4088_ == 0)
{
lean_ctor_set(v___x_4087_, 0, v___x_4090_);
v___x_4092_ = v___x_4087_;
goto v_reusejp_4091_;
}
else
{
lean_object* v_reuseFailAlloc_4093_; 
v_reuseFailAlloc_4093_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4093_, 0, v___x_4090_);
v___x_4092_ = v_reuseFailAlloc_4093_;
goto v_reusejp_4091_;
}
v_reusejp_4091_:
{
return v___x_4092_;
}
}
}
else
{
lean_dec(v_a_4081_);
lean_dec_ref(v___x_4078_);
lean_dec_ref(v___x_4077_);
return v___x_4084_;
}
}
else
{
lean_object* v_a_4095_; lean_object* v___x_4097_; uint8_t v_isShared_4098_; uint8_t v_isSharedCheck_4102_; 
lean_dec(v_a_4081_);
lean_dec_ref(v___x_4078_);
lean_dec_ref(v___x_4077_);
lean_dec_ref(v___y_4066_);
lean_dec(v___y_4065_);
lean_dec_ref(v_f_x27_4056_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4095_ = lean_ctor_get(v___x_4082_, 0);
v_isSharedCheck_4102_ = !lean_is_exclusive(v___x_4082_);
if (v_isSharedCheck_4102_ == 0)
{
v___x_4097_ = v___x_4082_;
v_isShared_4098_ = v_isSharedCheck_4102_;
goto v_resetjp_4096_;
}
else
{
lean_inc(v_a_4095_);
lean_dec(v___x_4082_);
v___x_4097_ = lean_box(0);
v_isShared_4098_ = v_isSharedCheck_4102_;
goto v_resetjp_4096_;
}
v_resetjp_4096_:
{
lean_object* v___x_4100_; 
if (v_isShared_4098_ == 0)
{
v___x_4100_ = v___x_4097_;
goto v_reusejp_4099_;
}
else
{
lean_object* v_reuseFailAlloc_4101_; 
v_reuseFailAlloc_4101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4101_, 0, v_a_4095_);
v___x_4100_ = v_reuseFailAlloc_4101_;
goto v_reusejp_4099_;
}
v_reusejp_4099_:
{
return v___x_4100_;
}
}
}
}
else
{
lean_object* v_a_4103_; lean_object* v___x_4105_; uint8_t v_isShared_4106_; uint8_t v_isSharedCheck_4110_; 
lean_dec_ref(v___x_4078_);
lean_dec_ref(v___x_4077_);
lean_dec_ref(v___y_4066_);
lean_dec(v___y_4065_);
lean_dec_ref(v_f_x27_4056_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4103_ = lean_ctor_get(v___x_4080_, 0);
v_isSharedCheck_4110_ = !lean_is_exclusive(v___x_4080_);
if (v_isSharedCheck_4110_ == 0)
{
v___x_4105_ = v___x_4080_;
v_isShared_4106_ = v_isSharedCheck_4110_;
goto v_resetjp_4104_;
}
else
{
lean_inc(v_a_4103_);
lean_dec(v___x_4080_);
v___x_4105_ = lean_box(0);
v_isShared_4106_ = v_isSharedCheck_4110_;
goto v_resetjp_4104_;
}
v_resetjp_4104_:
{
lean_object* v___x_4108_; 
if (v_isShared_4106_ == 0)
{
v___x_4108_ = v___x_4105_;
goto v_reusejp_4107_;
}
else
{
lean_object* v_reuseFailAlloc_4109_; 
v_reuseFailAlloc_4109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4109_, 0, v_a_4103_);
v___x_4108_ = v_reuseFailAlloc_4109_;
goto v_reusejp_4107_;
}
v_reusejp_4107_:
{
return v___x_4108_;
}
}
}
}
v___jp_4111_:
{
lean_object* v___x_4121_; 
lean_inc(v___y_4114_);
lean_inc_ref(v_f_4055_);
v___x_4121_ = lp_mathlib_Lean_Meta_mkHCongrWithArity_x27(v_f_4055_, v___y_4114_, v___y_4117_, v___y_4118_, v___y_4119_, v___y_4120_);
if (lean_obj_tag(v___x_4121_) == 0)
{
lean_object* v_a_4122_; lean_object* v_proof_4123_; lean_object* v_argKinds_4124_; lean_object* v___x_4125_; lean_object* v___x_4126_; lean_object* v___x_4127_; lean_object* v___x_4128_; lean_object* v___x_4129_; lean_object* v___x_4130_; lean_object* v___x_4131_; lean_object* v___x_4132_; lean_object* v___x_4133_; lean_object* v___x_4134_; lean_object* v___x_4135_; lean_object* v___x_4136_; lean_object* v___x_4137_; 
v_a_4122_ = lean_ctor_get(v___x_4121_, 0);
lean_inc(v_a_4122_);
lean_dec_ref_known(v___x_4121_, 1);
v_proof_4123_ = lean_ctor_get(v_a_4122_, 1);
lean_inc_ref(v_proof_4123_);
v_argKinds_4124_ = lean_ctor_get(v_a_4122_, 2);
lean_inc_ref(v_argKinds_4124_);
lean_dec(v_a_4122_);
v___x_4125_ = lean_array_get_size(v_rhsArgs_4051_);
v___x_4126_ = lean_unsigned_to_nat(0u);
v___x_4127_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__0));
lean_inc_n(v_i_4052_, 2);
lean_inc_ref(v_rhsArgs_4051_);
v___x_4128_ = l_Array_toSubarray___redArg(v_rhsArgs_4051_, v_i_4052_, v___x_4125_);
v___x_4129_ = lean_array_get_size(v_argKinds_4124_);
v___x_4130_ = l_Array_toSubarray___redArg(v_argKinds_4124_, v___x_4126_, v___x_4129_);
v___x_4131_ = lean_array_get_size(v_lhsArgs_4050_);
lean_inc_ref(v_lhsArgs_4050_);
v___x_4132_ = l_Array_toSubarray___redArg(v_lhsArgs_4050_, v_i_4052_, v___x_4131_);
v___x_4133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4133_, 0, v___x_4128_);
lean_ctor_set(v___x_4133_, 1, v___x_4130_);
v___x_4134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4134_, 0, v___x_4127_);
lean_ctor_set(v___x_4134_, 1, v___x_4133_);
v___x_4135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4135_, 0, v___x_4127_);
lean_ctor_set(v___x_4135_, 1, v___x_4134_);
v___x_4136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4136_, 0, v___x_4127_);
lean_ctor_set(v___x_4136_, 1, v___x_4135_);
lean_inc(v_mvarCounterSaved_4048_);
v___x_4137_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg(v___y_4113_, v_mvarCounterSaved_4048_, v___x_4132_, v___x_4136_, v___y_4116_, v___y_4117_, v___y_4118_, v___y_4119_, v___y_4120_);
if (lean_obj_tag(v___x_4137_) == 0)
{
lean_object* v_a_4138_; lean_object* v___x_4140_; uint8_t v_isShared_4141_; uint8_t v_isSharedCheck_4167_; 
v_a_4138_ = lean_ctor_get(v___x_4137_, 0);
v_isSharedCheck_4167_ = !lean_is_exclusive(v___x_4137_);
if (v_isSharedCheck_4167_ == 0)
{
v___x_4140_ = v___x_4137_;
v_isShared_4141_ = v_isSharedCheck_4167_;
goto v_resetjp_4139_;
}
else
{
lean_inc(v_a_4138_);
lean_dec(v___x_4137_);
v___x_4140_ = lean_box(0);
v_isShared_4141_ = v_isSharedCheck_4167_;
goto v_resetjp_4139_;
}
v_resetjp_4139_:
{
lean_object* v_snd_4142_; lean_object* v_snd_4143_; lean_object* v_fst_4144_; lean_object* v_fst_4145_; lean_object* v_fst_4146_; lean_object* v___x_4147_; lean_object* v___x_4148_; lean_object* v___x_4149_; lean_object* v___x_4150_; lean_object* v___x_4151_; uint8_t v___x_4152_; 
v_snd_4142_ = lean_ctor_get(v_a_4138_, 1);
lean_inc(v_snd_4142_);
v_snd_4143_ = lean_ctor_get(v_snd_4142_, 1);
lean_inc(v_snd_4143_);
v_fst_4144_ = lean_ctor_get(v_a_4138_, 0);
lean_inc(v_fst_4144_);
lean_dec(v_a_4138_);
v_fst_4145_ = lean_ctor_get(v_snd_4142_, 0);
lean_inc(v_fst_4145_);
lean_dec(v_snd_4142_);
v_fst_4146_ = lean_ctor_get(v_snd_4143_, 0);
lean_inc(v_fst_4146_);
lean_dec(v_snd_4143_);
v___x_4147_ = l_Lean_mkAppN(v_f_4055_, v_fst_4145_);
lean_dec(v_fst_4145_);
v___x_4148_ = l_Lean_mkAppN(v_f_x27_4056_, v_fst_4146_);
lean_dec(v_fst_4146_);
v___x_4149_ = l_Lean_mkAppN(v_proof_4123_, v_fst_4144_);
lean_dec(v_fst_4144_);
lean_inc_ref(v___x_4148_);
lean_inc_ref(v___x_4147_);
v___x_4150_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v___x_4147_, v___x_4148_, v___x_4149_);
v___x_4151_ = lean_nat_add(v_i_4052_, v___y_4114_);
lean_dec(v___y_4114_);
lean_dec(v_i_4052_);
v___x_4152_ = lean_nat_dec_lt(v___x_4151_, v_arity_4049_);
if (v___x_4152_ == 0)
{
lean_object* v___x_4154_; 
lean_dec(v___x_4151_);
lean_dec_ref(v___x_4148_);
lean_dec_ref(v___x_4147_);
lean_dec_ref(v___y_4115_);
lean_dec(v___y_4112_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
if (v_isShared_4141_ == 0)
{
lean_ctor_set(v___x_4140_, 0, v___x_4150_);
v___x_4154_ = v___x_4140_;
goto v_reusejp_4153_;
}
else
{
lean_object* v_reuseFailAlloc_4155_; 
v_reuseFailAlloc_4155_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4155_, 0, v___x_4150_);
v___x_4154_ = v_reuseFailAlloc_4155_;
goto v_reusejp_4153_;
}
v_reusejp_4153_:
{
return v___x_4154_;
}
}
else
{
lean_object* v___x_4156_; 
lean_del_object(v___x_4140_);
v___x_4156_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v___x_4150_, v___y_4117_, v___y_4118_, v___y_4119_, v___y_4120_);
if (lean_obj_tag(v___x_4156_) == 0)
{
lean_object* v_a_4157_; 
v_a_4157_ = lean_ctor_get(v___x_4156_, 0);
lean_inc(v_a_4157_);
lean_dec_ref_known(v___x_4156_, 1);
v_i_4052_ = v___x_4151_;
v_finfo_4053_ = v___y_4115_;
v_finfoIdx_4054_ = v___y_4112_;
v_f_4055_ = v___x_4147_;
v_f_x27_4056_ = v___x_4148_;
v_pf_4057_ = v_a_4157_;
v_a_4058_ = v___y_4116_;
v_a_4059_ = v___y_4117_;
v_a_4060_ = v___y_4118_;
v_a_4061_ = v___y_4119_;
v_a_4062_ = v___y_4120_;
goto _start;
}
else
{
lean_object* v_a_4159_; lean_object* v___x_4161_; uint8_t v_isShared_4162_; uint8_t v_isSharedCheck_4166_; 
lean_dec(v___x_4151_);
lean_dec_ref(v___x_4148_);
lean_dec_ref(v___x_4147_);
lean_dec_ref(v___y_4115_);
lean_dec(v___y_4112_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4159_ = lean_ctor_get(v___x_4156_, 0);
v_isSharedCheck_4166_ = !lean_is_exclusive(v___x_4156_);
if (v_isSharedCheck_4166_ == 0)
{
v___x_4161_ = v___x_4156_;
v_isShared_4162_ = v_isSharedCheck_4166_;
goto v_resetjp_4160_;
}
else
{
lean_inc(v_a_4159_);
lean_dec(v___x_4156_);
v___x_4161_ = lean_box(0);
v_isShared_4162_ = v_isSharedCheck_4166_;
goto v_resetjp_4160_;
}
v_resetjp_4160_:
{
lean_object* v___x_4164_; 
if (v_isShared_4162_ == 0)
{
v___x_4164_ = v___x_4161_;
goto v_reusejp_4163_;
}
else
{
lean_object* v_reuseFailAlloc_4165_; 
v_reuseFailAlloc_4165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4165_, 0, v_a_4159_);
v___x_4164_ = v_reuseFailAlloc_4165_;
goto v_reusejp_4163_;
}
v_reusejp_4163_:
{
return v___x_4164_;
}
}
}
}
}
}
else
{
lean_object* v_a_4168_; lean_object* v___x_4170_; uint8_t v_isShared_4171_; uint8_t v_isSharedCheck_4175_; 
lean_dec_ref(v_proof_4123_);
lean_dec_ref(v___y_4115_);
lean_dec(v___y_4114_);
lean_dec(v___y_4112_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4168_ = lean_ctor_get(v___x_4137_, 0);
v_isSharedCheck_4175_ = !lean_is_exclusive(v___x_4137_);
if (v_isSharedCheck_4175_ == 0)
{
v___x_4170_ = v___x_4137_;
v_isShared_4171_ = v_isSharedCheck_4175_;
goto v_resetjp_4169_;
}
else
{
lean_inc(v_a_4168_);
lean_dec(v___x_4137_);
v___x_4170_ = lean_box(0);
v_isShared_4171_ = v_isSharedCheck_4175_;
goto v_resetjp_4169_;
}
v_resetjp_4169_:
{
lean_object* v___x_4173_; 
if (v_isShared_4171_ == 0)
{
v___x_4173_ = v___x_4170_;
goto v_reusejp_4172_;
}
else
{
lean_object* v_reuseFailAlloc_4174_; 
v_reuseFailAlloc_4174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4174_, 0, v_a_4168_);
v___x_4173_ = v_reuseFailAlloc_4174_;
goto v_reusejp_4172_;
}
v_reusejp_4172_:
{
return v___x_4173_;
}
}
}
}
else
{
lean_object* v_a_4176_; lean_object* v___x_4178_; uint8_t v_isShared_4179_; uint8_t v_isSharedCheck_4183_; 
lean_dec_ref(v___y_4115_);
lean_dec(v___y_4114_);
lean_dec(v___y_4113_);
lean_dec(v___y_4112_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4176_ = lean_ctor_get(v___x_4121_, 0);
v_isSharedCheck_4183_ = !lean_is_exclusive(v___x_4121_);
if (v_isSharedCheck_4183_ == 0)
{
v___x_4178_ = v___x_4121_;
v_isShared_4179_ = v_isSharedCheck_4183_;
goto v_resetjp_4177_;
}
else
{
lean_inc(v_a_4176_);
lean_dec(v___x_4121_);
v___x_4178_ = lean_box(0);
v_isShared_4179_ = v_isSharedCheck_4183_;
goto v_resetjp_4177_;
}
v_resetjp_4177_:
{
lean_object* v___x_4181_; 
if (v_isShared_4179_ == 0)
{
v___x_4181_ = v___x_4178_;
goto v_reusejp_4180_;
}
else
{
lean_object* v_reuseFailAlloc_4182_; 
v_reuseFailAlloc_4182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4182_, 0, v_a_4176_);
v___x_4181_ = v_reuseFailAlloc_4182_;
goto v_reusejp_4180_;
}
v_reusejp_4180_:
{
return v___x_4181_;
}
}
}
}
v___jp_4184_:
{
lean_object* v_options_4195_; uint8_t v_hasTrace_4196_; 
v_options_4195_ = lean_ctor_get(v___y_4193_, 2);
v_hasTrace_4196_ = lean_ctor_get_uint8(v_options_4195_, sizeof(void*)*1);
if (v_hasTrace_4196_ == 0)
{
v___y_4112_ = v___y_4185_;
v___y_4113_ = v___y_4186_;
v___y_4114_ = v_fArity_4189_;
v___y_4115_ = v___y_4188_;
v___y_4116_ = v___y_4190_;
v___y_4117_ = v___y_4191_;
v___y_4118_ = v___y_4192_;
v___y_4119_ = v___y_4193_;
v___y_4120_ = v___y_4194_;
goto v___jp_4111_;
}
else
{
lean_object* v_inheritedTraceOptions_4197_; lean_object* v___x_4198_; lean_object* v___x_4199_; uint8_t v___x_4200_; 
v_inheritedTraceOptions_4197_ = lean_ctor_get(v___y_4193_, 13);
v___x_4198_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_4199_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23);
v___x_4200_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4197_, v_options_4195_, v___x_4199_);
if (v___x_4200_ == 0)
{
v___y_4112_ = v___y_4185_;
v___y_4113_ = v___y_4186_;
v___y_4114_ = v_fArity_4189_;
v___y_4115_ = v___y_4188_;
v___y_4116_ = v___y_4190_;
v___y_4117_ = v___y_4191_;
v___y_4118_ = v___y_4192_;
v___y_4119_ = v___y_4193_;
v___y_4120_ = v___y_4194_;
goto v___jp_4111_;
}
else
{
lean_object* v___x_4201_; lean_object* v___x_4202_; lean_object* v___x_4203_; lean_object* v___x_4204_; lean_object* v___x_4205_; lean_object* v___x_4206_; lean_object* v___x_4207_; lean_object* v___x_4208_; lean_object* v___x_4209_; lean_object* v___x_4210_; lean_object* v___x_4211_; lean_object* v___x_4212_; lean_object* v___x_4213_; lean_object* v___x_4214_; lean_object* v___x_4215_; lean_object* v___x_4216_; lean_object* v___x_4217_; lean_object* v___x_4218_; lean_object* v___x_4219_; lean_object* v___x_4220_; lean_object* v___x_4221_; lean_object* v___x_4222_; 
v___x_4201_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__2, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__2);
lean_inc(v_i_4052_);
v___x_4202_ = l_Nat_reprFast(v_i_4052_);
v___x_4203_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4203_, 0, v___x_4202_);
v___x_4204_ = l_Lean_MessageData_ofFormat(v___x_4203_);
v___x_4205_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4205_, 0, v___x_4201_);
lean_ctor_set(v___x_4205_, 1, v___x_4204_);
v___x_4206_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__4, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__4);
v___x_4207_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4207_, 0, v___x_4205_);
lean_ctor_set(v___x_4207_, 1, v___x_4206_);
v___x_4208_ = lean_nat_add(v_i_4052_, v_arity_4049_);
v___x_4209_ = lean_nat_sub(v___x_4208_, v___y_4187_);
lean_dec(v___x_4208_);
v___x_4210_ = l_Nat_reprFast(v___x_4209_);
v___x_4211_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4211_, 0, v___x_4210_);
v___x_4212_ = l_Lean_MessageData_ofFormat(v___x_4211_);
v___x_4213_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4213_, 0, v___x_4207_);
lean_ctor_set(v___x_4213_, 1, v___x_4212_);
v___x_4214_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__6, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__6);
v___x_4215_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4215_, 0, v___x_4213_);
lean_ctor_set(v___x_4215_, 1, v___x_4214_);
lean_inc(v_arity_4049_);
v___x_4216_ = l_Nat_reprFast(v_arity_4049_);
v___x_4217_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4217_, 0, v___x_4216_);
v___x_4218_ = l_Lean_MessageData_ofFormat(v___x_4217_);
v___x_4219_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4219_, 0, v___x_4215_);
lean_ctor_set(v___x_4219_, 1, v___x_4218_);
v___x_4220_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__8, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__8);
v___x_4221_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4221_, 0, v___x_4219_);
lean_ctor_set(v___x_4221_, 1, v___x_4220_);
v___x_4222_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v___x_4198_, v___x_4221_, v___y_4191_, v___y_4192_, v___y_4193_, v___y_4194_);
if (lean_obj_tag(v___x_4222_) == 0)
{
lean_dec_ref_known(v___x_4222_, 1);
v___y_4112_ = v___y_4185_;
v___y_4113_ = v___y_4186_;
v___y_4114_ = v_fArity_4189_;
v___y_4115_ = v___y_4188_;
v___y_4116_ = v___y_4190_;
v___y_4117_ = v___y_4191_;
v___y_4118_ = v___y_4192_;
v___y_4119_ = v___y_4193_;
v___y_4120_ = v___y_4194_;
goto v___jp_4111_;
}
else
{
lean_object* v_a_4223_; lean_object* v___x_4225_; uint8_t v_isShared_4226_; uint8_t v_isSharedCheck_4230_; 
lean_dec(v_fArity_4189_);
lean_dec_ref(v___y_4188_);
lean_dec(v___y_4186_);
lean_dec(v___y_4185_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4223_ = lean_ctor_get(v___x_4222_, 0);
v_isSharedCheck_4230_ = !lean_is_exclusive(v___x_4222_);
if (v_isSharedCheck_4230_ == 0)
{
v___x_4225_ = v___x_4222_;
v_isShared_4226_ = v_isSharedCheck_4230_;
goto v_resetjp_4224_;
}
else
{
lean_inc(v_a_4223_);
lean_dec(v___x_4222_);
v___x_4225_ = lean_box(0);
v_isShared_4226_ = v_isSharedCheck_4230_;
goto v_resetjp_4224_;
}
v_resetjp_4224_:
{
lean_object* v___x_4228_; 
if (v_isShared_4226_ == 0)
{
v___x_4228_ = v___x_4225_;
goto v_reusejp_4227_;
}
else
{
lean_object* v_reuseFailAlloc_4229_; 
v_reuseFailAlloc_4229_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4229_, 0, v_a_4223_);
v___x_4228_ = v_reuseFailAlloc_4229_;
goto v_reusejp_4227_;
}
v_reusejp_4227_:
{
return v___x_4228_;
}
}
}
}
}
}
v___jp_4231_:
{
lean_object* v___x_4241_; 
lean_inc_ref(v___y_4234_);
v___x_4241_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v___y_4234_, v___y_4237_, v___y_4238_, v___y_4239_, v___y_4240_);
if (lean_obj_tag(v___x_4241_) == 0)
{
lean_object* v_a_4242_; lean_object* v___x_4243_; 
v_a_4242_ = lean_ctor_get(v___x_4241_, 0);
lean_inc(v_a_4242_);
lean_dec_ref_known(v___x_4241_, 1);
v___x_4243_ = l_Lean_Meta_mkCongr(v_pf_4057_, v_a_4242_, v___y_4237_, v___y_4238_, v___y_4239_, v___y_4240_);
if (lean_obj_tag(v___x_4243_) == 0)
{
lean_object* v_a_4244_; lean_object* v_lhs_4245_; lean_object* v_rhs_4246_; lean_object* v___x_4247_; lean_object* v___x_4248_; lean_object* v___x_4249_; 
v_a_4244_ = lean_ctor_get(v___x_4243_, 0);
lean_inc(v_a_4244_);
lean_dec_ref_known(v___x_4243_, 1);
v_lhs_4245_ = lean_ctor_get(v___y_4234_, 0);
lean_inc_ref(v_lhs_4245_);
v_rhs_4246_ = lean_ctor_get(v___y_4234_, 1);
lean_inc_ref(v_rhs_4246_);
lean_dec_ref(v___y_4234_);
v___x_4247_ = lean_nat_add(v_i_4052_, v___y_4233_);
lean_dec(v_i_4052_);
v___x_4248_ = l_Lean_Expr_app___override(v_f_4055_, v_lhs_4245_);
v___x_4249_ = l_Lean_Expr_app___override(v_f_x27_4056_, v_rhs_4246_);
v_i_4052_ = v___x_4247_;
v_finfo_4053_ = v___y_4235_;
v_finfoIdx_4054_ = v___y_4232_;
v_f_4055_ = v___x_4248_;
v_f_x27_4056_ = v___x_4249_;
v_pf_4057_ = v_a_4244_;
v_a_4058_ = v___y_4236_;
v_a_4059_ = v___y_4237_;
v_a_4060_ = v___y_4238_;
v_a_4061_ = v___y_4239_;
v_a_4062_ = v___y_4240_;
goto _start;
}
else
{
lean_object* v_a_4251_; lean_object* v___x_4253_; uint8_t v_isShared_4254_; uint8_t v_isSharedCheck_4258_; 
lean_dec_ref(v___y_4235_);
lean_dec_ref(v___y_4234_);
lean_dec(v___y_4232_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4251_ = lean_ctor_get(v___x_4243_, 0);
v_isSharedCheck_4258_ = !lean_is_exclusive(v___x_4243_);
if (v_isSharedCheck_4258_ == 0)
{
v___x_4253_ = v___x_4243_;
v_isShared_4254_ = v_isSharedCheck_4258_;
goto v_resetjp_4252_;
}
else
{
lean_inc(v_a_4251_);
lean_dec(v___x_4243_);
v___x_4253_ = lean_box(0);
v_isShared_4254_ = v_isSharedCheck_4258_;
goto v_resetjp_4252_;
}
v_resetjp_4252_:
{
lean_object* v___x_4256_; 
if (v_isShared_4254_ == 0)
{
v___x_4256_ = v___x_4253_;
goto v_reusejp_4255_;
}
else
{
lean_object* v_reuseFailAlloc_4257_; 
v_reuseFailAlloc_4257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4257_, 0, v_a_4251_);
v___x_4256_ = v_reuseFailAlloc_4257_;
goto v_reusejp_4255_;
}
v_reusejp_4255_:
{
return v___x_4256_;
}
}
}
}
else
{
lean_object* v_a_4259_; lean_object* v___x_4261_; uint8_t v_isShared_4262_; uint8_t v_isSharedCheck_4266_; 
lean_dec_ref(v___y_4235_);
lean_dec_ref(v___y_4234_);
lean_dec(v___y_4232_);
lean_dec_ref(v_pf_4057_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4259_ = lean_ctor_get(v___x_4241_, 0);
v_isSharedCheck_4266_ = !lean_is_exclusive(v___x_4241_);
if (v_isSharedCheck_4266_ == 0)
{
v___x_4261_ = v___x_4241_;
v_isShared_4262_ = v_isSharedCheck_4266_;
goto v_resetjp_4260_;
}
else
{
lean_inc(v_a_4259_);
lean_dec(v___x_4241_);
v___x_4261_ = lean_box(0);
v_isShared_4262_ = v_isSharedCheck_4266_;
goto v_resetjp_4260_;
}
v_resetjp_4260_:
{
lean_object* v___x_4264_; 
if (v_isShared_4262_ == 0)
{
v___x_4264_ = v___x_4261_;
goto v_reusejp_4263_;
}
else
{
lean_object* v_reuseFailAlloc_4265_; 
v_reuseFailAlloc_4265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4265_, 0, v_a_4259_);
v___x_4264_ = v_reuseFailAlloc_4265_;
goto v_reusejp_4263_;
}
v_reusejp_4263_:
{
return v___x_4264_;
}
}
}
}
v___jp_4267_:
{
lean_object* v_options_4277_; uint8_t v_hasTrace_4278_; 
v_options_4277_ = lean_ctor_get(v___y_4275_, 2);
v_hasTrace_4278_ = lean_ctor_get_uint8(v_options_4277_, sizeof(void*)*1);
if (v_hasTrace_4278_ == 0)
{
v___y_4232_ = v___y_4269_;
v___y_4233_ = v___y_4273_;
v___y_4234_ = v___y_4274_;
v___y_4235_ = v___y_4276_;
v___y_4236_ = v___y_4272_;
v___y_4237_ = v___y_4271_;
v___y_4238_ = v___y_4268_;
v___y_4239_ = v___y_4275_;
v___y_4240_ = v___y_4270_;
goto v___jp_4231_;
}
else
{
lean_object* v_inheritedTraceOptions_4279_; lean_object* v___x_4280_; lean_object* v___x_4281_; uint8_t v___x_4282_; 
v_inheritedTraceOptions_4279_ = lean_ctor_get(v___y_4275_, 13);
v___x_4280_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_4281_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23);
v___x_4282_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4279_, v_options_4277_, v___x_4281_);
if (v___x_4282_ == 0)
{
v___y_4232_ = v___y_4269_;
v___y_4233_ = v___y_4273_;
v___y_4234_ = v___y_4274_;
v___y_4235_ = v___y_4276_;
v___y_4236_ = v___y_4272_;
v___y_4237_ = v___y_4271_;
v___y_4238_ = v___y_4268_;
v___y_4239_ = v___y_4275_;
v___y_4240_ = v___y_4270_;
goto v___jp_4231_;
}
else
{
lean_object* v___x_4283_; lean_object* v___x_4284_; lean_object* v___x_4285_; lean_object* v___x_4286_; lean_object* v___x_4287_; lean_object* v___x_4288_; lean_object* v___x_4289_; lean_object* v___x_4290_; 
v___x_4283_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__10);
lean_inc(v_i_4052_);
v___x_4284_ = l_Nat_reprFast(v_i_4052_);
v___x_4285_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4285_, 0, v___x_4284_);
v___x_4286_ = l_Lean_MessageData_ofFormat(v___x_4285_);
v___x_4287_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4287_, 0, v___x_4283_);
lean_ctor_set(v___x_4287_, 1, v___x_4286_);
v___x_4288_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__12, &lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__12_once, _init_lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___closed__12);
v___x_4289_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4289_, 0, v___x_4287_);
lean_ctor_set(v___x_4289_, 1, v___x_4288_);
v___x_4290_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v___x_4280_, v___x_4289_, v___y_4271_, v___y_4268_, v___y_4275_, v___y_4270_);
if (lean_obj_tag(v___x_4290_) == 0)
{
lean_dec_ref_known(v___x_4290_, 1);
v___y_4232_ = v___y_4269_;
v___y_4233_ = v___y_4273_;
v___y_4234_ = v___y_4274_;
v___y_4235_ = v___y_4276_;
v___y_4236_ = v___y_4272_;
v___y_4237_ = v___y_4271_;
v___y_4238_ = v___y_4268_;
v___y_4239_ = v___y_4275_;
v___y_4240_ = v___y_4270_;
goto v___jp_4231_;
}
else
{
lean_object* v_a_4291_; lean_object* v___x_4293_; uint8_t v_isShared_4294_; uint8_t v_isSharedCheck_4298_; 
lean_dec_ref(v___y_4276_);
lean_dec_ref(v___y_4274_);
lean_dec(v___y_4269_);
lean_dec_ref(v_pf_4057_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4291_ = lean_ctor_get(v___x_4290_, 0);
v_isSharedCheck_4298_ = !lean_is_exclusive(v___x_4290_);
if (v_isSharedCheck_4298_ == 0)
{
v___x_4293_ = v___x_4290_;
v_isShared_4294_ = v_isSharedCheck_4298_;
goto v_resetjp_4292_;
}
else
{
lean_inc(v_a_4291_);
lean_dec(v___x_4290_);
v___x_4293_ = lean_box(0);
v_isShared_4294_ = v_isSharedCheck_4298_;
goto v_resetjp_4292_;
}
v_resetjp_4292_:
{
lean_object* v___x_4296_; 
if (v_isShared_4294_ == 0)
{
v___x_4296_ = v___x_4293_;
goto v_reusejp_4295_;
}
else
{
lean_object* v_reuseFailAlloc_4297_; 
v_reuseFailAlloc_4297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4297_, 0, v_a_4291_);
v___x_4296_ = v_reuseFailAlloc_4297_;
goto v_reusejp_4295_;
}
v_reusejp_4295_:
{
return v___x_4296_;
}
}
}
}
}
}
v___jp_4299_:
{
lean_object* v_lhs_4309_; lean_object* v_rhs_4310_; lean_object* v___x_4311_; 
v_lhs_4309_ = lean_ctor_get(v___y_4302_, 0);
lean_inc_ref_n(v_lhs_4309_, 2);
v_rhs_4310_ = lean_ctor_get(v___y_4302_, 1);
lean_inc_ref(v_rhs_4310_);
lean_dec_ref(v___y_4302_);
v___x_4311_ = l_Lean_Meta_mkCongrFun(v_pf_4057_, v_lhs_4309_, v___y_4305_, v___y_4306_, v___y_4307_, v___y_4308_);
if (lean_obj_tag(v___x_4311_) == 0)
{
lean_object* v_a_4312_; lean_object* v___x_4313_; lean_object* v___x_4314_; lean_object* v___x_4315_; 
v_a_4312_ = lean_ctor_get(v___x_4311_, 0);
lean_inc(v_a_4312_);
lean_dec_ref_known(v___x_4311_, 1);
v___x_4313_ = lean_nat_add(v_i_4052_, v___y_4301_);
lean_dec(v_i_4052_);
v___x_4314_ = l_Lean_Expr_app___override(v_f_4055_, v_lhs_4309_);
v___x_4315_ = l_Lean_Expr_app___override(v_f_x27_4056_, v_rhs_4310_);
v_i_4052_ = v___x_4313_;
v_finfo_4053_ = v___y_4303_;
v_finfoIdx_4054_ = v___y_4300_;
v_f_4055_ = v___x_4314_;
v_f_x27_4056_ = v___x_4315_;
v_pf_4057_ = v_a_4312_;
v_a_4058_ = v___y_4304_;
v_a_4059_ = v___y_4305_;
v_a_4060_ = v___y_4306_;
v_a_4061_ = v___y_4307_;
v_a_4062_ = v___y_4308_;
goto _start;
}
else
{
lean_object* v_a_4317_; lean_object* v___x_4319_; uint8_t v_isShared_4320_; uint8_t v_isSharedCheck_4324_; 
lean_dec_ref(v_rhs_4310_);
lean_dec_ref(v_lhs_4309_);
lean_dec_ref(v___y_4303_);
lean_dec(v___y_4300_);
lean_dec_ref(v_f_x27_4056_);
lean_dec_ref(v_f_4055_);
lean_dec(v_i_4052_);
lean_dec_ref(v_rhsArgs_4051_);
lean_dec_ref(v_lhsArgs_4050_);
lean_dec(v_arity_4049_);
lean_dec(v_mvarCounterSaved_4048_);
v_a_4317_ = lean_ctor_get(v___x_4311_, 0);
v_isSharedCheck_4324_ = !lean_is_exclusive(v___x_4311_);
if (v_isSharedCheck_4324_ == 0)
{
v___x_4319_ = v___x_4311_;
v_isShared_4320_ = v_isSharedCheck_4324_;
goto v_resetjp_4318_;
}
else
{
lean_inc(v_a_4317_);
lean_dec(v___x_4311_);
v___x_4319_ = lean_box(0);
v_isShared_4320_ = v_isSharedCheck_4324_;
goto v_resetjp_4318_;
}
v_resetjp_4318_:
{
lean_object* v___x_4322_; 
if (v_isShared_4320_ == 0)
{
v___x_4322_ = v___x_4319_;
goto v_reusejp_4321_;
}
else
{
lean_object* v_reuseFailAlloc_4323_; 
v_reuseFailAlloc_4323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4323_, 0, v_a_4317_);
v___x_4322_ = v_reuseFailAlloc_4323_;
goto v_reusejp_4321_;
}
v_reusejp_4321_:
{
return v___x_4322_;
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__1(void){
_start:
{
lean_object* v___x_4463_; lean_object* v___x_4464_; 
v___x_4463_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__0));
v___x_4464_ = l_Lean_stringToMessageData(v___x_4463_);
return v___x_4464_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__3(void){
_start:
{
lean_object* v___x_4466_; lean_object* v___x_4467_; 
v___x_4466_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__2));
v___x_4467_ = l_Lean_stringToMessageData(v___x_4466_);
return v___x_4467_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__5(void){
_start:
{
lean_object* v___x_4469_; lean_object* v___x_4470_; 
v___x_4469_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__4));
v___x_4470_ = l_Lean_stringToMessageData(v___x_4469_);
return v___x_4470_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__7(void){
_start:
{
lean_object* v___x_4472_; lean_object* v___x_4473_; 
v___x_4472_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__6));
v___x_4473_ = l_Lean_stringToMessageData(v___x_4472_);
return v___x_4473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp(lean_object* v_depth_4474_, lean_object* v_mvarCounterSaved_4475_, lean_object* v_lhs_4476_, lean_object* v_rhs_4477_, lean_object* v_a_4478_, lean_object* v_a_4479_, lean_object* v_a_4480_, lean_object* v_a_4481_, lean_object* v_a_4482_){
_start:
{
lean_object* v___y_4485_; lean_object* v___y_4486_; lean_object* v___y_4487_; lean_object* v___y_4488_; lean_object* v___y_4489_; lean_object* v___y_4490_; lean_object* v___y_4491_; lean_object* v___y_4492_; lean_object* v___y_4493_; lean_object* v___y_4494_; lean_object* v___y_4495_; lean_object* v___y_4527_; lean_object* v___y_4528_; lean_object* v___y_4529_; lean_object* v___y_4530_; lean_object* v___y_4531_; lean_object* v___y_4532_; lean_object* v___y_4533_; lean_object* v___y_4534_; lean_object* v___y_4535_; lean_object* v___y_4536_; lean_object* v___y_4537_; lean_object* v_inheritedTraceOptions_4541_; lean_object* v_cls_4542_; lean_object* v___x_4543_; 
v_inheritedTraceOptions_4541_ = lean_ctor_get(v_a_4481_, 13);
v_cls_4542_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___x_4543_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0(v_cls_4542_, v_inheritedTraceOptions_4541_, v_a_4478_, v_a_4479_, v_a_4480_, v_a_4481_, v_a_4482_);
if (lean_obj_tag(v___x_4543_) == 0)
{
lean_object* v_a_4544_; lean_object* v_arity_4545_; lean_object* v___y_4547_; lean_object* v___y_4548_; lean_object* v___y_4549_; lean_object* v___y_4550_; lean_object* v___y_4551_; lean_object* v___y_4552_; lean_object* v___y_4553_; lean_object* v___y_4554_; lean_object* v___y_4555_; lean_object* v___y_4559_; lean_object* v___y_4560_; uint8_t v___y_4561_; lean_object* v___y_4562_; lean_object* v___y_4563_; lean_object* v___y_4564_; lean_object* v___y_4565_; lean_object* v___y_4566_; lean_object* v___y_4567_; lean_object* v___y_4568_; lean_object* v___y_4626_; lean_object* v___y_4627_; lean_object* v___y_4628_; lean_object* v___y_4629_; lean_object* v___y_4630_; uint8_t v___x_4688_; 
v_a_4544_ = lean_ctor_get(v___x_4543_, 0);
lean_inc(v_a_4544_);
lean_dec_ref_known(v___x_4543_, 1);
v_arity_4545_ = l_Lean_Expr_getAppNumArgs(v_lhs_4476_);
v___x_4688_ = lean_unbox(v_a_4544_);
lean_dec(v_a_4544_);
if (v___x_4688_ == 0)
{
v___y_4626_ = v_a_4478_;
v___y_4627_ = v_a_4479_;
v___y_4628_ = v_a_4480_;
v___y_4629_ = v_a_4481_;
v___y_4630_ = v_a_4482_;
goto v___jp_4625_;
}
else
{
lean_object* v___x_4689_; lean_object* v___x_4690_; lean_object* v___x_4691_; lean_object* v___x_4692_; lean_object* v___x_4693_; lean_object* v___x_4694_; 
v___x_4689_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__7, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__7);
lean_inc(v_arity_4545_);
v___x_4690_ = l_Nat_reprFast(v_arity_4545_);
v___x_4691_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4691_, 0, v___x_4690_);
v___x_4692_ = l_Lean_MessageData_ofFormat(v___x_4691_);
v___x_4693_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4693_, 0, v___x_4689_);
lean_ctor_set(v___x_4693_, 1, v___x_4692_);
v___x_4694_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4542_, v___x_4693_, v_a_4479_, v_a_4480_, v_a_4481_, v_a_4482_);
if (lean_obj_tag(v___x_4694_) == 0)
{
lean_dec_ref_known(v___x_4694_, 1);
v___y_4626_ = v_a_4478_;
v___y_4627_ = v_a_4479_;
v___y_4628_ = v_a_4480_;
v___y_4629_ = v_a_4481_;
v___y_4630_ = v_a_4482_;
goto v___jp_4625_;
}
else
{
lean_object* v_a_4695_; lean_object* v___x_4697_; uint8_t v_isShared_4698_; uint8_t v_isSharedCheck_4702_; 
lean_dec(v_arity_4545_);
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4695_ = lean_ctor_get(v___x_4694_, 0);
v_isSharedCheck_4702_ = !lean_is_exclusive(v___x_4694_);
if (v_isSharedCheck_4702_ == 0)
{
v___x_4697_ = v___x_4694_;
v_isShared_4698_ = v_isSharedCheck_4702_;
goto v_resetjp_4696_;
}
else
{
lean_inc(v_a_4695_);
lean_dec(v___x_4694_);
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
lean_ctor_set(v_reuseFailAlloc_4701_, 0, v_a_4695_);
v___x_4700_ = v_reuseFailAlloc_4701_;
goto v_reusejp_4699_;
}
v_reusejp_4699_:
{
return v___x_4700_;
}
}
}
}
v___jp_4546_:
{
lean_object* v_dummy_4556_; uint8_t v___x_4557_; 
v_dummy_4556_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0, &lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_cHole_x3f___closed__0);
v___x_4557_ = lean_nat_dec_le(v___y_4550_, v_arity_4545_);
if (v___x_4557_ == 0)
{
v___y_4527_ = v___y_4551_;
v___y_4528_ = v_dummy_4556_;
v___y_4529_ = v___y_4554_;
v___y_4530_ = v___y_4547_;
v___y_4531_ = v___y_4555_;
v___y_4532_ = v___y_4552_;
v___y_4533_ = v___y_4548_;
v___y_4534_ = v___y_4549_;
v___y_4535_ = v___y_4553_;
v___y_4536_ = v___y_4550_;
v___y_4537_ = v_arity_4545_;
goto v___jp_4526_;
}
else
{
lean_dec(v_arity_4545_);
lean_inc(v___y_4550_);
v___y_4527_ = v___y_4551_;
v___y_4528_ = v_dummy_4556_;
v___y_4529_ = v___y_4554_;
v___y_4530_ = v___y_4547_;
v___y_4531_ = v___y_4555_;
v___y_4532_ = v___y_4552_;
v___y_4533_ = v___y_4548_;
v___y_4534_ = v___y_4549_;
v___y_4535_ = v___y_4553_;
v___y_4536_ = v___y_4550_;
v___y_4537_ = v___y_4550_;
goto v___jp_4526_;
}
}
v___jp_4558_:
{
uint8_t v___x_4569_; 
v___x_4569_ = lean_expr_eqv(v___y_4559_, v___y_4562_);
if (v___x_4569_ == 0)
{
if (v___y_4561_ == 0)
{
v___y_4547_ = v___y_4559_;
v___y_4548_ = v___y_4560_;
v___y_4549_ = v___y_4562_;
v___y_4550_ = v___y_4563_;
v___y_4551_ = v___y_4564_;
v___y_4552_ = v___y_4565_;
v___y_4553_ = v___y_4566_;
v___y_4554_ = v___y_4567_;
v___y_4555_ = v___y_4568_;
goto v___jp_4546_;
}
else
{
lean_object* v___x_4570_; 
lean_inc(v___y_4568_);
lean_inc_ref(v___y_4567_);
lean_inc(v___y_4566_);
lean_inc_ref(v___y_4565_);
lean_inc_ref(v___y_4559_);
v___x_4570_ = lean_infer_type(v___y_4559_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_);
if (lean_obj_tag(v___x_4570_) == 0)
{
lean_object* v_a_4571_; lean_object* v___x_4572_; 
v_a_4571_ = lean_ctor_get(v___x_4570_, 0);
lean_inc(v_a_4571_);
lean_dec_ref_known(v___x_4570_, 1);
lean_inc(v___y_4568_);
lean_inc_ref(v___y_4567_);
lean_inc(v___y_4566_);
lean_inc_ref(v___y_4565_);
lean_inc_ref(v___y_4562_);
v___x_4572_ = lean_infer_type(v___y_4562_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_);
if (lean_obj_tag(v___x_4572_) == 0)
{
lean_object* v_a_4573_; lean_object* v___x_4574_; 
v_a_4573_ = lean_ctor_get(v___x_4572_, 0);
lean_inc(v_a_4573_);
lean_dec_ref_known(v___x_4572_, 1);
v___x_4574_ = l_Lean_Meta_isExprDefEq(v_a_4571_, v_a_4573_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_);
if (lean_obj_tag(v___x_4574_) == 0)
{
lean_object* v_a_4575_; uint8_t v___x_4576_; 
v_a_4575_ = lean_ctor_get(v___x_4574_, 0);
lean_inc(v_a_4575_);
lean_dec_ref_known(v___x_4574_, 1);
v___x_4576_ = lean_unbox(v_a_4575_);
lean_dec(v_a_4575_);
if (v___x_4576_ == 0)
{
lean_object* v_inheritedTraceOptions_4577_; lean_object* v___x_4578_; 
lean_dec(v___y_4563_);
lean_dec_ref(v___y_4562_);
lean_dec(v___y_4560_);
lean_dec_ref(v___y_4559_);
lean_dec(v_arity_4545_);
v_inheritedTraceOptions_4577_ = lean_ctor_get(v___y_4567_, 13);
v___x_4578_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0(v_cls_4542_, v_inheritedTraceOptions_4577_, v___y_4564_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_);
if (lean_obj_tag(v___x_4578_) == 0)
{
lean_object* v_a_4579_; uint8_t v___x_4580_; 
v_a_4579_ = lean_ctor_get(v___x_4578_, 0);
lean_inc(v_a_4579_);
lean_dec_ref_known(v___x_4578_, 1);
v___x_4580_ = lean_unbox(v_a_4579_);
lean_dec(v_a_4579_);
if (v___x_4580_ == 0)
{
lean_object* v___x_4581_; 
v___x_4581_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_4475_, v_lhs_4476_, v_rhs_4477_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_);
return v___x_4581_;
}
else
{
lean_object* v___x_4582_; lean_object* v___x_4583_; 
v___x_4582_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__1);
v___x_4583_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4542_, v___x_4582_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_);
if (lean_obj_tag(v___x_4583_) == 0)
{
lean_object* v___x_4584_; 
lean_dec_ref_known(v___x_4583_, 1);
v___x_4584_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_4475_, v_lhs_4476_, v_rhs_4477_, v___y_4565_, v___y_4566_, v___y_4567_, v___y_4568_);
return v___x_4584_;
}
else
{
lean_object* v_a_4585_; lean_object* v___x_4587_; uint8_t v_isShared_4588_; uint8_t v_isSharedCheck_4592_; 
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4585_ = lean_ctor_get(v___x_4583_, 0);
v_isSharedCheck_4592_ = !lean_is_exclusive(v___x_4583_);
if (v_isSharedCheck_4592_ == 0)
{
v___x_4587_ = v___x_4583_;
v_isShared_4588_ = v_isSharedCheck_4592_;
goto v_resetjp_4586_;
}
else
{
lean_inc(v_a_4585_);
lean_dec(v___x_4583_);
v___x_4587_ = lean_box(0);
v_isShared_4588_ = v_isSharedCheck_4592_;
goto v_resetjp_4586_;
}
v_resetjp_4586_:
{
lean_object* v___x_4590_; 
if (v_isShared_4588_ == 0)
{
v___x_4590_ = v___x_4587_;
goto v_reusejp_4589_;
}
else
{
lean_object* v_reuseFailAlloc_4591_; 
v_reuseFailAlloc_4591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4591_, 0, v_a_4585_);
v___x_4590_ = v_reuseFailAlloc_4591_;
goto v_reusejp_4589_;
}
v_reusejp_4589_:
{
return v___x_4590_;
}
}
}
}
}
else
{
lean_object* v_a_4593_; lean_object* v___x_4595_; uint8_t v_isShared_4596_; uint8_t v_isSharedCheck_4600_; 
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4593_ = lean_ctor_get(v___x_4578_, 0);
v_isSharedCheck_4600_ = !lean_is_exclusive(v___x_4578_);
if (v_isSharedCheck_4600_ == 0)
{
v___x_4595_ = v___x_4578_;
v_isShared_4596_ = v_isSharedCheck_4600_;
goto v_resetjp_4594_;
}
else
{
lean_inc(v_a_4593_);
lean_dec(v___x_4578_);
v___x_4595_ = lean_box(0);
v_isShared_4596_ = v_isSharedCheck_4600_;
goto v_resetjp_4594_;
}
v_resetjp_4594_:
{
lean_object* v___x_4598_; 
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
return v___x_4598_;
}
}
}
}
else
{
v___y_4547_ = v___y_4559_;
v___y_4548_ = v___y_4560_;
v___y_4549_ = v___y_4562_;
v___y_4550_ = v___y_4563_;
v___y_4551_ = v___y_4564_;
v___y_4552_ = v___y_4565_;
v___y_4553_ = v___y_4566_;
v___y_4554_ = v___y_4567_;
v___y_4555_ = v___y_4568_;
goto v___jp_4546_;
}
}
else
{
lean_object* v_a_4601_; lean_object* v___x_4603_; uint8_t v_isShared_4604_; uint8_t v_isSharedCheck_4608_; 
lean_dec(v___y_4563_);
lean_dec_ref(v___y_4562_);
lean_dec(v___y_4560_);
lean_dec_ref(v___y_4559_);
lean_dec(v_arity_4545_);
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4601_ = lean_ctor_get(v___x_4574_, 0);
v_isSharedCheck_4608_ = !lean_is_exclusive(v___x_4574_);
if (v_isSharedCheck_4608_ == 0)
{
v___x_4603_ = v___x_4574_;
v_isShared_4604_ = v_isSharedCheck_4608_;
goto v_resetjp_4602_;
}
else
{
lean_inc(v_a_4601_);
lean_dec(v___x_4574_);
v___x_4603_ = lean_box(0);
v_isShared_4604_ = v_isSharedCheck_4608_;
goto v_resetjp_4602_;
}
v_resetjp_4602_:
{
lean_object* v___x_4606_; 
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
return v___x_4606_;
}
}
}
}
else
{
lean_object* v_a_4609_; lean_object* v___x_4611_; uint8_t v_isShared_4612_; uint8_t v_isSharedCheck_4616_; 
lean_dec(v_a_4571_);
lean_dec(v___y_4563_);
lean_dec_ref(v___y_4562_);
lean_dec(v___y_4560_);
lean_dec_ref(v___y_4559_);
lean_dec(v_arity_4545_);
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4609_ = lean_ctor_get(v___x_4572_, 0);
v_isSharedCheck_4616_ = !lean_is_exclusive(v___x_4572_);
if (v_isSharedCheck_4616_ == 0)
{
v___x_4611_ = v___x_4572_;
v_isShared_4612_ = v_isSharedCheck_4616_;
goto v_resetjp_4610_;
}
else
{
lean_inc(v_a_4609_);
lean_dec(v___x_4572_);
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
else
{
lean_object* v_a_4617_; lean_object* v___x_4619_; uint8_t v_isShared_4620_; uint8_t v_isSharedCheck_4624_; 
lean_dec(v___y_4563_);
lean_dec_ref(v___y_4562_);
lean_dec(v___y_4560_);
lean_dec_ref(v___y_4559_);
lean_dec(v_arity_4545_);
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4617_ = lean_ctor_get(v___x_4570_, 0);
v_isSharedCheck_4624_ = !lean_is_exclusive(v___x_4570_);
if (v_isSharedCheck_4624_ == 0)
{
v___x_4619_ = v___x_4570_;
v_isShared_4620_ = v_isSharedCheck_4624_;
goto v_resetjp_4618_;
}
else
{
lean_inc(v_a_4617_);
lean_dec(v___x_4570_);
v___x_4619_ = lean_box(0);
v_isShared_4620_ = v_isSharedCheck_4624_;
goto v_resetjp_4618_;
}
v_resetjp_4618_:
{
lean_object* v___x_4622_; 
if (v_isShared_4620_ == 0)
{
v___x_4622_ = v___x_4619_;
goto v_reusejp_4621_;
}
else
{
lean_object* v_reuseFailAlloc_4623_; 
v_reuseFailAlloc_4623_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4623_, 0, v_a_4617_);
v___x_4622_ = v_reuseFailAlloc_4623_;
goto v_reusejp_4621_;
}
v_reusejp_4621_:
{
return v___x_4622_;
}
}
}
}
}
else
{
v___y_4547_ = v___y_4559_;
v___y_4548_ = v___y_4560_;
v___y_4549_ = v___y_4562_;
v___y_4550_ = v___y_4563_;
v___y_4551_ = v___y_4564_;
v___y_4552_ = v___y_4565_;
v___y_4553_ = v___y_4566_;
v___y_4554_ = v___y_4567_;
v___y_4555_ = v___y_4568_;
goto v___jp_4546_;
}
}
v___jp_4625_:
{
lean_object* v___x_4631_; uint8_t v___x_4632_; 
v___x_4631_ = l_Lean_Expr_getAppNumArgs(v_rhs_4477_);
v___x_4632_ = lean_nat_dec_eq(v_arity_4545_, v___x_4631_);
if (v___x_4632_ == 0)
{
lean_object* v_options_4633_; uint8_t v_hasTrace_4634_; 
lean_dec(v___x_4631_);
lean_dec(v_arity_4545_);
v_options_4633_ = lean_ctor_get(v___y_4629_, 2);
v_hasTrace_4634_ = lean_ctor_get_uint8(v_options_4633_, sizeof(void*)*1);
if (v_hasTrace_4634_ == 0)
{
lean_object* v___x_4635_; 
v___x_4635_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_4475_, v_lhs_4476_, v_rhs_4477_, v___y_4627_, v___y_4628_, v___y_4629_, v___y_4630_);
return v___x_4635_;
}
else
{
lean_object* v_inheritedTraceOptions_4636_; lean_object* v___x_4637_; uint8_t v___x_4638_; 
v_inheritedTraceOptions_4636_ = lean_ctor_get(v___y_4629_, 13);
v___x_4637_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23);
v___x_4638_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4636_, v_options_4633_, v___x_4637_);
if (v___x_4638_ == 0)
{
lean_object* v___x_4639_; 
v___x_4639_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_4475_, v_lhs_4476_, v_rhs_4477_, v___y_4627_, v___y_4628_, v___y_4629_, v___y_4630_);
return v___x_4639_;
}
else
{
lean_object* v___x_4640_; lean_object* v___x_4641_; 
v___x_4640_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__3, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__3);
v___x_4641_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4542_, v___x_4640_, v___y_4627_, v___y_4628_, v___y_4629_, v___y_4630_);
if (lean_obj_tag(v___x_4641_) == 0)
{
lean_object* v___x_4642_; 
lean_dec_ref_known(v___x_4641_, 1);
v___x_4642_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_4475_, v_lhs_4476_, v_rhs_4477_, v___y_4627_, v___y_4628_, v___y_4629_, v___y_4630_);
return v___x_4642_;
}
else
{
lean_object* v_a_4643_; lean_object* v___x_4645_; uint8_t v_isShared_4646_; uint8_t v_isSharedCheck_4650_; 
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4643_ = lean_ctor_get(v___x_4641_, 0);
v_isSharedCheck_4650_ = !lean_is_exclusive(v___x_4641_);
if (v_isSharedCheck_4650_ == 0)
{
v___x_4645_ = v___x_4641_;
v_isShared_4646_ = v_isSharedCheck_4650_;
goto v_resetjp_4644_;
}
else
{
lean_inc(v_a_4643_);
lean_dec(v___x_4641_);
v___x_4645_ = lean_box(0);
v_isShared_4646_ = v_isSharedCheck_4650_;
goto v_resetjp_4644_;
}
v_resetjp_4644_:
{
lean_object* v___x_4648_; 
if (v_isShared_4646_ == 0)
{
v___x_4648_ = v___x_4645_;
goto v_reusejp_4647_;
}
else
{
lean_object* v_reuseFailAlloc_4649_; 
v_reuseFailAlloc_4649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4649_, 0, v_a_4643_);
v___x_4648_ = v_reuseFailAlloc_4649_;
goto v_reusejp_4647_;
}
v_reusejp_4647_:
{
return v___x_4648_;
}
}
}
}
}
}
else
{
lean_object* v___x_4651_; lean_object* v_fst_4652_; lean_object* v_snd_4653_; lean_object* v___x_4655_; uint8_t v_isShared_4656_; uint8_t v_isSharedCheck_4687_; 
lean_inc_ref(v_rhs_4477_);
lean_inc_ref(v_lhs_4476_);
v___x_4651_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_getJointAppFns(v_lhs_4476_, v_rhs_4477_);
v_fst_4652_ = lean_ctor_get(v___x_4651_, 0);
v_snd_4653_ = lean_ctor_get(v___x_4651_, 1);
v_isSharedCheck_4687_ = !lean_is_exclusive(v___x_4651_);
if (v_isSharedCheck_4687_ == 0)
{
v___x_4655_ = v___x_4651_;
v_isShared_4656_ = v_isSharedCheck_4687_;
goto v_resetjp_4654_;
}
else
{
lean_inc(v_snd_4653_);
lean_inc(v_fst_4652_);
lean_dec(v___x_4651_);
v___x_4655_ = lean_box(0);
v_isShared_4656_ = v_isSharedCheck_4687_;
goto v_resetjp_4654_;
}
v_resetjp_4654_:
{
lean_object* v_inheritedTraceOptions_4657_; lean_object* v___x_4658_; 
v_inheritedTraceOptions_4657_ = lean_ctor_get(v___y_4629_, 13);
v___x_4658_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0(v_cls_4542_, v_inheritedTraceOptions_4657_, v___y_4626_, v___y_4627_, v___y_4628_, v___y_4629_, v___y_4630_);
if (lean_obj_tag(v___x_4658_) == 0)
{
lean_object* v_a_4659_; lean_object* v___x_4660_; lean_object* v___x_4661_; uint8_t v___x_4662_; 
v_a_4659_ = lean_ctor_get(v___x_4658_, 0);
lean_inc(v_a_4659_);
lean_dec_ref_known(v___x_4658_, 1);
v___x_4660_ = l_Lean_Expr_getAppNumArgs(v_fst_4652_);
v___x_4661_ = lean_nat_sub(v_arity_4545_, v___x_4660_);
lean_dec(v___x_4660_);
v___x_4662_ = lean_unbox(v_a_4659_);
lean_dec(v_a_4659_);
if (v___x_4662_ == 0)
{
lean_del_object(v___x_4655_);
v___y_4559_ = v_fst_4652_;
v___y_4560_ = v___x_4631_;
v___y_4561_ = v___x_4632_;
v___y_4562_ = v_snd_4653_;
v___y_4563_ = v___x_4661_;
v___y_4564_ = v___y_4626_;
v___y_4565_ = v___y_4627_;
v___y_4566_ = v___y_4628_;
v___y_4567_ = v___y_4629_;
v___y_4568_ = v___y_4630_;
goto v___jp_4558_;
}
else
{
lean_object* v___x_4663_; lean_object* v___x_4664_; lean_object* v___x_4665_; lean_object* v___x_4666_; lean_object* v___x_4668_; 
v___x_4663_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__5, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___closed__5);
lean_inc(v___x_4661_);
v___x_4664_ = l_Nat_reprFast(v___x_4661_);
v___x_4665_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4665_, 0, v___x_4664_);
v___x_4666_ = l_Lean_MessageData_ofFormat(v___x_4665_);
if (v_isShared_4656_ == 0)
{
lean_ctor_set_tag(v___x_4655_, 7);
lean_ctor_set(v___x_4655_, 1, v___x_4666_);
lean_ctor_set(v___x_4655_, 0, v___x_4663_);
v___x_4668_ = v___x_4655_;
goto v_reusejp_4667_;
}
else
{
lean_object* v_reuseFailAlloc_4678_; 
v_reuseFailAlloc_4678_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4678_, 0, v___x_4663_);
lean_ctor_set(v_reuseFailAlloc_4678_, 1, v___x_4666_);
v___x_4668_ = v_reuseFailAlloc_4678_;
goto v_reusejp_4667_;
}
v_reusejp_4667_:
{
lean_object* v___x_4669_; 
v___x_4669_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4542_, v___x_4668_, v___y_4627_, v___y_4628_, v___y_4629_, v___y_4630_);
if (lean_obj_tag(v___x_4669_) == 0)
{
lean_dec_ref_known(v___x_4669_, 1);
v___y_4559_ = v_fst_4652_;
v___y_4560_ = v___x_4631_;
v___y_4561_ = v___x_4632_;
v___y_4562_ = v_snd_4653_;
v___y_4563_ = v___x_4661_;
v___y_4564_ = v___y_4626_;
v___y_4565_ = v___y_4627_;
v___y_4566_ = v___y_4628_;
v___y_4567_ = v___y_4629_;
v___y_4568_ = v___y_4630_;
goto v___jp_4558_;
}
else
{
lean_object* v_a_4670_; lean_object* v___x_4672_; uint8_t v_isShared_4673_; uint8_t v_isSharedCheck_4677_; 
lean_dec(v___x_4661_);
lean_dec(v_snd_4653_);
lean_dec(v_fst_4652_);
lean_dec(v___x_4631_);
lean_dec(v_arity_4545_);
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4670_ = lean_ctor_get(v___x_4669_, 0);
v_isSharedCheck_4677_ = !lean_is_exclusive(v___x_4669_);
if (v_isSharedCheck_4677_ == 0)
{
v___x_4672_ = v___x_4669_;
v_isShared_4673_ = v_isSharedCheck_4677_;
goto v_resetjp_4671_;
}
else
{
lean_inc(v_a_4670_);
lean_dec(v___x_4669_);
v___x_4672_ = lean_box(0);
v_isShared_4673_ = v_isSharedCheck_4677_;
goto v_resetjp_4671_;
}
v_resetjp_4671_:
{
lean_object* v___x_4675_; 
if (v_isShared_4673_ == 0)
{
v___x_4675_ = v___x_4672_;
goto v_reusejp_4674_;
}
else
{
lean_object* v_reuseFailAlloc_4676_; 
v_reuseFailAlloc_4676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4676_, 0, v_a_4670_);
v___x_4675_ = v_reuseFailAlloc_4676_;
goto v_reusejp_4674_;
}
v_reusejp_4674_:
{
return v___x_4675_;
}
}
}
}
}
}
else
{
lean_object* v_a_4679_; lean_object* v___x_4681_; uint8_t v_isShared_4682_; uint8_t v_isSharedCheck_4686_; 
lean_del_object(v___x_4655_);
lean_dec(v_snd_4653_);
lean_dec(v_fst_4652_);
lean_dec(v___x_4631_);
lean_dec(v_arity_4545_);
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4679_ = lean_ctor_get(v___x_4658_, 0);
v_isSharedCheck_4686_ = !lean_is_exclusive(v___x_4658_);
if (v_isSharedCheck_4686_ == 0)
{
v___x_4681_ = v___x_4658_;
v_isShared_4682_ = v_isSharedCheck_4686_;
goto v_resetjp_4680_;
}
else
{
lean_inc(v_a_4679_);
lean_dec(v___x_4658_);
v___x_4681_ = lean_box(0);
v_isShared_4682_ = v_isSharedCheck_4686_;
goto v_resetjp_4680_;
}
v_resetjp_4680_:
{
lean_object* v___x_4684_; 
if (v_isShared_4682_ == 0)
{
v___x_4684_ = v___x_4681_;
goto v_reusejp_4683_;
}
else
{
lean_object* v_reuseFailAlloc_4685_; 
v_reuseFailAlloc_4685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4685_, 0, v_a_4679_);
v___x_4684_ = v_reuseFailAlloc_4685_;
goto v_reusejp_4683_;
}
v_reusejp_4683_:
{
return v___x_4684_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_4703_; lean_object* v___x_4705_; uint8_t v_isShared_4706_; uint8_t v_isSharedCheck_4710_; 
lean_dec_ref(v_rhs_4477_);
lean_dec_ref(v_lhs_4476_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4703_ = lean_ctor_get(v___x_4543_, 0);
v_isSharedCheck_4710_ = !lean_is_exclusive(v___x_4543_);
if (v_isSharedCheck_4710_ == 0)
{
v___x_4705_ = v___x_4543_;
v_isShared_4706_ = v_isSharedCheck_4710_;
goto v_resetjp_4704_;
}
else
{
lean_inc(v_a_4703_);
lean_dec(v___x_4543_);
v___x_4705_ = lean_box(0);
v_isShared_4706_ = v_isSharedCheck_4710_;
goto v_resetjp_4704_;
}
v_resetjp_4704_:
{
lean_object* v___x_4708_; 
if (v_isShared_4706_ == 0)
{
v___x_4708_ = v___x_4705_;
goto v_reusejp_4707_;
}
else
{
lean_object* v_reuseFailAlloc_4709_; 
v_reuseFailAlloc_4709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4709_, 0, v_a_4703_);
v___x_4708_ = v_reuseFailAlloc_4709_;
goto v_reusejp_4707_;
}
v_reusejp_4707_:
{
return v___x_4708_;
}
}
}
v___jp_4484_:
{
lean_object* v___x_4496_; lean_object* v___x_4497_; lean_object* v___x_4498_; 
v___x_4496_ = lean_unsigned_to_nat(1u);
v___x_4497_ = lean_nat_add(v_depth_4474_, v___x_4496_);
lean_inc_ref(v___y_4488_);
lean_inc(v_mvarCounterSaved_4475_);
v___x_4498_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_4497_, v_mvarCounterSaved_4475_, v___y_4488_, v___y_4493_, v___y_4485_, v___y_4490_, v___y_4492_, v___y_4487_, v___y_4489_);
if (lean_obj_tag(v___x_4498_) == 0)
{
lean_object* v_a_4499_; lean_object* v___x_4500_; 
v_a_4499_ = lean_ctor_get(v___x_4498_, 0);
lean_inc_n(v_a_4499_, 2);
lean_dec_ref_known(v___x_4498_, 1);
v___x_4500_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_a_4499_, v___y_4490_, v___y_4492_, v___y_4487_, v___y_4489_);
if (lean_obj_tag(v___x_4500_) == 0)
{
lean_object* v_a_4501_; lean_object* v___x_4502_; 
v_a_4501_ = lean_ctor_get(v___x_4500_, 0);
lean_inc(v_a_4501_);
lean_dec_ref_known(v___x_4500_, 1);
lean_inc(v___y_4494_);
v___x_4502_ = l_Lean_Meta_getFunInfoNArgs(v___y_4488_, v___y_4494_, v___y_4490_, v___y_4492_, v___y_4487_, v___y_4489_);
if (lean_obj_tag(v___x_4502_) == 0)
{
lean_object* v_a_4503_; lean_object* v_lhs_4504_; lean_object* v_rhs_4505_; lean_object* v___x_4506_; lean_object* v___x_4507_; lean_object* v___x_4508_; lean_object* v___x_4509_; 
v_a_4503_ = lean_ctor_get(v___x_4502_, 0);
lean_inc(v_a_4503_);
lean_dec_ref_known(v___x_4502_, 1);
v_lhs_4504_ = lean_ctor_get(v_a_4499_, 0);
lean_inc_ref(v_lhs_4504_);
v_rhs_4505_ = lean_ctor_get(v_a_4499_, 1);
lean_inc_ref(v_rhs_4505_);
lean_dec(v_a_4499_);
lean_inc_ref(v___y_4486_);
lean_inc(v___y_4495_);
v___x_4506_ = lean_mk_array(v___y_4495_, v___y_4486_);
v___x_4507_ = l___private_Lean_Expr_0__Lean_Expr_getBoundedAppArgsAux(v_rhs_4477_, v___x_4506_, v___y_4495_);
v___x_4508_ = lean_unsigned_to_nat(0u);
v___x_4509_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go(v_depth_4474_, v_mvarCounterSaved_4475_, v___y_4494_, v___y_4491_, v___x_4507_, v___x_4508_, v_a_4503_, v___x_4508_, v_lhs_4504_, v_rhs_4505_, v_a_4501_, v___y_4485_, v___y_4490_, v___y_4492_, v___y_4487_, v___y_4489_);
return v___x_4509_;
}
else
{
lean_object* v_a_4510_; lean_object* v___x_4512_; uint8_t v_isShared_4513_; uint8_t v_isSharedCheck_4517_; 
lean_dec(v_a_4501_);
lean_dec(v_a_4499_);
lean_dec(v___y_4495_);
lean_dec(v___y_4494_);
lean_dec_ref(v___y_4491_);
lean_dec_ref(v_rhs_4477_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4510_ = lean_ctor_get(v___x_4502_, 0);
v_isSharedCheck_4517_ = !lean_is_exclusive(v___x_4502_);
if (v_isSharedCheck_4517_ == 0)
{
v___x_4512_ = v___x_4502_;
v_isShared_4513_ = v_isSharedCheck_4517_;
goto v_resetjp_4511_;
}
else
{
lean_inc(v_a_4510_);
lean_dec(v___x_4502_);
v___x_4512_ = lean_box(0);
v_isShared_4513_ = v_isSharedCheck_4517_;
goto v_resetjp_4511_;
}
v_resetjp_4511_:
{
lean_object* v___x_4515_; 
if (v_isShared_4513_ == 0)
{
v___x_4515_ = v___x_4512_;
goto v_reusejp_4514_;
}
else
{
lean_object* v_reuseFailAlloc_4516_; 
v_reuseFailAlloc_4516_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4516_, 0, v_a_4510_);
v___x_4515_ = v_reuseFailAlloc_4516_;
goto v_reusejp_4514_;
}
v_reusejp_4514_:
{
return v___x_4515_;
}
}
}
}
else
{
lean_object* v_a_4518_; lean_object* v___x_4520_; uint8_t v_isShared_4521_; uint8_t v_isSharedCheck_4525_; 
lean_dec(v_a_4499_);
lean_dec(v___y_4495_);
lean_dec(v___y_4494_);
lean_dec_ref(v___y_4491_);
lean_dec_ref(v___y_4488_);
lean_dec_ref(v_rhs_4477_);
lean_dec(v_mvarCounterSaved_4475_);
v_a_4518_ = lean_ctor_get(v___x_4500_, 0);
v_isSharedCheck_4525_ = !lean_is_exclusive(v___x_4500_);
if (v_isSharedCheck_4525_ == 0)
{
v___x_4520_ = v___x_4500_;
v_isShared_4521_ = v_isSharedCheck_4525_;
goto v_resetjp_4519_;
}
else
{
lean_inc(v_a_4518_);
lean_dec(v___x_4500_);
v___x_4520_ = lean_box(0);
v_isShared_4521_ = v_isSharedCheck_4525_;
goto v_resetjp_4519_;
}
v_resetjp_4519_:
{
lean_object* v___x_4523_; 
if (v_isShared_4521_ == 0)
{
v___x_4523_ = v___x_4520_;
goto v_reusejp_4522_;
}
else
{
lean_object* v_reuseFailAlloc_4524_; 
v_reuseFailAlloc_4524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4524_, 0, v_a_4518_);
v___x_4523_ = v_reuseFailAlloc_4524_;
goto v_reusejp_4522_;
}
v_reusejp_4522_:
{
return v___x_4523_;
}
}
}
}
else
{
lean_dec(v___y_4495_);
lean_dec(v___y_4494_);
lean_dec_ref(v___y_4491_);
lean_dec_ref(v___y_4488_);
lean_dec_ref(v_rhs_4477_);
lean_dec(v_mvarCounterSaved_4475_);
return v___x_4498_;
}
}
v___jp_4526_:
{
lean_object* v___x_4538_; lean_object* v___x_4539_; uint8_t v___x_4540_; 
lean_inc_ref(v___y_4528_);
lean_inc(v___y_4537_);
v___x_4538_ = lean_mk_array(v___y_4537_, v___y_4528_);
v___x_4539_ = l___private_Lean_Expr_0__Lean_Expr_getBoundedAppArgsAux(v_lhs_4476_, v___x_4538_, v___y_4537_);
v___x_4540_ = lean_nat_dec_le(v___y_4536_, v___y_4533_);
if (v___x_4540_ == 0)
{
v___y_4485_ = v___y_4527_;
v___y_4486_ = v___y_4528_;
v___y_4487_ = v___y_4529_;
v___y_4488_ = v___y_4530_;
v___y_4489_ = v___y_4531_;
v___y_4490_ = v___y_4532_;
v___y_4491_ = v___x_4539_;
v___y_4492_ = v___y_4535_;
v___y_4493_ = v___y_4534_;
v___y_4494_ = v___y_4536_;
v___y_4495_ = v___y_4533_;
goto v___jp_4484_;
}
else
{
lean_dec(v___y_4533_);
lean_inc(v___y_4536_);
v___y_4485_ = v___y_4527_;
v___y_4486_ = v___y_4528_;
v___y_4487_ = v___y_4529_;
v___y_4488_ = v___y_4530_;
v___y_4489_ = v___y_4531_;
v___y_4490_ = v___y_4532_;
v___y_4491_ = v___x_4539_;
v___y_4492_ = v___y_4535_;
v___y_4493_ = v___y_4534_;
v___y_4494_ = v___y_4536_;
v___y_4495_ = v___y_4536_;
goto v___jp_4484_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__6(void){
_start:
{
lean_object* v___x_4712_; lean_object* v___x_4713_; 
v___x_4712_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__5));
v___x_4713_ = l_Lean_stringToMessageData(v___x_4712_);
return v___x_4713_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__8(void){
_start:
{
lean_object* v___x_4715_; lean_object* v___x_4716_; 
v___x_4715_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__7));
v___x_4716_ = l_Lean_stringToMessageData(v___x_4715_);
return v___x_4716_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__10(void){
_start:
{
lean_object* v___x_4718_; lean_object* v___x_4719_; 
v___x_4718_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__9));
v___x_4719_ = l_Lean_stringToMessageData(v___x_4718_);
return v___x_4719_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__12(void){
_start:
{
lean_object* v___x_4721_; lean_object* v___x_4722_; 
v___x_4721_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__11));
v___x_4722_ = l_Lean_stringToMessageData(v___x_4721_);
return v___x_4722_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__14(void){
_start:
{
lean_object* v___x_4724_; lean_object* v___x_4725_; 
v___x_4724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__13));
v___x_4725_ = l_Lean_stringToMessageData(v___x_4724_);
return v___x_4725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3(lean_object* v_a_4726_, lean_object* v_depth_4727_, lean_object* v_mvarCounterSaved_4728_, lean_object* v_a_4729_, lean_object* v___f_4730_, lean_object* v_cls_4731_, uint8_t v___x_4732_, lean_object* v_____r_4733_, lean_object* v___y_4734_, lean_object* v___y_4735_, lean_object* v___y_4736_, lean_object* v___y_4737_, lean_object* v___y_4738_){
_start:
{
lean_object* v___y_4741_; lean_object* v___y_4742_; lean_object* v___y_4743_; lean_object* v___y_4747_; lean_object* v___y_4748_; lean_object* v___y_4749_; lean_object* v___y_4750_; lean_object* v___y_4751_; lean_object* v___y_4757_; lean_object* v___y_4758_; lean_object* v___y_4759_; lean_object* v___y_4760_; lean_object* v___y_4773_; lean_object* v___y_4774_; lean_object* v___y_4779_; lean_object* v___y_4780_; lean_object* v___y_4781_; lean_object* v___y_4782_; lean_object* v___y_4783_; lean_object* v___y_4789_; lean_object* v___y_4790_; lean_object* v___y_4791_; lean_object* v___y_4792_; lean_object* v___y_4793_; lean_object* v___y_4794_; lean_object* v___y_4795_; lean_object* v___y_4796_; lean_object* v___y_4797_; lean_object* v___y_4798_; lean_object* v___y_4799_; lean_object* v___y_4812_; lean_object* v___y_4813_; lean_object* v___y_4814_; lean_object* v___y_4815_; lean_object* v___y_4816_; lean_object* v___y_4817_; lean_object* v___y_4818_; lean_object* v___y_4819_; lean_object* v___y_4820_; lean_object* v___y_4821_; lean_object* v___y_4822_; uint8_t v___y_4823_; lean_object* v___y_4835_; lean_object* v___y_4836_; lean_object* v___y_4837_; lean_object* v___y_4838_; lean_object* v___y_4839_; lean_object* v___y_4840_; lean_object* v___y_4869_; lean_object* v___y_4870_; lean_object* v___y_4871_; lean_object* v___y_4872_; lean_object* v___y_4873_; lean_object* v___y_4874_; lean_object* v___y_4875_; lean_object* v___y_4876_; uint8_t v___y_4877_; lean_object* v___y_4920_; lean_object* v___y_4921_; lean_object* v___y_4922_; lean_object* v___y_4923_; lean_object* v___y_4924_; lean_object* v___y_4925_; lean_object* v___y_4926_; lean_object* v___y_4927_; lean_object* v___y_4928_; uint8_t v___y_4929_; uint8_t v___y_4975_; lean_object* v___y_4976_; lean_object* v___y_4977_; lean_object* v___y_4978_; lean_object* v___y_4979_; lean_object* v___y_4980_; lean_object* v___y_4997_; lean_object* v___x_5132_; 
lean_inc_ref(v_a_4729_);
v___x_5132_ = l_Lean_Meta_isProof(v_a_4729_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_5132_) == 0)
{
lean_object* v_a_5133_; uint8_t v___x_5134_; 
v_a_5133_ = lean_ctor_get(v___x_5132_, 0);
lean_inc(v_a_5133_);
v___x_5134_ = lean_unbox(v_a_5133_);
lean_dec(v_a_5133_);
if (v___x_5134_ == 0)
{
lean_object* v___x_5135_; 
lean_dec_ref_known(v___x_5132_, 1);
lean_inc_ref(v_a_4726_);
v___x_5135_ = l_Lean_Meta_isProof(v_a_4726_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
v___y_4997_ = v___x_5135_;
goto v___jp_4996_;
}
else
{
v___y_4997_ = v___x_5132_;
goto v___jp_4996_;
}
}
else
{
v___y_4997_ = v___x_5132_;
goto v___jp_4996_;
}
v___jp_4740_:
{
lean_object* v___x_4744_; lean_object* v___x_4745_; 
v___x_4744_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4744_, 0, v___y_4742_);
lean_ctor_set(v___x_4744_, 1, v___y_4743_);
lean_ctor_set(v___x_4744_, 2, v___y_4741_);
v___x_4745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4745_, 0, v___x_4744_);
return v___x_4745_;
}
v___jp_4746_:
{
size_t v___x_4752_; size_t v___x_4753_; uint8_t v___x_4754_; 
v___x_4752_ = lean_ptr_addr(v___y_4749_);
lean_dec_ref(v___y_4749_);
v___x_4753_ = lean_ptr_addr(v___y_4750_);
v___x_4754_ = lean_usize_dec_eq(v___x_4752_, v___x_4753_);
if (v___x_4754_ == 0)
{
lean_object* v___x_4755_; 
lean_dec_ref(v_a_4726_);
v___x_4755_ = l_Lean_Expr_mdata___override(v___y_4747_, v___y_4750_);
v___y_4741_ = v___y_4748_;
v___y_4742_ = v___y_4751_;
v___y_4743_ = v___x_4755_;
goto v___jp_4740_;
}
else
{
lean_dec_ref(v___y_4750_);
lean_dec(v___y_4747_);
v___y_4741_ = v___y_4748_;
v___y_4742_ = v___y_4751_;
v___y_4743_ = v_a_4726_;
goto v___jp_4740_;
}
}
v___jp_4756_:
{
lean_object* v___x_4761_; lean_object* v___x_4762_; lean_object* v___x_4763_; 
v___x_4761_ = lean_unsigned_to_nat(1u);
v___x_4762_ = lean_nat_add(v_depth_4727_, v___x_4761_);
lean_inc_ref(v___y_4759_);
lean_inc_ref(v___y_4760_);
v___x_4763_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_4762_, v_mvarCounterSaved_4728_, v___y_4760_, v___y_4759_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_4763_) == 0)
{
lean_object* v_a_4764_; lean_object* v_lhs_4765_; lean_object* v_rhs_4766_; lean_object* v_pf_x3f_4767_; size_t v___x_4768_; size_t v___x_4769_; uint8_t v___x_4770_; 
v_a_4764_ = lean_ctor_get(v___x_4763_, 0);
lean_inc(v_a_4764_);
lean_dec_ref_known(v___x_4763_, 1);
v_lhs_4765_ = lean_ctor_get(v_a_4764_, 0);
lean_inc_ref(v_lhs_4765_);
v_rhs_4766_ = lean_ctor_get(v_a_4764_, 1);
lean_inc_ref(v_rhs_4766_);
v_pf_x3f_4767_ = lean_ctor_get(v_a_4764_, 2);
lean_inc(v_pf_x3f_4767_);
lean_dec(v_a_4764_);
v___x_4768_ = lean_ptr_addr(v___y_4760_);
lean_dec_ref(v___y_4760_);
v___x_4769_ = lean_ptr_addr(v_lhs_4765_);
v___x_4770_ = lean_usize_dec_eq(v___x_4768_, v___x_4769_);
if (v___x_4770_ == 0)
{
lean_object* v___x_4771_; 
lean_dec_ref(v_a_4729_);
v___x_4771_ = l_Lean_Expr_mdata___override(v___y_4758_, v_lhs_4765_);
v___y_4747_ = v___y_4757_;
v___y_4748_ = v_pf_x3f_4767_;
v___y_4749_ = v___y_4759_;
v___y_4750_ = v_rhs_4766_;
v___y_4751_ = v___x_4771_;
goto v___jp_4746_;
}
else
{
lean_dec_ref(v_lhs_4765_);
lean_dec(v___y_4758_);
v___y_4747_ = v___y_4757_;
v___y_4748_ = v_pf_x3f_4767_;
v___y_4749_ = v___y_4759_;
v___y_4750_ = v_rhs_4766_;
v___y_4751_ = v_a_4729_;
goto v___jp_4746_;
}
}
else
{
lean_dec_ref(v___y_4760_);
lean_dec_ref(v___y_4759_);
lean_dec(v___y_4758_);
lean_dec(v___y_4757_);
lean_dec_ref(v_a_4729_);
lean_dec_ref(v_a_4726_);
return v___x_4763_;
}
}
v___jp_4772_:
{
lean_object* v___x_4775_; lean_object* v___x_4776_; lean_object* v___x_4777_; 
v___x_4775_ = lean_box(0);
v___x_4776_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4776_, 0, v___y_4773_);
lean_ctor_set(v___x_4776_, 1, v___y_4774_);
lean_ctor_set(v___x_4776_, 2, v___x_4775_);
v___x_4777_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4777_, 0, v___x_4776_);
return v___x_4777_;
}
v___jp_4778_:
{
size_t v___x_4784_; size_t v___x_4785_; uint8_t v___x_4786_; 
v___x_4784_ = lean_ptr_addr(v___y_4780_);
lean_dec_ref(v___y_4780_);
v___x_4785_ = lean_ptr_addr(v___y_4781_);
v___x_4786_ = lean_usize_dec_eq(v___x_4784_, v___x_4785_);
if (v___x_4786_ == 0)
{
lean_object* v___x_4787_; 
lean_dec_ref(v_a_4726_);
v___x_4787_ = l_Lean_Expr_proj___override(v___y_4782_, v___y_4779_, v___y_4781_);
v___y_4773_ = v___y_4783_;
v___y_4774_ = v___x_4787_;
goto v___jp_4772_;
}
else
{
lean_dec(v___y_4782_);
lean_dec_ref(v___y_4781_);
lean_dec(v___y_4779_);
v___y_4773_ = v___y_4783_;
v___y_4774_ = v_a_4726_;
goto v___jp_4772_;
}
}
v___jp_4788_:
{
lean_object* v___x_4800_; lean_object* v___x_4801_; lean_object* v___x_4802_; 
v___x_4800_ = lean_unsigned_to_nat(1u);
v___x_4801_ = lean_nat_add(v_depth_4727_, v___x_4800_);
lean_inc_ref(v___y_4791_);
lean_inc_ref(v___y_4792_);
v___x_4802_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_4801_, v_mvarCounterSaved_4728_, v___y_4792_, v___y_4791_, v___y_4795_, v___y_4796_, v___y_4797_, v___y_4798_, v___y_4799_);
if (lean_obj_tag(v___x_4802_) == 0)
{
lean_object* v_a_4803_; lean_object* v___x_4804_; 
v_a_4803_ = lean_ctor_get(v___x_4802_, 0);
lean_inc_n(v_a_4803_, 2);
lean_dec_ref_known(v___x_4802_, 1);
v___x_4804_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq(v_a_4803_, v___y_4796_, v___y_4797_, v___y_4798_, v___y_4799_);
if (lean_obj_tag(v___x_4804_) == 0)
{
lean_object* v_lhs_4805_; lean_object* v_rhs_4806_; size_t v___x_4807_; size_t v___x_4808_; uint8_t v___x_4809_; 
lean_dec_ref_known(v___x_4804_, 1);
v_lhs_4805_ = lean_ctor_get(v_a_4803_, 0);
lean_inc_ref(v_lhs_4805_);
v_rhs_4806_ = lean_ctor_get(v_a_4803_, 1);
lean_inc_ref(v_rhs_4806_);
lean_dec(v_a_4803_);
v___x_4807_ = lean_ptr_addr(v___y_4792_);
lean_dec_ref(v___y_4792_);
v___x_4808_ = lean_ptr_addr(v_lhs_4805_);
v___x_4809_ = lean_usize_dec_eq(v___x_4807_, v___x_4808_);
if (v___x_4809_ == 0)
{
lean_object* v___x_4810_; 
lean_dec_ref(v_a_4729_);
v___x_4810_ = l_Lean_Expr_proj___override(v___y_4789_, v___y_4793_, v_lhs_4805_);
v___y_4779_ = v___y_4790_;
v___y_4780_ = v___y_4791_;
v___y_4781_ = v_rhs_4806_;
v___y_4782_ = v___y_4794_;
v___y_4783_ = v___x_4810_;
goto v___jp_4778_;
}
else
{
lean_dec_ref(v_lhs_4805_);
lean_dec(v___y_4793_);
lean_dec(v___y_4789_);
v___y_4779_ = v___y_4790_;
v___y_4780_ = v___y_4791_;
v___y_4781_ = v_rhs_4806_;
v___y_4782_ = v___y_4794_;
v___y_4783_ = v_a_4729_;
goto v___jp_4778_;
}
}
else
{
lean_dec(v_a_4803_);
lean_dec(v___y_4794_);
lean_dec(v___y_4793_);
lean_dec_ref(v___y_4792_);
lean_dec_ref(v___y_4791_);
lean_dec(v___y_4790_);
lean_dec(v___y_4789_);
lean_dec_ref(v_a_4729_);
lean_dec_ref(v_a_4726_);
return v___x_4804_;
}
}
else
{
lean_dec(v___y_4794_);
lean_dec(v___y_4793_);
lean_dec_ref(v___y_4792_);
lean_dec_ref(v___y_4791_);
lean_dec(v___y_4790_);
lean_dec(v___y_4789_);
lean_dec_ref(v_a_4729_);
lean_dec_ref(v_a_4726_);
return v___x_4802_;
}
}
v___jp_4811_:
{
if (v___y_4823_ == 0)
{
lean_object* v___x_4824_; lean_object* v___x_4825_; 
v___x_4824_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__2);
lean_inc_ref(v_a_4726_);
lean_inc_ref(v_a_4729_);
v___x_4825_ = lp_mathlib_Mathlib_Tactic_TermCongr_throwCongrEx___redArg(v_a_4729_, v_a_4726_, v___x_4824_, v___y_4812_, v___y_4818_, v___y_4814_, v___y_4822_);
if (lean_obj_tag(v___x_4825_) == 0)
{
lean_dec_ref_known(v___x_4825_, 1);
v___y_4789_ = v___y_4813_;
v___y_4790_ = v___y_4815_;
v___y_4791_ = v___y_4817_;
v___y_4792_ = v___y_4819_;
v___y_4793_ = v___y_4820_;
v___y_4794_ = v___y_4821_;
v___y_4795_ = v___y_4816_;
v___y_4796_ = v___y_4812_;
v___y_4797_ = v___y_4818_;
v___y_4798_ = v___y_4814_;
v___y_4799_ = v___y_4822_;
goto v___jp_4788_;
}
else
{
lean_object* v_a_4826_; lean_object* v___x_4828_; uint8_t v_isShared_4829_; uint8_t v_isSharedCheck_4833_; 
lean_dec(v___y_4821_);
lean_dec(v___y_4820_);
lean_dec_ref(v___y_4819_);
lean_dec_ref(v___y_4817_);
lean_dec(v___y_4815_);
lean_dec(v___y_4813_);
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
v_a_4826_ = lean_ctor_get(v___x_4825_, 0);
v_isSharedCheck_4833_ = !lean_is_exclusive(v___x_4825_);
if (v_isSharedCheck_4833_ == 0)
{
v___x_4828_ = v___x_4825_;
v_isShared_4829_ = v_isSharedCheck_4833_;
goto v_resetjp_4827_;
}
else
{
lean_inc(v_a_4826_);
lean_dec(v___x_4825_);
v___x_4828_ = lean_box(0);
v_isShared_4829_ = v_isSharedCheck_4833_;
goto v_resetjp_4827_;
}
v_resetjp_4827_:
{
lean_object* v___x_4831_; 
if (v_isShared_4829_ == 0)
{
v___x_4831_ = v___x_4828_;
goto v_reusejp_4830_;
}
else
{
lean_object* v_reuseFailAlloc_4832_; 
v_reuseFailAlloc_4832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4832_, 0, v_a_4826_);
v___x_4831_ = v_reuseFailAlloc_4832_;
goto v_reusejp_4830_;
}
v_reusejp_4830_:
{
return v___x_4831_;
}
}
}
}
else
{
v___y_4789_ = v___y_4813_;
v___y_4790_ = v___y_4815_;
v___y_4791_ = v___y_4817_;
v___y_4792_ = v___y_4819_;
v___y_4793_ = v___y_4820_;
v___y_4794_ = v___y_4821_;
v___y_4795_ = v___y_4816_;
v___y_4796_ = v___y_4812_;
v___y_4797_ = v___y_4818_;
v___y_4798_ = v___y_4814_;
v___y_4799_ = v___y_4822_;
goto v___jp_4788_;
}
}
v___jp_4834_:
{
uint8_t v___x_4841_; 
v___x_4841_ = lean_name_eq(v___y_4835_, v___y_4840_);
if (v___x_4841_ == 0)
{
v___y_4812_ = v___y_4735_;
v___y_4813_ = v___y_4835_;
v___y_4814_ = v___y_4737_;
v___y_4815_ = v___y_4836_;
v___y_4816_ = v___y_4734_;
v___y_4817_ = v___y_4837_;
v___y_4818_ = v___y_4736_;
v___y_4819_ = v___y_4838_;
v___y_4820_ = v___y_4839_;
v___y_4821_ = v___y_4840_;
v___y_4822_ = v___y_4738_;
v___y_4823_ = v___x_4841_;
goto v___jp_4811_;
}
else
{
uint8_t v___x_4842_; 
v___x_4842_ = lean_nat_dec_eq(v___y_4839_, v___y_4836_);
v___y_4812_ = v___y_4735_;
v___y_4813_ = v___y_4835_;
v___y_4814_ = v___y_4737_;
v___y_4815_ = v___y_4836_;
v___y_4816_ = v___y_4734_;
v___y_4817_ = v___y_4837_;
v___y_4818_ = v___y_4736_;
v___y_4819_ = v___y_4838_;
v___y_4820_ = v___y_4839_;
v___y_4821_ = v___y_4840_;
v___y_4822_ = v___y_4738_;
v___y_4823_ = v___x_4842_;
goto v___jp_4811_;
}
}
v___jp_4843_:
{
lean_object* v_inheritedTraceOptions_4844_; lean_object* v___x_4845_; 
v_inheritedTraceOptions_4844_ = lean_ctor_get(v___y_4737_, 13);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
lean_inc(v___y_4736_);
lean_inc_ref(v___y_4735_);
lean_inc(v___y_4734_);
lean_inc_ref(v_inheritedTraceOptions_4844_);
v___x_4845_ = lean_apply_7(v___f_4730_, v_inheritedTraceOptions_4844_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_, lean_box(0));
if (lean_obj_tag(v___x_4845_) == 0)
{
lean_object* v_a_4846_; uint8_t v___x_4847_; 
v_a_4846_ = lean_ctor_get(v___x_4845_, 0);
lean_inc(v_a_4846_);
lean_dec_ref_known(v___x_4845_, 1);
v___x_4847_ = lean_unbox(v_a_4846_);
lean_dec(v_a_4846_);
if (v___x_4847_ == 0)
{
lean_object* v___x_4848_; 
lean_dec(v_cls_4731_);
v___x_4848_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_4728_, v_a_4729_, v_a_4726_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
return v___x_4848_;
}
else
{
lean_object* v___x_4849_; lean_object* v___x_4850_; 
v___x_4849_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__4, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__4);
v___x_4850_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4731_, v___x_4849_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_4850_) == 0)
{
lean_object* v___x_4851_; 
lean_dec_ref_known(v___x_4850_, 1);
v___x_4851_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault_x27(v_mvarCounterSaved_4728_, v_a_4729_, v_a_4726_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
return v___x_4851_;
}
else
{
lean_object* v_a_4852_; lean_object* v___x_4854_; uint8_t v_isShared_4855_; uint8_t v_isSharedCheck_4859_; 
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
v_a_4852_ = lean_ctor_get(v___x_4850_, 0);
v_isSharedCheck_4859_ = !lean_is_exclusive(v___x_4850_);
if (v_isSharedCheck_4859_ == 0)
{
v___x_4854_ = v___x_4850_;
v_isShared_4855_ = v_isSharedCheck_4859_;
goto v_resetjp_4853_;
}
else
{
lean_inc(v_a_4852_);
lean_dec(v___x_4850_);
v___x_4854_ = lean_box(0);
v_isShared_4855_ = v_isSharedCheck_4859_;
goto v_resetjp_4853_;
}
v_resetjp_4853_:
{
lean_object* v___x_4857_; 
if (v_isShared_4855_ == 0)
{
v___x_4857_ = v___x_4854_;
goto v_reusejp_4856_;
}
else
{
lean_object* v_reuseFailAlloc_4858_; 
v_reuseFailAlloc_4858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4858_, 0, v_a_4852_);
v___x_4857_ = v_reuseFailAlloc_4858_;
goto v_reusejp_4856_;
}
v_reusejp_4856_:
{
return v___x_4857_;
}
}
}
}
}
else
{
lean_object* v_a_4860_; lean_object* v___x_4862_; uint8_t v_isShared_4863_; uint8_t v_isSharedCheck_4867_; 
lean_dec(v_cls_4731_);
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
v_a_4860_ = lean_ctor_get(v___x_4845_, 0);
v_isSharedCheck_4867_ = !lean_is_exclusive(v___x_4845_);
if (v_isSharedCheck_4867_ == 0)
{
v___x_4862_ = v___x_4845_;
v_isShared_4863_ = v_isSharedCheck_4867_;
goto v_resetjp_4861_;
}
else
{
lean_inc(v_a_4860_);
lean_dec(v___x_4845_);
v___x_4862_ = lean_box(0);
v_isShared_4863_ = v_isSharedCheck_4867_;
goto v_resetjp_4861_;
}
v_resetjp_4861_:
{
lean_object* v___x_4865_; 
if (v_isShared_4863_ == 0)
{
v___x_4865_ = v___x_4862_;
goto v_reusejp_4864_;
}
else
{
lean_object* v_reuseFailAlloc_4866_; 
v_reuseFailAlloc_4866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4866_, 0, v_a_4860_);
v___x_4865_ = v_reuseFailAlloc_4866_;
goto v_reusejp_4864_;
}
v_reusejp_4864_:
{
return v___x_4865_;
}
}
}
}
v___jp_4868_:
{
if (v___y_4877_ == 0)
{
lean_object* v___x_4878_; 
v___x_4878_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v___y_4871_, v___y_4872_, v___y_4874_, v___y_4869_, v___y_4873_);
if (lean_obj_tag(v___x_4878_) == 0)
{
lean_object* v_a_4879_; lean_object* v___x_4880_; 
v_a_4879_ = lean_ctor_get(v___x_4878_, 0);
lean_inc(v_a_4879_);
lean_dec_ref_known(v___x_4878_, 1);
v___x_4880_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v___y_4870_, v___y_4872_, v___y_4874_, v___y_4869_, v___y_4873_);
if (lean_obj_tag(v___x_4880_) == 0)
{
lean_object* v_a_4881_; lean_object* v___x_4882_; 
v_a_4881_ = lean_ctor_get(v___x_4880_, 0);
lean_inc(v_a_4881_);
lean_dec_ref_known(v___x_4880_, 1);
v___x_4882_ = l_Lean_Meta_mkImpCongr(v_a_4879_, v_a_4881_, v___y_4872_, v___y_4874_, v___y_4869_, v___y_4873_);
if (lean_obj_tag(v___x_4882_) == 0)
{
lean_object* v_a_4883_; lean_object* v___x_4885_; uint8_t v_isShared_4886_; uint8_t v_isSharedCheck_4891_; 
v_a_4883_ = lean_ctor_get(v___x_4882_, 0);
v_isSharedCheck_4891_ = !lean_is_exclusive(v___x_4882_);
if (v_isSharedCheck_4891_ == 0)
{
v___x_4885_ = v___x_4882_;
v_isShared_4886_ = v_isSharedCheck_4891_;
goto v_resetjp_4884_;
}
else
{
lean_inc(v_a_4883_);
lean_dec(v___x_4882_);
v___x_4885_ = lean_box(0);
v_isShared_4886_ = v_isSharedCheck_4891_;
goto v_resetjp_4884_;
}
v_resetjp_4884_:
{
lean_object* v___x_4887_; lean_object* v___x_4889_; 
v___x_4887_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v___y_4876_, v___y_4875_, v_a_4883_);
if (v_isShared_4886_ == 0)
{
lean_ctor_set(v___x_4885_, 0, v___x_4887_);
v___x_4889_ = v___x_4885_;
goto v_reusejp_4888_;
}
else
{
lean_object* v_reuseFailAlloc_4890_; 
v_reuseFailAlloc_4890_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4890_, 0, v___x_4887_);
v___x_4889_ = v_reuseFailAlloc_4890_;
goto v_reusejp_4888_;
}
v_reusejp_4888_:
{
return v___x_4889_;
}
}
}
else
{
lean_object* v_a_4892_; lean_object* v___x_4894_; uint8_t v_isShared_4895_; uint8_t v_isSharedCheck_4899_; 
lean_dec_ref(v___y_4876_);
lean_dec_ref(v___y_4875_);
v_a_4892_ = lean_ctor_get(v___x_4882_, 0);
v_isSharedCheck_4899_ = !lean_is_exclusive(v___x_4882_);
if (v_isSharedCheck_4899_ == 0)
{
v___x_4894_ = v___x_4882_;
v_isShared_4895_ = v_isSharedCheck_4899_;
goto v_resetjp_4893_;
}
else
{
lean_inc(v_a_4892_);
lean_dec(v___x_4882_);
v___x_4894_ = lean_box(0);
v_isShared_4895_ = v_isSharedCheck_4899_;
goto v_resetjp_4893_;
}
v_resetjp_4893_:
{
lean_object* v___x_4897_; 
if (v_isShared_4895_ == 0)
{
v___x_4897_ = v___x_4894_;
goto v_reusejp_4896_;
}
else
{
lean_object* v_reuseFailAlloc_4898_; 
v_reuseFailAlloc_4898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4898_, 0, v_a_4892_);
v___x_4897_ = v_reuseFailAlloc_4898_;
goto v_reusejp_4896_;
}
v_reusejp_4896_:
{
return v___x_4897_;
}
}
}
}
else
{
lean_object* v_a_4900_; lean_object* v___x_4902_; uint8_t v_isShared_4903_; uint8_t v_isSharedCheck_4907_; 
lean_dec(v_a_4879_);
lean_dec_ref(v___y_4876_);
lean_dec_ref(v___y_4875_);
v_a_4900_ = lean_ctor_get(v___x_4880_, 0);
v_isSharedCheck_4907_ = !lean_is_exclusive(v___x_4880_);
if (v_isSharedCheck_4907_ == 0)
{
v___x_4902_ = v___x_4880_;
v_isShared_4903_ = v_isSharedCheck_4907_;
goto v_resetjp_4901_;
}
else
{
lean_inc(v_a_4900_);
lean_dec(v___x_4880_);
v___x_4902_ = lean_box(0);
v_isShared_4903_ = v_isSharedCheck_4907_;
goto v_resetjp_4901_;
}
v_resetjp_4901_:
{
lean_object* v___x_4905_; 
if (v_isShared_4903_ == 0)
{
v___x_4905_ = v___x_4902_;
goto v_reusejp_4904_;
}
else
{
lean_object* v_reuseFailAlloc_4906_; 
v_reuseFailAlloc_4906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4906_, 0, v_a_4900_);
v___x_4905_ = v_reuseFailAlloc_4906_;
goto v_reusejp_4904_;
}
v_reusejp_4904_:
{
return v___x_4905_;
}
}
}
}
else
{
lean_object* v_a_4908_; lean_object* v___x_4910_; uint8_t v_isShared_4911_; uint8_t v_isSharedCheck_4915_; 
lean_dec_ref(v___y_4876_);
lean_dec_ref(v___y_4875_);
lean_dec_ref(v___y_4870_);
v_a_4908_ = lean_ctor_get(v___x_4878_, 0);
v_isSharedCheck_4915_ = !lean_is_exclusive(v___x_4878_);
if (v_isSharedCheck_4915_ == 0)
{
v___x_4910_ = v___x_4878_;
v_isShared_4911_ = v_isSharedCheck_4915_;
goto v_resetjp_4909_;
}
else
{
lean_inc(v_a_4908_);
lean_dec(v___x_4878_);
v___x_4910_ = lean_box(0);
v_isShared_4911_ = v_isSharedCheck_4915_;
goto v_resetjp_4909_;
}
v_resetjp_4909_:
{
lean_object* v___x_4913_; 
if (v_isShared_4911_ == 0)
{
v___x_4913_ = v___x_4910_;
goto v_reusejp_4912_;
}
else
{
lean_object* v_reuseFailAlloc_4914_; 
v_reuseFailAlloc_4914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4914_, 0, v_a_4908_);
v___x_4913_ = v_reuseFailAlloc_4914_;
goto v_reusejp_4912_;
}
v_reusejp_4912_:
{
return v___x_4913_;
}
}
}
}
else
{
lean_object* v___x_4916_; lean_object* v___x_4917_; lean_object* v___x_4918_; 
lean_dec_ref(v___y_4871_);
lean_dec_ref(v___y_4870_);
v___x_4916_ = lean_box(0);
v___x_4917_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4917_, 0, v___y_4876_);
lean_ctor_set(v___x_4917_, 1, v___y_4875_);
lean_ctor_set(v___x_4917_, 2, v___x_4916_);
v___x_4918_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4918_, 0, v___x_4917_);
return v___x_4918_;
}
}
v___jp_4919_:
{
if (v___y_4929_ == 0)
{
lean_object* v___x_4930_; 
lean_dec(v___y_4924_);
lean_inc_ref(v___y_4920_);
v___x_4930_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq(v___y_4920_, v___y_4926_, v___y_4928_, v___y_4923_, v___y_4927_);
if (lean_obj_tag(v___x_4930_) == 0)
{
lean_object* v_lhs_4931_; lean_object* v___x_4932_; lean_object* v___x_4933_; lean_object* v___f_4934_; lean_object* v___x_4935_; uint8_t v___x_4936_; uint8_t v___x_4937_; lean_object* v___x_4938_; 
lean_dec_ref_known(v___x_4930_, 1);
v_lhs_4931_ = lean_ctor_get(v___y_4920_, 0);
lean_inc_ref(v_lhs_4931_);
lean_dec_ref(v___y_4920_);
v___x_4932_ = lean_box(v___y_4929_);
v___x_4933_ = lean_box(v___x_4732_);
lean_inc_ref(v_a_4729_);
v___f_4934_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___boxed), 14, 7);
lean_closure_set(v___f_4934_, 0, v_a_4729_);
lean_closure_set(v___f_4934_, 1, v_a_4726_);
lean_closure_set(v___f_4934_, 2, v___y_4922_);
lean_closure_set(v___f_4934_, 3, v_mvarCounterSaved_4728_);
lean_closure_set(v___f_4934_, 4, v___y_4921_);
lean_closure_set(v___f_4934_, 5, v___x_4932_);
lean_closure_set(v___f_4934_, 6, v___x_4933_);
v___x_4935_ = l_Lean_Expr_bindingName_x21(v_a_4729_);
v___x_4936_ = l_Lean_Expr_bindingInfo_x21(v_a_4729_);
lean_dec_ref(v_a_4729_);
v___x_4937_ = 0;
v___x_4938_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg(v___x_4935_, v___x_4936_, v_lhs_4931_, v___f_4934_, v___x_4937_, v___y_4925_, v___y_4926_, v___y_4928_, v___y_4923_, v___y_4927_);
return v___x_4938_;
}
else
{
lean_dec(v___y_4922_);
lean_dec(v___y_4921_);
lean_dec_ref(v___y_4920_);
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
return v___x_4930_;
}
}
else
{
lean_object* v___x_4939_; lean_object* v___x_4940_; lean_object* v___x_4941_; 
lean_dec(v___y_4922_);
lean_dec(v___y_4921_);
v___x_4939_ = l_Lean_Expr_bindingBody_x21(v_a_4729_);
v___x_4940_ = l_Lean_Expr_bindingBody_x21(v_a_4726_);
v___x_4941_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___y_4924_, v_mvarCounterSaved_4728_, v___x_4939_, v___x_4940_, v___y_4925_, v___y_4926_, v___y_4928_, v___y_4923_, v___y_4927_);
if (lean_obj_tag(v___x_4941_) == 0)
{
lean_object* v_a_4942_; lean_object* v_lhs_4943_; lean_object* v_rhs_4944_; lean_object* v_lhs_4945_; lean_object* v_rhs_4946_; lean_object* v___x_4947_; uint8_t v___x_4948_; lean_object* v___x_4949_; lean_object* v___x_4950_; uint8_t v___x_4951_; lean_object* v___x_4952_; uint8_t v___x_4953_; 
v_a_4942_ = lean_ctor_get(v___x_4941_, 0);
lean_inc(v_a_4942_);
lean_dec_ref_known(v___x_4941_, 1);
v_lhs_4943_ = lean_ctor_get(v___y_4920_, 0);
v_rhs_4944_ = lean_ctor_get(v___y_4920_, 1);
v_lhs_4945_ = lean_ctor_get(v_a_4942_, 0);
v_rhs_4946_ = lean_ctor_get(v_a_4942_, 1);
v___x_4947_ = l_Lean_Expr_bindingName_x21(v_a_4729_);
v___x_4948_ = l_Lean_Expr_bindingInfo_x21(v_a_4729_);
lean_dec_ref(v_a_4729_);
lean_inc_ref(v_lhs_4945_);
lean_inc_ref(v_lhs_4943_);
v___x_4949_ = l_Lean_Expr_forallE___override(v___x_4947_, v_lhs_4943_, v_lhs_4945_, v___x_4948_);
v___x_4950_ = l_Lean_Expr_bindingName_x21(v_a_4726_);
v___x_4951_ = l_Lean_Expr_bindingInfo_x21(v_a_4726_);
lean_dec_ref(v_a_4726_);
lean_inc_ref(v_rhs_4946_);
lean_inc_ref(v_rhs_4944_);
v___x_4952_ = l_Lean_Expr_forallE___override(v___x_4950_, v_rhs_4944_, v_rhs_4946_, v___x_4951_);
v___x_4953_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v___y_4920_);
if (v___x_4953_ == 0)
{
v___y_4869_ = v___y_4923_;
v___y_4870_ = v_a_4942_;
v___y_4871_ = v___y_4920_;
v___y_4872_ = v___y_4926_;
v___y_4873_ = v___y_4927_;
v___y_4874_ = v___y_4928_;
v___y_4875_ = v___x_4952_;
v___y_4876_ = v___x_4949_;
v___y_4877_ = v___x_4953_;
goto v___jp_4868_;
}
else
{
uint8_t v___x_4954_; 
v___x_4954_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_a_4942_);
v___y_4869_ = v___y_4923_;
v___y_4870_ = v_a_4942_;
v___y_4871_ = v___y_4920_;
v___y_4872_ = v___y_4926_;
v___y_4873_ = v___y_4927_;
v___y_4874_ = v___y_4928_;
v___y_4875_ = v___x_4952_;
v___y_4876_ = v___x_4949_;
v___y_4877_ = v___x_4954_;
goto v___jp_4868_;
}
}
else
{
lean_dec_ref(v___y_4920_);
lean_dec_ref(v_a_4729_);
lean_dec_ref(v_a_4726_);
return v___x_4941_;
}
}
}
v___jp_4955_:
{
lean_object* v___x_4956_; lean_object* v___x_4957_; lean_object* v___x_4958_; lean_object* v___x_4959_; lean_object* v___x_4960_; 
v___x_4956_ = lean_unsigned_to_nat(1u);
v___x_4957_ = lean_nat_add(v_depth_4727_, v___x_4956_);
v___x_4958_ = l_Lean_Expr_bindingDomain_x21(v_a_4729_);
v___x_4959_ = l_Lean_Expr_bindingDomain_x21(v_a_4726_);
lean_inc(v_mvarCounterSaved_4728_);
lean_inc(v___x_4957_);
v___x_4960_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_4957_, v_mvarCounterSaved_4728_, v___x_4958_, v___x_4959_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_4960_) == 0)
{
lean_object* v_a_4961_; uint8_t v___x_4962_; 
v_a_4961_ = lean_ctor_get(v___x_4960_, 0);
lean_inc(v_a_4961_);
lean_dec_ref_known(v___x_4960_, 1);
v___x_4962_ = l_Lean_Expr_isArrow(v_a_4729_);
if (v___x_4962_ == 0)
{
lean_inc(v___x_4957_);
v___y_4920_ = v_a_4961_;
v___y_4921_ = v___x_4956_;
v___y_4922_ = v___x_4957_;
v___y_4923_ = v___y_4737_;
v___y_4924_ = v___x_4957_;
v___y_4925_ = v___y_4734_;
v___y_4926_ = v___y_4735_;
v___y_4927_ = v___y_4738_;
v___y_4928_ = v___y_4736_;
v___y_4929_ = v___x_4962_;
goto v___jp_4919_;
}
else
{
uint8_t v___x_4963_; 
v___x_4963_ = l_Lean_Expr_isArrow(v_a_4726_);
lean_inc(v___x_4957_);
v___y_4920_ = v_a_4961_;
v___y_4921_ = v___x_4956_;
v___y_4922_ = v___x_4957_;
v___y_4923_ = v___y_4737_;
v___y_4924_ = v___x_4957_;
v___y_4925_ = v___y_4734_;
v___y_4926_ = v___y_4735_;
v___y_4927_ = v___y_4738_;
v___y_4928_ = v___y_4736_;
v___y_4929_ = v___x_4963_;
goto v___jp_4919_;
}
}
else
{
lean_dec(v___x_4957_);
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
return v___x_4960_;
}
}
v___jp_4964_:
{
lean_object* v___x_4965_; lean_object* v___x_4966_; lean_object* v___x_4967_; lean_object* v___x_4968_; lean_object* v___x_4969_; lean_object* v___x_4970_; lean_object* v___x_4971_; lean_object* v___x_4972_; lean_object* v___x_4973_; 
v___x_4965_ = l_Lean_Expr_letBody_x21(v_a_4729_);
v___x_4966_ = l_Lean_Expr_letValue_x21(v_a_4729_);
lean_dec_ref(v_a_4729_);
v___x_4967_ = lean_expr_instantiate1(v___x_4965_, v___x_4966_);
lean_dec_ref(v___x_4966_);
lean_dec_ref(v___x_4965_);
v___x_4968_ = l_Lean_Expr_letBody_x21(v_a_4726_);
v___x_4969_ = l_Lean_Expr_letValue_x21(v_a_4726_);
lean_dec_ref(v_a_4726_);
v___x_4970_ = lean_expr_instantiate1(v___x_4968_, v___x_4969_);
lean_dec_ref(v___x_4969_);
lean_dec_ref(v___x_4968_);
v___x_4971_ = lean_unsigned_to_nat(1u);
v___x_4972_ = lean_nat_add(v_depth_4727_, v___x_4971_);
v___x_4973_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_4972_, v_mvarCounterSaved_4728_, v___x_4967_, v___x_4970_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
return v___x_4973_;
}
v___jp_4974_:
{
lean_object* v___x_4981_; lean_object* v___x_4982_; lean_object* v___x_4983_; lean_object* v___x_4984_; lean_object* v___x_4985_; 
v___x_4981_ = lean_unsigned_to_nat(1u);
v___x_4982_ = lean_nat_add(v_depth_4727_, v___x_4981_);
v___x_4983_ = l_Lean_Expr_bindingDomain_x21(v_a_4729_);
v___x_4984_ = l_Lean_Expr_bindingDomain_x21(v_a_4726_);
lean_inc(v_mvarCounterSaved_4728_);
lean_inc(v___x_4982_);
v___x_4985_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_4982_, v_mvarCounterSaved_4728_, v___x_4983_, v___x_4984_, v___y_4976_, v___y_4977_, v___y_4978_, v___y_4979_, v___y_4980_);
if (lean_obj_tag(v___x_4985_) == 0)
{
lean_object* v_a_4986_; lean_object* v___x_4987_; 
v_a_4986_ = lean_ctor_get(v___x_4985_, 0);
lean_inc_n(v_a_4986_, 2);
lean_dec_ref_known(v___x_4985_, 1);
v___x_4987_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_defeq(v_a_4986_, v___y_4977_, v___y_4978_, v___y_4979_, v___y_4980_);
if (lean_obj_tag(v___x_4987_) == 0)
{
lean_object* v_lhs_4988_; lean_object* v___x_4989_; lean_object* v___x_4990_; lean_object* v___f_4991_; lean_object* v___x_4992_; uint8_t v___x_4993_; uint8_t v___x_4994_; lean_object* v___x_4995_; 
lean_dec_ref_known(v___x_4987_, 1);
v_lhs_4988_ = lean_ctor_get(v_a_4986_, 0);
lean_inc_ref(v_lhs_4988_);
lean_dec(v_a_4986_);
v___x_4989_ = lean_box(v___y_4975_);
v___x_4990_ = lean_box(v___x_4732_);
lean_inc_ref(v_a_4729_);
v___f_4991_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__2___boxed), 14, 7);
lean_closure_set(v___f_4991_, 0, v_a_4729_);
lean_closure_set(v___f_4991_, 1, v_a_4726_);
lean_closure_set(v___f_4991_, 2, v___x_4982_);
lean_closure_set(v___f_4991_, 3, v_mvarCounterSaved_4728_);
lean_closure_set(v___f_4991_, 4, v___x_4981_);
lean_closure_set(v___f_4991_, 5, v___x_4989_);
lean_closure_set(v___f_4991_, 6, v___x_4990_);
v___x_4992_ = l_Lean_Expr_bindingName_x21(v_a_4729_);
v___x_4993_ = l_Lean_Expr_bindingInfo_x21(v_a_4729_);
lean_dec_ref(v_a_4729_);
v___x_4994_ = 0;
v___x_4995_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg(v___x_4992_, v___x_4993_, v_lhs_4988_, v___f_4991_, v___x_4994_, v___y_4976_, v___y_4977_, v___y_4978_, v___y_4979_, v___y_4980_);
return v___x_4995_;
}
else
{
lean_dec(v_a_4986_);
lean_dec(v___x_4982_);
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
return v___x_4987_;
}
}
else
{
lean_dec(v___x_4982_);
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
return v___x_4985_;
}
}
v___jp_4996_:
{
if (lean_obj_tag(v___y_4997_) == 0)
{
lean_object* v_a_4998_; uint8_t v___x_4999_; 
v_a_4998_ = lean_ctor_get(v___y_4997_, 0);
lean_inc(v_a_4998_);
lean_dec_ref_known(v___y_4997_, 1);
v___x_4999_ = lean_unbox(v_a_4998_);
if (v___x_4999_ == 0)
{
switch(lean_obj_tag(v_a_4729_))
{
case 5:
{
lean_dec(v_a_4998_);
if (lean_obj_tag(v_a_4726_) == 5)
{
lean_object* v___x_5000_; 
lean_dec(v_cls_4731_);
lean_dec_ref(v___f_4730_);
v___x_5000_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp(v_depth_4727_, v_mvarCounterSaved_4728_, v_a_4729_, v_a_4726_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
return v___x_5000_;
}
else
{
goto v___jp_4843_;
}
}
case 6:
{
if (lean_obj_tag(v_a_4726_) == 6)
{
lean_object* v_inheritedTraceOptions_5001_; lean_object* v___x_5002_; 
v_inheritedTraceOptions_5001_ = lean_ctor_get(v___y_4737_, 13);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
lean_inc(v___y_4736_);
lean_inc_ref(v___y_4735_);
lean_inc(v___y_4734_);
lean_inc_ref(v_inheritedTraceOptions_5001_);
v___x_5002_ = lean_apply_7(v___f_4730_, v_inheritedTraceOptions_5001_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_, lean_box(0));
if (lean_obj_tag(v___x_5002_) == 0)
{
lean_object* v_a_5003_; uint8_t v___x_5004_; 
v_a_5003_ = lean_ctor_get(v___x_5002_, 0);
lean_inc(v_a_5003_);
lean_dec_ref_known(v___x_5002_, 1);
v___x_5004_ = lean_unbox(v_a_5003_);
lean_dec(v_a_5003_);
if (v___x_5004_ == 0)
{
uint8_t v___x_5005_; 
lean_dec(v_cls_4731_);
v___x_5005_ = lean_unbox(v_a_4998_);
lean_dec(v_a_4998_);
v___y_4975_ = v___x_5005_;
v___y_4976_ = v___y_4734_;
v___y_4977_ = v___y_4735_;
v___y_4978_ = v___y_4736_;
v___y_4979_ = v___y_4737_;
v___y_4980_ = v___y_4738_;
goto v___jp_4974_;
}
else
{
lean_object* v___x_5006_; lean_object* v___x_5007_; 
v___x_5006_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__6, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__6);
v___x_5007_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4731_, v___x_5006_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_5007_) == 0)
{
uint8_t v___x_5008_; 
lean_dec_ref_known(v___x_5007_, 1);
v___x_5008_ = lean_unbox(v_a_4998_);
lean_dec(v_a_4998_);
v___y_4975_ = v___x_5008_;
v___y_4976_ = v___y_4734_;
v___y_4977_ = v___y_4735_;
v___y_4978_ = v___y_4736_;
v___y_4979_ = v___y_4737_;
v___y_4980_ = v___y_4738_;
goto v___jp_4974_;
}
else
{
lean_object* v_a_5009_; lean_object* v___x_5011_; uint8_t v_isShared_5012_; uint8_t v_isSharedCheck_5016_; 
lean_dec_ref_known(v_a_4726_, 3);
lean_dec_ref_known(v_a_4729_, 3);
lean_dec(v_a_4998_);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5009_ = lean_ctor_get(v___x_5007_, 0);
v_isSharedCheck_5016_ = !lean_is_exclusive(v___x_5007_);
if (v_isSharedCheck_5016_ == 0)
{
v___x_5011_ = v___x_5007_;
v_isShared_5012_ = v_isSharedCheck_5016_;
goto v_resetjp_5010_;
}
else
{
lean_inc(v_a_5009_);
lean_dec(v___x_5007_);
v___x_5011_ = lean_box(0);
v_isShared_5012_ = v_isSharedCheck_5016_;
goto v_resetjp_5010_;
}
v_resetjp_5010_:
{
lean_object* v___x_5014_; 
if (v_isShared_5012_ == 0)
{
v___x_5014_ = v___x_5011_;
goto v_reusejp_5013_;
}
else
{
lean_object* v_reuseFailAlloc_5015_; 
v_reuseFailAlloc_5015_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5015_, 0, v_a_5009_);
v___x_5014_ = v_reuseFailAlloc_5015_;
goto v_reusejp_5013_;
}
v_reusejp_5013_:
{
return v___x_5014_;
}
}
}
}
}
else
{
lean_object* v_a_5017_; lean_object* v___x_5019_; uint8_t v_isShared_5020_; uint8_t v_isSharedCheck_5024_; 
lean_dec_ref_known(v_a_4726_, 3);
lean_dec_ref_known(v_a_4729_, 3);
lean_dec(v_a_4998_);
lean_dec(v_cls_4731_);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5017_ = lean_ctor_get(v___x_5002_, 0);
v_isSharedCheck_5024_ = !lean_is_exclusive(v___x_5002_);
if (v_isSharedCheck_5024_ == 0)
{
v___x_5019_ = v___x_5002_;
v_isShared_5020_ = v_isSharedCheck_5024_;
goto v_resetjp_5018_;
}
else
{
lean_inc(v_a_5017_);
lean_dec(v___x_5002_);
v___x_5019_ = lean_box(0);
v_isShared_5020_ = v_isSharedCheck_5024_;
goto v_resetjp_5018_;
}
v_resetjp_5018_:
{
lean_object* v___x_5022_; 
if (v_isShared_5020_ == 0)
{
v___x_5022_ = v___x_5019_;
goto v_reusejp_5021_;
}
else
{
lean_object* v_reuseFailAlloc_5023_; 
v_reuseFailAlloc_5023_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5023_, 0, v_a_5017_);
v___x_5022_ = v_reuseFailAlloc_5023_;
goto v_reusejp_5021_;
}
v_reusejp_5021_:
{
return v___x_5022_;
}
}
}
}
else
{
lean_dec(v_a_4998_);
goto v___jp_4843_;
}
}
case 7:
{
lean_dec(v_a_4998_);
if (lean_obj_tag(v_a_4726_) == 7)
{
lean_object* v_inheritedTraceOptions_5025_; lean_object* v___x_5026_; 
v_inheritedTraceOptions_5025_ = lean_ctor_get(v___y_4737_, 13);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
lean_inc(v___y_4736_);
lean_inc_ref(v___y_4735_);
lean_inc(v___y_4734_);
lean_inc_ref(v_inheritedTraceOptions_5025_);
v___x_5026_ = lean_apply_7(v___f_4730_, v_inheritedTraceOptions_5025_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_, lean_box(0));
if (lean_obj_tag(v___x_5026_) == 0)
{
lean_object* v_a_5027_; uint8_t v___x_5028_; 
v_a_5027_ = lean_ctor_get(v___x_5026_, 0);
lean_inc(v_a_5027_);
lean_dec_ref_known(v___x_5026_, 1);
v___x_5028_ = lean_unbox(v_a_5027_);
lean_dec(v_a_5027_);
if (v___x_5028_ == 0)
{
lean_dec(v_cls_4731_);
goto v___jp_4955_;
}
else
{
lean_object* v___x_5029_; lean_object* v___x_5030_; 
v___x_5029_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__8, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__8);
v___x_5030_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4731_, v___x_5029_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_5030_) == 0)
{
lean_dec_ref_known(v___x_5030_, 1);
goto v___jp_4955_;
}
else
{
lean_object* v_a_5031_; lean_object* v___x_5033_; uint8_t v_isShared_5034_; uint8_t v_isSharedCheck_5038_; 
lean_dec_ref_known(v_a_4726_, 3);
lean_dec_ref_known(v_a_4729_, 3);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5031_ = lean_ctor_get(v___x_5030_, 0);
v_isSharedCheck_5038_ = !lean_is_exclusive(v___x_5030_);
if (v_isSharedCheck_5038_ == 0)
{
v___x_5033_ = v___x_5030_;
v_isShared_5034_ = v_isSharedCheck_5038_;
goto v_resetjp_5032_;
}
else
{
lean_inc(v_a_5031_);
lean_dec(v___x_5030_);
v___x_5033_ = lean_box(0);
v_isShared_5034_ = v_isSharedCheck_5038_;
goto v_resetjp_5032_;
}
v_resetjp_5032_:
{
lean_object* v___x_5036_; 
if (v_isShared_5034_ == 0)
{
v___x_5036_ = v___x_5033_;
goto v_reusejp_5035_;
}
else
{
lean_object* v_reuseFailAlloc_5037_; 
v_reuseFailAlloc_5037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5037_, 0, v_a_5031_);
v___x_5036_ = v_reuseFailAlloc_5037_;
goto v_reusejp_5035_;
}
v_reusejp_5035_:
{
return v___x_5036_;
}
}
}
}
}
else
{
lean_object* v_a_5039_; lean_object* v___x_5041_; uint8_t v_isShared_5042_; uint8_t v_isSharedCheck_5046_; 
lean_dec_ref_known(v_a_4726_, 3);
lean_dec_ref_known(v_a_4729_, 3);
lean_dec(v_cls_4731_);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5039_ = lean_ctor_get(v___x_5026_, 0);
v_isSharedCheck_5046_ = !lean_is_exclusive(v___x_5026_);
if (v_isSharedCheck_5046_ == 0)
{
v___x_5041_ = v___x_5026_;
v_isShared_5042_ = v_isSharedCheck_5046_;
goto v_resetjp_5040_;
}
else
{
lean_inc(v_a_5039_);
lean_dec(v___x_5026_);
v___x_5041_ = lean_box(0);
v_isShared_5042_ = v_isSharedCheck_5046_;
goto v_resetjp_5040_;
}
v_resetjp_5040_:
{
lean_object* v___x_5044_; 
if (v_isShared_5042_ == 0)
{
v___x_5044_ = v___x_5041_;
goto v_reusejp_5043_;
}
else
{
lean_object* v_reuseFailAlloc_5045_; 
v_reuseFailAlloc_5045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5045_, 0, v_a_5039_);
v___x_5044_ = v_reuseFailAlloc_5045_;
goto v_reusejp_5043_;
}
v_reusejp_5043_:
{
return v___x_5044_;
}
}
}
}
else
{
goto v___jp_4843_;
}
}
case 8:
{
lean_dec(v_a_4998_);
if (lean_obj_tag(v_a_4726_) == 8)
{
lean_object* v_inheritedTraceOptions_5047_; lean_object* v___x_5048_; 
v_inheritedTraceOptions_5047_ = lean_ctor_get(v___y_4737_, 13);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
lean_inc(v___y_4736_);
lean_inc_ref(v___y_4735_);
lean_inc(v___y_4734_);
lean_inc_ref(v_inheritedTraceOptions_5047_);
v___x_5048_ = lean_apply_7(v___f_4730_, v_inheritedTraceOptions_5047_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_, lean_box(0));
if (lean_obj_tag(v___x_5048_) == 0)
{
lean_object* v_a_5049_; uint8_t v___x_5050_; 
v_a_5049_ = lean_ctor_get(v___x_5048_, 0);
lean_inc(v_a_5049_);
lean_dec_ref_known(v___x_5048_, 1);
v___x_5050_ = lean_unbox(v_a_5049_);
lean_dec(v_a_5049_);
if (v___x_5050_ == 0)
{
lean_dec(v_cls_4731_);
goto v___jp_4964_;
}
else
{
lean_object* v___x_5051_; lean_object* v___x_5052_; 
v___x_5051_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__10, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__10);
v___x_5052_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4731_, v___x_5051_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_5052_) == 0)
{
lean_dec_ref_known(v___x_5052_, 1);
goto v___jp_4964_;
}
else
{
lean_object* v_a_5053_; lean_object* v___x_5055_; uint8_t v_isShared_5056_; uint8_t v_isSharedCheck_5060_; 
lean_dec_ref_known(v_a_4726_, 4);
lean_dec_ref_known(v_a_4729_, 4);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5053_ = lean_ctor_get(v___x_5052_, 0);
v_isSharedCheck_5060_ = !lean_is_exclusive(v___x_5052_);
if (v_isSharedCheck_5060_ == 0)
{
v___x_5055_ = v___x_5052_;
v_isShared_5056_ = v_isSharedCheck_5060_;
goto v_resetjp_5054_;
}
else
{
lean_inc(v_a_5053_);
lean_dec(v___x_5052_);
v___x_5055_ = lean_box(0);
v_isShared_5056_ = v_isSharedCheck_5060_;
goto v_resetjp_5054_;
}
v_resetjp_5054_:
{
lean_object* v___x_5058_; 
if (v_isShared_5056_ == 0)
{
v___x_5058_ = v___x_5055_;
goto v_reusejp_5057_;
}
else
{
lean_object* v_reuseFailAlloc_5059_; 
v_reuseFailAlloc_5059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5059_, 0, v_a_5053_);
v___x_5058_ = v_reuseFailAlloc_5059_;
goto v_reusejp_5057_;
}
v_reusejp_5057_:
{
return v___x_5058_;
}
}
}
}
}
else
{
lean_object* v_a_5061_; lean_object* v___x_5063_; uint8_t v_isShared_5064_; uint8_t v_isSharedCheck_5068_; 
lean_dec_ref_known(v_a_4726_, 4);
lean_dec_ref_known(v_a_4729_, 4);
lean_dec(v_cls_4731_);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5061_ = lean_ctor_get(v___x_5048_, 0);
v_isSharedCheck_5068_ = !lean_is_exclusive(v___x_5048_);
if (v_isSharedCheck_5068_ == 0)
{
v___x_5063_ = v___x_5048_;
v_isShared_5064_ = v_isSharedCheck_5068_;
goto v_resetjp_5062_;
}
else
{
lean_inc(v_a_5061_);
lean_dec(v___x_5048_);
v___x_5063_ = lean_box(0);
v_isShared_5064_ = v_isSharedCheck_5068_;
goto v_resetjp_5062_;
}
v_resetjp_5062_:
{
lean_object* v___x_5066_; 
if (v_isShared_5064_ == 0)
{
v___x_5066_ = v___x_5063_;
goto v_reusejp_5065_;
}
else
{
lean_object* v_reuseFailAlloc_5067_; 
v_reuseFailAlloc_5067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5067_, 0, v_a_5061_);
v___x_5066_ = v_reuseFailAlloc_5067_;
goto v_reusejp_5065_;
}
v_reusejp_5065_:
{
return v___x_5066_;
}
}
}
}
else
{
goto v___jp_4843_;
}
}
case 10:
{
lean_dec(v_a_4998_);
if (lean_obj_tag(v_a_4726_) == 10)
{
lean_object* v_data_5069_; lean_object* v_expr_5070_; lean_object* v_data_5071_; lean_object* v_expr_5072_; lean_object* v_inheritedTraceOptions_5073_; lean_object* v___x_5074_; 
v_data_5069_ = lean_ctor_get(v_a_4729_, 0);
v_expr_5070_ = lean_ctor_get(v_a_4729_, 1);
v_data_5071_ = lean_ctor_get(v_a_4726_, 0);
v_expr_5072_ = lean_ctor_get(v_a_4726_, 1);
v_inheritedTraceOptions_5073_ = lean_ctor_get(v___y_4737_, 13);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
lean_inc(v___y_4736_);
lean_inc_ref(v___y_4735_);
lean_inc(v___y_4734_);
lean_inc_ref(v_inheritedTraceOptions_5073_);
v___x_5074_ = lean_apply_7(v___f_4730_, v_inheritedTraceOptions_5073_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_, lean_box(0));
if (lean_obj_tag(v___x_5074_) == 0)
{
lean_object* v_a_5075_; uint8_t v___x_5076_; 
v_a_5075_ = lean_ctor_get(v___x_5074_, 0);
lean_inc(v_a_5075_);
lean_dec_ref_known(v___x_5074_, 1);
v___x_5076_ = lean_unbox(v_a_5075_);
lean_dec(v_a_5075_);
if (v___x_5076_ == 0)
{
lean_dec(v_cls_4731_);
lean_inc_ref(v_expr_5070_);
lean_inc_ref(v_expr_5072_);
lean_inc(v_data_5069_);
lean_inc(v_data_5071_);
v___y_4757_ = v_data_5071_;
v___y_4758_ = v_data_5069_;
v___y_4759_ = v_expr_5072_;
v___y_4760_ = v_expr_5070_;
goto v___jp_4756_;
}
else
{
lean_object* v___x_5077_; lean_object* v___x_5078_; 
v___x_5077_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__12, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__12);
v___x_5078_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4731_, v___x_5077_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_5078_) == 0)
{
lean_dec_ref_known(v___x_5078_, 1);
lean_inc_ref(v_expr_5070_);
lean_inc_ref(v_expr_5072_);
lean_inc(v_data_5069_);
lean_inc(v_data_5071_);
v___y_4757_ = v_data_5071_;
v___y_4758_ = v_data_5069_;
v___y_4759_ = v_expr_5072_;
v___y_4760_ = v_expr_5070_;
goto v___jp_4756_;
}
else
{
lean_object* v_a_5079_; lean_object* v___x_5081_; uint8_t v_isShared_5082_; uint8_t v_isSharedCheck_5086_; 
lean_dec_ref_known(v_a_4726_, 2);
lean_dec_ref_known(v_a_4729_, 2);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5079_ = lean_ctor_get(v___x_5078_, 0);
v_isSharedCheck_5086_ = !lean_is_exclusive(v___x_5078_);
if (v_isSharedCheck_5086_ == 0)
{
v___x_5081_ = v___x_5078_;
v_isShared_5082_ = v_isSharedCheck_5086_;
goto v_resetjp_5080_;
}
else
{
lean_inc(v_a_5079_);
lean_dec(v___x_5078_);
v___x_5081_ = lean_box(0);
v_isShared_5082_ = v_isSharedCheck_5086_;
goto v_resetjp_5080_;
}
v_resetjp_5080_:
{
lean_object* v___x_5084_; 
if (v_isShared_5082_ == 0)
{
v___x_5084_ = v___x_5081_;
goto v_reusejp_5083_;
}
else
{
lean_object* v_reuseFailAlloc_5085_; 
v_reuseFailAlloc_5085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5085_, 0, v_a_5079_);
v___x_5084_ = v_reuseFailAlloc_5085_;
goto v_reusejp_5083_;
}
v_reusejp_5083_:
{
return v___x_5084_;
}
}
}
}
}
else
{
lean_object* v_a_5087_; lean_object* v___x_5089_; uint8_t v_isShared_5090_; uint8_t v_isSharedCheck_5094_; 
lean_dec_ref_known(v_a_4726_, 2);
lean_dec_ref_known(v_a_4729_, 2);
lean_dec(v_cls_4731_);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5087_ = lean_ctor_get(v___x_5074_, 0);
v_isSharedCheck_5094_ = !lean_is_exclusive(v___x_5074_);
if (v_isSharedCheck_5094_ == 0)
{
v___x_5089_ = v___x_5074_;
v_isShared_5090_ = v_isSharedCheck_5094_;
goto v_resetjp_5088_;
}
else
{
lean_inc(v_a_5087_);
lean_dec(v___x_5074_);
v___x_5089_ = lean_box(0);
v_isShared_5090_ = v_isSharedCheck_5094_;
goto v_resetjp_5088_;
}
v_resetjp_5088_:
{
lean_object* v___x_5092_; 
if (v_isShared_5090_ == 0)
{
v___x_5092_ = v___x_5089_;
goto v_reusejp_5091_;
}
else
{
lean_object* v_reuseFailAlloc_5093_; 
v_reuseFailAlloc_5093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5093_, 0, v_a_5087_);
v___x_5092_ = v_reuseFailAlloc_5093_;
goto v_reusejp_5091_;
}
v_reusejp_5091_:
{
return v___x_5092_;
}
}
}
}
else
{
goto v___jp_4843_;
}
}
case 11:
{
lean_dec(v_a_4998_);
if (lean_obj_tag(v_a_4726_) == 11)
{
lean_object* v_typeName_5095_; lean_object* v_idx_5096_; lean_object* v_struct_5097_; lean_object* v_typeName_5098_; lean_object* v_idx_5099_; lean_object* v_struct_5100_; lean_object* v_inheritedTraceOptions_5101_; lean_object* v___x_5102_; 
v_typeName_5095_ = lean_ctor_get(v_a_4729_, 0);
v_idx_5096_ = lean_ctor_get(v_a_4729_, 1);
v_struct_5097_ = lean_ctor_get(v_a_4729_, 2);
v_typeName_5098_ = lean_ctor_get(v_a_4726_, 0);
v_idx_5099_ = lean_ctor_get(v_a_4726_, 1);
v_struct_5100_ = lean_ctor_get(v_a_4726_, 2);
v_inheritedTraceOptions_5101_ = lean_ctor_get(v___y_4737_, 13);
lean_inc(v___y_4738_);
lean_inc_ref(v___y_4737_);
lean_inc(v___y_4736_);
lean_inc_ref(v___y_4735_);
lean_inc(v___y_4734_);
lean_inc_ref(v_inheritedTraceOptions_5101_);
v___x_5102_ = lean_apply_7(v___f_4730_, v_inheritedTraceOptions_5101_, v___y_4734_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_, lean_box(0));
if (lean_obj_tag(v___x_5102_) == 0)
{
lean_object* v_a_5103_; uint8_t v___x_5104_; 
v_a_5103_ = lean_ctor_get(v___x_5102_, 0);
lean_inc(v_a_5103_);
lean_dec_ref_known(v___x_5102_, 1);
v___x_5104_ = lean_unbox(v_a_5103_);
lean_dec(v_a_5103_);
if (v___x_5104_ == 0)
{
lean_dec(v_cls_4731_);
lean_inc(v_typeName_5098_);
lean_inc(v_idx_5096_);
lean_inc_ref(v_struct_5097_);
lean_inc_ref(v_struct_5100_);
lean_inc(v_idx_5099_);
lean_inc(v_typeName_5095_);
v___y_4835_ = v_typeName_5095_;
v___y_4836_ = v_idx_5099_;
v___y_4837_ = v_struct_5100_;
v___y_4838_ = v_struct_5097_;
v___y_4839_ = v_idx_5096_;
v___y_4840_ = v_typeName_5098_;
goto v___jp_4834_;
}
else
{
lean_object* v___x_5105_; lean_object* v___x_5106_; 
v___x_5105_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__14, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___closed__14);
v___x_5106_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_4731_, v___x_5105_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
if (lean_obj_tag(v___x_5106_) == 0)
{
lean_dec_ref_known(v___x_5106_, 1);
lean_inc(v_typeName_5098_);
lean_inc(v_idx_5096_);
lean_inc_ref(v_struct_5097_);
lean_inc_ref(v_struct_5100_);
lean_inc(v_idx_5099_);
lean_inc(v_typeName_5095_);
v___y_4835_ = v_typeName_5095_;
v___y_4836_ = v_idx_5099_;
v___y_4837_ = v_struct_5100_;
v___y_4838_ = v_struct_5097_;
v___y_4839_ = v_idx_5096_;
v___y_4840_ = v_typeName_5098_;
goto v___jp_4834_;
}
else
{
lean_object* v_a_5107_; lean_object* v___x_5109_; uint8_t v_isShared_5110_; uint8_t v_isSharedCheck_5114_; 
lean_dec_ref_known(v_a_4726_, 3);
lean_dec_ref_known(v_a_4729_, 3);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5107_ = lean_ctor_get(v___x_5106_, 0);
v_isSharedCheck_5114_ = !lean_is_exclusive(v___x_5106_);
if (v_isSharedCheck_5114_ == 0)
{
v___x_5109_ = v___x_5106_;
v_isShared_5110_ = v_isSharedCheck_5114_;
goto v_resetjp_5108_;
}
else
{
lean_inc(v_a_5107_);
lean_dec(v___x_5106_);
v___x_5109_ = lean_box(0);
v_isShared_5110_ = v_isSharedCheck_5114_;
goto v_resetjp_5108_;
}
v_resetjp_5108_:
{
lean_object* v___x_5112_; 
if (v_isShared_5110_ == 0)
{
v___x_5112_ = v___x_5109_;
goto v_reusejp_5111_;
}
else
{
lean_object* v_reuseFailAlloc_5113_; 
v_reuseFailAlloc_5113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5113_, 0, v_a_5107_);
v___x_5112_ = v_reuseFailAlloc_5113_;
goto v_reusejp_5111_;
}
v_reusejp_5111_:
{
return v___x_5112_;
}
}
}
}
}
else
{
lean_object* v_a_5115_; lean_object* v___x_5117_; uint8_t v_isShared_5118_; uint8_t v_isSharedCheck_5122_; 
lean_dec_ref_known(v_a_4726_, 3);
lean_dec_ref_known(v_a_4729_, 3);
lean_dec(v_cls_4731_);
lean_dec(v_mvarCounterSaved_4728_);
v_a_5115_ = lean_ctor_get(v___x_5102_, 0);
v_isSharedCheck_5122_ = !lean_is_exclusive(v___x_5102_);
if (v_isSharedCheck_5122_ == 0)
{
v___x_5117_ = v___x_5102_;
v_isShared_5118_ = v_isSharedCheck_5122_;
goto v_resetjp_5116_;
}
else
{
lean_inc(v_a_5115_);
lean_dec(v___x_5102_);
v___x_5117_ = lean_box(0);
v_isShared_5118_ = v_isSharedCheck_5122_;
goto v_resetjp_5116_;
}
v_resetjp_5116_:
{
lean_object* v___x_5120_; 
if (v_isShared_5118_ == 0)
{
v___x_5120_ = v___x_5117_;
goto v_reusejp_5119_;
}
else
{
lean_object* v_reuseFailAlloc_5121_; 
v_reuseFailAlloc_5121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5121_, 0, v_a_5115_);
v___x_5120_ = v_reuseFailAlloc_5121_;
goto v_reusejp_5119_;
}
v_reusejp_5119_:
{
return v___x_5120_;
}
}
}
}
else
{
goto v___jp_4843_;
}
}
default: 
{
lean_dec(v_a_4998_);
goto v___jp_4843_;
}
}
}
else
{
lean_object* v___x_5123_; 
lean_dec(v_a_4998_);
lean_dec(v_cls_4731_);
lean_dec_ref(v___f_4730_);
lean_dec(v_mvarCounterSaved_4728_);
v___x_5123_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mkDefault(v_a_4729_, v_a_4726_, v___y_4735_, v___y_4736_, v___y_4737_, v___y_4738_);
return v___x_5123_;
}
}
else
{
lean_object* v_a_5124_; lean_object* v___x_5126_; uint8_t v_isShared_5127_; uint8_t v_isSharedCheck_5131_; 
lean_dec(v_cls_4731_);
lean_dec_ref(v___f_4730_);
lean_dec_ref(v_a_4729_);
lean_dec(v_mvarCounterSaved_4728_);
lean_dec_ref(v_a_4726_);
v_a_5124_ = lean_ctor_get(v___y_4997_, 0);
v_isSharedCheck_5131_ = !lean_is_exclusive(v___y_4997_);
if (v_isSharedCheck_5131_ == 0)
{
v___x_5126_ = v___y_4997_;
v_isShared_5127_ = v_isSharedCheck_5131_;
goto v_resetjp_5125_;
}
else
{
lean_inc(v_a_5124_);
lean_dec(v___y_4997_);
v___x_5126_ = lean_box(0);
v_isShared_5127_ = v_isSharedCheck_5131_;
goto v_resetjp_5125_;
}
v_resetjp_5125_:
{
lean_object* v___x_5129_; 
if (v_isShared_5127_ == 0)
{
v___x_5129_ = v___x_5126_;
goto v_reusejp_5128_;
}
else
{
lean_object* v_reuseFailAlloc_5130_; 
v_reuseFailAlloc_5130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5130_, 0, v_a_5124_);
v___x_5129_ = v_reuseFailAlloc_5130_;
goto v_reusejp_5128_;
}
v_reusejp_5128_:
{
return v___x_5129_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___boxed(lean_object* v_a_5136_, lean_object* v_depth_5137_, lean_object* v_mvarCounterSaved_5138_, lean_object* v_a_5139_, lean_object* v___f_5140_, lean_object* v_cls_5141_, lean_object* v___x_5142_, lean_object* v_____r_5143_, lean_object* v___y_5144_, lean_object* v___y_5145_, lean_object* v___y_5146_, lean_object* v___y_5147_, lean_object* v___y_5148_, lean_object* v___y_5149_){
_start:
{
uint8_t v___x_84224__boxed_5150_; lean_object* v_res_5151_; 
v___x_84224__boxed_5150_ = lean_unbox(v___x_5142_);
v_res_5151_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3(v_a_5136_, v_depth_5137_, v_mvarCounterSaved_5138_, v_a_5139_, v___f_5140_, v_cls_5141_, v___x_84224__boxed_5150_, v_____r_5143_, v___y_5144_, v___y_5145_, v___y_5146_, v___y_5147_, v___y_5148_);
lean_dec(v___y_5148_);
lean_dec_ref(v___y_5147_);
lean_dec(v___y_5146_);
lean_dec_ref(v___y_5145_);
lean_dec(v___y_5144_);
lean_dec(v_depth_5137_);
return v_res_5151_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__4(void){
_start:
{
lean_object* v___x_5153_; lean_object* v___x_5154_; 
v___x_5153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__3));
v___x_5154_ = l_Lean_stringToMessageData(v___x_5153_);
return v___x_5154_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__6(void){
_start:
{
lean_object* v___x_5156_; lean_object* v___x_5157_; 
v___x_5156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__5));
v___x_5157_ = l_Lean_stringToMessageData(v___x_5156_);
return v___x_5157_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__8(void){
_start:
{
lean_object* v___x_5159_; lean_object* v___x_5160_; 
v___x_5159_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__7));
v___x_5160_ = l_Lean_stringToMessageData(v___x_5159_);
return v___x_5160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(lean_object* v_depth_5161_, lean_object* v_mvarCounterSaved_5162_, lean_object* v_lhs_5163_, lean_object* v_rhs_5164_, lean_object* v_a_5165_, lean_object* v_a_5166_, lean_object* v_a_5167_, lean_object* v_a_5168_, lean_object* v_a_5169_){
_start:
{
lean_object* v___y_5172_; lean_object* v___y_5173_; lean_object* v_a_5174_; lean_object* v___y_5180_; lean_object* v___y_5181_; lean_object* v___y_5182_; lean_object* v___y_5185_; lean_object* v___y_5186_; lean_object* v___y_5187_; lean_object* v___y_5188_; lean_object* v___y_5189_; lean_object* v___y_5190_; lean_object* v___y_5191_; lean_object* v___y_5192_; lean_object* v___y_5193_; lean_object* v___y_5210_; lean_object* v___y_5211_; lean_object* v___y_5212_; lean_object* v___y_5213_; lean_object* v___y_5214_; lean_object* v___y_5215_; lean_object* v___y_5216_; lean_object* v___y_5220_; lean_object* v___y_5221_; uint8_t v___y_5222_; lean_object* v___y_5223_; lean_object* v___y_5224_; lean_object* v___y_5225_; lean_object* v___y_5226_; lean_object* v___y_5227_; lean_object* v___y_5228_; lean_object* v___y_5229_; lean_object* v_inheritedTraceOptions_5231_; lean_object* v_cls_5232_; lean_object* v___f_5233_; lean_object* v___y_5235_; lean_object* v___y_5236_; lean_object* v___y_5237_; lean_object* v___y_5238_; lean_object* v___y_5239_; lean_object* v___y_5308_; lean_object* v___y_5309_; lean_object* v___y_5310_; lean_object* v___y_5311_; lean_object* v___y_5312_; lean_object* v___x_5325_; 
v_inheritedTraceOptions_5231_ = lean_ctor_get(v_a_5168_, 13);
v_cls_5232_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn___closed__2_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_));
v___f_5233_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__0));
v___x_5325_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__0(v_cls_5232_, v_inheritedTraceOptions_5231_, v_a_5165_, v_a_5166_, v_a_5167_, v_a_5168_, v_a_5169_);
if (lean_obj_tag(v___x_5325_) == 0)
{
lean_object* v_a_5326_; uint8_t v___x_5327_; 
v_a_5326_ = lean_ctor_get(v___x_5325_, 0);
lean_inc(v_a_5326_);
lean_dec_ref_known(v___x_5325_, 1);
v___x_5327_ = lean_unbox(v_a_5326_);
lean_dec(v_a_5326_);
if (v___x_5327_ == 0)
{
v___y_5308_ = v_a_5165_;
v___y_5309_ = v_a_5166_;
v___y_5310_ = v_a_5167_;
v___y_5311_ = v_a_5168_;
v___y_5312_ = v_a_5169_;
goto v___jp_5307_;
}
else
{
lean_object* v___x_5328_; uint8_t v___x_5329_; lean_object* v___x_5330_; lean_object* v___x_5331_; 
v___x_5328_ = lean_box(0);
v___x_5329_ = 0;
v___x_5330_ = lean_box(0);
v___x_5331_ = l_Lean_Meta_mkFreshExprMVar(v___x_5328_, v___x_5329_, v___x_5330_, v_a_5166_, v_a_5167_, v_a_5168_, v_a_5169_);
if (lean_obj_tag(v___x_5331_) == 0)
{
lean_object* v_a_5332_; lean_object* v___x_5333_; lean_object* v___x_5334_; lean_object* v___x_5335_; lean_object* v___x_5336_; lean_object* v___x_5337_; lean_object* v___x_5338_; lean_object* v___x_5339_; lean_object* v___x_5340_; lean_object* v___x_5341_; lean_object* v___x_5342_; lean_object* v___x_5343_; lean_object* v___x_5344_; lean_object* v___x_5345_; lean_object* v___x_5346_; lean_object* v___x_5347_; lean_object* v___x_5348_; lean_object* v___x_5349_; 
v_a_5332_ = lean_ctor_get(v___x_5331_, 0);
lean_inc(v_a_5332_);
lean_dec_ref_known(v___x_5331_, 1);
v___x_5333_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__6, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__6);
lean_inc(v_depth_5161_);
v___x_5334_ = l_Nat_reprFast(v_depth_5161_);
v___x_5335_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_5335_, 0, v___x_5334_);
v___x_5336_ = l_Lean_MessageData_ofFormat(v___x_5335_);
v___x_5337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5337_, 0, v___x_5333_);
lean_ctor_set(v___x_5337_, 1, v___x_5336_);
v___x_5338_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__8, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__8);
v___x_5339_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5339_, 0, v___x_5337_);
lean_ctor_set(v___x_5339_, 1, v___x_5338_);
lean_inc_ref(v_lhs_5163_);
v___x_5340_ = l_Lean_MessageData_ofExpr(v_lhs_5163_);
v___x_5341_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5341_, 0, v___x_5339_);
lean_ctor_set(v___x_5341_, 1, v___x_5340_);
v___x_5342_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5342_, 0, v___x_5341_);
lean_ctor_set(v___x_5342_, 1, v___x_5338_);
lean_inc_ref(v_rhs_5164_);
v___x_5343_ = l_Lean_MessageData_ofExpr(v_rhs_5164_);
v___x_5344_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5344_, 0, v___x_5342_);
lean_ctor_set(v___x_5344_, 1, v___x_5343_);
v___x_5345_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5345_, 0, v___x_5344_);
lean_ctor_set(v___x_5345_, 1, v___x_5338_);
v___x_5346_ = l_Lean_Expr_mvarId_x21(v_a_5332_);
lean_dec(v_a_5332_);
v___x_5347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5347_, 0, v___x_5346_);
v___x_5348_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5348_, 0, v___x_5345_);
lean_ctor_set(v___x_5348_, 1, v___x_5347_);
v___x_5349_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_5232_, v___x_5348_, v_a_5166_, v_a_5167_, v_a_5168_, v_a_5169_);
if (lean_obj_tag(v___x_5349_) == 0)
{
lean_dec_ref_known(v___x_5349_, 1);
v___y_5308_ = v_a_5165_;
v___y_5309_ = v_a_5166_;
v___y_5310_ = v_a_5167_;
v___y_5311_ = v_a_5168_;
v___y_5312_ = v_a_5169_;
goto v___jp_5307_;
}
else
{
lean_object* v_a_5350_; lean_object* v___x_5352_; uint8_t v_isShared_5353_; uint8_t v_isSharedCheck_5357_; 
lean_dec_ref(v_rhs_5164_);
lean_dec_ref(v_lhs_5163_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_a_5350_ = lean_ctor_get(v___x_5349_, 0);
v_isSharedCheck_5357_ = !lean_is_exclusive(v___x_5349_);
if (v_isSharedCheck_5357_ == 0)
{
v___x_5352_ = v___x_5349_;
v_isShared_5353_ = v_isSharedCheck_5357_;
goto v_resetjp_5351_;
}
else
{
lean_inc(v_a_5350_);
lean_dec(v___x_5349_);
v___x_5352_ = lean_box(0);
v_isShared_5353_ = v_isSharedCheck_5357_;
goto v_resetjp_5351_;
}
v_resetjp_5351_:
{
lean_object* v___x_5355_; 
if (v_isShared_5353_ == 0)
{
v___x_5355_ = v___x_5352_;
goto v_reusejp_5354_;
}
else
{
lean_object* v_reuseFailAlloc_5356_; 
v_reuseFailAlloc_5356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5356_, 0, v_a_5350_);
v___x_5355_ = v_reuseFailAlloc_5356_;
goto v_reusejp_5354_;
}
v_reusejp_5354_:
{
return v___x_5355_;
}
}
}
}
else
{
lean_object* v_a_5358_; lean_object* v___x_5360_; uint8_t v_isShared_5361_; uint8_t v_isSharedCheck_5365_; 
lean_dec_ref(v_rhs_5164_);
lean_dec_ref(v_lhs_5163_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_a_5358_ = lean_ctor_get(v___x_5331_, 0);
v_isSharedCheck_5365_ = !lean_is_exclusive(v___x_5331_);
if (v_isSharedCheck_5365_ == 0)
{
v___x_5360_ = v___x_5331_;
v_isShared_5361_ = v_isSharedCheck_5365_;
goto v_resetjp_5359_;
}
else
{
lean_inc(v_a_5358_);
lean_dec(v___x_5331_);
v___x_5360_ = lean_box(0);
v_isShared_5361_ = v_isSharedCheck_5365_;
goto v_resetjp_5359_;
}
v_resetjp_5359_:
{
lean_object* v___x_5363_; 
if (v_isShared_5361_ == 0)
{
v___x_5363_ = v___x_5360_;
goto v_reusejp_5362_;
}
else
{
lean_object* v_reuseFailAlloc_5364_; 
v_reuseFailAlloc_5364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5364_, 0, v_a_5358_);
v___x_5363_ = v_reuseFailAlloc_5364_;
goto v_reusejp_5362_;
}
v_reusejp_5362_:
{
return v___x_5363_;
}
}
}
}
}
else
{
lean_object* v_a_5366_; lean_object* v___x_5368_; uint8_t v_isShared_5369_; uint8_t v_isSharedCheck_5373_; 
lean_dec_ref(v_rhs_5164_);
lean_dec_ref(v_lhs_5163_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_a_5366_ = lean_ctor_get(v___x_5325_, 0);
v_isSharedCheck_5373_ = !lean_is_exclusive(v___x_5325_);
if (v_isSharedCheck_5373_ == 0)
{
v___x_5368_ = v___x_5325_;
v_isShared_5369_ = v_isSharedCheck_5373_;
goto v_resetjp_5367_;
}
else
{
lean_inc(v_a_5366_);
lean_dec(v___x_5325_);
v___x_5368_ = lean_box(0);
v_isShared_5369_ = v_isSharedCheck_5373_;
goto v_resetjp_5367_;
}
v_resetjp_5367_:
{
lean_object* v___x_5371_; 
if (v_isShared_5369_ == 0)
{
v___x_5371_ = v___x_5368_;
goto v_reusejp_5370_;
}
else
{
lean_object* v_reuseFailAlloc_5372_; 
v_reuseFailAlloc_5372_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5372_, 0, v_a_5366_);
v___x_5371_ = v_reuseFailAlloc_5372_;
goto v_reusejp_5370_;
}
v_reusejp_5370_:
{
return v___x_5371_;
}
}
}
v___jp_5171_:
{
lean_object* v___x_5175_; lean_object* v___x_5176_; lean_object* v___x_5177_; lean_object* v___x_5178_; 
v___x_5175_ = lean_st_ref_take(v___y_5172_);
lean_inc_ref(v_a_5174_);
v___x_5176_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8___redArg(v___x_5175_, v___y_5173_, v_a_5174_);
v___x_5177_ = lean_st_ref_set(v___y_5172_, v___x_5176_);
v___x_5178_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5178_, 0, v_a_5174_);
return v___x_5178_;
}
v___jp_5179_:
{
if (lean_obj_tag(v___y_5182_) == 0)
{
lean_object* v_a_5183_; 
v_a_5183_ = lean_ctor_get(v___y_5182_, 0);
lean_inc(v_a_5183_);
lean_dec_ref_known(v___y_5182_, 1);
v___y_5172_ = v___y_5180_;
v___y_5173_ = v___y_5181_;
v_a_5174_ = v_a_5183_;
goto v___jp_5171_;
}
else
{
lean_dec_ref(v___y_5181_);
return v___y_5182_;
}
}
v___jp_5184_:
{
lean_object* v___x_5194_; 
lean_inc_ref(v___y_5190_);
lean_inc_ref(v___y_5186_);
v___x_5194_ = l_Lean_Meta_isExprDefEq(v___y_5186_, v___y_5190_, v___y_5191_, v___y_5192_, v___y_5189_, v___y_5187_);
if (lean_obj_tag(v___x_5194_) == 0)
{
lean_object* v_a_5195_; uint8_t v___x_5196_; 
v_a_5195_ = lean_ctor_get(v___x_5194_, 0);
lean_inc(v_a_5195_);
lean_dec_ref_known(v___x_5194_, 1);
v___x_5196_ = lean_unbox(v_a_5195_);
lean_dec(v_a_5195_);
if (v___x_5196_ == 0)
{
lean_object* v___x_5197_; lean_object* v___x_5198_; 
lean_dec_ref(v___y_5190_);
lean_dec_ref(v___y_5186_);
v___x_5197_ = lean_box(0);
lean_inc(v___y_5187_);
lean_inc_ref(v___y_5189_);
lean_inc(v___y_5192_);
lean_inc_ref(v___y_5191_);
lean_inc(v___y_5188_);
v___x_5198_ = lean_apply_7(v___y_5185_, v___x_5197_, v___y_5188_, v___y_5191_, v___y_5192_, v___y_5189_, v___y_5187_, lean_box(0));
v___y_5180_ = v___y_5188_;
v___y_5181_ = v___y_5193_;
v___y_5182_ = v___x_5198_;
goto v___jp_5179_;
}
else
{
lean_object* v___x_5199_; lean_object* v___x_5200_; 
lean_dec_ref(v___y_5185_);
v___x_5199_ = lean_box(0);
v___x_5200_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_5200_, 0, v___y_5186_);
lean_ctor_set(v___x_5200_, 1, v___y_5190_);
lean_ctor_set(v___x_5200_, 2, v___x_5199_);
v___y_5172_ = v___y_5188_;
v___y_5173_ = v___y_5193_;
v_a_5174_ = v___x_5200_;
goto v___jp_5171_;
}
}
else
{
lean_object* v_a_5201_; lean_object* v___x_5203_; uint8_t v_isShared_5204_; uint8_t v_isSharedCheck_5208_; 
lean_dec_ref(v___y_5193_);
lean_dec_ref(v___y_5190_);
lean_dec_ref(v___y_5186_);
lean_dec_ref(v___y_5185_);
v_a_5201_ = lean_ctor_get(v___x_5194_, 0);
v_isSharedCheck_5208_ = !lean_is_exclusive(v___x_5194_);
if (v_isSharedCheck_5208_ == 0)
{
v___x_5203_ = v___x_5194_;
v_isShared_5204_ = v_isSharedCheck_5208_;
goto v_resetjp_5202_;
}
else
{
lean_inc(v_a_5201_);
lean_dec(v___x_5194_);
v___x_5203_ = lean_box(0);
v_isShared_5204_ = v_isSharedCheck_5208_;
goto v_resetjp_5202_;
}
v_resetjp_5202_:
{
lean_object* v___x_5206_; 
if (v_isShared_5204_ == 0)
{
v___x_5206_ = v___x_5203_;
goto v_reusejp_5205_;
}
else
{
lean_object* v_reuseFailAlloc_5207_; 
v_reuseFailAlloc_5207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5207_, 0, v_a_5201_);
v___x_5206_ = v_reuseFailAlloc_5207_;
goto v_reusejp_5205_;
}
v_reusejp_5205_:
{
return v___x_5206_;
}
}
}
}
v___jp_5209_:
{
lean_object* v___x_5217_; lean_object* v___x_5218_; 
v___x_5217_ = lean_box(0);
lean_inc(v___y_5211_);
lean_inc_ref(v___y_5212_);
lean_inc(v___y_5215_);
lean_inc_ref(v___y_5214_);
lean_inc(v___y_5213_);
v___x_5218_ = lean_apply_7(v___y_5210_, v___x_5217_, v___y_5213_, v___y_5214_, v___y_5215_, v___y_5212_, v___y_5211_, lean_box(0));
v___y_5180_ = v___y_5213_;
v___y_5181_ = v___y_5216_;
v___y_5182_ = v___x_5218_;
goto v___jp_5179_;
}
v___jp_5219_:
{
lean_object* v___x_5230_; 
v___x_5230_ = lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole(v_mvarCounterSaved_5162_, v___y_5226_);
if (lean_obj_tag(v___x_5230_) == 0)
{
v___y_5185_ = v___y_5221_;
v___y_5186_ = v___y_5220_;
v___y_5187_ = v___y_5223_;
v___y_5188_ = v___y_5225_;
v___y_5189_ = v___y_5224_;
v___y_5190_ = v___y_5226_;
v___y_5191_ = v___y_5227_;
v___y_5192_ = v___y_5228_;
v___y_5193_ = v___y_5229_;
goto v___jp_5184_;
}
else
{
lean_dec_ref_known(v___x_5230_, 1);
if (v___y_5222_ == 0)
{
lean_dec_ref(v___y_5226_);
lean_dec_ref(v___y_5220_);
v___y_5210_ = v___y_5221_;
v___y_5211_ = v___y_5223_;
v___y_5212_ = v___y_5224_;
v___y_5213_ = v___y_5225_;
v___y_5214_ = v___y_5227_;
v___y_5215_ = v___y_5228_;
v___y_5216_ = v___y_5229_;
goto v___jp_5209_;
}
else
{
v___y_5185_ = v___y_5221_;
v___y_5186_ = v___y_5220_;
v___y_5187_ = v___y_5223_;
v___y_5188_ = v___y_5225_;
v___y_5189_ = v___y_5224_;
v___y_5190_ = v___y_5226_;
v___y_5191_ = v___y_5227_;
v___y_5192_ = v___y_5228_;
v___y_5193_ = v___y_5229_;
goto v___jp_5184_;
}
}
}
v___jp_5234_:
{
lean_object* v___x_5240_; 
v___x_5240_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg(v_lhs_5163_, v___y_5237_);
if (lean_obj_tag(v___x_5240_) == 0)
{
lean_object* v_a_5241_; lean_object* v___x_5242_; 
v_a_5241_ = lean_ctor_get(v___x_5240_, 0);
lean_inc(v_a_5241_);
lean_dec_ref_known(v___x_5240_, 1);
v___x_5242_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg(v_rhs_5164_, v___y_5237_);
if (lean_obj_tag(v___x_5242_) == 0)
{
lean_object* v_a_5243_; lean_object* v___x_5245_; uint8_t v_isShared_5246_; uint8_t v_isSharedCheck_5290_; 
v_a_5243_ = lean_ctor_get(v___x_5242_, 0);
v_isSharedCheck_5290_ = !lean_is_exclusive(v___x_5242_);
if (v_isSharedCheck_5290_ == 0)
{
v___x_5245_ = v___x_5242_;
v_isShared_5246_ = v_isSharedCheck_5290_;
goto v_resetjp_5244_;
}
else
{
lean_inc(v_a_5243_);
lean_dec(v___x_5242_);
v___x_5245_ = lean_box(0);
v_isShared_5246_ = v_isSharedCheck_5290_;
goto v_resetjp_5244_;
}
v_resetjp_5244_:
{
lean_object* v___x_5247_; lean_object* v___x_5248_; lean_object* v___x_5249_; 
v___x_5247_ = lean_st_ref_get(v___y_5235_);
lean_inc(v_a_5243_);
lean_inc(v_a_5241_);
v___x_5248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5248_, 0, v_a_5241_);
lean_ctor_set(v___x_5248_, 1, v_a_5243_);
v___x_5249_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___redArg(v___x_5247_, v___x_5248_);
lean_dec(v___x_5247_);
if (lean_obj_tag(v___x_5249_) == 0)
{
lean_object* v___x_5250_; 
lean_del_object(v___x_5245_);
lean_inc(v_a_5243_);
lean_inc(v_a_5241_);
lean_inc(v_mvarCounterSaved_5162_);
v___x_5250_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f(v_mvarCounterSaved_5162_, v_a_5241_, v_a_5243_, v___y_5236_, v___y_5237_, v___y_5238_, v___y_5239_);
if (lean_obj_tag(v___x_5250_) == 0)
{
lean_object* v_a_5251_; 
v_a_5251_ = lean_ctor_get(v___x_5250_, 0);
lean_inc(v_a_5251_);
lean_dec_ref_known(v___x_5250_, 1);
if (lean_obj_tag(v_a_5251_) == 1)
{
lean_object* v_options_5252_; uint8_t v_hasTrace_5253_; 
lean_dec(v_a_5243_);
lean_dec(v_a_5241_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_options_5252_ = lean_ctor_get(v___y_5238_, 2);
v_hasTrace_5253_ = lean_ctor_get_uint8(v_options_5252_, sizeof(void*)*1);
if (v_hasTrace_5253_ == 0)
{
lean_object* v_val_5254_; 
v_val_5254_ = lean_ctor_get(v_a_5251_, 0);
lean_inc(v_val_5254_);
lean_dec_ref_known(v_a_5251_, 1);
v___y_5172_ = v___y_5235_;
v___y_5173_ = v___x_5248_;
v_a_5174_ = v_val_5254_;
goto v___jp_5171_;
}
else
{
lean_object* v_val_5255_; lean_object* v_inheritedTraceOptions_5256_; lean_object* v___x_5257_; uint8_t v___x_5258_; 
v_val_5255_ = lean_ctor_get(v_a_5251_, 0);
lean_inc(v_val_5255_);
lean_dec_ref_known(v_a_5251_, 1);
v_inheritedTraceOptions_5256_ = lean_ctor_get(v___y_5238_, 13);
v___x_5257_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfCHole_x3f___closed__23);
v___x_5258_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_5256_, v_options_5252_, v___x_5257_);
if (v___x_5258_ == 0)
{
v___y_5172_ = v___y_5235_;
v___y_5173_ = v___x_5248_;
v_a_5174_ = v_val_5255_;
goto v___jp_5171_;
}
else
{
lean_object* v___x_5259_; lean_object* v___x_5260_; 
v___x_5259_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__2, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__2);
v___x_5260_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_5232_, v___x_5259_, v___y_5236_, v___y_5237_, v___y_5238_, v___y_5239_);
if (lean_obj_tag(v___x_5260_) == 0)
{
lean_dec_ref_known(v___x_5260_, 1);
v___y_5172_ = v___y_5235_;
v___y_5173_ = v___x_5248_;
v_a_5174_ = v_val_5255_;
goto v___jp_5171_;
}
else
{
lean_object* v_a_5261_; lean_object* v___x_5263_; uint8_t v_isShared_5264_; uint8_t v_isSharedCheck_5268_; 
lean_dec(v_val_5255_);
lean_dec_ref_known(v___x_5248_, 2);
v_a_5261_ = lean_ctor_get(v___x_5260_, 0);
v_isSharedCheck_5268_ = !lean_is_exclusive(v___x_5260_);
if (v_isSharedCheck_5268_ == 0)
{
v___x_5263_ = v___x_5260_;
v_isShared_5264_ = v_isSharedCheck_5268_;
goto v_resetjp_5262_;
}
else
{
lean_inc(v_a_5261_);
lean_dec(v___x_5260_);
v___x_5263_ = lean_box(0);
v_isShared_5264_ = v_isSharedCheck_5268_;
goto v_resetjp_5262_;
}
v_resetjp_5262_:
{
lean_object* v___x_5266_; 
if (v_isShared_5264_ == 0)
{
v___x_5266_ = v___x_5263_;
goto v_reusejp_5265_;
}
else
{
lean_object* v_reuseFailAlloc_5267_; 
v_reuseFailAlloc_5267_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5267_, 0, v_a_5261_);
v___x_5266_ = v_reuseFailAlloc_5267_;
goto v_reusejp_5265_;
}
v_reusejp_5265_:
{
return v___x_5266_;
}
}
}
}
}
}
else
{
uint8_t v___x_5269_; 
lean_dec(v_a_5251_);
v___x_5269_ = lean_expr_eqv(v_a_5241_, v_a_5243_);
if (v___x_5269_ == 0)
{
uint8_t v___x_5270_; lean_object* v___x_5271_; lean_object* v___f_5272_; lean_object* v___x_5273_; 
v___x_5270_ = 1;
v___x_5271_ = lean_box(v___x_5270_);
lean_inc(v_a_5241_);
lean_inc_n(v_mvarCounterSaved_5162_, 2);
lean_inc(v_a_5243_);
v___f_5272_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__3___boxed), 14, 7);
lean_closure_set(v___f_5272_, 0, v_a_5243_);
lean_closure_set(v___f_5272_, 1, v_depth_5161_);
lean_closure_set(v___f_5272_, 2, v_mvarCounterSaved_5162_);
lean_closure_set(v___f_5272_, 3, v_a_5241_);
lean_closure_set(v___f_5272_, 4, v___f_5233_);
lean_closure_set(v___f_5272_, 5, v_cls_5232_);
lean_closure_set(v___f_5272_, 6, v___x_5271_);
v___x_5273_ = lp_mathlib_Mathlib_Tactic_TermCongr_hasCHole(v_mvarCounterSaved_5162_, v_a_5241_);
if (lean_obj_tag(v___x_5273_) == 0)
{
v___y_5220_ = v_a_5241_;
v___y_5221_ = v___f_5272_;
v___y_5222_ = v___x_5269_;
v___y_5223_ = v___y_5239_;
v___y_5224_ = v___y_5238_;
v___y_5225_ = v___y_5235_;
v___y_5226_ = v_a_5243_;
v___y_5227_ = v___y_5236_;
v___y_5228_ = v___y_5237_;
v___y_5229_ = v___x_5248_;
goto v___jp_5219_;
}
else
{
lean_dec_ref_known(v___x_5273_, 1);
if (v___x_5269_ == 0)
{
lean_dec(v_a_5243_);
lean_dec(v_a_5241_);
lean_dec(v_mvarCounterSaved_5162_);
v___y_5210_ = v___f_5272_;
v___y_5211_ = v___y_5239_;
v___y_5212_ = v___y_5238_;
v___y_5213_ = v___y_5235_;
v___y_5214_ = v___y_5236_;
v___y_5215_ = v___y_5237_;
v___y_5216_ = v___x_5248_;
goto v___jp_5209_;
}
else
{
v___y_5220_ = v_a_5241_;
v___y_5221_ = v___f_5272_;
v___y_5222_ = v___x_5269_;
v___y_5223_ = v___y_5239_;
v___y_5224_ = v___y_5238_;
v___y_5225_ = v___y_5235_;
v___y_5226_ = v_a_5243_;
v___y_5227_ = v___y_5236_;
v___y_5228_ = v___y_5237_;
v___y_5229_ = v___x_5248_;
goto v___jp_5219_;
}
}
}
else
{
lean_object* v___x_5274_; lean_object* v___x_5275_; lean_object* v___x_5276_; lean_object* v___x_5277_; 
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v___x_5274_ = lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(v_a_5241_);
lean_dec(v_a_5241_);
v___x_5275_ = lp_mathlib_Mathlib_Tactic_TermCongr_removeCHoles(v_a_5243_);
lean_dec(v_a_5243_);
v___x_5276_ = lean_box(0);
v___x_5277_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_5277_, 0, v___x_5274_);
lean_ctor_set(v___x_5277_, 1, v___x_5275_);
lean_ctor_set(v___x_5277_, 2, v___x_5276_);
v___y_5172_ = v___y_5235_;
v___y_5173_ = v___x_5248_;
v_a_5174_ = v___x_5277_;
goto v___jp_5171_;
}
}
}
else
{
lean_object* v_a_5278_; lean_object* v___x_5280_; uint8_t v_isShared_5281_; uint8_t v_isSharedCheck_5285_; 
lean_dec_ref_known(v___x_5248_, 2);
lean_dec(v_a_5243_);
lean_dec(v_a_5241_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_a_5278_ = lean_ctor_get(v___x_5250_, 0);
v_isSharedCheck_5285_ = !lean_is_exclusive(v___x_5250_);
if (v_isSharedCheck_5285_ == 0)
{
v___x_5280_ = v___x_5250_;
v_isShared_5281_ = v_isSharedCheck_5285_;
goto v_resetjp_5279_;
}
else
{
lean_inc(v_a_5278_);
lean_dec(v___x_5250_);
v___x_5280_ = lean_box(0);
v_isShared_5281_ = v_isSharedCheck_5285_;
goto v_resetjp_5279_;
}
v_resetjp_5279_:
{
lean_object* v___x_5283_; 
if (v_isShared_5281_ == 0)
{
v___x_5283_ = v___x_5280_;
goto v_reusejp_5282_;
}
else
{
lean_object* v_reuseFailAlloc_5284_; 
v_reuseFailAlloc_5284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5284_, 0, v_a_5278_);
v___x_5283_ = v_reuseFailAlloc_5284_;
goto v_reusejp_5282_;
}
v_reusejp_5282_:
{
return v___x_5283_;
}
}
}
}
else
{
lean_object* v_val_5286_; lean_object* v___x_5288_; 
lean_dec_ref_known(v___x_5248_, 2);
lean_dec(v_a_5243_);
lean_dec(v_a_5241_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_val_5286_ = lean_ctor_get(v___x_5249_, 0);
lean_inc(v_val_5286_);
lean_dec_ref_known(v___x_5249_, 1);
if (v_isShared_5246_ == 0)
{
lean_ctor_set(v___x_5245_, 0, v_val_5286_);
v___x_5288_ = v___x_5245_;
goto v_reusejp_5287_;
}
else
{
lean_object* v_reuseFailAlloc_5289_; 
v_reuseFailAlloc_5289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5289_, 0, v_val_5286_);
v___x_5288_ = v_reuseFailAlloc_5289_;
goto v_reusejp_5287_;
}
v_reusejp_5287_:
{
return v___x_5288_;
}
}
}
}
else
{
lean_object* v_a_5291_; lean_object* v___x_5293_; uint8_t v_isShared_5294_; uint8_t v_isSharedCheck_5298_; 
lean_dec(v_a_5241_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_a_5291_ = lean_ctor_get(v___x_5242_, 0);
v_isSharedCheck_5298_ = !lean_is_exclusive(v___x_5242_);
if (v_isSharedCheck_5298_ == 0)
{
v___x_5293_ = v___x_5242_;
v_isShared_5294_ = v_isSharedCheck_5298_;
goto v_resetjp_5292_;
}
else
{
lean_inc(v_a_5291_);
lean_dec(v___x_5242_);
v___x_5293_ = lean_box(0);
v_isShared_5294_ = v_isSharedCheck_5298_;
goto v_resetjp_5292_;
}
v_resetjp_5292_:
{
lean_object* v___x_5296_; 
if (v_isShared_5294_ == 0)
{
v___x_5296_ = v___x_5293_;
goto v_reusejp_5295_;
}
else
{
lean_object* v_reuseFailAlloc_5297_; 
v_reuseFailAlloc_5297_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5297_, 0, v_a_5291_);
v___x_5296_ = v_reuseFailAlloc_5297_;
goto v_reusejp_5295_;
}
v_reusejp_5295_:
{
return v___x_5296_;
}
}
}
}
else
{
lean_object* v_a_5299_; lean_object* v___x_5301_; uint8_t v_isShared_5302_; uint8_t v_isSharedCheck_5306_; 
lean_dec_ref(v_rhs_5164_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_a_5299_ = lean_ctor_get(v___x_5240_, 0);
v_isSharedCheck_5306_ = !lean_is_exclusive(v___x_5240_);
if (v_isSharedCheck_5306_ == 0)
{
v___x_5301_ = v___x_5240_;
v_isShared_5302_ = v_isSharedCheck_5306_;
goto v_resetjp_5300_;
}
else
{
lean_inc(v_a_5299_);
lean_dec(v___x_5240_);
v___x_5301_ = lean_box(0);
v_isShared_5302_ = v_isSharedCheck_5306_;
goto v_resetjp_5300_;
}
v_resetjp_5300_:
{
lean_object* v___x_5304_; 
if (v_isShared_5302_ == 0)
{
v___x_5304_ = v___x_5301_;
goto v_reusejp_5303_;
}
else
{
lean_object* v_reuseFailAlloc_5305_; 
v_reuseFailAlloc_5305_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5305_, 0, v_a_5299_);
v___x_5304_ = v_reuseFailAlloc_5305_;
goto v_reusejp_5303_;
}
v_reusejp_5303_:
{
return v___x_5304_;
}
}
}
}
v___jp_5307_:
{
lean_object* v___x_5313_; uint8_t v___x_5314_; 
v___x_5313_ = lean_unsigned_to_nat(1000u);
v___x_5314_ = lean_nat_dec_lt(v___x_5313_, v_depth_5161_);
if (v___x_5314_ == 0)
{
v___y_5235_ = v___y_5308_;
v___y_5236_ = v___y_5309_;
v___y_5237_ = v___y_5310_;
v___y_5238_ = v___y_5311_;
v___y_5239_ = v___y_5312_;
goto v___jp_5234_;
}
else
{
lean_object* v___x_5315_; lean_object* v___x_5316_; 
v___x_5315_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__4, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___closed__4);
v___x_5316_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___redArg(v___x_5315_, v___y_5309_, v___y_5310_, v___y_5311_, v___y_5312_);
if (lean_obj_tag(v___x_5316_) == 0)
{
lean_dec_ref_known(v___x_5316_, 1);
v___y_5235_ = v___y_5308_;
v___y_5236_ = v___y_5309_;
v___y_5237_ = v___y_5310_;
v___y_5238_ = v___y_5311_;
v___y_5239_ = v___y_5312_;
goto v___jp_5234_;
}
else
{
lean_object* v_a_5317_; lean_object* v___x_5319_; uint8_t v_isShared_5320_; uint8_t v_isSharedCheck_5324_; 
lean_dec_ref(v_rhs_5164_);
lean_dec_ref(v_lhs_5163_);
lean_dec(v_mvarCounterSaved_5162_);
lean_dec(v_depth_5161_);
v_a_5317_ = lean_ctor_get(v___x_5316_, 0);
v_isSharedCheck_5324_ = !lean_is_exclusive(v___x_5316_);
if (v_isSharedCheck_5324_ == 0)
{
v___x_5319_ = v___x_5316_;
v_isShared_5320_ = v_isSharedCheck_5324_;
goto v_resetjp_5318_;
}
else
{
lean_inc(v_a_5317_);
lean_dec(v___x_5316_);
v___x_5319_ = lean_box(0);
v_isShared_5320_ = v_isSharedCheck_5324_;
goto v_resetjp_5318_;
}
v_resetjp_5318_:
{
lean_object* v___x_5322_; 
if (v_isShared_5320_ == 0)
{
v___x_5322_ = v___x_5319_;
goto v_reusejp_5321_;
}
else
{
lean_object* v_reuseFailAlloc_5323_; 
v_reuseFailAlloc_5323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5323_, 0, v_a_5317_);
v___x_5322_ = v_reuseFailAlloc_5323_;
goto v_reusejp_5321_;
}
v_reusejp_5321_:
{
return v___x_5322_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1(lean_object* v_a_5374_, lean_object* v_a_5375_, lean_object* v___x_5376_, lean_object* v_mvarCounterSaved_5377_, lean_object* v___x_5378_, uint8_t v___y_5379_, uint8_t v___x_5380_, lean_object* v_x_5381_, lean_object* v___y_5382_, lean_object* v___y_5383_, lean_object* v___y_5384_, lean_object* v___y_5385_, lean_object* v___y_5386_){
_start:
{
lean_object* v___x_5388_; lean_object* v___x_5389_; lean_object* v___x_5390_; lean_object* v___x_5391_; lean_object* v___x_5392_; 
v___x_5388_ = l_Lean_Expr_bindingBody_x21(v_a_5374_);
v___x_5389_ = lean_expr_instantiate1(v___x_5388_, v_x_5381_);
lean_dec_ref(v___x_5388_);
v___x_5390_ = l_Lean_Expr_bindingBody_x21(v_a_5375_);
v___x_5391_ = lean_expr_instantiate1(v___x_5390_, v_x_5381_);
lean_dec_ref(v___x_5390_);
v___x_5392_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v___x_5376_, v_mvarCounterSaved_5377_, v___x_5389_, v___x_5391_, v___y_5382_, v___y_5383_, v___y_5384_, v___y_5385_, v___y_5386_);
if (lean_obj_tag(v___x_5392_) == 0)
{
lean_object* v_a_5393_; lean_object* v_lhs_5394_; lean_object* v_rhs_5395_; lean_object* v___x_5396_; lean_object* v___x_5397_; uint8_t v___x_5398_; lean_object* v___x_5399_; 
v_a_5393_ = lean_ctor_get(v___x_5392_, 0);
lean_inc(v_a_5393_);
lean_dec_ref_known(v___x_5392_, 1);
v_lhs_5394_ = lean_ctor_get(v_a_5393_, 0);
v_rhs_5395_ = lean_ctor_get(v_a_5393_, 1);
v___x_5396_ = lean_mk_empty_array_with_capacity(v___x_5378_);
lean_inc_ref(v___x_5396_);
v___x_5397_ = lean_array_push(v___x_5396_, v_x_5381_);
v___x_5398_ = 1;
lean_inc_ref(v_lhs_5394_);
v___x_5399_ = l_Lean_Meta_mkForallFVars(v___x_5397_, v_lhs_5394_, v___y_5379_, v___x_5380_, v___x_5380_, v___x_5398_, v___y_5383_, v___y_5384_, v___y_5385_, v___y_5386_);
if (lean_obj_tag(v___x_5399_) == 0)
{
lean_object* v_a_5400_; lean_object* v___x_5401_; 
v_a_5400_ = lean_ctor_get(v___x_5399_, 0);
lean_inc(v_a_5400_);
lean_dec_ref_known(v___x_5399_, 1);
lean_inc_ref(v_rhs_5395_);
v___x_5401_ = l_Lean_Meta_mkForallFVars(v___x_5397_, v_rhs_5395_, v___y_5379_, v___x_5380_, v___x_5380_, v___x_5398_, v___y_5383_, v___y_5384_, v___y_5385_, v___y_5386_);
if (lean_obj_tag(v___x_5401_) == 0)
{
lean_object* v_a_5402_; lean_object* v___x_5404_; uint8_t v_isShared_5405_; uint8_t v_isSharedCheck_5461_; 
v_a_5402_ = lean_ctor_get(v___x_5401_, 0);
v_isSharedCheck_5461_ = !lean_is_exclusive(v___x_5401_);
if (v_isSharedCheck_5461_ == 0)
{
v___x_5404_ = v___x_5401_;
v_isShared_5405_ = v_isSharedCheck_5461_;
goto v_resetjp_5403_;
}
else
{
lean_inc(v_a_5402_);
lean_dec(v___x_5401_);
v___x_5404_ = lean_box(0);
v_isShared_5405_ = v_isSharedCheck_5461_;
goto v_resetjp_5403_;
}
v_resetjp_5403_:
{
uint8_t v___x_5406_; 
v___x_5406_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_isRfl(v_a_5393_);
if (v___x_5406_ == 0)
{
lean_object* v___x_5407_; 
lean_del_object(v___x_5404_);
v___x_5407_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_a_5393_, v___y_5383_, v___y_5384_, v___y_5385_, v___y_5386_);
if (lean_obj_tag(v___x_5407_) == 0)
{
lean_object* v_a_5408_; lean_object* v___x_5409_; 
v_a_5408_ = lean_ctor_get(v___x_5407_, 0);
lean_inc(v_a_5408_);
lean_dec_ref_known(v___x_5407_, 1);
v___x_5409_ = l_Lean_Meta_mkLambdaFVars(v___x_5397_, v_a_5408_, v___y_5379_, v___x_5380_, v___y_5379_, v___x_5380_, v___x_5398_, v___y_5383_, v___y_5384_, v___y_5385_, v___y_5386_);
lean_dec_ref(v___x_5397_);
if (lean_obj_tag(v___x_5409_) == 0)
{
lean_object* v_a_5410_; lean_object* v___x_5411_; lean_object* v___x_5412_; lean_object* v___x_5413_; 
v_a_5410_ = lean_ctor_get(v___x_5409_, 0);
lean_inc(v_a_5410_);
lean_dec_ref_known(v___x_5409_, 1);
v___x_5411_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___lam__1___closed__1));
v___x_5412_ = lean_array_push(v___x_5396_, v_a_5410_);
v___x_5413_ = l_Lean_Meta_mkAppM(v___x_5411_, v___x_5412_, v___y_5383_, v___y_5384_, v___y_5385_, v___y_5386_);
if (lean_obj_tag(v___x_5413_) == 0)
{
lean_object* v_a_5414_; lean_object* v___x_5416_; uint8_t v_isShared_5417_; uint8_t v_isSharedCheck_5422_; 
v_a_5414_ = lean_ctor_get(v___x_5413_, 0);
v_isSharedCheck_5422_ = !lean_is_exclusive(v___x_5413_);
if (v_isSharedCheck_5422_ == 0)
{
v___x_5416_ = v___x_5413_;
v_isShared_5417_ = v_isSharedCheck_5422_;
goto v_resetjp_5415_;
}
else
{
lean_inc(v_a_5414_);
lean_dec(v___x_5413_);
v___x_5416_ = lean_box(0);
v_isShared_5417_ = v_isSharedCheck_5422_;
goto v_resetjp_5415_;
}
v_resetjp_5415_:
{
lean_object* v___x_5418_; lean_object* v___x_5420_; 
v___x_5418_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_mk_x27(v_a_5400_, v_a_5402_, v_a_5414_);
if (v_isShared_5417_ == 0)
{
lean_ctor_set(v___x_5416_, 0, v___x_5418_);
v___x_5420_ = v___x_5416_;
goto v_reusejp_5419_;
}
else
{
lean_object* v_reuseFailAlloc_5421_; 
v_reuseFailAlloc_5421_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5421_, 0, v___x_5418_);
v___x_5420_ = v_reuseFailAlloc_5421_;
goto v_reusejp_5419_;
}
v_reusejp_5419_:
{
return v___x_5420_;
}
}
}
else
{
lean_object* v_a_5423_; lean_object* v___x_5425_; uint8_t v_isShared_5426_; uint8_t v_isSharedCheck_5430_; 
lean_dec(v_a_5402_);
lean_dec(v_a_5400_);
v_a_5423_ = lean_ctor_get(v___x_5413_, 0);
v_isSharedCheck_5430_ = !lean_is_exclusive(v___x_5413_);
if (v_isSharedCheck_5430_ == 0)
{
v___x_5425_ = v___x_5413_;
v_isShared_5426_ = v_isSharedCheck_5430_;
goto v_resetjp_5424_;
}
else
{
lean_inc(v_a_5423_);
lean_dec(v___x_5413_);
v___x_5425_ = lean_box(0);
v_isShared_5426_ = v_isSharedCheck_5430_;
goto v_resetjp_5424_;
}
v_resetjp_5424_:
{
lean_object* v___x_5428_; 
if (v_isShared_5426_ == 0)
{
v___x_5428_ = v___x_5425_;
goto v_reusejp_5427_;
}
else
{
lean_object* v_reuseFailAlloc_5429_; 
v_reuseFailAlloc_5429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5429_, 0, v_a_5423_);
v___x_5428_ = v_reuseFailAlloc_5429_;
goto v_reusejp_5427_;
}
v_reusejp_5427_:
{
return v___x_5428_;
}
}
}
}
else
{
lean_object* v_a_5431_; lean_object* v___x_5433_; uint8_t v_isShared_5434_; uint8_t v_isSharedCheck_5438_; 
lean_dec(v_a_5402_);
lean_dec(v_a_5400_);
lean_dec_ref(v___x_5396_);
v_a_5431_ = lean_ctor_get(v___x_5409_, 0);
v_isSharedCheck_5438_ = !lean_is_exclusive(v___x_5409_);
if (v_isSharedCheck_5438_ == 0)
{
v___x_5433_ = v___x_5409_;
v_isShared_5434_ = v_isSharedCheck_5438_;
goto v_resetjp_5432_;
}
else
{
lean_inc(v_a_5431_);
lean_dec(v___x_5409_);
v___x_5433_ = lean_box(0);
v_isShared_5434_ = v_isSharedCheck_5438_;
goto v_resetjp_5432_;
}
v_resetjp_5432_:
{
lean_object* v___x_5436_; 
if (v_isShared_5434_ == 0)
{
v___x_5436_ = v___x_5433_;
goto v_reusejp_5435_;
}
else
{
lean_object* v_reuseFailAlloc_5437_; 
v_reuseFailAlloc_5437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5437_, 0, v_a_5431_);
v___x_5436_ = v_reuseFailAlloc_5437_;
goto v_reusejp_5435_;
}
v_reusejp_5435_:
{
return v___x_5436_;
}
}
}
}
else
{
lean_object* v_a_5439_; lean_object* v___x_5441_; uint8_t v_isShared_5442_; uint8_t v_isSharedCheck_5446_; 
lean_dec(v_a_5402_);
lean_dec(v_a_5400_);
lean_dec_ref(v___x_5397_);
lean_dec_ref(v___x_5396_);
v_a_5439_ = lean_ctor_get(v___x_5407_, 0);
v_isSharedCheck_5446_ = !lean_is_exclusive(v___x_5407_);
if (v_isSharedCheck_5446_ == 0)
{
v___x_5441_ = v___x_5407_;
v_isShared_5442_ = v_isSharedCheck_5446_;
goto v_resetjp_5440_;
}
else
{
lean_inc(v_a_5439_);
lean_dec(v___x_5407_);
v___x_5441_ = lean_box(0);
v_isShared_5442_ = v_isSharedCheck_5446_;
goto v_resetjp_5440_;
}
v_resetjp_5440_:
{
lean_object* v___x_5444_; 
if (v_isShared_5442_ == 0)
{
v___x_5444_ = v___x_5441_;
goto v_reusejp_5443_;
}
else
{
lean_object* v_reuseFailAlloc_5445_; 
v_reuseFailAlloc_5445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5445_, 0, v_a_5439_);
v___x_5444_ = v_reuseFailAlloc_5445_;
goto v_reusejp_5443_;
}
v_reusejp_5443_:
{
return v___x_5444_;
}
}
}
}
else
{
lean_object* v___x_5448_; uint8_t v_isShared_5449_; uint8_t v_isSharedCheck_5457_; 
lean_dec_ref(v___x_5397_);
lean_dec_ref(v___x_5396_);
v_isSharedCheck_5457_ = !lean_is_exclusive(v_a_5393_);
if (v_isSharedCheck_5457_ == 0)
{
lean_object* v_unused_5458_; lean_object* v_unused_5459_; lean_object* v_unused_5460_; 
v_unused_5458_ = lean_ctor_get(v_a_5393_, 2);
lean_dec(v_unused_5458_);
v_unused_5459_ = lean_ctor_get(v_a_5393_, 1);
lean_dec(v_unused_5459_);
v_unused_5460_ = lean_ctor_get(v_a_5393_, 0);
lean_dec(v_unused_5460_);
v___x_5448_ = v_a_5393_;
v_isShared_5449_ = v_isSharedCheck_5457_;
goto v_resetjp_5447_;
}
else
{
lean_dec(v_a_5393_);
v___x_5448_ = lean_box(0);
v_isShared_5449_ = v_isSharedCheck_5457_;
goto v_resetjp_5447_;
}
v_resetjp_5447_:
{
lean_object* v___x_5450_; lean_object* v___x_5452_; 
v___x_5450_ = lean_box(0);
if (v_isShared_5449_ == 0)
{
lean_ctor_set(v___x_5448_, 2, v___x_5450_);
lean_ctor_set(v___x_5448_, 1, v_a_5402_);
lean_ctor_set(v___x_5448_, 0, v_a_5400_);
v___x_5452_ = v___x_5448_;
goto v_reusejp_5451_;
}
else
{
lean_object* v_reuseFailAlloc_5456_; 
v_reuseFailAlloc_5456_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_5456_, 0, v_a_5400_);
lean_ctor_set(v_reuseFailAlloc_5456_, 1, v_a_5402_);
lean_ctor_set(v_reuseFailAlloc_5456_, 2, v___x_5450_);
v___x_5452_ = v_reuseFailAlloc_5456_;
goto v_reusejp_5451_;
}
v_reusejp_5451_:
{
lean_object* v___x_5454_; 
if (v_isShared_5405_ == 0)
{
lean_ctor_set(v___x_5404_, 0, v___x_5452_);
v___x_5454_ = v___x_5404_;
goto v_reusejp_5453_;
}
else
{
lean_object* v_reuseFailAlloc_5455_; 
v_reuseFailAlloc_5455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5455_, 0, v___x_5452_);
v___x_5454_ = v_reuseFailAlloc_5455_;
goto v_reusejp_5453_;
}
v_reusejp_5453_:
{
return v___x_5454_;
}
}
}
}
}
}
else
{
lean_object* v_a_5462_; lean_object* v___x_5464_; uint8_t v_isShared_5465_; uint8_t v_isSharedCheck_5469_; 
lean_dec(v_a_5400_);
lean_dec_ref(v___x_5397_);
lean_dec_ref(v___x_5396_);
lean_dec(v_a_5393_);
v_a_5462_ = lean_ctor_get(v___x_5401_, 0);
v_isSharedCheck_5469_ = !lean_is_exclusive(v___x_5401_);
if (v_isSharedCheck_5469_ == 0)
{
v___x_5464_ = v___x_5401_;
v_isShared_5465_ = v_isSharedCheck_5469_;
goto v_resetjp_5463_;
}
else
{
lean_inc(v_a_5462_);
lean_dec(v___x_5401_);
v___x_5464_ = lean_box(0);
v_isShared_5465_ = v_isSharedCheck_5469_;
goto v_resetjp_5463_;
}
v_resetjp_5463_:
{
lean_object* v___x_5467_; 
if (v_isShared_5465_ == 0)
{
v___x_5467_ = v___x_5464_;
goto v_reusejp_5466_;
}
else
{
lean_object* v_reuseFailAlloc_5468_; 
v_reuseFailAlloc_5468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5468_, 0, v_a_5462_);
v___x_5467_ = v_reuseFailAlloc_5468_;
goto v_reusejp_5466_;
}
v_reusejp_5466_:
{
return v___x_5467_;
}
}
}
}
else
{
lean_object* v_a_5470_; lean_object* v___x_5472_; uint8_t v_isShared_5473_; uint8_t v_isSharedCheck_5477_; 
lean_dec_ref(v___x_5397_);
lean_dec_ref(v___x_5396_);
lean_dec(v_a_5393_);
v_a_5470_ = lean_ctor_get(v___x_5399_, 0);
v_isSharedCheck_5477_ = !lean_is_exclusive(v___x_5399_);
if (v_isSharedCheck_5477_ == 0)
{
v___x_5472_ = v___x_5399_;
v_isShared_5473_ = v_isSharedCheck_5477_;
goto v_resetjp_5471_;
}
else
{
lean_inc(v_a_5470_);
lean_dec(v___x_5399_);
v___x_5472_ = lean_box(0);
v_isShared_5473_ = v_isSharedCheck_5477_;
goto v_resetjp_5471_;
}
v_resetjp_5471_:
{
lean_object* v___x_5475_; 
if (v_isShared_5473_ == 0)
{
v___x_5475_ = v___x_5472_;
goto v_reusejp_5474_;
}
else
{
lean_object* v_reuseFailAlloc_5476_; 
v_reuseFailAlloc_5476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5476_, 0, v_a_5470_);
v___x_5475_ = v_reuseFailAlloc_5476_;
goto v_reusejp_5474_;
}
v_reusejp_5474_:
{
return v___x_5475_;
}
}
}
}
else
{
lean_dec_ref(v_x_5381_);
return v___x_5392_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg___boxed(lean_object* v___x_5478_, lean_object* v_mvarCounterSaved_5479_, lean_object* v_a_5480_, lean_object* v_b_5481_, lean_object* v___y_5482_, lean_object* v___y_5483_, lean_object* v___y_5484_, lean_object* v___y_5485_, lean_object* v___y_5486_, lean_object* v___y_5487_){
_start:
{
lean_object* v_res_5488_; 
v_res_5488_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg(v___x_5478_, v_mvarCounterSaved_5479_, v_a_5480_, v_b_5481_, v___y_5482_, v___y_5483_, v___y_5484_, v___y_5485_, v___y_5486_);
lean_dec(v___y_5486_);
lean_dec_ref(v___y_5485_);
lean_dec(v___y_5484_);
lean_dec_ref(v___y_5483_);
lean_dec(v___y_5482_);
return v_res_5488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp___boxed(lean_object* v_depth_5489_, lean_object* v_mvarCounterSaved_5490_, lean_object* v_lhs_5491_, lean_object* v_rhs_5492_, lean_object* v_a_5493_, lean_object* v_a_5494_, lean_object* v_a_5495_, lean_object* v_a_5496_, lean_object* v_a_5497_, lean_object* v_a_5498_){
_start:
{
lean_object* v_res_5499_; 
v_res_5499_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfApp(v_depth_5489_, v_mvarCounterSaved_5490_, v_lhs_5491_, v_rhs_5492_, v_a_5493_, v_a_5494_, v_a_5495_, v_a_5496_, v_a_5497_);
lean_dec(v_a_5497_);
lean_dec_ref(v_a_5496_);
lean_dec(v_a_5495_);
lean_dec_ref(v_a_5494_);
lean_dec(v_a_5493_);
lean_dec(v_depth_5489_);
return v_res_5499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux___boxed(lean_object* v_depth_5500_, lean_object* v_mvarCounterSaved_5501_, lean_object* v_lhs_5502_, lean_object* v_rhs_5503_, lean_object* v_a_5504_, lean_object* v_a_5505_, lean_object* v_a_5506_, lean_object* v_a_5507_, lean_object* v_a_5508_, lean_object* v_a_5509_){
_start:
{
lean_object* v_res_5510_; 
v_res_5510_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v_depth_5500_, v_mvarCounterSaved_5501_, v_lhs_5502_, v_rhs_5503_, v_a_5504_, v_a_5505_, v_a_5506_, v_a_5507_, v_a_5508_);
lean_dec(v_a_5508_);
lean_dec_ref(v_a_5507_);
lean_dec(v_a_5506_);
lean_dec_ref(v_a_5505_);
lean_dec(v_a_5504_);
return v_res_5510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go___boxed(lean_object** _args){
lean_object* v_depth_5511_ = _args[0];
lean_object* v_mvarCounterSaved_5512_ = _args[1];
lean_object* v_arity_5513_ = _args[2];
lean_object* v_lhsArgs_5514_ = _args[3];
lean_object* v_rhsArgs_5515_ = _args[4];
lean_object* v_i_5516_ = _args[5];
lean_object* v_finfo_5517_ = _args[6];
lean_object* v_finfoIdx_5518_ = _args[7];
lean_object* v_f_5519_ = _args[8];
lean_object* v_f_x27_5520_ = _args[9];
lean_object* v_pf_5521_ = _args[10];
lean_object* v_a_5522_ = _args[11];
lean_object* v_a_5523_ = _args[12];
lean_object* v_a_5524_ = _args[13];
lean_object* v_a_5525_ = _args[14];
lean_object* v_a_5526_ = _args[15];
lean_object* v_a_5527_ = _args[16];
_start:
{
lean_object* v_res_5528_; 
v_res_5528_ = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go(v_depth_5511_, v_mvarCounterSaved_5512_, v_arity_5513_, v_lhsArgs_5514_, v_rhsArgs_5515_, v_i_5516_, v_finfo_5517_, v_finfoIdx_5518_, v_f_5519_, v_f_x27_5520_, v_pf_5521_, v_a_5522_, v_a_5523_, v_a_5524_, v_a_5525_, v_a_5526_);
lean_dec(v_a_5526_);
lean_dec_ref(v_a_5525_);
lean_dec(v_a_5524_);
lean_dec_ref(v_a_5523_);
lean_dec(v_a_5522_);
lean_dec(v_depth_5511_);
return v_res_5528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7(lean_object* v_e_5529_, lean_object* v___y_5530_, lean_object* v___y_5531_, lean_object* v___y_5532_, lean_object* v___y_5533_, lean_object* v___y_5534_){
_start:
{
lean_object* v___x_5536_; 
v___x_5536_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___redArg(v_e_5529_, v___y_5532_);
return v___x_5536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7___boxed(lean_object* v_e_5537_, lean_object* v___y_5538_, lean_object* v___y_5539_, lean_object* v___y_5540_, lean_object* v___y_5541_, lean_object* v___y_5542_, lean_object* v___y_5543_){
_start:
{
lean_object* v_res_5544_; 
v_res_5544_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__7(v_e_5537_, v___y_5538_, v___y_5539_, v___y_5540_, v___y_5541_, v___y_5542_);
lean_dec(v___y_5542_);
lean_dec_ref(v___y_5541_);
lean_dec(v___y_5540_);
lean_dec_ref(v___y_5539_);
lean_dec(v___y_5538_);
return v_res_5544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10(lean_object* v_00_u03b1_5545_, lean_object* v_name_5546_, uint8_t v_bi_5547_, lean_object* v_type_5548_, lean_object* v_k_5549_, uint8_t v_kind_5550_, lean_object* v___y_5551_, lean_object* v___y_5552_, lean_object* v___y_5553_, lean_object* v___y_5554_, lean_object* v___y_5555_){
_start:
{
lean_object* v___x_5557_; 
v___x_5557_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___redArg(v_name_5546_, v_bi_5547_, v_type_5548_, v_k_5549_, v_kind_5550_, v___y_5551_, v___y_5552_, v___y_5553_, v___y_5554_, v___y_5555_);
return v___x_5557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10___boxed(lean_object* v_00_u03b1_5558_, lean_object* v_name_5559_, lean_object* v_bi_5560_, lean_object* v_type_5561_, lean_object* v_k_5562_, lean_object* v_kind_5563_, lean_object* v___y_5564_, lean_object* v___y_5565_, lean_object* v___y_5566_, lean_object* v___y_5567_, lean_object* v___y_5568_, lean_object* v___y_5569_){
_start:
{
uint8_t v_bi_boxed_5570_; uint8_t v_kind_boxed_5571_; lean_object* v_res_5572_; 
v_bi_boxed_5570_ = lean_unbox(v_bi_5560_);
v_kind_boxed_5571_ = lean_unbox(v_kind_5563_);
v_res_5572_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__10(v_00_u03b1_5558_, v_name_5559_, v_bi_boxed_5570_, v_type_5561_, v_k_5562_, v_kind_boxed_5571_, v___y_5564_, v___y_5565_, v___y_5566_, v___y_5567_, v___y_5568_);
lean_dec(v___y_5568_);
lean_dec_ref(v___y_5567_);
lean_dec(v___y_5566_);
lean_dec_ref(v___y_5565_);
lean_dec(v___y_5564_);
return v_res_5572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1(lean_object* v_as_5573_, size_t v_sz_5574_, size_t v_i_5575_, lean_object* v_b_5576_, lean_object* v___y_5577_, lean_object* v___y_5578_, lean_object* v___y_5579_, lean_object* v___y_5580_, lean_object* v___y_5581_){
_start:
{
lean_object* v___x_5583_; 
v___x_5583_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___redArg(v_as_5573_, v_sz_5574_, v_i_5575_, v_b_5576_, v___y_5578_, v___y_5579_, v___y_5580_, v___y_5581_);
return v___x_5583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1___boxed(lean_object* v_as_5584_, lean_object* v_sz_5585_, lean_object* v_i_5586_, lean_object* v_b_5587_, lean_object* v___y_5588_, lean_object* v___y_5589_, lean_object* v___y_5590_, lean_object* v___y_5591_, lean_object* v___y_5592_, lean_object* v___y_5593_){
_start:
{
size_t v_sz_boxed_5594_; size_t v_i_boxed_5595_; lean_object* v_res_5596_; 
v_sz_boxed_5594_ = lean_unbox_usize(v_sz_5585_);
lean_dec(v_sz_5585_);
v_i_boxed_5595_ = lean_unbox_usize(v_i_5586_);
lean_dec(v_i_5586_);
v_res_5596_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__1(v_as_5584_, v_sz_boxed_5594_, v_i_boxed_5595_, v_b_5587_, v___y_5588_, v___y_5589_, v___y_5590_, v___y_5591_, v___y_5592_);
lean_dec(v___y_5592_);
lean_dec_ref(v___y_5591_);
lean_dec(v___y_5590_);
lean_dec_ref(v___y_5589_);
lean_dec(v___y_5588_);
lean_dec_ref(v_as_5584_);
return v_res_5596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3(lean_object* v___x_5597_, lean_object* v_mvarCounterSaved_5598_, lean_object* v_inst_5599_, lean_object* v_R_5600_, lean_object* v_a_5601_, lean_object* v_b_5602_, lean_object* v_c_5603_, lean_object* v___y_5604_, lean_object* v___y_5605_, lean_object* v___y_5606_, lean_object* v___y_5607_, lean_object* v___y_5608_){
_start:
{
lean_object* v___x_5610_; 
v___x_5610_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___redArg(v___x_5597_, v_mvarCounterSaved_5598_, v_a_5601_, v_b_5602_, v___y_5604_, v___y_5605_, v___y_5606_, v___y_5607_, v___y_5608_);
return v___x_5610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3___boxed(lean_object* v___x_5611_, lean_object* v_mvarCounterSaved_5612_, lean_object* v_inst_5613_, lean_object* v_R_5614_, lean_object* v_a_5615_, lean_object* v_b_5616_, lean_object* v_c_5617_, lean_object* v___y_5618_, lean_object* v___y_5619_, lean_object* v___y_5620_, lean_object* v___y_5621_, lean_object* v___y_5622_, lean_object* v___y_5623_){
_start:
{
lean_object* v_res_5624_; 
v_res_5624_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__3(v___x_5611_, v_mvarCounterSaved_5612_, v_inst_5613_, v_R_5614_, v_a_5615_, v_b_5616_, v_c_5617_, v___y_5618_, v___y_5619_, v___y_5620_, v___y_5621_, v___y_5622_);
lean_dec(v___y_5622_);
lean_dec_ref(v___y_5621_);
lean_dec(v___y_5620_);
lean_dec_ref(v___y_5619_);
lean_dec(v___y_5618_);
return v_res_5624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4(lean_object* v_cls_5625_, lean_object* v_msg_5626_, lean_object* v___y_5627_, lean_object* v___y_5628_, lean_object* v___y_5629_, lean_object* v___y_5630_, lean_object* v___y_5631_){
_start:
{
lean_object* v___x_5633_; 
v___x_5633_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___redArg(v_cls_5625_, v_msg_5626_, v___y_5628_, v___y_5629_, v___y_5630_, v___y_5631_);
return v___x_5633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4___boxed(lean_object* v_cls_5634_, lean_object* v_msg_5635_, lean_object* v___y_5636_, lean_object* v___y_5637_, lean_object* v___y_5638_, lean_object* v___y_5639_, lean_object* v___y_5640_, lean_object* v___y_5641_){
_start:
{
lean_object* v_res_5642_; 
v_res_5642_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_mkCongrOfApp_go_spec__4(v_cls_5634_, v_msg_5635_, v___y_5636_, v___y_5637_, v___y_5638_, v___y_5639_, v___y_5640_);
lean_dec(v___y_5640_);
lean_dec_ref(v___y_5639_);
lean_dec(v___y_5638_);
lean_dec_ref(v___y_5637_);
lean_dec(v___y_5636_);
return v_res_5642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8(lean_object* v_00_u03b2_5643_, lean_object* v_m_5644_, lean_object* v_a_5645_, lean_object* v_b_5646_){
_start:
{
lean_object* v___x_5647_; 
v___x_5647_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8___redArg(v_m_5644_, v_a_5645_, v_b_5646_);
return v___x_5647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9(lean_object* v_00_u03b2_5648_, lean_object* v_m_5649_, lean_object* v_a_5650_){
_start:
{
lean_object* v___x_5651_; 
v___x_5651_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___redArg(v_m_5649_, v_a_5650_);
return v___x_5651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9___boxed(lean_object* v_00_u03b2_5652_, lean_object* v_m_5653_, lean_object* v_a_5654_){
_start:
{
lean_object* v_res_5655_; 
v_res_5655_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9(v_00_u03b2_5652_, v_m_5653_, v_a_5654_);
lean_dec_ref(v_a_5654_);
lean_dec_ref(v_m_5653_);
return v_res_5655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11(lean_object* v_00_u03b1_5656_, lean_object* v_msg_5657_, lean_object* v___y_5658_, lean_object* v___y_5659_, lean_object* v___y_5660_, lean_object* v___y_5661_, lean_object* v___y_5662_){
_start:
{
lean_object* v___x_5664_; 
v___x_5664_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___redArg(v_msg_5657_, v___y_5659_, v___y_5660_, v___y_5661_, v___y_5662_);
return v___x_5664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11___boxed(lean_object* v_00_u03b1_5665_, lean_object* v_msg_5666_, lean_object* v___y_5667_, lean_object* v___y_5668_, lean_object* v___y_5669_, lean_object* v___y_5670_, lean_object* v___y_5671_, lean_object* v___y_5672_){
_start:
{
lean_object* v_res_5673_; 
v_res_5673_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__11(v_00_u03b1_5665_, v_msg_5666_, v___y_5667_, v___y_5668_, v___y_5669_, v___y_5670_, v___y_5671_);
lean_dec(v___y_5671_);
lean_dec_ref(v___y_5670_);
lean_dec(v___y_5669_);
lean_dec_ref(v___y_5668_);
lean_dec(v___y_5667_);
return v_res_5673_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8(lean_object* v_00_u03b2_5674_, lean_object* v_a_5675_, lean_object* v_x_5676_){
_start:
{
uint8_t v___x_5677_; 
v___x_5677_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___redArg(v_a_5675_, v_x_5676_);
return v___x_5677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8___boxed(lean_object* v_00_u03b2_5678_, lean_object* v_a_5679_, lean_object* v_x_5680_){
_start:
{
uint8_t v_res_5681_; lean_object* v_r_5682_; 
v_res_5681_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__8(v_00_u03b2_5678_, v_a_5679_, v_x_5680_);
lean_dec(v_x_5680_);
lean_dec_ref(v_a_5679_);
v_r_5682_ = lean_box(v_res_5681_);
return v_r_5682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9(lean_object* v_00_u03b2_5683_, lean_object* v_data_5684_){
_start:
{
lean_object* v___x_5685_; 
v___x_5685_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9___redArg(v_data_5684_);
return v___x_5685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__10(lean_object* v_00_u03b2_5686_, lean_object* v_a_5687_, lean_object* v_b_5688_, lean_object* v_x_5689_){
_start:
{
lean_object* v___x_5690_; 
v___x_5690_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__10___redArg(v_a_5687_, v_b_5688_, v_x_5689_);
return v___x_5690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12(lean_object* v_00_u03b2_5691_, lean_object* v_a_5692_, lean_object* v_x_5693_){
_start:
{
lean_object* v___x_5694_; 
v___x_5694_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___redArg(v_a_5692_, v_x_5693_);
return v___x_5694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12___boxed(lean_object* v_00_u03b2_5695_, lean_object* v_a_5696_, lean_object* v_x_5697_){
_start:
{
lean_object* v_res_5698_; 
v_res_5698_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__9_spec__12(v_00_u03b2_5695_, v_a_5696_, v_x_5697_);
lean_dec(v_x_5697_);
lean_dec_ref(v_a_5696_);
return v_res_5698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12(lean_object* v_00_u03b2_5699_, lean_object* v_i_5700_, lean_object* v_source_5701_, lean_object* v_target_5702_){
_start:
{
lean_object* v___x_5703_; 
v___x_5703_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12___redArg(v_i_5700_, v_source_5701_, v_target_5702_);
return v___x_5703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12_spec__15(lean_object* v_00_u03b2_5704_, lean_object* v_x_5705_, lean_object* v_x_5706_){
_start:
{
lean_object* v___x_5707_; 
v___x_5707_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_TermCongr_mkCongrOfAux_spec__8_spec__9_spec__12_spec__15___redArg(v_x_5705_, v_x_5706_);
return v___x_5707_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__0(void){
_start:
{
lean_object* v___x_5708_; lean_object* v___x_5709_; lean_object* v___x_5710_; 
v___x_5708_ = lean_box(0);
v___x_5709_ = lean_unsigned_to_nat(16u);
v___x_5710_ = lean_mk_array(v___x_5709_, v___x_5708_);
return v___x_5710_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__1(void){
_start:
{
lean_object* v___x_5711_; lean_object* v___x_5712_; lean_object* v___x_5713_; 
v___x_5711_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__0, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__0);
v___x_5712_ = lean_unsigned_to_nat(0u);
v___x_5713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5713_, 0, v___x_5712_);
lean_ctor_set(v___x_5713_, 1, v___x_5711_);
return v___x_5713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf(lean_object* v_depth_5714_, lean_object* v_mvarCounterSaved_5715_, lean_object* v_lhs_5716_, lean_object* v_rhs_5717_, lean_object* v_a_5718_, lean_object* v_a_5719_, lean_object* v_a_5720_, lean_object* v_a_5721_){
_start:
{
lean_object* v___x_5723_; lean_object* v___x_5724_; lean_object* v___x_5725_; 
v___x_5723_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__1, &lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___closed__1);
v___x_5724_ = lean_st_mk_ref(v___x_5723_);
v___x_5725_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOfAux(v_depth_5714_, v_mvarCounterSaved_5715_, v_lhs_5716_, v_rhs_5717_, v___x_5724_, v_a_5718_, v_a_5719_, v_a_5720_, v_a_5721_);
if (lean_obj_tag(v___x_5725_) == 0)
{
lean_object* v_a_5726_; lean_object* v___x_5728_; uint8_t v_isShared_5729_; uint8_t v_isSharedCheck_5734_; 
v_a_5726_ = lean_ctor_get(v___x_5725_, 0);
v_isSharedCheck_5734_ = !lean_is_exclusive(v___x_5725_);
if (v_isSharedCheck_5734_ == 0)
{
v___x_5728_ = v___x_5725_;
v_isShared_5729_ = v_isSharedCheck_5734_;
goto v_resetjp_5727_;
}
else
{
lean_inc(v_a_5726_);
lean_dec(v___x_5725_);
v___x_5728_ = lean_box(0);
v_isShared_5729_ = v_isSharedCheck_5734_;
goto v_resetjp_5727_;
}
v_resetjp_5727_:
{
lean_object* v___x_5730_; lean_object* v___x_5732_; 
v___x_5730_ = lean_st_ref_get(v___x_5724_);
lean_dec(v___x_5724_);
lean_dec(v___x_5730_);
if (v_isShared_5729_ == 0)
{
v___x_5732_ = v___x_5728_;
goto v_reusejp_5731_;
}
else
{
lean_object* v_reuseFailAlloc_5733_; 
v_reuseFailAlloc_5733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5733_, 0, v_a_5726_);
v___x_5732_ = v_reuseFailAlloc_5733_;
goto v_reusejp_5731_;
}
v_reusejp_5731_:
{
return v___x_5732_;
}
}
}
else
{
lean_dec(v___x_5724_);
return v___x_5725_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf___boxed(lean_object* v_depth_5735_, lean_object* v_mvarCounterSaved_5736_, lean_object* v_lhs_5737_, lean_object* v_rhs_5738_, lean_object* v_a_5739_, lean_object* v_a_5740_, lean_object* v_a_5741_, lean_object* v_a_5742_, lean_object* v_a_5743_){
_start:
{
lean_object* v_res_5744_; 
v_res_5744_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf(v_depth_5735_, v_mvarCounterSaved_5736_, v_lhs_5737_, v_rhs_5738_, v_a_5739_, v_a_5740_, v_a_5741_, v_a_5742_);
lean_dec(v_a_5742_);
lean_dec_ref(v_a_5741_);
lean_dec(v_a_5740_);
lean_dec_ref(v_a_5739_);
return v_res_5744_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___closed__0(void){
_start:
{
lean_object* v___x_5745_; 
v___x_5745_ = l_Lean_Elab_Term_instInhabitedTermElabM(lean_box(0));
return v___x_5745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0(lean_object* v_msg_5746_, lean_object* v___y_5747_, lean_object* v___y_5748_, lean_object* v___y_5749_, lean_object* v___y_5750_, lean_object* v___y_5751_, lean_object* v___y_5752_){
_start:
{
lean_object* v___x_5754_; lean_object* v___x_6895__overap_5755_; lean_object* v___x_5756_; 
v___x_5754_ = lean_obj_once(&lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___closed__0, &lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___closed__0_once, _init_lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___closed__0);
v___x_6895__overap_5755_ = lean_panic_fn_borrowed(v___x_5754_, v_msg_5746_);
lean_inc(v___y_5752_);
lean_inc_ref(v___y_5751_);
lean_inc(v___y_5750_);
lean_inc_ref(v___y_5749_);
lean_inc(v___y_5748_);
lean_inc_ref(v___y_5747_);
v___x_5756_ = lean_apply_7(v___x_6895__overap_5755_, v___y_5747_, v___y_5748_, v___y_5749_, v___y_5750_, v___y_5751_, v___y_5752_, lean_box(0));
return v___x_5756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0___boxed(lean_object* v_msg_5757_, lean_object* v___y_5758_, lean_object* v___y_5759_, lean_object* v___y_5760_, lean_object* v___y_5761_, lean_object* v___y_5762_, lean_object* v___y_5763_, lean_object* v___y_5764_){
_start:
{
lean_object* v_res_5765_; 
v_res_5765_ = lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0(v_msg_5757_, v___y_5758_, v___y_5759_, v___y_5760_, v___y_5761_, v___y_5762_, v___y_5763_);
lean_dec(v___y_5763_);
lean_dec_ref(v___y_5762_);
lean_dec(v___y_5761_);
lean_dec_ref(v___y_5760_);
lean_dec(v___y_5759_);
lean_dec_ref(v___y_5758_);
return v_res_5765_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__2(void){
_start:
{
lean_object* v___x_5768_; lean_object* v___x_5769_; lean_object* v___x_5770_; lean_object* v___x_5771_; lean_object* v___x_5772_; lean_object* v___x_5773_; 
v___x_5768_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__1));
v___x_5769_ = lean_unsigned_to_nat(23u);
v___x_5770_ = lean_unsigned_to_nat(727u);
v___x_5771_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__0));
v___x_5772_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_CongrResult_mk_x27_ensureSidesDefeq___closed__6));
v___x_5773_ = l_mkPanicMessageWithDecl(v___x_5772_, v___x_5771_, v___x_5770_, v___x_5769_, v___x_5768_);
return v___x_5773_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__4(void){
_start:
{
lean_object* v___x_5775_; lean_object* v___x_5776_; 
v___x_5775_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__3));
v___x_5776_ = l_Lean_stringToMessageData(v___x_5775_);
return v___x_5776_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__6(void){
_start:
{
lean_object* v___x_5778_; lean_object* v___x_5779_; 
v___x_5778_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__5));
v___x_5779_ = l_Lean_stringToMessageData(v___x_5778_);
return v___x_5779_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__8(void){
_start:
{
lean_object* v___x_5781_; lean_object* v___x_5782_; 
v___x_5781_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__7));
v___x_5782_ = l_Lean_stringToMessageData(v___x_5781_);
return v___x_5782_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__10(void){
_start:
{
lean_object* v___x_5784_; lean_object* v___x_5785_; 
v___x_5784_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__9));
v___x_5785_ = l_Lean_stringToMessageData(v___x_5784_);
return v___x_5785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr(lean_object* v_stx_5786_, lean_object* v_expectedType_x3f_5787_, lean_object* v_a_5788_, lean_object* v_a_5789_, lean_object* v_a_5790_, lean_object* v_a_5791_, lean_object* v_a_5792_, lean_object* v_a_5793_){
_start:
{
lean_object* v___x_5795_; uint8_t v___x_5796_; 
v___x_5795_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_termCongr___closed__1));
lean_inc(v_stx_5786_);
v___x_5796_ = l_Lean_Syntax_isOfKind(v_stx_5786_, v___x_5795_);
if (v___x_5796_ == 0)
{
lean_object* v___x_5797_; 
lean_dec(v_expectedType_x3f_5787_);
lean_dec(v_stx_5786_);
v___x_5797_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_TermCongr_elabCHoleExpand_spec__0___redArg();
return v___x_5797_;
}
else
{
lean_object* v___x_5798_; lean_object* v_mctx_5799_; lean_object* v_mvarCounter_5800_; lean_object* v___x_5801_; lean_object* v___x_5802_; lean_object* v_t_5803_; lean_object* v___y_5805_; lean_object* v___y_5806_; lean_object* v___y_5807_; lean_object* v___y_5808_; lean_object* v___y_5809_; lean_object* v___y_5810_; 
v___x_5798_ = lean_st_ref_get(v_a_5791_);
v_mctx_5799_ = lean_ctor_get(v___x_5798_, 0);
lean_inc_ref(v_mctx_5799_);
lean_dec(v___x_5798_);
v_mvarCounter_5800_ = lean_ctor_get(v_mctx_5799_, 3);
lean_inc(v_mvarCounter_5800_);
lean_dec_ref(v_mctx_5799_);
v___x_5801_ = lean_unsigned_to_nat(0u);
v___x_5802_ = lean_unsigned_to_nat(1u);
v_t_5803_ = l_Lean_Syntax_getArg(v_stx_5786_, v___x_5802_);
lean_dec(v_stx_5786_);
if (lean_obj_tag(v_expectedType_x3f_5787_) == 1)
{
lean_object* v_val_5844_; lean_object* v___x_5846_; uint8_t v_isShared_5847_; uint8_t v_isSharedCheck_6014_; 
v_val_5844_ = lean_ctor_get(v_expectedType_x3f_5787_, 0);
v_isSharedCheck_6014_ = !lean_is_exclusive(v_expectedType_x3f_5787_);
if (v_isSharedCheck_6014_ == 0)
{
v___x_5846_ = v_expectedType_x3f_5787_;
v_isShared_5847_ = v_isSharedCheck_6014_;
goto v_resetjp_5845_;
}
else
{
lean_inc(v_val_5844_);
lean_dec(v_expectedType_x3f_5787_);
v___x_5846_ = lean_box(0);
v_isShared_5847_ = v_isSharedCheck_6014_;
goto v_resetjp_5845_;
}
v_resetjp_5845_:
{
lean_object* v___y_5849_; lean_object* v___y_5850_; lean_object* v___y_5851_; lean_object* v___y_5852_; lean_object* v___y_5853_; lean_object* v___y_5854_; lean_object* v___y_5855_; lean_object* v___y_5856_; lean_object* v___x_5869_; 
lean_inc(v_a_5793_);
lean_inc_ref(v_a_5792_);
lean_inc(v_a_5791_);
lean_inc_ref(v_a_5790_);
lean_inc(v_val_5844_);
v___x_5869_ = lean_whnf(v_val_5844_, v_a_5790_, v_a_5791_, v_a_5792_, v_a_5793_);
if (lean_obj_tag(v___x_5869_) == 0)
{
lean_object* v_a_5870_; lean_object* v___x_5871_; 
v_a_5870_ = lean_ctor_get(v___x_5869_, 0);
lean_inc(v_a_5870_);
lean_dec_ref_known(v___x_5869_, 1);
v___x_5871_ = lp_mathlib_Lean_Expr_sides_x3f(v_a_5870_);
lean_dec(v_a_5870_);
if (lean_obj_tag(v___x_5871_) == 1)
{
lean_object* v_val_5872_; lean_object* v___x_5874_; uint8_t v_isShared_5875_; uint8_t v_isSharedCheck_6013_; 
v_val_5872_ = lean_ctor_get(v___x_5871_, 0);
v_isSharedCheck_6013_ = !lean_is_exclusive(v___x_5871_);
if (v_isSharedCheck_6013_ == 0)
{
v___x_5874_ = v___x_5871_;
v_isShared_5875_ = v_isSharedCheck_6013_;
goto v_resetjp_5873_;
}
else
{
lean_inc(v_val_5872_);
lean_dec(v___x_5871_);
v___x_5874_ = lean_box(0);
v_isShared_5875_ = v_isSharedCheck_6013_;
goto v_resetjp_5873_;
}
v_resetjp_5873_:
{
lean_object* v_snd_5876_; lean_object* v_snd_5877_; lean_object* v_fst_5878_; lean_object* v___x_5880_; uint8_t v_isShared_5881_; uint8_t v_isSharedCheck_6011_; 
v_snd_5876_ = lean_ctor_get(v_val_5872_, 1);
lean_inc(v_snd_5876_);
v_snd_5877_ = lean_ctor_get(v_snd_5876_, 1);
lean_inc(v_snd_5877_);
v_fst_5878_ = lean_ctor_get(v_val_5872_, 0);
v_isSharedCheck_6011_ = !lean_is_exclusive(v_val_5872_);
if (v_isSharedCheck_6011_ == 0)
{
lean_object* v_unused_6012_; 
v_unused_6012_ = lean_ctor_get(v_val_5872_, 1);
lean_dec(v_unused_6012_);
v___x_5880_ = v_val_5872_;
v_isShared_5881_ = v_isSharedCheck_6011_;
goto v_resetjp_5879_;
}
else
{
lean_inc(v_fst_5878_);
lean_dec(v_val_5872_);
v___x_5880_ = lean_box(0);
v_isShared_5881_ = v_isSharedCheck_6011_;
goto v_resetjp_5879_;
}
v_resetjp_5879_:
{
lean_object* v_fst_5882_; lean_object* v___x_5884_; uint8_t v_isShared_5885_; uint8_t v_isSharedCheck_6009_; 
v_fst_5882_ = lean_ctor_get(v_snd_5876_, 0);
v_isSharedCheck_6009_ = !lean_is_exclusive(v_snd_5876_);
if (v_isSharedCheck_6009_ == 0)
{
lean_object* v_unused_6010_; 
v_unused_6010_ = lean_ctor_get(v_snd_5876_, 1);
lean_dec(v_unused_6010_);
v___x_5884_ = v_snd_5876_;
v_isShared_5885_ = v_isSharedCheck_6009_;
goto v_resetjp_5883_;
}
else
{
lean_inc(v_fst_5882_);
lean_dec(v_snd_5876_);
v___x_5884_ = lean_box(0);
v_isShared_5885_ = v_isSharedCheck_6009_;
goto v_resetjp_5883_;
}
v_resetjp_5883_:
{
lean_object* v_fst_5886_; lean_object* v_snd_5887_; lean_object* v___x_5889_; uint8_t v_isShared_5890_; uint8_t v_isSharedCheck_6008_; 
v_fst_5886_ = lean_ctor_get(v_snd_5877_, 0);
v_snd_5887_ = lean_ctor_get(v_snd_5877_, 1);
v_isSharedCheck_6008_ = !lean_is_exclusive(v_snd_5877_);
if (v_isSharedCheck_6008_ == 0)
{
v___x_5889_ = v_snd_5877_;
v_isShared_5890_ = v_isSharedCheck_6008_;
goto v_resetjp_5888_;
}
else
{
lean_inc(v_snd_5887_);
lean_inc(v_fst_5886_);
lean_dec(v_snd_5877_);
v___x_5889_ = lean_box(0);
v_isShared_5890_ = v_isSharedCheck_6008_;
goto v_resetjp_5888_;
}
v_resetjp_5888_:
{
lean_object* v___x_5892_; 
if (v_isShared_5875_ == 0)
{
lean_ctor_set(v___x_5874_, 0, v_fst_5878_);
v___x_5892_ = v___x_5874_;
goto v_reusejp_5891_;
}
else
{
lean_object* v_reuseFailAlloc_6007_; 
v_reuseFailAlloc_6007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6007_, 0, v_fst_5878_);
v___x_5892_ = v_reuseFailAlloc_6007_;
goto v_reusejp_5891_;
}
v_reusejp_5891_:
{
lean_object* v___x_5893_; 
lean_inc(v_t_5803_);
v___x_5893_ = lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern(v_t_5803_, v___x_5892_, v___x_5796_, v_a_5788_, v_a_5789_, v_a_5790_, v_a_5791_, v_a_5792_, v_a_5793_);
if (lean_obj_tag(v___x_5893_) == 0)
{
lean_object* v_a_5894_; lean_object* v___x_5896_; 
v_a_5894_ = lean_ctor_get(v___x_5893_, 0);
lean_inc(v_a_5894_);
lean_dec_ref_known(v___x_5893_, 1);
if (v_isShared_5847_ == 0)
{
lean_ctor_set(v___x_5846_, 0, v_fst_5886_);
v___x_5896_ = v___x_5846_;
goto v_reusejp_5895_;
}
else
{
lean_object* v_reuseFailAlloc_6006_; 
v_reuseFailAlloc_6006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6006_, 0, v_fst_5886_);
v___x_5896_ = v_reuseFailAlloc_6006_;
goto v_reusejp_5895_;
}
v_reusejp_5895_:
{
uint8_t v___x_5897_; lean_object* v___x_5898_; 
v___x_5897_ = 0;
v___x_5898_ = lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern(v_t_5803_, v___x_5896_, v___x_5897_, v_a_5788_, v_a_5789_, v_a_5790_, v_a_5791_, v_a_5792_, v_a_5793_);
if (lean_obj_tag(v___x_5898_) == 0)
{
lean_object* v_a_5899_; lean_object* v___y_5901_; lean_object* v___y_5902_; lean_object* v___y_5903_; lean_object* v___y_5904_; lean_object* v___y_5905_; lean_object* v___y_5906_; lean_object* v___y_5936_; lean_object* v___y_5937_; lean_object* v___y_5938_; lean_object* v___y_5939_; lean_object* v___y_5940_; lean_object* v___y_5941_; lean_object* v___x_5977_; 
v_a_5899_ = lean_ctor_get(v___x_5898_, 0);
lean_inc(v_a_5899_);
lean_dec_ref_known(v___x_5898_, 1);
lean_inc(v_a_5894_);
v___x_5977_ = l_Lean_Meta_isExprDefEq(v_fst_5882_, v_a_5894_, v_a_5790_, v_a_5791_, v_a_5792_, v_a_5793_);
if (lean_obj_tag(v___x_5977_) == 0)
{
lean_object* v_a_5978_; uint8_t v___x_5979_; 
v_a_5978_ = lean_ctor_get(v___x_5977_, 0);
lean_inc(v_a_5978_);
lean_dec_ref_known(v___x_5977_, 1);
v___x_5979_ = lean_unbox(v_a_5978_);
lean_dec(v_a_5978_);
if (v___x_5979_ == 0)
{
lean_object* v___x_5980_; lean_object* v___x_5981_; lean_object* v___x_5982_; lean_object* v___x_5983_; lean_object* v___x_5984_; lean_object* v___x_5985_; lean_object* v___x_5986_; lean_object* v___x_5987_; lean_object* v___x_5988_; lean_object* v___x_5989_; lean_object* v_a_5990_; lean_object* v___x_5992_; uint8_t v_isShared_5993_; uint8_t v_isSharedCheck_5997_; 
lean_dec(v_a_5899_);
lean_del_object(v___x_5889_);
lean_dec(v_snd_5887_);
lean_del_object(v___x_5884_);
lean_del_object(v___x_5880_);
lean_dec(v_mvarCounter_5800_);
v___x_5980_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__8, &lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__8);
v___x_5981_ = l_Lean_MessageData_ofExpr(v_a_5894_);
v___x_5982_ = l_Lean_indentD(v___x_5981_);
v___x_5983_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5983_, 0, v___x_5980_);
lean_ctor_set(v___x_5983_, 1, v___x_5982_);
v___x_5984_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__10, &lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__10);
v___x_5985_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5985_, 0, v___x_5983_);
lean_ctor_set(v___x_5985_, 1, v___x_5984_);
v___x_5986_ = l_Lean_MessageData_ofExpr(v_val_5844_);
v___x_5987_ = l_Lean_indentD(v___x_5986_);
v___x_5988_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_5988_, 0, v___x_5985_);
lean_ctor_set(v___x_5988_, 1, v___x_5987_);
v___x_5989_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(v___x_5988_, v_a_5788_, v_a_5789_, v_a_5790_, v_a_5791_, v_a_5792_, v_a_5793_);
v_a_5990_ = lean_ctor_get(v___x_5989_, 0);
v_isSharedCheck_5997_ = !lean_is_exclusive(v___x_5989_);
if (v_isSharedCheck_5997_ == 0)
{
v___x_5992_ = v___x_5989_;
v_isShared_5993_ = v_isSharedCheck_5997_;
goto v_resetjp_5991_;
}
else
{
lean_inc(v_a_5990_);
lean_dec(v___x_5989_);
v___x_5992_ = lean_box(0);
v_isShared_5993_ = v_isSharedCheck_5997_;
goto v_resetjp_5991_;
}
v_resetjp_5991_:
{
lean_object* v___x_5995_; 
if (v_isShared_5993_ == 0)
{
v___x_5995_ = v___x_5992_;
goto v_reusejp_5994_;
}
else
{
lean_object* v_reuseFailAlloc_5996_; 
v_reuseFailAlloc_5996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5996_, 0, v_a_5990_);
v___x_5995_ = v_reuseFailAlloc_5996_;
goto v_reusejp_5994_;
}
v_reusejp_5994_:
{
return v___x_5995_;
}
}
}
else
{
v___y_5936_ = v_a_5788_;
v___y_5937_ = v_a_5789_;
v___y_5938_ = v_a_5790_;
v___y_5939_ = v_a_5791_;
v___y_5940_ = v_a_5792_;
v___y_5941_ = v_a_5793_;
goto v___jp_5935_;
}
}
else
{
lean_object* v_a_5998_; lean_object* v___x_6000_; uint8_t v_isShared_6001_; uint8_t v_isSharedCheck_6005_; 
lean_dec(v_a_5899_);
lean_dec(v_a_5894_);
lean_del_object(v___x_5889_);
lean_dec(v_snd_5887_);
lean_del_object(v___x_5884_);
lean_del_object(v___x_5880_);
lean_dec(v_val_5844_);
lean_dec(v_mvarCounter_5800_);
v_a_5998_ = lean_ctor_get(v___x_5977_, 0);
v_isSharedCheck_6005_ = !lean_is_exclusive(v___x_5977_);
if (v_isSharedCheck_6005_ == 0)
{
v___x_6000_ = v___x_5977_;
v_isShared_6001_ = v_isSharedCheck_6005_;
goto v_resetjp_5999_;
}
else
{
lean_inc(v_a_5998_);
lean_dec(v___x_5977_);
v___x_6000_ = lean_box(0);
v_isShared_6001_ = v_isSharedCheck_6005_;
goto v_resetjp_5999_;
}
v_resetjp_5999_:
{
lean_object* v___x_6003_; 
if (v_isShared_6001_ == 0)
{
v___x_6003_ = v___x_6000_;
goto v_reusejp_6002_;
}
else
{
lean_object* v_reuseFailAlloc_6004_; 
v_reuseFailAlloc_6004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6004_, 0, v_a_5998_);
v___x_6003_ = v_reuseFailAlloc_6004_;
goto v_reusejp_6002_;
}
v_reusejp_6002_:
{
return v___x_6003_;
}
}
}
v___jp_5900_:
{
uint8_t v___x_5907_; lean_object* v___x_5908_; 
v___x_5907_ = 0;
v___x_5908_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_5907_, v___x_5897_, v___y_5901_, v___y_5902_, v___y_5903_, v___y_5904_, v___y_5905_, v___y_5906_);
if (lean_obj_tag(v___x_5908_) == 0)
{
lean_object* v___x_5909_; 
lean_dec_ref_known(v___x_5908_, 1);
v___x_5909_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf(v___x_5801_, v_mvarCounter_5800_, v_a_5894_, v_a_5899_, v___y_5903_, v___y_5904_, v___y_5905_, v___y_5906_);
if (lean_obj_tag(v___x_5909_) == 0)
{
lean_object* v_a_5910_; lean_object* v___x_5911_; 
v_a_5910_ = lean_ctor_get(v___x_5909_, 0);
lean_inc(v_a_5910_);
lean_dec_ref_known(v___x_5909_, 1);
lean_inc(v___y_5906_);
lean_inc_ref(v___y_5905_);
lean_inc(v___y_5904_);
lean_inc_ref(v___y_5903_);
lean_inc(v_val_5844_);
v___x_5911_ = lean_whnf(v_val_5844_, v___y_5903_, v___y_5904_, v___y_5905_, v___y_5906_);
if (lean_obj_tag(v___x_5911_) == 0)
{
lean_object* v_a_5912_; lean_object* v___x_5913_; lean_object* v___x_5914_; uint8_t v___x_5915_; 
v_a_5912_ = lean_ctor_get(v___x_5911_, 0);
lean_inc(v_a_5912_);
lean_dec_ref_known(v___x_5911_, 1);
v___x_5913_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_TermCongr_mkIffForExpectedType___closed__2));
v___x_5914_ = lean_unsigned_to_nat(2u);
v___x_5915_ = l_Lean_Expr_isAppOfArity(v_a_5912_, v___x_5913_, v___x_5914_);
if (v___x_5915_ == 0)
{
v___y_5849_ = v___y_5906_;
v___y_5850_ = v___y_5904_;
v___y_5851_ = v___y_5903_;
v___y_5852_ = v___y_5905_;
v___y_5853_ = v_a_5910_;
v___y_5854_ = v_a_5912_;
v___y_5855_ = v___y_5901_;
v___y_5856_ = v___y_5902_;
goto v___jp_5848_;
}
else
{
if (v___x_5796_ == 0)
{
v___y_5849_ = v___y_5906_;
v___y_5850_ = v___y_5904_;
v___y_5851_ = v___y_5903_;
v___y_5852_ = v___y_5905_;
v___y_5853_ = v_a_5910_;
v___y_5854_ = v_a_5912_;
v___y_5855_ = v___y_5901_;
v___y_5856_ = v___y_5902_;
goto v___jp_5848_;
}
else
{
lean_object* v___x_5916_; 
lean_dec(v_a_5912_);
v___x_5916_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_iff(v_a_5910_, v___y_5903_, v___y_5904_, v___y_5905_, v___y_5906_);
if (lean_obj_tag(v___x_5916_) == 0)
{
lean_object* v_a_5917_; lean_object* v___x_5918_; 
v_a_5917_ = lean_ctor_get(v___x_5916_, 0);
lean_inc(v_a_5917_);
lean_dec_ref_known(v___x_5916_, 1);
v___x_5918_ = l_Lean_Meta_mkExpectedTypeHint(v_a_5917_, v_val_5844_, v___y_5903_, v___y_5904_, v___y_5905_, v___y_5906_);
return v___x_5918_;
}
else
{
lean_dec(v_val_5844_);
return v___x_5916_;
}
}
}
}
else
{
lean_dec(v_a_5910_);
lean_dec(v_val_5844_);
return v___x_5911_;
}
}
else
{
lean_object* v_a_5919_; lean_object* v___x_5921_; uint8_t v_isShared_5922_; uint8_t v_isSharedCheck_5926_; 
lean_dec(v_val_5844_);
v_a_5919_ = lean_ctor_get(v___x_5909_, 0);
v_isSharedCheck_5926_ = !lean_is_exclusive(v___x_5909_);
if (v_isSharedCheck_5926_ == 0)
{
v___x_5921_ = v___x_5909_;
v_isShared_5922_ = v_isSharedCheck_5926_;
goto v_resetjp_5920_;
}
else
{
lean_inc(v_a_5919_);
lean_dec(v___x_5909_);
v___x_5921_ = lean_box(0);
v_isShared_5922_ = v_isSharedCheck_5926_;
goto v_resetjp_5920_;
}
v_resetjp_5920_:
{
lean_object* v___x_5924_; 
if (v_isShared_5922_ == 0)
{
v___x_5924_ = v___x_5921_;
goto v_reusejp_5923_;
}
else
{
lean_object* v_reuseFailAlloc_5925_; 
v_reuseFailAlloc_5925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5925_, 0, v_a_5919_);
v___x_5924_ = v_reuseFailAlloc_5925_;
goto v_reusejp_5923_;
}
v_reusejp_5923_:
{
return v___x_5924_;
}
}
}
}
else
{
lean_object* v_a_5927_; lean_object* v___x_5929_; uint8_t v_isShared_5930_; uint8_t v_isSharedCheck_5934_; 
lean_dec(v_a_5899_);
lean_dec(v_a_5894_);
lean_dec(v_val_5844_);
lean_dec(v_mvarCounter_5800_);
v_a_5927_ = lean_ctor_get(v___x_5908_, 0);
v_isSharedCheck_5934_ = !lean_is_exclusive(v___x_5908_);
if (v_isSharedCheck_5934_ == 0)
{
v___x_5929_ = v___x_5908_;
v_isShared_5930_ = v_isSharedCheck_5934_;
goto v_resetjp_5928_;
}
else
{
lean_inc(v_a_5927_);
lean_dec(v___x_5908_);
v___x_5929_ = lean_box(0);
v_isShared_5930_ = v_isSharedCheck_5934_;
goto v_resetjp_5928_;
}
v_resetjp_5928_:
{
lean_object* v___x_5932_; 
if (v_isShared_5930_ == 0)
{
v___x_5932_ = v___x_5929_;
goto v_reusejp_5931_;
}
else
{
lean_object* v_reuseFailAlloc_5933_; 
v_reuseFailAlloc_5933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5933_, 0, v_a_5927_);
v___x_5932_ = v_reuseFailAlloc_5933_;
goto v_reusejp_5931_;
}
v_reusejp_5931_:
{
return v___x_5932_;
}
}
}
}
v___jp_5935_:
{
lean_object* v___x_5942_; 
lean_inc(v_a_5899_);
v___x_5942_ = l_Lean_Meta_isExprDefEq(v_snd_5887_, v_a_5899_, v___y_5938_, v___y_5939_, v___y_5940_, v___y_5941_);
if (lean_obj_tag(v___x_5942_) == 0)
{
lean_object* v_a_5943_; uint8_t v___x_5944_; 
v_a_5943_ = lean_ctor_get(v___x_5942_, 0);
lean_inc(v_a_5943_);
lean_dec_ref_known(v___x_5942_, 1);
v___x_5944_ = lean_unbox(v_a_5943_);
lean_dec(v_a_5943_);
if (v___x_5944_ == 0)
{
lean_object* v___x_5945_; lean_object* v___x_5946_; lean_object* v___x_5947_; lean_object* v___x_5949_; 
lean_dec(v_a_5894_);
lean_dec(v_mvarCounter_5800_);
v___x_5945_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__4, &lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__4);
v___x_5946_ = l_Lean_MessageData_ofExpr(v_a_5899_);
v___x_5947_ = l_Lean_indentD(v___x_5946_);
if (v_isShared_5890_ == 0)
{
lean_ctor_set_tag(v___x_5889_, 7);
lean_ctor_set(v___x_5889_, 1, v___x_5947_);
lean_ctor_set(v___x_5889_, 0, v___x_5945_);
v___x_5949_ = v___x_5889_;
goto v_reusejp_5948_;
}
else
{
lean_object* v_reuseFailAlloc_5968_; 
v_reuseFailAlloc_5968_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5968_, 0, v___x_5945_);
lean_ctor_set(v_reuseFailAlloc_5968_, 1, v___x_5947_);
v___x_5949_ = v_reuseFailAlloc_5968_;
goto v_reusejp_5948_;
}
v_reusejp_5948_:
{
lean_object* v___x_5950_; lean_object* v___x_5952_; 
v___x_5950_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__6, &lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__6);
if (v_isShared_5885_ == 0)
{
lean_ctor_set_tag(v___x_5884_, 7);
lean_ctor_set(v___x_5884_, 1, v___x_5950_);
lean_ctor_set(v___x_5884_, 0, v___x_5949_);
v___x_5952_ = v___x_5884_;
goto v_reusejp_5951_;
}
else
{
lean_object* v_reuseFailAlloc_5967_; 
v_reuseFailAlloc_5967_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5967_, 0, v___x_5949_);
lean_ctor_set(v_reuseFailAlloc_5967_, 1, v___x_5950_);
v___x_5952_ = v_reuseFailAlloc_5967_;
goto v_reusejp_5951_;
}
v_reusejp_5951_:
{
lean_object* v___x_5953_; lean_object* v___x_5954_; lean_object* v___x_5956_; 
v___x_5953_ = l_Lean_MessageData_ofExpr(v_val_5844_);
v___x_5954_ = l_Lean_indentD(v___x_5953_);
if (v_isShared_5881_ == 0)
{
lean_ctor_set_tag(v___x_5880_, 7);
lean_ctor_set(v___x_5880_, 1, v___x_5954_);
lean_ctor_set(v___x_5880_, 0, v___x_5952_);
v___x_5956_ = v___x_5880_;
goto v_reusejp_5955_;
}
else
{
lean_object* v_reuseFailAlloc_5966_; 
v_reuseFailAlloc_5966_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5966_, 0, v___x_5952_);
lean_ctor_set(v_reuseFailAlloc_5966_, 1, v___x_5954_);
v___x_5956_ = v_reuseFailAlloc_5966_;
goto v_reusejp_5955_;
}
v_reusejp_5955_:
{
lean_object* v___x_5957_; lean_object* v_a_5958_; lean_object* v___x_5960_; uint8_t v_isShared_5961_; uint8_t v_isSharedCheck_5965_; 
v___x_5957_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_TermCongr_elabCHole_spec__0___redArg(v___x_5956_, v___y_5936_, v___y_5937_, v___y_5938_, v___y_5939_, v___y_5940_, v___y_5941_);
v_a_5958_ = lean_ctor_get(v___x_5957_, 0);
v_isSharedCheck_5965_ = !lean_is_exclusive(v___x_5957_);
if (v_isSharedCheck_5965_ == 0)
{
v___x_5960_ = v___x_5957_;
v_isShared_5961_ = v_isSharedCheck_5965_;
goto v_resetjp_5959_;
}
else
{
lean_inc(v_a_5958_);
lean_dec(v___x_5957_);
v___x_5960_ = lean_box(0);
v_isShared_5961_ = v_isSharedCheck_5965_;
goto v_resetjp_5959_;
}
v_resetjp_5959_:
{
lean_object* v___x_5963_; 
if (v_isShared_5961_ == 0)
{
v___x_5963_ = v___x_5960_;
goto v_reusejp_5962_;
}
else
{
lean_object* v_reuseFailAlloc_5964_; 
v_reuseFailAlloc_5964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5964_, 0, v_a_5958_);
v___x_5963_ = v_reuseFailAlloc_5964_;
goto v_reusejp_5962_;
}
v_reusejp_5962_:
{
return v___x_5963_;
}
}
}
}
}
}
else
{
lean_del_object(v___x_5889_);
lean_del_object(v___x_5884_);
lean_del_object(v___x_5880_);
v___y_5901_ = v___y_5936_;
v___y_5902_ = v___y_5937_;
v___y_5903_ = v___y_5938_;
v___y_5904_ = v___y_5939_;
v___y_5905_ = v___y_5940_;
v___y_5906_ = v___y_5941_;
goto v___jp_5900_;
}
}
else
{
lean_object* v_a_5969_; lean_object* v___x_5971_; uint8_t v_isShared_5972_; uint8_t v_isSharedCheck_5976_; 
lean_dec(v_a_5899_);
lean_dec(v_a_5894_);
lean_del_object(v___x_5889_);
lean_del_object(v___x_5884_);
lean_del_object(v___x_5880_);
lean_dec(v_val_5844_);
lean_dec(v_mvarCounter_5800_);
v_a_5969_ = lean_ctor_get(v___x_5942_, 0);
v_isSharedCheck_5976_ = !lean_is_exclusive(v___x_5942_);
if (v_isSharedCheck_5976_ == 0)
{
v___x_5971_ = v___x_5942_;
v_isShared_5972_ = v_isSharedCheck_5976_;
goto v_resetjp_5970_;
}
else
{
lean_inc(v_a_5969_);
lean_dec(v___x_5942_);
v___x_5971_ = lean_box(0);
v_isShared_5972_ = v_isSharedCheck_5976_;
goto v_resetjp_5970_;
}
v_resetjp_5970_:
{
lean_object* v___x_5974_; 
if (v_isShared_5972_ == 0)
{
v___x_5974_ = v___x_5971_;
goto v_reusejp_5973_;
}
else
{
lean_object* v_reuseFailAlloc_5975_; 
v_reuseFailAlloc_5975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5975_, 0, v_a_5969_);
v___x_5974_ = v_reuseFailAlloc_5975_;
goto v_reusejp_5973_;
}
v_reusejp_5973_:
{
return v___x_5974_;
}
}
}
}
}
else
{
lean_dec(v_a_5894_);
lean_del_object(v___x_5889_);
lean_dec(v_snd_5887_);
lean_del_object(v___x_5884_);
lean_dec(v_fst_5882_);
lean_del_object(v___x_5880_);
lean_dec(v_val_5844_);
lean_dec(v_mvarCounter_5800_);
return v___x_5898_;
}
}
}
else
{
lean_del_object(v___x_5889_);
lean_dec(v_snd_5887_);
lean_dec(v_fst_5886_);
lean_del_object(v___x_5884_);
lean_dec(v_fst_5882_);
lean_del_object(v___x_5880_);
lean_del_object(v___x_5846_);
lean_dec(v_val_5844_);
lean_dec(v_t_5803_);
lean_dec(v_mvarCounter_5800_);
return v___x_5893_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_5871_);
lean_del_object(v___x_5846_);
lean_dec(v_val_5844_);
v___y_5805_ = v_a_5788_;
v___y_5806_ = v_a_5789_;
v___y_5807_ = v_a_5790_;
v___y_5808_ = v_a_5791_;
v___y_5809_ = v_a_5792_;
v___y_5810_ = v_a_5793_;
goto v___jp_5804_;
}
}
else
{
lean_del_object(v___x_5846_);
lean_dec(v_val_5844_);
lean_dec(v_t_5803_);
lean_dec(v_mvarCounter_5800_);
return v___x_5869_;
}
v___jp_5848_:
{
uint8_t v___x_5857_; 
v___x_5857_ = l_Lean_Expr_isEq(v___y_5854_);
if (v___x_5857_ == 0)
{
uint8_t v___x_5858_; 
v___x_5858_ = l_Lean_Expr_isHEq(v___y_5854_);
lean_dec_ref(v___y_5854_);
if (v___x_5858_ == 0)
{
lean_object* v___x_5859_; lean_object* v___x_5860_; 
lean_dec_ref(v___y_5853_);
v___x_5859_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__2, &lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___closed__2);
v___x_5860_ = lp_mathlib_panic___at___00Mathlib_Tactic_TermCongr_elabTermCongr_spec__0(v___x_5859_, v___y_5855_, v___y_5856_, v___y_5851_, v___y_5850_, v___y_5852_, v___y_5849_);
if (lean_obj_tag(v___x_5860_) == 0)
{
lean_object* v_a_5861_; lean_object* v___x_5862_; 
v_a_5861_ = lean_ctor_get(v___x_5860_, 0);
lean_inc(v_a_5861_);
lean_dec_ref_known(v___x_5860_, 1);
v___x_5862_ = l_Lean_Meta_mkExpectedTypeHint(v_a_5861_, v_val_5844_, v___y_5851_, v___y_5850_, v___y_5852_, v___y_5849_);
return v___x_5862_;
}
else
{
lean_dec(v_val_5844_);
return v___x_5860_;
}
}
else
{
lean_object* v___x_5863_; 
v___x_5863_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_heq(v___y_5853_, v___y_5851_, v___y_5850_, v___y_5852_, v___y_5849_);
if (lean_obj_tag(v___x_5863_) == 0)
{
lean_object* v_a_5864_; lean_object* v___x_5865_; 
v_a_5864_ = lean_ctor_get(v___x_5863_, 0);
lean_inc(v_a_5864_);
lean_dec_ref_known(v___x_5863_, 1);
v___x_5865_ = l_Lean_Meta_mkExpectedTypeHint(v_a_5864_, v_val_5844_, v___y_5851_, v___y_5850_, v___y_5852_, v___y_5849_);
return v___x_5865_;
}
else
{
lean_dec(v_val_5844_);
return v___x_5863_;
}
}
}
else
{
lean_object* v___x_5866_; 
lean_dec_ref(v___y_5854_);
v___x_5866_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v___y_5853_, v___y_5851_, v___y_5850_, v___y_5852_, v___y_5849_);
if (lean_obj_tag(v___x_5866_) == 0)
{
lean_object* v_a_5867_; lean_object* v___x_5868_; 
v_a_5867_ = lean_ctor_get(v___x_5866_, 0);
lean_inc(v_a_5867_);
lean_dec_ref_known(v___x_5866_, 1);
v___x_5868_ = l_Lean_Meta_mkExpectedTypeHint(v_a_5867_, v_val_5844_, v___y_5851_, v___y_5850_, v___y_5852_, v___y_5849_);
return v___x_5868_;
}
else
{
lean_dec(v_val_5844_);
return v___x_5866_;
}
}
}
}
}
else
{
lean_dec(v_expectedType_x3f_5787_);
v___y_5805_ = v_a_5788_;
v___y_5806_ = v_a_5789_;
v___y_5807_ = v_a_5790_;
v___y_5808_ = v_a_5791_;
v___y_5809_ = v_a_5792_;
v___y_5810_ = v_a_5793_;
goto v___jp_5804_;
}
v___jp_5804_:
{
lean_object* v___x_5811_; lean_object* v___x_5812_; 
v___x_5811_ = lean_box(0);
lean_inc(v_t_5803_);
v___x_5812_ = lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern(v_t_5803_, v___x_5811_, v___x_5796_, v___y_5805_, v___y_5806_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_);
if (lean_obj_tag(v___x_5812_) == 0)
{
lean_object* v_a_5813_; uint8_t v___x_5814_; lean_object* v___x_5815_; 
v_a_5813_ = lean_ctor_get(v___x_5812_, 0);
lean_inc(v_a_5813_);
lean_dec_ref_known(v___x_5812_, 1);
v___x_5814_ = 0;
v___x_5815_ = lp_mathlib_Mathlib_Tactic_TermCongr_elaboratePattern(v_t_5803_, v___x_5811_, v___x_5814_, v___y_5805_, v___y_5806_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_);
if (lean_obj_tag(v___x_5815_) == 0)
{
lean_object* v_a_5816_; uint8_t v___x_5817_; lean_object* v___x_5818_; 
v_a_5816_ = lean_ctor_get(v___x_5815_, 0);
lean_inc(v_a_5816_);
lean_dec_ref_known(v___x_5815_, 1);
v___x_5817_ = 0;
v___x_5818_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_5817_, v___x_5814_, v___y_5805_, v___y_5806_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_);
if (lean_obj_tag(v___x_5818_) == 0)
{
lean_object* v___x_5819_; 
lean_dec_ref_known(v___x_5818_, 1);
v___x_5819_ = lp_mathlib_Mathlib_Tactic_TermCongr_mkCongrOf(v___x_5801_, v_mvarCounter_5800_, v_a_5813_, v_a_5816_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_);
if (lean_obj_tag(v___x_5819_) == 0)
{
lean_object* v_a_5820_; lean_object* v___x_5821_; 
v_a_5820_ = lean_ctor_get(v___x_5819_, 0);
lean_inc_n(v_a_5820_, 2);
lean_dec_ref_known(v___x_5819_, 1);
v___x_5821_ = lp_mathlib_Mathlib_Tactic_TermCongr_CongrResult_eq(v_a_5820_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_);
if (lean_obj_tag(v___x_5821_) == 0)
{
lean_object* v_a_5822_; lean_object* v_lhs_5823_; lean_object* v_rhs_5824_; lean_object* v___x_5825_; 
v_a_5822_ = lean_ctor_get(v___x_5821_, 0);
lean_inc(v_a_5822_);
lean_dec_ref_known(v___x_5821_, 1);
v_lhs_5823_ = lean_ctor_get(v_a_5820_, 0);
lean_inc_ref(v_lhs_5823_);
v_rhs_5824_ = lean_ctor_get(v_a_5820_, 1);
lean_inc_ref(v_rhs_5824_);
lean_dec(v_a_5820_);
v___x_5825_ = l_Lean_Meta_mkEq(v_lhs_5823_, v_rhs_5824_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_);
if (lean_obj_tag(v___x_5825_) == 0)
{
lean_object* v_a_5826_; lean_object* v___x_5827_; 
v_a_5826_ = lean_ctor_get(v___x_5825_, 0);
lean_inc(v_a_5826_);
lean_dec_ref_known(v___x_5825_, 1);
v___x_5827_ = l_Lean_Meta_mkExpectedTypeHint(v_a_5822_, v_a_5826_, v___y_5807_, v___y_5808_, v___y_5809_, v___y_5810_);
return v___x_5827_;
}
else
{
lean_dec(v_a_5822_);
return v___x_5825_;
}
}
else
{
lean_dec(v_a_5820_);
return v___x_5821_;
}
}
else
{
lean_object* v_a_5828_; lean_object* v___x_5830_; uint8_t v_isShared_5831_; uint8_t v_isSharedCheck_5835_; 
v_a_5828_ = lean_ctor_get(v___x_5819_, 0);
v_isSharedCheck_5835_ = !lean_is_exclusive(v___x_5819_);
if (v_isSharedCheck_5835_ == 0)
{
v___x_5830_ = v___x_5819_;
v_isShared_5831_ = v_isSharedCheck_5835_;
goto v_resetjp_5829_;
}
else
{
lean_inc(v_a_5828_);
lean_dec(v___x_5819_);
v___x_5830_ = lean_box(0);
v_isShared_5831_ = v_isSharedCheck_5835_;
goto v_resetjp_5829_;
}
v_resetjp_5829_:
{
lean_object* v___x_5833_; 
if (v_isShared_5831_ == 0)
{
v___x_5833_ = v___x_5830_;
goto v_reusejp_5832_;
}
else
{
lean_object* v_reuseFailAlloc_5834_; 
v_reuseFailAlloc_5834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5834_, 0, v_a_5828_);
v___x_5833_ = v_reuseFailAlloc_5834_;
goto v_reusejp_5832_;
}
v_reusejp_5832_:
{
return v___x_5833_;
}
}
}
}
else
{
lean_object* v_a_5836_; lean_object* v___x_5838_; uint8_t v_isShared_5839_; uint8_t v_isSharedCheck_5843_; 
lean_dec(v_a_5816_);
lean_dec(v_a_5813_);
lean_dec(v_mvarCounter_5800_);
v_a_5836_ = lean_ctor_get(v___x_5818_, 0);
v_isSharedCheck_5843_ = !lean_is_exclusive(v___x_5818_);
if (v_isSharedCheck_5843_ == 0)
{
v___x_5838_ = v___x_5818_;
v_isShared_5839_ = v_isSharedCheck_5843_;
goto v_resetjp_5837_;
}
else
{
lean_inc(v_a_5836_);
lean_dec(v___x_5818_);
v___x_5838_ = lean_box(0);
v_isShared_5839_ = v_isSharedCheck_5843_;
goto v_resetjp_5837_;
}
v_resetjp_5837_:
{
lean_object* v___x_5841_; 
if (v_isShared_5839_ == 0)
{
v___x_5841_ = v___x_5838_;
goto v_reusejp_5840_;
}
else
{
lean_object* v_reuseFailAlloc_5842_; 
v_reuseFailAlloc_5842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5842_, 0, v_a_5836_);
v___x_5841_ = v_reuseFailAlloc_5842_;
goto v_reusejp_5840_;
}
v_reusejp_5840_:
{
return v___x_5841_;
}
}
}
}
else
{
lean_dec(v_a_5813_);
lean_dec(v_mvarCounter_5800_);
return v___x_5815_;
}
}
else
{
lean_dec(v_t_5803_);
lean_dec(v_mvarCounter_5800_);
return v___x_5812_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr___boxed(lean_object* v_stx_6015_, lean_object* v_expectedType_x3f_6016_, lean_object* v_a_6017_, lean_object* v_a_6018_, lean_object* v_a_6019_, lean_object* v_a_6020_, lean_object* v_a_6021_, lean_object* v_a_6022_, lean_object* v_a_6023_){
_start:
{
lean_object* v_res_6024_; 
v_res_6024_ = lp_mathlib_Mathlib_Tactic_TermCongr_elabTermCongr(v_stx_6015_, v_expectedType_x3f_6016_, v_a_6017_, v_a_6018_, v_a_6019_, v_a_6020_, v_a_6021_, v_a_6022_);
lean_dec(v_a_6022_);
lean_dec_ref(v_a_6021_);
lean_dec(v_a_6020_);
lean_dec_ref(v_a_6019_);
lean_dec(v_a_6018_);
lean_dec_ref(v_a_6017_);
return v_res_6024_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_TermCongr_0__Mathlib_Tactic_TermCongr_initFn_00___x40_Mathlib_Tactic_TermCongr_920020848____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
}
#ifdef __cplusplus
}
#endif
