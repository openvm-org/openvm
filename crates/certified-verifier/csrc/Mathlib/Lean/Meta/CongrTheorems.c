// Lean compiler output
// Module: Mathlib.Lean.Meta.CongrTheorems
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Refl public import Mathlib.Basic.IsEmpty.Defs public import Lean.Meta.CongrTheorems public meta import Mathlib.Basic.IsEmpty.Defs
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
lean_object* l_Lean_Expr_replaceFVars(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_MVarId_proofIrrelHeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_heqOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallTelescopeReducing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_inferType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withNewLocalInstances___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withLocalDeclsD___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_hrefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Tactic_Cleanup_0__Lean_Meta_cleanupCore(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_intros(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_substEqs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_refl(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Meta_instReprCongrArgKind_repr(uint8_t, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Std_Format_fill(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqHEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkHEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedParamInfo_default;
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_name_append_index_after(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkHCongrWithArity(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_abstract(lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_clear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
uint8_t l_Lean_Expr_isEq(lean_object*);
uint8_t l_Lean_Expr_isHEq(lean_object*);
lean_object* l_Lean_MVarId_casesRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_subsingletonElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingName_x21(lean_object*);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
lean_object* lean_name_append_after(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isForall(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Meta_FunInfo_getArity(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__1_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "CongrTheorems"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__1_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__1_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(211, 174, 49, 251, 64, 24, 251, 1)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__1_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(134, 187, 99, 157, 92, 226, 58, 150)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__3_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__3_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__3_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__4_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__3_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__4_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__4_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__5_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__5_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__5_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__6_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__4_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__5_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__6_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__6_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__8_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__6_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(107, 51, 169, 24, 39, 49, 110, 91)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__8_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__8_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__9_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__8_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(99, 9, 190, 9, 238, 15, 137, 228)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__9_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__9_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__10_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__9_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__1_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(182, 100, 18, 61, 81, 84, 159, 187)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__10_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__10_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__11_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__10_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(95, 0, 217, 117, 152, 24, 198, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__11_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__11_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__12_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__11_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(18, 5, 94, 85, 55, 196, 39, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__12_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__12_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__13_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__12_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(6, 57, 170, 154, 53, 62, 117, 23)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__13_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__13_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__14_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__14_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__14_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__15_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__13_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__14_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 229, 179, 77, 107, 65, 252, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__15_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__15_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__16_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__16_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__16_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__17_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__15_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__16_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(14, 148, 247, 24, 149, 2, 76, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__17_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__17_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__18_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__17_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__5_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(135, 130, 14, 79, 153, 239, 158, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__18_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__18_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__19_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__18_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(250, 96, 23, 51, 41, 234, 157, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__19_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__19_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__20_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__19_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(254, 213, 125, 136, 177, 114, 184, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__20_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__20_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__21_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__20_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__1_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(127, 152, 30, 72, 154, 190, 4, 52)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__21_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__21_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__22_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__22_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__23_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__23_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__23_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__24_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__24_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__25_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__25_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__25_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__26_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__26_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__27_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__27_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Mathlib.Lean.Meta.CongrTheorems"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "_private.Mathlib.Lean.Meta.CongrTheorems.0.Lean.Meta.mkHCongrWithArity'.process"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Unexpected CongrArgKind"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Meta_mkHCongrWithArity_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_mkHCongrWithArity_x27___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_mkHCongrWithArity_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkHCongrWithArity_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkHCongrWithArity_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "IsEmpty"};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(44, 14, 182, 92, 174, 89, 100, 107)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "inst"};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(170, 188, 240, 205, 110, 63, 170, 91)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "FastSubsingleton"};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(3, 43, 17, 5, 235, 183, 145, 18)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__1_value),LEAN_SCALAR_PTR_LITERAL(183, 54, 246, 109, 48, 150, 71, 175)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "FastIsEmpty"};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(207, 186, 71, 204, 12, 48, 62, 72)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__1_value),LEAN_SCALAR_PTR_LITERAL(35, 37, 165, 75, 182, 158, 45, 108)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Subsingleton"};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 130, 42, 228, 248, 162, 23, 186)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__1;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__2_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__3_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__4_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__6_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__7 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__7_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_inferType___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__8_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__9 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__9_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__10_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__11 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__11_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__12_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__13 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__13_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__14_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__15 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__9_value),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__16_value),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__11_value),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__12_value),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__13_value),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__17 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__17_value),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__15_value)}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__18_value;
static const lean_array_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__19 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__19_value;
static const lean_closure_object lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__20 = (const lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "elim"};
static const lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__7_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__0_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(3, 43, 17, 5, 235, 183, 145, 18)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(83, 127, 17, 151, 251, 235, 223, 76)}};
static const lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "fastSubsingletonElim"};
static const lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__0_value),LEAN_SCALAR_PTR_LITERAL(188, 134, 184, 24, 238, 162, 119, 53)}};
static const lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__0___boxed(lean_object**);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "doubleTelescope: function doesn't have enough parameters"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "e'"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 5, 27, 128, 192, 63, 73, 43)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "e"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 154, 90, 102, 217, 192, 49, 255)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___boxed__const__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "was not able to solve for proof"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "trySolve success!"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "trySolve "};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4_spec__5_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "#["};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__0 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__0_value;
static const lean_string_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__1 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__1_value;
static const lean_ctor_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__1_value)}};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__2 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__2_value;
static const lean_ctor_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__2_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__3 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__3_value;
static const lean_string_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__4 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__4_value;
static lean_once_cell_t lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__5;
static lean_once_cell_t lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__6;
static const lean_ctor_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__0_value)}};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__7 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__7_value;
static const lean_ctor_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__4_value)}};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__8 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__8_value;
static const lean_string_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "#[]"};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__9 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__9_value;
static const lean_ctor_object lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__9_value)}};
static const lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__10 = (const lean_object*)&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "rich congrType: "};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__1;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "CongrArgKinds: "};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__3;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Internal error when constructing congruence lemma proof"};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__5;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "simple congrType: "};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__3___boxed(lean_object**);
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "f'"};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(176, 166, 137, 10, 240, 99, 97, 180)}};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___boxed(lean_object**);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__0_value;
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__1 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__6(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "computed fixedParams="};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__3;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ys = "};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__5;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "xs = "};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_mkRichHCongr_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_mkRichHCongr_spec__7___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_mkRichHCongr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_mkRichHCongr___lam__0___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = ", fixedParams="};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__2;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "fixedFun="};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__4;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "deps: "};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__6;
static const lean_string_object lp_mathlib_Lean_Meta_mkRichHCongr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ftype: "};
static const lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__7 = (const lean_object*)&lp_mathlib_Lean_Meta_mkRichHCongr___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_mkRichHCongr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__5(lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__22_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lean_unsigned_to_nat(3122272532u);
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__21_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_));
v___x_55_ = l_Lean_Name_num___override(v___x_54_, v___x_53_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__24_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__23_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_));
v___x_58_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__22_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__22_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__22_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_);
v___x_59_ = l_Lean_Name_str___override(v___x_58_, v___x_57_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__26_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__25_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_));
v___x_62_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__24_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__24_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__24_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_);
v___x_63_ = l_Lean_Name_str___override(v___x_62_, v___x_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__27_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_64_ = lean_unsigned_to_nat(2u);
v___x_65_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__26_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__26_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__26_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_);
v___x_66_ = l_Lean_Name_num___override(v___x_65_, v___x_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_68_; uint8_t v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_68_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_));
v___x_69_ = 0;
v___x_70_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__27_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__27_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__27_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_);
v___x_71_ = l_Lean_registerTraceClass(v___x_68_, v___x_69_, v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2____boxed(lean_object* v_a_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_();
return v_res_73_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1_spec__2(lean_object* v_a_74_, lean_object* v_as_75_, size_t v_i_76_, size_t v_stop_77_){
_start:
{
uint8_t v___x_78_; 
v___x_78_ = lean_usize_dec_eq(v_i_76_, v_stop_77_);
if (v___x_78_ == 0)
{
lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_79_ = lean_array_uget_borrowed(v_as_75_, v_i_76_);
v___x_80_ = lean_expr_eqv(v_a_74_, v___x_79_);
if (v___x_80_ == 0)
{
size_t v___x_81_; size_t v___x_82_; 
v___x_81_ = ((size_t)1ULL);
v___x_82_ = lean_usize_add(v_i_76_, v___x_81_);
v_i_76_ = v___x_82_;
goto _start;
}
else
{
return v___x_80_;
}
}
else
{
uint8_t v___x_84_; 
v___x_84_ = 0;
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1_spec__2___boxed(lean_object* v_a_85_, lean_object* v_as_86_, lean_object* v_i_87_, lean_object* v_stop_88_){
_start:
{
size_t v_i_boxed_89_; size_t v_stop_boxed_90_; uint8_t v_res_91_; lean_object* v_r_92_; 
v_i_boxed_89_ = lean_unbox_usize(v_i_87_);
lean_dec(v_i_87_);
v_stop_boxed_90_ = lean_unbox_usize(v_stop_88_);
lean_dec(v_stop_88_);
v_res_91_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1_spec__2(v_a_85_, v_as_86_, v_i_boxed_89_, v_stop_boxed_90_);
lean_dec_ref(v_as_86_);
lean_dec_ref(v_a_85_);
v_r_92_ = lean_box(v_res_91_);
return v_r_92_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1(lean_object* v_as_93_, lean_object* v_a_94_){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_95_ = lean_unsigned_to_nat(0u);
v___x_96_ = lean_array_get_size(v_as_93_);
v___x_97_ = lean_nat_dec_lt(v___x_95_, v___x_96_);
if (v___x_97_ == 0)
{
return v___x_97_;
}
else
{
if (v___x_97_ == 0)
{
return v___x_97_;
}
else
{
size_t v___x_98_; size_t v___x_99_; uint8_t v___x_100_; 
v___x_98_ = ((size_t)0ULL);
v___x_99_ = lean_usize_of_nat(v___x_96_);
v___x_100_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1_spec__2(v_a_94_, v_as_93_, v___x_98_, v___x_99_);
return v___x_100_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1___boxed(lean_object* v_as_101_, lean_object* v_a_102_){
_start:
{
uint8_t v_res_103_; lean_object* v_r_104_; 
v_res_103_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1(v_as_101_, v_a_102_);
lean_dec_ref(v_a_102_);
lean_dec_ref(v_as_101_);
v_r_104_ = lean_box(v_res_103_);
return v_r_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___lam__0(lean_object* v_params_105_, lean_object* v_localDecl_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v___x_117_; uint8_t v___x_118_; 
v___x_117_ = l_Lean_LocalDecl_type(v_localDecl_106_);
v___x_118_ = l_Lean_Expr_isEq(v___x_117_);
if (v___x_118_ == 0)
{
uint8_t v___x_119_; 
v___x_119_ = l_Lean_Expr_isHEq(v___x_117_);
lean_dec_ref(v___x_117_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; lean_object* v___x_121_; 
lean_dec_ref(v_localDecl_106_);
v___x_120_ = lean_box(v___x_119_);
v___x_121_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
return v___x_121_;
}
else
{
goto v___jp_112_;
}
}
else
{
lean_dec_ref(v___x_117_);
goto v___jp_112_;
}
v___jp_112_:
{
lean_object* v___x_113_; uint8_t v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_113_ = l_Lean_LocalDecl_toExpr(v_localDecl_106_);
v___x_114_ = lp_mathlib_Array_contains___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__1(v_params_105_, v___x_113_);
lean_dec_ref(v___x_113_);
v___x_115_ = lean_box(v___x_114_);
v___x_116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
return v___x_116_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___lam__0___boxed(lean_object* v_params_122_, lean_object* v_localDecl_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___lam__0(v_params_122_, v_localDecl_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
lean_dec_ref(v_params_122_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0_spec__0(lean_object* v_msgData_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
lean_object* v___x_136_; lean_object* v_env_137_; lean_object* v___x_138_; lean_object* v_mctx_139_; lean_object* v_lctx_140_; lean_object* v_options_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_136_ = lean_st_ref_get(v___y_134_);
v_env_137_ = lean_ctor_get(v___x_136_, 0);
lean_inc_ref(v_env_137_);
lean_dec(v___x_136_);
v___x_138_ = lean_st_ref_get(v___y_132_);
v_mctx_139_ = lean_ctor_get(v___x_138_, 0);
lean_inc_ref(v_mctx_139_);
lean_dec(v___x_138_);
v_lctx_140_ = lean_ctor_get(v___y_131_, 2);
v_options_141_ = lean_ctor_get(v___y_133_, 2);
lean_inc_ref(v_options_141_);
lean_inc_ref(v_lctx_140_);
v___x_142_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_142_, 0, v_env_137_);
lean_ctor_set(v___x_142_, 1, v_mctx_139_);
lean_ctor_set(v___x_142_, 2, v_lctx_140_);
lean_ctor_set(v___x_142_, 3, v_options_141_);
v___x_143_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
lean_ctor_set(v___x_143_, 1, v_msgData_130_);
v___x_144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0_spec__0___boxed(lean_object* v_msgData_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0_spec__0(v_msgData_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec(v___y_147_);
lean_dec_ref(v___y_146_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(lean_object* v_msg_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_){
_start:
{
lean_object* v_ref_158_; lean_object* v___x_159_; lean_object* v_a_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_168_; 
v_ref_158_ = lean_ctor_get(v___y_155_, 5);
v___x_159_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0_spec__0(v_msg_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_);
v_a_160_ = lean_ctor_get(v___x_159_, 0);
v_isSharedCheck_168_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_168_ == 0)
{
v___x_162_ = v___x_159_;
v_isShared_163_ = v_isSharedCheck_168_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_a_160_);
lean_dec(v___x_159_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_168_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v___x_164_; lean_object* v___x_166_; 
lean_inc(v_ref_158_);
v___x_164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_164_, 0, v_ref_158_);
lean_ctor_set(v___x_164_, 1, v_a_160_);
if (v_isShared_163_ == 0)
{
lean_ctor_set_tag(v___x_162_, 1);
lean_ctor_set(v___x_162_, 0, v___x_164_);
v___x_166_ = v___x_162_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v___x_164_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg___boxed(lean_object* v_msg_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v_msg_169_, v___y_170_, v___y_171_, v___y_172_, v___y_173_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
lean_dec(v___y_171_);
lean_dec_ref(v___y_170_);
return v_res_175_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2(void){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_179_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__1));
v___x_180_ = l_Lean_stringToMessageData(v___x_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove(lean_object* v_g_181_, lean_object* v_params_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_){
_start:
{
lean_object* v___x_188_; uint8_t v___x_189_; lean_object* v___x_190_; 
v___x_188_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__0));
v___x_189_ = 1;
v___x_190_ = l___private_Lean_Meta_Tactic_Cleanup_0__Lean_Meta_cleanupCore(v_g_181_, v___x_188_, v___x_189_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_190_) == 0)
{
lean_object* v_a_191_; lean_object* v___f_192_; lean_object* v___x_193_; 
v_a_191_ = lean_ctor_get(v___x_190_, 0);
lean_inc(v_a_191_);
lean_dec_ref_known(v___x_190_, 1);
v___f_192_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___lam__0___boxed), 7, 1);
lean_closure_set(v___f_192_, 0, v_params_182_);
v___x_193_ = l_Lean_MVarId_casesRec(v_a_191_, v___f_192_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_193_) == 0)
{
lean_object* v_a_194_; lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v___y_198_; lean_object* v___y_199_; 
v_a_194_ = lean_ctor_get(v___x_193_, 0);
lean_inc(v_a_194_);
lean_dec_ref_known(v___x_193_, 1);
if (lean_obj_tag(v_a_194_) == 1)
{
lean_object* v_head_202_; lean_object* v_tail_203_; lean_object* v___y_205_; uint8_t v___y_206_; 
v_head_202_ = lean_ctor_get(v_a_194_, 0);
lean_inc(v_head_202_);
v_tail_203_ = lean_ctor_get(v_a_194_, 1);
lean_inc(v_tail_203_);
lean_dec_ref_known(v_a_194_, 2);
if (lean_obj_tag(v_tail_203_) == 0)
{
lean_object* v___x_257_; 
lean_inc(v_head_202_);
v___x_257_ = l_Lean_MVarId_refl(v_head_202_, v___x_189_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_257_) == 0)
{
lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_265_; 
lean_dec(v_head_202_);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_257_);
if (v_isSharedCheck_265_ == 0)
{
lean_object* v_unused_266_; 
v_unused_266_ = lean_ctor_get(v___x_257_, 0);
lean_dec(v_unused_266_);
v___x_259_ = v___x_257_;
v_isShared_260_ = v_isSharedCheck_265_;
goto v_resetjp_258_;
}
else
{
lean_dec(v___x_257_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_265_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_261_; lean_object* v___x_263_; 
v___x_261_ = lean_box(0);
if (v_isShared_260_ == 0)
{
lean_ctor_set(v___x_259_, 0, v___x_261_);
v___x_263_ = v___x_259_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_267_; uint8_t v___y_269_; uint8_t v___x_283_; 
v_a_267_ = lean_ctor_get(v___x_257_, 0);
lean_inc(v_a_267_);
v___x_283_ = l_Lean_Exception_isInterrupt(v_a_267_);
if (v___x_283_ == 0)
{
uint8_t v___x_284_; 
v___x_284_ = l_Lean_Exception_isRuntime(v_a_267_);
v___y_269_ = v___x_284_;
goto v___jp_268_;
}
else
{
lean_dec(v_a_267_);
v___y_269_ = v___x_283_;
goto v___jp_268_;
}
v___jp_268_:
{
if (v___y_269_ == 0)
{
lean_object* v___x_270_; 
lean_dec_ref_known(v___x_257_, 1);
lean_inc(v_head_202_);
v___x_270_ = l_Lean_MVarId_hrefl(v_head_202_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_270_) == 0)
{
lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_278_; 
lean_dec(v_head_202_);
v_isSharedCheck_278_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_278_ == 0)
{
lean_object* v_unused_279_; 
v_unused_279_ = lean_ctor_get(v___x_270_, 0);
lean_dec(v_unused_279_);
v___x_272_ = v___x_270_;
v_isShared_273_ = v_isSharedCheck_278_;
goto v_resetjp_271_;
}
else
{
lean_dec(v___x_270_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_278_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v___x_274_; lean_object* v___x_276_; 
v___x_274_ = lean_box(0);
if (v_isShared_273_ == 0)
{
lean_ctor_set(v___x_272_, 0, v___x_274_);
v___x_276_ = v___x_272_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v___x_274_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
}
else
{
lean_object* v_a_280_; uint8_t v___x_281_; 
v_a_280_ = lean_ctor_get(v___x_270_, 0);
lean_inc(v_a_280_);
v___x_281_ = l_Lean_Exception_isInterrupt(v_a_280_);
if (v___x_281_ == 0)
{
uint8_t v___x_282_; 
v___x_282_ = l_Lean_Exception_isRuntime(v_a_280_);
v___y_205_ = v___x_270_;
v___y_206_ = v___x_282_;
goto v___jp_204_;
}
else
{
lean_dec(v_a_280_);
v___y_205_ = v___x_270_;
v___y_206_ = v___x_281_;
goto v___jp_204_;
}
}
}
else
{
lean_dec(v_head_202_);
return v___x_257_;
}
}
}
}
else
{
lean_dec(v_tail_203_);
lean_dec(v_head_202_);
v___y_196_ = v_a_183_;
v___y_197_ = v_a_184_;
v___y_198_ = v_a_185_;
v___y_199_ = v_a_186_;
goto v___jp_195_;
}
v___jp_204_:
{
if (v___y_206_ == 0)
{
lean_object* v___x_207_; 
lean_dec_ref(v___y_205_);
lean_inc(v_head_202_);
v___x_207_ = l_Lean_MVarId_proofIrrelHeq(v_head_202_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_207_) == 0)
{
lean_object* v_a_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_248_; 
v_a_208_ = lean_ctor_get(v___x_207_, 0);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_207_);
if (v_isSharedCheck_248_ == 0)
{
v___x_210_ = v___x_207_;
v_isShared_211_ = v_isSharedCheck_248_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_a_208_);
lean_dec(v___x_207_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_248_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
uint8_t v___x_212_; 
v___x_212_ = lean_unbox(v_a_208_);
lean_dec(v_a_208_);
if (v___x_212_ == 0)
{
lean_object* v___x_213_; 
lean_del_object(v___x_210_);
v___x_213_ = l_Lean_MVarId_heqOfEq(v_head_202_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_213_) == 0)
{
lean_object* v_a_214_; lean_object* v___x_215_; 
v_a_214_ = lean_ctor_get(v___x_213_, 0);
lean_inc(v_a_214_);
lean_dec_ref_known(v___x_213_, 1);
v___x_215_ = l_Lean_MVarId_subsingletonElim(v_a_214_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_227_; 
v_a_216_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_227_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_227_ == 0)
{
v___x_218_ = v___x_215_;
v_isShared_219_ = v_isSharedCheck_227_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_215_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_227_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
uint8_t v___x_220_; 
v___x_220_ = lean_unbox(v_a_216_);
lean_dec(v_a_216_);
if (v___x_220_ == 0)
{
lean_object* v___x_221_; lean_object* v___x_222_; 
lean_del_object(v___x_218_);
v___x_221_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2);
v___x_222_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v___x_221_, v_a_183_, v_a_184_, v_a_185_, v_a_186_);
return v___x_222_;
}
else
{
lean_object* v___x_223_; lean_object* v___x_225_; 
v___x_223_ = lean_box(0);
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 0, v___x_223_);
v___x_225_ = v___x_218_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v___x_223_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
}
else
{
lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
v_a_228_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_235_ == 0)
{
v___x_230_ = v___x_215_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_215_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_a_228_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
else
{
lean_object* v_a_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_243_; 
v_a_236_ = lean_ctor_get(v___x_213_, 0);
v_isSharedCheck_243_ = !lean_is_exclusive(v___x_213_);
if (v_isSharedCheck_243_ == 0)
{
v___x_238_ = v___x_213_;
v_isShared_239_ = v_isSharedCheck_243_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_a_236_);
lean_dec(v___x_213_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_243_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_241_; 
if (v_isShared_239_ == 0)
{
v___x_241_ = v___x_238_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_242_; 
v_reuseFailAlloc_242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_242_, 0, v_a_236_);
v___x_241_ = v_reuseFailAlloc_242_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
return v___x_241_;
}
}
}
}
else
{
lean_object* v___x_244_; lean_object* v___x_246_; 
lean_dec(v_head_202_);
v___x_244_ = lean_box(0);
if (v_isShared_211_ == 0)
{
lean_ctor_set(v___x_210_, 0, v___x_244_);
v___x_246_ = v___x_210_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v___x_244_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
}
else
{
lean_object* v_a_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_256_; 
lean_dec(v_head_202_);
v_a_249_ = lean_ctor_get(v___x_207_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v___x_207_);
if (v_isSharedCheck_256_ == 0)
{
v___x_251_ = v___x_207_;
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_a_249_);
lean_dec(v___x_207_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_254_; 
if (v_isShared_252_ == 0)
{
v___x_254_ = v___x_251_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_a_249_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
return v___x_254_;
}
}
}
}
else
{
lean_dec(v_head_202_);
return v___y_205_;
}
}
}
else
{
lean_dec(v_a_194_);
v___y_196_ = v_a_183_;
v___y_197_ = v_a_184_;
v___y_198_ = v_a_185_;
v___y_199_ = v_a_186_;
goto v___jp_195_;
}
v___jp_195_:
{
lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_200_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2);
v___x_201_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v___x_200_, v___y_196_, v___y_197_, v___y_198_, v___y_199_);
return v___x_201_;
}
}
else
{
lean_object* v_a_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_292_; 
v_a_285_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_292_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_292_ == 0)
{
v___x_287_ = v___x_193_;
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_a_285_);
lean_dec(v___x_193_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_292_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_290_; 
if (v_isShared_288_ == 0)
{
v___x_290_ = v___x_287_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v_a_285_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
}
}
else
{
lean_object* v_a_293_; lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_300_; 
lean_dec_ref(v_params_182_);
v_a_293_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_300_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_300_ == 0)
{
v___x_295_ = v___x_190_;
v_isShared_296_ = v_isSharedCheck_300_;
goto v_resetjp_294_;
}
else
{
lean_inc(v_a_293_);
lean_dec(v___x_190_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_300_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
lean_object* v___x_298_; 
if (v_isShared_296_ == 0)
{
v___x_298_ = v___x_295_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_a_293_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___boxed(lean_object* v_g_301_, lean_object* v_params_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_, lean_object* v_a_306_, lean_object* v_a_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove(v_g_301_, v_params_302_, v_a_303_, v_a_304_, v_a_305_, v_a_306_);
lean_dec(v_a_306_);
lean_dec_ref(v_a_305_);
lean_dec(v_a_304_);
lean_dec_ref(v_a_303_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0(lean_object* v_00_u03b1_309_, lean_object* v_msg_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v_msg_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___boxed(lean_object* v_00_u03b1_317_, lean_object* v_msg_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0(v_00_u03b1_317_, v_msg_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
lean_dec(v___y_320_);
lean_dec_ref(v___y_319_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1(lean_object* v_msg_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_){
_start:
{
lean_object* v___f_332_; lean_object* v___x_2006__overap_333_; lean_object* v___x_334_; 
v___f_332_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1___closed__0));
v___x_2006__overap_333_ = lean_panic_fn_borrowed(v___f_332_, v_msg_326_);
lean_inc(v___y_330_);
lean_inc_ref(v___y_329_);
lean_inc(v___y_328_);
lean_inc_ref(v___y_327_);
v___x_334_ = lean_apply_5(v___x_2006__overap_333_, v___y_327_, v___y_328_, v___y_329_, v___y_330_, lean_box(0));
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1___boxed(lean_object* v_msg_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1(v_msg_335_, v___y_336_, v___y_337_, v___y_338_, v___y_339_);
lean_dec(v___y_339_);
lean_dec_ref(v___y_338_);
lean_dec(v___y_337_);
lean_dec_ref(v___y_336_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg(lean_object* v_x_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = l_Lean_Meta_saveState___redArg(v___y_344_, v___y_346_);
if (lean_obj_tag(v___x_348_) == 0)
{
lean_object* v_a_349_; lean_object* v___x_350_; 
v_a_349_ = lean_ctor_get(v___x_348_, 0);
lean_inc(v_a_349_);
lean_dec_ref_known(v___x_348_, 1);
lean_inc(v___y_346_);
lean_inc_ref(v___y_345_);
lean_inc(v___y_344_);
lean_inc_ref(v___y_343_);
v___x_350_ = lean_apply_5(v_x_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, lean_box(0));
if (lean_obj_tag(v___x_350_) == 0)
{
lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_359_; 
lean_dec(v_a_349_);
v_a_351_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_359_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_359_ == 0)
{
v___x_353_ = v___x_350_;
v_isShared_354_ = v_isSharedCheck_359_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_350_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_359_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v___x_355_; lean_object* v___x_357_; 
v___x_355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_355_, 0, v_a_351_);
if (v_isShared_354_ == 0)
{
lean_ctor_set(v___x_353_, 0, v___x_355_);
v___x_357_ = v___x_353_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v___x_355_);
v___x_357_ = v_reuseFailAlloc_358_;
goto v_reusejp_356_;
}
v_reusejp_356_:
{
return v___x_357_;
}
}
}
else
{
lean_object* v_a_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_389_; 
v_a_360_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_389_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_389_ == 0)
{
v___x_362_ = v___x_350_;
v_isShared_363_ = v_isSharedCheck_389_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_a_360_);
lean_dec(v___x_350_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_389_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
uint8_t v___y_365_; uint8_t v___x_387_; 
v___x_387_ = l_Lean_Exception_isInterrupt(v_a_360_);
if (v___x_387_ == 0)
{
uint8_t v___x_388_; 
lean_inc(v_a_360_);
v___x_388_ = l_Lean_Exception_isRuntime(v_a_360_);
v___y_365_ = v___x_388_;
goto v___jp_364_;
}
else
{
v___y_365_ = v___x_387_;
goto v___jp_364_;
}
v___jp_364_:
{
if (v___y_365_ == 0)
{
lean_object* v___x_366_; 
lean_del_object(v___x_362_);
lean_dec(v_a_360_);
v___x_366_ = l_Lean_Meta_SavedState_restore___redArg(v_a_349_, v___y_344_, v___y_346_);
lean_dec(v_a_349_);
if (lean_obj_tag(v___x_366_) == 0)
{
lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_374_; 
v_isSharedCheck_374_ = !lean_is_exclusive(v___x_366_);
if (v_isSharedCheck_374_ == 0)
{
lean_object* v_unused_375_; 
v_unused_375_ = lean_ctor_get(v___x_366_, 0);
lean_dec(v_unused_375_);
v___x_368_ = v___x_366_;
v_isShared_369_ = v_isSharedCheck_374_;
goto v_resetjp_367_;
}
else
{
lean_dec(v___x_366_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_374_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_370_; lean_object* v___x_372_; 
v___x_370_ = lean_box(0);
if (v_isShared_369_ == 0)
{
lean_ctor_set(v___x_368_, 0, v___x_370_);
v___x_372_ = v___x_368_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v___x_370_);
v___x_372_ = v_reuseFailAlloc_373_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
return v___x_372_;
}
}
}
else
{
lean_object* v_a_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_383_; 
v_a_376_ = lean_ctor_get(v___x_366_, 0);
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_366_);
if (v_isSharedCheck_383_ == 0)
{
v___x_378_ = v___x_366_;
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_a_376_);
lean_dec(v___x_366_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_381_; 
if (v_isShared_379_ == 0)
{
v___x_381_ = v___x_378_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_a_376_);
v___x_381_ = v_reuseFailAlloc_382_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
return v___x_381_;
}
}
}
}
else
{
lean_object* v___x_385_; 
lean_dec(v_a_349_);
if (v_isShared_363_ == 0)
{
v___x_385_ = v___x_362_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v_a_360_);
v___x_385_ = v_reuseFailAlloc_386_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
return v___x_385_;
}
}
}
}
}
}
else
{
lean_object* v_a_390_; lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_397_; 
lean_dec_ref(v_x_342_);
v_a_390_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_397_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_397_ == 0)
{
v___x_392_ = v___x_348_;
v_isShared_393_ = v_isSharedCheck_397_;
goto v_resetjp_391_;
}
else
{
lean_inc(v_a_390_);
lean_dec(v___x_348_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_397_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
lean_object* v___x_395_; 
if (v_isShared_393_ == 0)
{
v___x_395_ = v___x_392_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_396_; 
v_reuseFailAlloc_396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_396_, 0, v_a_390_);
v___x_395_ = v_reuseFailAlloc_396_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
return v___x_395_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg___boxed(lean_object* v_x_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg(v_x_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
lean_dec(v___y_400_);
lean_dec_ref(v___y_399_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2(lean_object* v_00_u03b1_405_, lean_object* v_x_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg(v_x_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___boxed(lean_object* v_00_u03b1_413_, lean_object* v_x_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2(v_00_u03b1_413_, v_x_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
lean_dec(v___y_418_);
lean_dec_ref(v___y_417_);
lean_dec(v___y_416_);
lean_dec_ref(v___y_415_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(lean_object* v_e_421_, lean_object* v___y_422_){
_start:
{
uint8_t v___x_424_; 
v___x_424_ = l_Lean_Expr_hasMVar(v_e_421_);
if (v___x_424_ == 0)
{
lean_object* v___x_425_; 
v___x_425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_425_, 0, v_e_421_);
return v___x_425_;
}
else
{
lean_object* v___x_426_; lean_object* v_mctx_427_; lean_object* v___x_428_; lean_object* v_fst_429_; lean_object* v_snd_430_; lean_object* v___x_431_; lean_object* v_cache_432_; lean_object* v_zetaDeltaFVarIds_433_; lean_object* v_postponed_434_; lean_object* v_diag_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_444_; 
v___x_426_ = lean_st_ref_get(v___y_422_);
v_mctx_427_ = lean_ctor_get(v___x_426_, 0);
lean_inc_ref(v_mctx_427_);
lean_dec(v___x_426_);
v___x_428_ = l_Lean_instantiateMVarsCore(v_mctx_427_, v_e_421_);
v_fst_429_ = lean_ctor_get(v___x_428_, 0);
lean_inc(v_fst_429_);
v_snd_430_ = lean_ctor_get(v___x_428_, 1);
lean_inc(v_snd_430_);
lean_dec_ref(v___x_428_);
v___x_431_ = lean_st_ref_take(v___y_422_);
v_cache_432_ = lean_ctor_get(v___x_431_, 1);
v_zetaDeltaFVarIds_433_ = lean_ctor_get(v___x_431_, 2);
v_postponed_434_ = lean_ctor_get(v___x_431_, 3);
v_diag_435_ = lean_ctor_get(v___x_431_, 4);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_444_ == 0)
{
lean_object* v_unused_445_; 
v_unused_445_ = lean_ctor_get(v___x_431_, 0);
lean_dec(v_unused_445_);
v___x_437_ = v___x_431_;
v_isShared_438_ = v_isSharedCheck_444_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_diag_435_);
lean_inc(v_postponed_434_);
lean_inc(v_zetaDeltaFVarIds_433_);
lean_inc(v_cache_432_);
lean_dec(v___x_431_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_444_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v___x_440_; 
if (v_isShared_438_ == 0)
{
lean_ctor_set(v___x_437_, 0, v_snd_430_);
v___x_440_ = v___x_437_;
goto v_reusejp_439_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v_snd_430_);
lean_ctor_set(v_reuseFailAlloc_443_, 1, v_cache_432_);
lean_ctor_set(v_reuseFailAlloc_443_, 2, v_zetaDeltaFVarIds_433_);
lean_ctor_set(v_reuseFailAlloc_443_, 3, v_postponed_434_);
lean_ctor_set(v_reuseFailAlloc_443_, 4, v_diag_435_);
v___x_440_ = v_reuseFailAlloc_443_;
goto v_reusejp_439_;
}
v_reusejp_439_:
{
lean_object* v___x_441_; lean_object* v___x_442_; 
v___x_441_ = lean_st_ref_set(v___y_422_, v___x_440_);
v___x_442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_442_, 0, v_fst_429_);
return v___x_442_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg___boxed(lean_object* v_e_446_, lean_object* v___y_447_, lean_object* v___y_448_){
_start:
{
lean_object* v_res_449_; 
v_res_449_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(v_e_446_, v___y_447_);
lean_dec(v___y_447_);
return v_res_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3(lean_object* v_e_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(v_e_450_, v___y_452_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___boxed(lean_object* v_e_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_){
_start:
{
lean_object* v_res_463_; 
v_res_463_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3(v_e_457_, v___y_458_, v___y_459_, v___y_460_, v___y_461_);
lean_dec(v___y_461_);
lean_dec_ref(v___y_460_);
lean_dec(v___y_459_);
lean_dec_ref(v___y_458_);
return v_res_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___lam__0(lean_object* v_k_464_, lean_object* v_b_465_, lean_object* v_c_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_){
_start:
{
lean_object* v___x_472_; 
lean_inc(v___y_470_);
lean_inc_ref(v___y_469_);
lean_inc(v___y_468_);
lean_inc_ref(v___y_467_);
v___x_472_ = lean_apply_7(v_k_464_, v_b_465_, v_c_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_, lean_box(0));
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___lam__0___boxed(lean_object* v_k_473_, lean_object* v_b_474_, lean_object* v_c_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___lam__0(v_k_473_, v_b_474_, v_c_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_);
lean_dec(v___y_479_);
lean_dec_ref(v___y_478_);
lean_dec(v___y_477_);
lean_dec_ref(v___y_476_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg(lean_object* v_type_482_, lean_object* v_maxFVars_x3f_483_, lean_object* v_k_484_, uint8_t v_cleanupAnnotations_485_, uint8_t v_whnfType_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_){
_start:
{
lean_object* v___f_492_; lean_object* v___x_493_; 
v___f_492_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_492_, 0, v_k_484_);
v___x_493_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_482_, v_maxFVars_x3f_483_, v___f_492_, v_cleanupAnnotations_485_, v_whnfType_486_, v___y_487_, v___y_488_, v___y_489_, v___y_490_);
if (lean_obj_tag(v___x_493_) == 0)
{
lean_object* v_a_494_; lean_object* v___x_496_; uint8_t v_isShared_497_; uint8_t v_isSharedCheck_501_; 
v_a_494_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_501_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_501_ == 0)
{
v___x_496_ = v___x_493_;
v_isShared_497_ = v_isSharedCheck_501_;
goto v_resetjp_495_;
}
else
{
lean_inc(v_a_494_);
lean_dec(v___x_493_);
v___x_496_ = lean_box(0);
v_isShared_497_ = v_isSharedCheck_501_;
goto v_resetjp_495_;
}
v_resetjp_495_:
{
lean_object* v___x_499_; 
if (v_isShared_497_ == 0)
{
v___x_499_ = v___x_496_;
goto v_reusejp_498_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v_a_494_);
v___x_499_ = v_reuseFailAlloc_500_;
goto v_reusejp_498_;
}
v_reusejp_498_:
{
return v___x_499_;
}
}
}
else
{
lean_object* v_a_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_509_; 
v_a_502_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_509_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_509_ == 0)
{
v___x_504_ = v___x_493_;
v_isShared_505_ = v_isSharedCheck_509_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_a_502_);
lean_dec(v___x_493_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_509_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
lean_object* v___x_507_; 
if (v_isShared_505_ == 0)
{
v___x_507_ = v___x_504_;
goto v_reusejp_506_;
}
else
{
lean_object* v_reuseFailAlloc_508_; 
v_reuseFailAlloc_508_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_508_, 0, v_a_502_);
v___x_507_ = v_reuseFailAlloc_508_;
goto v_reusejp_506_;
}
v_reusejp_506_:
{
return v___x_507_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg___boxed(lean_object* v_type_510_, lean_object* v_maxFVars_x3f_511_, lean_object* v_k_512_, lean_object* v_cleanupAnnotations_513_, lean_object* v_whnfType_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_520_; uint8_t v_whnfType_boxed_521_; lean_object* v_res_522_; 
v_cleanupAnnotations_boxed_520_ = lean_unbox(v_cleanupAnnotations_513_);
v_whnfType_boxed_521_ = lean_unbox(v_whnfType_514_);
v_res_522_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg(v_type_510_, v_maxFVars_x3f_511_, v_k_512_, v_cleanupAnnotations_boxed_520_, v_whnfType_boxed_521_, v___y_515_, v___y_516_, v___y_517_, v___y_518_);
lean_dec(v___y_518_);
lean_dec_ref(v___y_517_);
lean_dec(v___y_516_);
lean_dec_ref(v___y_515_);
return v_res_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4(lean_object* v_00_u03b1_523_, lean_object* v_type_524_, lean_object* v_maxFVars_x3f_525_, lean_object* v_k_526_, uint8_t v_cleanupAnnotations_527_, uint8_t v_whnfType_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg(v_type_524_, v_maxFVars_x3f_525_, v_k_526_, v_cleanupAnnotations_527_, v_whnfType_528_, v___y_529_, v___y_530_, v___y_531_, v___y_532_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___boxed(lean_object* v_00_u03b1_535_, lean_object* v_type_536_, lean_object* v_maxFVars_x3f_537_, lean_object* v_k_538_, lean_object* v_cleanupAnnotations_539_, lean_object* v_whnfType_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_546_; uint8_t v_whnfType_boxed_547_; lean_object* v_res_548_; 
v_cleanupAnnotations_boxed_546_ = lean_unbox(v_cleanupAnnotations_539_);
v_whnfType_boxed_547_ = lean_unbox(v_whnfType_540_);
v_res_548_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4(v_00_u03b1_535_, v_type_536_, v_maxFVars_x3f_537_, v_k_538_, v_cleanupAnnotations_boxed_546_, v_whnfType_boxed_547_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__0(lean_object* v_as_549_, size_t v_i_550_, size_t v_stop_551_, lean_object* v_b_552_){
_start:
{
uint8_t v___x_553_; 
v___x_553_ = lean_usize_dec_eq(v_i_550_, v_stop_551_);
if (v___x_553_ == 0)
{
size_t v___x_554_; size_t v___x_555_; lean_object* v___x_556_; lean_object* v_fst_557_; lean_object* v_snd_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; 
v___x_554_ = ((size_t)1ULL);
v___x_555_ = lean_usize_sub(v_i_550_, v___x_554_);
v___x_556_ = lean_array_uget_borrowed(v_as_549_, v___x_555_);
v_fst_557_ = lean_ctor_get(v___x_556_, 0);
v_snd_558_ = lean_ctor_get(v___x_556_, 1);
v___x_559_ = lean_unsigned_to_nat(1u);
v___x_560_ = lean_mk_empty_array_with_capacity(v___x_559_);
lean_inc(v_fst_557_);
v___x_561_ = lean_array_push(v___x_560_, v_fst_557_);
v___x_562_ = lean_expr_abstract(v_b_552_, v___x_561_);
lean_dec_ref(v___x_561_);
lean_dec_ref(v_b_552_);
v___x_563_ = lean_expr_instantiate1(v___x_562_, v_snd_558_);
lean_dec_ref(v___x_562_);
v_i_550_ = v___x_555_;
v_b_552_ = v___x_563_;
goto _start;
}
else
{
return v_b_552_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__0___boxed(lean_object* v_as_565_, lean_object* v_i_566_, lean_object* v_stop_567_, lean_object* v_b_568_){
_start:
{
size_t v_i_boxed_569_; size_t v_stop_boxed_570_; lean_object* v_res_571_; 
v_i_boxed_569_ = lean_unbox_usize(v_i_566_);
lean_dec(v_i_566_);
v_stop_boxed_570_ = lean_unbox_usize(v_stop_567_);
lean_dec(v_stop_567_);
v_res_571_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__0(v_as_565_, v_i_boxed_569_, v_stop_boxed_570_, v_b_568_);
lean_dec_ref(v_as_565_);
return v_res_571_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__4(void){
_start:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v___x_577_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__3));
v___x_578_ = lean_unsigned_to_nat(13u);
v___x_579_ = lean_unsigned_to_nat(82u);
v___x_580_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__2));
v___x_581_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__1));
v___x_582_ = l_mkPanicMessageWithDecl(v___x_581_, v___x_580_, v___x_579_, v___x_578_, v___x_577_);
return v___x_582_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__1(void){
_start:
{
lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_584_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__0));
v___x_585_ = lean_unsigned_to_nat(39u);
v___x_586_ = lean_unsigned_to_nat(69u);
v___x_587_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__2));
v___x_588_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__1));
v___x_589_ = l_mkPanicMessageWithDecl(v___x_588_, v___x_587_, v___x_586_, v___x_585_, v___x_584_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0(lean_object* v___x_590_, lean_object* v_args_591_, lean_object* v_argKinds_x27_592_, uint8_t v_head_593_, lean_object* v_params_594_, lean_object* v_cthm_595_, lean_object* v_tail_596_, lean_object* v_letArgs_597_, lean_object* v_params_x27_598_, lean_object* v_type_x27_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_){
_start:
{
lean_object* v___x_605_; uint8_t v___x_606_; 
v___x_605_ = lean_array_get_size(v_params_x27_598_);
v___x_606_ = lean_nat_dec_eq(v___x_605_, v___x_590_);
if (v___x_606_ == 0)
{
lean_object* v___x_607_; lean_object* v___x_608_; 
lean_dec_ref(v_type_x27_599_);
lean_dec_ref(v_letArgs_597_);
lean_dec(v_tail_596_);
lean_dec_ref(v_cthm_595_);
lean_dec_ref(v_params_594_);
lean_dec_ref(v_argKinds_x27_592_);
lean_dec_ref(v_args_591_);
v___x_607_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__1, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___closed__1);
v___x_608_ = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1(v___x_607_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
return v___x_608_;
}
else
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_609_ = lean_unsigned_to_nat(2u);
v___x_610_ = lean_array_fget_borrowed(v_params_x27_598_, v___x_609_);
lean_inc(v___y_603_);
lean_inc_ref(v___y_602_);
lean_inc(v___y_601_);
lean_inc_ref(v___y_600_);
lean_inc(v___x_610_);
v___x_611_ = lean_infer_type(v___x_610_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
if (lean_obj_tag(v___x_611_) == 0)
{
lean_object* v_a_612_; lean_object* v___x_613_; uint8_t v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
v_a_612_ = lean_ctor_get(v___x_611_, 0);
lean_inc(v_a_612_);
lean_dec_ref_known(v___x_611_, 1);
v___x_613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_613_, 0, v_a_612_);
v___x_614_ = 0;
v___x_615_ = lean_box(0);
v___x_616_ = l_Lean_Meta_mkFreshExprMVar(v___x_613_, v___x_614_, v___x_615_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
if (lean_obj_tag(v___x_616_) == 0)
{
lean_object* v_a_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; 
v_a_617_ = lean_ctor_get(v___x_616_, 0);
lean_inc(v_a_617_);
lean_dec_ref_known(v___x_616_, 1);
v___x_618_ = l_Lean_Expr_mvarId_x21(v_a_617_);
lean_dec(v_a_617_);
v___x_619_ = l_Lean_Expr_fvarId_x21(v___x_610_);
v___x_620_ = l_Lean_MVarId_clear(v___x_618_, v___x_619_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
if (lean_obj_tag(v___x_620_) == 0)
{
lean_object* v_a_621_; lean_object* v___x_622_; lean_object* v___x_623_; 
v_a_621_ = lean_ctor_get(v___x_620_, 0);
lean_inc_n(v_a_621_, 2);
lean_dec_ref_known(v___x_620_, 1);
lean_inc_ref(v_args_591_);
v___x_622_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___boxed), 7, 2);
lean_closure_set(v___x_622_, 0, v_a_621_);
lean_closure_set(v___x_622_, 1, v_args_591_);
v___x_623_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg(v___x_622_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
if (lean_obj_tag(v___x_623_) == 0)
{
lean_object* v_a_624_; 
v_a_624_ = lean_ctor_get(v___x_623_, 0);
lean_inc(v_a_624_);
lean_dec_ref_known(v___x_623_, 1);
if (lean_obj_tag(v_a_624_) == 0)
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
lean_dec(v_a_621_);
v___x_625_ = lean_box(v_head_593_);
v___x_626_ = lean_array_push(v_argKinds_x27_592_, v___x_625_);
v___x_627_ = l_Array_append___redArg(v_params_594_, v_params_x27_598_);
v___x_628_ = l_Array_append___redArg(v_args_591_, v_params_x27_598_);
v___x_629_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process(v_cthm_595_, v_type_x27_599_, v_tail_596_, v___x_626_, v___x_627_, v___x_628_, v_letArgs_597_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
return v___x_629_;
}
else
{
lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v_a_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; uint8_t v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
lean_dec_ref_known(v_a_624_, 1);
v___x_630_ = l_Lean_Expr_mvar___override(v_a_621_);
v___x_631_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(v___x_630_, v___y_601_);
v_a_632_ = lean_ctor_get(v___x_631_, 0);
lean_inc(v_a_632_);
lean_dec_ref(v___x_631_);
v___x_633_ = lean_unsigned_to_nat(0u);
v___x_634_ = lean_array_fget_borrowed(v_params_x27_598_, v___x_633_);
v___x_635_ = lean_unsigned_to_nat(1u);
v___x_636_ = lean_array_fget_borrowed(v_params_x27_598_, v___x_635_);
v___x_637_ = 5;
v___x_638_ = lean_box(v___x_637_);
v___x_639_ = lean_array_push(v_argKinds_x27_592_, v___x_638_);
v___x_640_ = lean_mk_empty_array_with_capacity(v___x_609_);
lean_inc(v___x_634_);
v___x_641_ = lean_array_push(v___x_640_, v___x_634_);
lean_inc(v___x_636_);
v___x_642_ = lean_array_push(v___x_641_, v___x_636_);
v___x_643_ = l_Array_append___redArg(v_params_594_, v___x_642_);
lean_dec_ref(v___x_642_);
v___x_644_ = l_Array_append___redArg(v_args_591_, v_params_x27_598_);
lean_inc(v___x_610_);
v___x_645_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_645_, 0, v___x_610_);
lean_ctor_set(v___x_645_, 1, v_a_632_);
v___x_646_ = lean_array_push(v_letArgs_597_, v___x_645_);
v___x_647_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process(v_cthm_595_, v_type_x27_599_, v_tail_596_, v___x_639_, v___x_643_, v___x_644_, v___x_646_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
return v___x_647_;
}
}
else
{
lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_655_; 
lean_dec(v_a_621_);
lean_dec_ref(v_type_x27_599_);
lean_dec_ref(v_letArgs_597_);
lean_dec(v_tail_596_);
lean_dec_ref(v_cthm_595_);
lean_dec_ref(v_params_594_);
lean_dec_ref(v_argKinds_x27_592_);
lean_dec_ref(v_args_591_);
v_a_648_ = lean_ctor_get(v___x_623_, 0);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_623_);
if (v_isSharedCheck_655_ == 0)
{
v___x_650_ = v___x_623_;
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_623_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_a_648_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
}
else
{
lean_object* v_a_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_663_; 
lean_dec_ref(v_type_x27_599_);
lean_dec_ref(v_letArgs_597_);
lean_dec(v_tail_596_);
lean_dec_ref(v_cthm_595_);
lean_dec_ref(v_params_594_);
lean_dec_ref(v_argKinds_x27_592_);
lean_dec_ref(v_args_591_);
v_a_656_ = lean_ctor_get(v___x_620_, 0);
v_isSharedCheck_663_ = !lean_is_exclusive(v___x_620_);
if (v_isSharedCheck_663_ == 0)
{
v___x_658_ = v___x_620_;
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_a_656_);
lean_dec(v___x_620_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v___x_661_; 
if (v_isShared_659_ == 0)
{
v___x_661_ = v___x_658_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v_a_656_);
v___x_661_ = v_reuseFailAlloc_662_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
return v___x_661_;
}
}
}
}
else
{
lean_object* v_a_664_; lean_object* v___x_666_; uint8_t v_isShared_667_; uint8_t v_isSharedCheck_671_; 
lean_dec_ref(v_type_x27_599_);
lean_dec_ref(v_letArgs_597_);
lean_dec(v_tail_596_);
lean_dec_ref(v_cthm_595_);
lean_dec_ref(v_params_594_);
lean_dec_ref(v_argKinds_x27_592_);
lean_dec_ref(v_args_591_);
v_a_664_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_671_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_671_ == 0)
{
v___x_666_ = v___x_616_;
v_isShared_667_ = v_isSharedCheck_671_;
goto v_resetjp_665_;
}
else
{
lean_inc(v_a_664_);
lean_dec(v___x_616_);
v___x_666_ = lean_box(0);
v_isShared_667_ = v_isSharedCheck_671_;
goto v_resetjp_665_;
}
v_resetjp_665_:
{
lean_object* v___x_669_; 
if (v_isShared_667_ == 0)
{
v___x_669_ = v___x_666_;
goto v_reusejp_668_;
}
else
{
lean_object* v_reuseFailAlloc_670_; 
v_reuseFailAlloc_670_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_670_, 0, v_a_664_);
v___x_669_ = v_reuseFailAlloc_670_;
goto v_reusejp_668_;
}
v_reusejp_668_:
{
return v___x_669_;
}
}
}
}
else
{
lean_object* v_a_672_; lean_object* v___x_674_; uint8_t v_isShared_675_; uint8_t v_isSharedCheck_679_; 
lean_dec_ref(v_type_x27_599_);
lean_dec_ref(v_letArgs_597_);
lean_dec(v_tail_596_);
lean_dec_ref(v_cthm_595_);
lean_dec_ref(v_params_594_);
lean_dec_ref(v_argKinds_x27_592_);
lean_dec_ref(v_args_591_);
v_a_672_ = lean_ctor_get(v___x_611_, 0);
v_isSharedCheck_679_ = !lean_is_exclusive(v___x_611_);
if (v_isSharedCheck_679_ == 0)
{
v___x_674_ = v___x_611_;
v_isShared_675_ = v_isSharedCheck_679_;
goto v_resetjp_673_;
}
else
{
lean_inc(v_a_672_);
lean_dec(v___x_611_);
v___x_674_ = lean_box(0);
v_isShared_675_ = v_isSharedCheck_679_;
goto v_resetjp_673_;
}
v_resetjp_673_:
{
lean_object* v___x_677_; 
if (v_isShared_675_ == 0)
{
v___x_677_ = v___x_674_;
goto v_reusejp_676_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v_a_672_);
v___x_677_ = v_reuseFailAlloc_678_;
goto v_reusejp_676_;
}
v_reusejp_676_:
{
return v___x_677_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___boxed(lean_object* v___x_680_, lean_object* v_args_681_, lean_object* v_argKinds_x27_682_, lean_object* v_head_683_, lean_object* v_params_684_, lean_object* v_cthm_685_, lean_object* v_tail_686_, lean_object* v_letArgs_687_, lean_object* v_params_x27_688_, lean_object* v_type_x27_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_){
_start:
{
uint8_t v_head_4299__boxed_695_; lean_object* v_res_696_; 
v_head_4299__boxed_695_ = lean_unbox(v_head_683_);
v_res_696_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0(v___x_680_, v_args_681_, v_argKinds_x27_682_, v_head_4299__boxed_695_, v_params_684_, v_cthm_685_, v_tail_686_, v_letArgs_687_, v_params_x27_688_, v_type_x27_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_);
lean_dec(v___y_693_);
lean_dec_ref(v___y_692_);
lean_dec(v___y_691_);
lean_dec_ref(v___y_690_);
lean_dec_ref(v_params_x27_688_);
lean_dec(v___x_680_);
return v_res_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process(lean_object* v_cthm_697_, lean_object* v_type_698_, lean_object* v_argKinds_699_, lean_object* v_argKinds_x27_700_, lean_object* v_params_701_, lean_object* v_args_702_, lean_object* v_letArgs_703_, lean_object* v_a_704_, lean_object* v_a_705_, lean_object* v_a_706_, lean_object* v_a_707_){
_start:
{
if (lean_obj_tag(v_argKinds_699_) == 0)
{
lean_object* v___x_709_; lean_object* v___x_710_; uint8_t v___x_711_; 
lean_dec_ref(v_type_698_);
v___x_709_ = lean_array_get_size(v_letArgs_703_);
v___x_710_ = lean_unsigned_to_nat(0u);
v___x_711_ = lean_nat_dec_eq(v___x_709_, v___x_710_);
if (v___x_711_ == 0)
{
lean_object* v_proof_712_; lean_object* v___x_714_; uint8_t v_isShared_715_; uint8_t v_isSharedCheck_755_; 
v_proof_712_ = lean_ctor_get(v_cthm_697_, 1);
v_isSharedCheck_755_ = !lean_is_exclusive(v_cthm_697_);
if (v_isSharedCheck_755_ == 0)
{
lean_object* v_unused_756_; lean_object* v_unused_757_; 
v_unused_756_ = lean_ctor_get(v_cthm_697_, 2);
lean_dec(v_unused_756_);
v_unused_757_ = lean_ctor_get(v_cthm_697_, 0);
lean_dec(v_unused_757_);
v___x_714_ = v_cthm_697_;
v_isShared_715_ = v_isSharedCheck_755_;
goto v_resetjp_713_;
}
else
{
lean_inc(v_proof_712_);
lean_dec(v_cthm_697_);
v___x_714_ = lean_box(0);
v_isShared_715_ = v_isSharedCheck_755_;
goto v_resetjp_713_;
}
v_resetjp_713_:
{
uint8_t v___x_716_; lean_object* v___y_718_; lean_object* v___x_750_; uint8_t v___x_751_; 
v___x_716_ = 1;
v___x_750_ = l_Lean_mkAppN(v_proof_712_, v_args_702_);
lean_dec_ref(v_args_702_);
v___x_751_ = lean_nat_dec_lt(v___x_710_, v___x_709_);
if (v___x_751_ == 0)
{
lean_dec_ref(v_letArgs_703_);
v___y_718_ = v___x_750_;
goto v___jp_717_;
}
else
{
size_t v___x_752_; size_t v___x_753_; lean_object* v___x_754_; 
v___x_752_ = lean_usize_of_nat(v___x_709_);
v___x_753_ = ((size_t)0ULL);
v___x_754_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__0(v_letArgs_703_, v___x_752_, v___x_753_, v___x_750_);
lean_dec_ref(v_letArgs_703_);
v___y_718_ = v___x_754_;
goto v___jp_717_;
}
v___jp_717_:
{
uint8_t v___x_719_; lean_object* v___x_720_; 
v___x_719_ = 1;
v___x_720_ = l_Lean_Meta_mkLambdaFVars(v_params_701_, v___y_718_, v___x_711_, v___x_716_, v___x_711_, v___x_716_, v___x_719_, v_a_704_, v_a_705_, v_a_706_, v_a_707_);
lean_dec_ref(v_params_701_);
if (lean_obj_tag(v___x_720_) == 0)
{
lean_object* v_a_721_; lean_object* v___x_722_; 
v_a_721_ = lean_ctor_get(v___x_720_, 0);
lean_inc_n(v_a_721_, 2);
lean_dec_ref_known(v___x_720_, 1);
lean_inc(v_a_707_);
lean_inc_ref(v_a_706_);
lean_inc(v_a_705_);
lean_inc_ref(v_a_704_);
v___x_722_ = lean_infer_type(v_a_721_, v_a_704_, v_a_705_, v_a_706_, v_a_707_);
if (lean_obj_tag(v___x_722_) == 0)
{
lean_object* v_a_723_; lean_object* v___x_725_; uint8_t v_isShared_726_; uint8_t v_isSharedCheck_733_; 
v_a_723_ = lean_ctor_get(v___x_722_, 0);
v_isSharedCheck_733_ = !lean_is_exclusive(v___x_722_);
if (v_isSharedCheck_733_ == 0)
{
v___x_725_ = v___x_722_;
v_isShared_726_ = v_isSharedCheck_733_;
goto v_resetjp_724_;
}
else
{
lean_inc(v_a_723_);
lean_dec(v___x_722_);
v___x_725_ = lean_box(0);
v_isShared_726_ = v_isSharedCheck_733_;
goto v_resetjp_724_;
}
v_resetjp_724_:
{
lean_object* v___x_728_; 
if (v_isShared_715_ == 0)
{
lean_ctor_set(v___x_714_, 2, v_argKinds_x27_700_);
lean_ctor_set(v___x_714_, 1, v_a_721_);
lean_ctor_set(v___x_714_, 0, v_a_723_);
v___x_728_ = v___x_714_;
goto v_reusejp_727_;
}
else
{
lean_object* v_reuseFailAlloc_732_; 
v_reuseFailAlloc_732_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_732_, 0, v_a_723_);
lean_ctor_set(v_reuseFailAlloc_732_, 1, v_a_721_);
lean_ctor_set(v_reuseFailAlloc_732_, 2, v_argKinds_x27_700_);
v___x_728_ = v_reuseFailAlloc_732_;
goto v_reusejp_727_;
}
v_reusejp_727_:
{
lean_object* v___x_730_; 
if (v_isShared_726_ == 0)
{
lean_ctor_set(v___x_725_, 0, v___x_728_);
v___x_730_ = v___x_725_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v___x_728_);
v___x_730_ = v_reuseFailAlloc_731_;
goto v_reusejp_729_;
}
v_reusejp_729_:
{
return v___x_730_;
}
}
}
}
else
{
lean_object* v_a_734_; lean_object* v___x_736_; uint8_t v_isShared_737_; uint8_t v_isSharedCheck_741_; 
lean_dec(v_a_721_);
lean_del_object(v___x_714_);
lean_dec_ref(v_argKinds_x27_700_);
v_a_734_ = lean_ctor_get(v___x_722_, 0);
v_isSharedCheck_741_ = !lean_is_exclusive(v___x_722_);
if (v_isSharedCheck_741_ == 0)
{
v___x_736_ = v___x_722_;
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
else
{
lean_inc(v_a_734_);
lean_dec(v___x_722_);
v___x_736_ = lean_box(0);
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
v_resetjp_735_:
{
lean_object* v___x_739_; 
if (v_isShared_737_ == 0)
{
v___x_739_ = v___x_736_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_a_734_);
v___x_739_ = v_reuseFailAlloc_740_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
return v___x_739_;
}
}
}
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
lean_del_object(v___x_714_);
lean_dec_ref(v_argKinds_x27_700_);
v_a_742_ = lean_ctor_get(v___x_720_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_720_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_720_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v___x_720_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_747_; 
if (v_isShared_745_ == 0)
{
v___x_747_ = v___x_744_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_a_742_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
}
}
}
else
{
lean_object* v___x_758_; 
lean_dec_ref(v_letArgs_703_);
lean_dec_ref(v_args_702_);
lean_dec_ref(v_params_701_);
lean_dec_ref(v_argKinds_x27_700_);
v___x_758_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_758_, 0, v_cthm_697_);
return v___x_758_;
}
}
else
{
lean_object* v_head_759_; lean_object* v_tail_760_; lean_object* v___y_762_; lean_object* v___y_763_; lean_object* v___y_764_; lean_object* v___y_765_; uint8_t v___x_771_; 
v_head_759_ = lean_ctor_get(v_argKinds_699_, 0);
lean_inc(v_head_759_);
v_tail_760_ = lean_ctor_get(v_argKinds_699_, 1);
lean_inc(v_tail_760_);
lean_dec_ref_known(v_argKinds_699_, 2);
v___x_771_ = lean_unbox(v_head_759_);
switch(v___x_771_)
{
case 2:
{
v___y_762_ = v_a_704_;
v___y_763_ = v_a_705_;
v___y_764_ = v_a_706_;
v___y_765_ = v_a_707_;
goto v___jp_761_;
}
case 4:
{
v___y_762_ = v_a_704_;
v___y_763_ = v_a_705_;
v___y_764_ = v_a_706_;
v___y_765_ = v_a_707_;
goto v___jp_761_;
}
default: 
{
lean_object* v___x_772_; lean_object* v___x_773_; 
lean_dec(v_tail_760_);
lean_dec(v_head_759_);
lean_dec_ref(v_letArgs_703_);
lean_dec_ref(v_args_702_);
lean_dec_ref(v_params_701_);
lean_dec_ref(v_argKinds_x27_700_);
lean_dec_ref(v_type_698_);
lean_dec_ref(v_cthm_697_);
v___x_772_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__4, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__4_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__4);
v___x_773_ = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__1(v___x_772_, v_a_704_, v_a_705_, v_a_706_, v_a_707_);
return v___x_773_;
}
}
v___jp_761_:
{
lean_object* v___x_766_; lean_object* v___f_767_; lean_object* v___x_768_; uint8_t v___x_769_; lean_object* v___x_770_; 
v___x_766_ = lean_unsigned_to_nat(3u);
v___f_767_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___lam__0___boxed), 15, 8);
lean_closure_set(v___f_767_, 0, v___x_766_);
lean_closure_set(v___f_767_, 1, v_args_702_);
lean_closure_set(v___f_767_, 2, v_argKinds_x27_700_);
lean_closure_set(v___f_767_, 3, v_head_759_);
lean_closure_set(v___f_767_, 4, v_params_701_);
lean_closure_set(v___f_767_, 5, v_cthm_697_);
lean_closure_set(v___f_767_, 6, v_tail_760_);
lean_closure_set(v___f_767_, 7, v_letArgs_703_);
v___x_768_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___closed__0));
v___x_769_ = 0;
v___x_770_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__4___redArg(v_type_698_, v___x_768_, v___f_767_, v___x_769_, v___x_769_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
return v___x_770_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process___boxed(lean_object* v_cthm_774_, lean_object* v_type_775_, lean_object* v_argKinds_776_, lean_object* v_argKinds_x27_777_, lean_object* v_params_778_, lean_object* v_args_779_, lean_object* v_letArgs_780_, lean_object* v_a_781_, lean_object* v_a_782_, lean_object* v_a_783_, lean_object* v_a_784_, lean_object* v_a_785_){
_start:
{
lean_object* v_res_786_; 
v_res_786_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process(v_cthm_774_, v_type_775_, v_argKinds_776_, v_argKinds_x27_777_, v_params_778_, v_args_779_, v_letArgs_780_, v_a_781_, v_a_782_, v_a_783_, v_a_784_);
lean_dec(v_a_784_);
lean_dec_ref(v_a_783_);
lean_dec(v_a_782_);
lean_dec_ref(v_a_781_);
return v_res_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkHCongrWithArity_x27(lean_object* v_f_789_, lean_object* v_numArgs_790_, lean_object* v_a_791_, lean_object* v_a_792_, lean_object* v_a_793_, lean_object* v_a_794_){
_start:
{
lean_object* v___x_796_; 
v___x_796_ = l_Lean_Meta_mkHCongrWithArity(v_f_789_, v_numArgs_790_, v_a_791_, v_a_792_, v_a_793_, v_a_794_);
if (lean_obj_tag(v___x_796_) == 0)
{
lean_object* v_a_797_; lean_object* v_type_798_; lean_object* v_argKinds_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; 
v_a_797_ = lean_ctor_get(v___x_796_, 0);
lean_inc(v_a_797_);
lean_dec_ref_known(v___x_796_, 1);
v_type_798_ = lean_ctor_get(v_a_797_, 0);
lean_inc_ref(v_type_798_);
v_argKinds_799_ = lean_ctor_get(v_a_797_, 2);
lean_inc_ref(v_argKinds_799_);
v___x_800_ = lean_array_to_list(v_argKinds_799_);
v___x_801_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkHCongrWithArity_x27___closed__0));
v___x_802_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process(v_a_797_, v_type_798_, v___x_800_, v___x_801_, v___x_801_, v___x_801_, v___x_801_, v_a_791_, v_a_792_, v_a_793_, v_a_794_);
return v___x_802_;
}
else
{
return v___x_796_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkHCongrWithArity_x27___boxed(lean_object* v_f_803_, lean_object* v_numArgs_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_mathlib_Lean_Meta_mkHCongrWithArity_x27(v_f_803_, v_numArgs_804_, v_a_805_, v_a_806_, v_a_807_, v_a_808_);
lean_dec(v_a_808_);
lean_dec_ref(v_a_807_);
lean_dec(v_a_806_);
lean_dec_ref(v_a_805_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0(lean_object* v_x1_814_, lean_object* v_x2_815_){
_start:
{
lean_object* v_className_816_; lean_object* v___x_817_; uint8_t v___x_818_; 
v_className_816_ = lean_ctor_get(v_x2_815_, 0);
v___x_817_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__0___closed__1));
v___x_818_ = lean_name_eq(v_className_816_, v___x_817_);
if (v___x_818_ == 0)
{
lean_dec_ref(v_x2_815_);
return v_x1_814_;
}
else
{
lean_object* v___x_819_; 
v___x_819_ = lean_array_push(v_x1_814_, v_x2_815_);
return v___x_819_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__1(lean_object* v_x_820_, lean_object* v_x_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v___x_827_; 
v___x_827_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_827_, 0, v_x_820_);
return v___x_827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__1___boxed(lean_object* v_x_828_, lean_object* v_x_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_){
_start:
{
lean_object* v_res_835_; 
v_res_835_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__1(v_x_828_, v_x_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
lean_dec(v___y_831_);
lean_dec_ref(v___y_830_);
lean_dec_ref(v_x_829_);
return v_res_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2(lean_object* v_x_839_){
_start:
{
lean_object* v___f_840_; lean_object* v___x_841_; lean_object* v___x_842_; 
v___f_840_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__1___boxed), 7, 1);
lean_closure_set(v___f_840_, 0, v_x_839_);
v___x_841_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__2___closed__1));
v___x_842_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_842_, 0, v___x_841_);
lean_ctor_set(v___x_842_, 1, v___f_840_);
return v___x_842_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___closed__0(void){
_start:
{
lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; 
v___x_843_ = lean_box(0);
v___x_844_ = lean_unsigned_to_nat(2u);
v___x_845_ = lean_mk_empty_array_with_capacity(v___x_844_);
v___x_846_ = lean_array_push(v___x_845_, v___x_843_);
return v___x_846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3(lean_object* v_inst_847_, lean_object* v_f_848_, lean_object* v_args_849_, lean_object* v_x_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_){
_start:
{
lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_856_ = l_Lean_mkAppN(v_inst_847_, v_args_849_);
v___x_857_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_857_, 0, v___x_856_);
v___x_858_ = lean_obj_once(&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___closed__0, &lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___closed__0_once, _init_lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___closed__0);
v___x_859_ = lean_array_push(v___x_858_, v___x_857_);
v___x_860_ = l_Lean_Meta_mkAppOptM(v_f_848_, v___x_859_, v___y_851_, v___y_852_, v___y_853_, v___y_854_);
if (lean_obj_tag(v___x_860_) == 0)
{
lean_object* v_a_861_; uint8_t v___x_862_; uint8_t v___x_863_; uint8_t v___x_864_; lean_object* v___x_865_; 
v_a_861_ = lean_ctor_get(v___x_860_, 0);
lean_inc(v_a_861_);
lean_dec_ref_known(v___x_860_, 1);
v___x_862_ = 0;
v___x_863_ = 1;
v___x_864_ = 1;
v___x_865_ = l_Lean_Meta_mkLambdaFVars(v_args_849_, v_a_861_, v___x_862_, v___x_863_, v___x_862_, v___x_863_, v___x_864_, v___y_851_, v___y_852_, v___y_853_, v___y_854_);
return v___x_865_;
}
else
{
return v___x_860_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___boxed(lean_object* v_inst_866_, lean_object* v_f_867_, lean_object* v_args_868_, lean_object* v_x_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3(v_inst_866_, v_f_867_, v_args_868_, v_x_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_);
lean_dec(v___y_873_);
lean_dec_ref(v___y_872_);
lean_dec(v___y_871_);
lean_dec_ref(v___y_870_);
lean_dec_ref(v_x_869_);
lean_dec_ref(v_args_868_);
return v_res_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__4(lean_object* v___x_876_, lean_object* v___x_877_, lean_object* v_f_878_, lean_object* v_inst_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_){
_start:
{
lean_object* v___x_885_; 
lean_inc(v___y_883_);
lean_inc_ref(v___y_882_);
lean_inc(v___y_881_);
lean_inc_ref(v___y_880_);
lean_inc_ref(v_inst_879_);
v___x_885_ = lean_infer_type(v_inst_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_);
if (lean_obj_tag(v___x_885_) == 0)
{
lean_object* v_a_886_; lean_object* v___f_887_; uint8_t v___x_888_; lean_object* v___x_1499__overap_889_; lean_object* v___x_890_; 
v_a_886_ = lean_ctor_get(v___x_885_, 0);
lean_inc(v_a_886_);
lean_dec_ref_known(v___x_885_, 1);
v___f_887_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__3___boxed), 9, 2);
lean_closure_set(v___f_887_, 0, v_inst_879_);
lean_closure_set(v___f_887_, 1, v_f_878_);
v___x_888_ = 0;
v___x_1499__overap_889_ = l_Lean_Meta_forallTelescopeReducing___redArg(v___x_876_, v___x_877_, v_a_886_, v___f_887_, v___x_888_, v___x_888_);
lean_inc(v___y_883_);
lean_inc_ref(v___y_882_);
lean_inc(v___y_881_);
lean_inc_ref(v___y_880_);
v___x_890_ = lean_apply_5(v___x_1499__overap_889_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, lean_box(0));
return v___x_890_;
}
else
{
lean_dec_ref(v_inst_879_);
lean_dec(v_f_878_);
lean_dec_ref(v___x_877_);
lean_dec_ref(v___x_876_);
return v___x_885_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__4___boxed(lean_object* v___x_891_, lean_object* v___x_892_, lean_object* v_f_893_, lean_object* v_inst_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__4(v___x_891_, v___x_892_, v_f_893_, v_inst_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_);
lean_dec(v___y_898_);
lean_dec_ref(v___y_897_);
lean_dec(v___y_896_);
lean_dec_ref(v___y_895_);
return v_res_900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5(lean_object* v___f_908_, lean_object* v_inst_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_){
_start:
{
lean_object* v_fvar_915_; lean_object* v___x_916_; lean_object* v___x_917_; 
v_fvar_915_ = lean_ctor_get(v_inst_909_, 1);
lean_inc_ref(v_fvar_915_);
lean_dec_ref(v_inst_909_);
v___x_916_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___closed__2));
lean_inc(v___y_913_);
lean_inc_ref(v___y_912_);
lean_inc(v___y_911_);
lean_inc_ref(v___y_910_);
v___x_917_ = lean_apply_7(v___f_908_, v___x_916_, v_fvar_915_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, lean_box(0));
return v___x_917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___boxed(lean_object* v___f_918_, lean_object* v_inst_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_){
_start:
{
lean_object* v_res_925_; 
v_res_925_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5(v___f_918_, v_inst_919_, v___y_920_, v___y_921_, v___y_922_, v___y_923_);
lean_dec(v___y_923_);
lean_dec_ref(v___y_922_);
lean_dec(v___y_921_);
lean_dec_ref(v___y_920_);
return v_res_925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6(lean_object* v___f_932_, lean_object* v_inst_933_, lean_object* v___y_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_){
_start:
{
lean_object* v_fvar_939_; lean_object* v___x_940_; lean_object* v___x_941_; 
v_fvar_939_ = lean_ctor_get(v_inst_933_, 1);
lean_inc_ref(v_fvar_939_);
lean_dec_ref(v_inst_933_);
v___x_940_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___closed__1));
lean_inc(v___y_937_);
lean_inc_ref(v___y_936_);
lean_inc(v___y_935_);
lean_inc_ref(v___y_934_);
v___x_941_ = lean_apply_7(v___f_932_, v___x_940_, v_fvar_939_, v___y_934_, v___y_935_, v___y_936_, v___y_937_, lean_box(0));
return v___x_941_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___boxed(lean_object* v___f_942_, lean_object* v_inst_943_, lean_object* v___y_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_){
_start:
{
lean_object* v_res_949_; 
v_res_949_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6(v___f_942_, v_inst_943_, v___y_944_, v___y_945_, v___y_946_, v___y_947_);
lean_dec(v___y_947_);
lean_dec_ref(v___y_946_);
lean_dec(v___y_945_);
lean_dec_ref(v___y_944_);
return v_res_949_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__7(lean_object* v_args_950_, lean_object* v___x_951_, lean_object* v_e_952_){
_start:
{
lean_object* v___x_953_; 
v___x_953_ = l_Lean_Expr_replaceFVars(v_e_952_, v_args_950_, v___x_951_);
return v___x_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__7___boxed(lean_object* v_args_954_, lean_object* v___x_955_, lean_object* v_e_956_){
_start:
{
lean_object* v_res_957_; 
v_res_957_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__7(v_args_954_, v___x_955_, v_e_956_);
lean_dec_ref(v_e_956_);
lean_dec_ref(v___x_955_);
lean_dec_ref(v_args_954_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__8(lean_object* v___x_958_, lean_object* v_mx_959_, lean_object* v___x_960_, lean_object* v___x_961_, lean_object* v___x_962_, lean_object* v_args_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_){
_start:
{
lean_object* v___f_969_; lean_object* v___x_970_; lean_object* v___x_1539__overap_971_; lean_object* v___x_972_; 
lean_inc_ref(v_args_963_);
v___f_969_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__7___boxed), 3, 2);
lean_closure_set(v___f_969_, 0, v_args_963_);
lean_closure_set(v___f_969_, 1, v___x_958_);
v___x_970_ = lean_apply_1(v_mx_959_, v___f_969_);
v___x_1539__overap_971_ = l_Lean_Meta_withNewLocalInstances___redArg(v___x_960_, v___x_961_, v_args_963_, v___x_962_, v___x_970_);
lean_inc(v___y_967_);
lean_inc_ref(v___y_966_);
lean_inc(v___y_965_);
lean_inc_ref(v___y_964_);
v___x_972_ = lean_apply_5(v___x_1539__overap_971_, v___y_964_, v___y_965_, v___y_966_, v___y_967_, lean_box(0));
return v___x_972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__8___boxed(lean_object* v___x_973_, lean_object* v_mx_974_, lean_object* v___x_975_, lean_object* v___x_976_, lean_object* v___x_977_, lean_object* v_args_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_){
_start:
{
lean_object* v_res_984_; 
v_res_984_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__8(v___x_973_, v_mx_974_, v___x_975_, v___x_976_, v___x_977_, v_args_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
return v_res_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9(lean_object* v_x1_988_, lean_object* v_x2_989_){
_start:
{
lean_object* v_className_990_; lean_object* v___x_991_; uint8_t v___x_992_; 
v_className_990_ = lean_ctor_get(v_x2_989_, 0);
v___x_991_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__9___closed__1));
v___x_992_ = lean_name_eq(v_className_990_, v___x_991_);
if (v___x_992_ == 0)
{
lean_dec_ref(v_x2_989_);
return v_x1_988_;
}
else
{
lean_object* v___x_993_; 
v___x_993_ = lean_array_push(v_x1_988_, v_x2_989_);
return v___x_993_;
}
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__0(void){
_start:
{
lean_object* v___x_994_; 
v___x_994_ = l_instMonadEIO(lean_box(0));
return v___x_994_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__1(void){
_start:
{
lean_object* v___x_995_; lean_object* v___x_996_; 
v___x_995_ = lean_obj_once(&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__0, &lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__0_once, _init_lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__0);
v___x_996_ = l_StateRefT_x27_instMonad___redArg(v___x_995_);
return v___x_996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg(lean_object* v_mx_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_){
_start:
{
lean_object* v___x_1032_; lean_object* v_toApplicative_1033_; lean_object* v_toFunctor_1034_; lean_object* v_toSeq_1035_; lean_object* v_toSeqLeft_1036_; lean_object* v_toSeqRight_1037_; lean_object* v___f_1038_; lean_object* v___f_1039_; lean_object* v___f_1040_; lean_object* v___f_1041_; lean_object* v___x_1042_; lean_object* v___f_1043_; lean_object* v___f_1044_; lean_object* v___f_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v_toApplicative_1049_; lean_object* v___x_1051_; uint8_t v_isShared_1052_; uint8_t v_isSharedCheck_1173_; 
v___x_1032_ = lean_obj_once(&lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__1, &lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__1);
v_toApplicative_1033_ = lean_ctor_get(v___x_1032_, 0);
v_toFunctor_1034_ = lean_ctor_get(v_toApplicative_1033_, 0);
v_toSeq_1035_ = lean_ctor_get(v_toApplicative_1033_, 2);
v_toSeqLeft_1036_ = lean_ctor_get(v_toApplicative_1033_, 3);
v_toSeqRight_1037_ = lean_ctor_get(v_toApplicative_1033_, 4);
v___f_1038_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__2));
v___f_1039_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_1034_, 2);
v___f_1040_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1040_, 0, v_toFunctor_1034_);
v___f_1041_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1041_, 0, v_toFunctor_1034_);
v___x_1042_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1042_, 0, v___f_1040_);
lean_ctor_set(v___x_1042_, 1, v___f_1041_);
lean_inc(v_toSeqRight_1037_);
v___f_1043_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1043_, 0, v_toSeqRight_1037_);
lean_inc(v_toSeqLeft_1036_);
v___f_1044_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1044_, 0, v_toSeqLeft_1036_);
lean_inc(v_toSeq_1035_);
v___f_1045_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1045_, 0, v_toSeq_1035_);
v___x_1046_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1046_, 0, v___x_1042_);
lean_ctor_set(v___x_1046_, 1, v___f_1038_);
lean_ctor_set(v___x_1046_, 2, v___f_1045_);
lean_ctor_set(v___x_1046_, 3, v___f_1044_);
lean_ctor_set(v___x_1046_, 4, v___f_1043_);
v___x_1047_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1047_, 0, v___x_1046_);
lean_ctor_set(v___x_1047_, 1, v___f_1039_);
v___x_1048_ = l_StateRefT_x27_instMonad___redArg(v___x_1047_);
v_toApplicative_1049_ = lean_ctor_get(v___x_1048_, 0);
v_isSharedCheck_1173_ = !lean_is_exclusive(v___x_1048_);
if (v_isSharedCheck_1173_ == 0)
{
lean_object* v_unused_1174_; 
v_unused_1174_ = lean_ctor_get(v___x_1048_, 1);
lean_dec(v_unused_1174_);
v___x_1051_ = v___x_1048_;
v_isShared_1052_ = v_isSharedCheck_1173_;
goto v_resetjp_1050_;
}
else
{
lean_inc(v_toApplicative_1049_);
lean_dec(v___x_1048_);
v___x_1051_ = lean_box(0);
v_isShared_1052_ = v_isSharedCheck_1173_;
goto v_resetjp_1050_;
}
v_resetjp_1050_:
{
lean_object* v_toFunctor_1053_; lean_object* v_toSeq_1054_; lean_object* v_toSeqLeft_1055_; lean_object* v_toSeqRight_1056_; lean_object* v___x_1058_; uint8_t v_isShared_1059_; uint8_t v_isSharedCheck_1171_; 
v_toFunctor_1053_ = lean_ctor_get(v_toApplicative_1049_, 0);
v_toSeq_1054_ = lean_ctor_get(v_toApplicative_1049_, 2);
v_toSeqLeft_1055_ = lean_ctor_get(v_toApplicative_1049_, 3);
v_toSeqRight_1056_ = lean_ctor_get(v_toApplicative_1049_, 4);
v_isSharedCheck_1171_ = !lean_is_exclusive(v_toApplicative_1049_);
if (v_isSharedCheck_1171_ == 0)
{
lean_object* v_unused_1172_; 
v_unused_1172_ = lean_ctor_get(v_toApplicative_1049_, 1);
lean_dec(v_unused_1172_);
v___x_1058_ = v_toApplicative_1049_;
v_isShared_1059_ = v_isSharedCheck_1171_;
goto v_resetjp_1057_;
}
else
{
lean_inc(v_toSeqRight_1056_);
lean_inc(v_toSeqLeft_1055_);
lean_inc(v_toSeq_1054_);
lean_inc(v_toFunctor_1053_);
lean_dec(v_toApplicative_1049_);
v___x_1058_ = lean_box(0);
v_isShared_1059_ = v_isSharedCheck_1171_;
goto v_resetjp_1057_;
}
v_resetjp_1057_:
{
lean_object* v___f_1060_; lean_object* v___f_1061_; lean_object* v___f_1062_; lean_object* v___f_1063_; lean_object* v___x_1064_; lean_object* v___f_1065_; lean_object* v___f_1066_; lean_object* v___f_1067_; lean_object* v___x_1069_; 
v___f_1060_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__4));
v___f_1061_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__5));
lean_inc_ref(v_toFunctor_1053_);
v___f_1062_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1062_, 0, v_toFunctor_1053_);
v___f_1063_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1063_, 0, v_toFunctor_1053_);
v___x_1064_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1064_, 0, v___f_1062_);
lean_ctor_set(v___x_1064_, 1, v___f_1063_);
v___f_1065_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1065_, 0, v_toSeqRight_1056_);
v___f_1066_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1066_, 0, v_toSeqLeft_1055_);
v___f_1067_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1067_, 0, v_toSeq_1054_);
if (v_isShared_1059_ == 0)
{
lean_ctor_set(v___x_1058_, 4, v___f_1065_);
lean_ctor_set(v___x_1058_, 3, v___f_1066_);
lean_ctor_set(v___x_1058_, 2, v___f_1067_);
lean_ctor_set(v___x_1058_, 1, v___f_1060_);
lean_ctor_set(v___x_1058_, 0, v___x_1064_);
v___x_1069_ = v___x_1058_;
goto v_reusejp_1068_;
}
else
{
lean_object* v_reuseFailAlloc_1170_; 
v_reuseFailAlloc_1170_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1170_, 0, v___x_1064_);
lean_ctor_set(v_reuseFailAlloc_1170_, 1, v___f_1060_);
lean_ctor_set(v_reuseFailAlloc_1170_, 2, v___f_1067_);
lean_ctor_set(v_reuseFailAlloc_1170_, 3, v___f_1066_);
lean_ctor_set(v_reuseFailAlloc_1170_, 4, v___f_1065_);
v___x_1069_ = v_reuseFailAlloc_1170_;
goto v_reusejp_1068_;
}
v_reusejp_1068_:
{
lean_object* v___x_1071_; 
if (v_isShared_1052_ == 0)
{
lean_ctor_set(v___x_1051_, 1, v___f_1061_);
lean_ctor_set(v___x_1051_, 0, v___x_1069_);
v___x_1071_ = v___x_1051_;
goto v_reusejp_1070_;
}
else
{
lean_object* v_reuseFailAlloc_1169_; 
v_reuseFailAlloc_1169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1169_, 0, v___x_1069_);
lean_ctor_set(v_reuseFailAlloc_1169_, 1, v___f_1061_);
v___x_1071_ = v_reuseFailAlloc_1169_;
goto v_reusejp_1070_;
}
v_reusejp_1070_:
{
lean_object* v_toApplicative_1072_; lean_object* v_toFunctor_1073_; lean_object* v_toSeq_1074_; lean_object* v_toSeqLeft_1075_; lean_object* v_toSeqRight_1076_; lean_object* v___f_1077_; lean_object* v___f_1078_; lean_object* v___x_1079_; lean_object* v___f_1080_; lean_object* v___f_1081_; lean_object* v___f_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v_localInstances_1088_; lean_object* v___f_1089_; lean_object* v___f_1090_; lean_object* v___f_1091_; lean_object* v___f_1092_; lean_object* v___f_1093_; lean_object* v___x_1094_; lean_object* v___y_1096_; lean_object* v___y_1097_; lean_object* v___y_1145_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; uint8_t v___x_1160_; 
v_toApplicative_1072_ = lean_ctor_get(v___x_1032_, 0);
v_toFunctor_1073_ = lean_ctor_get(v_toApplicative_1072_, 0);
v_toSeq_1074_ = lean_ctor_get(v_toApplicative_1072_, 2);
v_toSeqLeft_1075_ = lean_ctor_get(v_toApplicative_1072_, 3);
v_toSeqRight_1076_ = lean_ctor_get(v_toApplicative_1072_, 4);
lean_inc_ref_n(v_toFunctor_1073_, 2);
v___f_1077_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1077_, 0, v_toFunctor_1073_);
v___f_1078_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1078_, 0, v_toFunctor_1073_);
v___x_1079_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1079_, 0, v___f_1077_);
lean_ctor_set(v___x_1079_, 1, v___f_1078_);
lean_inc(v_toSeqRight_1076_);
v___f_1080_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1080_, 0, v_toSeqRight_1076_);
lean_inc(v_toSeqLeft_1075_);
v___f_1081_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1081_, 0, v_toSeqLeft_1075_);
lean_inc(v_toSeq_1074_);
v___f_1082_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1082_, 0, v_toSeq_1074_);
v___x_1083_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1083_, 0, v___x_1079_);
lean_ctor_set(v___x_1083_, 1, v___f_1038_);
lean_ctor_set(v___x_1083_, 2, v___f_1082_);
lean_ctor_set(v___x_1083_, 3, v___f_1081_);
lean_ctor_set(v___x_1083_, 4, v___f_1080_);
v___x_1084_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1084_, 0, v___x_1083_);
lean_ctor_set(v___x_1084_, 1, v___f_1039_);
v___x_1085_ = l_StateRefT_x27_instMonad___redArg(v___x_1084_);
v___x_1086_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_1086_, 0, lean_box(0));
lean_closure_set(v___x_1086_, 1, lean_box(0));
lean_closure_set(v___x_1086_, 2, v___x_1085_);
v___x_1087_ = l_instMonadControlTOfPure___redArg(v___x_1086_);
v_localInstances_1088_ = lean_ctor_get(v_a_1027_, 3);
v___f_1089_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__6));
v___f_1090_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__7));
lean_inc_ref(v___x_1071_);
lean_inc_ref(v___x_1087_);
v___f_1091_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__4___boxed), 9, 2);
lean_closure_set(v___f_1091_, 0, v___x_1087_);
lean_closure_set(v___f_1091_, 1, v___x_1071_);
lean_inc_ref(v___f_1091_);
v___f_1092_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__5___boxed), 7, 1);
lean_closure_set(v___f_1092_, 0, v___f_1091_);
v___f_1093_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__6___boxed), 7, 1);
lean_closure_set(v___f_1093_, 0, v___f_1091_);
v___x_1094_ = lean_unsigned_to_nat(0u);
v___x_1157_ = lean_array_get_size(v_localInstances_1088_);
v___x_1158_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__19));
v___x_1159_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__18));
v___x_1160_ = lean_nat_dec_lt(v___x_1094_, v___x_1157_);
if (v___x_1160_ == 0)
{
v___y_1145_ = v___x_1158_;
goto v___jp_1144_;
}
else
{
lean_object* v___f_1161_; uint8_t v___x_1162_; 
v___f_1161_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__20));
v___x_1162_ = lean_nat_dec_le(v___x_1157_, v___x_1157_);
if (v___x_1162_ == 0)
{
if (v___x_1160_ == 0)
{
v___y_1145_ = v___x_1158_;
goto v___jp_1144_;
}
else
{
size_t v___x_1163_; size_t v___x_1164_; lean_object* v___x_1165_; 
v___x_1163_ = ((size_t)0ULL);
v___x_1164_ = lean_usize_of_nat(v___x_1157_);
lean_inc_ref(v_localInstances_1088_);
v___x_1165_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1159_, v___f_1161_, v_localInstances_1088_, v___x_1163_, v___x_1164_, v___x_1158_);
v___y_1145_ = v___x_1165_;
goto v___jp_1144_;
}
}
else
{
size_t v___x_1166_; size_t v___x_1167_; lean_object* v___x_1168_; 
v___x_1166_ = ((size_t)0ULL);
v___x_1167_ = lean_usize_of_nat(v___x_1157_);
lean_inc_ref(v_localInstances_1088_);
v___x_1168_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1159_, v___f_1161_, v_localInstances_1088_, v___x_1166_, v___x_1167_, v___x_1158_);
v___y_1145_ = v___x_1168_;
goto v___jp_1144_;
}
}
v___jp_1095_:
{
size_t v_sz_1098_; size_t v___x_1099_; lean_object* v___x_1362__overap_1100_; lean_object* v___x_1101_; 
v_sz_1098_ = lean_array_size(v___y_1096_);
v___x_1099_ = ((size_t)0ULL);
lean_inc_ref(v___x_1071_);
v___x_1362__overap_1100_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_1071_, v___f_1092_, v_sz_1098_, v___x_1099_, v___y_1096_);
lean_inc(v_a_1030_);
lean_inc_ref(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc_ref(v_a_1027_);
v___x_1101_ = lean_apply_5(v___x_1362__overap_1100_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, lean_box(0));
if (lean_obj_tag(v___x_1101_) == 0)
{
lean_object* v_a_1102_; size_t v_sz_1103_; lean_object* v___x_1365__overap_1104_; lean_object* v___x_1105_; 
v_a_1102_ = lean_ctor_get(v___x_1101_, 0);
lean_inc(v_a_1102_);
lean_dec_ref_known(v___x_1101_, 1);
v_sz_1103_ = lean_array_size(v___y_1097_);
lean_inc_ref(v___x_1071_);
v___x_1365__overap_1104_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_1071_, v___f_1093_, v_sz_1103_, v___x_1099_, v___y_1097_);
lean_inc(v_a_1030_);
lean_inc_ref(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc_ref(v_a_1027_);
v___x_1105_ = lean_apply_5(v___x_1365__overap_1104_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, lean_box(0));
if (lean_obj_tag(v___x_1105_) == 0)
{
lean_object* v_a_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; size_t v_sz_1109_; lean_object* v___x_1368__overap_1110_; lean_object* v___x_1111_; 
v_a_1106_ = lean_ctor_get(v___x_1105_, 0);
lean_inc(v_a_1106_);
lean_dec_ref_known(v___x_1105_, 1);
v___x_1107_ = l_Array_append___redArg(v_a_1102_, v_a_1106_);
lean_dec(v_a_1106_);
v___x_1108_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__8));
v_sz_1109_ = lean_array_size(v___x_1107_);
lean_inc_ref(v___x_1107_);
lean_inc_ref(v___x_1071_);
v___x_1368__overap_1110_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_1071_, v___x_1108_, v_sz_1109_, v___x_1099_, v___x_1107_);
lean_inc(v_a_1030_);
lean_inc_ref(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc_ref(v_a_1027_);
v___x_1111_ = lean_apply_5(v___x_1368__overap_1110_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, lean_box(0));
if (lean_obj_tag(v___x_1111_) == 0)
{
lean_object* v_a_1112_; lean_object* v___f_1113_; lean_object* v___x_1114_; size_t v_sz_1115_; lean_object* v___x_1116_; uint8_t v___x_1117_; lean_object* v___x_1319__overap_1118_; lean_object* v___x_1119_; 
v_a_1112_ = lean_ctor_get(v___x_1111_, 0);
lean_inc(v_a_1112_);
lean_dec_ref_known(v___x_1111_, 1);
lean_inc_ref(v___x_1071_);
lean_inc_ref(v___x_1087_);
v___f_1113_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___lam__8___boxed), 11, 5);
lean_closure_set(v___f_1113_, 0, v___x_1107_);
lean_closure_set(v___f_1113_, 1, v_mx_1026_);
lean_closure_set(v___f_1113_, 2, v___x_1087_);
lean_closure_set(v___f_1113_, 3, v___x_1071_);
lean_closure_set(v___f_1113_, 4, v___x_1094_);
v___x_1114_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__18));
v_sz_1115_ = lean_array_size(v_a_1112_);
v___x_1116_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_1114_, v___f_1090_, v_sz_1115_, v___x_1099_, v_a_1112_);
v___x_1117_ = 0;
v___x_1319__overap_1118_ = l_Lean_Meta_withLocalDeclsD___redArg(v___x_1087_, v___x_1071_, v___x_1116_, v___f_1113_, v___x_1117_);
lean_inc(v_a_1030_);
lean_inc_ref(v_a_1029_);
lean_inc(v_a_1028_);
lean_inc_ref(v_a_1027_);
v___x_1119_ = lean_apply_5(v___x_1319__overap_1118_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, lean_box(0));
return v___x_1119_;
}
else
{
lean_object* v_a_1120_; lean_object* v___x_1122_; uint8_t v_isShared_1123_; uint8_t v_isSharedCheck_1127_; 
lean_dec_ref(v___x_1107_);
lean_dec_ref(v___x_1087_);
lean_dec_ref(v___x_1071_);
lean_dec_ref(v_mx_1026_);
v_a_1120_ = lean_ctor_get(v___x_1111_, 0);
v_isSharedCheck_1127_ = !lean_is_exclusive(v___x_1111_);
if (v_isSharedCheck_1127_ == 0)
{
v___x_1122_ = v___x_1111_;
v_isShared_1123_ = v_isSharedCheck_1127_;
goto v_resetjp_1121_;
}
else
{
lean_inc(v_a_1120_);
lean_dec(v___x_1111_);
v___x_1122_ = lean_box(0);
v_isShared_1123_ = v_isSharedCheck_1127_;
goto v_resetjp_1121_;
}
v_resetjp_1121_:
{
lean_object* v___x_1125_; 
if (v_isShared_1123_ == 0)
{
v___x_1125_ = v___x_1122_;
goto v_reusejp_1124_;
}
else
{
lean_object* v_reuseFailAlloc_1126_; 
v_reuseFailAlloc_1126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1126_, 0, v_a_1120_);
v___x_1125_ = v_reuseFailAlloc_1126_;
goto v_reusejp_1124_;
}
v_reusejp_1124_:
{
return v___x_1125_;
}
}
}
}
else
{
lean_object* v_a_1128_; lean_object* v___x_1130_; uint8_t v_isShared_1131_; uint8_t v_isSharedCheck_1135_; 
lean_dec(v_a_1102_);
lean_dec_ref(v___x_1087_);
lean_dec_ref(v___x_1071_);
lean_dec_ref(v_mx_1026_);
v_a_1128_ = lean_ctor_get(v___x_1105_, 0);
v_isSharedCheck_1135_ = !lean_is_exclusive(v___x_1105_);
if (v_isSharedCheck_1135_ == 0)
{
v___x_1130_ = v___x_1105_;
v_isShared_1131_ = v_isSharedCheck_1135_;
goto v_resetjp_1129_;
}
else
{
lean_inc(v_a_1128_);
lean_dec(v___x_1105_);
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
lean_dec_ref(v___y_1097_);
lean_dec_ref(v___f_1093_);
lean_dec_ref(v___x_1087_);
lean_dec_ref(v___x_1071_);
lean_dec_ref(v_mx_1026_);
v_a_1136_ = lean_ctor_get(v___x_1101_, 0);
v_isSharedCheck_1143_ = !lean_is_exclusive(v___x_1101_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1138_ = v___x_1101_;
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
else
{
lean_inc(v_a_1136_);
lean_dec(v___x_1101_);
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
v___jp_1144_:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; uint8_t v___x_1149_; 
v___x_1146_ = lean_array_get_size(v_localInstances_1088_);
v___x_1147_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__19));
v___x_1148_ = ((lean_object*)(lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___closed__18));
v___x_1149_ = lean_nat_dec_lt(v___x_1094_, v___x_1146_);
if (v___x_1149_ == 0)
{
v___y_1096_ = v___y_1145_;
v___y_1097_ = v___x_1147_;
goto v___jp_1095_;
}
else
{
uint8_t v___x_1150_; 
v___x_1150_ = lean_nat_dec_le(v___x_1146_, v___x_1146_);
if (v___x_1150_ == 0)
{
if (v___x_1149_ == 0)
{
v___y_1096_ = v___y_1145_;
v___y_1097_ = v___x_1147_;
goto v___jp_1095_;
}
else
{
size_t v___x_1151_; size_t v___x_1152_; lean_object* v___x_1153_; 
v___x_1151_ = ((size_t)0ULL);
v___x_1152_ = lean_usize_of_nat(v___x_1146_);
lean_inc_ref(v_localInstances_1088_);
v___x_1153_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1148_, v___f_1089_, v_localInstances_1088_, v___x_1151_, v___x_1152_, v___x_1147_);
v___y_1096_ = v___y_1145_;
v___y_1097_ = v___x_1153_;
goto v___jp_1095_;
}
}
else
{
size_t v___x_1154_; size_t v___x_1155_; lean_object* v___x_1156_; 
v___x_1154_ = ((size_t)0ULL);
v___x_1155_ = lean_usize_of_nat(v___x_1146_);
lean_inc_ref(v_localInstances_1088_);
v___x_1156_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1148_, v___f_1089_, v_localInstances_1088_, v___x_1154_, v___x_1155_, v___x_1147_);
v___y_1096_ = v___y_1145_;
v___y_1097_ = v___x_1156_;
goto v___jp_1095_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg___boxed(lean_object* v_mx_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_){
_start:
{
lean_object* v_res_1181_; 
v_res_1181_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg(v_mx_1175_, v_a_1176_, v_a_1177_, v_a_1178_, v_a_1179_);
lean_dec(v_a_1179_);
lean_dec_ref(v_a_1178_);
lean_dec(v_a_1177_);
lean_dec_ref(v_a_1176_);
return v_res_1181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast(lean_object* v_00_u03b1_1182_, lean_object* v_inst_1183_, lean_object* v_mx_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_, lean_object* v_a_1188_){
_start:
{
lean_object* v___x_1190_; 
v___x_1190_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg(v_mx_1184_, v_a_1185_, v_a_1186_, v_a_1187_, v_a_1188_);
return v___x_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withSubsingletonAsFast___boxed(lean_object* v_00_u03b1_1191_, lean_object* v_inst_1192_, lean_object* v_mx_1193_, lean_object* v_a_1194_, lean_object* v_a_1195_, lean_object* v_a_1196_, lean_object* v_a_1197_, lean_object* v_a_1198_){
_start:
{
lean_object* v_res_1199_; 
v_res_1199_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast(v_00_u03b1_1191_, v_inst_1192_, v_mx_1193_, v_a_1194_, v_a_1195_, v_a_1196_, v_a_1197_);
lean_dec(v_a_1197_);
lean_dec_ref(v_a_1196_);
lean_dec(v_a_1195_);
lean_dec_ref(v_a_1194_);
lean_dec(v_inst_1192_);
return v_res_1199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___redArg(lean_object* v_mvarId_1200_, lean_object* v_x_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_){
_start:
{
lean_object* v___x_1207_; 
v___x_1207_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1200_, v_x_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
if (lean_obj_tag(v___x_1207_) == 0)
{
lean_object* v_a_1208_; lean_object* v___x_1210_; uint8_t v_isShared_1211_; uint8_t v_isSharedCheck_1215_; 
v_a_1208_ = lean_ctor_get(v___x_1207_, 0);
v_isSharedCheck_1215_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1215_ == 0)
{
v___x_1210_ = v___x_1207_;
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
else
{
lean_inc(v_a_1208_);
lean_dec(v___x_1207_);
v___x_1210_ = lean_box(0);
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
v_resetjp_1209_:
{
lean_object* v___x_1213_; 
if (v_isShared_1211_ == 0)
{
v___x_1213_ = v___x_1210_;
goto v_reusejp_1212_;
}
else
{
lean_object* v_reuseFailAlloc_1214_; 
v_reuseFailAlloc_1214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1214_, 0, v_a_1208_);
v___x_1213_ = v_reuseFailAlloc_1214_;
goto v_reusejp_1212_;
}
v_reusejp_1212_:
{
return v___x_1213_;
}
}
}
else
{
lean_object* v_a_1216_; lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1223_; 
v_a_1216_ = lean_ctor_get(v___x_1207_, 0);
v_isSharedCheck_1223_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1218_ = v___x_1207_;
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
else
{
lean_inc(v_a_1216_);
lean_dec(v___x_1207_);
v___x_1218_ = lean_box(0);
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
v_resetjp_1217_:
{
lean_object* v___x_1221_; 
if (v_isShared_1219_ == 0)
{
v___x_1221_ = v___x_1218_;
goto v_reusejp_1220_;
}
else
{
lean_object* v_reuseFailAlloc_1222_; 
v_reuseFailAlloc_1222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1222_, 0, v_a_1216_);
v___x_1221_ = v_reuseFailAlloc_1222_;
goto v_reusejp_1220_;
}
v_reusejp_1220_:
{
return v___x_1221_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___redArg___boxed(lean_object* v_mvarId_1224_, lean_object* v_x_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_){
_start:
{
lean_object* v_res_1231_; 
v_res_1231_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___redArg(v_mvarId_1224_, v_x_1225_, v___y_1226_, v___y_1227_, v___y_1228_, v___y_1229_);
lean_dec(v___y_1229_);
lean_dec_ref(v___y_1228_);
lean_dec(v___y_1227_);
lean_dec_ref(v___y_1226_);
return v_res_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1(lean_object* v_00_u03b1_1232_, lean_object* v_mvarId_1233_, lean_object* v_x_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
lean_object* v___x_1240_; 
v___x_1240_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___redArg(v_mvarId_1233_, v_x_1234_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
return v___x_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___boxed(lean_object* v_00_u03b1_1241_, lean_object* v_mvarId_1242_, lean_object* v_x_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_){
_start:
{
lean_object* v_res_1249_; 
v_res_1249_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1(v_00_u03b1_1241_, v_mvarId_1242_, v_x_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_);
lean_dec(v___y_1247_);
lean_dec_ref(v___y_1246_);
lean_dec(v___y_1245_);
lean_dec_ref(v___y_1244_);
return v_res_1249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0(lean_object* v___x_1256_, lean_object* v___x_1257_, lean_object* v_elim_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_){
_start:
{
lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; 
v___x_1264_ = ((lean_object*)(lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___closed__1));
v___x_1265_ = lean_unsigned_to_nat(2u);
v___x_1266_ = lean_mk_empty_array_with_capacity(v___x_1265_);
v___x_1267_ = lean_array_push(v___x_1266_, v___x_1256_);
v___x_1268_ = lean_array_push(v___x_1267_, v___x_1257_);
v___x_1269_ = l_Lean_Meta_mkAppM(v___x_1264_, v___x_1268_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_);
if (lean_obj_tag(v___x_1269_) == 0)
{
lean_object* v_a_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1278_; 
v_a_1270_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1278_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1278_ == 0)
{
v___x_1272_ = v___x_1269_;
v_isShared_1273_ = v_isSharedCheck_1278_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_a_1270_);
lean_dec(v___x_1269_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1278_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1274_; lean_object* v___x_1276_; 
v___x_1274_ = lean_apply_1(v_elim_1258_, v_a_1270_);
if (v_isShared_1273_ == 0)
{
lean_ctor_set(v___x_1272_, 0, v___x_1274_);
v___x_1276_ = v___x_1272_;
goto v_reusejp_1275_;
}
else
{
lean_object* v_reuseFailAlloc_1277_; 
v_reuseFailAlloc_1277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1277_, 0, v___x_1274_);
v___x_1276_ = v_reuseFailAlloc_1277_;
goto v_reusejp_1275_;
}
v_reusejp_1275_:
{
return v___x_1276_;
}
}
}
else
{
lean_dec_ref(v_elim_1258_);
return v___x_1269_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___boxed(lean_object* v___x_1279_, lean_object* v___x_1280_, lean_object* v_elim_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_){
_start:
{
lean_object* v_res_1287_; 
v_res_1287_ = lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0(v___x_1279_, v___x_1280_, v_elim_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
return v_res_1287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(lean_object* v_x_1288_, lean_object* v_x_1289_, lean_object* v_x_1290_, lean_object* v_x_1291_){
_start:
{
lean_object* v_ks_1292_; lean_object* v_vs_1293_; lean_object* v___x_1295_; uint8_t v_isShared_1296_; uint8_t v_isSharedCheck_1317_; 
v_ks_1292_ = lean_ctor_get(v_x_1288_, 0);
v_vs_1293_ = lean_ctor_get(v_x_1288_, 1);
v_isSharedCheck_1317_ = !lean_is_exclusive(v_x_1288_);
if (v_isSharedCheck_1317_ == 0)
{
v___x_1295_ = v_x_1288_;
v_isShared_1296_ = v_isSharedCheck_1317_;
goto v_resetjp_1294_;
}
else
{
lean_inc(v_vs_1293_);
lean_inc(v_ks_1292_);
lean_dec(v_x_1288_);
v___x_1295_ = lean_box(0);
v_isShared_1296_ = v_isSharedCheck_1317_;
goto v_resetjp_1294_;
}
v_resetjp_1294_:
{
lean_object* v___x_1297_; uint8_t v___x_1298_; 
v___x_1297_ = lean_array_get_size(v_ks_1292_);
v___x_1298_ = lean_nat_dec_lt(v_x_1289_, v___x_1297_);
if (v___x_1298_ == 0)
{
lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1302_; 
lean_dec(v_x_1289_);
v___x_1299_ = lean_array_push(v_ks_1292_, v_x_1290_);
v___x_1300_ = lean_array_push(v_vs_1293_, v_x_1291_);
if (v_isShared_1296_ == 0)
{
lean_ctor_set(v___x_1295_, 1, v___x_1300_);
lean_ctor_set(v___x_1295_, 0, v___x_1299_);
v___x_1302_ = v___x_1295_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v___x_1299_);
lean_ctor_set(v_reuseFailAlloc_1303_, 1, v___x_1300_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
else
{
lean_object* v_k_x27_1304_; uint8_t v___x_1305_; 
v_k_x27_1304_ = lean_array_fget_borrowed(v_ks_1292_, v_x_1289_);
v___x_1305_ = l_Lean_instBEqMVarId_beq(v_x_1290_, v_k_x27_1304_);
if (v___x_1305_ == 0)
{
lean_object* v___x_1307_; 
if (v_isShared_1296_ == 0)
{
v___x_1307_ = v___x_1295_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1311_; 
v_reuseFailAlloc_1311_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1311_, 0, v_ks_1292_);
lean_ctor_set(v_reuseFailAlloc_1311_, 1, v_vs_1293_);
v___x_1307_ = v_reuseFailAlloc_1311_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
lean_object* v___x_1308_; lean_object* v___x_1309_; 
v___x_1308_ = lean_unsigned_to_nat(1u);
v___x_1309_ = lean_nat_add(v_x_1289_, v___x_1308_);
lean_dec(v_x_1289_);
v_x_1288_ = v___x_1307_;
v_x_1289_ = v___x_1309_;
goto _start;
}
}
else
{
lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1315_; 
v___x_1312_ = lean_array_fset(v_ks_1292_, v_x_1289_, v_x_1290_);
v___x_1313_ = lean_array_fset(v_vs_1293_, v_x_1289_, v_x_1291_);
lean_dec(v_x_1289_);
if (v_isShared_1296_ == 0)
{
lean_ctor_set(v___x_1295_, 1, v___x_1313_);
lean_ctor_set(v___x_1295_, 0, v___x_1312_);
v___x_1315_ = v___x_1295_;
goto v_reusejp_1314_;
}
else
{
lean_object* v_reuseFailAlloc_1316_; 
v_reuseFailAlloc_1316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1316_, 0, v___x_1312_);
lean_ctor_set(v_reuseFailAlloc_1316_, 1, v___x_1313_);
v___x_1315_ = v_reuseFailAlloc_1316_;
goto v_reusejp_1314_;
}
v_reusejp_1314_:
{
return v___x_1315_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_n_1318_, lean_object* v_k_1319_, lean_object* v_v_1320_){
_start:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; 
v___x_1321_ = lean_unsigned_to_nat(0u);
v___x_1322_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(v_n_1318_, v___x_1321_, v_k_1319_, v_v_1320_);
return v___x_1322_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_1323_; 
v___x_1323_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg(lean_object* v_x_1324_, size_t v_x_1325_, size_t v_x_1326_, lean_object* v_x_1327_, lean_object* v_x_1328_){
_start:
{
if (lean_obj_tag(v_x_1324_) == 0)
{
lean_object* v_es_1329_; size_t v___x_1330_; size_t v___x_1331_; lean_object* v_j_1332_; lean_object* v___x_1333_; uint8_t v___x_1334_; 
v_es_1329_ = lean_ctor_get(v_x_1324_, 0);
v___x_1330_ = ((size_t)31ULL);
v___x_1331_ = lean_usize_land(v_x_1325_, v___x_1330_);
v_j_1332_ = lean_usize_to_nat(v___x_1331_);
v___x_1333_ = lean_array_get_size(v_es_1329_);
v___x_1334_ = lean_nat_dec_lt(v_j_1332_, v___x_1333_);
if (v___x_1334_ == 0)
{
lean_dec(v_j_1332_);
lean_dec(v_x_1328_);
lean_dec(v_x_1327_);
return v_x_1324_;
}
else
{
lean_object* v___x_1336_; uint8_t v_isShared_1337_; uint8_t v_isSharedCheck_1373_; 
lean_inc_ref(v_es_1329_);
v_isSharedCheck_1373_ = !lean_is_exclusive(v_x_1324_);
if (v_isSharedCheck_1373_ == 0)
{
lean_object* v_unused_1374_; 
v_unused_1374_ = lean_ctor_get(v_x_1324_, 0);
lean_dec(v_unused_1374_);
v___x_1336_ = v_x_1324_;
v_isShared_1337_ = v_isSharedCheck_1373_;
goto v_resetjp_1335_;
}
else
{
lean_dec(v_x_1324_);
v___x_1336_ = lean_box(0);
v_isShared_1337_ = v_isSharedCheck_1373_;
goto v_resetjp_1335_;
}
v_resetjp_1335_:
{
lean_object* v_v_1338_; lean_object* v___x_1339_; lean_object* v_xs_x27_1340_; lean_object* v___y_1342_; 
v_v_1338_ = lean_array_fget(v_es_1329_, v_j_1332_);
v___x_1339_ = lean_box(0);
v_xs_x27_1340_ = lean_array_fset(v_es_1329_, v_j_1332_, v___x_1339_);
switch(lean_obj_tag(v_v_1338_))
{
case 0:
{
lean_object* v_key_1347_; lean_object* v_val_1348_; lean_object* v___x_1350_; uint8_t v_isShared_1351_; uint8_t v_isSharedCheck_1358_; 
v_key_1347_ = lean_ctor_get(v_v_1338_, 0);
v_val_1348_ = lean_ctor_get(v_v_1338_, 1);
v_isSharedCheck_1358_ = !lean_is_exclusive(v_v_1338_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1350_ = v_v_1338_;
v_isShared_1351_ = v_isSharedCheck_1358_;
goto v_resetjp_1349_;
}
else
{
lean_inc(v_val_1348_);
lean_inc(v_key_1347_);
lean_dec(v_v_1338_);
v___x_1350_ = lean_box(0);
v_isShared_1351_ = v_isSharedCheck_1358_;
goto v_resetjp_1349_;
}
v_resetjp_1349_:
{
uint8_t v___x_1352_; 
v___x_1352_ = l_Lean_instBEqMVarId_beq(v_x_1327_, v_key_1347_);
if (v___x_1352_ == 0)
{
lean_object* v___x_1353_; lean_object* v___x_1354_; 
lean_del_object(v___x_1350_);
v___x_1353_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1347_, v_val_1348_, v_x_1327_, v_x_1328_);
v___x_1354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1354_, 0, v___x_1353_);
v___y_1342_ = v___x_1354_;
goto v___jp_1341_;
}
else
{
lean_object* v___x_1356_; 
lean_dec(v_val_1348_);
lean_dec(v_key_1347_);
if (v_isShared_1351_ == 0)
{
lean_ctor_set(v___x_1350_, 1, v_x_1328_);
lean_ctor_set(v___x_1350_, 0, v_x_1327_);
v___x_1356_ = v___x_1350_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v_x_1327_);
lean_ctor_set(v_reuseFailAlloc_1357_, 1, v_x_1328_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
v___y_1342_ = v___x_1356_;
goto v___jp_1341_;
}
}
}
}
case 1:
{
lean_object* v_node_1359_; lean_object* v___x_1361_; uint8_t v_isShared_1362_; uint8_t v_isSharedCheck_1371_; 
v_node_1359_ = lean_ctor_get(v_v_1338_, 0);
v_isSharedCheck_1371_ = !lean_is_exclusive(v_v_1338_);
if (v_isSharedCheck_1371_ == 0)
{
v___x_1361_ = v_v_1338_;
v_isShared_1362_ = v_isSharedCheck_1371_;
goto v_resetjp_1360_;
}
else
{
lean_inc(v_node_1359_);
lean_dec(v_v_1338_);
v___x_1361_ = lean_box(0);
v_isShared_1362_ = v_isSharedCheck_1371_;
goto v_resetjp_1360_;
}
v_resetjp_1360_:
{
size_t v___x_1363_; size_t v___x_1364_; size_t v___x_1365_; size_t v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1369_; 
v___x_1363_ = ((size_t)5ULL);
v___x_1364_ = lean_usize_shift_right(v_x_1325_, v___x_1363_);
v___x_1365_ = ((size_t)1ULL);
v___x_1366_ = lean_usize_add(v_x_1326_, v___x_1365_);
v___x_1367_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg(v_node_1359_, v___x_1364_, v___x_1366_, v_x_1327_, v_x_1328_);
if (v_isShared_1362_ == 0)
{
lean_ctor_set(v___x_1361_, 0, v___x_1367_);
v___x_1369_ = v___x_1361_;
goto v_reusejp_1368_;
}
else
{
lean_object* v_reuseFailAlloc_1370_; 
v_reuseFailAlloc_1370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1370_, 0, v___x_1367_);
v___x_1369_ = v_reuseFailAlloc_1370_;
goto v_reusejp_1368_;
}
v_reusejp_1368_:
{
v___y_1342_ = v___x_1369_;
goto v___jp_1341_;
}
}
}
default: 
{
lean_object* v___x_1372_; 
v___x_1372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1372_, 0, v_x_1327_);
lean_ctor_set(v___x_1372_, 1, v_x_1328_);
v___y_1342_ = v___x_1372_;
goto v___jp_1341_;
}
}
v___jp_1341_:
{
lean_object* v___x_1343_; lean_object* v___x_1345_; 
v___x_1343_ = lean_array_fset(v_xs_x27_1340_, v_j_1332_, v___y_1342_);
lean_dec(v_j_1332_);
if (v_isShared_1337_ == 0)
{
lean_ctor_set(v___x_1336_, 0, v___x_1343_);
v___x_1345_ = v___x_1336_;
goto v_reusejp_1344_;
}
else
{
lean_object* v_reuseFailAlloc_1346_; 
v_reuseFailAlloc_1346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1346_, 0, v___x_1343_);
v___x_1345_ = v_reuseFailAlloc_1346_;
goto v_reusejp_1344_;
}
v_reusejp_1344_:
{
return v___x_1345_;
}
}
}
}
}
else
{
lean_object* v_ks_1375_; lean_object* v_vs_1376_; lean_object* v___x_1378_; uint8_t v_isShared_1379_; uint8_t v_isSharedCheck_1396_; 
v_ks_1375_ = lean_ctor_get(v_x_1324_, 0);
v_vs_1376_ = lean_ctor_get(v_x_1324_, 1);
v_isSharedCheck_1396_ = !lean_is_exclusive(v_x_1324_);
if (v_isSharedCheck_1396_ == 0)
{
v___x_1378_ = v_x_1324_;
v_isShared_1379_ = v_isSharedCheck_1396_;
goto v_resetjp_1377_;
}
else
{
lean_inc(v_vs_1376_);
lean_inc(v_ks_1375_);
lean_dec(v_x_1324_);
v___x_1378_ = lean_box(0);
v_isShared_1379_ = v_isSharedCheck_1396_;
goto v_resetjp_1377_;
}
v_resetjp_1377_:
{
lean_object* v___x_1381_; 
if (v_isShared_1379_ == 0)
{
v___x_1381_ = v___x_1378_;
goto v_reusejp_1380_;
}
else
{
lean_object* v_reuseFailAlloc_1395_; 
v_reuseFailAlloc_1395_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1395_, 0, v_ks_1375_);
lean_ctor_set(v_reuseFailAlloc_1395_, 1, v_vs_1376_);
v___x_1381_ = v_reuseFailAlloc_1395_;
goto v_reusejp_1380_;
}
v_reusejp_1380_:
{
lean_object* v_newNode_1382_; uint8_t v___y_1384_; size_t v___x_1390_; uint8_t v___x_1391_; 
v_newNode_1382_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3___redArg(v___x_1381_, v_x_1327_, v_x_1328_);
v___x_1390_ = ((size_t)7ULL);
v___x_1391_ = lean_usize_dec_le(v___x_1390_, v_x_1326_);
if (v___x_1391_ == 0)
{
lean_object* v___x_1392_; lean_object* v___x_1393_; uint8_t v___x_1394_; 
v___x_1392_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1382_);
v___x_1393_ = lean_unsigned_to_nat(4u);
v___x_1394_ = lean_nat_dec_lt(v___x_1392_, v___x_1393_);
lean_dec(v___x_1392_);
v___y_1384_ = v___x_1394_;
goto v___jp_1383_;
}
else
{
v___y_1384_ = v___x_1391_;
goto v___jp_1383_;
}
v___jp_1383_:
{
if (v___y_1384_ == 0)
{
lean_object* v_ks_1385_; lean_object* v_vs_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; 
v_ks_1385_ = lean_ctor_get(v_newNode_1382_, 0);
lean_inc_ref(v_ks_1385_);
v_vs_1386_ = lean_ctor_get(v_newNode_1382_, 1);
lean_inc_ref(v_vs_1386_);
lean_dec_ref(v_newNode_1382_);
v___x_1387_ = lean_unsigned_to_nat(0u);
v___x_1388_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___closed__0);
v___x_1389_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___redArg(v_x_1326_, v_ks_1385_, v_vs_1386_, v___x_1387_, v___x_1388_);
lean_dec_ref(v_vs_1386_);
lean_dec_ref(v_ks_1385_);
return v___x_1389_;
}
else
{
return v_newNode_1382_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___redArg(size_t v_depth_1397_, lean_object* v_keys_1398_, lean_object* v_vals_1399_, lean_object* v_i_1400_, lean_object* v_entries_1401_){
_start:
{
lean_object* v___x_1402_; uint8_t v___x_1403_; 
v___x_1402_ = lean_array_get_size(v_keys_1398_);
v___x_1403_ = lean_nat_dec_lt(v_i_1400_, v___x_1402_);
if (v___x_1403_ == 0)
{
lean_dec(v_i_1400_);
return v_entries_1401_;
}
else
{
lean_object* v_k_1404_; lean_object* v_v_1405_; uint64_t v___x_1406_; size_t v_h_1407_; size_t v___x_1408_; lean_object* v___x_1409_; size_t v___x_1410_; size_t v___x_1411_; size_t v___x_1412_; size_t v_h_1413_; lean_object* v___x_1414_; lean_object* v___x_1415_; 
v_k_1404_ = lean_array_fget_borrowed(v_keys_1398_, v_i_1400_);
v_v_1405_ = lean_array_fget_borrowed(v_vals_1399_, v_i_1400_);
v___x_1406_ = l_Lean_instHashableMVarId_hash(v_k_1404_);
v_h_1407_ = lean_uint64_to_usize(v___x_1406_);
v___x_1408_ = ((size_t)5ULL);
v___x_1409_ = lean_unsigned_to_nat(1u);
v___x_1410_ = ((size_t)1ULL);
v___x_1411_ = lean_usize_sub(v_depth_1397_, v___x_1410_);
v___x_1412_ = lean_usize_mul(v___x_1408_, v___x_1411_);
v_h_1413_ = lean_usize_shift_right(v_h_1407_, v___x_1412_);
v___x_1414_ = lean_nat_add(v_i_1400_, v___x_1409_);
lean_dec(v_i_1400_);
lean_inc(v_v_1405_);
lean_inc(v_k_1404_);
v___x_1415_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg(v_entries_1401_, v_h_1413_, v_depth_1397_, v_k_1404_, v_v_1405_);
v_i_1400_ = v___x_1414_;
v_entries_1401_ = v___x_1415_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___redArg___boxed(lean_object* v_depth_1417_, lean_object* v_keys_1418_, lean_object* v_vals_1419_, lean_object* v_i_1420_, lean_object* v_entries_1421_){
_start:
{
size_t v_depth_boxed_1422_; lean_object* v_res_1423_; 
v_depth_boxed_1422_ = lean_unbox_usize(v_depth_1417_);
lean_dec(v_depth_1417_);
v_res_1423_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___redArg(v_depth_boxed_1422_, v_keys_1418_, v_vals_1419_, v_i_1420_, v_entries_1421_);
lean_dec_ref(v_vals_1419_);
lean_dec_ref(v_keys_1418_);
return v_res_1423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_x_1424_, lean_object* v_x_1425_, lean_object* v_x_1426_, lean_object* v_x_1427_, lean_object* v_x_1428_){
_start:
{
size_t v_x_2647__boxed_1429_; size_t v_x_2648__boxed_1430_; lean_object* v_res_1431_; 
v_x_2647__boxed_1429_ = lean_unbox_usize(v_x_1425_);
lean_dec(v_x_1425_);
v_x_2648__boxed_1430_ = lean_unbox_usize(v_x_1426_);
lean_dec(v_x_1426_);
v_res_1431_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg(v_x_1424_, v_x_2647__boxed_1429_, v_x_2648__boxed_1430_, v_x_1427_, v_x_1428_);
return v_res_1431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0___redArg(lean_object* v_x_1432_, lean_object* v_x_1433_, lean_object* v_x_1434_){
_start:
{
uint64_t v___x_1435_; size_t v___x_1436_; size_t v___x_1437_; lean_object* v___x_1438_; 
v___x_1435_ = l_Lean_instHashableMVarId_hash(v_x_1433_);
v___x_1436_ = lean_uint64_to_usize(v___x_1435_);
v___x_1437_ = ((size_t)1ULL);
v___x_1438_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg(v_x_1432_, v___x_1436_, v___x_1437_, v_x_1433_, v_x_1434_);
return v___x_1438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___redArg(lean_object* v_mvarId_1439_, lean_object* v_val_1440_, lean_object* v___y_1441_){
_start:
{
lean_object* v___x_1443_; lean_object* v_mctx_1444_; lean_object* v_cache_1445_; lean_object* v_zetaDeltaFVarIds_1446_; lean_object* v_postponed_1447_; lean_object* v_diag_1448_; lean_object* v___x_1450_; uint8_t v_isShared_1451_; uint8_t v_isSharedCheck_1476_; 
v___x_1443_ = lean_st_ref_take(v___y_1441_);
v_mctx_1444_ = lean_ctor_get(v___x_1443_, 0);
v_cache_1445_ = lean_ctor_get(v___x_1443_, 1);
v_zetaDeltaFVarIds_1446_ = lean_ctor_get(v___x_1443_, 2);
v_postponed_1447_ = lean_ctor_get(v___x_1443_, 3);
v_diag_1448_ = lean_ctor_get(v___x_1443_, 4);
v_isSharedCheck_1476_ = !lean_is_exclusive(v___x_1443_);
if (v_isSharedCheck_1476_ == 0)
{
v___x_1450_ = v___x_1443_;
v_isShared_1451_ = v_isSharedCheck_1476_;
goto v_resetjp_1449_;
}
else
{
lean_inc(v_diag_1448_);
lean_inc(v_postponed_1447_);
lean_inc(v_zetaDeltaFVarIds_1446_);
lean_inc(v_cache_1445_);
lean_inc(v_mctx_1444_);
lean_dec(v___x_1443_);
v___x_1450_ = lean_box(0);
v_isShared_1451_ = v_isSharedCheck_1476_;
goto v_resetjp_1449_;
}
v_resetjp_1449_:
{
lean_object* v_depth_1452_; lean_object* v_levelAssignDepth_1453_; lean_object* v_lmvarCounter_1454_; lean_object* v_mvarCounter_1455_; lean_object* v_lDecls_1456_; lean_object* v_decls_1457_; lean_object* v_userNames_1458_; lean_object* v_lAssignment_1459_; lean_object* v_eAssignment_1460_; lean_object* v_dAssignment_1461_; lean_object* v___x_1463_; uint8_t v_isShared_1464_; uint8_t v_isSharedCheck_1475_; 
v_depth_1452_ = lean_ctor_get(v_mctx_1444_, 0);
v_levelAssignDepth_1453_ = lean_ctor_get(v_mctx_1444_, 1);
v_lmvarCounter_1454_ = lean_ctor_get(v_mctx_1444_, 2);
v_mvarCounter_1455_ = lean_ctor_get(v_mctx_1444_, 3);
v_lDecls_1456_ = lean_ctor_get(v_mctx_1444_, 4);
v_decls_1457_ = lean_ctor_get(v_mctx_1444_, 5);
v_userNames_1458_ = lean_ctor_get(v_mctx_1444_, 6);
v_lAssignment_1459_ = lean_ctor_get(v_mctx_1444_, 7);
v_eAssignment_1460_ = lean_ctor_get(v_mctx_1444_, 8);
v_dAssignment_1461_ = lean_ctor_get(v_mctx_1444_, 9);
v_isSharedCheck_1475_ = !lean_is_exclusive(v_mctx_1444_);
if (v_isSharedCheck_1475_ == 0)
{
v___x_1463_ = v_mctx_1444_;
v_isShared_1464_ = v_isSharedCheck_1475_;
goto v_resetjp_1462_;
}
else
{
lean_inc(v_dAssignment_1461_);
lean_inc(v_eAssignment_1460_);
lean_inc(v_lAssignment_1459_);
lean_inc(v_userNames_1458_);
lean_inc(v_decls_1457_);
lean_inc(v_lDecls_1456_);
lean_inc(v_mvarCounter_1455_);
lean_inc(v_lmvarCounter_1454_);
lean_inc(v_levelAssignDepth_1453_);
lean_inc(v_depth_1452_);
lean_dec(v_mctx_1444_);
v___x_1463_ = lean_box(0);
v_isShared_1464_ = v_isSharedCheck_1475_;
goto v_resetjp_1462_;
}
v_resetjp_1462_:
{
lean_object* v___x_1465_; lean_object* v___x_1467_; 
v___x_1465_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0___redArg(v_eAssignment_1460_, v_mvarId_1439_, v_val_1440_);
if (v_isShared_1464_ == 0)
{
lean_ctor_set(v___x_1463_, 8, v___x_1465_);
v___x_1467_ = v___x_1463_;
goto v_reusejp_1466_;
}
else
{
lean_object* v_reuseFailAlloc_1474_; 
v_reuseFailAlloc_1474_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1474_, 0, v_depth_1452_);
lean_ctor_set(v_reuseFailAlloc_1474_, 1, v_levelAssignDepth_1453_);
lean_ctor_set(v_reuseFailAlloc_1474_, 2, v_lmvarCounter_1454_);
lean_ctor_set(v_reuseFailAlloc_1474_, 3, v_mvarCounter_1455_);
lean_ctor_set(v_reuseFailAlloc_1474_, 4, v_lDecls_1456_);
lean_ctor_set(v_reuseFailAlloc_1474_, 5, v_decls_1457_);
lean_ctor_set(v_reuseFailAlloc_1474_, 6, v_userNames_1458_);
lean_ctor_set(v_reuseFailAlloc_1474_, 7, v_lAssignment_1459_);
lean_ctor_set(v_reuseFailAlloc_1474_, 8, v___x_1465_);
lean_ctor_set(v_reuseFailAlloc_1474_, 9, v_dAssignment_1461_);
v___x_1467_ = v_reuseFailAlloc_1474_;
goto v_reusejp_1466_;
}
v_reusejp_1466_:
{
lean_object* v___x_1469_; 
if (v_isShared_1451_ == 0)
{
lean_ctor_set(v___x_1450_, 0, v___x_1467_);
v___x_1469_ = v___x_1450_;
goto v_reusejp_1468_;
}
else
{
lean_object* v_reuseFailAlloc_1473_; 
v_reuseFailAlloc_1473_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1473_, 0, v___x_1467_);
lean_ctor_set(v_reuseFailAlloc_1473_, 1, v_cache_1445_);
lean_ctor_set(v_reuseFailAlloc_1473_, 2, v_zetaDeltaFVarIds_1446_);
lean_ctor_set(v_reuseFailAlloc_1473_, 3, v_postponed_1447_);
lean_ctor_set(v_reuseFailAlloc_1473_, 4, v_diag_1448_);
v___x_1469_ = v_reuseFailAlloc_1473_;
goto v_reusejp_1468_;
}
v_reusejp_1468_:
{
lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; 
v___x_1470_ = lean_st_ref_set(v___y_1441_, v___x_1469_);
v___x_1471_ = lean_box(0);
v___x_1472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1472_, 0, v___x_1471_);
return v___x_1472_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___redArg___boxed(lean_object* v_mvarId_1477_, lean_object* v_val_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_){
_start:
{
lean_object* v_res_1481_; 
v_res_1481_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___redArg(v_mvarId_1477_, v_val_1478_, v___y_1479_);
lean_dec(v___y_1479_);
return v_res_1481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1(lean_object* v_mvarId_1485_, lean_object* v___x_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_){
_start:
{
lean_object* v___x_1492_; 
lean_inc(v_mvarId_1485_);
v___x_1492_ = l_Lean_MVarId_checkNotAssigned(v_mvarId_1485_, v___x_1486_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_);
if (lean_obj_tag(v___x_1492_) == 0)
{
lean_object* v_keyedConfig_1493_; uint8_t v_trackZetaDelta_1494_; lean_object* v_zetaDeltaSet_1495_; lean_object* v_lctx_1496_; lean_object* v_localInstances_1497_; lean_object* v_defEqCtx_x3f_1498_; lean_object* v_synthPendingDepth_1499_; lean_object* v_customCanUnfoldPredicate_x3f_1500_; uint8_t v_univApprox_1501_; uint8_t v_inTypeClassResolution_1502_; uint8_t v_cacheInferType_1503_; uint8_t v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; 
lean_dec_ref_known(v___x_1492_, 1);
v_keyedConfig_1493_ = lean_ctor_get(v___y_1487_, 0);
v_trackZetaDelta_1494_ = lean_ctor_get_uint8(v___y_1487_, sizeof(void*)*7);
v_zetaDeltaSet_1495_ = lean_ctor_get(v___y_1487_, 1);
v_lctx_1496_ = lean_ctor_get(v___y_1487_, 2);
v_localInstances_1497_ = lean_ctor_get(v___y_1487_, 3);
v_defEqCtx_x3f_1498_ = lean_ctor_get(v___y_1487_, 4);
v_synthPendingDepth_1499_ = lean_ctor_get(v___y_1487_, 5);
v_customCanUnfoldPredicate_x3f_1500_ = lean_ctor_get(v___y_1487_, 6);
v_univApprox_1501_ = lean_ctor_get_uint8(v___y_1487_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1502_ = lean_ctor_get_uint8(v___y_1487_, sizeof(void*)*7 + 2);
v_cacheInferType_1503_ = lean_ctor_get_uint8(v___y_1487_, sizeof(void*)*7 + 3);
v___x_1504_ = 2;
lean_inc_ref(v_keyedConfig_1493_);
v___x_1505_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1504_, v_keyedConfig_1493_);
lean_inc(v_customCanUnfoldPredicate_x3f_1500_);
lean_inc(v_synthPendingDepth_1499_);
lean_inc(v_defEqCtx_x3f_1498_);
lean_inc_ref(v_localInstances_1497_);
lean_inc_ref(v_lctx_1496_);
lean_inc(v_zetaDeltaSet_1495_);
v___x_1506_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1506_, 0, v___x_1505_);
lean_ctor_set(v___x_1506_, 1, v_zetaDeltaSet_1495_);
lean_ctor_set(v___x_1506_, 2, v_lctx_1496_);
lean_ctor_set(v___x_1506_, 3, v_localInstances_1497_);
lean_ctor_set(v___x_1506_, 4, v_defEqCtx_x3f_1498_);
lean_ctor_set(v___x_1506_, 5, v_synthPendingDepth_1499_);
lean_ctor_set(v___x_1506_, 6, v_customCanUnfoldPredicate_x3f_1500_);
lean_ctor_set_uint8(v___x_1506_, sizeof(void*)*7, v_trackZetaDelta_1494_);
lean_ctor_set_uint8(v___x_1506_, sizeof(void*)*7 + 1, v_univApprox_1501_);
lean_ctor_set_uint8(v___x_1506_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1502_);
lean_ctor_set_uint8(v___x_1506_, sizeof(void*)*7 + 3, v_cacheInferType_1503_);
lean_inc(v_mvarId_1485_);
v___x_1507_ = l_Lean_MVarId_getType_x27(v_mvarId_1485_, v___x_1506_, v___y_1488_, v___y_1489_, v___y_1490_);
lean_dec_ref_known(v___x_1506_, 7);
if (lean_obj_tag(v___x_1507_) == 0)
{
lean_object* v_a_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; uint8_t v___x_1511_; 
v_a_1508_ = lean_ctor_get(v___x_1507_, 0);
lean_inc(v_a_1508_);
lean_dec_ref_known(v___x_1507_, 1);
v___x_1509_ = ((lean_object*)(lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__1));
v___x_1510_ = lean_unsigned_to_nat(3u);
v___x_1511_ = l_Lean_Expr_isAppOfArity(v_a_1508_, v___x_1509_, v___x_1510_);
if (v___x_1511_ == 0)
{
lean_object* v___x_1512_; lean_object* v___x_1513_; 
lean_dec(v_a_1508_);
lean_dec(v_mvarId_1485_);
v___x_1512_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__2);
v___x_1513_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v___x_1512_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_);
lean_dec_ref(v___y_1487_);
return v___x_1513_;
}
else
{
lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___f_1517_; lean_object* v___x_1518_; 
v___x_1514_ = l_Lean_Expr_appFn_x21(v_a_1508_);
v___x_1515_ = l_Lean_Expr_appArg_x21(v___x_1514_);
lean_dec_ref(v___x_1514_);
v___x_1516_ = l_Lean_Expr_appArg_x21(v_a_1508_);
lean_dec(v_a_1508_);
v___f_1517_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__0___boxed), 8, 2);
lean_closure_set(v___f_1517_, 0, v___x_1515_);
lean_closure_set(v___f_1517_, 1, v___x_1516_);
v___x_1518_ = lp_mathlib_Lean_Meta_withSubsingletonAsFast___redArg(v___f_1517_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_);
lean_dec_ref(v___y_1487_);
if (lean_obj_tag(v___x_1518_) == 0)
{
lean_object* v_a_1519_; lean_object* v___x_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1528_; 
v_a_1519_ = lean_ctor_get(v___x_1518_, 0);
lean_inc(v_a_1519_);
lean_dec_ref_known(v___x_1518_, 1);
v___x_1520_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___redArg(v_mvarId_1485_, v_a_1519_, v___y_1488_);
v_isSharedCheck_1528_ = !lean_is_exclusive(v___x_1520_);
if (v_isSharedCheck_1528_ == 0)
{
lean_object* v_unused_1529_; 
v_unused_1529_ = lean_ctor_get(v___x_1520_, 0);
lean_dec(v_unused_1529_);
v___x_1522_ = v___x_1520_;
v_isShared_1523_ = v_isSharedCheck_1528_;
goto v_resetjp_1521_;
}
else
{
lean_dec(v___x_1520_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1528_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
lean_object* v___x_1524_; lean_object* v___x_1526_; 
v___x_1524_ = lean_box(v___x_1511_);
if (v_isShared_1523_ == 0)
{
lean_ctor_set(v___x_1522_, 0, v___x_1524_);
v___x_1526_ = v___x_1522_;
goto v_reusejp_1525_;
}
else
{
lean_object* v_reuseFailAlloc_1527_; 
v_reuseFailAlloc_1527_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1527_, 0, v___x_1524_);
v___x_1526_ = v_reuseFailAlloc_1527_;
goto v_reusejp_1525_;
}
v_reusejp_1525_:
{
return v___x_1526_;
}
}
}
else
{
lean_object* v_a_1530_; lean_object* v___x_1532_; uint8_t v_isShared_1533_; uint8_t v_isSharedCheck_1537_; 
lean_dec(v_mvarId_1485_);
v_a_1530_ = lean_ctor_get(v___x_1518_, 0);
v_isSharedCheck_1537_ = !lean_is_exclusive(v___x_1518_);
if (v_isSharedCheck_1537_ == 0)
{
v___x_1532_ = v___x_1518_;
v_isShared_1533_ = v_isSharedCheck_1537_;
goto v_resetjp_1531_;
}
else
{
lean_inc(v_a_1530_);
lean_dec(v___x_1518_);
v___x_1532_ = lean_box(0);
v_isShared_1533_ = v_isSharedCheck_1537_;
goto v_resetjp_1531_;
}
v_resetjp_1531_:
{
lean_object* v___x_1535_; 
if (v_isShared_1533_ == 0)
{
v___x_1535_ = v___x_1532_;
goto v_reusejp_1534_;
}
else
{
lean_object* v_reuseFailAlloc_1536_; 
v_reuseFailAlloc_1536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1536_, 0, v_a_1530_);
v___x_1535_ = v_reuseFailAlloc_1536_;
goto v_reusejp_1534_;
}
v_reusejp_1534_:
{
return v___x_1535_;
}
}
}
}
}
else
{
lean_object* v_a_1538_; lean_object* v___x_1540_; uint8_t v_isShared_1541_; uint8_t v_isSharedCheck_1545_; 
lean_dec_ref(v___y_1487_);
lean_dec(v_mvarId_1485_);
v_a_1538_ = lean_ctor_get(v___x_1507_, 0);
v_isSharedCheck_1545_ = !lean_is_exclusive(v___x_1507_);
if (v_isSharedCheck_1545_ == 0)
{
v___x_1540_ = v___x_1507_;
v_isShared_1541_ = v_isSharedCheck_1545_;
goto v_resetjp_1539_;
}
else
{
lean_inc(v_a_1538_);
lean_dec(v___x_1507_);
v___x_1540_ = lean_box(0);
v_isShared_1541_ = v_isSharedCheck_1545_;
goto v_resetjp_1539_;
}
v_resetjp_1539_:
{
lean_object* v___x_1543_; 
if (v_isShared_1541_ == 0)
{
v___x_1543_ = v___x_1540_;
goto v_reusejp_1542_;
}
else
{
lean_object* v_reuseFailAlloc_1544_; 
v_reuseFailAlloc_1544_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1544_, 0, v_a_1538_);
v___x_1543_ = v_reuseFailAlloc_1544_;
goto v_reusejp_1542_;
}
v_reusejp_1542_:
{
return v___x_1543_;
}
}
}
}
else
{
lean_object* v_a_1546_; lean_object* v___x_1548_; uint8_t v_isShared_1549_; uint8_t v_isSharedCheck_1553_; 
lean_dec_ref(v___y_1487_);
lean_dec(v_mvarId_1485_);
v_a_1546_ = lean_ctor_get(v___x_1492_, 0);
v_isSharedCheck_1553_ = !lean_is_exclusive(v___x_1492_);
if (v_isSharedCheck_1553_ == 0)
{
v___x_1548_ = v___x_1492_;
v_isShared_1549_ = v_isSharedCheck_1553_;
goto v_resetjp_1547_;
}
else
{
lean_inc(v_a_1546_);
lean_dec(v___x_1492_);
v___x_1548_ = lean_box(0);
v_isShared_1549_ = v_isSharedCheck_1553_;
goto v_resetjp_1547_;
}
v_resetjp_1547_:
{
lean_object* v___x_1551_; 
if (v_isShared_1549_ == 0)
{
v___x_1551_ = v___x_1548_;
goto v_reusejp_1550_;
}
else
{
lean_object* v_reuseFailAlloc_1552_; 
v_reuseFailAlloc_1552_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1552_, 0, v_a_1546_);
v___x_1551_ = v_reuseFailAlloc_1552_;
goto v_reusejp_1550_;
}
v_reusejp_1550_:
{
return v___x_1551_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___boxed(lean_object* v_mvarId_1554_, lean_object* v___x_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_){
_start:
{
lean_object* v_res_1561_; 
v_res_1561_ = lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1(v_mvarId_1554_, v___x_1555_, v___y_1556_, v___y_1557_, v___y_1558_, v___y_1559_);
lean_dec(v___y_1559_);
lean_dec_ref(v___y_1558_);
lean_dec(v___y_1557_);
return v_res_1561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__2(lean_object* v___f_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_){
_start:
{
lean_object* v___x_1568_; 
v___x_1568_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg(v___f_1562_, v___y_1563_, v___y_1564_, v___y_1565_, v___y_1566_);
if (lean_obj_tag(v___x_1568_) == 0)
{
lean_object* v_a_1569_; lean_object* v___x_1571_; uint8_t v_isShared_1572_; uint8_t v_isSharedCheck_1582_; 
v_a_1569_ = lean_ctor_get(v___x_1568_, 0);
v_isSharedCheck_1582_ = !lean_is_exclusive(v___x_1568_);
if (v_isSharedCheck_1582_ == 0)
{
v___x_1571_ = v___x_1568_;
v_isShared_1572_ = v_isSharedCheck_1582_;
goto v_resetjp_1570_;
}
else
{
lean_inc(v_a_1569_);
lean_dec(v___x_1568_);
v___x_1571_ = lean_box(0);
v_isShared_1572_ = v_isSharedCheck_1582_;
goto v_resetjp_1570_;
}
v_resetjp_1570_:
{
if (lean_obj_tag(v_a_1569_) == 0)
{
uint8_t v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1576_; 
v___x_1573_ = 0;
v___x_1574_ = lean_box(v___x_1573_);
if (v_isShared_1572_ == 0)
{
lean_ctor_set(v___x_1571_, 0, v___x_1574_);
v___x_1576_ = v___x_1571_;
goto v_reusejp_1575_;
}
else
{
lean_object* v_reuseFailAlloc_1577_; 
v_reuseFailAlloc_1577_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1577_, 0, v___x_1574_);
v___x_1576_ = v_reuseFailAlloc_1577_;
goto v_reusejp_1575_;
}
v_reusejp_1575_:
{
return v___x_1576_;
}
}
else
{
lean_object* v_val_1578_; lean_object* v___x_1580_; 
v_val_1578_ = lean_ctor_get(v_a_1569_, 0);
lean_inc(v_val_1578_);
lean_dec_ref_known(v_a_1569_, 1);
if (v_isShared_1572_ == 0)
{
lean_ctor_set(v___x_1571_, 0, v_val_1578_);
v___x_1580_ = v___x_1571_;
goto v_reusejp_1579_;
}
else
{
lean_object* v_reuseFailAlloc_1581_; 
v_reuseFailAlloc_1581_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1581_, 0, v_val_1578_);
v___x_1580_ = v_reuseFailAlloc_1581_;
goto v_reusejp_1579_;
}
v_reusejp_1579_:
{
return v___x_1580_;
}
}
}
}
else
{
lean_object* v_a_1583_; lean_object* v___x_1585_; uint8_t v_isShared_1586_; uint8_t v_isSharedCheck_1590_; 
v_a_1583_ = lean_ctor_get(v___x_1568_, 0);
v_isSharedCheck_1590_ = !lean_is_exclusive(v___x_1568_);
if (v_isSharedCheck_1590_ == 0)
{
v___x_1585_ = v___x_1568_;
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
else
{
lean_inc(v_a_1583_);
lean_dec(v___x_1568_);
v___x_1585_ = lean_box(0);
v_isShared_1586_ = v_isSharedCheck_1590_;
goto v_resetjp_1584_;
}
v_resetjp_1584_:
{
lean_object* v___x_1588_; 
if (v_isShared_1586_ == 0)
{
v___x_1588_ = v___x_1585_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v_a_1583_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__2___boxed(lean_object* v___f_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_){
_start:
{
lean_object* v_res_1597_; 
v_res_1597_ = lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__2(v___f_1591_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
lean_dec(v___y_1595_);
lean_dec_ref(v___y_1594_);
lean_dec(v___y_1593_);
lean_dec_ref(v___y_1592_);
return v_res_1597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim(lean_object* v_mvarId_1601_, lean_object* v_a_1602_, lean_object* v_a_1603_, lean_object* v_a_1604_, lean_object* v_a_1605_){
_start:
{
lean_object* v___x_1607_; lean_object* v___f_1608_; lean_object* v___f_1609_; lean_object* v___x_1610_; 
v___x_1607_ = ((lean_object*)(lp_mathlib_Lean_Meta_fastSubsingletonElim___closed__1));
lean_inc(v_mvarId_1601_);
v___f_1608_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___boxed), 7, 2);
lean_closure_set(v___f_1608_, 0, v_mvarId_1601_);
lean_closure_set(v___f_1608_, 1, v___x_1607_);
v___f_1609_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__2___boxed), 6, 1);
lean_closure_set(v___f_1609_, 0, v___f_1608_);
v___x_1610_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Meta_fastSubsingletonElim_spec__1___redArg(v_mvarId_1601_, v___f_1609_, v_a_1602_, v_a_1603_, v_a_1604_, v_a_1605_);
return v___x_1610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_fastSubsingletonElim___boxed(lean_object* v_mvarId_1611_, lean_object* v_a_1612_, lean_object* v_a_1613_, lean_object* v_a_1614_, lean_object* v_a_1615_, lean_object* v_a_1616_){
_start:
{
lean_object* v_res_1617_; 
v_res_1617_ = lp_mathlib_Lean_Meta_fastSubsingletonElim(v_mvarId_1611_, v_a_1612_, v_a_1613_, v_a_1614_, v_a_1615_);
lean_dec(v_a_1615_);
lean_dec_ref(v_a_1614_);
lean_dec(v_a_1613_);
lean_dec_ref(v_a_1612_);
return v_res_1617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0(lean_object* v_mvarId_1618_, lean_object* v_val_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v___x_1625_; 
v___x_1625_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___redArg(v_mvarId_1618_, v_val_1619_, v___y_1621_);
return v___x_1625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0___boxed(lean_object* v_mvarId_1626_, lean_object* v_val_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_){
_start:
{
lean_object* v_res_1633_; 
v_res_1633_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0(v_mvarId_1626_, v_val_1627_, v___y_1628_, v___y_1629_, v___y_1630_, v___y_1631_);
lean_dec(v___y_1631_);
lean_dec_ref(v___y_1630_);
lean_dec(v___y_1629_);
lean_dec_ref(v___y_1628_);
return v_res_1633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0(lean_object* v_00_u03b2_1634_, lean_object* v_x_1635_, lean_object* v_x_1636_, lean_object* v_x_1637_){
_start:
{
lean_object* v___x_1638_; 
v___x_1638_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0___redArg(v_x_1635_, v_x_1636_, v_x_1637_);
return v___x_1638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_1639_, lean_object* v_x_1640_, size_t v_x_1641_, size_t v_x_1642_, lean_object* v_x_1643_, lean_object* v_x_1644_){
_start:
{
lean_object* v___x_1645_; 
v___x_1645_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___redArg(v_x_1640_, v_x_1641_, v_x_1642_, v_x_1643_, v_x_1644_);
return v___x_1645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_1646_, lean_object* v_x_1647_, lean_object* v_x_1648_, lean_object* v_x_1649_, lean_object* v_x_1650_, lean_object* v_x_1651_){
_start:
{
size_t v_x_3104__boxed_1652_; size_t v_x_3105__boxed_1653_; lean_object* v_res_1654_; 
v_x_3104__boxed_1652_ = lean_unbox_usize(v_x_1648_);
lean_dec(v_x_1648_);
v_x_3105__boxed_1653_ = lean_unbox_usize(v_x_1649_);
lean_dec(v_x_1649_);
v_res_1654_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2(v_00_u03b2_1646_, v_x_1647_, v_x_3104__boxed_1652_, v_x_3105__boxed_1653_, v_x_1650_, v_x_1651_);
return v_res_1654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3(lean_object* v_00_u03b2_1655_, lean_object* v_n_1656_, lean_object* v_k_1657_, lean_object* v_v_1658_){
_start:
{
lean_object* v___x_1659_; 
v___x_1659_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3___redArg(v_n_1656_, v_k_1657_, v_v_1658_);
return v___x_1659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4(lean_object* v_00_u03b2_1660_, size_t v_depth_1661_, lean_object* v_keys_1662_, lean_object* v_vals_1663_, lean_object* v_heq_1664_, lean_object* v_i_1665_, lean_object* v_entries_1666_){
_start:
{
lean_object* v___x_1667_; 
v___x_1667_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___redArg(v_depth_1661_, v_keys_1662_, v_vals_1663_, v_i_1665_, v_entries_1666_);
return v___x_1667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_00_u03b2_1668_, lean_object* v_depth_1669_, lean_object* v_keys_1670_, lean_object* v_vals_1671_, lean_object* v_heq_1672_, lean_object* v_i_1673_, lean_object* v_entries_1674_){
_start:
{
size_t v_depth_boxed_1675_; lean_object* v_res_1676_; 
v_depth_boxed_1675_ = lean_unbox_usize(v_depth_1669_);
lean_dec(v_depth_1669_);
v_res_1676_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__4(v_00_u03b2_1668_, v_depth_boxed_1675_, v_keys_1670_, v_vals_1671_, v_heq_1672_, v_i_1673_, v_entries_1674_);
lean_dec_ref(v_vals_1671_);
lean_dec_ref(v_keys_1670_);
return v_res_1676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_1677_, lean_object* v_x_1678_, lean_object* v_x_1679_, lean_object* v_x_1680_, lean_object* v_x_1681_){
_start:
{
lean_object* v___x_1682_; 
v___x_1682_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Meta_fastSubsingletonElim_spec__0_spec__0_spec__2_spec__3_spec__4___redArg(v_x_1678_, v_x_1679_, v_x_1680_, v_x_1681_);
return v___x_1682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___lam__0(lean_object* v_k_1683_, lean_object* v_b_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_){
_start:
{
lean_object* v___x_1690_; 
lean_inc(v___y_1688_);
lean_inc_ref(v___y_1687_);
lean_inc(v___y_1686_);
lean_inc_ref(v___y_1685_);
v___x_1690_ = lean_apply_6(v_k_1683_, v_b_1684_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_, lean_box(0));
return v___x_1690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_k_1691_, lean_object* v_b_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_){
_start:
{
lean_object* v_res_1698_; 
v_res_1698_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___lam__0(v_k_1691_, v_b_1692_, v___y_1693_, v___y_1694_, v___y_1695_, v___y_1696_);
lean_dec(v___y_1696_);
lean_dec_ref(v___y_1695_);
lean_dec(v___y_1694_);
lean_dec_ref(v___y_1693_);
return v_res_1698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg(lean_object* v_name_1699_, uint8_t v_bi_1700_, lean_object* v_type_1701_, lean_object* v_k_1702_, uint8_t v_kind_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_){
_start:
{
lean_object* v___f_1709_; lean_object* v___x_1710_; 
v___f_1709_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1709_, 0, v_k_1702_);
v___x_1710_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1699_, v_bi_1700_, v_type_1701_, v___f_1709_, v_kind_1703_, v___y_1704_, v___y_1705_, v___y_1706_, v___y_1707_);
if (lean_obj_tag(v___x_1710_) == 0)
{
lean_object* v_a_1711_; lean_object* v___x_1713_; uint8_t v_isShared_1714_; uint8_t v_isSharedCheck_1718_; 
v_a_1711_ = lean_ctor_get(v___x_1710_, 0);
v_isSharedCheck_1718_ = !lean_is_exclusive(v___x_1710_);
if (v_isSharedCheck_1718_ == 0)
{
v___x_1713_ = v___x_1710_;
v_isShared_1714_ = v_isSharedCheck_1718_;
goto v_resetjp_1712_;
}
else
{
lean_inc(v_a_1711_);
lean_dec(v___x_1710_);
v___x_1713_ = lean_box(0);
v_isShared_1714_ = v_isSharedCheck_1718_;
goto v_resetjp_1712_;
}
v_resetjp_1712_:
{
lean_object* v___x_1716_; 
if (v_isShared_1714_ == 0)
{
v___x_1716_ = v___x_1713_;
goto v_reusejp_1715_;
}
else
{
lean_object* v_reuseFailAlloc_1717_; 
v_reuseFailAlloc_1717_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1717_, 0, v_a_1711_);
v___x_1716_ = v_reuseFailAlloc_1717_;
goto v_reusejp_1715_;
}
v_reusejp_1715_:
{
return v___x_1716_;
}
}
}
else
{
lean_object* v_a_1719_; lean_object* v___x_1721_; uint8_t v_isShared_1722_; uint8_t v_isSharedCheck_1726_; 
v_a_1719_ = lean_ctor_get(v___x_1710_, 0);
v_isSharedCheck_1726_ = !lean_is_exclusive(v___x_1710_);
if (v_isSharedCheck_1726_ == 0)
{
v___x_1721_ = v___x_1710_;
v_isShared_1722_ = v_isSharedCheck_1726_;
goto v_resetjp_1720_;
}
else
{
lean_inc(v_a_1719_);
lean_dec(v___x_1710_);
v___x_1721_ = lean_box(0);
v_isShared_1722_ = v_isSharedCheck_1726_;
goto v_resetjp_1720_;
}
v_resetjp_1720_:
{
lean_object* v___x_1724_; 
if (v_isShared_1722_ == 0)
{
v___x_1724_ = v___x_1721_;
goto v_reusejp_1723_;
}
else
{
lean_object* v_reuseFailAlloc_1725_; 
v_reuseFailAlloc_1725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1725_, 0, v_a_1719_);
v___x_1724_ = v_reuseFailAlloc_1725_;
goto v_reusejp_1723_;
}
v_reusejp_1723_:
{
return v___x_1724_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg___boxed(lean_object* v_name_1727_, lean_object* v_bi_1728_, lean_object* v_type_1729_, lean_object* v_k_1730_, lean_object* v_kind_1731_, lean_object* v___y_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_){
_start:
{
uint8_t v_bi_boxed_1737_; uint8_t v_kind_boxed_1738_; lean_object* v_res_1739_; 
v_bi_boxed_1737_ = lean_unbox(v_bi_1728_);
v_kind_boxed_1738_ = lean_unbox(v_kind_1731_);
v_res_1739_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg(v_name_1727_, v_bi_boxed_1737_, v_type_1729_, v_k_1730_, v_kind_boxed_1738_, v___y_1732_, v___y_1733_, v___y_1734_, v___y_1735_);
lean_dec(v___y_1735_);
lean_dec_ref(v___y_1734_);
lean_dec(v___y_1733_);
lean_dec_ref(v___y_1732_);
return v_res_1739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(lean_object* v_name_1740_, lean_object* v_type_1741_, lean_object* v_k_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_){
_start:
{
uint8_t v___x_1748_; uint8_t v___x_1749_; lean_object* v___x_1750_; 
v___x_1748_ = 0;
v___x_1749_ = 0;
v___x_1750_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg(v_name_1740_, v___x_1748_, v_type_1741_, v_k_1742_, v___x_1749_, v___y_1743_, v___y_1744_, v___y_1745_, v___y_1746_);
return v___x_1750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg___boxed(lean_object* v_name_1751_, lean_object* v_type_1752_, lean_object* v_k_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_){
_start:
{
lean_object* v_res_1759_; 
v_res_1759_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v_name_1751_, v_type_1752_, v_k_1753_, v___y_1754_, v___y_1755_, v___y_1756_, v___y_1757_);
lean_dec(v___y_1757_);
lean_dec_ref(v___y_1756_);
lean_dec(v___y_1755_);
lean_dec_ref(v___y_1754_);
return v_res_1759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__0___boxed(lean_object** _args){
lean_object* v_a_1760_ = _args[0];
lean_object* v_i_1761_ = _args[1];
lean_object* v_xs_1762_ = _args[2];
lean_object* v_fvarx_1763_ = _args[3];
lean_object* v_ys_1764_ = _args[4];
lean_object* v_fixed_x27_1765_ = _args[5];
lean_object* v___x_1766_ = _args[6];
lean_object* v_numVars_1767_ = _args[7];
lean_object* v_fixed_1768_ = _args[8];
lean_object* v_k_1769_ = _args[9];
lean_object* v___x_1770_ = _args[10];
lean_object* v_fvary_1771_ = _args[11];
lean_object* v___y_1772_ = _args[12];
lean_object* v___y_1773_ = _args[13];
lean_object* v___y_1774_ = _args[14];
lean_object* v___y_1775_ = _args[15];
lean_object* v___y_1776_ = _args[16];
_start:
{
uint8_t v___x_1489__boxed_1777_; lean_object* v_res_1778_; 
v___x_1489__boxed_1777_ = lean_unbox(v___x_1766_);
v_res_1778_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__0(v_a_1760_, v_i_1761_, v_xs_1762_, v_fvarx_1763_, v_ys_1764_, v_fixed_x27_1765_, v___x_1489__boxed_1777_, v_numVars_1767_, v_fixed_1768_, v_k_1769_, v___x_1770_, v_fvary_1771_, v___y_1772_, v___y_1773_, v___y_1774_, v___y_1775_);
lean_dec(v___y_1775_);
lean_dec_ref(v___y_1774_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v_i_1761_);
lean_dec_ref(v_a_1760_);
return v_res_1778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1(lean_object* v_a_1780_, lean_object* v_a_1781_, lean_object* v_i_1782_, lean_object* v_xs_1783_, lean_object* v_ys_1784_, lean_object* v_fixed_x27_1785_, lean_object* v_numVars_1786_, lean_object* v_fixed_1787_, lean_object* v_k_1788_, lean_object* v___x_1789_, lean_object* v_fvarx_1790_, lean_object* v___y_1791_, lean_object* v___y_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_){
_start:
{
lean_object* v___x_1796_; lean_object* v___x_1797_; uint8_t v___x_1798_; lean_object* v___x_1799_; lean_object* v___f_1800_; lean_object* v___x_1807_; uint8_t v___x_1808_; 
v___x_1796_ = l_Lean_Expr_bindingBody_x21(v_a_1780_);
v___x_1797_ = lean_expr_instantiate1(v___x_1796_, v_fvarx_1790_);
lean_dec_ref(v___x_1796_);
v___x_1798_ = 0;
v___x_1799_ = lean_box(v___x_1798_);
lean_inc_ref(v___x_1797_);
lean_inc_ref(v_k_1788_);
lean_inc_ref(v_fixed_1787_);
lean_inc(v_numVars_1786_);
lean_inc_ref(v_fixed_x27_1785_);
lean_inc_ref(v_ys_1784_);
lean_inc_ref(v_fvarx_1790_);
lean_inc_ref(v_xs_1783_);
lean_inc(v_i_1782_);
lean_inc_ref(v_a_1781_);
v___f_1800_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__0___boxed), 17, 11);
lean_closure_set(v___f_1800_, 0, v_a_1781_);
lean_closure_set(v___f_1800_, 1, v_i_1782_);
lean_closure_set(v___f_1800_, 2, v_xs_1783_);
lean_closure_set(v___f_1800_, 3, v_fvarx_1790_);
lean_closure_set(v___f_1800_, 4, v_ys_1784_);
lean_closure_set(v___f_1800_, 5, v_fixed_x27_1785_);
lean_closure_set(v___f_1800_, 6, v___x_1799_);
lean_closure_set(v___f_1800_, 7, v_numVars_1786_);
lean_closure_set(v___f_1800_, 8, v_fixed_1787_);
lean_closure_set(v___f_1800_, 9, v_k_1788_);
lean_closure_set(v___f_1800_, 10, v___x_1797_);
v___x_1807_ = lean_array_get_size(v_fixed_1787_);
v___x_1808_ = lean_nat_dec_lt(v_i_1782_, v___x_1807_);
if (v___x_1808_ == 0)
{
lean_dec_ref(v___x_1797_);
lean_dec_ref(v_fvarx_1790_);
lean_dec_ref(v_k_1788_);
lean_dec_ref(v_fixed_1787_);
lean_dec(v_numVars_1786_);
lean_dec_ref(v_fixed_x27_1785_);
lean_dec_ref(v_ys_1784_);
lean_dec_ref(v_xs_1783_);
lean_dec(v_i_1782_);
goto v___jp_1801_;
}
else
{
lean_object* v___x_1809_; uint8_t v___x_1810_; 
v___x_1809_ = lean_array_fget_borrowed(v_fixed_1787_, v_i_1782_);
v___x_1810_ = lean_unbox(v___x_1809_);
if (v___x_1810_ == 0)
{
lean_dec_ref(v___x_1797_);
lean_dec_ref(v_fvarx_1790_);
lean_dec_ref(v_k_1788_);
lean_dec_ref(v_fixed_1787_);
lean_dec(v_numVars_1786_);
lean_dec_ref(v_fixed_x27_1785_);
lean_dec_ref(v_ys_1784_);
lean_dec_ref(v_xs_1783_);
lean_dec(v_i_1782_);
goto v___jp_1801_;
}
else
{
lean_object* v___x_1811_; uint8_t v___x_1812_; 
v___x_1811_ = l_Lean_Expr_bindingDomain_x21(v_a_1781_);
v___x_1812_ = lean_expr_eqv(v___x_1789_, v___x_1811_);
lean_dec_ref(v___x_1811_);
if (v___x_1812_ == 0)
{
lean_dec_ref(v___x_1797_);
lean_dec_ref(v_fvarx_1790_);
lean_dec_ref(v_k_1788_);
lean_dec_ref(v_fixed_1787_);
lean_dec(v_numVars_1786_);
lean_dec_ref(v_fixed_x27_1785_);
lean_dec_ref(v_ys_1784_);
lean_dec_ref(v_xs_1783_);
lean_dec(v_i_1782_);
goto v___jp_1801_;
}
else
{
lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; 
lean_dec_ref(v___f_1800_);
v___x_1813_ = l_Lean_Expr_bindingBody_x21(v_a_1781_);
lean_dec_ref(v_a_1781_);
v___x_1814_ = lean_expr_instantiate1(v___x_1813_, v_fvarx_1790_);
lean_dec_ref(v___x_1813_);
v___x_1815_ = lean_unsigned_to_nat(1u);
v___x_1816_ = lean_nat_add(v_i_1782_, v___x_1815_);
lean_dec(v_i_1782_);
lean_inc_ref(v_fvarx_1790_);
v___x_1817_ = lean_array_push(v_xs_1783_, v_fvarx_1790_);
v___x_1818_ = lean_array_push(v_ys_1784_, v_fvarx_1790_);
lean_inc(v___x_1809_);
v___x_1819_ = lean_array_push(v_fixed_x27_1785_, v___x_1809_);
v___x_1820_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg(v_numVars_1786_, v_fixed_1787_, v_k_1788_, v___x_1816_, v___x_1797_, v___x_1814_, v___x_1817_, v___x_1818_, v___x_1819_, v___y_1791_, v___y_1792_, v___y_1793_, v___y_1794_);
return v___x_1820_;
}
}
}
v___jp_1801_:
{
lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; 
v___x_1802_ = l_Lean_Expr_bindingName_x21(v_a_1781_);
v___x_1803_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1___closed__0));
v___x_1804_ = lean_name_append_after(v___x_1802_, v___x_1803_);
v___x_1805_ = l_Lean_Expr_bindingDomain_x21(v_a_1781_);
lean_dec_ref(v_a_1781_);
v___x_1806_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v___x_1804_, v___x_1805_, v___f_1800_, v___y_1791_, v___y_1792_, v___y_1793_, v___y_1794_);
return v___x_1806_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1___boxed(lean_object* v_a_1821_, lean_object* v_a_1822_, lean_object* v_i_1823_, lean_object* v_xs_1824_, lean_object* v_ys_1825_, lean_object* v_fixed_x27_1826_, lean_object* v_numVars_1827_, lean_object* v_fixed_1828_, lean_object* v_k_1829_, lean_object* v___x_1830_, lean_object* v_fvarx_1831_, lean_object* v___y_1832_, lean_object* v___y_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_){
_start:
{
lean_object* v_res_1837_; 
v_res_1837_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1(v_a_1821_, v_a_1822_, v_i_1823_, v_xs_1824_, v_ys_1825_, v_fixed_x27_1826_, v_numVars_1827_, v_fixed_1828_, v_k_1829_, v___x_1830_, v_fvarx_1831_, v___y_1832_, v___y_1833_, v___y_1834_, v___y_1835_);
lean_dec(v___y_1835_);
lean_dec_ref(v___y_1834_);
lean_dec(v___y_1833_);
lean_dec_ref(v___y_1832_);
lean_dec_ref(v___x_1830_);
lean_dec_ref(v_a_1821_);
return v_res_1837_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__1(void){
_start:
{
lean_object* v___x_1839_; lean_object* v___x_1840_; 
v___x_1839_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__0));
v___x_1840_ = l_Lean_stringToMessageData(v___x_1839_);
return v___x_1840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg(lean_object* v_numVars_1841_, lean_object* v_fixed_1842_, lean_object* v_k_1843_, lean_object* v_i_1844_, lean_object* v_ftyx_1845_, lean_object* v_ftyy_1846_, lean_object* v_xs_1847_, lean_object* v_ys_1848_, lean_object* v_fixed_x27_1849_, lean_object* v_a_1850_, lean_object* v_a_1851_, lean_object* v_a_1852_, lean_object* v_a_1853_){
_start:
{
uint8_t v___x_1855_; 
v___x_1855_ = lean_nat_dec_lt(v_i_1844_, v_numVars_1841_);
if (v___x_1855_ == 0)
{
lean_object* v___x_1856_; 
lean_dec_ref(v_ftyy_1846_);
lean_dec_ref(v_ftyx_1845_);
lean_dec(v_i_1844_);
lean_dec_ref(v_fixed_1842_);
lean_dec(v_numVars_1841_);
lean_inc(v_a_1853_);
lean_inc_ref(v_a_1852_);
lean_inc(v_a_1851_);
lean_inc_ref(v_a_1850_);
v___x_1856_ = lean_apply_8(v_k_1843_, v_xs_1847_, v_ys_1848_, v_fixed_x27_1849_, v_a_1850_, v_a_1851_, v_a_1852_, v_a_1853_, lean_box(0));
return v___x_1856_;
}
else
{
lean_object* v___x_1857_; 
v___x_1857_ = l_Lean_Meta_whnfD(v_ftyx_1845_, v_a_1850_, v_a_1851_, v_a_1852_, v_a_1853_);
if (lean_obj_tag(v___x_1857_) == 0)
{
lean_object* v_a_1858_; lean_object* v___x_1859_; 
v_a_1858_ = lean_ctor_get(v___x_1857_, 0);
lean_inc(v_a_1858_);
lean_dec_ref_known(v___x_1857_, 1);
v___x_1859_ = l_Lean_Meta_whnfD(v_ftyy_1846_, v_a_1850_, v_a_1851_, v_a_1852_, v_a_1853_);
if (lean_obj_tag(v___x_1859_) == 0)
{
lean_object* v_a_1860_; lean_object* v___y_1862_; lean_object* v___y_1863_; lean_object* v___y_1864_; lean_object* v___y_1865_; uint8_t v___x_1870_; 
v_a_1860_ = lean_ctor_get(v___x_1859_, 0);
lean_inc(v_a_1860_);
lean_dec_ref_known(v___x_1859_, 1);
v___x_1870_ = l_Lean_Expr_isForall(v_a_1858_);
if (v___x_1870_ == 0)
{
lean_object* v___x_1871_; lean_object* v___x_1872_; 
v___x_1871_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___closed__1);
v___x_1872_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v___x_1871_, v_a_1850_, v_a_1851_, v_a_1852_, v_a_1853_);
if (lean_obj_tag(v___x_1872_) == 0)
{
lean_dec_ref_known(v___x_1872_, 1);
v___y_1862_ = v_a_1850_;
v___y_1863_ = v_a_1851_;
v___y_1864_ = v_a_1852_;
v___y_1865_ = v_a_1853_;
goto v___jp_1861_;
}
else
{
lean_object* v_a_1873_; lean_object* v___x_1875_; uint8_t v_isShared_1876_; uint8_t v_isSharedCheck_1880_; 
lean_dec(v_a_1860_);
lean_dec(v_a_1858_);
lean_dec_ref(v_fixed_x27_1849_);
lean_dec_ref(v_ys_1848_);
lean_dec_ref(v_xs_1847_);
lean_dec(v_i_1844_);
lean_dec_ref(v_k_1843_);
lean_dec_ref(v_fixed_1842_);
lean_dec(v_numVars_1841_);
v_a_1873_ = lean_ctor_get(v___x_1872_, 0);
v_isSharedCheck_1880_ = !lean_is_exclusive(v___x_1872_);
if (v_isSharedCheck_1880_ == 0)
{
v___x_1875_ = v___x_1872_;
v_isShared_1876_ = v_isSharedCheck_1880_;
goto v_resetjp_1874_;
}
else
{
lean_inc(v_a_1873_);
lean_dec(v___x_1872_);
v___x_1875_ = lean_box(0);
v_isShared_1876_ = v_isSharedCheck_1880_;
goto v_resetjp_1874_;
}
v_resetjp_1874_:
{
lean_object* v___x_1878_; 
if (v_isShared_1876_ == 0)
{
v___x_1878_ = v___x_1875_;
goto v_reusejp_1877_;
}
else
{
lean_object* v_reuseFailAlloc_1879_; 
v_reuseFailAlloc_1879_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1879_, 0, v_a_1873_);
v___x_1878_ = v_reuseFailAlloc_1879_;
goto v_reusejp_1877_;
}
v_reusejp_1877_:
{
return v___x_1878_;
}
}
}
}
else
{
v___y_1862_ = v_a_1850_;
v___y_1863_ = v_a_1851_;
v___y_1864_ = v_a_1852_;
v___y_1865_ = v_a_1853_;
goto v___jp_1861_;
}
v___jp_1861_:
{
lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___f_1868_; lean_object* v___x_1869_; 
v___x_1866_ = l_Lean_Expr_bindingName_x21(v_a_1858_);
v___x_1867_ = l_Lean_Expr_bindingDomain_x21(v_a_1858_);
lean_inc_ref(v___x_1867_);
v___f_1868_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__1___boxed), 16, 10);
lean_closure_set(v___f_1868_, 0, v_a_1858_);
lean_closure_set(v___f_1868_, 1, v_a_1860_);
lean_closure_set(v___f_1868_, 2, v_i_1844_);
lean_closure_set(v___f_1868_, 3, v_xs_1847_);
lean_closure_set(v___f_1868_, 4, v_ys_1848_);
lean_closure_set(v___f_1868_, 5, v_fixed_x27_1849_);
lean_closure_set(v___f_1868_, 6, v_numVars_1841_);
lean_closure_set(v___f_1868_, 7, v_fixed_1842_);
lean_closure_set(v___f_1868_, 8, v_k_1843_);
lean_closure_set(v___f_1868_, 9, v___x_1867_);
v___x_1869_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v___x_1866_, v___x_1867_, v___f_1868_, v___y_1862_, v___y_1863_, v___y_1864_, v___y_1865_);
return v___x_1869_;
}
}
else
{
lean_object* v_a_1881_; lean_object* v___x_1883_; uint8_t v_isShared_1884_; uint8_t v_isSharedCheck_1888_; 
lean_dec(v_a_1858_);
lean_dec_ref(v_fixed_x27_1849_);
lean_dec_ref(v_ys_1848_);
lean_dec_ref(v_xs_1847_);
lean_dec(v_i_1844_);
lean_dec_ref(v_k_1843_);
lean_dec_ref(v_fixed_1842_);
lean_dec(v_numVars_1841_);
v_a_1881_ = lean_ctor_get(v___x_1859_, 0);
v_isSharedCheck_1888_ = !lean_is_exclusive(v___x_1859_);
if (v_isSharedCheck_1888_ == 0)
{
v___x_1883_ = v___x_1859_;
v_isShared_1884_ = v_isSharedCheck_1888_;
goto v_resetjp_1882_;
}
else
{
lean_inc(v_a_1881_);
lean_dec(v___x_1859_);
v___x_1883_ = lean_box(0);
v_isShared_1884_ = v_isSharedCheck_1888_;
goto v_resetjp_1882_;
}
v_resetjp_1882_:
{
lean_object* v___x_1886_; 
if (v_isShared_1884_ == 0)
{
v___x_1886_ = v___x_1883_;
goto v_reusejp_1885_;
}
else
{
lean_object* v_reuseFailAlloc_1887_; 
v_reuseFailAlloc_1887_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1887_, 0, v_a_1881_);
v___x_1886_ = v_reuseFailAlloc_1887_;
goto v_reusejp_1885_;
}
v_reusejp_1885_:
{
return v___x_1886_;
}
}
}
}
else
{
lean_object* v_a_1889_; lean_object* v___x_1891_; uint8_t v_isShared_1892_; uint8_t v_isSharedCheck_1896_; 
lean_dec_ref(v_fixed_x27_1849_);
lean_dec_ref(v_ys_1848_);
lean_dec_ref(v_xs_1847_);
lean_dec_ref(v_ftyy_1846_);
lean_dec(v_i_1844_);
lean_dec_ref(v_k_1843_);
lean_dec_ref(v_fixed_1842_);
lean_dec(v_numVars_1841_);
v_a_1889_ = lean_ctor_get(v___x_1857_, 0);
v_isSharedCheck_1896_ = !lean_is_exclusive(v___x_1857_);
if (v_isSharedCheck_1896_ == 0)
{
v___x_1891_ = v___x_1857_;
v_isShared_1892_ = v_isSharedCheck_1896_;
goto v_resetjp_1890_;
}
else
{
lean_inc(v_a_1889_);
lean_dec(v___x_1857_);
v___x_1891_ = lean_box(0);
v_isShared_1892_ = v_isSharedCheck_1896_;
goto v_resetjp_1890_;
}
v_resetjp_1890_:
{
lean_object* v___x_1894_; 
if (v_isShared_1892_ == 0)
{
v___x_1894_ = v___x_1891_;
goto v_reusejp_1893_;
}
else
{
lean_object* v_reuseFailAlloc_1895_; 
v_reuseFailAlloc_1895_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1895_, 0, v_a_1889_);
v___x_1894_ = v_reuseFailAlloc_1895_;
goto v_reusejp_1893_;
}
v_reusejp_1893_:
{
return v___x_1894_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___lam__0(lean_object* v_a_1897_, lean_object* v_i_1898_, lean_object* v_xs_1899_, lean_object* v_fvarx_1900_, lean_object* v_ys_1901_, lean_object* v_fixed_x27_1902_, uint8_t v___x_1903_, lean_object* v_numVars_1904_, lean_object* v_fixed_1905_, lean_object* v_k_1906_, lean_object* v___x_1907_, lean_object* v_fvary_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_){
_start:
{
lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; 
v___x_1914_ = l_Lean_Expr_bindingBody_x21(v_a_1897_);
v___x_1915_ = lean_expr_instantiate1(v___x_1914_, v_fvary_1908_);
lean_dec_ref(v___x_1914_);
v___x_1916_ = lean_unsigned_to_nat(1u);
v___x_1917_ = lean_nat_add(v_i_1898_, v___x_1916_);
v___x_1918_ = lean_array_push(v_xs_1899_, v_fvarx_1900_);
v___x_1919_ = lean_array_push(v_ys_1901_, v_fvary_1908_);
v___x_1920_ = lean_box(v___x_1903_);
v___x_1921_ = lean_array_push(v_fixed_x27_1902_, v___x_1920_);
v___x_1922_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg(v_numVars_1904_, v_fixed_1905_, v_k_1906_, v___x_1917_, v___x_1907_, v___x_1915_, v___x_1918_, v___x_1919_, v___x_1921_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_);
return v___x_1922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg___boxed(lean_object* v_numVars_1923_, lean_object* v_fixed_1924_, lean_object* v_k_1925_, lean_object* v_i_1926_, lean_object* v_ftyx_1927_, lean_object* v_ftyy_1928_, lean_object* v_xs_1929_, lean_object* v_ys_1930_, lean_object* v_fixed_x27_1931_, lean_object* v_a_1932_, lean_object* v_a_1933_, lean_object* v_a_1934_, lean_object* v_a_1935_, lean_object* v_a_1936_){
_start:
{
lean_object* v_res_1937_; 
v_res_1937_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg(v_numVars_1923_, v_fixed_1924_, v_k_1925_, v_i_1926_, v_ftyx_1927_, v_ftyy_1928_, v_xs_1929_, v_ys_1930_, v_fixed_x27_1931_, v_a_1932_, v_a_1933_, v_a_1934_, v_a_1935_);
lean_dec(v_a_1935_);
lean_dec_ref(v_a_1934_);
lean_dec(v_a_1933_);
lean_dec_ref(v_a_1932_);
return v_res_1937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop(lean_object* v_00_u03b1_1938_, lean_object* v_numVars_1939_, lean_object* v_fixed_1940_, lean_object* v_k_1941_, lean_object* v_i_1942_, lean_object* v_ftyx_1943_, lean_object* v_ftyy_1944_, lean_object* v_xs_1945_, lean_object* v_ys_1946_, lean_object* v_fixed_x27_1947_, lean_object* v_a_1948_, lean_object* v_a_1949_, lean_object* v_a_1950_, lean_object* v_a_1951_){
_start:
{
lean_object* v___x_1953_; 
v___x_1953_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg(v_numVars_1939_, v_fixed_1940_, v_k_1941_, v_i_1942_, v_ftyx_1943_, v_ftyy_1944_, v_xs_1945_, v_ys_1946_, v_fixed_x27_1947_, v_a_1948_, v_a_1949_, v_a_1950_, v_a_1951_);
return v___x_1953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___boxed(lean_object* v_00_u03b1_1954_, lean_object* v_numVars_1955_, lean_object* v_fixed_1956_, lean_object* v_k_1957_, lean_object* v_i_1958_, lean_object* v_ftyx_1959_, lean_object* v_ftyy_1960_, lean_object* v_xs_1961_, lean_object* v_ys_1962_, lean_object* v_fixed_x27_1963_, lean_object* v_a_1964_, lean_object* v_a_1965_, lean_object* v_a_1966_, lean_object* v_a_1967_, lean_object* v_a_1968_){
_start:
{
lean_object* v_res_1969_; 
v_res_1969_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop(v_00_u03b1_1954_, v_numVars_1955_, v_fixed_1956_, v_k_1957_, v_i_1958_, v_ftyx_1959_, v_ftyy_1960_, v_xs_1961_, v_ys_1962_, v_fixed_x27_1963_, v_a_1964_, v_a_1965_, v_a_1966_, v_a_1967_);
lean_dec(v_a_1967_);
lean_dec_ref(v_a_1966_);
lean_dec(v_a_1965_);
lean_dec_ref(v_a_1964_);
return v_res_1969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0(lean_object* v_00_u03b1_1970_, lean_object* v_name_1971_, uint8_t v_bi_1972_, lean_object* v_type_1973_, lean_object* v_k_1974_, uint8_t v_kind_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_){
_start:
{
lean_object* v___x_1981_; 
v___x_1981_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___redArg(v_name_1971_, v_bi_1972_, v_type_1973_, v_k_1974_, v_kind_1975_, v___y_1976_, v___y_1977_, v___y_1978_, v___y_1979_);
return v___x_1981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1982_, lean_object* v_name_1983_, lean_object* v_bi_1984_, lean_object* v_type_1985_, lean_object* v_k_1986_, lean_object* v_kind_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_){
_start:
{
uint8_t v_bi_boxed_1993_; uint8_t v_kind_boxed_1994_; lean_object* v_res_1995_; 
v_bi_boxed_1993_ = lean_unbox(v_bi_1984_);
v_kind_boxed_1994_ = lean_unbox(v_kind_1987_);
v_res_1995_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0_spec__0(v_00_u03b1_1982_, v_name_1983_, v_bi_boxed_1993_, v_type_1985_, v_k_1986_, v_kind_boxed_1994_, v___y_1988_, v___y_1989_, v___y_1990_, v___y_1991_);
lean_dec(v___y_1991_);
lean_dec_ref(v___y_1990_);
lean_dec(v___y_1989_);
lean_dec_ref(v___y_1988_);
return v_res_1995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0(lean_object* v_00_u03b1_1996_, lean_object* v_name_1997_, lean_object* v_type_1998_, lean_object* v_k_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_){
_start:
{
lean_object* v___x_2005_; 
v___x_2005_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v_name_1997_, v_type_1998_, v_k_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_);
return v___x_2005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___boxed(lean_object* v_00_u03b1_2006_, lean_object* v_name_2007_, lean_object* v_type_2008_, lean_object* v_k_2009_, lean_object* v___y_2010_, lean_object* v___y_2011_, lean_object* v___y_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_){
_start:
{
lean_object* v_res_2015_; 
v_res_2015_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0(v_00_u03b1_2006_, v_name_2007_, v_type_2008_, v_k_2009_, v___y_2010_, v___y_2011_, v___y_2012_, v___y_2013_);
lean_dec(v___y_2013_);
lean_dec_ref(v___y_2012_);
lean_dec(v___y_2011_);
lean_dec_ref(v___y_2010_);
return v_res_2015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg(lean_object* v_fty_2018_, lean_object* v_numVars_2019_, lean_object* v_fixed_2020_, lean_object* v_k_2021_, lean_object* v_a_2022_, lean_object* v_a_2023_, lean_object* v_a_2024_, lean_object* v_a_2025_){
_start:
{
lean_object* v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; 
v___x_2027_ = lean_unsigned_to_nat(0u);
v___x_2028_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg___closed__0));
lean_inc_ref(v_fty_2018_);
v___x_2029_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop___redArg(v_numVars_2019_, v_fixed_2020_, v_k_2021_, v___x_2027_, v_fty_2018_, v_fty_2018_, v___x_2028_, v___x_2028_, v___x_2028_, v_a_2022_, v_a_2023_, v_a_2024_, v_a_2025_);
return v___x_2029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg___boxed(lean_object* v_fty_2030_, lean_object* v_numVars_2031_, lean_object* v_fixed_2032_, lean_object* v_k_2033_, lean_object* v_a_2034_, lean_object* v_a_2035_, lean_object* v_a_2036_, lean_object* v_a_2037_, lean_object* v_a_2038_){
_start:
{
lean_object* v_res_2039_; 
v_res_2039_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg(v_fty_2030_, v_numVars_2031_, v_fixed_2032_, v_k_2033_, v_a_2034_, v_a_2035_, v_a_2036_, v_a_2037_);
lean_dec(v_a_2037_);
lean_dec_ref(v_a_2036_);
lean_dec(v_a_2035_);
lean_dec_ref(v_a_2034_);
return v_res_2039_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope(lean_object* v_00_u03b1_2040_, lean_object* v_fty_2041_, lean_object* v_numVars_2042_, lean_object* v_fixed_2043_, lean_object* v_k_2044_, lean_object* v_a_2045_, lean_object* v_a_2046_, lean_object* v_a_2047_, lean_object* v_a_2048_){
_start:
{
lean_object* v___x_2050_; 
v___x_2050_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg(v_fty_2041_, v_numVars_2042_, v_fixed_2043_, v_k_2044_, v_a_2045_, v_a_2046_, v_a_2047_, v_a_2048_);
return v___x_2050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___boxed(lean_object* v_00_u03b1_2051_, lean_object* v_fty_2052_, lean_object* v_numVars_2053_, lean_object* v_fixed_2054_, lean_object* v_k_2055_, lean_object* v_a_2056_, lean_object* v_a_2057_, lean_object* v_a_2058_, lean_object* v_a_2059_, lean_object* v_a_2060_){
_start:
{
lean_object* v_res_2061_; 
v_res_2061_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope(v_00_u03b1_2051_, v_fty_2052_, v_numVars_2053_, v_fixed_2054_, v_k_2055_, v_a_2056_, v_a_2057_, v_a_2058_, v_a_2059_);
lean_dec(v_a_2059_);
lean_dec_ref(v_a_2058_);
lean_dec(v_a_2057_);
lean_dec_ref(v_a_2056_);
return v_res_2061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__1(size_t v_sz_2062_, size_t v_i_2063_, lean_object* v_bs_2064_){
_start:
{
uint8_t v___x_2065_; 
v___x_2065_ = lean_usize_dec_lt(v_i_2063_, v_sz_2062_);
if (v___x_2065_ == 0)
{
return v_bs_2064_;
}
else
{
lean_object* v_v_2066_; lean_object* v_fst_2067_; lean_object* v___x_2068_; lean_object* v_bs_x27_2069_; size_t v___x_2070_; size_t v___x_2071_; lean_object* v___x_2072_; 
v_v_2066_ = lean_array_uget_borrowed(v_bs_2064_, v_i_2063_);
v_fst_2067_ = lean_ctor_get(v_v_2066_, 0);
lean_inc(v_fst_2067_);
v___x_2068_ = lean_unsigned_to_nat(0u);
v_bs_x27_2069_ = lean_array_uset(v_bs_2064_, v_i_2063_, v___x_2068_);
v___x_2070_ = ((size_t)1ULL);
v___x_2071_ = lean_usize_add(v_i_2063_, v___x_2070_);
v___x_2072_ = lean_array_uset(v_bs_x27_2069_, v_i_2063_, v_fst_2067_);
v_i_2063_ = v___x_2071_;
v_bs_2064_ = v___x_2072_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__1___boxed(lean_object* v_sz_2074_, lean_object* v_i_2075_, lean_object* v_bs_2076_){
_start:
{
size_t v_sz_boxed_2077_; size_t v_i_boxed_2078_; lean_object* v_res_2079_; 
v_sz_boxed_2077_ = lean_unbox_usize(v_sz_2074_);
lean_dec(v_sz_2074_);
v_i_boxed_2078_ = lean_unbox_usize(v_i_2075_);
lean_dec(v_i_2075_);
v_res_2079_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__1(v_sz_boxed_2077_, v_i_boxed_2078_, v_bs_2076_);
return v_res_2079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0_spec__0(lean_object* v_eqs_2080_, lean_object* v_as_2081_, size_t v_i_2082_, size_t v_stop_2083_, lean_object* v_b_2084_){
_start:
{
lean_object* v___y_2086_; uint8_t v___x_2090_; 
v___x_2090_ = lean_usize_dec_eq(v_i_2082_, v_stop_2083_);
if (v___x_2090_ == 0)
{
lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; 
v___x_2091_ = lean_array_uget_borrowed(v_as_2081_, v_i_2082_);
v___x_2092_ = lean_box(0);
v___x_2093_ = lean_array_get_borrowed(v___x_2092_, v_eqs_2080_, v___x_2091_);
if (lean_obj_tag(v___x_2093_) == 0)
{
v___y_2086_ = v_b_2084_;
goto v___jp_2085_;
}
else
{
lean_object* v_val_2094_; lean_object* v___x_2095_; 
v_val_2094_ = lean_ctor_get(v___x_2093_, 0);
lean_inc(v_val_2094_);
v___x_2095_ = lean_array_push(v_b_2084_, v_val_2094_);
v___y_2086_ = v___x_2095_;
goto v___jp_2085_;
}
}
else
{
return v_b_2084_;
}
v___jp_2085_:
{
size_t v___x_2087_; size_t v___x_2088_; 
v___x_2087_ = ((size_t)1ULL);
v___x_2088_ = lean_usize_add(v_i_2082_, v___x_2087_);
v_i_2082_ = v___x_2088_;
v_b_2084_ = v___y_2086_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0_spec__0___boxed(lean_object* v_eqs_2096_, lean_object* v_as_2097_, lean_object* v_i_2098_, lean_object* v_stop_2099_, lean_object* v_b_2100_){
_start:
{
size_t v_i_boxed_2101_; size_t v_stop_boxed_2102_; lean_object* v_res_2103_; 
v_i_boxed_2101_ = lean_unbox_usize(v_i_2098_);
lean_dec(v_i_2098_);
v_stop_boxed_2102_ = lean_unbox_usize(v_stop_2099_);
lean_dec(v_stop_2099_);
v_res_2103_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0_spec__0(v_eqs_2096_, v_as_2097_, v_i_boxed_2101_, v_stop_boxed_2102_, v_b_2100_);
lean_dec_ref(v_as_2097_);
lean_dec_ref(v_eqs_2096_);
return v_res_2103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0(lean_object* v_eqs_2106_, lean_object* v_as_2107_, lean_object* v_start_2108_, lean_object* v_stop_2109_){
_start:
{
lean_object* v___x_2110_; uint8_t v___x_2111_; 
v___x_2110_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0___closed__0));
v___x_2111_ = lean_nat_dec_lt(v_start_2108_, v_stop_2109_);
if (v___x_2111_ == 0)
{
return v___x_2110_;
}
else
{
lean_object* v___x_2112_; uint8_t v___x_2113_; 
v___x_2112_ = lean_array_get_size(v_as_2107_);
v___x_2113_ = lean_nat_dec_le(v_stop_2109_, v___x_2112_);
if (v___x_2113_ == 0)
{
uint8_t v___x_2114_; 
v___x_2114_ = lean_nat_dec_lt(v_start_2108_, v___x_2112_);
if (v___x_2114_ == 0)
{
return v___x_2110_;
}
else
{
size_t v___x_2115_; size_t v___x_2116_; lean_object* v___x_2117_; 
v___x_2115_ = lean_usize_of_nat(v_start_2108_);
v___x_2116_ = lean_usize_of_nat(v___x_2112_);
v___x_2117_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0_spec__0(v_eqs_2106_, v_as_2107_, v___x_2115_, v___x_2116_, v___x_2110_);
return v___x_2117_;
}
}
else
{
size_t v___x_2118_; size_t v___x_2119_; lean_object* v___x_2120_; 
v___x_2118_ = lean_usize_of_nat(v_start_2108_);
v___x_2119_ = lean_usize_of_nat(v_stop_2109_);
v___x_2120_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0_spec__0(v_eqs_2106_, v_as_2107_, v___x_2118_, v___x_2119_, v___x_2110_);
return v___x_2120_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0___boxed(lean_object* v_eqs_2121_, lean_object* v_as_2122_, lean_object* v_start_2123_, lean_object* v_stop_2124_){
_start:
{
lean_object* v_res_2125_; 
v_res_2125_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0(v_eqs_2121_, v_as_2122_, v_start_2123_, v_stop_2124_);
lean_dec(v_stop_2124_);
lean_dec(v_start_2123_);
lean_dec_ref(v_as_2122_);
lean_dec_ref(v_eqs_2121_);
return v_res_2125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__2(size_t v_sz_2126_, size_t v_i_2127_, lean_object* v_bs_2128_){
_start:
{
uint8_t v___x_2129_; 
v___x_2129_ = lean_usize_dec_lt(v_i_2127_, v_sz_2126_);
if (v___x_2129_ == 0)
{
return v_bs_2128_;
}
else
{
lean_object* v_v_2130_; lean_object* v_snd_2131_; lean_object* v_snd_2132_; lean_object* v___x_2133_; lean_object* v_bs_x27_2134_; size_t v___x_2135_; size_t v___x_2136_; lean_object* v___x_2137_; 
v_v_2130_ = lean_array_uget_borrowed(v_bs_2128_, v_i_2127_);
v_snd_2131_ = lean_ctor_get(v_v_2130_, 1);
v_snd_2132_ = lean_ctor_get(v_snd_2131_, 1);
lean_inc(v_snd_2132_);
v___x_2133_ = lean_unsigned_to_nat(0u);
v_bs_x27_2134_ = lean_array_uset(v_bs_2128_, v_i_2127_, v___x_2133_);
v___x_2135_ = ((size_t)1ULL);
v___x_2136_ = lean_usize_add(v_i_2127_, v___x_2135_);
v___x_2137_ = lean_array_uset(v_bs_x27_2134_, v_i_2127_, v_snd_2132_);
v_i_2127_ = v___x_2136_;
v_bs_2128_ = v___x_2137_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__2___boxed(lean_object* v_sz_2139_, lean_object* v_i_2140_, lean_object* v_bs_2141_){
_start:
{
size_t v_sz_boxed_2142_; size_t v_i_boxed_2143_; lean_object* v_res_2144_; 
v_sz_boxed_2142_ = lean_unbox_usize(v_sz_2139_);
lean_dec(v_sz_2139_);
v_i_boxed_2143_ = lean_unbox_usize(v_i_2140_);
lean_dec(v_i_2140_);
v_res_2144_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__2(v_sz_boxed_2142_, v_i_boxed_2143_, v_bs_2141_);
return v_res_2144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___boxed(lean_object** _args){
lean_object* v_sz_2153_ = _args[0];
lean_object* v___x_2154_ = _args[1];
lean_object* v_deps_2155_ = _args[2];
lean_object* v_kinds_2156_ = _args[3];
lean_object* v_eqs_2157_ = _args[4];
lean_object* v_info_2158_ = _args[5];
lean_object* v_xs_2159_ = _args[6];
lean_object* v_ys_2160_ = _args[7];
lean_object* v_fixedParams_2161_ = _args[8];
lean_object* v_k_2162_ = _args[9];
lean_object* v___x_2163_ = _args[10];
lean_object* v___x_2164_ = _args[11];
lean_object* v_a_2165_ = _args[12];
lean_object* v_h_2166_ = _args[13];
lean_object* v___y_2167_ = _args[14];
lean_object* v___y_2168_ = _args[15];
lean_object* v___y_2169_ = _args[16];
lean_object* v___y_2170_ = _args[17];
lean_object* v___y_2171_ = _args[18];
_start:
{
size_t v_sz_boxed_2172_; size_t v___x_2234__boxed_2173_; uint8_t v___x_2236__boxed_2174_; lean_object* v_res_2175_; 
v_sz_boxed_2172_ = lean_unbox_usize(v_sz_2153_);
lean_dec(v_sz_2153_);
v___x_2234__boxed_2173_ = lean_unbox_usize(v___x_2154_);
lean_dec(v___x_2154_);
v___x_2236__boxed_2174_ = lean_unbox(v___x_2164_);
v_res_2175_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1(v_sz_boxed_2172_, v___x_2234__boxed_2173_, v_deps_2155_, v_kinds_2156_, v_eqs_2157_, v_info_2158_, v_xs_2159_, v_ys_2160_, v_fixedParams_2161_, v_k_2162_, v___x_2163_, v___x_2236__boxed_2174_, v_a_2165_, v_h_2166_, v___y_2167_, v___y_2168_, v___y_2169_, v___y_2170_);
lean_dec(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec(v___y_2168_);
lean_dec_ref(v___y_2167_);
return v_res_2175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg(lean_object* v_info_2176_, lean_object* v_xs_2177_, lean_object* v_ys_2178_, lean_object* v_fixedParams_2179_, lean_object* v_k_2180_, lean_object* v_i_2181_, lean_object* v_kinds_2182_, lean_object* v_eqs_2183_, lean_object* v_a_2184_, lean_object* v_a_2185_, lean_object* v_a_2186_, lean_object* v_a_2187_){
_start:
{
lean_object* v___x_2189_; uint8_t v___x_2190_; 
v___x_2189_ = lean_array_get_size(v_xs_2177_);
v___x_2190_ = lean_nat_dec_lt(v_i_2181_, v___x_2189_);
if (v___x_2190_ == 0)
{
lean_object* v___x_2191_; 
lean_dec(v_i_2181_);
lean_dec_ref(v_fixedParams_2179_);
lean_dec_ref(v_ys_2178_);
lean_dec_ref(v_xs_2177_);
lean_dec_ref(v_info_2176_);
lean_inc(v_a_2187_);
lean_inc_ref(v_a_2186_);
lean_inc(v_a_2185_);
lean_inc_ref(v_a_2184_);
v___x_2191_ = lean_apply_7(v_k_2180_, v_kinds_2182_, v_eqs_2183_, v_a_2184_, v_a_2185_, v_a_2186_, v_a_2187_, lean_box(0));
return v___x_2191_;
}
else
{
uint8_t v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; uint8_t v___x_2195_; 
v___x_2192_ = 0;
v___x_2193_ = lean_box(v___x_2192_);
v___x_2194_ = lean_array_get(v___x_2193_, v_fixedParams_2179_, v_i_2181_);
lean_dec(v___x_2193_);
v___x_2195_ = lean_unbox(v___x_2194_);
if (v___x_2195_ == 0)
{
lean_object* v_paramInfo_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v_backDeps_2199_; lean_object* v___x_2200_; lean_object* v_x_2201_; lean_object* v_y_2202_; lean_object* v___x_2203_; 
v_paramInfo_2196_ = lean_ctor_get(v_info_2176_, 0);
v___x_2197_ = l_Lean_Meta_instInhabitedParamInfo_default;
v___x_2198_ = lean_array_get_borrowed(v___x_2197_, v_paramInfo_2196_, v_i_2181_);
v_backDeps_2199_ = lean_ctor_get(v___x_2198_, 0);
v___x_2200_ = l_Lean_instInhabitedExpr;
v_x_2201_ = lean_array_get_borrowed(v___x_2200_, v_xs_2177_, v_i_2181_);
v_y_2202_ = lean_array_get_borrowed(v___x_2200_, v_ys_2178_, v_i_2181_);
lean_inc(v_y_2202_);
lean_inc(v_x_2201_);
v___x_2203_ = l_Lean_Meta_mkEqHEq(v_x_2201_, v_y_2202_, v_a_2184_, v_a_2185_, v_a_2186_, v_a_2187_);
if (lean_obj_tag(v___x_2203_) == 0)
{
lean_object* v_a_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v_deps_2207_; size_t v_sz_2208_; size_t v___x_2209_; lean_object* v___x_2210_; uint8_t v___x_2211_; uint8_t v___x_2212_; lean_object* v___x_2213_; 
v_a_2204_ = lean_ctor_get(v___x_2203_, 0);
lean_inc(v_a_2204_);
lean_dec_ref_known(v___x_2203_, 1);
v___x_2205_ = lean_array_get_size(v_backDeps_2199_);
v___x_2206_ = lean_unsigned_to_nat(0u);
v_deps_2207_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__0(v_eqs_2183_, v_backDeps_2199_, v___x_2206_, v___x_2205_);
v_sz_2208_ = lean_array_size(v_deps_2207_);
v___x_2209_ = ((size_t)0ULL);
lean_inc_ref(v_deps_2207_);
v___x_2210_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__1(v_sz_2208_, v___x_2209_, v_deps_2207_);
v___x_2211_ = 1;
v___x_2212_ = lean_unbox(v___x_2194_);
v___x_2213_ = l_Lean_Meta_mkForallFVars(v___x_2210_, v_a_2204_, v___x_2212_, v___x_2190_, v___x_2190_, v___x_2211_, v_a_2184_, v_a_2185_, v_a_2186_, v_a_2187_);
lean_dec_ref(v___x_2210_);
if (lean_obj_tag(v___x_2213_) == 0)
{
lean_object* v_a_2214_; lean_object* v___x_2215_; 
v_a_2214_ = lean_ctor_get(v___x_2213_, 0);
lean_inc(v_a_2214_);
lean_dec_ref_known(v___x_2213_, 1);
lean_inc(v_y_2202_);
lean_inc(v_x_2201_);
v___x_2215_ = l_Lean_Meta_mkEqHEq(v_x_2201_, v_y_2202_, v_a_2184_, v_a_2185_, v_a_2186_, v_a_2187_);
if (lean_obj_tag(v___x_2215_) == 0)
{
lean_object* v_a_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___f_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; 
v_a_2216_ = lean_ctor_get(v___x_2215_, 0);
lean_inc(v_a_2216_);
lean_dec_ref_known(v___x_2215_, 1);
v___x_2217_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__1));
v___x_2218_ = lean_unsigned_to_nat(1u);
v___x_2219_ = lean_nat_add(v_i_2181_, v___x_2218_);
lean_dec(v_i_2181_);
v___x_2220_ = lean_box_usize(v_sz_2208_);
v___x_2221_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___boxed__const__1));
lean_inc(v___x_2219_);
v___f_2222_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___boxed), 19, 13);
lean_closure_set(v___f_2222_, 0, v___x_2220_);
lean_closure_set(v___f_2222_, 1, v___x_2221_);
lean_closure_set(v___f_2222_, 2, v_deps_2207_);
lean_closure_set(v___f_2222_, 3, v_kinds_2182_);
lean_closure_set(v___f_2222_, 4, v_eqs_2183_);
lean_closure_set(v___f_2222_, 5, v_info_2176_);
lean_closure_set(v___f_2222_, 6, v_xs_2177_);
lean_closure_set(v___f_2222_, 7, v_ys_2178_);
lean_closure_set(v___f_2222_, 8, v_fixedParams_2179_);
lean_closure_set(v___f_2222_, 9, v_k_2180_);
lean_closure_set(v___f_2222_, 10, v___x_2219_);
lean_closure_set(v___f_2222_, 11, v___x_2194_);
lean_closure_set(v___f_2222_, 12, v_a_2214_);
v___x_2223_ = lean_name_append_index_after(v___x_2217_, v___x_2219_);
v___x_2224_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v___x_2223_, v_a_2216_, v___f_2222_, v_a_2184_, v_a_2185_, v_a_2186_, v_a_2187_);
return v___x_2224_;
}
else
{
lean_object* v_a_2225_; lean_object* v___x_2227_; uint8_t v_isShared_2228_; uint8_t v_isSharedCheck_2232_; 
lean_dec(v_a_2214_);
lean_dec_ref(v_deps_2207_);
lean_dec(v___x_2194_);
lean_dec_ref(v_eqs_2183_);
lean_dec_ref(v_kinds_2182_);
lean_dec(v_i_2181_);
lean_dec_ref(v_k_2180_);
lean_dec_ref(v_fixedParams_2179_);
lean_dec_ref(v_ys_2178_);
lean_dec_ref(v_xs_2177_);
lean_dec_ref(v_info_2176_);
v_a_2225_ = lean_ctor_get(v___x_2215_, 0);
v_isSharedCheck_2232_ = !lean_is_exclusive(v___x_2215_);
if (v_isSharedCheck_2232_ == 0)
{
v___x_2227_ = v___x_2215_;
v_isShared_2228_ = v_isSharedCheck_2232_;
goto v_resetjp_2226_;
}
else
{
lean_inc(v_a_2225_);
lean_dec(v___x_2215_);
v___x_2227_ = lean_box(0);
v_isShared_2228_ = v_isSharedCheck_2232_;
goto v_resetjp_2226_;
}
v_resetjp_2226_:
{
lean_object* v___x_2230_; 
if (v_isShared_2228_ == 0)
{
v___x_2230_ = v___x_2227_;
goto v_reusejp_2229_;
}
else
{
lean_object* v_reuseFailAlloc_2231_; 
v_reuseFailAlloc_2231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2231_, 0, v_a_2225_);
v___x_2230_ = v_reuseFailAlloc_2231_;
goto v_reusejp_2229_;
}
v_reusejp_2229_:
{
return v___x_2230_;
}
}
}
}
else
{
lean_object* v_a_2233_; lean_object* v___x_2235_; uint8_t v_isShared_2236_; uint8_t v_isSharedCheck_2240_; 
lean_dec_ref(v_deps_2207_);
lean_dec(v___x_2194_);
lean_dec_ref(v_eqs_2183_);
lean_dec_ref(v_kinds_2182_);
lean_dec(v_i_2181_);
lean_dec_ref(v_k_2180_);
lean_dec_ref(v_fixedParams_2179_);
lean_dec_ref(v_ys_2178_);
lean_dec_ref(v_xs_2177_);
lean_dec_ref(v_info_2176_);
v_a_2233_ = lean_ctor_get(v___x_2213_, 0);
v_isSharedCheck_2240_ = !lean_is_exclusive(v___x_2213_);
if (v_isSharedCheck_2240_ == 0)
{
v___x_2235_ = v___x_2213_;
v_isShared_2236_ = v_isSharedCheck_2240_;
goto v_resetjp_2234_;
}
else
{
lean_inc(v_a_2233_);
lean_dec(v___x_2213_);
v___x_2235_ = lean_box(0);
v_isShared_2236_ = v_isSharedCheck_2240_;
goto v_resetjp_2234_;
}
v_resetjp_2234_:
{
lean_object* v___x_2238_; 
if (v_isShared_2236_ == 0)
{
v___x_2238_ = v___x_2235_;
goto v_reusejp_2237_;
}
else
{
lean_object* v_reuseFailAlloc_2239_; 
v_reuseFailAlloc_2239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2239_, 0, v_a_2233_);
v___x_2238_ = v_reuseFailAlloc_2239_;
goto v_reusejp_2237_;
}
v_reusejp_2237_:
{
return v___x_2238_;
}
}
}
}
else
{
lean_object* v_a_2241_; lean_object* v___x_2243_; uint8_t v_isShared_2244_; uint8_t v_isSharedCheck_2248_; 
lean_dec(v___x_2194_);
lean_dec_ref(v_eqs_2183_);
lean_dec_ref(v_kinds_2182_);
lean_dec(v_i_2181_);
lean_dec_ref(v_k_2180_);
lean_dec_ref(v_fixedParams_2179_);
lean_dec_ref(v_ys_2178_);
lean_dec_ref(v_xs_2177_);
lean_dec_ref(v_info_2176_);
v_a_2241_ = lean_ctor_get(v___x_2203_, 0);
v_isSharedCheck_2248_ = !lean_is_exclusive(v___x_2203_);
if (v_isSharedCheck_2248_ == 0)
{
v___x_2243_ = v___x_2203_;
v_isShared_2244_ = v_isSharedCheck_2248_;
goto v_resetjp_2242_;
}
else
{
lean_inc(v_a_2241_);
lean_dec(v___x_2203_);
v___x_2243_ = lean_box(0);
v_isShared_2244_ = v_isSharedCheck_2248_;
goto v_resetjp_2242_;
}
v_resetjp_2242_:
{
lean_object* v___x_2246_; 
if (v_isShared_2244_ == 0)
{
v___x_2246_ = v___x_2243_;
goto v_reusejp_2245_;
}
else
{
lean_object* v_reuseFailAlloc_2247_; 
v_reuseFailAlloc_2247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2247_, 0, v_a_2241_);
v___x_2246_ = v_reuseFailAlloc_2247_;
goto v_reusejp_2245_;
}
v_reusejp_2245_:
{
return v___x_2246_;
}
}
}
}
else
{
lean_object* v___x_2249_; lean_object* v___x_2250_; uint8_t v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; 
lean_dec(v___x_2194_);
v___x_2249_ = lean_unsigned_to_nat(1u);
v___x_2250_ = lean_nat_add(v_i_2181_, v___x_2249_);
lean_dec(v_i_2181_);
v___x_2251_ = 0;
v___x_2252_ = lean_box(v___x_2251_);
v___x_2253_ = lean_array_push(v_kinds_2182_, v___x_2252_);
v___x_2254_ = lean_box(0);
v___x_2255_ = lean_array_push(v_eqs_2183_, v___x_2254_);
v_i_2181_ = v___x_2250_;
v_kinds_2182_ = v___x_2253_;
v_eqs_2183_ = v___x_2255_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__0(lean_object* v_h_2257_, size_t v_sz_2258_, size_t v___x_2259_, lean_object* v_deps_2260_, lean_object* v_kinds_2261_, lean_object* v_eqs_2262_, lean_object* v_info_2263_, lean_object* v_xs_2264_, lean_object* v_ys_2265_, lean_object* v_fixedParams_2266_, lean_object* v_k_2267_, lean_object* v___x_2268_, uint8_t v___x_2269_, lean_object* v_h_x27_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_){
_start:
{
lean_object* v___x_2276_; 
lean_inc(v___y_2274_);
lean_inc_ref(v___y_2273_);
lean_inc(v___y_2272_);
lean_inc_ref(v___y_2271_);
lean_inc_ref(v_h_2257_);
v___x_2276_ = lean_infer_type(v_h_2257_, v___y_2271_, v___y_2272_, v___y_2273_, v___y_2274_);
if (lean_obj_tag(v___x_2276_) == 0)
{
lean_object* v_a_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; uint8_t v___y_2281_; lean_object* v___x_2291_; lean_object* v___x_2292_; uint8_t v___x_2293_; 
v_a_2277_ = lean_ctor_get(v___x_2276_, 0);
lean_inc(v_a_2277_);
lean_dec_ref_known(v___x_2276_, 1);
v___x_2278_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop_spec__2(v_sz_2258_, v___x_2259_, v_deps_2260_);
lean_inc_ref(v_h_x27_2270_);
v___x_2279_ = l_Lean_mkAppN(v_h_x27_2270_, v___x_2278_);
lean_dec_ref(v___x_2278_);
v___x_2291_ = ((lean_object*)(lp_mathlib_Lean_Meta_fastSubsingletonElim___lam__1___closed__1));
v___x_2292_ = lean_unsigned_to_nat(3u);
v___x_2293_ = l_Lean_Expr_isAppOfArity(v_a_2277_, v___x_2291_, v___x_2292_);
lean_dec(v_a_2277_);
if (v___x_2293_ == 0)
{
if (v___x_2269_ == 0)
{
uint8_t v___x_2294_; 
v___x_2294_ = 4;
v___y_2281_ = v___x_2294_;
goto v___jp_2280_;
}
else
{
goto v___jp_2289_;
}
}
else
{
goto v___jp_2289_;
}
v___jp_2280_:
{
lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; 
v___x_2282_ = lean_box(v___y_2281_);
v___x_2283_ = lean_array_push(v_kinds_2261_, v___x_2282_);
v___x_2284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2284_, 0, v_h_x27_2270_);
lean_ctor_set(v___x_2284_, 1, v___x_2279_);
v___x_2285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2285_, 0, v_h_2257_);
lean_ctor_set(v___x_2285_, 1, v___x_2284_);
v___x_2286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2286_, 0, v___x_2285_);
v___x_2287_ = lean_array_push(v_eqs_2262_, v___x_2286_);
v___x_2288_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg(v_info_2263_, v_xs_2264_, v_ys_2265_, v_fixedParams_2266_, v_k_2267_, v___x_2268_, v___x_2283_, v___x_2287_, v___y_2271_, v___y_2272_, v___y_2273_, v___y_2274_);
return v___x_2288_;
}
v___jp_2289_:
{
uint8_t v___x_2290_; 
v___x_2290_ = 2;
v___y_2281_ = v___x_2290_;
goto v___jp_2280_;
}
}
else
{
lean_object* v_a_2295_; lean_object* v___x_2297_; uint8_t v_isShared_2298_; uint8_t v_isSharedCheck_2302_; 
lean_dec_ref(v_h_x27_2270_);
lean_dec(v___x_2268_);
lean_dec_ref(v_k_2267_);
lean_dec_ref(v_fixedParams_2266_);
lean_dec_ref(v_ys_2265_);
lean_dec_ref(v_xs_2264_);
lean_dec_ref(v_info_2263_);
lean_dec_ref(v_eqs_2262_);
lean_dec_ref(v_kinds_2261_);
lean_dec_ref(v_deps_2260_);
lean_dec_ref(v_h_2257_);
v_a_2295_ = lean_ctor_get(v___x_2276_, 0);
v_isSharedCheck_2302_ = !lean_is_exclusive(v___x_2276_);
if (v_isSharedCheck_2302_ == 0)
{
v___x_2297_ = v___x_2276_;
v_isShared_2298_ = v_isSharedCheck_2302_;
goto v_resetjp_2296_;
}
else
{
lean_inc(v_a_2295_);
lean_dec(v___x_2276_);
v___x_2297_ = lean_box(0);
v_isShared_2298_ = v_isSharedCheck_2302_;
goto v_resetjp_2296_;
}
v_resetjp_2296_:
{
lean_object* v___x_2300_; 
if (v_isShared_2298_ == 0)
{
v___x_2300_ = v___x_2297_;
goto v_reusejp_2299_;
}
else
{
lean_object* v_reuseFailAlloc_2301_; 
v_reuseFailAlloc_2301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2301_, 0, v_a_2295_);
v___x_2300_ = v_reuseFailAlloc_2301_;
goto v_reusejp_2299_;
}
v_reusejp_2299_:
{
return v___x_2300_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__0___boxed(lean_object** _args){
lean_object* v_h_2303_ = _args[0];
lean_object* v_sz_2304_ = _args[1];
lean_object* v___x_2305_ = _args[2];
lean_object* v_deps_2306_ = _args[3];
lean_object* v_kinds_2307_ = _args[4];
lean_object* v_eqs_2308_ = _args[5];
lean_object* v_info_2309_ = _args[6];
lean_object* v_xs_2310_ = _args[7];
lean_object* v_ys_2311_ = _args[8];
lean_object* v_fixedParams_2312_ = _args[9];
lean_object* v_k_2313_ = _args[10];
lean_object* v___x_2314_ = _args[11];
lean_object* v___x_2315_ = _args[12];
lean_object* v_h_x27_2316_ = _args[13];
lean_object* v___y_2317_ = _args[14];
lean_object* v___y_2318_ = _args[15];
lean_object* v___y_2319_ = _args[16];
lean_object* v___y_2320_ = _args[17];
lean_object* v___y_2321_ = _args[18];
_start:
{
size_t v_sz_boxed_2322_; size_t v___x_2247__boxed_2323_; uint8_t v___x_2249__boxed_2324_; lean_object* v_res_2325_; 
v_sz_boxed_2322_ = lean_unbox_usize(v_sz_2304_);
lean_dec(v_sz_2304_);
v___x_2247__boxed_2323_ = lean_unbox_usize(v___x_2305_);
lean_dec(v___x_2305_);
v___x_2249__boxed_2324_ = lean_unbox(v___x_2315_);
v_res_2325_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__0(v_h_2303_, v_sz_boxed_2322_, v___x_2247__boxed_2323_, v_deps_2306_, v_kinds_2307_, v_eqs_2308_, v_info_2309_, v_xs_2310_, v_ys_2311_, v_fixedParams_2312_, v_k_2313_, v___x_2314_, v___x_2249__boxed_2324_, v_h_x27_2316_, v___y_2317_, v___y_2318_, v___y_2319_, v___y_2320_);
lean_dec(v___y_2320_);
lean_dec_ref(v___y_2319_);
lean_dec(v___y_2318_);
lean_dec_ref(v___y_2317_);
return v_res_2325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1(size_t v_sz_2326_, size_t v___x_2327_, lean_object* v_deps_2328_, lean_object* v_kinds_2329_, lean_object* v_eqs_2330_, lean_object* v_info_2331_, lean_object* v_xs_2332_, lean_object* v_ys_2333_, lean_object* v_fixedParams_2334_, lean_object* v_k_2335_, lean_object* v___x_2336_, uint8_t v___x_2337_, lean_object* v_a_2338_, lean_object* v_h_2339_, lean_object* v___y_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_){
_start:
{
lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___f_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; 
v___x_2345_ = lean_box_usize(v_sz_2326_);
v___x_2346_ = lean_box_usize(v___x_2327_);
v___x_2347_ = lean_box(v___x_2337_);
lean_inc(v___x_2336_);
v___f_2348_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__0___boxed), 19, 13);
lean_closure_set(v___f_2348_, 0, v_h_2339_);
lean_closure_set(v___f_2348_, 1, v___x_2345_);
lean_closure_set(v___f_2348_, 2, v___x_2346_);
lean_closure_set(v___f_2348_, 3, v_deps_2328_);
lean_closure_set(v___f_2348_, 4, v_kinds_2329_);
lean_closure_set(v___f_2348_, 5, v_eqs_2330_);
lean_closure_set(v___f_2348_, 6, v_info_2331_);
lean_closure_set(v___f_2348_, 7, v_xs_2332_);
lean_closure_set(v___f_2348_, 8, v_ys_2333_);
lean_closure_set(v___f_2348_, 9, v_fixedParams_2334_);
lean_closure_set(v___f_2348_, 10, v_k_2335_);
lean_closure_set(v___f_2348_, 11, v___x_2336_);
lean_closure_set(v___f_2348_, 12, v___x_2347_);
v___x_2349_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___lam__1___closed__1));
v___x_2350_ = lean_name_append_index_after(v___x_2349_, v___x_2336_);
v___x_2351_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v___x_2350_, v_a_2338_, v___f_2348_, v___y_2340_, v___y_2341_, v___y_2342_, v___y_2343_);
return v___x_2351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___boxed(lean_object* v_info_2352_, lean_object* v_xs_2353_, lean_object* v_ys_2354_, lean_object* v_fixedParams_2355_, lean_object* v_k_2356_, lean_object* v_i_2357_, lean_object* v_kinds_2358_, lean_object* v_eqs_2359_, lean_object* v_a_2360_, lean_object* v_a_2361_, lean_object* v_a_2362_, lean_object* v_a_2363_, lean_object* v_a_2364_){
_start:
{
lean_object* v_res_2365_; 
v_res_2365_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg(v_info_2352_, v_xs_2353_, v_ys_2354_, v_fixedParams_2355_, v_k_2356_, v_i_2357_, v_kinds_2358_, v_eqs_2359_, v_a_2360_, v_a_2361_, v_a_2362_, v_a_2363_);
lean_dec(v_a_2363_);
lean_dec_ref(v_a_2362_);
lean_dec(v_a_2361_);
lean_dec_ref(v_a_2360_);
return v_res_2365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop(lean_object* v_info_2366_, lean_object* v_00_u03b1_2367_, lean_object* v_xs_2368_, lean_object* v_ys_2369_, lean_object* v_fixedParams_2370_, lean_object* v_k_2371_, lean_object* v_i_2372_, lean_object* v_kinds_2373_, lean_object* v_eqs_2374_, lean_object* v_a_2375_, lean_object* v_a_2376_, lean_object* v_a_2377_, lean_object* v_a_2378_){
_start:
{
lean_object* v___x_2380_; 
v___x_2380_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg(v_info_2366_, v_xs_2368_, v_ys_2369_, v_fixedParams_2370_, v_k_2371_, v_i_2372_, v_kinds_2373_, v_eqs_2374_, v_a_2375_, v_a_2376_, v_a_2377_, v_a_2378_);
return v___x_2380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___boxed(lean_object* v_info_2381_, lean_object* v_00_u03b1_2382_, lean_object* v_xs_2383_, lean_object* v_ys_2384_, lean_object* v_fixedParams_2385_, lean_object* v_k_2386_, lean_object* v_i_2387_, lean_object* v_kinds_2388_, lean_object* v_eqs_2389_, lean_object* v_a_2390_, lean_object* v_a_2391_, lean_object* v_a_2392_, lean_object* v_a_2393_, lean_object* v_a_2394_){
_start:
{
lean_object* v_res_2395_; 
v_res_2395_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop(v_info_2381_, v_00_u03b1_2382_, v_xs_2383_, v_ys_2384_, v_fixedParams_2385_, v_k_2386_, v_i_2387_, v_kinds_2388_, v_eqs_2389_, v_a_2390_, v_a_2391_, v_a_2392_, v_a_2393_);
lean_dec(v_a_2393_);
lean_dec_ref(v_a_2392_);
lean_dec(v_a_2391_);
lean_dec_ref(v_a_2390_);
return v_res_2395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___redArg(lean_object* v_info_2396_, lean_object* v_xs_2397_, lean_object* v_ys_2398_, lean_object* v_fixedParams_2399_, lean_object* v_k_2400_, lean_object* v_a_2401_, lean_object* v_a_2402_, lean_object* v_a_2403_, lean_object* v_a_2404_){
_start:
{
lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; 
v___x_2406_ = lean_unsigned_to_nat(0u);
v___x_2407_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkHCongrWithArity_x27___closed__0));
v___x_2408_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg(v_info_2396_, v_xs_2397_, v_ys_2398_, v_fixedParams_2399_, v_k_2400_, v___x_2406_, v___x_2407_, v___x_2407_, v_a_2401_, v_a_2402_, v_a_2403_, v_a_2404_);
return v___x_2408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___redArg___boxed(lean_object* v_info_2409_, lean_object* v_xs_2410_, lean_object* v_ys_2411_, lean_object* v_fixedParams_2412_, lean_object* v_k_2413_, lean_object* v_a_2414_, lean_object* v_a_2415_, lean_object* v_a_2416_, lean_object* v_a_2417_, lean_object* v_a_2418_){
_start:
{
lean_object* v_res_2419_; 
v_res_2419_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___redArg(v_info_2409_, v_xs_2410_, v_ys_2411_, v_fixedParams_2412_, v_k_2413_, v_a_2414_, v_a_2415_, v_a_2416_, v_a_2417_);
lean_dec(v_a_2417_);
lean_dec_ref(v_a_2416_);
lean_dec(v_a_2415_);
lean_dec_ref(v_a_2414_);
return v_res_2419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs(lean_object* v_info_2420_, lean_object* v_00_u03b1_2421_, lean_object* v_xs_2422_, lean_object* v_ys_2423_, lean_object* v_fixedParams_2424_, lean_object* v_k_2425_, lean_object* v_a_2426_, lean_object* v_a_2427_, lean_object* v_a_2428_, lean_object* v_a_2429_){
_start:
{
lean_object* v___x_2431_; 
v___x_2431_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___redArg(v_info_2420_, v_xs_2422_, v_ys_2423_, v_fixedParams_2424_, v_k_2425_, v_a_2426_, v_a_2427_, v_a_2428_, v_a_2429_);
return v___x_2431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___boxed(lean_object* v_info_2432_, lean_object* v_00_u03b1_2433_, lean_object* v_xs_2434_, lean_object* v_ys_2435_, lean_object* v_fixedParams_2436_, lean_object* v_k_2437_, lean_object* v_a_2438_, lean_object* v_a_2439_, lean_object* v_a_2440_, lean_object* v_a_2441_, lean_object* v_a_2442_){
_start:
{
lean_object* v_res_2443_; 
v_res_2443_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs(v_info_2432_, v_00_u03b1_2433_, v_xs_2434_, v_ys_2435_, v_fixedParams_2436_, v_k_2437_, v_a_2438_, v_a_2439_, v_a_2440_, v_a_2441_);
lean_dec(v_a_2441_);
lean_dec_ref(v_a_2440_);
lean_dec(v_a_2439_);
lean_dec_ref(v_a_2438_);
return v_res_2443_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__1(void){
_start:
{
lean_object* v___x_2445_; lean_object* v___x_2446_; 
v___x_2445_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__0));
v___x_2446_ = l_Lean_stringToMessageData(v___x_2445_);
return v___x_2446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore(lean_object* v_mvarId_2447_, lean_object* v_a_2448_, lean_object* v_a_2449_, lean_object* v_a_2450_, lean_object* v_a_2451_){
_start:
{
lean_object* v___y_2454_; lean_object* v___y_2455_; uint8_t v___y_2456_; lean_object* v___y_2508_; lean_object* v___y_2509_; uint8_t v___y_2510_; lean_object* v___x_2524_; uint8_t v___x_2525_; lean_object* v___x_2526_; 
v___x_2524_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove___closed__0));
v___x_2525_ = 1;
v___x_2526_ = l___private_Lean_Meta_Tactic_Cleanup_0__Lean_Meta_cleanupCore(v_mvarId_2447_, v___x_2524_, v___x_2525_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2526_) == 0)
{
lean_object* v_a_2527_; lean_object* v___x_2528_; 
v_a_2527_ = lean_ctor_get(v___x_2526_, 0);
lean_inc(v_a_2527_);
lean_dec_ref_known(v___x_2526_, 1);
v___x_2528_ = l_Lean_MVarId_intros(v_a_2527_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2528_) == 0)
{
lean_object* v_a_2529_; lean_object* v_snd_2530_; lean_object* v___x_2531_; 
v_a_2529_ = lean_ctor_get(v___x_2528_, 0);
lean_inc(v_a_2529_);
lean_dec_ref_known(v___x_2528_, 1);
v_snd_2530_ = lean_ctor_get(v_a_2529_, 1);
lean_inc_n(v_snd_2530_, 2);
lean_dec(v_a_2529_);
v___x_2531_ = l_Lean_MVarId_substEqs(v_snd_2530_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2531_) == 0)
{
lean_object* v_a_2532_; lean_object* v___y_2534_; 
v_a_2532_ = lean_ctor_get(v___x_2531_, 0);
lean_inc(v_a_2532_);
lean_dec_ref_known(v___x_2531_, 1);
if (lean_obj_tag(v_a_2532_) == 0)
{
v___y_2534_ = v_snd_2530_;
goto v___jp_2533_;
}
else
{
lean_object* v_val_2548_; 
lean_dec(v_snd_2530_);
v_val_2548_ = lean_ctor_get(v_a_2532_, 0);
lean_inc(v_val_2548_);
lean_dec_ref_known(v_a_2532_, 1);
v___y_2534_ = v_val_2548_;
goto v___jp_2533_;
}
v___jp_2533_:
{
lean_object* v___x_2535_; 
lean_inc(v___y_2534_);
v___x_2535_ = l_Lean_MVarId_refl(v___y_2534_, v___x_2525_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2535_) == 0)
{
lean_object* v___x_2537_; uint8_t v_isShared_2538_; uint8_t v_isSharedCheck_2543_; 
lean_dec(v___y_2534_);
v_isSharedCheck_2543_ = !lean_is_exclusive(v___x_2535_);
if (v_isSharedCheck_2543_ == 0)
{
lean_object* v_unused_2544_; 
v_unused_2544_ = lean_ctor_get(v___x_2535_, 0);
lean_dec(v_unused_2544_);
v___x_2537_ = v___x_2535_;
v_isShared_2538_ = v_isSharedCheck_2543_;
goto v_resetjp_2536_;
}
else
{
lean_dec(v___x_2535_);
v___x_2537_ = lean_box(0);
v_isShared_2538_ = v_isSharedCheck_2543_;
goto v_resetjp_2536_;
}
v_resetjp_2536_:
{
lean_object* v___x_2539_; lean_object* v___x_2541_; 
v___x_2539_ = lean_box(0);
if (v_isShared_2538_ == 0)
{
lean_ctor_set(v___x_2537_, 0, v___x_2539_);
v___x_2541_ = v___x_2537_;
goto v_reusejp_2540_;
}
else
{
lean_object* v_reuseFailAlloc_2542_; 
v_reuseFailAlloc_2542_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2542_, 0, v___x_2539_);
v___x_2541_ = v_reuseFailAlloc_2542_;
goto v_reusejp_2540_;
}
v_reusejp_2540_:
{
return v___x_2541_;
}
}
}
else
{
lean_object* v_a_2545_; uint8_t v___x_2546_; 
v_a_2545_ = lean_ctor_get(v___x_2535_, 0);
lean_inc(v_a_2545_);
v___x_2546_ = l_Lean_Exception_isInterrupt(v_a_2545_);
if (v___x_2546_ == 0)
{
uint8_t v___x_2547_; 
v___x_2547_ = l_Lean_Exception_isRuntime(v_a_2545_);
v___y_2508_ = v___y_2534_;
v___y_2509_ = v___x_2535_;
v___y_2510_ = v___x_2547_;
goto v___jp_2507_;
}
else
{
lean_dec(v_a_2545_);
v___y_2508_ = v___y_2534_;
v___y_2509_ = v___x_2535_;
v___y_2510_ = v___x_2546_;
goto v___jp_2507_;
}
}
}
}
else
{
lean_object* v_a_2549_; lean_object* v___x_2551_; uint8_t v_isShared_2552_; uint8_t v_isSharedCheck_2556_; 
lean_dec(v_snd_2530_);
v_a_2549_ = lean_ctor_get(v___x_2531_, 0);
v_isSharedCheck_2556_ = !lean_is_exclusive(v___x_2531_);
if (v_isSharedCheck_2556_ == 0)
{
v___x_2551_ = v___x_2531_;
v_isShared_2552_ = v_isSharedCheck_2556_;
goto v_resetjp_2550_;
}
else
{
lean_inc(v_a_2549_);
lean_dec(v___x_2531_);
v___x_2551_ = lean_box(0);
v_isShared_2552_ = v_isSharedCheck_2556_;
goto v_resetjp_2550_;
}
v_resetjp_2550_:
{
lean_object* v___x_2554_; 
if (v_isShared_2552_ == 0)
{
v___x_2554_ = v___x_2551_;
goto v_reusejp_2553_;
}
else
{
lean_object* v_reuseFailAlloc_2555_; 
v_reuseFailAlloc_2555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2555_, 0, v_a_2549_);
v___x_2554_ = v_reuseFailAlloc_2555_;
goto v_reusejp_2553_;
}
v_reusejp_2553_:
{
return v___x_2554_;
}
}
}
}
else
{
lean_object* v_a_2557_; lean_object* v___x_2559_; uint8_t v_isShared_2560_; uint8_t v_isSharedCheck_2564_; 
v_a_2557_ = lean_ctor_get(v___x_2528_, 0);
v_isSharedCheck_2564_ = !lean_is_exclusive(v___x_2528_);
if (v_isSharedCheck_2564_ == 0)
{
v___x_2559_ = v___x_2528_;
v_isShared_2560_ = v_isSharedCheck_2564_;
goto v_resetjp_2558_;
}
else
{
lean_inc(v_a_2557_);
lean_dec(v___x_2528_);
v___x_2559_ = lean_box(0);
v_isShared_2560_ = v_isSharedCheck_2564_;
goto v_resetjp_2558_;
}
v_resetjp_2558_:
{
lean_object* v___x_2562_; 
if (v_isShared_2560_ == 0)
{
v___x_2562_ = v___x_2559_;
goto v_reusejp_2561_;
}
else
{
lean_object* v_reuseFailAlloc_2563_; 
v_reuseFailAlloc_2563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2563_, 0, v_a_2557_);
v___x_2562_ = v_reuseFailAlloc_2563_;
goto v_reusejp_2561_;
}
v_reusejp_2561_:
{
return v___x_2562_;
}
}
}
}
else
{
lean_object* v_a_2565_; lean_object* v___x_2567_; uint8_t v_isShared_2568_; uint8_t v_isSharedCheck_2572_; 
v_a_2565_ = lean_ctor_get(v___x_2526_, 0);
v_isSharedCheck_2572_ = !lean_is_exclusive(v___x_2526_);
if (v_isSharedCheck_2572_ == 0)
{
v___x_2567_ = v___x_2526_;
v_isShared_2568_ = v_isSharedCheck_2572_;
goto v_resetjp_2566_;
}
else
{
lean_inc(v_a_2565_);
lean_dec(v___x_2526_);
v___x_2567_ = lean_box(0);
v_isShared_2568_ = v_isSharedCheck_2572_;
goto v_resetjp_2566_;
}
v_resetjp_2566_:
{
lean_object* v___x_2570_; 
if (v_isShared_2568_ == 0)
{
v___x_2570_ = v___x_2567_;
goto v_reusejp_2569_;
}
else
{
lean_object* v_reuseFailAlloc_2571_; 
v_reuseFailAlloc_2571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2571_, 0, v_a_2565_);
v___x_2570_ = v_reuseFailAlloc_2571_;
goto v_reusejp_2569_;
}
v_reusejp_2569_:
{
return v___x_2570_;
}
}
}
v___jp_2453_:
{
if (v___y_2456_ == 0)
{
lean_object* v___x_2457_; 
lean_dec_ref(v___y_2454_);
lean_inc(v___y_2455_);
v___x_2457_ = l_Lean_MVarId_proofIrrelHeq(v___y_2455_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2457_) == 0)
{
lean_object* v_a_2458_; lean_object* v___x_2460_; uint8_t v_isShared_2461_; uint8_t v_isSharedCheck_2498_; 
v_a_2458_ = lean_ctor_get(v___x_2457_, 0);
v_isSharedCheck_2498_ = !lean_is_exclusive(v___x_2457_);
if (v_isSharedCheck_2498_ == 0)
{
v___x_2460_ = v___x_2457_;
v_isShared_2461_ = v_isSharedCheck_2498_;
goto v_resetjp_2459_;
}
else
{
lean_inc(v_a_2458_);
lean_dec(v___x_2457_);
v___x_2460_ = lean_box(0);
v_isShared_2461_ = v_isSharedCheck_2498_;
goto v_resetjp_2459_;
}
v_resetjp_2459_:
{
uint8_t v___x_2462_; 
v___x_2462_ = lean_unbox(v_a_2458_);
lean_dec(v_a_2458_);
if (v___x_2462_ == 0)
{
lean_object* v___x_2463_; 
lean_del_object(v___x_2460_);
v___x_2463_ = l_Lean_MVarId_heqOfEq(v___y_2455_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2463_) == 0)
{
lean_object* v_a_2464_; lean_object* v___x_2465_; 
v_a_2464_ = lean_ctor_get(v___x_2463_, 0);
lean_inc(v_a_2464_);
lean_dec_ref_known(v___x_2463_, 1);
v___x_2465_ = lp_mathlib_Lean_Meta_fastSubsingletonElim(v_a_2464_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2465_) == 0)
{
lean_object* v_a_2466_; lean_object* v___x_2468_; uint8_t v_isShared_2469_; uint8_t v_isSharedCheck_2477_; 
v_a_2466_ = lean_ctor_get(v___x_2465_, 0);
v_isSharedCheck_2477_ = !lean_is_exclusive(v___x_2465_);
if (v_isSharedCheck_2477_ == 0)
{
v___x_2468_ = v___x_2465_;
v_isShared_2469_ = v_isSharedCheck_2477_;
goto v_resetjp_2467_;
}
else
{
lean_inc(v_a_2466_);
lean_dec(v___x_2465_);
v___x_2468_ = lean_box(0);
v_isShared_2469_ = v_isSharedCheck_2477_;
goto v_resetjp_2467_;
}
v_resetjp_2467_:
{
uint8_t v___x_2470_; 
v___x_2470_ = lean_unbox(v_a_2466_);
lean_dec(v_a_2466_);
if (v___x_2470_ == 0)
{
lean_object* v___x_2471_; lean_object* v___x_2472_; 
lean_del_object(v___x_2468_);
v___x_2471_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__1, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___closed__1);
v___x_2472_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v___x_2471_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
return v___x_2472_;
}
else
{
lean_object* v___x_2473_; lean_object* v___x_2475_; 
v___x_2473_ = lean_box(0);
if (v_isShared_2469_ == 0)
{
lean_ctor_set(v___x_2468_, 0, v___x_2473_);
v___x_2475_ = v___x_2468_;
goto v_reusejp_2474_;
}
else
{
lean_object* v_reuseFailAlloc_2476_; 
v_reuseFailAlloc_2476_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2476_, 0, v___x_2473_);
v___x_2475_ = v_reuseFailAlloc_2476_;
goto v_reusejp_2474_;
}
v_reusejp_2474_:
{
return v___x_2475_;
}
}
}
}
else
{
lean_object* v_a_2478_; lean_object* v___x_2480_; uint8_t v_isShared_2481_; uint8_t v_isSharedCheck_2485_; 
v_a_2478_ = lean_ctor_get(v___x_2465_, 0);
v_isSharedCheck_2485_ = !lean_is_exclusive(v___x_2465_);
if (v_isSharedCheck_2485_ == 0)
{
v___x_2480_ = v___x_2465_;
v_isShared_2481_ = v_isSharedCheck_2485_;
goto v_resetjp_2479_;
}
else
{
lean_inc(v_a_2478_);
lean_dec(v___x_2465_);
v___x_2480_ = lean_box(0);
v_isShared_2481_ = v_isSharedCheck_2485_;
goto v_resetjp_2479_;
}
v_resetjp_2479_:
{
lean_object* v___x_2483_; 
if (v_isShared_2481_ == 0)
{
v___x_2483_ = v___x_2480_;
goto v_reusejp_2482_;
}
else
{
lean_object* v_reuseFailAlloc_2484_; 
v_reuseFailAlloc_2484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2484_, 0, v_a_2478_);
v___x_2483_ = v_reuseFailAlloc_2484_;
goto v_reusejp_2482_;
}
v_reusejp_2482_:
{
return v___x_2483_;
}
}
}
}
else
{
lean_object* v_a_2486_; lean_object* v___x_2488_; uint8_t v_isShared_2489_; uint8_t v_isSharedCheck_2493_; 
v_a_2486_ = lean_ctor_get(v___x_2463_, 0);
v_isSharedCheck_2493_ = !lean_is_exclusive(v___x_2463_);
if (v_isSharedCheck_2493_ == 0)
{
v___x_2488_ = v___x_2463_;
v_isShared_2489_ = v_isSharedCheck_2493_;
goto v_resetjp_2487_;
}
else
{
lean_inc(v_a_2486_);
lean_dec(v___x_2463_);
v___x_2488_ = lean_box(0);
v_isShared_2489_ = v_isSharedCheck_2493_;
goto v_resetjp_2487_;
}
v_resetjp_2487_:
{
lean_object* v___x_2491_; 
if (v_isShared_2489_ == 0)
{
v___x_2491_ = v___x_2488_;
goto v_reusejp_2490_;
}
else
{
lean_object* v_reuseFailAlloc_2492_; 
v_reuseFailAlloc_2492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2492_, 0, v_a_2486_);
v___x_2491_ = v_reuseFailAlloc_2492_;
goto v_reusejp_2490_;
}
v_reusejp_2490_:
{
return v___x_2491_;
}
}
}
}
else
{
lean_object* v___x_2494_; lean_object* v___x_2496_; 
lean_dec(v___y_2455_);
v___x_2494_ = lean_box(0);
if (v_isShared_2461_ == 0)
{
lean_ctor_set(v___x_2460_, 0, v___x_2494_);
v___x_2496_ = v___x_2460_;
goto v_reusejp_2495_;
}
else
{
lean_object* v_reuseFailAlloc_2497_; 
v_reuseFailAlloc_2497_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2497_, 0, v___x_2494_);
v___x_2496_ = v_reuseFailAlloc_2497_;
goto v_reusejp_2495_;
}
v_reusejp_2495_:
{
return v___x_2496_;
}
}
}
}
else
{
lean_object* v_a_2499_; lean_object* v___x_2501_; uint8_t v_isShared_2502_; uint8_t v_isSharedCheck_2506_; 
lean_dec(v___y_2455_);
v_a_2499_ = lean_ctor_get(v___x_2457_, 0);
v_isSharedCheck_2506_ = !lean_is_exclusive(v___x_2457_);
if (v_isSharedCheck_2506_ == 0)
{
v___x_2501_ = v___x_2457_;
v_isShared_2502_ = v_isSharedCheck_2506_;
goto v_resetjp_2500_;
}
else
{
lean_inc(v_a_2499_);
lean_dec(v___x_2457_);
v___x_2501_ = lean_box(0);
v_isShared_2502_ = v_isSharedCheck_2506_;
goto v_resetjp_2500_;
}
v_resetjp_2500_:
{
lean_object* v___x_2504_; 
if (v_isShared_2502_ == 0)
{
v___x_2504_ = v___x_2501_;
goto v_reusejp_2503_;
}
else
{
lean_object* v_reuseFailAlloc_2505_; 
v_reuseFailAlloc_2505_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2505_, 0, v_a_2499_);
v___x_2504_ = v_reuseFailAlloc_2505_;
goto v_reusejp_2503_;
}
v_reusejp_2503_:
{
return v___x_2504_;
}
}
}
}
else
{
lean_dec(v___y_2455_);
return v___y_2454_;
}
}
v___jp_2507_:
{
if (v___y_2510_ == 0)
{
lean_object* v___x_2511_; 
lean_dec_ref(v___y_2509_);
lean_inc(v___y_2508_);
v___x_2511_ = l_Lean_MVarId_hrefl(v___y_2508_, v_a_2448_, v_a_2449_, v_a_2450_, v_a_2451_);
if (lean_obj_tag(v___x_2511_) == 0)
{
lean_object* v___x_2513_; uint8_t v_isShared_2514_; uint8_t v_isSharedCheck_2519_; 
lean_dec(v___y_2508_);
v_isSharedCheck_2519_ = !lean_is_exclusive(v___x_2511_);
if (v_isSharedCheck_2519_ == 0)
{
lean_object* v_unused_2520_; 
v_unused_2520_ = lean_ctor_get(v___x_2511_, 0);
lean_dec(v_unused_2520_);
v___x_2513_ = v___x_2511_;
v_isShared_2514_ = v_isSharedCheck_2519_;
goto v_resetjp_2512_;
}
else
{
lean_dec(v___x_2511_);
v___x_2513_ = lean_box(0);
v_isShared_2514_ = v_isSharedCheck_2519_;
goto v_resetjp_2512_;
}
v_resetjp_2512_:
{
lean_object* v___x_2515_; lean_object* v___x_2517_; 
v___x_2515_ = lean_box(0);
if (v_isShared_2514_ == 0)
{
lean_ctor_set(v___x_2513_, 0, v___x_2515_);
v___x_2517_ = v___x_2513_;
goto v_reusejp_2516_;
}
else
{
lean_object* v_reuseFailAlloc_2518_; 
v_reuseFailAlloc_2518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2518_, 0, v___x_2515_);
v___x_2517_ = v_reuseFailAlloc_2518_;
goto v_reusejp_2516_;
}
v_reusejp_2516_:
{
return v___x_2517_;
}
}
}
else
{
lean_object* v_a_2521_; uint8_t v___x_2522_; 
v_a_2521_ = lean_ctor_get(v___x_2511_, 0);
lean_inc(v_a_2521_);
v___x_2522_ = l_Lean_Exception_isInterrupt(v_a_2521_);
if (v___x_2522_ == 0)
{
uint8_t v___x_2523_; 
v___x_2523_ = l_Lean_Exception_isRuntime(v_a_2521_);
v___y_2454_ = v___x_2511_;
v___y_2455_ = v___y_2508_;
v___y_2456_ = v___x_2523_;
goto v___jp_2453_;
}
else
{
lean_dec(v_a_2521_);
v___y_2454_ = v___x_2511_;
v___y_2455_ = v___y_2508_;
v___y_2456_ = v___x_2522_;
goto v___jp_2453_;
}
}
}
else
{
lean_dec(v___y_2508_);
return v___y_2509_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore___boxed(lean_object* v_mvarId_2573_, lean_object* v_a_2574_, lean_object* v_a_2575_, lean_object* v_a_2576_, lean_object* v_a_2577_, lean_object* v_a_2578_){
_start:
{
lean_object* v_res_2579_; 
v_res_2579_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore(v_mvarId_2573_, v_a_2574_, v_a_2575_, v_a_2576_, v_a_2577_);
lean_dec(v_a_2577_);
lean_dec_ref(v_a_2576_);
lean_dec(v_a_2575_);
lean_dec_ref(v_a_2574_);
return v_res_2579_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__0(void){
_start:
{
lean_object* v___x_2580_; double v___x_2581_; 
v___x_2580_ = lean_unsigned_to_nat(0u);
v___x_2581_ = lean_float_of_nat(v___x_2580_);
return v___x_2581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(lean_object* v_cls_2585_, lean_object* v_msg_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_, lean_object* v___y_2590_){
_start:
{
lean_object* v_ref_2592_; lean_object* v___x_2593_; lean_object* v_a_2594_; lean_object* v___x_2596_; uint8_t v_isShared_2597_; uint8_t v_isSharedCheck_2638_; 
v_ref_2592_ = lean_ctor_get(v___y_2589_, 5);
v___x_2593_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0_spec__0(v_msg_2586_, v___y_2587_, v___y_2588_, v___y_2589_, v___y_2590_);
v_a_2594_ = lean_ctor_get(v___x_2593_, 0);
v_isSharedCheck_2638_ = !lean_is_exclusive(v___x_2593_);
if (v_isSharedCheck_2638_ == 0)
{
v___x_2596_ = v___x_2593_;
v_isShared_2597_ = v_isSharedCheck_2638_;
goto v_resetjp_2595_;
}
else
{
lean_inc(v_a_2594_);
lean_dec(v___x_2593_);
v___x_2596_ = lean_box(0);
v_isShared_2597_ = v_isSharedCheck_2638_;
goto v_resetjp_2595_;
}
v_resetjp_2595_:
{
lean_object* v___x_2598_; lean_object* v_traceState_2599_; lean_object* v_env_2600_; lean_object* v_nextMacroScope_2601_; lean_object* v_ngen_2602_; lean_object* v_auxDeclNGen_2603_; lean_object* v_cache_2604_; lean_object* v_messages_2605_; lean_object* v_infoState_2606_; lean_object* v_snapshotTasks_2607_; lean_object* v___x_2609_; uint8_t v_isShared_2610_; uint8_t v_isSharedCheck_2637_; 
v___x_2598_ = lean_st_ref_take(v___y_2590_);
v_traceState_2599_ = lean_ctor_get(v___x_2598_, 4);
v_env_2600_ = lean_ctor_get(v___x_2598_, 0);
v_nextMacroScope_2601_ = lean_ctor_get(v___x_2598_, 1);
v_ngen_2602_ = lean_ctor_get(v___x_2598_, 2);
v_auxDeclNGen_2603_ = lean_ctor_get(v___x_2598_, 3);
v_cache_2604_ = lean_ctor_get(v___x_2598_, 5);
v_messages_2605_ = lean_ctor_get(v___x_2598_, 6);
v_infoState_2606_ = lean_ctor_get(v___x_2598_, 7);
v_snapshotTasks_2607_ = lean_ctor_get(v___x_2598_, 8);
v_isSharedCheck_2637_ = !lean_is_exclusive(v___x_2598_);
if (v_isSharedCheck_2637_ == 0)
{
v___x_2609_ = v___x_2598_;
v_isShared_2610_ = v_isSharedCheck_2637_;
goto v_resetjp_2608_;
}
else
{
lean_inc(v_snapshotTasks_2607_);
lean_inc(v_infoState_2606_);
lean_inc(v_messages_2605_);
lean_inc(v_cache_2604_);
lean_inc(v_traceState_2599_);
lean_inc(v_auxDeclNGen_2603_);
lean_inc(v_ngen_2602_);
lean_inc(v_nextMacroScope_2601_);
lean_inc(v_env_2600_);
lean_dec(v___x_2598_);
v___x_2609_ = lean_box(0);
v_isShared_2610_ = v_isSharedCheck_2637_;
goto v_resetjp_2608_;
}
v_resetjp_2608_:
{
uint64_t v_tid_2611_; lean_object* v_traces_2612_; lean_object* v___x_2614_; uint8_t v_isShared_2615_; uint8_t v_isSharedCheck_2636_; 
v_tid_2611_ = lean_ctor_get_uint64(v_traceState_2599_, sizeof(void*)*1);
v_traces_2612_ = lean_ctor_get(v_traceState_2599_, 0);
v_isSharedCheck_2636_ = !lean_is_exclusive(v_traceState_2599_);
if (v_isSharedCheck_2636_ == 0)
{
v___x_2614_ = v_traceState_2599_;
v_isShared_2615_ = v_isSharedCheck_2636_;
goto v_resetjp_2613_;
}
else
{
lean_inc(v_traces_2612_);
lean_dec(v_traceState_2599_);
v___x_2614_ = lean_box(0);
v_isShared_2615_ = v_isSharedCheck_2636_;
goto v_resetjp_2613_;
}
v_resetjp_2613_:
{
lean_object* v___x_2616_; double v___x_2617_; uint8_t v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2626_; 
v___x_2616_ = lean_box(0);
v___x_2617_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__0);
v___x_2618_ = 0;
v___x_2619_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__1));
v___x_2620_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2620_, 0, v_cls_2585_);
lean_ctor_set(v___x_2620_, 1, v___x_2616_);
lean_ctor_set(v___x_2620_, 2, v___x_2619_);
lean_ctor_set_float(v___x_2620_, sizeof(void*)*3, v___x_2617_);
lean_ctor_set_float(v___x_2620_, sizeof(void*)*3 + 8, v___x_2617_);
lean_ctor_set_uint8(v___x_2620_, sizeof(void*)*3 + 16, v___x_2618_);
v___x_2621_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___closed__2));
v___x_2622_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2622_, 0, v___x_2620_);
lean_ctor_set(v___x_2622_, 1, v_a_2594_);
lean_ctor_set(v___x_2622_, 2, v___x_2621_);
lean_inc(v_ref_2592_);
v___x_2623_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2623_, 0, v_ref_2592_);
lean_ctor_set(v___x_2623_, 1, v___x_2622_);
v___x_2624_ = l_Lean_PersistentArray_push___redArg(v_traces_2612_, v___x_2623_);
if (v_isShared_2615_ == 0)
{
lean_ctor_set(v___x_2614_, 0, v___x_2624_);
v___x_2626_ = v___x_2614_;
goto v_reusejp_2625_;
}
else
{
lean_object* v_reuseFailAlloc_2635_; 
v_reuseFailAlloc_2635_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2635_, 0, v___x_2624_);
lean_ctor_set_uint64(v_reuseFailAlloc_2635_, sizeof(void*)*1, v_tid_2611_);
v___x_2626_ = v_reuseFailAlloc_2635_;
goto v_reusejp_2625_;
}
v_reusejp_2625_:
{
lean_object* v___x_2628_; 
if (v_isShared_2610_ == 0)
{
lean_ctor_set(v___x_2609_, 4, v___x_2626_);
v___x_2628_ = v___x_2609_;
goto v_reusejp_2627_;
}
else
{
lean_object* v_reuseFailAlloc_2634_; 
v_reuseFailAlloc_2634_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2634_, 0, v_env_2600_);
lean_ctor_set(v_reuseFailAlloc_2634_, 1, v_nextMacroScope_2601_);
lean_ctor_set(v_reuseFailAlloc_2634_, 2, v_ngen_2602_);
lean_ctor_set(v_reuseFailAlloc_2634_, 3, v_auxDeclNGen_2603_);
lean_ctor_set(v_reuseFailAlloc_2634_, 4, v___x_2626_);
lean_ctor_set(v_reuseFailAlloc_2634_, 5, v_cache_2604_);
lean_ctor_set(v_reuseFailAlloc_2634_, 6, v_messages_2605_);
lean_ctor_set(v_reuseFailAlloc_2634_, 7, v_infoState_2606_);
lean_ctor_set(v_reuseFailAlloc_2634_, 8, v_snapshotTasks_2607_);
v___x_2628_ = v_reuseFailAlloc_2634_;
goto v_reusejp_2627_;
}
v_reusejp_2627_:
{
lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2632_; 
v___x_2629_ = lean_st_ref_set(v___y_2590_, v___x_2628_);
v___x_2630_ = lean_box(0);
if (v_isShared_2597_ == 0)
{
lean_ctor_set(v___x_2596_, 0, v___x_2630_);
v___x_2632_ = v___x_2596_;
goto v_reusejp_2631_;
}
else
{
lean_object* v_reuseFailAlloc_2633_; 
v_reuseFailAlloc_2633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2633_, 0, v___x_2630_);
v___x_2632_ = v_reuseFailAlloc_2633_;
goto v_reusejp_2631_;
}
v_reusejp_2631_:
{
return v___x_2632_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0___boxed(lean_object* v_cls_2639_, lean_object* v_msg_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_, lean_object* v___y_2645_){
_start:
{
lean_object* v_res_2646_; 
v_res_2646_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_2639_, v_msg_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_);
lean_dec(v___y_2644_);
lean_dec_ref(v___y_2643_);
lean_dec(v___y_2642_);
lean_dec_ref(v___y_2641_);
return v_res_2646_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2(void){
_start:
{
lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; 
v___x_2650_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_));
v___x_2651_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__1));
v___x_2652_ = l_Lean_Name_append(v___x_2651_, v___x_2650_);
return v___x_2652_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__4(void){
_start:
{
lean_object* v___x_2654_; lean_object* v___x_2655_; 
v___x_2654_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__3));
v___x_2655_ = l_Lean_stringToMessageData(v___x_2654_);
return v___x_2655_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__6(void){
_start:
{
lean_object* v___x_2657_; lean_object* v___x_2658_; 
v___x_2657_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__5));
v___x_2658_ = l_Lean_stringToMessageData(v___x_2657_);
return v___x_2658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0(lean_object* v___x_2659_, uint8_t v___x_2660_, lean_object* v___x_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_){
_start:
{
lean_object* v___x_2667_; 
v___x_2667_ = l_Lean_Meta_mkFreshExprMVar(v___x_2659_, v___x_2660_, v___x_2661_, v___y_2662_, v___y_2663_, v___y_2664_, v___y_2665_);
if (lean_obj_tag(v___x_2667_) == 0)
{
lean_object* v_options_2668_; lean_object* v_a_2669_; lean_object* v___x_2671_; uint8_t v_isShared_2672_; uint8_t v_isSharedCheck_2740_; 
v_options_2668_ = lean_ctor_get(v___y_2664_, 2);
v_a_2669_ = lean_ctor_get(v___x_2667_, 0);
v_isSharedCheck_2740_ = !lean_is_exclusive(v___x_2667_);
if (v_isSharedCheck_2740_ == 0)
{
v___x_2671_ = v___x_2667_;
v_isShared_2672_ = v_isSharedCheck_2740_;
goto v_resetjp_2670_;
}
else
{
lean_inc(v_a_2669_);
lean_dec(v___x_2667_);
v___x_2671_ = lean_box(0);
v_isShared_2672_ = v_isSharedCheck_2740_;
goto v_resetjp_2670_;
}
v_resetjp_2670_:
{
lean_object* v_inheritedTraceOptions_2673_; uint8_t v_hasTrace_2674_; lean_object* v___x_2675_; lean_object* v___y_2677_; lean_object* v___y_2678_; lean_object* v___y_2679_; lean_object* v___y_2680_; 
v_inheritedTraceOptions_2673_ = lean_ctor_get(v___y_2664_, 13);
v_hasTrace_2674_ = lean_ctor_get_uint8(v_options_2668_, sizeof(void*)*1);
v___x_2675_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_));
if (v_hasTrace_2674_ == 0)
{
lean_del_object(v___x_2671_);
v___y_2677_ = v___y_2662_;
v___y_2678_ = v___y_2663_;
v___y_2679_ = v___y_2664_;
v___y_2680_ = v___y_2665_;
goto v___jp_2676_;
}
else
{
lean_object* v___x_2723_; uint8_t v___x_2724_; 
v___x_2723_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2);
v___x_2724_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2673_, v_options_2668_, v___x_2723_);
if (v___x_2724_ == 0)
{
lean_del_object(v___x_2671_);
v___y_2677_ = v___y_2662_;
v___y_2678_ = v___y_2663_;
v___y_2679_ = v___y_2664_;
v___y_2680_ = v___y_2665_;
goto v___jp_2676_;
}
else
{
lean_object* v___x_2725_; lean_object* v___x_2726_; lean_object* v___x_2728_; 
v___x_2725_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__6, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__6_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__6);
v___x_2726_ = l_Lean_Expr_mvarId_x21(v_a_2669_);
if (v_isShared_2672_ == 0)
{
lean_ctor_set_tag(v___x_2671_, 1);
lean_ctor_set(v___x_2671_, 0, v___x_2726_);
v___x_2728_ = v___x_2671_;
goto v_reusejp_2727_;
}
else
{
lean_object* v_reuseFailAlloc_2739_; 
v_reuseFailAlloc_2739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2739_, 0, v___x_2726_);
v___x_2728_ = v_reuseFailAlloc_2739_;
goto v_reusejp_2727_;
}
v_reusejp_2727_:
{
lean_object* v___x_2729_; lean_object* v___x_2730_; 
v___x_2729_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2729_, 0, v___x_2725_);
lean_ctor_set(v___x_2729_, 1, v___x_2728_);
v___x_2730_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v___x_2675_, v___x_2729_, v___y_2662_, v___y_2663_, v___y_2664_, v___y_2665_);
if (lean_obj_tag(v___x_2730_) == 0)
{
lean_dec_ref_known(v___x_2730_, 1);
v___y_2677_ = v___y_2662_;
v___y_2678_ = v___y_2663_;
v___y_2679_ = v___y_2664_;
v___y_2680_ = v___y_2665_;
goto v___jp_2676_;
}
else
{
lean_object* v_a_2731_; lean_object* v___x_2733_; uint8_t v_isShared_2734_; uint8_t v_isSharedCheck_2738_; 
lean_dec(v_a_2669_);
lean_dec_ref(v___y_2662_);
v_a_2731_ = lean_ctor_get(v___x_2730_, 0);
v_isSharedCheck_2738_ = !lean_is_exclusive(v___x_2730_);
if (v_isSharedCheck_2738_ == 0)
{
v___x_2733_ = v___x_2730_;
v_isShared_2734_ = v_isSharedCheck_2738_;
goto v_resetjp_2732_;
}
else
{
lean_inc(v_a_2731_);
lean_dec(v___x_2730_);
v___x_2733_ = lean_box(0);
v_isShared_2734_ = v_isSharedCheck_2738_;
goto v_resetjp_2732_;
}
v_resetjp_2732_:
{
lean_object* v___x_2736_; 
if (v_isShared_2734_ == 0)
{
v___x_2736_ = v___x_2733_;
goto v_reusejp_2735_;
}
else
{
lean_object* v_reuseFailAlloc_2737_; 
v_reuseFailAlloc_2737_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2737_, 0, v_a_2731_);
v___x_2736_ = v_reuseFailAlloc_2737_;
goto v_reusejp_2735_;
}
v_reusejp_2735_:
{
return v___x_2736_;
}
}
}
}
}
}
v___jp_2676_:
{
lean_object* v_keyedConfig_2681_; uint8_t v_trackZetaDelta_2682_; lean_object* v_zetaDeltaSet_2683_; lean_object* v_lctx_2684_; lean_object* v_localInstances_2685_; lean_object* v_defEqCtx_x3f_2686_; lean_object* v_synthPendingDepth_2687_; lean_object* v_customCanUnfoldPredicate_x3f_2688_; uint8_t v_univApprox_2689_; uint8_t v_inTypeClassResolution_2690_; uint8_t v_cacheInferType_2691_; lean_object* v___x_2692_; uint8_t v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; 
v_keyedConfig_2681_ = lean_ctor_get(v___y_2677_, 0);
v_trackZetaDelta_2682_ = lean_ctor_get_uint8(v___y_2677_, sizeof(void*)*7);
v_zetaDeltaSet_2683_ = lean_ctor_get(v___y_2677_, 1);
v_lctx_2684_ = lean_ctor_get(v___y_2677_, 2);
v_localInstances_2685_ = lean_ctor_get(v___y_2677_, 3);
v_defEqCtx_x3f_2686_ = lean_ctor_get(v___y_2677_, 4);
v_synthPendingDepth_2687_ = lean_ctor_get(v___y_2677_, 5);
v_customCanUnfoldPredicate_x3f_2688_ = lean_ctor_get(v___y_2677_, 6);
v_univApprox_2689_ = lean_ctor_get_uint8(v___y_2677_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2690_ = lean_ctor_get_uint8(v___y_2677_, sizeof(void*)*7 + 2);
v_cacheInferType_2691_ = lean_ctor_get_uint8(v___y_2677_, sizeof(void*)*7 + 3);
v___x_2692_ = l_Lean_Expr_mvarId_x21(v_a_2669_);
v___x_2693_ = 2;
lean_inc_ref(v_keyedConfig_2681_);
v___x_2694_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2693_, v_keyedConfig_2681_);
lean_inc(v_customCanUnfoldPredicate_x3f_2688_);
lean_inc(v_synthPendingDepth_2687_);
lean_inc(v_defEqCtx_x3f_2686_);
lean_inc_ref(v_localInstances_2685_);
lean_inc_ref(v_lctx_2684_);
lean_inc(v_zetaDeltaSet_2683_);
v___x_2695_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2695_, 0, v___x_2694_);
lean_ctor_set(v___x_2695_, 1, v_zetaDeltaSet_2683_);
lean_ctor_set(v___x_2695_, 2, v_lctx_2684_);
lean_ctor_set(v___x_2695_, 3, v_localInstances_2685_);
lean_ctor_set(v___x_2695_, 4, v_defEqCtx_x3f_2686_);
lean_ctor_set(v___x_2695_, 5, v_synthPendingDepth_2687_);
lean_ctor_set(v___x_2695_, 6, v_customCanUnfoldPredicate_x3f_2688_);
lean_ctor_set_uint8(v___x_2695_, sizeof(void*)*7, v_trackZetaDelta_2682_);
lean_ctor_set_uint8(v___x_2695_, sizeof(void*)*7 + 1, v_univApprox_2689_);
lean_ctor_set_uint8(v___x_2695_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2690_);
lean_ctor_set_uint8(v___x_2695_, sizeof(void*)*7 + 3, v_cacheInferType_2691_);
v___x_2696_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolveCore(v___x_2692_, v___x_2695_, v___y_2678_, v___y_2679_, v___y_2680_);
lean_dec_ref_known(v___x_2695_, 7);
if (lean_obj_tag(v___x_2696_) == 0)
{
lean_object* v_options_2697_; uint8_t v_hasTrace_2698_; 
lean_dec_ref_known(v___x_2696_, 1);
v_options_2697_ = lean_ctor_get(v___y_2679_, 2);
v_hasTrace_2698_ = lean_ctor_get_uint8(v_options_2697_, sizeof(void*)*1);
if (v_hasTrace_2698_ == 0)
{
lean_object* v___x_2699_; 
lean_dec_ref(v___y_2677_);
v___x_2699_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(v_a_2669_, v___y_2678_);
return v___x_2699_;
}
else
{
lean_object* v_inheritedTraceOptions_2700_; lean_object* v___x_2701_; uint8_t v___x_2702_; 
v_inheritedTraceOptions_2700_ = lean_ctor_get(v___y_2679_, 13);
v___x_2701_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__2);
v___x_2702_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2700_, v_options_2697_, v___x_2701_);
if (v___x_2702_ == 0)
{
lean_object* v___x_2703_; 
lean_dec_ref(v___y_2677_);
v___x_2703_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(v_a_2669_, v___y_2678_);
return v___x_2703_;
}
else
{
lean_object* v___x_2704_; lean_object* v___x_2705_; 
v___x_2704_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__4, &lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__4_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__4);
v___x_2705_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v___x_2675_, v___x_2704_, v___y_2677_, v___y_2678_, v___y_2679_, v___y_2680_);
lean_dec_ref(v___y_2677_);
if (lean_obj_tag(v___x_2705_) == 0)
{
lean_object* v___x_2706_; 
lean_dec_ref_known(v___x_2705_, 1);
v___x_2706_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__3___redArg(v_a_2669_, v___y_2678_);
return v___x_2706_;
}
else
{
lean_object* v_a_2707_; lean_object* v___x_2709_; uint8_t v_isShared_2710_; uint8_t v_isSharedCheck_2714_; 
lean_dec(v_a_2669_);
v_a_2707_ = lean_ctor_get(v___x_2705_, 0);
v_isSharedCheck_2714_ = !lean_is_exclusive(v___x_2705_);
if (v_isSharedCheck_2714_ == 0)
{
v___x_2709_ = v___x_2705_;
v_isShared_2710_ = v_isSharedCheck_2714_;
goto v_resetjp_2708_;
}
else
{
lean_inc(v_a_2707_);
lean_dec(v___x_2705_);
v___x_2709_ = lean_box(0);
v_isShared_2710_ = v_isSharedCheck_2714_;
goto v_resetjp_2708_;
}
v_resetjp_2708_:
{
lean_object* v___x_2712_; 
if (v_isShared_2710_ == 0)
{
v___x_2712_ = v___x_2709_;
goto v_reusejp_2711_;
}
else
{
lean_object* v_reuseFailAlloc_2713_; 
v_reuseFailAlloc_2713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2713_, 0, v_a_2707_);
v___x_2712_ = v_reuseFailAlloc_2713_;
goto v_reusejp_2711_;
}
v_reusejp_2711_:
{
return v___x_2712_;
}
}
}
}
}
}
else
{
lean_object* v_a_2715_; lean_object* v___x_2717_; uint8_t v_isShared_2718_; uint8_t v_isSharedCheck_2722_; 
lean_dec_ref(v___y_2677_);
lean_dec(v_a_2669_);
v_a_2715_ = lean_ctor_get(v___x_2696_, 0);
v_isSharedCheck_2722_ = !lean_is_exclusive(v___x_2696_);
if (v_isSharedCheck_2722_ == 0)
{
v___x_2717_ = v___x_2696_;
v_isShared_2718_ = v_isSharedCheck_2722_;
goto v_resetjp_2716_;
}
else
{
lean_inc(v_a_2715_);
lean_dec(v___x_2696_);
v___x_2717_ = lean_box(0);
v_isShared_2718_ = v_isSharedCheck_2722_;
goto v_resetjp_2716_;
}
v_resetjp_2716_:
{
lean_object* v___x_2720_; 
if (v_isShared_2718_ == 0)
{
v___x_2720_ = v___x_2717_;
goto v_reusejp_2719_;
}
else
{
lean_object* v_reuseFailAlloc_2721_; 
v_reuseFailAlloc_2721_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2721_, 0, v_a_2715_);
v___x_2720_ = v_reuseFailAlloc_2721_;
goto v_reusejp_2719_;
}
v_reusejp_2719_:
{
return v___x_2720_;
}
}
}
}
}
}
else
{
lean_dec_ref(v___y_2662_);
return v___x_2667_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___boxed(lean_object* v___x_2741_, lean_object* v___x_2742_, lean_object* v___x_2743_, lean_object* v___y_2744_, lean_object* v___y_2745_, lean_object* v___y_2746_, lean_object* v___y_2747_, lean_object* v___y_2748_){
_start:
{
uint8_t v___x_2592__boxed_2749_; lean_object* v_res_2750_; 
v___x_2592__boxed_2749_ = lean_unbox(v___x_2742_);
v_res_2750_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0(v___x_2741_, v___x_2592__boxed_2749_, v___x_2743_, v___y_2744_, v___y_2745_, v___y_2746_, v___y_2747_);
lean_dec(v___y_2747_);
lean_dec_ref(v___y_2746_);
lean_dec(v___y_2745_);
return v_res_2750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve(lean_object* v_ty_2751_, lean_object* v_a_2752_, lean_object* v_a_2753_, lean_object* v_a_2754_, lean_object* v_a_2755_){
_start:
{
lean_object* v___x_2757_; uint8_t v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___f_2761_; lean_object* v___x_2762_; 
v___x_2757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2757_, 0, v_ty_2751_);
v___x_2758_ = 0;
v___x_2759_ = lean_box(0);
v___x_2760_ = lean_box(v___x_2758_);
v___f_2761_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2761_, 0, v___x_2757_);
lean_closure_set(v___f_2761_, 1, v___x_2760_);
lean_closure_set(v___f_2761_, 2, v___x_2759_);
v___x_2762_ = lp_mathlib_Lean_observing_x3f___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_process_spec__2___redArg(v___f_2761_, v_a_2752_, v_a_2753_, v_a_2754_, v_a_2755_);
return v___x_2762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___boxed(lean_object* v_ty_2763_, lean_object* v_a_2764_, lean_object* v_a_2765_, lean_object* v_a_2766_, lean_object* v_a_2767_, lean_object* v_a_2768_){
_start:
{
lean_object* v_res_2769_; 
v_res_2769_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve(v_ty_2763_, v_a_2764_, v_a_2765_, v_a_2766_, v_a_2767_);
lean_dec(v_a_2767_);
lean_dec_ref(v_a_2766_);
lean_dec(v_a_2765_);
lean_dec_ref(v_a_2764_);
return v_res_2769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg(lean_object* v_lctx_2770_, lean_object* v_localInsts_2771_, lean_object* v_x_2772_, lean_object* v___y_2773_, lean_object* v___y_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_){
_start:
{
lean_object* v___x_2778_; 
v___x_2778_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_2770_, v_localInsts_2771_, v_x_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_);
if (lean_obj_tag(v___x_2778_) == 0)
{
lean_object* v_a_2779_; lean_object* v___x_2781_; uint8_t v_isShared_2782_; uint8_t v_isSharedCheck_2786_; 
v_a_2779_ = lean_ctor_get(v___x_2778_, 0);
v_isSharedCheck_2786_ = !lean_is_exclusive(v___x_2778_);
if (v_isSharedCheck_2786_ == 0)
{
v___x_2781_ = v___x_2778_;
v_isShared_2782_ = v_isSharedCheck_2786_;
goto v_resetjp_2780_;
}
else
{
lean_inc(v_a_2779_);
lean_dec(v___x_2778_);
v___x_2781_ = lean_box(0);
v_isShared_2782_ = v_isSharedCheck_2786_;
goto v_resetjp_2780_;
}
v_resetjp_2780_:
{
lean_object* v___x_2784_; 
if (v_isShared_2782_ == 0)
{
v___x_2784_ = v___x_2781_;
goto v_reusejp_2783_;
}
else
{
lean_object* v_reuseFailAlloc_2785_; 
v_reuseFailAlloc_2785_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2785_, 0, v_a_2779_);
v___x_2784_ = v_reuseFailAlloc_2785_;
goto v_reusejp_2783_;
}
v_reusejp_2783_:
{
return v___x_2784_;
}
}
}
else
{
lean_object* v_a_2787_; lean_object* v___x_2789_; uint8_t v_isShared_2790_; uint8_t v_isSharedCheck_2794_; 
v_a_2787_ = lean_ctor_get(v___x_2778_, 0);
v_isSharedCheck_2794_ = !lean_is_exclusive(v___x_2778_);
if (v_isSharedCheck_2794_ == 0)
{
v___x_2789_ = v___x_2778_;
v_isShared_2790_ = v_isSharedCheck_2794_;
goto v_resetjp_2788_;
}
else
{
lean_inc(v_a_2787_);
lean_dec(v___x_2778_);
v___x_2789_ = lean_box(0);
v_isShared_2790_ = v_isSharedCheck_2794_;
goto v_resetjp_2788_;
}
v_resetjp_2788_:
{
lean_object* v___x_2792_; 
if (v_isShared_2790_ == 0)
{
v___x_2792_ = v___x_2789_;
goto v_reusejp_2791_;
}
else
{
lean_object* v_reuseFailAlloc_2793_; 
v_reuseFailAlloc_2793_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2793_, 0, v_a_2787_);
v___x_2792_ = v_reuseFailAlloc_2793_;
goto v_reusejp_2791_;
}
v_reusejp_2791_:
{
return v___x_2792_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg___boxed(lean_object* v_lctx_2795_, lean_object* v_localInsts_2796_, lean_object* v_x_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_, lean_object* v___y_2801_, lean_object* v___y_2802_){
_start:
{
lean_object* v_res_2803_; 
v_res_2803_ = lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg(v_lctx_2795_, v_localInsts_2796_, v_x_2797_, v___y_2798_, v___y_2799_, v___y_2800_, v___y_2801_);
lean_dec(v___y_2801_);
lean_dec_ref(v___y_2800_);
lean_dec(v___y_2799_);
lean_dec_ref(v___y_2798_);
return v_res_2803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1(lean_object* v_00_u03b1_2804_, lean_object* v_lctx_2805_, lean_object* v_localInsts_2806_, lean_object* v_x_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_){
_start:
{
lean_object* v___x_2813_; 
v___x_2813_ = lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg(v_lctx_2805_, v_localInsts_2806_, v_x_2807_, v___y_2808_, v___y_2809_, v___y_2810_, v___y_2811_);
return v___x_2813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___boxed(lean_object* v_00_u03b1_2814_, lean_object* v_lctx_2815_, lean_object* v_localInsts_2816_, lean_object* v_x_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_, lean_object* v___y_2820_, lean_object* v___y_2821_, lean_object* v___y_2822_){
_start:
{
lean_object* v_res_2823_; 
v_res_2823_ = lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1(v_00_u03b1_2814_, v_lctx_2815_, v_localInsts_2816_, v_x_2817_, v___y_2818_, v___y_2819_, v___y_2820_, v___y_2821_);
lean_dec(v___y_2821_);
lean_dec_ref(v___y_2820_);
lean_dec(v___y_2819_);
lean_dec_ref(v___y_2818_);
return v_res_2823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__0(lean_object* v_cls_2824_, lean_object* v___y_2825_, lean_object* v___y_2826_, lean_object* v___y_2827_, lean_object* v___y_2828_){
_start:
{
lean_object* v_options_2830_; uint8_t v_hasTrace_2831_; 
v_options_2830_ = lean_ctor_get(v___y_2827_, 2);
v_hasTrace_2831_ = lean_ctor_get_uint8(v_options_2830_, sizeof(void*)*1);
if (v_hasTrace_2831_ == 0)
{
lean_object* v___x_2832_; lean_object* v___x_2833_; 
lean_dec(v_cls_2824_);
v___x_2832_ = lean_box(v_hasTrace_2831_);
v___x_2833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2833_, 0, v___x_2832_);
return v___x_2833_;
}
else
{
lean_object* v_inheritedTraceOptions_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; uint8_t v___x_2837_; lean_object* v___x_2838_; lean_object* v___x_2839_; 
v_inheritedTraceOptions_2834_ = lean_ctor_get(v___y_2827_, 13);
v___x_2835_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__1));
v___x_2836_ = l_Lean_Name_append(v___x_2835_, v_cls_2824_);
v___x_2837_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2834_, v_options_2830_, v___x_2836_);
lean_dec(v___x_2836_);
v___x_2838_ = lean_box(v___x_2837_);
v___x_2839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2839_, 0, v___x_2838_);
return v___x_2839_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__0___boxed(lean_object* v_cls_2840_, lean_object* v___y_2841_, lean_object* v___y_2842_, lean_object* v___y_2843_, lean_object* v___y_2844_, lean_object* v___y_2845_){
_start:
{
lean_object* v_res_2846_; 
v_res_2846_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__0(v_cls_2840_, v___y_2841_, v___y_2842_, v___y_2843_, v___y_2844_);
lean_dec(v___y_2844_);
lean_dec_ref(v___y_2843_);
lean_dec(v___y_2842_);
lean_dec_ref(v___y_2841_);
return v_res_2846_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___redArg(lean_object* v_xs_2847_, lean_object* v_eqs_2848_, lean_object* v___x_2849_, lean_object* v_ys_2850_, lean_object* v_kinds_2851_, lean_object* v_range_2852_, lean_object* v_b_2853_, lean_object* v_i_2854_, lean_object* v___y_2855_, lean_object* v___y_2856_, lean_object* v___y_2857_, lean_object* v___y_2858_){
_start:
{
lean_object* v_stop_2860_; lean_object* v_step_2861_; lean_object* v_a_2863_; uint8_t v___x_2866_; 
v_stop_2860_ = lean_ctor_get(v_range_2852_, 1);
v_step_2861_ = lean_ctor_get(v_range_2852_, 2);
v___x_2866_ = lean_nat_dec_lt(v_i_2854_, v_stop_2860_);
if (v___x_2866_ == 0)
{
lean_object* v___x_2867_; 
lean_dec(v_i_2854_);
lean_dec_ref(v___x_2849_);
v___x_2867_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2867_, 0, v_b_2853_);
return v___x_2867_;
}
else
{
lean_object* v_snd_2868_; lean_object* v_snd_2869_; lean_object* v_fst_2870_; lean_object* v___x_2872_; uint8_t v_isShared_2873_; uint8_t v_isSharedCheck_2969_; 
v_snd_2868_ = lean_ctor_get(v_b_2853_, 1);
lean_inc(v_snd_2868_);
v_snd_2869_ = lean_ctor_get(v_snd_2868_, 1);
lean_inc(v_snd_2869_);
v_fst_2870_ = lean_ctor_get(v_b_2853_, 0);
v_isSharedCheck_2969_ = !lean_is_exclusive(v_b_2853_);
if (v_isSharedCheck_2969_ == 0)
{
lean_object* v_unused_2970_; 
v_unused_2970_ = lean_ctor_get(v_b_2853_, 1);
lean_dec(v_unused_2970_);
v___x_2872_ = v_b_2853_;
v_isShared_2873_ = v_isSharedCheck_2969_;
goto v_resetjp_2871_;
}
else
{
lean_inc(v_fst_2870_);
lean_dec(v_b_2853_);
v___x_2872_ = lean_box(0);
v_isShared_2873_ = v_isSharedCheck_2969_;
goto v_resetjp_2871_;
}
v_resetjp_2871_:
{
lean_object* v_fst_2874_; lean_object* v___x_2876_; uint8_t v_isShared_2877_; uint8_t v_isSharedCheck_2967_; 
v_fst_2874_ = lean_ctor_get(v_snd_2868_, 0);
v_isSharedCheck_2967_ = !lean_is_exclusive(v_snd_2868_);
if (v_isSharedCheck_2967_ == 0)
{
lean_object* v_unused_2968_; 
v_unused_2968_ = lean_ctor_get(v_snd_2868_, 1);
lean_dec(v_unused_2968_);
v___x_2876_ = v_snd_2868_;
v_isShared_2877_ = v_isSharedCheck_2967_;
goto v_resetjp_2875_;
}
else
{
lean_inc(v_fst_2874_);
lean_dec(v_snd_2868_);
v___x_2876_ = lean_box(0);
v_isShared_2877_ = v_isSharedCheck_2967_;
goto v_resetjp_2875_;
}
v_resetjp_2875_:
{
lean_object* v_fst_2878_; lean_object* v_snd_2879_; lean_object* v___x_2881_; uint8_t v_isShared_2882_; uint8_t v_isSharedCheck_2966_; 
v_fst_2878_ = lean_ctor_get(v_snd_2869_, 0);
v_snd_2879_ = lean_ctor_get(v_snd_2869_, 1);
v_isSharedCheck_2966_ = !lean_is_exclusive(v_snd_2869_);
if (v_isSharedCheck_2966_ == 0)
{
v___x_2881_ = v_snd_2869_;
v_isShared_2882_ = v_isSharedCheck_2966_;
goto v_resetjp_2880_;
}
else
{
lean_inc(v_snd_2879_);
lean_inc(v_fst_2878_);
lean_dec(v_snd_2869_);
v___x_2881_ = lean_box(0);
v_isShared_2882_ = v_isSharedCheck_2966_;
goto v_resetjp_2880_;
}
v_resetjp_2880_:
{
lean_object* v___x_2883_; lean_object* v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2887_; 
v___x_2883_ = l_Lean_instInhabitedExpr;
v___x_2884_ = lean_box(0);
v___x_2885_ = lean_array_get_borrowed(v___x_2883_, v_xs_2847_, v_i_2854_);
lean_inc(v___x_2885_);
v___x_2886_ = lean_array_push(v_fst_2870_, v___x_2885_);
v___x_2887_ = lean_array_get_borrowed(v___x_2884_, v_eqs_2848_, v_i_2854_);
if (lean_obj_tag(v___x_2887_) == 1)
{
lean_object* v_val_2888_; lean_object* v_snd_2889_; lean_object* v___x_2891_; uint8_t v_isShared_2892_; uint8_t v_isSharedCheck_2952_; 
lean_del_object(v___x_2876_);
lean_del_object(v___x_2872_);
v_val_2888_ = lean_ctor_get(v___x_2887_, 0);
lean_inc(v_val_2888_);
v_snd_2889_ = lean_ctor_get(v_val_2888_, 1);
v_isSharedCheck_2952_ = !lean_is_exclusive(v_val_2888_);
if (v_isSharedCheck_2952_ == 0)
{
lean_object* v_unused_2953_; 
v_unused_2953_ = lean_ctor_get(v_val_2888_, 0);
lean_dec(v_unused_2953_);
v___x_2891_ = v_val_2888_;
v_isShared_2892_ = v_isSharedCheck_2952_;
goto v_resetjp_2890_;
}
else
{
lean_inc(v_snd_2889_);
lean_dec(v_val_2888_);
v___x_2891_ = lean_box(0);
v_isShared_2892_ = v_isSharedCheck_2952_;
goto v_resetjp_2890_;
}
v_resetjp_2890_:
{
lean_object* v_fst_2893_; lean_object* v___x_2895_; uint8_t v_isShared_2896_; uint8_t v_isSharedCheck_2950_; 
v_fst_2893_ = lean_ctor_get(v_snd_2889_, 0);
v_isSharedCheck_2950_ = !lean_is_exclusive(v_snd_2889_);
if (v_isSharedCheck_2950_ == 0)
{
lean_object* v_unused_2951_; 
v_unused_2951_ = lean_ctor_get(v_snd_2889_, 1);
lean_dec(v_unused_2951_);
v___x_2895_ = v_snd_2889_;
v_isShared_2896_ = v_isSharedCheck_2950_;
goto v_resetjp_2894_;
}
else
{
lean_inc(v_fst_2893_);
lean_dec(v_snd_2889_);
v___x_2895_ = lean_box(0);
v_isShared_2896_ = v_isSharedCheck_2950_;
goto v_resetjp_2894_;
}
v_resetjp_2894_:
{
lean_object* v_localInstances_2897_; lean_object* v___x_2898_; 
v_localInstances_2897_ = lean_ctor_get(v___y_2855_, 3);
lean_inc(v___y_2858_);
lean_inc_ref(v___y_2857_);
lean_inc(v___y_2856_);
lean_inc_ref(v___y_2855_);
lean_inc(v_fst_2893_);
v___x_2898_ = lean_infer_type(v_fst_2893_, v___y_2855_, v___y_2856_, v___y_2857_, v___y_2858_);
if (lean_obj_tag(v___x_2898_) == 0)
{
lean_object* v_a_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; 
v_a_2899_ = lean_ctor_get(v___x_2898_, 0);
lean_inc(v_a_2899_);
lean_dec_ref_known(v___x_2898_, 1);
v___x_2900_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___boxed), 6, 1);
lean_closure_set(v___x_2900_, 0, v_a_2899_);
lean_inc_ref(v_localInstances_2897_);
lean_inc_ref(v___x_2849_);
v___x_2901_ = lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg(v___x_2849_, v_localInstances_2897_, v___x_2900_, v___y_2855_, v___y_2856_, v___y_2857_, v___y_2858_);
if (lean_obj_tag(v___x_2901_) == 0)
{
lean_object* v_a_2902_; lean_object* v___x_2903_; lean_object* v___x_2904_; 
v_a_2902_ = lean_ctor_get(v___x_2901_, 0);
lean_inc(v_a_2902_);
lean_dec_ref_known(v___x_2901_, 1);
v___x_2903_ = lean_array_get_borrowed(v___x_2883_, v_ys_2850_, v_i_2854_);
lean_inc(v___x_2903_);
v___x_2904_ = lean_array_push(v___x_2886_, v___x_2903_);
if (lean_obj_tag(v_a_2902_) == 1)
{
lean_object* v_val_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; uint8_t v___x_2908_; lean_object* v___x_2909_; lean_object* v___x_2910_; lean_object* v___x_2912_; 
v_val_2905_ = lean_ctor_get(v_a_2902_, 0);
lean_inc(v_val_2905_);
lean_dec_ref_known(v_a_2902_, 1);
v___x_2906_ = lean_array_push(v_fst_2874_, v_fst_2893_);
v___x_2907_ = lean_array_push(v_fst_2878_, v_val_2905_);
v___x_2908_ = 5;
v___x_2909_ = lean_box(v___x_2908_);
v___x_2910_ = lean_array_push(v_snd_2879_, v___x_2909_);
if (v_isShared_2896_ == 0)
{
lean_ctor_set(v___x_2895_, 1, v___x_2910_);
lean_ctor_set(v___x_2895_, 0, v___x_2907_);
v___x_2912_ = v___x_2895_;
goto v_reusejp_2911_;
}
else
{
lean_object* v_reuseFailAlloc_2919_; 
v_reuseFailAlloc_2919_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2919_, 0, v___x_2907_);
lean_ctor_set(v_reuseFailAlloc_2919_, 1, v___x_2910_);
v___x_2912_ = v_reuseFailAlloc_2919_;
goto v_reusejp_2911_;
}
v_reusejp_2911_:
{
lean_object* v___x_2914_; 
if (v_isShared_2892_ == 0)
{
lean_ctor_set(v___x_2891_, 1, v___x_2912_);
lean_ctor_set(v___x_2891_, 0, v___x_2906_);
v___x_2914_ = v___x_2891_;
goto v_reusejp_2913_;
}
else
{
lean_object* v_reuseFailAlloc_2918_; 
v_reuseFailAlloc_2918_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2918_, 0, v___x_2906_);
lean_ctor_set(v_reuseFailAlloc_2918_, 1, v___x_2912_);
v___x_2914_ = v_reuseFailAlloc_2918_;
goto v_reusejp_2913_;
}
v_reusejp_2913_:
{
lean_object* v___x_2916_; 
if (v_isShared_2882_ == 0)
{
lean_ctor_set(v___x_2881_, 1, v___x_2914_);
lean_ctor_set(v___x_2881_, 0, v___x_2904_);
v___x_2916_ = v___x_2881_;
goto v_reusejp_2915_;
}
else
{
lean_object* v_reuseFailAlloc_2917_; 
v_reuseFailAlloc_2917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2917_, 0, v___x_2904_);
lean_ctor_set(v_reuseFailAlloc_2917_, 1, v___x_2914_);
v___x_2916_ = v_reuseFailAlloc_2917_;
goto v_reusejp_2915_;
}
v_reusejp_2915_:
{
v_a_2863_ = v___x_2916_;
goto v___jp_2862_;
}
}
}
}
else
{
uint8_t v___x_2920_; lean_object* v___x_2921_; lean_object* v___x_2922_; lean_object* v___x_2923_; lean_object* v___x_2924_; lean_object* v___x_2926_; 
lean_dec(v_a_2902_);
v___x_2920_ = 0;
v___x_2921_ = lean_array_push(v___x_2904_, v_fst_2893_);
v___x_2922_ = lean_box(v___x_2920_);
v___x_2923_ = lean_array_get(v___x_2922_, v_kinds_2851_, v_i_2854_);
lean_dec(v___x_2922_);
v___x_2924_ = lean_array_push(v_snd_2879_, v___x_2923_);
if (v_isShared_2896_ == 0)
{
lean_ctor_set(v___x_2895_, 1, v___x_2924_);
lean_ctor_set(v___x_2895_, 0, v_fst_2878_);
v___x_2926_ = v___x_2895_;
goto v_reusejp_2925_;
}
else
{
lean_object* v_reuseFailAlloc_2933_; 
v_reuseFailAlloc_2933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2933_, 0, v_fst_2878_);
lean_ctor_set(v_reuseFailAlloc_2933_, 1, v___x_2924_);
v___x_2926_ = v_reuseFailAlloc_2933_;
goto v_reusejp_2925_;
}
v_reusejp_2925_:
{
lean_object* v___x_2928_; 
if (v_isShared_2892_ == 0)
{
lean_ctor_set(v___x_2891_, 1, v___x_2926_);
lean_ctor_set(v___x_2891_, 0, v_fst_2874_);
v___x_2928_ = v___x_2891_;
goto v_reusejp_2927_;
}
else
{
lean_object* v_reuseFailAlloc_2932_; 
v_reuseFailAlloc_2932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2932_, 0, v_fst_2874_);
lean_ctor_set(v_reuseFailAlloc_2932_, 1, v___x_2926_);
v___x_2928_ = v_reuseFailAlloc_2932_;
goto v_reusejp_2927_;
}
v_reusejp_2927_:
{
lean_object* v___x_2930_; 
if (v_isShared_2882_ == 0)
{
lean_ctor_set(v___x_2881_, 1, v___x_2928_);
lean_ctor_set(v___x_2881_, 0, v___x_2921_);
v___x_2930_ = v___x_2881_;
goto v_reusejp_2929_;
}
else
{
lean_object* v_reuseFailAlloc_2931_; 
v_reuseFailAlloc_2931_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2931_, 0, v___x_2921_);
lean_ctor_set(v_reuseFailAlloc_2931_, 1, v___x_2928_);
v___x_2930_ = v_reuseFailAlloc_2931_;
goto v_reusejp_2929_;
}
v_reusejp_2929_:
{
v_a_2863_ = v___x_2930_;
goto v___jp_2862_;
}
}
}
}
}
else
{
lean_object* v_a_2934_; lean_object* v___x_2936_; uint8_t v_isShared_2937_; uint8_t v_isSharedCheck_2941_; 
lean_del_object(v___x_2895_);
lean_dec(v_fst_2893_);
lean_del_object(v___x_2891_);
lean_dec_ref(v___x_2886_);
lean_del_object(v___x_2881_);
lean_dec(v_snd_2879_);
lean_dec(v_fst_2878_);
lean_dec(v_fst_2874_);
lean_dec(v_i_2854_);
lean_dec_ref(v___x_2849_);
v_a_2934_ = lean_ctor_get(v___x_2901_, 0);
v_isSharedCheck_2941_ = !lean_is_exclusive(v___x_2901_);
if (v_isSharedCheck_2941_ == 0)
{
v___x_2936_ = v___x_2901_;
v_isShared_2937_ = v_isSharedCheck_2941_;
goto v_resetjp_2935_;
}
else
{
lean_inc(v_a_2934_);
lean_dec(v___x_2901_);
v___x_2936_ = lean_box(0);
v_isShared_2937_ = v_isSharedCheck_2941_;
goto v_resetjp_2935_;
}
v_resetjp_2935_:
{
lean_object* v___x_2939_; 
if (v_isShared_2937_ == 0)
{
v___x_2939_ = v___x_2936_;
goto v_reusejp_2938_;
}
else
{
lean_object* v_reuseFailAlloc_2940_; 
v_reuseFailAlloc_2940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2940_, 0, v_a_2934_);
v___x_2939_ = v_reuseFailAlloc_2940_;
goto v_reusejp_2938_;
}
v_reusejp_2938_:
{
return v___x_2939_;
}
}
}
}
else
{
lean_object* v_a_2942_; lean_object* v___x_2944_; uint8_t v_isShared_2945_; uint8_t v_isSharedCheck_2949_; 
lean_del_object(v___x_2895_);
lean_dec(v_fst_2893_);
lean_del_object(v___x_2891_);
lean_dec_ref(v___x_2886_);
lean_del_object(v___x_2881_);
lean_dec(v_snd_2879_);
lean_dec(v_fst_2878_);
lean_dec(v_fst_2874_);
lean_dec(v_i_2854_);
lean_dec_ref(v___x_2849_);
v_a_2942_ = lean_ctor_get(v___x_2898_, 0);
v_isSharedCheck_2949_ = !lean_is_exclusive(v___x_2898_);
if (v_isSharedCheck_2949_ == 0)
{
v___x_2944_ = v___x_2898_;
v_isShared_2945_ = v_isSharedCheck_2949_;
goto v_resetjp_2943_;
}
else
{
lean_inc(v_a_2942_);
lean_dec(v___x_2898_);
v___x_2944_ = lean_box(0);
v_isShared_2945_ = v_isSharedCheck_2949_;
goto v_resetjp_2943_;
}
v_resetjp_2943_:
{
lean_object* v___x_2947_; 
if (v_isShared_2945_ == 0)
{
v___x_2947_ = v___x_2944_;
goto v_reusejp_2946_;
}
else
{
lean_object* v_reuseFailAlloc_2948_; 
v_reuseFailAlloc_2948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2948_, 0, v_a_2942_);
v___x_2947_ = v_reuseFailAlloc_2948_;
goto v_reusejp_2946_;
}
v_reusejp_2946_:
{
return v___x_2947_;
}
}
}
}
}
}
else
{
uint8_t v___x_2954_; lean_object* v___x_2955_; lean_object* v___x_2956_; lean_object* v___x_2958_; 
v___x_2954_ = 0;
v___x_2955_ = lean_box(v___x_2954_);
v___x_2956_ = lean_array_push(v_snd_2879_, v___x_2955_);
if (v_isShared_2882_ == 0)
{
lean_ctor_set(v___x_2881_, 1, v___x_2956_);
v___x_2958_ = v___x_2881_;
goto v_reusejp_2957_;
}
else
{
lean_object* v_reuseFailAlloc_2965_; 
v_reuseFailAlloc_2965_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2965_, 0, v_fst_2878_);
lean_ctor_set(v_reuseFailAlloc_2965_, 1, v___x_2956_);
v___x_2958_ = v_reuseFailAlloc_2965_;
goto v_reusejp_2957_;
}
v_reusejp_2957_:
{
lean_object* v___x_2960_; 
if (v_isShared_2877_ == 0)
{
lean_ctor_set(v___x_2876_, 1, v___x_2958_);
v___x_2960_ = v___x_2876_;
goto v_reusejp_2959_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v_fst_2874_);
lean_ctor_set(v_reuseFailAlloc_2964_, 1, v___x_2958_);
v___x_2960_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2959_;
}
v_reusejp_2959_:
{
lean_object* v___x_2962_; 
if (v_isShared_2873_ == 0)
{
lean_ctor_set(v___x_2872_, 1, v___x_2960_);
lean_ctor_set(v___x_2872_, 0, v___x_2886_);
v___x_2962_ = v___x_2872_;
goto v_reusejp_2961_;
}
else
{
lean_object* v_reuseFailAlloc_2963_; 
v_reuseFailAlloc_2963_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2963_, 0, v___x_2886_);
lean_ctor_set(v_reuseFailAlloc_2963_, 1, v___x_2960_);
v___x_2962_ = v_reuseFailAlloc_2963_;
goto v_reusejp_2961_;
}
v_reusejp_2961_:
{
v_a_2863_ = v___x_2962_;
goto v___jp_2862_;
}
}
}
}
}
}
}
}
v___jp_2862_:
{
lean_object* v___x_2864_; 
v___x_2864_ = lean_nat_add(v_i_2854_, v_step_2861_);
lean_dec(v_i_2854_);
v_b_2853_ = v_a_2863_;
v_i_2854_ = v___x_2864_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___redArg___boxed(lean_object* v_xs_2971_, lean_object* v_eqs_2972_, lean_object* v___x_2973_, lean_object* v_ys_2974_, lean_object* v_kinds_2975_, lean_object* v_range_2976_, lean_object* v_b_2977_, lean_object* v_i_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_){
_start:
{
lean_object* v_res_2984_; 
v_res_2984_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___redArg(v_xs_2971_, v_eqs_2972_, v___x_2973_, v_ys_2974_, v_kinds_2975_, v_range_2976_, v_b_2977_, v_i_2978_, v___y_2979_, v___y_2980_, v___y_2981_, v___y_2982_);
lean_dec(v___y_2982_);
lean_dec_ref(v___y_2981_);
lean_dec(v___y_2980_);
lean_dec_ref(v___y_2979_);
lean_dec_ref(v_range_2976_);
lean_dec_ref(v_kinds_2975_);
lean_dec_ref(v_ys_2974_);
lean_dec_ref(v_eqs_2972_);
lean_dec_ref(v_xs_2971_);
return v_res_2984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4___lam__0(uint8_t v___y_2985_){
_start:
{
lean_object* v___x_2986_; lean_object* v___x_2987_; 
v___x_2986_ = lean_unsigned_to_nat(0u);
v___x_2987_ = l_Lean_Meta_instReprCongrArgKind_repr(v___y_2985_, v___x_2986_);
return v___x_2987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4___lam__0___boxed(lean_object* v___y_2988_){
_start:
{
uint8_t v___y_18781__boxed_2989_; lean_object* v_res_2990_; 
v___y_18781__boxed_2989_ = lean_unbox(v___y_2988_);
v_res_2990_ = lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4___lam__0(v___y_18781__boxed_2989_);
return v_res_2990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4_spec__5_spec__11(lean_object* v_x_2991_, lean_object* v_x_2992_, lean_object* v_x_2993_){
_start:
{
if (lean_obj_tag(v_x_2993_) == 0)
{
lean_dec(v_x_2991_);
return v_x_2992_;
}
else
{
lean_object* v_head_2994_; lean_object* v_tail_2995_; lean_object* v___x_2997_; uint8_t v_isShared_2998_; uint8_t v_isSharedCheck_3007_; 
v_head_2994_ = lean_ctor_get(v_x_2993_, 0);
v_tail_2995_ = lean_ctor_get(v_x_2993_, 1);
v_isSharedCheck_3007_ = !lean_is_exclusive(v_x_2993_);
if (v_isSharedCheck_3007_ == 0)
{
v___x_2997_ = v_x_2993_;
v_isShared_2998_ = v_isSharedCheck_3007_;
goto v_resetjp_2996_;
}
else
{
lean_inc(v_tail_2995_);
lean_inc(v_head_2994_);
lean_dec(v_x_2993_);
v___x_2997_ = lean_box(0);
v_isShared_2998_ = v_isSharedCheck_3007_;
goto v_resetjp_2996_;
}
v_resetjp_2996_:
{
lean_object* v___x_3000_; 
lean_inc(v_x_2991_);
if (v_isShared_2998_ == 0)
{
lean_ctor_set_tag(v___x_2997_, 5);
lean_ctor_set(v___x_2997_, 1, v_x_2991_);
lean_ctor_set(v___x_2997_, 0, v_x_2992_);
v___x_3000_ = v___x_2997_;
goto v_reusejp_2999_;
}
else
{
lean_object* v_reuseFailAlloc_3006_; 
v_reuseFailAlloc_3006_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3006_, 0, v_x_2992_);
lean_ctor_set(v_reuseFailAlloc_3006_, 1, v_x_2991_);
v___x_3000_ = v_reuseFailAlloc_3006_;
goto v_reusejp_2999_;
}
v_reusejp_2999_:
{
lean_object* v___x_3001_; uint8_t v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; 
v___x_3001_ = lean_unsigned_to_nat(0u);
v___x_3002_ = lean_unbox(v_head_2994_);
lean_dec(v_head_2994_);
v___x_3003_ = l_Lean_Meta_instReprCongrArgKind_repr(v___x_3002_, v___x_3001_);
v___x_3004_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_3004_, 0, v___x_3000_);
lean_ctor_set(v___x_3004_, 1, v___x_3003_);
v_x_2992_ = v___x_3004_;
v_x_2993_ = v_tail_2995_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4_spec__5(lean_object* v_x_3008_, lean_object* v_x_3009_, lean_object* v_x_3010_){
_start:
{
if (lean_obj_tag(v_x_3010_) == 0)
{
lean_dec(v_x_3008_);
return v_x_3009_;
}
else
{
lean_object* v_head_3011_; lean_object* v_tail_3012_; lean_object* v___x_3014_; uint8_t v_isShared_3015_; uint8_t v_isSharedCheck_3024_; 
v_head_3011_ = lean_ctor_get(v_x_3010_, 0);
v_tail_3012_ = lean_ctor_get(v_x_3010_, 1);
v_isSharedCheck_3024_ = !lean_is_exclusive(v_x_3010_);
if (v_isSharedCheck_3024_ == 0)
{
v___x_3014_ = v_x_3010_;
v_isShared_3015_ = v_isSharedCheck_3024_;
goto v_resetjp_3013_;
}
else
{
lean_inc(v_tail_3012_);
lean_inc(v_head_3011_);
lean_dec(v_x_3010_);
v___x_3014_ = lean_box(0);
v_isShared_3015_ = v_isSharedCheck_3024_;
goto v_resetjp_3013_;
}
v_resetjp_3013_:
{
lean_object* v___x_3017_; 
lean_inc(v_x_3008_);
if (v_isShared_3015_ == 0)
{
lean_ctor_set_tag(v___x_3014_, 5);
lean_ctor_set(v___x_3014_, 1, v_x_3008_);
lean_ctor_set(v___x_3014_, 0, v_x_3009_);
v___x_3017_ = v___x_3014_;
goto v_reusejp_3016_;
}
else
{
lean_object* v_reuseFailAlloc_3023_; 
v_reuseFailAlloc_3023_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3023_, 0, v_x_3009_);
lean_ctor_set(v_reuseFailAlloc_3023_, 1, v_x_3008_);
v___x_3017_ = v_reuseFailAlloc_3023_;
goto v_reusejp_3016_;
}
v_reusejp_3016_:
{
lean_object* v___x_3018_; uint8_t v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; 
v___x_3018_ = lean_unsigned_to_nat(0u);
v___x_3019_ = lean_unbox(v_head_3011_);
lean_dec(v_head_3011_);
v___x_3020_ = l_Lean_Meta_instReprCongrArgKind_repr(v___x_3019_, v___x_3018_);
v___x_3021_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_3021_, 0, v___x_3017_);
lean_ctor_set(v___x_3021_, 1, v___x_3020_);
v___x_3022_ = lp_mathlib_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4_spec__5_spec__11(v_x_3008_, v___x_3021_, v_tail_3012_);
return v___x_3022_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4(lean_object* v_x_3025_, lean_object* v_x_3026_){
_start:
{
if (lean_obj_tag(v_x_3025_) == 0)
{
lean_object* v___x_3027_; 
lean_dec(v_x_3026_);
v___x_3027_ = lean_box(0);
return v___x_3027_;
}
else
{
lean_object* v_tail_3028_; 
v_tail_3028_ = lean_ctor_get(v_x_3025_, 1);
if (lean_obj_tag(v_tail_3028_) == 0)
{
lean_object* v_head_3029_; uint8_t v___x_3030_; lean_object* v___x_3031_; 
lean_dec(v_x_3026_);
v_head_3029_ = lean_ctor_get(v_x_3025_, 0);
lean_inc(v_head_3029_);
lean_dec_ref_known(v_x_3025_, 2);
v___x_3030_ = lean_unbox(v_head_3029_);
lean_dec(v_head_3029_);
v___x_3031_ = lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4___lam__0(v___x_3030_);
return v___x_3031_;
}
else
{
lean_object* v_head_3032_; uint8_t v___x_3033_; lean_object* v___x_3034_; lean_object* v___x_3035_; 
lean_inc(v_tail_3028_);
v_head_3032_ = lean_ctor_get(v_x_3025_, 0);
lean_inc(v_head_3032_);
lean_dec_ref_known(v_x_3025_, 2);
v___x_3033_ = lean_unbox(v_head_3032_);
lean_dec(v_head_3032_);
v___x_3034_ = lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4___lam__0(v___x_3033_);
v___x_3035_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4_spec__5(v_x_3026_, v___x_3034_, v_tail_3028_);
return v___x_3035_;
}
}
}
}
static lean_object* _init_lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__5(void){
_start:
{
lean_object* v___x_3044_; lean_object* v___x_3045_; 
v___x_3044_ = ((lean_object*)(lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__0));
v___x_3045_ = lean_string_length(v___x_3044_);
return v___x_3045_;
}
}
static lean_object* _init_lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__6(void){
_start:
{
lean_object* v___x_3046_; lean_object* v___x_3047_; 
v___x_3046_ = lean_obj_once(&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__5, &lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__5_once, _init_lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__5);
v___x_3047_ = lean_nat_to_int(v___x_3046_);
return v___x_3047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4(lean_object* v_xs_3055_){
_start:
{
lean_object* v___x_3056_; lean_object* v___x_3057_; uint8_t v___x_3058_; 
v___x_3056_ = lean_array_get_size(v_xs_3055_);
v___x_3057_ = lean_unsigned_to_nat(0u);
v___x_3058_ = lean_nat_dec_eq(v___x_3056_, v___x_3057_);
if (v___x_3058_ == 0)
{
lean_object* v___x_3059_; lean_object* v___x_3060_; lean_object* v___x_3061_; lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; lean_object* v___x_3065_; lean_object* v___x_3066_; lean_object* v___x_3067_; lean_object* v___x_3068_; 
v___x_3059_ = lean_array_to_list(v_xs_3055_);
v___x_3060_ = ((lean_object*)(lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__3));
v___x_3061_ = lp_mathlib_Std_Format_joinSep___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__4(v___x_3059_, v___x_3060_);
v___x_3062_ = lean_obj_once(&lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__6, &lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__6_once, _init_lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__6);
v___x_3063_ = ((lean_object*)(lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__7));
v___x_3064_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_3064_, 0, v___x_3063_);
lean_ctor_set(v___x_3064_, 1, v___x_3061_);
v___x_3065_ = ((lean_object*)(lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__8));
v___x_3066_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_3066_, 0, v___x_3064_);
lean_ctor_set(v___x_3066_, 1, v___x_3065_);
v___x_3067_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_3067_, 0, v___x_3062_);
lean_ctor_set(v___x_3067_, 1, v___x_3066_);
v___x_3068_ = l_Std_Format_fill(v___x_3067_);
return v___x_3068_;
}
else
{
lean_object* v___x_3069_; 
lean_dec_ref(v_xs_3055_);
v___x_3069_ = ((lean_object*)(lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4___closed__10));
return v___x_3069_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___redArg(lean_object* v_xs_3070_, lean_object* v_eqs_3071_, lean_object* v_ys_3072_, lean_object* v_range_3073_, lean_object* v_b_3074_, lean_object* v_i_3075_){
_start:
{
lean_object* v_stop_3077_; lean_object* v_step_3078_; lean_object* v_a_3080_; uint8_t v___x_3083_; 
v_stop_3077_ = lean_ctor_get(v_range_3073_, 1);
v_step_3078_ = lean_ctor_get(v_range_3073_, 2);
v___x_3083_ = lean_nat_dec_lt(v_i_3075_, v_stop_3077_);
if (v___x_3083_ == 0)
{
lean_object* v___x_3084_; 
lean_dec(v_i_3075_);
v___x_3084_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3084_, 0, v_b_3074_);
return v___x_3084_;
}
else
{
lean_object* v_snd_3085_; lean_object* v_fst_3086_; lean_object* v___x_3088_; uint8_t v_isShared_3089_; uint8_t v_isSharedCheck_3135_; 
v_snd_3085_ = lean_ctor_get(v_b_3074_, 1);
v_fst_3086_ = lean_ctor_get(v_b_3074_, 0);
v_isSharedCheck_3135_ = !lean_is_exclusive(v_b_3074_);
if (v_isSharedCheck_3135_ == 0)
{
v___x_3088_ = v_b_3074_;
v_isShared_3089_ = v_isSharedCheck_3135_;
goto v_resetjp_3087_;
}
else
{
lean_inc(v_snd_3085_);
lean_inc(v_fst_3086_);
lean_dec(v_b_3074_);
v___x_3088_ = lean_box(0);
v_isShared_3089_ = v_isSharedCheck_3135_;
goto v_resetjp_3087_;
}
v_resetjp_3087_:
{
lean_object* v_fst_3090_; lean_object* v_snd_3091_; lean_object* v___x_3093_; uint8_t v_isShared_3094_; uint8_t v_isSharedCheck_3134_; 
v_fst_3090_ = lean_ctor_get(v_snd_3085_, 0);
v_snd_3091_ = lean_ctor_get(v_snd_3085_, 1);
v_isSharedCheck_3134_ = !lean_is_exclusive(v_snd_3085_);
if (v_isSharedCheck_3134_ == 0)
{
v___x_3093_ = v_snd_3085_;
v_isShared_3094_ = v_isSharedCheck_3134_;
goto v_resetjp_3092_;
}
else
{
lean_inc(v_snd_3091_);
lean_inc(v_fst_3090_);
lean_dec(v_snd_3085_);
v___x_3093_ = lean_box(0);
v_isShared_3094_ = v_isSharedCheck_3134_;
goto v_resetjp_3092_;
}
v_resetjp_3092_:
{
lean_object* v___x_3095_; lean_object* v___x_3096_; lean_object* v___x_3097_; lean_object* v___x_3098_; lean_object* v___x_3099_; lean_object* v___x_3100_; lean_object* v___x_3101_; 
v___x_3095_ = l_Lean_instInhabitedExpr;
v___x_3096_ = lean_box(0);
v___x_3097_ = lean_array_get_borrowed(v___x_3095_, v_xs_3070_, v_i_3075_);
lean_inc_n(v___x_3097_, 3);
v___x_3098_ = lean_array_push(v_fst_3086_, v___x_3097_);
v___x_3099_ = lean_array_push(v_fst_3090_, v___x_3097_);
v___x_3100_ = lean_array_push(v_snd_3091_, v___x_3097_);
v___x_3101_ = lean_array_get_borrowed(v___x_3096_, v_eqs_3071_, v_i_3075_);
if (lean_obj_tag(v___x_3101_) == 1)
{
lean_object* v_val_3102_; lean_object* v_snd_3103_; lean_object* v_fst_3104_; lean_object* v___x_3106_; uint8_t v_isShared_3107_; uint8_t v_isSharedCheck_3127_; 
lean_del_object(v___x_3093_);
lean_del_object(v___x_3088_);
v_val_3102_ = lean_ctor_get(v___x_3101_, 0);
lean_inc(v_val_3102_);
v_snd_3103_ = lean_ctor_get(v_val_3102_, 1);
v_fst_3104_ = lean_ctor_get(v_val_3102_, 0);
v_isSharedCheck_3127_ = !lean_is_exclusive(v_val_3102_);
if (v_isSharedCheck_3127_ == 0)
{
v___x_3106_ = v_val_3102_;
v_isShared_3107_ = v_isSharedCheck_3127_;
goto v_resetjp_3105_;
}
else
{
lean_inc(v_snd_3103_);
lean_inc(v_fst_3104_);
lean_dec(v_val_3102_);
v___x_3106_ = lean_box(0);
v_isShared_3107_ = v_isSharedCheck_3127_;
goto v_resetjp_3105_;
}
v_resetjp_3105_:
{
lean_object* v_fst_3108_; lean_object* v_snd_3109_; lean_object* v___x_3111_; uint8_t v_isShared_3112_; uint8_t v_isSharedCheck_3126_; 
v_fst_3108_ = lean_ctor_get(v_snd_3103_, 0);
v_snd_3109_ = lean_ctor_get(v_snd_3103_, 1);
v_isSharedCheck_3126_ = !lean_is_exclusive(v_snd_3103_);
if (v_isSharedCheck_3126_ == 0)
{
v___x_3111_ = v_snd_3103_;
v_isShared_3112_ = v_isSharedCheck_3126_;
goto v_resetjp_3110_;
}
else
{
lean_inc(v_snd_3109_);
lean_inc(v_fst_3108_);
lean_dec(v_snd_3103_);
v___x_3111_ = lean_box(0);
v_isShared_3112_ = v_isSharedCheck_3126_;
goto v_resetjp_3110_;
}
v_resetjp_3110_:
{
lean_object* v___x_3113_; lean_object* v___x_3114_; lean_object* v___x_3115_; lean_object* v___x_3116_; lean_object* v___x_3117_; lean_object* v___x_3118_; lean_object* v___x_3119_; lean_object* v___x_3121_; 
v___x_3113_ = lean_array_get_borrowed(v___x_3095_, v_ys_3072_, v_i_3075_);
lean_inc_n(v___x_3113_, 3);
v___x_3114_ = lean_array_push(v___x_3098_, v___x_3113_);
v___x_3115_ = lean_array_push(v___x_3114_, v_fst_3104_);
v___x_3116_ = lean_array_push(v___x_3099_, v___x_3113_);
v___x_3117_ = lean_array_push(v___x_3116_, v_fst_3108_);
v___x_3118_ = lean_array_push(v___x_3100_, v___x_3113_);
v___x_3119_ = lean_array_push(v___x_3118_, v_snd_3109_);
if (v_isShared_3112_ == 0)
{
lean_ctor_set(v___x_3111_, 1, v___x_3119_);
lean_ctor_set(v___x_3111_, 0, v___x_3117_);
v___x_3121_ = v___x_3111_;
goto v_reusejp_3120_;
}
else
{
lean_object* v_reuseFailAlloc_3125_; 
v_reuseFailAlloc_3125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3125_, 0, v___x_3117_);
lean_ctor_set(v_reuseFailAlloc_3125_, 1, v___x_3119_);
v___x_3121_ = v_reuseFailAlloc_3125_;
goto v_reusejp_3120_;
}
v_reusejp_3120_:
{
lean_object* v___x_3123_; 
if (v_isShared_3107_ == 0)
{
lean_ctor_set(v___x_3106_, 1, v___x_3121_);
lean_ctor_set(v___x_3106_, 0, v___x_3115_);
v___x_3123_ = v___x_3106_;
goto v_reusejp_3122_;
}
else
{
lean_object* v_reuseFailAlloc_3124_; 
v_reuseFailAlloc_3124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3124_, 0, v___x_3115_);
lean_ctor_set(v_reuseFailAlloc_3124_, 1, v___x_3121_);
v___x_3123_ = v_reuseFailAlloc_3124_;
goto v_reusejp_3122_;
}
v_reusejp_3122_:
{
v_a_3080_ = v___x_3123_;
goto v___jp_3079_;
}
}
}
}
}
else
{
lean_object* v___x_3129_; 
if (v_isShared_3094_ == 0)
{
lean_ctor_set(v___x_3093_, 1, v___x_3100_);
lean_ctor_set(v___x_3093_, 0, v___x_3099_);
v___x_3129_ = v___x_3093_;
goto v_reusejp_3128_;
}
else
{
lean_object* v_reuseFailAlloc_3133_; 
v_reuseFailAlloc_3133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3133_, 0, v___x_3099_);
lean_ctor_set(v_reuseFailAlloc_3133_, 1, v___x_3100_);
v___x_3129_ = v_reuseFailAlloc_3133_;
goto v_reusejp_3128_;
}
v_reusejp_3128_:
{
lean_object* v___x_3131_; 
if (v_isShared_3089_ == 0)
{
lean_ctor_set(v___x_3088_, 1, v___x_3129_);
lean_ctor_set(v___x_3088_, 0, v___x_3098_);
v___x_3131_ = v___x_3088_;
goto v_reusejp_3130_;
}
else
{
lean_object* v_reuseFailAlloc_3132_; 
v_reuseFailAlloc_3132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3132_, 0, v___x_3098_);
lean_ctor_set(v_reuseFailAlloc_3132_, 1, v___x_3129_);
v___x_3131_ = v_reuseFailAlloc_3132_;
goto v_reusejp_3130_;
}
v_reusejp_3130_:
{
v_a_3080_ = v___x_3131_;
goto v___jp_3079_;
}
}
}
}
}
}
v___jp_3079_:
{
lean_object* v___x_3081_; 
v___x_3081_ = lean_nat_add(v_i_3075_, v_step_3078_);
lean_dec(v_i_3075_);
v_b_3074_ = v_a_3080_;
v_i_3075_ = v___x_3081_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___redArg___boxed(lean_object* v_xs_3136_, lean_object* v_eqs_3137_, lean_object* v_ys_3138_, lean_object* v_range_3139_, lean_object* v_b_3140_, lean_object* v_i_3141_, lean_object* v___y_3142_){
_start:
{
lean_object* v_res_3143_; 
v_res_3143_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___redArg(v_xs_3136_, v_eqs_3137_, v_ys_3138_, v_range_3139_, v_b_3140_, v_i_3141_);
lean_dec_ref(v_range_3139_);
lean_dec_ref(v_ys_3138_);
lean_dec_ref(v_eqs_3137_);
lean_dec_ref(v_xs_3136_);
return v_res_3143_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__1(void){
_start:
{
lean_object* v___x_3145_; lean_object* v___x_3146_; 
v___x_3145_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__0));
v___x_3146_ = l_Lean_stringToMessageData(v___x_3145_);
return v___x_3146_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__3(void){
_start:
{
lean_object* v___x_3148_; lean_object* v___x_3149_; 
v___x_3148_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__2));
v___x_3149_ = l_Lean_stringToMessageData(v___x_3148_);
return v___x_3149_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__5(void){
_start:
{
lean_object* v___x_3151_; lean_object* v___x_3152_; 
v___x_3151_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__4));
v___x_3152_ = l_Lean_stringToMessageData(v___x_3151_);
return v___x_3152_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__7(void){
_start:
{
lean_object* v___x_3154_; lean_object* v___x_3155_; 
v___x_3154_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__6));
v___x_3155_ = l_Lean_stringToMessageData(v___x_3154_);
return v___x_3155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1(uint8_t v___x_3156_, lean_object* v_cls_3157_, lean_object* v_xs_3158_, lean_object* v_lctx_3159_, lean_object* v_ys_3160_, lean_object* v___f_3161_, uint8_t v_fixedFun_3162_, lean_object* v___x_3163_, lean_object* v_ef_3164_, lean_object* v___y_3165_, uint8_t v_forceHEq_3166_, lean_object* v_ee_3167_, lean_object* v_kinds_3168_, lean_object* v_eqs_3169_, lean_object* v___y_3170_, lean_object* v___y_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_){
_start:
{
lean_object* v___y_3176_; lean_object* v___y_3177_; lean_object* v___y_3178_; lean_object* v___y_3182_; lean_object* v___y_3183_; uint8_t v___y_3184_; lean_object* v___y_3185_; lean_object* v___y_3186_; lean_object* v___y_3187_; lean_object* v___y_3188_; uint8_t v___y_3189_; lean_object* v___y_3190_; lean_object* v___y_3191_; lean_object* v___y_3192_; lean_object* v___y_3193_; lean_object* v___y_3194_; lean_object* v___y_3257_; lean_object* v___y_3258_; uint8_t v___y_3259_; lean_object* v___y_3260_; lean_object* v___y_3261_; lean_object* v___y_3262_; lean_object* v___y_3263_; lean_object* v___y_3264_; lean_object* v___y_3265_; lean_object* v___y_3266_; lean_object* v___y_3267_; uint8_t v___y_3268_; lean_object* v___y_3269_; uint8_t v___y_3270_; lean_object* v___y_3328_; uint8_t v___y_3329_; lean_object* v___y_3330_; lean_object* v___y_3331_; lean_object* v___y_3332_; lean_object* v___y_3333_; uint8_t v___y_3334_; lean_object* v___y_3335_; lean_object* v___y_3355_; lean_object* v___y_3356_; lean_object* v___y_3357_; lean_object* v___y_3358_; lean_object* v___y_3359_; lean_object* v___y_3360_; lean_object* v___y_3361_; lean_object* v___y_3407_; 
if (v_fixedFun_3162_ == 0)
{
lean_object* v___x_3422_; lean_object* v___x_3423_; lean_object* v___x_3424_; lean_object* v___x_3425_; lean_object* v___x_3426_; 
v___x_3422_ = lean_unsigned_to_nat(3u);
v___x_3423_ = lean_mk_empty_array_with_capacity(v___x_3422_);
lean_inc_ref(v_ef_3164_);
v___x_3424_ = lean_array_push(v___x_3423_, v_ef_3164_);
lean_inc_ref(v___y_3165_);
v___x_3425_ = lean_array_push(v___x_3424_, v___y_3165_);
v___x_3426_ = lean_array_push(v___x_3425_, v_ee_3167_);
v___y_3407_ = v___x_3426_;
goto v___jp_3406_;
}
else
{
lean_object* v___x_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; 
lean_dec_ref(v_ee_3167_);
v___x_3427_ = lean_unsigned_to_nat(1u);
v___x_3428_ = lean_mk_empty_array_with_capacity(v___x_3427_);
lean_inc_ref(v_ef_3164_);
v___x_3429_ = lean_array_push(v___x_3428_, v_ef_3164_);
v___y_3407_ = v___x_3429_;
goto v___jp_3406_;
}
v___jp_3175_:
{
lean_object* v___x_3179_; lean_object* v___x_3180_; 
v___x_3179_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3179_, 0, v___y_3176_);
lean_ctor_set(v___x_3179_, 1, v___y_3177_);
lean_ctor_set(v___x_3179_, 2, v___y_3178_);
v___x_3180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3180_, 0, v___x_3179_);
return v___x_3180_;
}
v___jp_3181_:
{
lean_object* v___x_3195_; lean_object* v___x_3196_; 
v___x_3195_ = l_Lean_Expr_beta(v___y_3186_, v___y_3183_);
v___x_3196_ = l_Lean_Meta_mkLambdaFVars(v___y_3187_, v___x_3195_, v___y_3184_, v___x_3156_, v___y_3184_, v___x_3156_, v___y_3189_, v___y_3191_, v___y_3192_, v___y_3193_, v___y_3194_);
lean_dec(v___y_3187_);
if (lean_obj_tag(v___x_3196_) == 0)
{
lean_object* v_a_3197_; lean_object* v___x_3198_; lean_object* v___x_3199_; 
v_a_3197_ = lean_ctor_get(v___x_3196_, 0);
lean_inc(v_a_3197_);
lean_dec_ref_known(v___x_3196_, 1);
v___x_3198_ = l_Lean_mkAppN(v_a_3197_, v___y_3182_);
lean_dec(v___y_3182_);
v___x_3199_ = l_Lean_Meta_mkLambdaFVars(v___y_3185_, v___x_3198_, v___x_3156_, v___x_3156_, v___y_3184_, v___x_3156_, v___y_3189_, v___y_3191_, v___y_3192_, v___y_3193_, v___y_3194_);
lean_dec(v___y_3185_);
if (lean_obj_tag(v___x_3199_) == 0)
{
lean_object* v_a_3200_; lean_object* v___x_3201_; 
v_a_3200_ = lean_ctor_get(v___x_3199_, 0);
lean_inc(v_a_3200_);
lean_dec_ref_known(v___x_3199_, 1);
v___x_3201_ = l_Lean_Meta_mkLambdaFVars(v___y_3188_, v_a_3200_, v___y_3184_, v___x_3156_, v___y_3184_, v___x_3156_, v___y_3189_, v___y_3191_, v___y_3192_, v___y_3193_, v___y_3194_);
lean_dec_ref(v___y_3188_);
if (lean_obj_tag(v___x_3201_) == 0)
{
lean_object* v_a_3202_; lean_object* v___x_3203_; 
v_a_3202_ = lean_ctor_get(v___x_3201_, 0);
lean_inc_n(v_a_3202_, 2);
lean_dec_ref_known(v___x_3201_, 1);
lean_inc(v___y_3194_);
lean_inc_ref(v___y_3193_);
lean_inc(v___y_3192_);
lean_inc_ref(v___y_3191_);
v___x_3203_ = lean_infer_type(v_a_3202_, v___y_3191_, v___y_3192_, v___y_3193_, v___y_3194_);
if (lean_obj_tag(v___x_3203_) == 0)
{
lean_object* v_options_3204_; uint8_t v_hasTrace_3205_; 
v_options_3204_ = lean_ctor_get(v___y_3193_, 2);
v_hasTrace_3205_ = lean_ctor_get_uint8(v_options_3204_, sizeof(void*)*1);
if (v_hasTrace_3205_ == 0)
{
lean_object* v_a_3206_; 
lean_dec(v_cls_3157_);
v_a_3206_ = lean_ctor_get(v___x_3203_, 0);
lean_inc(v_a_3206_);
lean_dec_ref_known(v___x_3203_, 1);
v___y_3176_ = v_a_3206_;
v___y_3177_ = v_a_3202_;
v___y_3178_ = v___y_3190_;
goto v___jp_3175_;
}
else
{
lean_object* v_a_3207_; lean_object* v_inheritedTraceOptions_3208_; lean_object* v___x_3209_; lean_object* v___x_3210_; uint8_t v___x_3211_; 
v_a_3207_ = lean_ctor_get(v___x_3203_, 0);
lean_inc(v_a_3207_);
lean_dec_ref_known(v___x_3203_, 1);
v_inheritedTraceOptions_3208_ = lean_ctor_get(v___y_3193_, 13);
v___x_3209_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___lam__0___closed__1));
lean_inc(v_cls_3157_);
v___x_3210_ = l_Lean_Name_append(v___x_3209_, v_cls_3157_);
v___x_3211_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3208_, v_options_3204_, v___x_3210_);
lean_dec(v___x_3210_);
if (v___x_3211_ == 0)
{
lean_dec(v_cls_3157_);
v___y_3176_ = v_a_3207_;
v___y_3177_ = v_a_3202_;
v___y_3178_ = v___y_3190_;
goto v___jp_3175_;
}
else
{
lean_object* v___x_3212_; lean_object* v___x_3213_; lean_object* v___x_3214_; lean_object* v___x_3215_; 
v___x_3212_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__1, &lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__1_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__1);
lean_inc(v_a_3207_);
v___x_3213_ = l_Lean_MessageData_ofExpr(v_a_3207_);
v___x_3214_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3214_, 0, v___x_3212_);
lean_ctor_set(v___x_3214_, 1, v___x_3213_);
v___x_3215_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3157_, v___x_3214_, v___y_3191_, v___y_3192_, v___y_3193_, v___y_3194_);
if (lean_obj_tag(v___x_3215_) == 0)
{
lean_dec_ref_known(v___x_3215_, 1);
v___y_3176_ = v_a_3207_;
v___y_3177_ = v_a_3202_;
v___y_3178_ = v___y_3190_;
goto v___jp_3175_;
}
else
{
lean_object* v_a_3216_; lean_object* v___x_3218_; uint8_t v_isShared_3219_; uint8_t v_isSharedCheck_3223_; 
lean_dec(v_a_3207_);
lean_dec(v_a_3202_);
lean_dec(v___y_3190_);
v_a_3216_ = lean_ctor_get(v___x_3215_, 0);
v_isSharedCheck_3223_ = !lean_is_exclusive(v___x_3215_);
if (v_isSharedCheck_3223_ == 0)
{
v___x_3218_ = v___x_3215_;
v_isShared_3219_ = v_isSharedCheck_3223_;
goto v_resetjp_3217_;
}
else
{
lean_inc(v_a_3216_);
lean_dec(v___x_3215_);
v___x_3218_ = lean_box(0);
v_isShared_3219_ = v_isSharedCheck_3223_;
goto v_resetjp_3217_;
}
v_resetjp_3217_:
{
lean_object* v___x_3221_; 
if (v_isShared_3219_ == 0)
{
v___x_3221_ = v___x_3218_;
goto v_reusejp_3220_;
}
else
{
lean_object* v_reuseFailAlloc_3222_; 
v_reuseFailAlloc_3222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3222_, 0, v_a_3216_);
v___x_3221_ = v_reuseFailAlloc_3222_;
goto v_reusejp_3220_;
}
v_reusejp_3220_:
{
return v___x_3221_;
}
}
}
}
}
}
else
{
lean_object* v_a_3224_; lean_object* v___x_3226_; uint8_t v_isShared_3227_; uint8_t v_isSharedCheck_3231_; 
lean_dec(v_a_3202_);
lean_dec(v___y_3190_);
lean_dec(v_cls_3157_);
v_a_3224_ = lean_ctor_get(v___x_3203_, 0);
v_isSharedCheck_3231_ = !lean_is_exclusive(v___x_3203_);
if (v_isSharedCheck_3231_ == 0)
{
v___x_3226_ = v___x_3203_;
v_isShared_3227_ = v_isSharedCheck_3231_;
goto v_resetjp_3225_;
}
else
{
lean_inc(v_a_3224_);
lean_dec(v___x_3203_);
v___x_3226_ = lean_box(0);
v_isShared_3227_ = v_isSharedCheck_3231_;
goto v_resetjp_3225_;
}
v_resetjp_3225_:
{
lean_object* v___x_3229_; 
if (v_isShared_3227_ == 0)
{
v___x_3229_ = v___x_3226_;
goto v_reusejp_3228_;
}
else
{
lean_object* v_reuseFailAlloc_3230_; 
v_reuseFailAlloc_3230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3230_, 0, v_a_3224_);
v___x_3229_ = v_reuseFailAlloc_3230_;
goto v_reusejp_3228_;
}
v_reusejp_3228_:
{
return v___x_3229_;
}
}
}
}
else
{
lean_object* v_a_3232_; lean_object* v___x_3234_; uint8_t v_isShared_3235_; uint8_t v_isSharedCheck_3239_; 
lean_dec(v___y_3190_);
lean_dec(v_cls_3157_);
v_a_3232_ = lean_ctor_get(v___x_3201_, 0);
v_isSharedCheck_3239_ = !lean_is_exclusive(v___x_3201_);
if (v_isSharedCheck_3239_ == 0)
{
v___x_3234_ = v___x_3201_;
v_isShared_3235_ = v_isSharedCheck_3239_;
goto v_resetjp_3233_;
}
else
{
lean_inc(v_a_3232_);
lean_dec(v___x_3201_);
v___x_3234_ = lean_box(0);
v_isShared_3235_ = v_isSharedCheck_3239_;
goto v_resetjp_3233_;
}
v_resetjp_3233_:
{
lean_object* v___x_3237_; 
if (v_isShared_3235_ == 0)
{
v___x_3237_ = v___x_3234_;
goto v_reusejp_3236_;
}
else
{
lean_object* v_reuseFailAlloc_3238_; 
v_reuseFailAlloc_3238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3238_, 0, v_a_3232_);
v___x_3237_ = v_reuseFailAlloc_3238_;
goto v_reusejp_3236_;
}
v_reusejp_3236_:
{
return v___x_3237_;
}
}
}
}
else
{
lean_object* v_a_3240_; lean_object* v___x_3242_; uint8_t v_isShared_3243_; uint8_t v_isSharedCheck_3247_; 
lean_dec(v___y_3190_);
lean_dec_ref(v___y_3188_);
lean_dec(v_cls_3157_);
v_a_3240_ = lean_ctor_get(v___x_3199_, 0);
v_isSharedCheck_3247_ = !lean_is_exclusive(v___x_3199_);
if (v_isSharedCheck_3247_ == 0)
{
v___x_3242_ = v___x_3199_;
v_isShared_3243_ = v_isSharedCheck_3247_;
goto v_resetjp_3241_;
}
else
{
lean_inc(v_a_3240_);
lean_dec(v___x_3199_);
v___x_3242_ = lean_box(0);
v_isShared_3243_ = v_isSharedCheck_3247_;
goto v_resetjp_3241_;
}
v_resetjp_3241_:
{
lean_object* v___x_3245_; 
if (v_isShared_3243_ == 0)
{
v___x_3245_ = v___x_3242_;
goto v_reusejp_3244_;
}
else
{
lean_object* v_reuseFailAlloc_3246_; 
v_reuseFailAlloc_3246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3246_, 0, v_a_3240_);
v___x_3245_ = v_reuseFailAlloc_3246_;
goto v_reusejp_3244_;
}
v_reusejp_3244_:
{
return v___x_3245_;
}
}
}
}
else
{
lean_object* v_a_3248_; lean_object* v___x_3250_; uint8_t v_isShared_3251_; uint8_t v_isSharedCheck_3255_; 
lean_dec(v___y_3190_);
lean_dec_ref(v___y_3188_);
lean_dec(v___y_3185_);
lean_dec(v___y_3182_);
lean_dec(v_cls_3157_);
v_a_3248_ = lean_ctor_get(v___x_3196_, 0);
v_isSharedCheck_3255_ = !lean_is_exclusive(v___x_3196_);
if (v_isSharedCheck_3255_ == 0)
{
v___x_3250_ = v___x_3196_;
v_isShared_3251_ = v_isSharedCheck_3255_;
goto v_resetjp_3249_;
}
else
{
lean_inc(v_a_3248_);
lean_dec(v___x_3196_);
v___x_3250_ = lean_box(0);
v_isShared_3251_ = v_isSharedCheck_3255_;
goto v_resetjp_3249_;
}
v_resetjp_3249_:
{
lean_object* v___x_3253_; 
if (v_isShared_3251_ == 0)
{
v___x_3253_ = v___x_3250_;
goto v_reusejp_3252_;
}
else
{
lean_object* v_reuseFailAlloc_3254_; 
v_reuseFailAlloc_3254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3254_, 0, v_a_3248_);
v___x_3253_ = v_reuseFailAlloc_3254_;
goto v_reusejp_3252_;
}
v_reusejp_3252_:
{
return v___x_3253_;
}
}
}
}
v___jp_3256_:
{
lean_object* v___x_3271_; lean_object* v___x_3272_; lean_object* v___x_3273_; lean_object* v___x_3274_; lean_object* v___x_3275_; lean_object* v___x_3276_; lean_object* v___x_3277_; 
v___x_3271_ = lean_mk_empty_array_with_capacity(v___y_3269_);
v___x_3272_ = lean_box(v___y_3270_);
v___x_3273_ = lean_array_push(v___x_3271_, v___x_3272_);
lean_inc_ref_n(v___y_3263_, 2);
v___x_3274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3274_, 0, v___y_3263_);
lean_ctor_set(v___x_3274_, 1, v___x_3273_);
v___x_3275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3275_, 0, v___y_3263_);
lean_ctor_set(v___x_3275_, 1, v___x_3274_);
v___x_3276_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3276_, 0, v___y_3263_);
lean_ctor_set(v___x_3276_, 1, v___x_3275_);
v___x_3277_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___redArg(v_xs_3158_, v_eqs_3169_, v_lctx_3159_, v_ys_3160_, v_kinds_3168_, v___y_3260_, v___x_3276_, v___y_3266_, v___y_3261_, v___y_3267_, v___y_3258_, v___y_3264_);
lean_dec_ref(v___y_3260_);
if (lean_obj_tag(v___x_3277_) == 0)
{
lean_object* v_a_3278_; lean_object* v___x_3279_; 
v_a_3278_ = lean_ctor_get(v___x_3277_, 0);
lean_inc(v_a_3278_);
lean_dec_ref_known(v___x_3277_, 1);
lean_inc(v___y_3264_);
lean_inc_ref(v___y_3258_);
lean_inc(v___y_3267_);
lean_inc_ref(v___y_3261_);
v___x_3279_ = lean_apply_5(v___f_3161_, v___y_3261_, v___y_3267_, v___y_3258_, v___y_3264_, lean_box(0));
if (lean_obj_tag(v___x_3279_) == 0)
{
lean_object* v_snd_3280_; lean_object* v_snd_3281_; lean_object* v_a_3282_; uint8_t v___x_3283_; 
v_snd_3280_ = lean_ctor_get(v_a_3278_, 1);
lean_inc(v_snd_3280_);
v_snd_3281_ = lean_ctor_get(v_snd_3280_, 1);
lean_inc(v_snd_3281_);
v_a_3282_ = lean_ctor_get(v___x_3279_, 0);
lean_inc(v_a_3282_);
lean_dec_ref_known(v___x_3279_, 1);
v___x_3283_ = lean_unbox(v_a_3282_);
lean_dec(v_a_3282_);
if (v___x_3283_ == 0)
{
lean_object* v_fst_3284_; lean_object* v_fst_3285_; lean_object* v_fst_3286_; lean_object* v_snd_3287_; 
v_fst_3284_ = lean_ctor_get(v_a_3278_, 0);
lean_inc(v_fst_3284_);
lean_dec(v_a_3278_);
v_fst_3285_ = lean_ctor_get(v_snd_3280_, 0);
lean_inc(v_fst_3285_);
lean_dec(v_snd_3280_);
v_fst_3286_ = lean_ctor_get(v_snd_3281_, 0);
lean_inc(v_fst_3286_);
v_snd_3287_ = lean_ctor_get(v_snd_3281_, 1);
lean_inc(v_snd_3287_);
lean_dec(v_snd_3281_);
v___y_3182_ = v_fst_3286_;
v___y_3183_ = v___y_3257_;
v___y_3184_ = v___y_3259_;
v___y_3185_ = v_fst_3284_;
v___y_3186_ = v___y_3262_;
v___y_3187_ = v_fst_3285_;
v___y_3188_ = v___y_3265_;
v___y_3189_ = v___y_3268_;
v___y_3190_ = v_snd_3287_;
v___y_3191_ = v___y_3261_;
v___y_3192_ = v___y_3267_;
v___y_3193_ = v___y_3258_;
v___y_3194_ = v___y_3264_;
goto v___jp_3181_;
}
else
{
lean_object* v_fst_3288_; lean_object* v_fst_3289_; lean_object* v_fst_3290_; lean_object* v_snd_3291_; lean_object* v___x_3293_; uint8_t v_isShared_3294_; uint8_t v_isSharedCheck_3310_; 
v_fst_3288_ = lean_ctor_get(v_a_3278_, 0);
lean_inc(v_fst_3288_);
lean_dec(v_a_3278_);
v_fst_3289_ = lean_ctor_get(v_snd_3280_, 0);
lean_inc(v_fst_3289_);
lean_dec(v_snd_3280_);
v_fst_3290_ = lean_ctor_get(v_snd_3281_, 0);
v_snd_3291_ = lean_ctor_get(v_snd_3281_, 1);
v_isSharedCheck_3310_ = !lean_is_exclusive(v_snd_3281_);
if (v_isSharedCheck_3310_ == 0)
{
v___x_3293_ = v_snd_3281_;
v_isShared_3294_ = v_isSharedCheck_3310_;
goto v_resetjp_3292_;
}
else
{
lean_inc(v_snd_3291_);
lean_inc(v_fst_3290_);
lean_dec(v_snd_3281_);
v___x_3293_ = lean_box(0);
v_isShared_3294_ = v_isSharedCheck_3310_;
goto v_resetjp_3292_;
}
v_resetjp_3292_:
{
lean_object* v___x_3295_; lean_object* v___x_3296_; lean_object* v___x_3297_; lean_object* v___x_3299_; 
v___x_3295_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__3, &lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__3_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__3);
lean_inc(v_snd_3291_);
v___x_3296_ = lp_mathlib_Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4(v_snd_3291_);
v___x_3297_ = l_Lean_MessageData_ofFormat(v___x_3296_);
if (v_isShared_3294_ == 0)
{
lean_ctor_set_tag(v___x_3293_, 7);
lean_ctor_set(v___x_3293_, 1, v___x_3297_);
lean_ctor_set(v___x_3293_, 0, v___x_3295_);
v___x_3299_ = v___x_3293_;
goto v_reusejp_3298_;
}
else
{
lean_object* v_reuseFailAlloc_3309_; 
v_reuseFailAlloc_3309_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3309_, 0, v___x_3295_);
lean_ctor_set(v_reuseFailAlloc_3309_, 1, v___x_3297_);
v___x_3299_ = v_reuseFailAlloc_3309_;
goto v_reusejp_3298_;
}
v_reusejp_3298_:
{
lean_object* v___x_3300_; 
lean_inc(v_cls_3157_);
v___x_3300_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3157_, v___x_3299_, v___y_3261_, v___y_3267_, v___y_3258_, v___y_3264_);
if (lean_obj_tag(v___x_3300_) == 0)
{
lean_dec_ref_known(v___x_3300_, 1);
v___y_3182_ = v_fst_3290_;
v___y_3183_ = v___y_3257_;
v___y_3184_ = v___y_3259_;
v___y_3185_ = v_fst_3288_;
v___y_3186_ = v___y_3262_;
v___y_3187_ = v_fst_3289_;
v___y_3188_ = v___y_3265_;
v___y_3189_ = v___y_3268_;
v___y_3190_ = v_snd_3291_;
v___y_3191_ = v___y_3261_;
v___y_3192_ = v___y_3267_;
v___y_3193_ = v___y_3258_;
v___y_3194_ = v___y_3264_;
goto v___jp_3181_;
}
else
{
lean_object* v_a_3301_; lean_object* v___x_3303_; uint8_t v_isShared_3304_; uint8_t v_isSharedCheck_3308_; 
lean_dec(v_snd_3291_);
lean_dec(v_fst_3290_);
lean_dec(v_fst_3289_);
lean_dec(v_fst_3288_);
lean_dec_ref(v___y_3265_);
lean_dec_ref(v___y_3262_);
lean_dec(v___y_3257_);
lean_dec(v_cls_3157_);
v_a_3301_ = lean_ctor_get(v___x_3300_, 0);
v_isSharedCheck_3308_ = !lean_is_exclusive(v___x_3300_);
if (v_isSharedCheck_3308_ == 0)
{
v___x_3303_ = v___x_3300_;
v_isShared_3304_ = v_isSharedCheck_3308_;
goto v_resetjp_3302_;
}
else
{
lean_inc(v_a_3301_);
lean_dec(v___x_3300_);
v___x_3303_ = lean_box(0);
v_isShared_3304_ = v_isSharedCheck_3308_;
goto v_resetjp_3302_;
}
v_resetjp_3302_:
{
lean_object* v___x_3306_; 
if (v_isShared_3304_ == 0)
{
v___x_3306_ = v___x_3303_;
goto v_reusejp_3305_;
}
else
{
lean_object* v_reuseFailAlloc_3307_; 
v_reuseFailAlloc_3307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3307_, 0, v_a_3301_);
v___x_3306_ = v_reuseFailAlloc_3307_;
goto v_reusejp_3305_;
}
v_reusejp_3305_:
{
return v___x_3306_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3311_; lean_object* v___x_3313_; uint8_t v_isShared_3314_; uint8_t v_isSharedCheck_3318_; 
lean_dec(v_a_3278_);
lean_dec_ref(v___y_3265_);
lean_dec_ref(v___y_3262_);
lean_dec(v___y_3257_);
lean_dec(v_cls_3157_);
v_a_3311_ = lean_ctor_get(v___x_3279_, 0);
v_isSharedCheck_3318_ = !lean_is_exclusive(v___x_3279_);
if (v_isSharedCheck_3318_ == 0)
{
v___x_3313_ = v___x_3279_;
v_isShared_3314_ = v_isSharedCheck_3318_;
goto v_resetjp_3312_;
}
else
{
lean_inc(v_a_3311_);
lean_dec(v___x_3279_);
v___x_3313_ = lean_box(0);
v_isShared_3314_ = v_isSharedCheck_3318_;
goto v_resetjp_3312_;
}
v_resetjp_3312_:
{
lean_object* v___x_3316_; 
if (v_isShared_3314_ == 0)
{
v___x_3316_ = v___x_3313_;
goto v_reusejp_3315_;
}
else
{
lean_object* v_reuseFailAlloc_3317_; 
v_reuseFailAlloc_3317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3317_, 0, v_a_3311_);
v___x_3316_ = v_reuseFailAlloc_3317_;
goto v_reusejp_3315_;
}
v_reusejp_3315_:
{
return v___x_3316_;
}
}
}
}
else
{
lean_object* v_a_3319_; lean_object* v___x_3321_; uint8_t v_isShared_3322_; uint8_t v_isSharedCheck_3326_; 
lean_dec_ref(v___y_3265_);
lean_dec_ref(v___y_3262_);
lean_dec(v___y_3257_);
lean_dec_ref(v___f_3161_);
lean_dec(v_cls_3157_);
v_a_3319_ = lean_ctor_get(v___x_3277_, 0);
v_isSharedCheck_3326_ = !lean_is_exclusive(v___x_3277_);
if (v_isSharedCheck_3326_ == 0)
{
v___x_3321_ = v___x_3277_;
v_isShared_3322_ = v_isSharedCheck_3326_;
goto v_resetjp_3320_;
}
else
{
lean_inc(v_a_3319_);
lean_dec(v___x_3277_);
v___x_3321_ = lean_box(0);
v_isShared_3322_ = v_isSharedCheck_3326_;
goto v_resetjp_3320_;
}
v_resetjp_3320_:
{
lean_object* v___x_3324_; 
if (v_isShared_3322_ == 0)
{
v___x_3324_ = v___x_3321_;
goto v_reusejp_3323_;
}
else
{
lean_object* v_reuseFailAlloc_3325_; 
v_reuseFailAlloc_3325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3325_, 0, v_a_3319_);
v___x_3324_ = v_reuseFailAlloc_3325_;
goto v_reusejp_3323_;
}
v_reusejp_3323_:
{
return v___x_3324_;
}
}
}
}
v___jp_3327_:
{
lean_object* v_localInstances_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; 
v_localInstances_3336_ = lean_ctor_get(v___y_3170_, 3);
v___x_3337_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve___boxed), 6, 1);
lean_closure_set(v___x_3337_, 0, v___y_3331_);
lean_inc_ref(v_localInstances_3336_);
lean_inc_ref(v_lctx_3159_);
v___x_3338_ = lp_mathlib_Lean_Meta_withLCtx___at___00Lean_Meta_mkRichHCongr_spec__1___redArg(v_lctx_3159_, v_localInstances_3336_, v___x_3337_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
if (lean_obj_tag(v___x_3338_) == 0)
{
lean_object* v_a_3339_; 
v_a_3339_ = lean_ctor_get(v___x_3338_, 0);
lean_inc(v_a_3339_);
lean_dec_ref_known(v___x_3338_, 1);
if (lean_obj_tag(v_a_3339_) == 1)
{
lean_object* v_val_3340_; lean_object* v___x_3341_; 
v_val_3340_ = lean_ctor_get(v_a_3339_, 0);
lean_inc(v_val_3340_);
lean_dec_ref_known(v_a_3339_, 1);
v___x_3341_ = lean_mk_empty_array_with_capacity(v___y_3333_);
if (v_fixedFun_3162_ == 0)
{
uint8_t v___x_3342_; 
v___x_3342_ = 2;
v___y_3257_ = v___y_3328_;
v___y_3258_ = v___y_3172_;
v___y_3259_ = v___y_3329_;
v___y_3260_ = v___y_3330_;
v___y_3261_ = v___y_3170_;
v___y_3262_ = v_val_3340_;
v___y_3263_ = v___x_3341_;
v___y_3264_ = v___y_3173_;
v___y_3265_ = v___y_3332_;
v___y_3266_ = v___y_3333_;
v___y_3267_ = v___y_3171_;
v___y_3268_ = v___y_3334_;
v___y_3269_ = v___y_3335_;
v___y_3270_ = v___x_3342_;
goto v___jp_3256_;
}
else
{
uint8_t v___x_3343_; 
v___x_3343_ = 0;
v___y_3257_ = v___y_3328_;
v___y_3258_ = v___y_3172_;
v___y_3259_ = v___y_3329_;
v___y_3260_ = v___y_3330_;
v___y_3261_ = v___y_3170_;
v___y_3262_ = v_val_3340_;
v___y_3263_ = v___x_3341_;
v___y_3264_ = v___y_3173_;
v___y_3265_ = v___y_3332_;
v___y_3266_ = v___y_3333_;
v___y_3267_ = v___y_3171_;
v___y_3268_ = v___y_3334_;
v___y_3269_ = v___y_3335_;
v___y_3270_ = v___x_3343_;
goto v___jp_3256_;
}
}
else
{
lean_object* v___x_3344_; lean_object* v___x_3345_; 
lean_dec(v_a_3339_);
lean_dec(v___y_3333_);
lean_dec_ref(v___y_3332_);
lean_dec_ref(v___y_3330_);
lean_dec(v___y_3328_);
lean_dec_ref(v___f_3161_);
lean_dec_ref(v_lctx_3159_);
lean_dec(v_cls_3157_);
v___x_3344_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__5, &lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__5_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__5);
v___x_3345_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkHCongrWithArity_x27_prove_spec__0___redArg(v___x_3344_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
return v___x_3345_;
}
}
else
{
lean_object* v_a_3346_; lean_object* v___x_3348_; uint8_t v_isShared_3349_; uint8_t v_isSharedCheck_3353_; 
lean_dec(v___y_3333_);
lean_dec_ref(v___y_3332_);
lean_dec_ref(v___y_3330_);
lean_dec(v___y_3328_);
lean_dec_ref(v___f_3161_);
lean_dec_ref(v_lctx_3159_);
lean_dec(v_cls_3157_);
v_a_3346_ = lean_ctor_get(v___x_3338_, 0);
v_isSharedCheck_3353_ = !lean_is_exclusive(v___x_3338_);
if (v_isSharedCheck_3353_ == 0)
{
v___x_3348_ = v___x_3338_;
v_isShared_3349_ = v_isSharedCheck_3353_;
goto v_resetjp_3347_;
}
else
{
lean_inc(v_a_3346_);
lean_dec(v___x_3338_);
v___x_3348_ = lean_box(0);
v_isShared_3349_ = v_isSharedCheck_3353_;
goto v_resetjp_3347_;
}
v_resetjp_3347_:
{
lean_object* v___x_3351_; 
if (v_isShared_3349_ == 0)
{
v___x_3351_ = v___x_3348_;
goto v_reusejp_3350_;
}
else
{
lean_object* v_reuseFailAlloc_3352_; 
v_reuseFailAlloc_3352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3352_, 0, v_a_3346_);
v___x_3351_ = v_reuseFailAlloc_3352_;
goto v_reusejp_3350_;
}
v_reusejp_3350_:
{
return v___x_3351_;
}
}
}
}
v___jp_3354_:
{
if (lean_obj_tag(v___y_3361_) == 0)
{
lean_object* v_a_3362_; uint8_t v___x_3363_; uint8_t v___x_3364_; lean_object* v___x_3365_; 
v_a_3362_ = lean_ctor_get(v___y_3361_, 0);
lean_inc(v_a_3362_);
lean_dec_ref_known(v___y_3361_, 1);
v___x_3363_ = 0;
v___x_3364_ = 1;
v___x_3365_ = l_Lean_Meta_mkForallFVars(v___y_3357_, v_a_3362_, v___x_3363_, v___x_3156_, v___x_3156_, v___x_3364_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
lean_dec(v___y_3357_);
if (lean_obj_tag(v___x_3365_) == 0)
{
lean_object* v_a_3366_; lean_object* v___x_3367_; 
v_a_3366_ = lean_ctor_get(v___x_3365_, 0);
lean_inc(v_a_3366_);
lean_dec_ref_known(v___x_3365_, 1);
lean_inc_ref(v___f_3161_);
lean_inc(v___y_3173_);
lean_inc_ref(v___y_3172_);
lean_inc(v___y_3171_);
lean_inc_ref(v___y_3170_);
v___x_3367_ = lean_apply_5(v___f_3161_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_, lean_box(0));
if (lean_obj_tag(v___x_3367_) == 0)
{
lean_object* v_a_3368_; uint8_t v___x_3369_; 
v_a_3368_ = lean_ctor_get(v___x_3367_, 0);
lean_inc(v_a_3368_);
lean_dec_ref_known(v___x_3367_, 1);
v___x_3369_ = lean_unbox(v_a_3368_);
lean_dec(v_a_3368_);
if (v___x_3369_ == 0)
{
v___y_3328_ = v___y_3355_;
v___y_3329_ = v___x_3363_;
v___y_3330_ = v___y_3356_;
v___y_3331_ = v_a_3366_;
v___y_3332_ = v___y_3358_;
v___y_3333_ = v___y_3359_;
v___y_3334_ = v___x_3364_;
v___y_3335_ = v___y_3360_;
goto v___jp_3327_;
}
else
{
lean_object* v___x_3370_; lean_object* v___x_3371_; lean_object* v___x_3372_; lean_object* v___x_3373_; 
v___x_3370_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__7, &lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__7_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___closed__7);
lean_inc(v_a_3366_);
v___x_3371_ = l_Lean_MessageData_ofExpr(v_a_3366_);
v___x_3372_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3372_, 0, v___x_3370_);
lean_ctor_set(v___x_3372_, 1, v___x_3371_);
lean_inc(v_cls_3157_);
v___x_3373_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3157_, v___x_3372_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
if (lean_obj_tag(v___x_3373_) == 0)
{
lean_dec_ref_known(v___x_3373_, 1);
v___y_3328_ = v___y_3355_;
v___y_3329_ = v___x_3363_;
v___y_3330_ = v___y_3356_;
v___y_3331_ = v_a_3366_;
v___y_3332_ = v___y_3358_;
v___y_3333_ = v___y_3359_;
v___y_3334_ = v___x_3364_;
v___y_3335_ = v___y_3360_;
goto v___jp_3327_;
}
else
{
lean_object* v_a_3374_; lean_object* v___x_3376_; uint8_t v_isShared_3377_; uint8_t v_isSharedCheck_3381_; 
lean_dec(v_a_3366_);
lean_dec(v___y_3359_);
lean_dec_ref(v___y_3358_);
lean_dec_ref(v___y_3356_);
lean_dec(v___y_3355_);
lean_dec_ref(v___f_3161_);
lean_dec_ref(v_lctx_3159_);
lean_dec(v_cls_3157_);
v_a_3374_ = lean_ctor_get(v___x_3373_, 0);
v_isSharedCheck_3381_ = !lean_is_exclusive(v___x_3373_);
if (v_isSharedCheck_3381_ == 0)
{
v___x_3376_ = v___x_3373_;
v_isShared_3377_ = v_isSharedCheck_3381_;
goto v_resetjp_3375_;
}
else
{
lean_inc(v_a_3374_);
lean_dec(v___x_3373_);
v___x_3376_ = lean_box(0);
v_isShared_3377_ = v_isSharedCheck_3381_;
goto v_resetjp_3375_;
}
v_resetjp_3375_:
{
lean_object* v___x_3379_; 
if (v_isShared_3377_ == 0)
{
v___x_3379_ = v___x_3376_;
goto v_reusejp_3378_;
}
else
{
lean_object* v_reuseFailAlloc_3380_; 
v_reuseFailAlloc_3380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3380_, 0, v_a_3374_);
v___x_3379_ = v_reuseFailAlloc_3380_;
goto v_reusejp_3378_;
}
v_reusejp_3378_:
{
return v___x_3379_;
}
}
}
}
}
else
{
lean_object* v_a_3382_; lean_object* v___x_3384_; uint8_t v_isShared_3385_; uint8_t v_isSharedCheck_3389_; 
lean_dec(v_a_3366_);
lean_dec(v___y_3359_);
lean_dec_ref(v___y_3358_);
lean_dec_ref(v___y_3356_);
lean_dec(v___y_3355_);
lean_dec_ref(v___f_3161_);
lean_dec_ref(v_lctx_3159_);
lean_dec(v_cls_3157_);
v_a_3382_ = lean_ctor_get(v___x_3367_, 0);
v_isSharedCheck_3389_ = !lean_is_exclusive(v___x_3367_);
if (v_isSharedCheck_3389_ == 0)
{
v___x_3384_ = v___x_3367_;
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
else
{
lean_inc(v_a_3382_);
lean_dec(v___x_3367_);
v___x_3384_ = lean_box(0);
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
v_resetjp_3383_:
{
lean_object* v___x_3387_; 
if (v_isShared_3385_ == 0)
{
v___x_3387_ = v___x_3384_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3388_; 
v_reuseFailAlloc_3388_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3388_, 0, v_a_3382_);
v___x_3387_ = v_reuseFailAlloc_3388_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
return v___x_3387_;
}
}
}
}
else
{
lean_object* v_a_3390_; lean_object* v___x_3392_; uint8_t v_isShared_3393_; uint8_t v_isSharedCheck_3397_; 
lean_dec(v___y_3359_);
lean_dec_ref(v___y_3358_);
lean_dec_ref(v___y_3356_);
lean_dec(v___y_3355_);
lean_dec_ref(v___f_3161_);
lean_dec_ref(v_lctx_3159_);
lean_dec(v_cls_3157_);
v_a_3390_ = lean_ctor_get(v___x_3365_, 0);
v_isSharedCheck_3397_ = !lean_is_exclusive(v___x_3365_);
if (v_isSharedCheck_3397_ == 0)
{
v___x_3392_ = v___x_3365_;
v_isShared_3393_ = v_isSharedCheck_3397_;
goto v_resetjp_3391_;
}
else
{
lean_inc(v_a_3390_);
lean_dec(v___x_3365_);
v___x_3392_ = lean_box(0);
v_isShared_3393_ = v_isSharedCheck_3397_;
goto v_resetjp_3391_;
}
v_resetjp_3391_:
{
lean_object* v___x_3395_; 
if (v_isShared_3393_ == 0)
{
v___x_3395_ = v___x_3392_;
goto v_reusejp_3394_;
}
else
{
lean_object* v_reuseFailAlloc_3396_; 
v_reuseFailAlloc_3396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3396_, 0, v_a_3390_);
v___x_3395_ = v_reuseFailAlloc_3396_;
goto v_reusejp_3394_;
}
v_reusejp_3394_:
{
return v___x_3395_;
}
}
}
}
else
{
lean_object* v_a_3398_; lean_object* v___x_3400_; uint8_t v_isShared_3401_; uint8_t v_isSharedCheck_3405_; 
lean_dec(v___y_3359_);
lean_dec_ref(v___y_3358_);
lean_dec(v___y_3357_);
lean_dec_ref(v___y_3356_);
lean_dec(v___y_3355_);
lean_dec_ref(v___f_3161_);
lean_dec_ref(v_lctx_3159_);
lean_dec(v_cls_3157_);
v_a_3398_ = lean_ctor_get(v___y_3361_, 0);
v_isSharedCheck_3405_ = !lean_is_exclusive(v___y_3361_);
if (v_isSharedCheck_3405_ == 0)
{
v___x_3400_ = v___y_3361_;
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
else
{
lean_inc(v_a_3398_);
lean_dec(v___y_3361_);
v___x_3400_ = lean_box(0);
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
v_resetjp_3399_:
{
lean_object* v___x_3403_; 
if (v_isShared_3401_ == 0)
{
v___x_3403_ = v___x_3400_;
goto v_reusejp_3402_;
}
else
{
lean_object* v_reuseFailAlloc_3404_; 
v_reuseFailAlloc_3404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3404_, 0, v_a_3398_);
v___x_3403_ = v_reuseFailAlloc_3404_;
goto v_reusejp_3402_;
}
v_reusejp_3402_:
{
return v___x_3403_;
}
}
}
}
v___jp_3406_:
{
lean_object* v___x_3408_; lean_object* v___x_3409_; lean_object* v___x_3410_; lean_object* v___x_3411_; lean_object* v___x_3412_; lean_object* v___x_3413_; lean_object* v_a_3414_; lean_object* v_snd_3415_; lean_object* v_fst_3416_; lean_object* v_snd_3417_; lean_object* v___x_3418_; lean_object* v___x_3419_; 
v___x_3408_ = lean_unsigned_to_nat(0u);
v___x_3409_ = lean_unsigned_to_nat(1u);
v___x_3410_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3410_, 0, v___x_3408_);
lean_ctor_set(v___x_3410_, 1, v___x_3163_);
lean_ctor_set(v___x_3410_, 2, v___x_3409_);
lean_inc_ref_n(v___y_3407_, 3);
v___x_3411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3411_, 0, v___y_3407_);
lean_ctor_set(v___x_3411_, 1, v___y_3407_);
v___x_3412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3412_, 0, v___y_3407_);
lean_ctor_set(v___x_3412_, 1, v___x_3411_);
v___x_3413_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___redArg(v_xs_3158_, v_eqs_3169_, v_ys_3160_, v___x_3410_, v___x_3412_, v___x_3408_);
v_a_3414_ = lean_ctor_get(v___x_3413_, 0);
lean_inc(v_a_3414_);
lean_dec_ref(v___x_3413_);
v_snd_3415_ = lean_ctor_get(v_a_3414_, 1);
lean_inc(v_snd_3415_);
v_fst_3416_ = lean_ctor_get(v_a_3414_, 0);
lean_inc(v_fst_3416_);
lean_dec(v_a_3414_);
v_snd_3417_ = lean_ctor_get(v_snd_3415_, 1);
lean_inc(v_snd_3417_);
lean_dec(v_snd_3415_);
v___x_3418_ = l_Lean_mkAppN(v_ef_3164_, v_xs_3158_);
v___x_3419_ = l_Lean_mkAppN(v___y_3165_, v_ys_3160_);
if (v_forceHEq_3166_ == 0)
{
lean_object* v___x_3420_; 
v___x_3420_ = l_Lean_Meta_mkEqHEq(v___x_3418_, v___x_3419_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
v___y_3355_ = v_snd_3417_;
v___y_3356_ = v___x_3410_;
v___y_3357_ = v_fst_3416_;
v___y_3358_ = v___y_3407_;
v___y_3359_ = v___x_3408_;
v___y_3360_ = v___x_3409_;
v___y_3361_ = v___x_3420_;
goto v___jp_3354_;
}
else
{
lean_object* v___x_3421_; 
v___x_3421_ = l_Lean_Meta_mkHEq(v___x_3418_, v___x_3419_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
v___y_3355_ = v_snd_3417_;
v___y_3356_ = v___x_3410_;
v___y_3357_ = v_fst_3416_;
v___y_3358_ = v___y_3407_;
v___y_3359_ = v___x_3408_;
v___y_3360_ = v___x_3409_;
v___y_3361_ = v___x_3421_;
goto v___jp_3354_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___boxed(lean_object** _args){
lean_object* v___x_3430_ = _args[0];
lean_object* v_cls_3431_ = _args[1];
lean_object* v_xs_3432_ = _args[2];
lean_object* v_lctx_3433_ = _args[3];
lean_object* v_ys_3434_ = _args[4];
lean_object* v___f_3435_ = _args[5];
lean_object* v_fixedFun_3436_ = _args[6];
lean_object* v___x_3437_ = _args[7];
lean_object* v_ef_3438_ = _args[8];
lean_object* v___y_3439_ = _args[9];
lean_object* v_forceHEq_3440_ = _args[10];
lean_object* v_ee_3441_ = _args[11];
lean_object* v_kinds_3442_ = _args[12];
lean_object* v_eqs_3443_ = _args[13];
lean_object* v___y_3444_ = _args[14];
lean_object* v___y_3445_ = _args[15];
lean_object* v___y_3446_ = _args[16];
lean_object* v___y_3447_ = _args[17];
lean_object* v___y_3448_ = _args[18];
_start:
{
uint8_t v___x_19099__boxed_3449_; uint8_t v_fixedFun_boxed_3450_; uint8_t v_forceHEq_boxed_3451_; lean_object* v_res_3452_; 
v___x_19099__boxed_3449_ = lean_unbox(v___x_3430_);
v_fixedFun_boxed_3450_ = lean_unbox(v_fixedFun_3436_);
v_forceHEq_boxed_3451_ = lean_unbox(v_forceHEq_3440_);
v_res_3452_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__1(v___x_19099__boxed_3449_, v_cls_3431_, v_xs_3432_, v_lctx_3433_, v_ys_3434_, v___f_3435_, v_fixedFun_boxed_3450_, v___x_3437_, v_ef_3438_, v___y_3439_, v_forceHEq_boxed_3451_, v_ee_3441_, v_kinds_3442_, v_eqs_3443_, v___y_3444_, v___y_3445_, v___y_3446_, v___y_3447_);
lean_dec(v___y_3447_);
lean_dec_ref(v___y_3446_);
lean_dec(v___y_3445_);
lean_dec_ref(v___y_3444_);
lean_dec_ref(v_eqs_3443_);
lean_dec_ref(v_kinds_3442_);
lean_dec_ref(v_ys_3434_);
lean_dec_ref(v_xs_3432_);
return v_res_3452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__2(uint8_t v___x_3453_, lean_object* v_cls_3454_, lean_object* v_xs_3455_, lean_object* v_lctx_3456_, lean_object* v_ys_3457_, lean_object* v___f_3458_, uint8_t v_fixedFun_3459_, lean_object* v___x_3460_, lean_object* v_ef_3461_, lean_object* v___y_3462_, uint8_t v_forceHEq_3463_, lean_object* v_info_3464_, lean_object* v_fixedParams_3465_, lean_object* v_ee_3466_, lean_object* v___y_3467_, lean_object* v___y_3468_, lean_object* v___y_3469_, lean_object* v___y_3470_){
_start:
{
lean_object* v___x_3472_; lean_object* v___x_3473_; lean_object* v___x_3474_; lean_object* v___f_3475_; lean_object* v___x_3476_; 
v___x_3472_ = lean_box(v___x_3453_);
v___x_3473_ = lean_box(v_fixedFun_3459_);
v___x_3474_ = lean_box(v_forceHEq_3463_);
lean_inc_ref(v_ys_3457_);
lean_inc_ref(v_xs_3455_);
v___f_3475_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__1___boxed), 19, 12);
lean_closure_set(v___f_3475_, 0, v___x_3472_);
lean_closure_set(v___f_3475_, 1, v_cls_3454_);
lean_closure_set(v___f_3475_, 2, v_xs_3455_);
lean_closure_set(v___f_3475_, 3, v_lctx_3456_);
lean_closure_set(v___f_3475_, 4, v_ys_3457_);
lean_closure_set(v___f_3475_, 5, v___f_3458_);
lean_closure_set(v___f_3475_, 6, v___x_3473_);
lean_closure_set(v___f_3475_, 7, v___x_3460_);
lean_closure_set(v___f_3475_, 8, v_ef_3461_);
lean_closure_set(v___f_3475_, 9, v___y_3462_);
lean_closure_set(v___f_3475_, 10, v___x_3474_);
lean_closure_set(v___f_3475_, 11, v_ee_3466_);
v___x_3476_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs___redArg(v_info_3464_, v_xs_3455_, v_ys_3457_, v_fixedParams_3465_, v___f_3475_, v___y_3467_, v___y_3468_, v___y_3469_, v___y_3470_);
return v___x_3476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__2___boxed(lean_object** _args){
lean_object* v___x_3477_ = _args[0];
lean_object* v_cls_3478_ = _args[1];
lean_object* v_xs_3479_ = _args[2];
lean_object* v_lctx_3480_ = _args[3];
lean_object* v_ys_3481_ = _args[4];
lean_object* v___f_3482_ = _args[5];
lean_object* v_fixedFun_3483_ = _args[6];
lean_object* v___x_3484_ = _args[7];
lean_object* v_ef_3485_ = _args[8];
lean_object* v___y_3486_ = _args[9];
lean_object* v_forceHEq_3487_ = _args[10];
lean_object* v_info_3488_ = _args[11];
lean_object* v_fixedParams_3489_ = _args[12];
lean_object* v_ee_3490_ = _args[13];
lean_object* v___y_3491_ = _args[14];
lean_object* v___y_3492_ = _args[15];
lean_object* v___y_3493_ = _args[16];
lean_object* v___y_3494_ = _args[17];
lean_object* v___y_3495_ = _args[18];
_start:
{
uint8_t v___x_19621__boxed_3496_; uint8_t v_fixedFun_boxed_3497_; uint8_t v_forceHEq_boxed_3498_; lean_object* v_res_3499_; 
v___x_19621__boxed_3496_ = lean_unbox(v___x_3477_);
v_fixedFun_boxed_3497_ = lean_unbox(v_fixedFun_3483_);
v_forceHEq_boxed_3498_ = lean_unbox(v_forceHEq_3487_);
v_res_3499_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__2(v___x_19621__boxed_3496_, v_cls_3478_, v_xs_3479_, v_lctx_3480_, v_ys_3481_, v___f_3482_, v_fixedFun_boxed_3497_, v___x_3484_, v_ef_3485_, v___y_3486_, v_forceHEq_boxed_3498_, v_info_3488_, v_fixedParams_3489_, v_ee_3490_, v___y_3491_, v___y_3492_, v___y_3493_, v___y_3494_);
lean_dec(v___y_3494_);
lean_dec_ref(v___y_3493_);
lean_dec(v___y_3492_);
lean_dec_ref(v___y_3491_);
return v_res_3499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__3(lean_object* v_ef_3500_, lean_object* v_cls_3501_, lean_object* v_xs_3502_, lean_object* v_lctx_3503_, lean_object* v_ys_3504_, lean_object* v___f_3505_, uint8_t v_fixedFun_3506_, lean_object* v___x_3507_, uint8_t v_forceHEq_3508_, lean_object* v_info_3509_, lean_object* v_fixedParams_3510_, lean_object* v_pef_x27_3511_, lean_object* v___y_3512_, lean_object* v___y_3513_, lean_object* v___y_3514_, lean_object* v___y_3515_){
_start:
{
uint8_t v___x_3517_; lean_object* v___y_3519_; 
v___x_3517_ = 1;
if (v_fixedFun_3506_ == 0)
{
v___y_3519_ = v_pef_x27_3511_;
goto v___jp_3518_;
}
else
{
lean_dec_ref(v_pef_x27_3511_);
lean_inc_ref(v_ef_3500_);
v___y_3519_ = v_ef_3500_;
goto v___jp_3518_;
}
v___jp_3518_:
{
lean_object* v___x_3520_; 
lean_inc_ref(v___y_3519_);
lean_inc_ref(v_ef_3500_);
v___x_3520_ = l_Lean_Meta_mkEq(v_ef_3500_, v___y_3519_, v___y_3512_, v___y_3513_, v___y_3514_, v___y_3515_);
if (lean_obj_tag(v___x_3520_) == 0)
{
lean_object* v_a_3521_; lean_object* v___x_3522_; lean_object* v___x_3523_; lean_object* v___x_3524_; lean_object* v___f_3525_; lean_object* v___x_3526_; lean_object* v___x_3527_; 
v_a_3521_ = lean_ctor_get(v___x_3520_, 0);
lean_inc(v_a_3521_);
lean_dec_ref_known(v___x_3520_, 1);
v___x_3522_ = lean_box(v___x_3517_);
v___x_3523_ = lean_box(v_fixedFun_3506_);
v___x_3524_ = lean_box(v_forceHEq_3508_);
v___f_3525_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__2___boxed), 19, 13);
lean_closure_set(v___f_3525_, 0, v___x_3522_);
lean_closure_set(v___f_3525_, 1, v_cls_3501_);
lean_closure_set(v___f_3525_, 2, v_xs_3502_);
lean_closure_set(v___f_3525_, 3, v_lctx_3503_);
lean_closure_set(v___f_3525_, 4, v_ys_3504_);
lean_closure_set(v___f_3525_, 5, v___f_3505_);
lean_closure_set(v___f_3525_, 6, v___x_3523_);
lean_closure_set(v___f_3525_, 7, v___x_3507_);
lean_closure_set(v___f_3525_, 8, v_ef_3500_);
lean_closure_set(v___f_3525_, 9, v___y_3519_);
lean_closure_set(v___f_3525_, 10, v___x_3524_);
lean_closure_set(v___f_3525_, 11, v_info_3509_);
lean_closure_set(v___f_3525_, 12, v_fixedParams_3510_);
v___x_3526_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_withNewEqs_loop___redArg___closed__1));
v___x_3527_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v___x_3526_, v_a_3521_, v___f_3525_, v___y_3512_, v___y_3513_, v___y_3514_, v___y_3515_);
return v___x_3527_;
}
else
{
lean_object* v_a_3528_; lean_object* v___x_3530_; uint8_t v_isShared_3531_; uint8_t v_isSharedCheck_3535_; 
lean_dec_ref(v___y_3519_);
lean_dec_ref(v_fixedParams_3510_);
lean_dec_ref(v_info_3509_);
lean_dec(v___x_3507_);
lean_dec_ref(v___f_3505_);
lean_dec_ref(v_ys_3504_);
lean_dec_ref(v_lctx_3503_);
lean_dec_ref(v_xs_3502_);
lean_dec(v_cls_3501_);
lean_dec_ref(v_ef_3500_);
v_a_3528_ = lean_ctor_get(v___x_3520_, 0);
v_isSharedCheck_3535_ = !lean_is_exclusive(v___x_3520_);
if (v_isSharedCheck_3535_ == 0)
{
v___x_3530_ = v___x_3520_;
v_isShared_3531_ = v_isSharedCheck_3535_;
goto v_resetjp_3529_;
}
else
{
lean_inc(v_a_3528_);
lean_dec(v___x_3520_);
v___x_3530_ = lean_box(0);
v_isShared_3531_ = v_isSharedCheck_3535_;
goto v_resetjp_3529_;
}
v_resetjp_3529_:
{
lean_object* v___x_3533_; 
if (v_isShared_3531_ == 0)
{
v___x_3533_ = v___x_3530_;
goto v_reusejp_3532_;
}
else
{
lean_object* v_reuseFailAlloc_3534_; 
v_reuseFailAlloc_3534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3534_, 0, v_a_3528_);
v___x_3533_ = v_reuseFailAlloc_3534_;
goto v_reusejp_3532_;
}
v_reusejp_3532_:
{
return v___x_3533_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__3___boxed(lean_object** _args){
lean_object* v_ef_3536_ = _args[0];
lean_object* v_cls_3537_ = _args[1];
lean_object* v_xs_3538_ = _args[2];
lean_object* v_lctx_3539_ = _args[3];
lean_object* v_ys_3540_ = _args[4];
lean_object* v___f_3541_ = _args[5];
lean_object* v_fixedFun_3542_ = _args[6];
lean_object* v___x_3543_ = _args[7];
lean_object* v_forceHEq_3544_ = _args[8];
lean_object* v_info_3545_ = _args[9];
lean_object* v_fixedParams_3546_ = _args[10];
lean_object* v_pef_x27_3547_ = _args[11];
lean_object* v___y_3548_ = _args[12];
lean_object* v___y_3549_ = _args[13];
lean_object* v___y_3550_ = _args[14];
lean_object* v___y_3551_ = _args[15];
lean_object* v___y_3552_ = _args[16];
_start:
{
uint8_t v_fixedFun_boxed_3553_; uint8_t v_forceHEq_boxed_3554_; lean_object* v_res_3555_; 
v_fixedFun_boxed_3553_ = lean_unbox(v_fixedFun_3542_);
v_forceHEq_boxed_3554_ = lean_unbox(v_forceHEq_3544_);
v_res_3555_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__3(v_ef_3536_, v_cls_3537_, v_xs_3538_, v_lctx_3539_, v_ys_3540_, v___f_3541_, v_fixedFun_boxed_3553_, v___x_3543_, v_forceHEq_boxed_3554_, v_info_3545_, v_fixedParams_3546_, v_pef_x27_3547_, v___y_3548_, v___y_3549_, v___y_3550_, v___y_3551_);
lean_dec(v___y_3551_);
lean_dec_ref(v___y_3550_);
lean_dec(v___y_3549_);
lean_dec_ref(v___y_3548_);
return v_res_3555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__4(lean_object* v_cls_3559_, lean_object* v_xs_3560_, lean_object* v_lctx_3561_, lean_object* v_ys_3562_, lean_object* v___f_3563_, uint8_t v_fixedFun_3564_, lean_object* v___x_3565_, uint8_t v_forceHEq_3566_, lean_object* v_info_3567_, lean_object* v_fixedParams_3568_, lean_object* v_fType_3569_, lean_object* v_ef_3570_, lean_object* v___y_3571_, lean_object* v___y_3572_, lean_object* v___y_3573_, lean_object* v___y_3574_){
_start:
{
lean_object* v___x_3576_; lean_object* v___x_3577_; lean_object* v___f_3578_; lean_object* v___x_3579_; lean_object* v___x_3580_; 
v___x_3576_ = lean_box(v_fixedFun_3564_);
v___x_3577_ = lean_box(v_forceHEq_3566_);
v___f_3578_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__3___boxed), 17, 11);
lean_closure_set(v___f_3578_, 0, v_ef_3570_);
lean_closure_set(v___f_3578_, 1, v_cls_3559_);
lean_closure_set(v___f_3578_, 2, v_xs_3560_);
lean_closure_set(v___f_3578_, 3, v_lctx_3561_);
lean_closure_set(v___f_3578_, 4, v_ys_3562_);
lean_closure_set(v___f_3578_, 5, v___f_3563_);
lean_closure_set(v___f_3578_, 6, v___x_3576_);
lean_closure_set(v___f_3578_, 7, v___x_3565_);
lean_closure_set(v___f_3578_, 8, v___x_3577_);
lean_closure_set(v___f_3578_, 9, v_info_3567_);
lean_closure_set(v___f_3578_, 10, v_fixedParams_3568_);
v___x_3579_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___closed__1));
v___x_3580_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v___x_3579_, v_fType_3569_, v___f_3578_, v___y_3571_, v___y_3572_, v___y_3573_, v___y_3574_);
return v___x_3580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___boxed(lean_object** _args){
lean_object* v_cls_3581_ = _args[0];
lean_object* v_xs_3582_ = _args[1];
lean_object* v_lctx_3583_ = _args[2];
lean_object* v_ys_3584_ = _args[3];
lean_object* v___f_3585_ = _args[4];
lean_object* v_fixedFun_3586_ = _args[5];
lean_object* v___x_3587_ = _args[6];
lean_object* v_forceHEq_3588_ = _args[7];
lean_object* v_info_3589_ = _args[8];
lean_object* v_fixedParams_3590_ = _args[9];
lean_object* v_fType_3591_ = _args[10];
lean_object* v_ef_3592_ = _args[11];
lean_object* v___y_3593_ = _args[12];
lean_object* v___y_3594_ = _args[13];
lean_object* v___y_3595_ = _args[14];
lean_object* v___y_3596_ = _args[15];
lean_object* v___y_3597_ = _args[16];
_start:
{
uint8_t v_fixedFun_boxed_3598_; uint8_t v_forceHEq_boxed_3599_; lean_object* v_res_3600_; 
v_fixedFun_boxed_3598_ = lean_unbox(v_fixedFun_3586_);
v_forceHEq_boxed_3599_ = lean_unbox(v_forceHEq_3588_);
v_res_3600_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__4(v_cls_3581_, v_xs_3582_, v_lctx_3583_, v_ys_3584_, v___f_3585_, v_fixedFun_boxed_3598_, v___x_3587_, v_forceHEq_boxed_3599_, v_info_3589_, v_fixedParams_3590_, v_fType_3591_, v_ef_3592_, v___y_3593_, v___y_3594_, v___y_3595_, v___y_3596_);
lean_dec(v___y_3596_);
lean_dec_ref(v___y_3595_);
lean_dec(v___y_3594_);
lean_dec_ref(v___y_3593_);
return v_res_3600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5(lean_object* v_a_3603_, lean_object* v_a_3604_){
_start:
{
if (lean_obj_tag(v_a_3603_) == 0)
{
lean_object* v___x_3605_; 
v___x_3605_ = l_List_reverse___redArg(v_a_3604_);
return v___x_3605_;
}
else
{
lean_object* v_head_3606_; lean_object* v_tail_3607_; lean_object* v___x_3609_; uint8_t v_isShared_3610_; uint8_t v_isSharedCheck_3622_; 
v_head_3606_ = lean_ctor_get(v_a_3603_, 0);
v_tail_3607_ = lean_ctor_get(v_a_3603_, 1);
v_isSharedCheck_3622_ = !lean_is_exclusive(v_a_3603_);
if (v_isSharedCheck_3622_ == 0)
{
v___x_3609_ = v_a_3603_;
v_isShared_3610_ = v_isSharedCheck_3622_;
goto v_resetjp_3608_;
}
else
{
lean_inc(v_tail_3607_);
lean_inc(v_head_3606_);
lean_dec(v_a_3603_);
v___x_3609_ = lean_box(0);
v_isShared_3610_ = v_isSharedCheck_3622_;
goto v_resetjp_3608_;
}
v_resetjp_3608_:
{
lean_object* v___y_3612_; uint8_t v___x_3619_; 
v___x_3619_ = lean_unbox(v_head_3606_);
lean_dec(v_head_3606_);
if (v___x_3619_ == 0)
{
lean_object* v___x_3620_; 
v___x_3620_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__0));
v___y_3612_ = v___x_3620_;
goto v___jp_3611_;
}
else
{
lean_object* v___x_3621_; 
v___x_3621_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__1));
v___y_3612_ = v___x_3621_;
goto v___jp_3611_;
}
v___jp_3611_:
{
lean_object* v___x_3613_; lean_object* v___x_3614_; lean_object* v___x_3616_; 
lean_inc_ref(v___y_3612_);
v___x_3613_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3613_, 0, v___y_3612_);
v___x_3614_ = l_Lean_MessageData_ofFormat(v___x_3613_);
if (v_isShared_3610_ == 0)
{
lean_ctor_set(v___x_3609_, 1, v_a_3604_);
lean_ctor_set(v___x_3609_, 0, v___x_3614_);
v___x_3616_ = v___x_3609_;
goto v_reusejp_3615_;
}
else
{
lean_object* v_reuseFailAlloc_3618_; 
v_reuseFailAlloc_3618_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3618_, 0, v___x_3614_);
lean_ctor_set(v_reuseFailAlloc_3618_, 1, v_a_3604_);
v___x_3616_ = v_reuseFailAlloc_3618_;
goto v_reusejp_3615_;
}
v_reusejp_3615_:
{
v_a_3603_ = v_tail_3607_;
v_a_3604_ = v___x_3616_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__6(lean_object* v_a_3623_, lean_object* v_a_3624_){
_start:
{
if (lean_obj_tag(v_a_3623_) == 0)
{
lean_object* v___x_3625_; 
v___x_3625_ = l_List_reverse___redArg(v_a_3624_);
return v___x_3625_;
}
else
{
lean_object* v_head_3626_; lean_object* v_tail_3627_; lean_object* v___x_3629_; uint8_t v_isShared_3630_; uint8_t v_isSharedCheck_3636_; 
v_head_3626_ = lean_ctor_get(v_a_3623_, 0);
v_tail_3627_ = lean_ctor_get(v_a_3623_, 1);
v_isSharedCheck_3636_ = !lean_is_exclusive(v_a_3623_);
if (v_isSharedCheck_3636_ == 0)
{
v___x_3629_ = v_a_3623_;
v_isShared_3630_ = v_isSharedCheck_3636_;
goto v_resetjp_3628_;
}
else
{
lean_inc(v_tail_3627_);
lean_inc(v_head_3626_);
lean_dec(v_a_3623_);
v___x_3629_ = lean_box(0);
v_isShared_3630_ = v_isSharedCheck_3636_;
goto v_resetjp_3628_;
}
v_resetjp_3628_:
{
lean_object* v___x_3631_; lean_object* v___x_3633_; 
v___x_3631_ = l_Lean_MessageData_ofExpr(v_head_3626_);
if (v_isShared_3630_ == 0)
{
lean_ctor_set(v___x_3629_, 1, v_a_3624_);
lean_ctor_set(v___x_3629_, 0, v___x_3631_);
v___x_3633_ = v___x_3629_;
goto v_reusejp_3632_;
}
else
{
lean_object* v_reuseFailAlloc_3635_; 
v_reuseFailAlloc_3635_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3635_, 0, v___x_3631_);
lean_ctor_set(v_reuseFailAlloc_3635_, 1, v_a_3624_);
v___x_3633_ = v_reuseFailAlloc_3635_;
goto v_reusejp_3632_;
}
v_reusejp_3632_:
{
v_a_3623_ = v_tail_3627_;
v_a_3624_ = v___x_3633_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__3(void){
_start:
{
lean_object* v___x_3641_; lean_object* v___x_3642_; 
v___x_3641_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__2));
v___x_3642_ = l_Lean_stringToMessageData(v___x_3641_);
return v___x_3642_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__5(void){
_start:
{
lean_object* v___x_3644_; lean_object* v___x_3645_; 
v___x_3644_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__4));
v___x_3645_ = l_Lean_stringToMessageData(v___x_3644_);
return v___x_3645_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__7(void){
_start:
{
lean_object* v___x_3647_; lean_object* v___x_3648_; 
v___x_3647_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__6));
v___x_3648_ = l_Lean_stringToMessageData(v___x_3647_);
return v___x_3648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5(lean_object* v___f_3649_, lean_object* v_cls_3650_, uint8_t v_fixedFun_3651_, lean_object* v___x_3652_, uint8_t v_forceHEq_3653_, lean_object* v_info_3654_, lean_object* v_fType_3655_, lean_object* v_xs_3656_, lean_object* v_ys_3657_, lean_object* v_fixedParams_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_){
_start:
{
lean_object* v___y_3665_; lean_object* v___y_3666_; lean_object* v___y_3667_; lean_object* v___y_3668_; lean_object* v___y_3676_; lean_object* v___y_3677_; lean_object* v___y_3678_; lean_object* v___y_3679_; lean_object* v___y_3707_; lean_object* v___y_3708_; lean_object* v___y_3709_; lean_object* v___y_3710_; lean_object* v___x_3737_; 
lean_inc_ref(v___f_3649_);
lean_inc(v___y_3662_);
lean_inc_ref(v___y_3661_);
lean_inc(v___y_3660_);
lean_inc_ref(v___y_3659_);
v___x_3737_ = lean_apply_5(v___f_3649_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, lean_box(0));
if (lean_obj_tag(v___x_3737_) == 0)
{
lean_object* v_a_3738_; uint8_t v___x_3739_; 
v_a_3738_ = lean_ctor_get(v___x_3737_, 0);
lean_inc(v_a_3738_);
lean_dec_ref_known(v___x_3737_, 1);
v___x_3739_ = lean_unbox(v_a_3738_);
lean_dec(v_a_3738_);
if (v___x_3739_ == 0)
{
v___y_3707_ = v___y_3659_;
v___y_3708_ = v___y_3660_;
v___y_3709_ = v___y_3661_;
v___y_3710_ = v___y_3662_;
goto v___jp_3706_;
}
else
{
lean_object* v___x_3740_; lean_object* v___x_3741_; lean_object* v___x_3742_; lean_object* v___x_3743_; lean_object* v___x_3744_; lean_object* v___x_3745_; lean_object* v___x_3746_; 
v___x_3740_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__7, &lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__7_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__7);
lean_inc_ref(v_xs_3656_);
v___x_3741_ = lean_array_to_list(v_xs_3656_);
v___x_3742_ = lean_box(0);
v___x_3743_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__6(v___x_3741_, v___x_3742_);
v___x_3744_ = l_Lean_MessageData_ofList(v___x_3743_);
v___x_3745_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3745_, 0, v___x_3740_);
lean_ctor_set(v___x_3745_, 1, v___x_3744_);
lean_inc(v_cls_3650_);
v___x_3746_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3650_, v___x_3745_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
if (lean_obj_tag(v___x_3746_) == 0)
{
lean_dec_ref_known(v___x_3746_, 1);
v___y_3707_ = v___y_3659_;
v___y_3708_ = v___y_3660_;
v___y_3709_ = v___y_3661_;
v___y_3710_ = v___y_3662_;
goto v___jp_3706_;
}
else
{
lean_object* v_a_3747_; lean_object* v___x_3749_; uint8_t v_isShared_3750_; uint8_t v_isSharedCheck_3754_; 
lean_dec_ref(v_fixedParams_3658_);
lean_dec_ref(v_ys_3657_);
lean_dec_ref(v_xs_3656_);
lean_dec_ref(v_fType_3655_);
lean_dec_ref(v_info_3654_);
lean_dec(v___x_3652_);
lean_dec(v_cls_3650_);
lean_dec_ref(v___f_3649_);
v_a_3747_ = lean_ctor_get(v___x_3746_, 0);
v_isSharedCheck_3754_ = !lean_is_exclusive(v___x_3746_);
if (v_isSharedCheck_3754_ == 0)
{
v___x_3749_ = v___x_3746_;
v_isShared_3750_ = v_isSharedCheck_3754_;
goto v_resetjp_3748_;
}
else
{
lean_inc(v_a_3747_);
lean_dec(v___x_3746_);
v___x_3749_ = lean_box(0);
v_isShared_3750_ = v_isSharedCheck_3754_;
goto v_resetjp_3748_;
}
v_resetjp_3748_:
{
lean_object* v___x_3752_; 
if (v_isShared_3750_ == 0)
{
v___x_3752_ = v___x_3749_;
goto v_reusejp_3751_;
}
else
{
lean_object* v_reuseFailAlloc_3753_; 
v_reuseFailAlloc_3753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3753_, 0, v_a_3747_);
v___x_3752_ = v_reuseFailAlloc_3753_;
goto v_reusejp_3751_;
}
v_reusejp_3751_:
{
return v___x_3752_;
}
}
}
}
}
else
{
lean_object* v_a_3755_; lean_object* v___x_3757_; uint8_t v_isShared_3758_; uint8_t v_isSharedCheck_3762_; 
lean_dec_ref(v_fixedParams_3658_);
lean_dec_ref(v_ys_3657_);
lean_dec_ref(v_xs_3656_);
lean_dec_ref(v_fType_3655_);
lean_dec_ref(v_info_3654_);
lean_dec(v___x_3652_);
lean_dec(v_cls_3650_);
lean_dec_ref(v___f_3649_);
v_a_3755_ = lean_ctor_get(v___x_3737_, 0);
v_isSharedCheck_3762_ = !lean_is_exclusive(v___x_3737_);
if (v_isSharedCheck_3762_ == 0)
{
v___x_3757_ = v___x_3737_;
v_isShared_3758_ = v_isSharedCheck_3762_;
goto v_resetjp_3756_;
}
else
{
lean_inc(v_a_3755_);
lean_dec(v___x_3737_);
v___x_3757_ = lean_box(0);
v_isShared_3758_ = v_isSharedCheck_3762_;
goto v_resetjp_3756_;
}
v_resetjp_3756_:
{
lean_object* v___x_3760_; 
if (v_isShared_3758_ == 0)
{
v___x_3760_ = v___x_3757_;
goto v_reusejp_3759_;
}
else
{
lean_object* v_reuseFailAlloc_3761_; 
v_reuseFailAlloc_3761_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3761_, 0, v_a_3755_);
v___x_3760_ = v_reuseFailAlloc_3761_;
goto v_reusejp_3759_;
}
v_reusejp_3759_:
{
return v___x_3760_;
}
}
}
v___jp_3664_:
{
lean_object* v_lctx_3669_; lean_object* v___x_3670_; lean_object* v___x_3671_; lean_object* v___f_3672_; lean_object* v___x_3673_; lean_object* v___x_3674_; 
v_lctx_3669_ = lean_ctor_get(v___y_3665_, 2);
v___x_3670_ = lean_box(v_fixedFun_3651_);
v___x_3671_ = lean_box(v_forceHEq_3653_);
lean_inc_ref(v_fType_3655_);
lean_inc_ref(v_lctx_3669_);
v___f_3672_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__4___boxed), 17, 11);
lean_closure_set(v___f_3672_, 0, v_cls_3650_);
lean_closure_set(v___f_3672_, 1, v_xs_3656_);
lean_closure_set(v___f_3672_, 2, v_lctx_3669_);
lean_closure_set(v___f_3672_, 3, v_ys_3657_);
lean_closure_set(v___f_3672_, 4, v___f_3649_);
lean_closure_set(v___f_3672_, 5, v___x_3670_);
lean_closure_set(v___f_3672_, 6, v___x_3652_);
lean_closure_set(v___f_3672_, 7, v___x_3671_);
lean_closure_set(v___f_3672_, 8, v_info_3654_);
lean_closure_set(v___f_3672_, 9, v_fixedParams_3658_);
lean_closure_set(v___f_3672_, 10, v_fType_3655_);
v___x_3673_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__1));
v___x_3674_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope_loop_spec__0___redArg(v___x_3673_, v_fType_3655_, v___f_3672_, v___y_3665_, v___y_3666_, v___y_3667_, v___y_3668_);
return v___x_3674_;
}
v___jp_3675_:
{
lean_object* v___x_3680_; 
lean_inc_ref(v___f_3649_);
lean_inc(v___y_3679_);
lean_inc_ref(v___y_3678_);
lean_inc(v___y_3677_);
lean_inc_ref(v___y_3676_);
v___x_3680_ = lean_apply_5(v___f_3649_, v___y_3676_, v___y_3677_, v___y_3678_, v___y_3679_, lean_box(0));
if (lean_obj_tag(v___x_3680_) == 0)
{
lean_object* v_a_3681_; uint8_t v___x_3682_; 
v_a_3681_ = lean_ctor_get(v___x_3680_, 0);
lean_inc(v_a_3681_);
lean_dec_ref_known(v___x_3680_, 1);
v___x_3682_ = lean_unbox(v_a_3681_);
lean_dec(v_a_3681_);
if (v___x_3682_ == 0)
{
v___y_3665_ = v___y_3676_;
v___y_3666_ = v___y_3677_;
v___y_3667_ = v___y_3678_;
v___y_3668_ = v___y_3679_;
goto v___jp_3664_;
}
else
{
lean_object* v___x_3683_; lean_object* v___x_3684_; lean_object* v___x_3685_; lean_object* v___x_3686_; lean_object* v___x_3687_; lean_object* v___x_3688_; lean_object* v___x_3689_; 
v___x_3683_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__3, &lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__3_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__3);
lean_inc_ref(v_fixedParams_3658_);
v___x_3684_ = lean_array_to_list(v_fixedParams_3658_);
v___x_3685_ = lean_box(0);
v___x_3686_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5(v___x_3684_, v___x_3685_);
v___x_3687_ = l_Lean_MessageData_ofList(v___x_3686_);
v___x_3688_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3688_, 0, v___x_3683_);
lean_ctor_set(v___x_3688_, 1, v___x_3687_);
lean_inc(v_cls_3650_);
v___x_3689_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3650_, v___x_3688_, v___y_3676_, v___y_3677_, v___y_3678_, v___y_3679_);
if (lean_obj_tag(v___x_3689_) == 0)
{
lean_dec_ref_known(v___x_3689_, 1);
v___y_3665_ = v___y_3676_;
v___y_3666_ = v___y_3677_;
v___y_3667_ = v___y_3678_;
v___y_3668_ = v___y_3679_;
goto v___jp_3664_;
}
else
{
lean_object* v_a_3690_; lean_object* v___x_3692_; uint8_t v_isShared_3693_; uint8_t v_isSharedCheck_3697_; 
lean_dec_ref(v_fixedParams_3658_);
lean_dec_ref(v_ys_3657_);
lean_dec_ref(v_xs_3656_);
lean_dec_ref(v_fType_3655_);
lean_dec_ref(v_info_3654_);
lean_dec(v___x_3652_);
lean_dec(v_cls_3650_);
lean_dec_ref(v___f_3649_);
v_a_3690_ = lean_ctor_get(v___x_3689_, 0);
v_isSharedCheck_3697_ = !lean_is_exclusive(v___x_3689_);
if (v_isSharedCheck_3697_ == 0)
{
v___x_3692_ = v___x_3689_;
v_isShared_3693_ = v_isSharedCheck_3697_;
goto v_resetjp_3691_;
}
else
{
lean_inc(v_a_3690_);
lean_dec(v___x_3689_);
v___x_3692_ = lean_box(0);
v_isShared_3693_ = v_isSharedCheck_3697_;
goto v_resetjp_3691_;
}
v_resetjp_3691_:
{
lean_object* v___x_3695_; 
if (v_isShared_3693_ == 0)
{
v___x_3695_ = v___x_3692_;
goto v_reusejp_3694_;
}
else
{
lean_object* v_reuseFailAlloc_3696_; 
v_reuseFailAlloc_3696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3696_, 0, v_a_3690_);
v___x_3695_ = v_reuseFailAlloc_3696_;
goto v_reusejp_3694_;
}
v_reusejp_3694_:
{
return v___x_3695_;
}
}
}
}
}
else
{
lean_object* v_a_3698_; lean_object* v___x_3700_; uint8_t v_isShared_3701_; uint8_t v_isSharedCheck_3705_; 
lean_dec_ref(v_fixedParams_3658_);
lean_dec_ref(v_ys_3657_);
lean_dec_ref(v_xs_3656_);
lean_dec_ref(v_fType_3655_);
lean_dec_ref(v_info_3654_);
lean_dec(v___x_3652_);
lean_dec(v_cls_3650_);
lean_dec_ref(v___f_3649_);
v_a_3698_ = lean_ctor_get(v___x_3680_, 0);
v_isSharedCheck_3705_ = !lean_is_exclusive(v___x_3680_);
if (v_isSharedCheck_3705_ == 0)
{
v___x_3700_ = v___x_3680_;
v_isShared_3701_ = v_isSharedCheck_3705_;
goto v_resetjp_3699_;
}
else
{
lean_inc(v_a_3698_);
lean_dec(v___x_3680_);
v___x_3700_ = lean_box(0);
v_isShared_3701_ = v_isSharedCheck_3705_;
goto v_resetjp_3699_;
}
v_resetjp_3699_:
{
lean_object* v___x_3703_; 
if (v_isShared_3701_ == 0)
{
v___x_3703_ = v___x_3700_;
goto v_reusejp_3702_;
}
else
{
lean_object* v_reuseFailAlloc_3704_; 
v_reuseFailAlloc_3704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3704_, 0, v_a_3698_);
v___x_3703_ = v_reuseFailAlloc_3704_;
goto v_reusejp_3702_;
}
v_reusejp_3702_:
{
return v___x_3703_;
}
}
}
}
v___jp_3706_:
{
lean_object* v___x_3711_; 
lean_inc_ref(v___f_3649_);
lean_inc(v___y_3710_);
lean_inc_ref(v___y_3709_);
lean_inc(v___y_3708_);
lean_inc_ref(v___y_3707_);
v___x_3711_ = lean_apply_5(v___f_3649_, v___y_3707_, v___y_3708_, v___y_3709_, v___y_3710_, lean_box(0));
if (lean_obj_tag(v___x_3711_) == 0)
{
lean_object* v_a_3712_; uint8_t v___x_3713_; 
v_a_3712_ = lean_ctor_get(v___x_3711_, 0);
lean_inc(v_a_3712_);
lean_dec_ref_known(v___x_3711_, 1);
v___x_3713_ = lean_unbox(v_a_3712_);
lean_dec(v_a_3712_);
if (v___x_3713_ == 0)
{
v___y_3676_ = v___y_3707_;
v___y_3677_ = v___y_3708_;
v___y_3678_ = v___y_3709_;
v___y_3679_ = v___y_3710_;
goto v___jp_3675_;
}
else
{
lean_object* v___x_3714_; lean_object* v___x_3715_; lean_object* v___x_3716_; lean_object* v___x_3717_; lean_object* v___x_3718_; lean_object* v___x_3719_; lean_object* v___x_3720_; 
v___x_3714_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__5, &lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__5_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___closed__5);
lean_inc_ref(v_ys_3657_);
v___x_3715_ = lean_array_to_list(v_ys_3657_);
v___x_3716_ = lean_box(0);
v___x_3717_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__6(v___x_3715_, v___x_3716_);
v___x_3718_ = l_Lean_MessageData_ofList(v___x_3717_);
v___x_3719_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3719_, 0, v___x_3714_);
lean_ctor_set(v___x_3719_, 1, v___x_3718_);
lean_inc(v_cls_3650_);
v___x_3720_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3650_, v___x_3719_, v___y_3707_, v___y_3708_, v___y_3709_, v___y_3710_);
if (lean_obj_tag(v___x_3720_) == 0)
{
lean_dec_ref_known(v___x_3720_, 1);
v___y_3676_ = v___y_3707_;
v___y_3677_ = v___y_3708_;
v___y_3678_ = v___y_3709_;
v___y_3679_ = v___y_3710_;
goto v___jp_3675_;
}
else
{
lean_object* v_a_3721_; lean_object* v___x_3723_; uint8_t v_isShared_3724_; uint8_t v_isSharedCheck_3728_; 
lean_dec_ref(v_fixedParams_3658_);
lean_dec_ref(v_ys_3657_);
lean_dec_ref(v_xs_3656_);
lean_dec_ref(v_fType_3655_);
lean_dec_ref(v_info_3654_);
lean_dec(v___x_3652_);
lean_dec(v_cls_3650_);
lean_dec_ref(v___f_3649_);
v_a_3721_ = lean_ctor_get(v___x_3720_, 0);
v_isSharedCheck_3728_ = !lean_is_exclusive(v___x_3720_);
if (v_isSharedCheck_3728_ == 0)
{
v___x_3723_ = v___x_3720_;
v_isShared_3724_ = v_isSharedCheck_3728_;
goto v_resetjp_3722_;
}
else
{
lean_inc(v_a_3721_);
lean_dec(v___x_3720_);
v___x_3723_ = lean_box(0);
v_isShared_3724_ = v_isSharedCheck_3728_;
goto v_resetjp_3722_;
}
v_resetjp_3722_:
{
lean_object* v___x_3726_; 
if (v_isShared_3724_ == 0)
{
v___x_3726_ = v___x_3723_;
goto v_reusejp_3725_;
}
else
{
lean_object* v_reuseFailAlloc_3727_; 
v_reuseFailAlloc_3727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3727_, 0, v_a_3721_);
v___x_3726_ = v_reuseFailAlloc_3727_;
goto v_reusejp_3725_;
}
v_reusejp_3725_:
{
return v___x_3726_;
}
}
}
}
}
else
{
lean_object* v_a_3729_; lean_object* v___x_3731_; uint8_t v_isShared_3732_; uint8_t v_isSharedCheck_3736_; 
lean_dec_ref(v_fixedParams_3658_);
lean_dec_ref(v_ys_3657_);
lean_dec_ref(v_xs_3656_);
lean_dec_ref(v_fType_3655_);
lean_dec_ref(v_info_3654_);
lean_dec(v___x_3652_);
lean_dec(v_cls_3650_);
lean_dec_ref(v___f_3649_);
v_a_3729_ = lean_ctor_get(v___x_3711_, 0);
v_isSharedCheck_3736_ = !lean_is_exclusive(v___x_3711_);
if (v_isSharedCheck_3736_ == 0)
{
v___x_3731_ = v___x_3711_;
v_isShared_3732_ = v_isSharedCheck_3736_;
goto v_resetjp_3730_;
}
else
{
lean_inc(v_a_3729_);
lean_dec(v___x_3711_);
v___x_3731_ = lean_box(0);
v_isShared_3732_ = v_isSharedCheck_3736_;
goto v_resetjp_3730_;
}
v_resetjp_3730_:
{
lean_object* v___x_3734_; 
if (v_isShared_3732_ == 0)
{
v___x_3734_ = v___x_3731_;
goto v_reusejp_3733_;
}
else
{
lean_object* v_reuseFailAlloc_3735_; 
v_reuseFailAlloc_3735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3735_, 0, v_a_3729_);
v___x_3734_ = v_reuseFailAlloc_3735_;
goto v_reusejp_3733_;
}
v_reusejp_3733_:
{
return v___x_3734_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___boxed(lean_object* v___f_3763_, lean_object* v_cls_3764_, lean_object* v_fixedFun_3765_, lean_object* v___x_3766_, lean_object* v_forceHEq_3767_, lean_object* v_info_3768_, lean_object* v_fType_3769_, lean_object* v_xs_3770_, lean_object* v_ys_3771_, lean_object* v_fixedParams_3772_, lean_object* v___y_3773_, lean_object* v___y_3774_, lean_object* v___y_3775_, lean_object* v___y_3776_, lean_object* v___y_3777_){
_start:
{
uint8_t v_fixedFun_boxed_3778_; uint8_t v_forceHEq_boxed_3779_; lean_object* v_res_3780_; 
v_fixedFun_boxed_3778_ = lean_unbox(v_fixedFun_3765_);
v_forceHEq_boxed_3779_ = lean_unbox(v_forceHEq_3767_);
v_res_3780_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__5(v___f_3763_, v_cls_3764_, v_fixedFun_boxed_3778_, v___x_3766_, v_forceHEq_boxed_3779_, v_info_3768_, v_fType_3769_, v_xs_3770_, v_ys_3771_, v_fixedParams_3772_, v___y_3773_, v___y_3774_, v___y_3775_, v___y_3776_);
lean_dec(v___y_3776_);
lean_dec_ref(v___y_3775_);
lean_dec(v___y_3774_);
lean_dec_ref(v___y_3773_);
return v_res_3780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__0(lean_object* v_a_3781_, lean_object* v_a_3782_){
_start:
{
if (lean_obj_tag(v_a_3781_) == 0)
{
lean_object* v___x_3783_; 
v___x_3783_ = l_List_reverse___redArg(v_a_3782_);
return v___x_3783_;
}
else
{
lean_object* v_head_3784_; lean_object* v_tail_3785_; lean_object* v___x_3787_; uint8_t v_isShared_3788_; uint8_t v_isSharedCheck_3796_; 
v_head_3784_ = lean_ctor_get(v_a_3781_, 0);
v_tail_3785_ = lean_ctor_get(v_a_3781_, 1);
v_isSharedCheck_3796_ = !lean_is_exclusive(v_a_3781_);
if (v_isSharedCheck_3796_ == 0)
{
v___x_3787_ = v_a_3781_;
v_isShared_3788_ = v_isSharedCheck_3796_;
goto v_resetjp_3786_;
}
else
{
lean_inc(v_tail_3785_);
lean_inc(v_head_3784_);
lean_dec(v_a_3781_);
v___x_3787_ = lean_box(0);
v_isShared_3788_ = v_isSharedCheck_3796_;
goto v_resetjp_3786_;
}
v_resetjp_3786_:
{
lean_object* v___x_3789_; lean_object* v___x_3790_; lean_object* v___x_3791_; lean_object* v___x_3793_; 
v___x_3789_ = l_Nat_reprFast(v_head_3784_);
v___x_3790_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3790_, 0, v___x_3789_);
v___x_3791_ = l_Lean_MessageData_ofFormat(v___x_3790_);
if (v_isShared_3788_ == 0)
{
lean_ctor_set(v___x_3787_, 1, v_a_3782_);
lean_ctor_set(v___x_3787_, 0, v___x_3791_);
v___x_3793_ = v___x_3787_;
goto v_reusejp_3792_;
}
else
{
lean_object* v_reuseFailAlloc_3795_; 
v_reuseFailAlloc_3795_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3795_, 0, v___x_3791_);
lean_ctor_set(v_reuseFailAlloc_3795_, 1, v_a_3782_);
v___x_3793_ = v_reuseFailAlloc_3795_;
goto v_reusejp_3792_;
}
v_reusejp_3792_:
{
v_a_3781_ = v_tail_3785_;
v_a_3782_ = v___x_3793_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__8(lean_object* v_a_3797_, lean_object* v_a_3798_){
_start:
{
if (lean_obj_tag(v_a_3797_) == 0)
{
lean_object* v___x_3799_; 
v___x_3799_ = l_List_reverse___redArg(v_a_3798_);
return v___x_3799_;
}
else
{
lean_object* v_head_3800_; lean_object* v_tail_3801_; lean_object* v___x_3803_; uint8_t v_isShared_3804_; uint8_t v_isSharedCheck_3813_; 
v_head_3800_ = lean_ctor_get(v_a_3797_, 0);
v_tail_3801_ = lean_ctor_get(v_a_3797_, 1);
v_isSharedCheck_3813_ = !lean_is_exclusive(v_a_3797_);
if (v_isSharedCheck_3813_ == 0)
{
v___x_3803_ = v_a_3797_;
v_isShared_3804_ = v_isSharedCheck_3813_;
goto v_resetjp_3802_;
}
else
{
lean_inc(v_tail_3801_);
lean_inc(v_head_3800_);
lean_dec(v_a_3797_);
v___x_3803_ = lean_box(0);
v_isShared_3804_ = v_isSharedCheck_3813_;
goto v_resetjp_3802_;
}
v_resetjp_3802_:
{
lean_object* v___x_3805_; lean_object* v___x_3806_; lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3810_; 
v___x_3805_ = lean_array_to_list(v_head_3800_);
v___x_3806_ = lean_box(0);
v___x_3807_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__0(v___x_3805_, v___x_3806_);
v___x_3808_ = l_Lean_MessageData_ofList(v___x_3807_);
if (v_isShared_3804_ == 0)
{
lean_ctor_set(v___x_3803_, 1, v_a_3798_);
lean_ctor_set(v___x_3803_, 0, v___x_3808_);
v___x_3810_ = v___x_3803_;
goto v_reusejp_3809_;
}
else
{
lean_object* v_reuseFailAlloc_3812_; 
v_reuseFailAlloc_3812_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3812_, 0, v___x_3808_);
lean_ctor_set(v_reuseFailAlloc_3812_, 1, v_a_3798_);
v___x_3810_ = v_reuseFailAlloc_3812_;
goto v_reusejp_3809_;
}
v_reusejp_3809_:
{
v_a_3797_ = v_tail_3801_;
v_a_3798_ = v___x_3810_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_mkRichHCongr_spec__7(size_t v_sz_3814_, size_t v_i_3815_, lean_object* v_bs_3816_){
_start:
{
uint8_t v___x_3817_; 
v___x_3817_ = lean_usize_dec_lt(v_i_3815_, v_sz_3814_);
if (v___x_3817_ == 0)
{
return v_bs_3816_;
}
else
{
lean_object* v_v_3818_; lean_object* v_backDeps_3819_; lean_object* v___x_3820_; lean_object* v_bs_x27_3821_; size_t v___x_3822_; size_t v___x_3823_; lean_object* v___x_3824_; 
v_v_3818_ = lean_array_uget_borrowed(v_bs_3816_, v_i_3815_);
v_backDeps_3819_ = lean_ctor_get(v_v_3818_, 0);
lean_inc_ref(v_backDeps_3819_);
v___x_3820_ = lean_unsigned_to_nat(0u);
v_bs_x27_3821_ = lean_array_uset(v_bs_3816_, v_i_3815_, v___x_3820_);
v___x_3822_ = ((size_t)1ULL);
v___x_3823_ = lean_usize_add(v_i_3815_, v___x_3822_);
v___x_3824_ = lean_array_uset(v_bs_x27_3821_, v_i_3815_, v_backDeps_3819_);
v_i_3815_ = v___x_3823_;
v_bs_3816_ = v___x_3824_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_mkRichHCongr_spec__7___boxed(lean_object* v_sz_3826_, lean_object* v_i_3827_, lean_object* v_bs_3828_){
_start:
{
size_t v_sz_boxed_3829_; size_t v_i_boxed_3830_; lean_object* v_res_3831_; 
v_sz_boxed_3829_ = lean_unbox_usize(v_sz_3826_);
lean_dec(v_sz_3826_);
v_i_boxed_3830_ = lean_unbox_usize(v_i_3827_);
lean_dec(v_i_3827_);
v_res_3831_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_mkRichHCongr_spec__7(v_sz_boxed_3829_, v_i_boxed_3830_, v_bs_3828_);
return v_res_3831_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__2(void){
_start:
{
lean_object* v___x_3835_; lean_object* v___x_3836_; 
v___x_3835_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___closed__1));
v___x_3836_ = l_Lean_stringToMessageData(v___x_3835_);
return v___x_3836_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__4(void){
_start:
{
lean_object* v___x_3838_; lean_object* v___x_3839_; 
v___x_3838_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___closed__3));
v___x_3839_ = l_Lean_stringToMessageData(v___x_3838_);
return v___x_3839_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__6(void){
_start:
{
lean_object* v___x_3841_; lean_object* v___x_3842_; 
v___x_3841_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___closed__5));
v___x_3842_ = l_Lean_stringToMessageData(v___x_3841_);
return v___x_3842_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__8(void){
_start:
{
lean_object* v___x_3844_; lean_object* v___x_3845_; 
v___x_3844_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___closed__7));
v___x_3845_ = l_Lean_stringToMessageData(v___x_3844_);
return v___x_3845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr(lean_object* v_fType_3846_, lean_object* v_info_3847_, uint8_t v_fixedFun_3848_, lean_object* v_fixedParams_3849_, uint8_t v_forceHEq_3850_, lean_object* v_a_3851_, lean_object* v_a_3852_, lean_object* v_a_3853_, lean_object* v_a_3854_){
_start:
{
lean_object* v_cls_3856_; lean_object* v___f_3857_; lean_object* v___y_3859_; lean_object* v___y_3860_; lean_object* v___y_3861_; lean_object* v___y_3862_; lean_object* v___y_3869_; lean_object* v___y_3870_; lean_object* v___y_3871_; lean_object* v___y_3872_; lean_object* v___y_3873_; lean_object* v___y_3874_; lean_object* v___y_3895_; lean_object* v___y_3896_; lean_object* v___y_3897_; lean_object* v___y_3898_; lean_object* v___y_3906_; lean_object* v___y_3907_; lean_object* v___y_3908_; lean_object* v___y_3909_; lean_object* v___x_3932_; lean_object* v_a_3933_; uint8_t v___x_3934_; 
v_cls_3856_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn___closed__2_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_));
v___f_3857_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkRichHCongr___closed__0));
v___x_3932_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__0(v_cls_3856_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
v_a_3933_ = lean_ctor_get(v___x_3932_, 0);
lean_inc(v_a_3933_);
lean_dec_ref(v___x_3932_);
v___x_3934_ = lean_unbox(v_a_3933_);
lean_dec(v_a_3933_);
if (v___x_3934_ == 0)
{
v___y_3906_ = v_a_3851_;
v___y_3907_ = v_a_3852_;
v___y_3908_ = v_a_3853_;
v___y_3909_ = v_a_3854_;
goto v___jp_3905_;
}
else
{
lean_object* v___x_3935_; lean_object* v___x_3936_; lean_object* v___x_3937_; lean_object* v___x_3938_; 
v___x_3935_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___closed__8, &lp_mathlib_Lean_Meta_mkRichHCongr___closed__8_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__8);
lean_inc_ref(v_fType_3846_);
v___x_3936_ = l_Lean_MessageData_ofExpr(v_fType_3846_);
v___x_3937_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3937_, 0, v___x_3935_);
lean_ctor_set(v___x_3937_, 1, v___x_3936_);
v___x_3938_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3856_, v___x_3937_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
if (lean_obj_tag(v___x_3938_) == 0)
{
lean_dec_ref_known(v___x_3938_, 1);
v___y_3906_ = v_a_3851_;
v___y_3907_ = v_a_3852_;
v___y_3908_ = v_a_3853_;
v___y_3909_ = v_a_3854_;
goto v___jp_3905_;
}
else
{
lean_object* v_a_3939_; lean_object* v___x_3941_; uint8_t v_isShared_3942_; uint8_t v_isSharedCheck_3946_; 
lean_dec_ref(v_fixedParams_3849_);
lean_dec_ref(v_info_3847_);
lean_dec_ref(v_fType_3846_);
v_a_3939_ = lean_ctor_get(v___x_3938_, 0);
v_isSharedCheck_3946_ = !lean_is_exclusive(v___x_3938_);
if (v_isSharedCheck_3946_ == 0)
{
v___x_3941_ = v___x_3938_;
v_isShared_3942_ = v_isSharedCheck_3946_;
goto v_resetjp_3940_;
}
else
{
lean_inc(v_a_3939_);
lean_dec(v___x_3938_);
v___x_3941_ = lean_box(0);
v_isShared_3942_ = v_isSharedCheck_3946_;
goto v_resetjp_3940_;
}
v_resetjp_3940_:
{
lean_object* v___x_3944_; 
if (v_isShared_3942_ == 0)
{
v___x_3944_ = v___x_3941_;
goto v_reusejp_3943_;
}
else
{
lean_object* v_reuseFailAlloc_3945_; 
v_reuseFailAlloc_3945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3945_, 0, v_a_3939_);
v___x_3944_ = v_reuseFailAlloc_3945_;
goto v_reusejp_3943_;
}
v_reusejp_3943_:
{
return v___x_3944_;
}
}
}
}
v___jp_3858_:
{
lean_object* v___x_3863_; lean_object* v___x_3864_; lean_object* v___x_3865_; lean_object* v___f_3866_; lean_object* v___x_3867_; 
v___x_3863_ = l_Lean_Meta_FunInfo_getArity(v_info_3847_);
v___x_3864_ = lean_box(v_fixedFun_3848_);
v___x_3865_ = lean_box(v_forceHEq_3850_);
lean_inc_ref(v_fType_3846_);
lean_inc(v___x_3863_);
v___f_3866_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mkRichHCongr___lam__5___boxed), 15, 7);
lean_closure_set(v___f_3866_, 0, v___f_3857_);
lean_closure_set(v___f_3866_, 1, v_cls_3856_);
lean_closure_set(v___f_3866_, 2, v___x_3864_);
lean_closure_set(v___f_3866_, 3, v___x_3863_);
lean_closure_set(v___f_3866_, 4, v___x_3865_);
lean_closure_set(v___f_3866_, 5, v_info_3847_);
lean_closure_set(v___f_3866_, 6, v_fType_3846_);
v___x_3867_ = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_doubleTelescope___redArg(v_fType_3846_, v___x_3863_, v_fixedParams_3849_, v___f_3866_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_);
return v___x_3867_;
}
v___jp_3868_:
{
lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v___x_3877_; lean_object* v___x_3878_; lean_object* v___x_3879_; lean_object* v___x_3880_; lean_object* v___x_3881_; lean_object* v___x_3882_; lean_object* v___x_3883_; lean_object* v___x_3884_; lean_object* v___x_3885_; 
lean_inc_ref(v___y_3874_);
v___x_3875_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3875_, 0, v___y_3874_);
v___x_3876_ = l_Lean_MessageData_ofFormat(v___x_3875_);
lean_inc_ref(v___y_3869_);
v___x_3877_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3877_, 0, v___y_3869_);
lean_ctor_set(v___x_3877_, 1, v___x_3876_);
v___x_3878_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___closed__2, &lp_mathlib_Lean_Meta_mkRichHCongr___closed__2_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__2);
v___x_3879_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3879_, 0, v___x_3877_);
lean_ctor_set(v___x_3879_, 1, v___x_3878_);
lean_inc_ref(v_fixedParams_3849_);
v___x_3880_ = lean_array_to_list(v_fixedParams_3849_);
v___x_3881_ = lean_box(0);
v___x_3882_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5(v___x_3880_, v___x_3881_);
v___x_3883_ = l_Lean_MessageData_ofList(v___x_3882_);
v___x_3884_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3884_, 0, v___x_3879_);
lean_ctor_set(v___x_3884_, 1, v___x_3883_);
v___x_3885_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3856_, v___x_3884_, v___y_3872_, v___y_3873_, v___y_3871_, v___y_3870_);
if (lean_obj_tag(v___x_3885_) == 0)
{
lean_dec_ref_known(v___x_3885_, 1);
v___y_3859_ = v___y_3872_;
v___y_3860_ = v___y_3873_;
v___y_3861_ = v___y_3871_;
v___y_3862_ = v___y_3870_;
goto v___jp_3858_;
}
else
{
lean_object* v_a_3886_; lean_object* v___x_3888_; uint8_t v_isShared_3889_; uint8_t v_isSharedCheck_3893_; 
lean_dec_ref(v_fixedParams_3849_);
lean_dec_ref(v_info_3847_);
lean_dec_ref(v_fType_3846_);
v_a_3886_ = lean_ctor_get(v___x_3885_, 0);
v_isSharedCheck_3893_ = !lean_is_exclusive(v___x_3885_);
if (v_isSharedCheck_3893_ == 0)
{
v___x_3888_ = v___x_3885_;
v_isShared_3889_ = v_isSharedCheck_3893_;
goto v_resetjp_3887_;
}
else
{
lean_inc(v_a_3886_);
lean_dec(v___x_3885_);
v___x_3888_ = lean_box(0);
v_isShared_3889_ = v_isSharedCheck_3893_;
goto v_resetjp_3887_;
}
v_resetjp_3887_:
{
lean_object* v___x_3891_; 
if (v_isShared_3889_ == 0)
{
v___x_3891_ = v___x_3888_;
goto v_reusejp_3890_;
}
else
{
lean_object* v_reuseFailAlloc_3892_; 
v_reuseFailAlloc_3892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3892_, 0, v_a_3886_);
v___x_3891_ = v_reuseFailAlloc_3892_;
goto v_reusejp_3890_;
}
v_reusejp_3890_:
{
return v___x_3891_;
}
}
}
}
v___jp_3894_:
{
lean_object* v___x_3899_; lean_object* v_a_3900_; uint8_t v___x_3901_; 
v___x_3899_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__0(v_cls_3856_, v___y_3895_, v___y_3896_, v___y_3897_, v___y_3898_);
v_a_3900_ = lean_ctor_get(v___x_3899_, 0);
lean_inc(v_a_3900_);
lean_dec_ref(v___x_3899_);
v___x_3901_ = lean_unbox(v_a_3900_);
lean_dec(v_a_3900_);
if (v___x_3901_ == 0)
{
v___y_3859_ = v___y_3895_;
v___y_3860_ = v___y_3896_;
v___y_3861_ = v___y_3897_;
v___y_3862_ = v___y_3898_;
goto v___jp_3858_;
}
else
{
lean_object* v___x_3902_; 
v___x_3902_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___closed__4, &lp_mathlib_Lean_Meta_mkRichHCongr___closed__4_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__4);
if (v_fixedFun_3848_ == 0)
{
lean_object* v___x_3903_; 
v___x_3903_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__0));
v___y_3869_ = v___x_3902_;
v___y_3870_ = v___y_3898_;
v___y_3871_ = v___y_3897_;
v___y_3872_ = v___y_3895_;
v___y_3873_ = v___y_3896_;
v___y_3874_ = v___x_3903_;
goto v___jp_3868_;
}
else
{
lean_object* v___x_3904_; 
v___x_3904_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__5___closed__1));
v___y_3869_ = v___x_3902_;
v___y_3870_ = v___y_3898_;
v___y_3871_ = v___y_3897_;
v___y_3872_ = v___y_3895_;
v___y_3873_ = v___y_3896_;
v___y_3874_ = v___x_3904_;
goto v___jp_3868_;
}
}
}
v___jp_3905_:
{
lean_object* v___x_3910_; lean_object* v_a_3911_; uint8_t v___x_3912_; 
v___x_3910_ = lp_mathlib_Lean_Meta_mkRichHCongr___lam__0(v_cls_3856_, v___y_3906_, v___y_3907_, v___y_3908_, v___y_3909_);
v_a_3911_ = lean_ctor_get(v___x_3910_, 0);
lean_inc(v_a_3911_);
lean_dec_ref(v___x_3910_);
v___x_3912_ = lean_unbox(v_a_3911_);
lean_dec(v_a_3911_);
if (v___x_3912_ == 0)
{
v___y_3895_ = v___y_3906_;
v___y_3896_ = v___y_3907_;
v___y_3897_ = v___y_3908_;
v___y_3898_ = v___y_3909_;
goto v___jp_3894_;
}
else
{
lean_object* v_paramInfo_3913_; lean_object* v___x_3914_; size_t v_sz_3915_; size_t v___x_3916_; lean_object* v___x_3917_; lean_object* v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; 
v_paramInfo_3913_ = lean_ctor_get(v_info_3847_, 0);
v___x_3914_ = lean_obj_once(&lp_mathlib_Lean_Meta_mkRichHCongr___closed__6, &lp_mathlib_Lean_Meta_mkRichHCongr___closed__6_once, _init_lp_mathlib_Lean_Meta_mkRichHCongr___closed__6);
v_sz_3915_ = lean_array_size(v_paramInfo_3913_);
v___x_3916_ = ((size_t)0ULL);
lean_inc_ref(v_paramInfo_3913_);
v___x_3917_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_mkRichHCongr_spec__7(v_sz_3915_, v___x_3916_, v_paramInfo_3913_);
v___x_3918_ = lean_array_to_list(v___x_3917_);
v___x_3919_ = lean_box(0);
v___x_3920_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_mkRichHCongr_spec__8(v___x_3918_, v___x_3919_);
v___x_3921_ = l_Lean_MessageData_ofList(v___x_3920_);
v___x_3922_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3922_, 0, v___x_3914_);
lean_ctor_set(v___x_3922_, 1, v___x_3921_);
v___x_3923_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_mkRichHCongr_trySolve_spec__0(v_cls_3856_, v___x_3922_, v___y_3906_, v___y_3907_, v___y_3908_, v___y_3909_);
if (lean_obj_tag(v___x_3923_) == 0)
{
lean_dec_ref_known(v___x_3923_, 1);
v___y_3895_ = v___y_3906_;
v___y_3896_ = v___y_3907_;
v___y_3897_ = v___y_3908_;
v___y_3898_ = v___y_3909_;
goto v___jp_3894_;
}
else
{
lean_object* v_a_3924_; lean_object* v___x_3926_; uint8_t v_isShared_3927_; uint8_t v_isSharedCheck_3931_; 
lean_dec_ref(v_fixedParams_3849_);
lean_dec_ref(v_info_3847_);
lean_dec_ref(v_fType_3846_);
v_a_3924_ = lean_ctor_get(v___x_3923_, 0);
v_isSharedCheck_3931_ = !lean_is_exclusive(v___x_3923_);
if (v_isSharedCheck_3931_ == 0)
{
v___x_3926_ = v___x_3923_;
v_isShared_3927_ = v_isSharedCheck_3931_;
goto v_resetjp_3925_;
}
else
{
lean_inc(v_a_3924_);
lean_dec(v___x_3923_);
v___x_3926_ = lean_box(0);
v_isShared_3927_ = v_isSharedCheck_3931_;
goto v_resetjp_3925_;
}
v_resetjp_3925_:
{
lean_object* v___x_3929_; 
if (v_isShared_3927_ == 0)
{
v___x_3929_ = v___x_3926_;
goto v_reusejp_3928_;
}
else
{
lean_object* v_reuseFailAlloc_3930_; 
v_reuseFailAlloc_3930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3930_, 0, v_a_3924_);
v___x_3929_ = v_reuseFailAlloc_3930_;
goto v_reusejp_3928_;
}
v_reusejp_3928_:
{
return v___x_3929_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkRichHCongr___boxed(lean_object* v_fType_3947_, lean_object* v_info_3948_, lean_object* v_fixedFun_3949_, lean_object* v_fixedParams_3950_, lean_object* v_forceHEq_3951_, lean_object* v_a_3952_, lean_object* v_a_3953_, lean_object* v_a_3954_, lean_object* v_a_3955_, lean_object* v_a_3956_){
_start:
{
uint8_t v_fixedFun_boxed_3957_; uint8_t v_forceHEq_boxed_3958_; lean_object* v_res_3959_; 
v_fixedFun_boxed_3957_ = lean_unbox(v_fixedFun_3949_);
v_forceHEq_boxed_3958_ = lean_unbox(v_forceHEq_3951_);
v_res_3959_ = lp_mathlib_Lean_Meta_mkRichHCongr(v_fType_3947_, v_info_3948_, v_fixedFun_boxed_3957_, v_fixedParams_3950_, v_forceHEq_boxed_3958_, v_a_3952_, v_a_3953_, v_a_3954_, v_a_3955_);
lean_dec(v_a_3955_);
lean_dec_ref(v_a_3954_);
lean_dec(v_a_3953_);
lean_dec_ref(v_a_3952_);
return v_res_3959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2(lean_object* v_xs_3960_, lean_object* v_eqs_3961_, lean_object* v_ys_3962_, lean_object* v_range_3963_, lean_object* v_b_3964_, lean_object* v_i_3965_, lean_object* v_hs_3966_, lean_object* v_hl_3967_, lean_object* v___y_3968_, lean_object* v___y_3969_, lean_object* v___y_3970_, lean_object* v___y_3971_){
_start:
{
lean_object* v___x_3973_; 
v___x_3973_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___redArg(v_xs_3960_, v_eqs_3961_, v_ys_3962_, v_range_3963_, v_b_3964_, v_i_3965_);
return v___x_3973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2___boxed(lean_object* v_xs_3974_, lean_object* v_eqs_3975_, lean_object* v_ys_3976_, lean_object* v_range_3977_, lean_object* v_b_3978_, lean_object* v_i_3979_, lean_object* v_hs_3980_, lean_object* v_hl_3981_, lean_object* v___y_3982_, lean_object* v___y_3983_, lean_object* v___y_3984_, lean_object* v___y_3985_, lean_object* v___y_3986_){
_start:
{
lean_object* v_res_3987_; 
v_res_3987_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__2(v_xs_3974_, v_eqs_3975_, v_ys_3976_, v_range_3977_, v_b_3978_, v_i_3979_, v_hs_3980_, v_hl_3981_, v___y_3982_, v___y_3983_, v___y_3984_, v___y_3985_);
lean_dec(v___y_3985_);
lean_dec_ref(v___y_3984_);
lean_dec(v___y_3983_);
lean_dec_ref(v___y_3982_);
lean_dec_ref(v_range_3977_);
lean_dec_ref(v_ys_3976_);
lean_dec_ref(v_eqs_3975_);
lean_dec_ref(v_xs_3974_);
return v_res_3987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3(lean_object* v_xs_3988_, lean_object* v_eqs_3989_, lean_object* v___x_3990_, lean_object* v_ys_3991_, lean_object* v_kinds_3992_, lean_object* v_range_3993_, lean_object* v_b_3994_, lean_object* v_i_3995_, lean_object* v_hs_3996_, lean_object* v_hl_3997_, lean_object* v___y_3998_, lean_object* v___y_3999_, lean_object* v___y_4000_, lean_object* v___y_4001_){
_start:
{
lean_object* v___x_4003_; 
v___x_4003_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___redArg(v_xs_3988_, v_eqs_3989_, v___x_3990_, v_ys_3991_, v_kinds_3992_, v_range_3993_, v_b_3994_, v_i_3995_, v___y_3998_, v___y_3999_, v___y_4000_, v___y_4001_);
return v___x_4003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3___boxed(lean_object* v_xs_4004_, lean_object* v_eqs_4005_, lean_object* v___x_4006_, lean_object* v_ys_4007_, lean_object* v_kinds_4008_, lean_object* v_range_4009_, lean_object* v_b_4010_, lean_object* v_i_4011_, lean_object* v_hs_4012_, lean_object* v_hl_4013_, lean_object* v___y_4014_, lean_object* v___y_4015_, lean_object* v___y_4016_, lean_object* v___y_4017_, lean_object* v___y_4018_){
_start:
{
lean_object* v_res_4019_; 
v_res_4019_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_mkRichHCongr_spec__3(v_xs_4004_, v_eqs_4005_, v___x_4006_, v_ys_4007_, v_kinds_4008_, v_range_4009_, v_b_4010_, v_i_4011_, v_hs_4012_, v_hl_4013_, v___y_4014_, v___y_4015_, v___y_4016_, v___y_4017_);
lean_dec(v___y_4017_);
lean_dec_ref(v___y_4016_);
lean_dec(v___y_4015_);
lean_dec_ref(v___y_4014_);
lean_dec_ref(v_range_4009_);
lean_dec_ref(v_kinds_4008_);
lean_dec_ref(v_ys_4007_);
lean_dec_ref(v_eqs_4005_);
lean_dec_ref(v_xs_4004_);
return v_res_4019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Array_repr___at___00Lean_Meta_mkRichHCongr_spec__4_spec__5(lean_object* v_a_4020_){
_start:
{
lean_object* v___x_4021_; 
v___x_4021_ = lean_nat_to_int(v_a_4020_);
return v___x_4021_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_CongrTheorems(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Refl(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Refl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Lean_Meta_CongrTheorems_0__Lean_Meta_initFn_00___x40_Mathlib_Lean_Meta_CongrTheorems_3122272532____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Refl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(uint8_t builtin);
lean_object* initialize_Lean_Meta_CongrTheorems(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Refl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_IsEmpty_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta_CongrTheorems(builtin);
}
#ifdef __cplusplus
}
#endif
