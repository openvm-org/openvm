// Lean compiler output
// Module: Mathlib.Lean.Meta.RefinedDiscrTree.Encode
// Imports: public import Init public meta import Init public import Mathlib.Lean.Meta.RefinedDiscrTree.Basic public import Lean.Meta.DiscrTree public import Lean.Meta.LazyDiscrTree
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_DiscrTree_hasNoindexAnnotation(lean_object*);
lean_object* l_Lean_Meta_withLocalDecl___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_DiscrTree_reduce___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(uint8_t, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_reduce(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_LazyDiscrTree_MatchClone_toNatLit_x3f(lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_instInhabitedForall___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isOutParam(lean_object*);
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev_range(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_throwFunctionExpected___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverseAux___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_isClass(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findIdx_x3f_go___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findIdx_x3f_go___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Mathlib.Lean.Meta.RefinedDiscrTree.Encode"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 99, .m_capacity = 99, .m_length = 98, .m_data = "_private.Mathlib.Lean.Meta.RefinedDiscrTree.Encode.0.Lean.Meta.RefinedDiscrTree.encodingStepAux.go"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go_fold(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "Lean.Meta.RefinedDiscrTree.LazyEntry.toList"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 67, .m_capacity = 67, .m_length = 66, .m_data = "`evalLazyEntry` with `eta := false` can only give a singleton list"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__1(lean_object* v_xs_1_, lean_object* v_v_2_, lean_object* v_i_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
v___x_4_ = lean_array_get_size(v_xs_1_);
v___x_5_ = lean_nat_dec_lt(v_i_3_, v___x_4_);
if (v___x_5_ == 0)
{
lean_object* v___x_6_; 
lean_dec(v_i_3_);
v___x_6_ = lean_box(0);
return v___x_6_;
}
else
{
lean_object* v___x_7_; uint8_t v___x_8_; 
v___x_7_ = lean_array_fget_borrowed(v_xs_1_, v_i_3_);
v___x_8_ = l_Lean_instBEqMVarId_beq(v___x_7_, v_v_2_);
if (v___x_8_ == 0)
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lean_unsigned_to_nat(1u);
v___x_10_ = lean_nat_add(v_i_3_, v___x_9_);
lean_dec(v_i_3_);
v_i_3_ = v___x_10_;
goto _start;
}
else
{
lean_object* v___x_12_; 
v___x_12_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_12_, 0, v_i_3_);
return v___x_12_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__1___boxed(lean_object* v_xs_13_, lean_object* v_v_14_, lean_object* v_i_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__1(v_xs_13_, v_v_14_, v_i_15_);
lean_dec(v_v_14_);
lean_dec_ref(v_xs_13_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(lean_object* v_xs_17_, lean_object* v_v_18_){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_unsigned_to_nat(0u);
v___x_20_ = lp_mathlib_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0_spec__1(v_xs_17_, v_v_18_, v___x_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0___boxed(lean_object* v_xs_21_, lean_object* v_v_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(v_xs_21_, v_v_22_);
lean_dec(v_v_22_);
lean_dec_ref(v_xs_21_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(lean_object* v_xs_24_, lean_object* v_v_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Array_finIdxOf_x3f___at___00Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0_spec__0(v_xs_24_, v_v_25_);
if (lean_obj_tag(v___x_26_) == 0)
{
lean_object* v___x_27_; 
v___x_27_ = lean_box(0);
return v___x_27_;
}
else
{
lean_object* v_val_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_35_; 
v_val_28_ = lean_ctor_get(v___x_26_, 0);
v_isSharedCheck_35_ = !lean_is_exclusive(v___x_26_);
if (v_isSharedCheck_35_ == 0)
{
v___x_30_ = v___x_26_;
v_isShared_31_ = v_isSharedCheck_35_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_val_28_);
lean_dec(v___x_26_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_35_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v___x_33_; 
if (v_isShared_31_ == 0)
{
v___x_33_ = v___x_30_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v_val_28_);
v___x_33_ = v_reuseFailAlloc_34_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
return v___x_33_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0___boxed(lean_object* v_xs_36_, lean_object* v_v_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(v_xs_36_, v_v_37_);
lean_dec(v_v_37_);
lean_dec_ref(v_xs_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(lean_object* v_mvarId_39_, lean_object* v_a_40_){
_start:
{
lean_object* v_labelledStars_x3f_42_; 
v_labelledStars_x3f_42_ = lean_ctor_get(v_a_40_, 3);
lean_inc(v_labelledStars_x3f_42_);
if (lean_obj_tag(v_labelledStars_x3f_42_) == 1)
{
lean_object* v_previous_43_; lean_object* v_stack_44_; lean_object* v_mctx_45_; lean_object* v_computedKeys_46_; lean_object* v_val_47_; lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_82_; 
v_previous_43_ = lean_ctor_get(v_a_40_, 0);
v_stack_44_ = lean_ctor_get(v_a_40_, 1);
v_mctx_45_ = lean_ctor_get(v_a_40_, 2);
v_computedKeys_46_ = lean_ctor_get(v_a_40_, 4);
v_val_47_ = lean_ctor_get(v_labelledStars_x3f_42_, 0);
v_isSharedCheck_82_ = !lean_is_exclusive(v_labelledStars_x3f_42_);
if (v_isSharedCheck_82_ == 0)
{
v___x_49_ = v_labelledStars_x3f_42_;
v_isShared_50_ = v_isSharedCheck_82_;
goto v_resetjp_48_;
}
else
{
lean_inc(v_val_47_);
lean_dec(v_labelledStars_x3f_42_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_82_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Array_idxOf_x3f___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar_spec__0(v_val_47_, v_mvarId_39_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v___x_53_; uint8_t v_isShared_54_; uint8_t v_isSharedCheck_66_; 
lean_inc(v_computedKeys_46_);
lean_inc_ref(v_mctx_45_);
lean_inc(v_stack_44_);
lean_inc(v_previous_43_);
v_isSharedCheck_66_ = !lean_is_exclusive(v_a_40_);
if (v_isSharedCheck_66_ == 0)
{
lean_object* v_unused_67_; lean_object* v_unused_68_; lean_object* v_unused_69_; lean_object* v_unused_70_; lean_object* v_unused_71_; 
v_unused_67_ = lean_ctor_get(v_a_40_, 4);
lean_dec(v_unused_67_);
v_unused_68_ = lean_ctor_get(v_a_40_, 3);
lean_dec(v_unused_68_);
v_unused_69_ = lean_ctor_get(v_a_40_, 2);
lean_dec(v_unused_69_);
v_unused_70_ = lean_ctor_get(v_a_40_, 1);
lean_dec(v_unused_70_);
v_unused_71_ = lean_ctor_get(v_a_40_, 0);
lean_dec(v_unused_71_);
v___x_53_ = v_a_40_;
v_isShared_54_ = v_isSharedCheck_66_;
goto v_resetjp_52_;
}
else
{
lean_dec(v_a_40_);
v___x_53_ = lean_box(0);
v_isShared_54_ = v_isSharedCheck_66_;
goto v_resetjp_52_;
}
v_resetjp_52_:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_59_; 
v___x_55_ = lean_array_get_size(v_val_47_);
v___x_56_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
v___x_57_ = lean_array_push(v_val_47_, v_mvarId_39_);
if (v_isShared_50_ == 0)
{
lean_ctor_set(v___x_49_, 0, v___x_57_);
v___x_59_ = v___x_49_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v___x_57_);
v___x_59_ = v_reuseFailAlloc_65_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
lean_object* v___x_61_; 
if (v_isShared_54_ == 0)
{
lean_ctor_set(v___x_53_, 3, v___x_59_);
v___x_61_ = v___x_53_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_64_; 
v_reuseFailAlloc_64_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_64_, 0, v_previous_43_);
lean_ctor_set(v_reuseFailAlloc_64_, 1, v_stack_44_);
lean_ctor_set(v_reuseFailAlloc_64_, 2, v_mctx_45_);
lean_ctor_set(v_reuseFailAlloc_64_, 3, v___x_59_);
lean_ctor_set(v_reuseFailAlloc_64_, 4, v_computedKeys_46_);
v___x_61_ = v_reuseFailAlloc_64_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_56_);
lean_ctor_set(v___x_62_, 1, v___x_61_);
v___x_63_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
return v___x_63_;
}
}
}
}
else
{
lean_object* v_val_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_81_; 
lean_del_object(v___x_49_);
lean_dec(v_val_47_);
lean_dec(v_mvarId_39_);
v_val_72_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_81_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_81_ == 0)
{
v___x_74_ = v___x_51_;
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_val_72_);
lean_dec(v___x_51_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_81_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v___x_77_; 
if (v_isShared_75_ == 0)
{
v___x_77_ = v___x_74_;
goto v_reusejp_76_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v_val_72_);
v___x_77_ = v_reuseFailAlloc_80_;
goto v_reusejp_76_;
}
v_reusejp_76_:
{
lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_77_);
lean_ctor_set(v___x_78_, 1, v_a_40_);
v___x_79_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
return v___x_79_;
}
}
}
}
}
else
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
lean_dec(v_labelledStars_x3f_42_);
lean_dec(v_mvarId_39_);
v___x_83_ = lean_box(0);
v___x_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_40_);
v___x_85_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
return v___x_85_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg___boxed(lean_object* v_mvarId_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(v_mvarId_86_, v_a_87_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar(lean_object* v_mvarId_90_, lean_object* v_a_91_, lean_object* v_a_92_, lean_object* v_a_93_, lean_object* v_a_94_, lean_object* v_a_95_, lean_object* v_a_96_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(v_mvarId_90_, v_a_92_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___boxed(lean_object* v_mvarId_99_, lean_object* v_a_100_, lean_object* v_a_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar(v_mvarId_99_, v_a_100_, v_a_101_, v_a_102_, v_a_103_, v_a_104_, v_a_105_);
lean_dec(v_a_105_);
lean_dec_ref(v_a_104_);
lean_dec(v_a_103_);
lean_dec_ref(v_a_102_);
lean_dec(v_a_100_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0(lean_object* v_x_108_, lean_object* v_x_109_){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_110_ = lean_box(8);
v___x_111_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
lean_ctor_set(v___x_111_, 1, v_x_108_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0___boxed(lean_object* v_x_112_, lean_object* v_x_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___lam__0(v_x_112_, v_x_113_);
lean_dec(v_x_113_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg(lean_object* v_lambdas_116_, lean_object* v_key_117_, lean_object* v_a_118_){
_start:
{
if (lean_obj_tag(v_lambdas_116_) == 0)
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v_key_117_);
lean_ctor_set(v___x_120_, 1, v_a_118_);
v___x_121_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
return v___x_121_;
}
else
{
lean_object* v_tail_122_; lean_object* v___x_124_; uint8_t v_isShared_125_; uint8_t v_isSharedCheck_147_; 
v_tail_122_ = lean_ctor_get(v_lambdas_116_, 1);
v_isSharedCheck_147_ = !lean_is_exclusive(v_lambdas_116_);
if (v_isSharedCheck_147_ == 0)
{
lean_object* v_unused_148_; 
v_unused_148_ = lean_ctor_get(v_lambdas_116_, 0);
lean_dec(v_unused_148_);
v___x_124_ = v_lambdas_116_;
v_isShared_125_ = v_isSharedCheck_147_;
goto v_resetjp_123_;
}
else
{
lean_inc(v_tail_122_);
lean_dec(v_lambdas_116_);
v___x_124_ = lean_box(0);
v_isShared_125_ = v_isSharedCheck_147_;
goto v_resetjp_123_;
}
v_resetjp_123_:
{
lean_object* v_previous_126_; lean_object* v_stack_127_; lean_object* v_mctx_128_; lean_object* v_labelledStars_x3f_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_145_; 
v_previous_126_ = lean_ctor_get(v_a_118_, 0);
v_stack_127_ = lean_ctor_get(v_a_118_, 1);
v_mctx_128_ = lean_ctor_get(v_a_118_, 2);
v_labelledStars_x3f_129_ = lean_ctor_get(v_a_118_, 3);
v_isSharedCheck_145_ = !lean_is_exclusive(v_a_118_);
if (v_isSharedCheck_145_ == 0)
{
lean_object* v_unused_146_; 
v_unused_146_ = lean_ctor_get(v_a_118_, 4);
lean_dec(v_unused_146_);
v___x_131_ = v_a_118_;
v_isShared_132_ = v_isSharedCheck_145_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_labelledStars_x3f_129_);
lean_inc(v_mctx_128_);
lean_inc(v_stack_127_);
lean_inc(v_previous_126_);
lean_dec(v_a_118_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_145_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
lean_object* v___f_133_; lean_object* v___x_134_; lean_object* v___x_136_; 
v___f_133_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___closed__0));
v___x_134_ = lean_box(0);
if (v_isShared_125_ == 0)
{
lean_ctor_set(v___x_124_, 1, v___x_134_);
lean_ctor_set(v___x_124_, 0, v_key_117_);
v___x_136_ = v___x_124_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v_key_117_);
lean_ctor_set(v_reuseFailAlloc_144_, 1, v___x_134_);
v___x_136_ = v_reuseFailAlloc_144_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
lean_object* v___x_137_; lean_object* v___x_139_; 
v___x_137_ = l_List_foldl___redArg(v___f_133_, v___x_136_, v_tail_122_);
if (v_isShared_132_ == 0)
{
lean_ctor_set(v___x_131_, 4, v___x_137_);
v___x_139_ = v___x_131_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v_previous_126_);
lean_ctor_set(v_reuseFailAlloc_143_, 1, v_stack_127_);
lean_ctor_set(v_reuseFailAlloc_143_, 2, v_mctx_128_);
lean_ctor_set(v_reuseFailAlloc_143_, 3, v_labelledStars_x3f_129_);
lean_ctor_set(v_reuseFailAlloc_143_, 4, v___x_137_);
v___x_139_ = v_reuseFailAlloc_143_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_140_ = lean_box(8);
v___x_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_140_);
lean_ctor_set(v___x_141_, 1, v___x_139_);
v___x_142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
return v___x_142_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___boxed(lean_object* v_lambdas_149_, lean_object* v_key_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg(v_lambdas_149_, v_key_150_, v_a_151_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams(lean_object* v_lambdas_154_, lean_object* v_key_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_, lean_object* v_a_160_){
_start:
{
if (lean_obj_tag(v_lambdas_154_) == 0)
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_162_, 0, v_key_155_);
lean_ctor_set(v___x_162_, 1, v_a_156_);
v___x_163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
return v___x_163_;
}
else
{
lean_object* v_tail_164_; lean_object* v___x_166_; uint8_t v_isShared_167_; uint8_t v_isSharedCheck_189_; 
v_tail_164_ = lean_ctor_get(v_lambdas_154_, 1);
v_isSharedCheck_189_ = !lean_is_exclusive(v_lambdas_154_);
if (v_isSharedCheck_189_ == 0)
{
lean_object* v_unused_190_; 
v_unused_190_ = lean_ctor_get(v_lambdas_154_, 0);
lean_dec(v_unused_190_);
v___x_166_ = v_lambdas_154_;
v_isShared_167_ = v_isSharedCheck_189_;
goto v_resetjp_165_;
}
else
{
lean_inc(v_tail_164_);
lean_dec(v_lambdas_154_);
v___x_166_ = lean_box(0);
v_isShared_167_ = v_isSharedCheck_189_;
goto v_resetjp_165_;
}
v_resetjp_165_:
{
lean_object* v_previous_168_; lean_object* v_stack_169_; lean_object* v_mctx_170_; lean_object* v_labelledStars_x3f_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_187_; 
v_previous_168_ = lean_ctor_get(v_a_156_, 0);
v_stack_169_ = lean_ctor_get(v_a_156_, 1);
v_mctx_170_ = lean_ctor_get(v_a_156_, 2);
v_labelledStars_x3f_171_ = lean_ctor_get(v_a_156_, 3);
v_isSharedCheck_187_ = !lean_is_exclusive(v_a_156_);
if (v_isSharedCheck_187_ == 0)
{
lean_object* v_unused_188_; 
v_unused_188_ = lean_ctor_get(v_a_156_, 4);
lean_dec(v_unused_188_);
v___x_173_ = v_a_156_;
v_isShared_174_ = v_isSharedCheck_187_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_labelledStars_x3f_171_);
lean_inc(v_mctx_170_);
lean_inc(v_stack_169_);
lean_inc(v_previous_168_);
lean_dec(v_a_156_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_187_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___f_175_; lean_object* v___x_176_; lean_object* v___x_178_; 
v___f_175_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___closed__0));
v___x_176_ = lean_box(0);
if (v_isShared_167_ == 0)
{
lean_ctor_set(v___x_166_, 1, v___x_176_);
lean_ctor_set(v___x_166_, 0, v_key_155_);
v___x_178_ = v___x_166_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v_key_155_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v___x_176_);
v___x_178_ = v_reuseFailAlloc_186_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
lean_object* v___x_179_; lean_object* v___x_181_; 
v___x_179_ = l_List_foldl___redArg(v___f_175_, v___x_178_, v_tail_164_);
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 4, v___x_179_);
v___x_181_ = v___x_173_;
goto v_reusejp_180_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v_previous_168_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v_stack_169_);
lean_ctor_set(v_reuseFailAlloc_185_, 2, v_mctx_170_);
lean_ctor_set(v_reuseFailAlloc_185_, 3, v_labelledStars_x3f_171_);
lean_ctor_set(v_reuseFailAlloc_185_, 4, v___x_179_);
v___x_181_ = v_reuseFailAlloc_185_;
goto v_reusejp_180_;
}
v_reusejp_180_:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = lean_box(8);
v___x_183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_182_);
lean_ctor_set(v___x_183_, 1, v___x_181_);
v___x_184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
return v___x_184_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___boxed(lean_object* v_lambdas_191_, lean_object* v_key_192_, lean_object* v_a_193_, lean_object* v_a_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams(v_lambdas_191_, v_key_192_, v_a_193_, v_a_194_, v_a_195_, v_a_196_, v_a_197_);
lean_dec(v_a_197_);
lean_dec_ref(v_a_196_);
lean_dec(v_a_195_);
lean_dec_ref(v_a_194_);
return v_res_199_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__0(void){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = l_instMonadEIO(lean_box(0));
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1(lean_object* v_msg_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v_toApplicative_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_287_; 
v___x_213_ = lean_obj_once(&lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__0, &lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__0_once, _init_lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__0);
v___x_214_ = l_StateRefT_x27_instMonad___redArg(v___x_213_);
v_toApplicative_215_ = lean_ctor_get(v___x_214_, 0);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_287_ == 0)
{
lean_object* v_unused_288_; 
v_unused_288_ = lean_ctor_get(v___x_214_, 1);
lean_dec(v_unused_288_);
v___x_217_ = v___x_214_;
v_isShared_218_ = v_isSharedCheck_287_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_toApplicative_215_);
lean_dec(v___x_214_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_287_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v_toFunctor_219_; lean_object* v_toSeq_220_; lean_object* v_toSeqLeft_221_; lean_object* v_toSeqRight_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_285_; 
v_toFunctor_219_ = lean_ctor_get(v_toApplicative_215_, 0);
v_toSeq_220_ = lean_ctor_get(v_toApplicative_215_, 2);
v_toSeqLeft_221_ = lean_ctor_get(v_toApplicative_215_, 3);
v_toSeqRight_222_ = lean_ctor_get(v_toApplicative_215_, 4);
v_isSharedCheck_285_ = !lean_is_exclusive(v_toApplicative_215_);
if (v_isSharedCheck_285_ == 0)
{
lean_object* v_unused_286_; 
v_unused_286_ = lean_ctor_get(v_toApplicative_215_, 1);
lean_dec(v_unused_286_);
v___x_224_ = v_toApplicative_215_;
v_isShared_225_ = v_isSharedCheck_285_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_toSeqRight_222_);
lean_inc(v_toSeqLeft_221_);
lean_inc(v_toSeq_220_);
lean_inc(v_toFunctor_219_);
lean_dec(v_toApplicative_215_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_285_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___f_226_; lean_object* v___f_227_; lean_object* v___f_228_; lean_object* v___f_229_; lean_object* v___x_230_; lean_object* v___f_231_; lean_object* v___f_232_; lean_object* v___f_233_; lean_object* v___x_235_; 
v___f_226_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__1));
v___f_227_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__2));
lean_inc_ref(v_toFunctor_219_);
v___f_228_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_228_, 0, v_toFunctor_219_);
v___f_229_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_229_, 0, v_toFunctor_219_);
v___x_230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_230_, 0, v___f_228_);
lean_ctor_set(v___x_230_, 1, v___f_229_);
v___f_231_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_231_, 0, v_toSeqRight_222_);
v___f_232_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_232_, 0, v_toSeqLeft_221_);
v___f_233_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_233_, 0, v_toSeq_220_);
if (v_isShared_225_ == 0)
{
lean_ctor_set(v___x_224_, 4, v___f_231_);
lean_ctor_set(v___x_224_, 3, v___f_232_);
lean_ctor_set(v___x_224_, 2, v___f_233_);
lean_ctor_set(v___x_224_, 1, v___f_226_);
lean_ctor_set(v___x_224_, 0, v___x_230_);
v___x_235_ = v___x_224_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v___x_230_);
lean_ctor_set(v_reuseFailAlloc_284_, 1, v___f_226_);
lean_ctor_set(v_reuseFailAlloc_284_, 2, v___f_233_);
lean_ctor_set(v_reuseFailAlloc_284_, 3, v___f_232_);
lean_ctor_set(v_reuseFailAlloc_284_, 4, v___f_231_);
v___x_235_ = v_reuseFailAlloc_284_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
lean_object* v___x_237_; 
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 1, v___f_227_);
lean_ctor_set(v___x_217_, 0, v___x_235_);
v___x_237_ = v___x_217_;
goto v_reusejp_236_;
}
else
{
lean_object* v_reuseFailAlloc_283_; 
v_reuseFailAlloc_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_283_, 0, v___x_235_);
lean_ctor_set(v_reuseFailAlloc_283_, 1, v___f_227_);
v___x_237_ = v_reuseFailAlloc_283_;
goto v_reusejp_236_;
}
v_reusejp_236_:
{
lean_object* v___x_238_; lean_object* v_toApplicative_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_281_; 
v___x_238_ = l_StateRefT_x27_instMonad___redArg(v___x_237_);
v_toApplicative_239_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_281_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_281_ == 0)
{
lean_object* v_unused_282_; 
v_unused_282_ = lean_ctor_get(v___x_238_, 1);
lean_dec(v_unused_282_);
v___x_241_ = v___x_238_;
v_isShared_242_ = v_isSharedCheck_281_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_toApplicative_239_);
lean_dec(v___x_238_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_281_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v_toFunctor_243_; lean_object* v_toSeq_244_; lean_object* v_toSeqLeft_245_; lean_object* v_toSeqRight_246_; lean_object* v___x_248_; uint8_t v_isShared_249_; uint8_t v_isSharedCheck_279_; 
v_toFunctor_243_ = lean_ctor_get(v_toApplicative_239_, 0);
v_toSeq_244_ = lean_ctor_get(v_toApplicative_239_, 2);
v_toSeqLeft_245_ = lean_ctor_get(v_toApplicative_239_, 3);
v_toSeqRight_246_ = lean_ctor_get(v_toApplicative_239_, 4);
v_isSharedCheck_279_ = !lean_is_exclusive(v_toApplicative_239_);
if (v_isSharedCheck_279_ == 0)
{
lean_object* v_unused_280_; 
v_unused_280_ = lean_ctor_get(v_toApplicative_239_, 1);
lean_dec(v_unused_280_);
v___x_248_ = v_toApplicative_239_;
v_isShared_249_ = v_isSharedCheck_279_;
goto v_resetjp_247_;
}
else
{
lean_inc(v_toSeqRight_246_);
lean_inc(v_toSeqLeft_245_);
lean_inc(v_toSeq_244_);
lean_inc(v_toFunctor_243_);
lean_dec(v_toApplicative_239_);
v___x_248_ = lean_box(0);
v_isShared_249_ = v_isSharedCheck_279_;
goto v_resetjp_247_;
}
v_resetjp_247_:
{
lean_object* v___f_250_; lean_object* v___f_251_; lean_object* v___f_252_; lean_object* v___f_253_; lean_object* v___x_254_; lean_object* v___f_255_; lean_object* v___f_256_; lean_object* v___f_257_; lean_object* v___x_259_; 
v___f_250_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__3));
v___f_251_ = ((lean_object*)(lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___closed__4));
lean_inc_ref(v_toFunctor_243_);
v___f_252_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_252_, 0, v_toFunctor_243_);
v___f_253_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_253_, 0, v_toFunctor_243_);
v___x_254_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_254_, 0, v___f_252_);
lean_ctor_set(v___x_254_, 1, v___f_253_);
v___f_255_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_255_, 0, v_toSeqRight_246_);
v___f_256_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_256_, 0, v_toSeqLeft_245_);
v___f_257_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_257_, 0, v_toSeq_244_);
if (v_isShared_249_ == 0)
{
lean_ctor_set(v___x_248_, 4, v___f_255_);
lean_ctor_set(v___x_248_, 3, v___f_256_);
lean_ctor_set(v___x_248_, 2, v___f_257_);
lean_ctor_set(v___x_248_, 1, v___f_250_);
lean_ctor_set(v___x_248_, 0, v___x_254_);
v___x_259_ = v___x_248_;
goto v_reusejp_258_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_254_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v___f_250_);
lean_ctor_set(v_reuseFailAlloc_278_, 2, v___f_257_);
lean_ctor_set(v_reuseFailAlloc_278_, 3, v___f_256_);
lean_ctor_set(v_reuseFailAlloc_278_, 4, v___f_255_);
v___x_259_ = v_reuseFailAlloc_278_;
goto v_reusejp_258_;
}
v_reusejp_258_:
{
lean_object* v___x_261_; 
if (v_isShared_242_ == 0)
{
lean_ctor_set(v___x_241_, 1, v___f_251_);
lean_ctor_set(v___x_241_, 0, v___x_259_);
v___x_261_ = v___x_241_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v___x_259_);
lean_ctor_set(v_reuseFailAlloc_277_, 1, v___f_251_);
v___x_261_ = v_reuseFailAlloc_277_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
lean_object* v___f_262_; lean_object* v___f_263_; lean_object* v___f_264_; lean_object* v___f_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___f_274_; lean_object* v___x_9482__overap_275_; lean_object* v___x_276_; 
lean_inc_ref_n(v___x_261_, 6);
v___f_262_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_262_, 0, v___x_261_);
v___f_263_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_263_, 0, v___x_261_);
v___f_264_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_264_, 0, v___x_261_);
v___f_265_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_265_, 0, v___x_261_);
v___x_266_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_266_, 0, lean_box(0));
lean_closure_set(v___x_266_, 1, lean_box(0));
lean_closure_set(v___x_266_, 2, v___x_261_);
v___x_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
lean_ctor_set(v___x_267_, 1, v___f_262_);
v___x_268_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_268_, 0, lean_box(0));
lean_closure_set(v___x_268_, 1, lean_box(0));
lean_closure_set(v___x_268_, 2, v___x_261_);
v___x_269_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_269_, 0, v___x_267_);
lean_ctor_set(v___x_269_, 1, v___x_268_);
lean_ctor_set(v___x_269_, 2, v___f_263_);
lean_ctor_set(v___x_269_, 3, v___f_264_);
lean_ctor_set(v___x_269_, 4, v___f_265_);
v___x_270_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_270_, 0, lean_box(0));
lean_closure_set(v___x_270_, 1, lean_box(0));
lean_closure_set(v___x_270_, 2, v___x_261_);
v___x_271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_269_);
lean_ctor_set(v___x_271_, 1, v___x_270_);
v___x_272_ = lean_box(0);
v___x_273_ = l_instInhabitedOfMonad___redArg(v___x_271_, v___x_272_);
v___f_274_ = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_274_, 0, v___x_273_);
v___x_9482__overap_275_ = lean_panic_fn_borrowed(v___f_274_, v_msg_205_);
lean_dec_ref(v___f_274_);
lean_inc(v___y_211_);
lean_inc_ref(v___y_210_);
lean_inc(v___y_209_);
lean_inc_ref(v___y_208_);
lean_inc(v___y_206_);
v___x_276_ = lean_apply_7(v___x_9482__overap_275_, v___y_206_, v___y_207_, v___y_208_, v___y_209_, v___y_210_, v___y_211_, lean_box(0));
return v___x_276_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1___boxed(lean_object* v_msg_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1(v_msg_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_);
lean_dec(v___y_295_);
lean_dec_ref(v___y_294_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
lean_dec(v___y_290_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(lean_object* v_lambdas_298_, lean_object* v_e_299_, lean_object* v_____do__lift_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_){
_start:
{
lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_308_ = l_List_appendTR___redArg(v_lambdas_298_, v_____do__lift_300_);
v___x_309_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v_e_299_, v___x_308_, v___y_303_);
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v_a_310_; lean_object* v___x_312_; uint8_t v_isShared_313_; uint8_t v_isSharedCheck_332_; 
v_a_310_ = lean_ctor_get(v___x_309_, 0);
v_isSharedCheck_332_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_332_ == 0)
{
v___x_312_ = v___x_309_;
v_isShared_313_ = v_isSharedCheck_332_;
goto v_resetjp_311_;
}
else
{
lean_inc(v_a_310_);
lean_dec(v___x_309_);
v___x_312_ = lean_box(0);
v_isShared_313_ = v_isSharedCheck_332_;
goto v_resetjp_311_;
}
v_resetjp_311_:
{
lean_object* v_stack_314_; lean_object* v_mctx_315_; lean_object* v_labelledStars_x3f_316_; lean_object* v_computedKeys_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_330_; 
v_stack_314_ = lean_ctor_get(v___y_302_, 1);
v_mctx_315_ = lean_ctor_get(v___y_302_, 2);
v_labelledStars_x3f_316_ = lean_ctor_get(v___y_302_, 3);
v_computedKeys_317_ = lean_ctor_get(v___y_302_, 4);
v_isSharedCheck_330_ = !lean_is_exclusive(v___y_302_);
if (v_isSharedCheck_330_ == 0)
{
lean_object* v_unused_331_; 
v_unused_331_ = lean_ctor_get(v___y_302_, 0);
lean_dec(v_unused_331_);
v___x_319_ = v___y_302_;
v_isShared_320_ = v_isSharedCheck_330_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_computedKeys_317_);
lean_inc(v_labelledStars_x3f_316_);
lean_inc(v_mctx_315_);
lean_inc(v_stack_314_);
lean_dec(v___y_302_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_330_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_324_; 
v___x_321_ = lean_box(0);
v___x_322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_322_, 0, v_a_310_);
if (v_isShared_320_ == 0)
{
lean_ctor_set(v___x_319_, 0, v___x_322_);
v___x_324_ = v___x_319_;
goto v_reusejp_323_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v___x_322_);
lean_ctor_set(v_reuseFailAlloc_329_, 1, v_stack_314_);
lean_ctor_set(v_reuseFailAlloc_329_, 2, v_mctx_315_);
lean_ctor_set(v_reuseFailAlloc_329_, 3, v_labelledStars_x3f_316_);
lean_ctor_set(v_reuseFailAlloc_329_, 4, v_computedKeys_317_);
v___x_324_ = v_reuseFailAlloc_329_;
goto v_reusejp_323_;
}
v_reusejp_323_:
{
lean_object* v___x_325_; lean_object* v___x_327_; 
v___x_325_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_321_);
lean_ctor_set(v___x_325_, 1, v___x_324_);
if (v_isShared_313_ == 0)
{
lean_ctor_set(v___x_312_, 0, v___x_325_);
v___x_327_ = v___x_312_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_328_; 
v_reuseFailAlloc_328_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_328_, 0, v___x_325_);
v___x_327_ = v_reuseFailAlloc_328_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
return v___x_327_;
}
}
}
}
}
else
{
lean_object* v_a_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_340_; 
lean_dec_ref(v___y_302_);
v_a_333_ = lean_ctor_get(v___x_309_, 0);
v_isSharedCheck_340_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_340_ == 0)
{
v___x_335_ = v___x_309_;
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_a_333_);
lean_dec(v___x_309_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_340_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_338_; 
if (v_isShared_336_ == 0)
{
v___x_338_ = v___x_335_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v_a_333_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
return v___x_338_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0___boxed(lean_object* v_lambdas_341_, lean_object* v_e_342_, lean_object* v_____do__lift_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(v_lambdas_341_, v_e_342_, v_____do__lift_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
lean_dec(v___y_344_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findIdx_x3f_go___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(lean_object* v_fvarId_352_, lean_object* v_a_353_, lean_object* v_a_354_){
_start:
{
if (lean_obj_tag(v_a_353_) == 0)
{
lean_object* v___x_355_; 
lean_dec(v_a_354_);
v___x_355_ = lean_box(0);
return v___x_355_;
}
else
{
lean_object* v_head_356_; lean_object* v_tail_357_; uint8_t v___x_358_; 
v_head_356_ = lean_ctor_get(v_a_353_, 0);
v_tail_357_ = lean_ctor_get(v_a_353_, 1);
v___x_358_ = l_Lean_instBEqFVarId_beq(v_head_356_, v_fvarId_352_);
if (v___x_358_ == 0)
{
lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_359_ = lean_unsigned_to_nat(1u);
v___x_360_ = lean_nat_add(v_a_354_, v___x_359_);
lean_dec(v_a_354_);
v_a_353_ = v_tail_357_;
v_a_354_ = v___x_360_;
goto _start;
}
else
{
lean_object* v___x_362_; 
v___x_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_362_, 0, v_a_354_);
return v___x_362_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findIdx_x3f_go___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0___boxed(lean_object* v_fvarId_363_, lean_object* v_a_364_, lean_object* v_a_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_List_findIdx_x3f_go___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(v_fvarId_363_, v_a_364_, v_a_365_);
lean_dec(v_a_364_);
lean_dec(v_fvarId_363_);
return v_res_366_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3(void){
_start:
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_370_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__2));
v___x_371_ = lean_unsigned_to_nat(19u);
v___x_372_ = lean_unsigned_to_nat(125u);
v___x_373_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__1));
v___x_374_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0));
v___x_375_ = l_mkPanicMessageWithDecl(v___x_374_, v___x_373_, v___x_372_, v___x_371_, v___x_370_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(lean_object* v_e_376_, lean_object* v_lambdas_377_, uint8_t v_root_378_, lean_object* v_a_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = l_Lean_Expr_getAppFn(v_e_376_);
switch(lean_obj_tag(v___x_386_))
{
case 4:
{
lean_object* v_declName_387_; lean_object* v___y_389_; lean_object* v___y_395_; lean_object* v___y_396_; lean_object* v___y_397_; 
v_declName_387_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_declName_387_);
lean_dec_ref_known(v___x_386_, 2);
if (v_root_378_ == 0)
{
lean_object* v___x_425_; 
lean_inc_ref(v_e_376_);
v___x_425_ = l_Lean_Meta_LazyDiscrTree_MatchClone_toNatLit_x3f(v_e_376_);
if (lean_obj_tag(v___x_425_) == 1)
{
lean_object* v_val_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_435_; 
lean_dec(v_declName_387_);
lean_dec(v_lambdas_377_);
lean_dec_ref(v_e_376_);
v_val_426_ = lean_ctor_get(v___x_425_, 0);
v_isSharedCheck_435_ = !lean_is_exclusive(v___x_425_);
if (v_isSharedCheck_435_ == 0)
{
v___x_428_ = v___x_425_;
v_isShared_429_ = v_isSharedCheck_435_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_val_426_);
lean_dec(v___x_425_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_435_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
lean_object* v___x_431_; 
if (v_isShared_429_ == 0)
{
lean_ctor_set_tag(v___x_428_, 6);
v___x_431_ = v___x_428_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_val_426_);
v___x_431_ = v_reuseFailAlloc_434_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_432_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_432_, 0, v___x_431_);
lean_ctor_set(v___x_432_, 1, v_a_380_);
v___x_433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
return v___x_433_;
}
}
}
else
{
lean_dec(v___x_425_);
v___y_395_ = v_a_379_;
v___y_396_ = v_a_380_;
v___y_397_ = v_a_381_;
goto v___jp_394_;
}
}
else
{
v___y_395_ = v_a_379_;
v___y_396_ = v_a_380_;
v___y_397_ = v_a_381_;
goto v___jp_394_;
}
v___jp_388_:
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; 
v___x_390_ = l_Lean_Expr_getAppNumArgs(v_e_376_);
lean_dec_ref(v_e_376_);
v___x_391_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_391_, 0, v_declName_387_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
lean_ctor_set(v___x_392_, 1, v___y_389_);
v___x_393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_393_, 0, v___x_392_);
return v___x_393_;
}
v___jp_394_:
{
lean_object* v___x_398_; lean_object* v___x_399_; uint8_t v___x_400_; 
v___x_398_ = l_Lean_Expr_getAppNumArgs(v_e_376_);
v___x_399_ = lean_unsigned_to_nat(0u);
v___x_400_ = lean_nat_dec_eq(v___x_398_, v___x_399_);
lean_dec(v___x_398_);
if (v___x_400_ == 0)
{
lean_object* v___x_401_; lean_object* v___x_402_; 
lean_inc(v___y_395_);
v___x_401_ = l_List_appendTR___redArg(v_lambdas_377_, v___y_395_);
lean_inc_ref(v_e_376_);
v___x_402_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v_e_376_, v___x_401_, v___y_397_);
if (lean_obj_tag(v___x_402_) == 0)
{
lean_object* v_a_403_; lean_object* v_stack_404_; lean_object* v_mctx_405_; lean_object* v_labelledStars_x3f_406_; lean_object* v_computedKeys_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_415_; 
v_a_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc(v_a_403_);
lean_dec_ref_known(v___x_402_, 1);
v_stack_404_ = lean_ctor_get(v___y_396_, 1);
v_mctx_405_ = lean_ctor_get(v___y_396_, 2);
v_labelledStars_x3f_406_ = lean_ctor_get(v___y_396_, 3);
v_computedKeys_407_ = lean_ctor_get(v___y_396_, 4);
v_isSharedCheck_415_ = !lean_is_exclusive(v___y_396_);
if (v_isSharedCheck_415_ == 0)
{
lean_object* v_unused_416_; 
v_unused_416_ = lean_ctor_get(v___y_396_, 0);
lean_dec(v_unused_416_);
v___x_409_ = v___y_396_;
v_isShared_410_ = v_isSharedCheck_415_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_computedKeys_407_);
lean_inc(v_labelledStars_x3f_406_);
lean_inc(v_mctx_405_);
lean_inc(v_stack_404_);
lean_dec(v___y_396_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_415_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v___x_411_; lean_object* v___x_413_; 
v___x_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_411_, 0, v_a_403_);
if (v_isShared_410_ == 0)
{
lean_ctor_set(v___x_409_, 0, v___x_411_);
v___x_413_ = v___x_409_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v___x_411_);
lean_ctor_set(v_reuseFailAlloc_414_, 1, v_stack_404_);
lean_ctor_set(v_reuseFailAlloc_414_, 2, v_mctx_405_);
lean_ctor_set(v_reuseFailAlloc_414_, 3, v_labelledStars_x3f_406_);
lean_ctor_set(v_reuseFailAlloc_414_, 4, v_computedKeys_407_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
v___y_389_ = v___x_413_;
goto v___jp_388_;
}
}
}
else
{
lean_object* v_a_417_; lean_object* v___x_419_; uint8_t v_isShared_420_; uint8_t v_isSharedCheck_424_; 
lean_dec_ref(v___y_396_);
lean_dec(v_declName_387_);
lean_dec_ref(v_e_376_);
v_a_417_ = lean_ctor_get(v___x_402_, 0);
v_isSharedCheck_424_ = !lean_is_exclusive(v___x_402_);
if (v_isSharedCheck_424_ == 0)
{
v___x_419_ = v___x_402_;
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
else
{
lean_inc(v_a_417_);
lean_dec(v___x_402_);
v___x_419_ = lean_box(0);
v_isShared_420_ = v_isSharedCheck_424_;
goto v_resetjp_418_;
}
v_resetjp_418_:
{
lean_object* v___x_422_; 
if (v_isShared_420_ == 0)
{
v___x_422_ = v___x_419_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_a_417_);
v___x_422_ = v_reuseFailAlloc_423_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
return v___x_422_;
}
}
}
}
else
{
lean_dec(v_lambdas_377_);
v___y_389_ = v___y_396_;
goto v___jp_388_;
}
}
}
case 11:
{
lean_object* v_typeName_436_; lean_object* v_idx_437_; lean_object* v___x_438_; 
v_typeName_436_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_typeName_436_);
v_idx_437_ = lean_ctor_get(v___x_386_, 1);
lean_inc(v_idx_437_);
lean_dec_ref_known(v___x_386_, 3);
lean_inc(v_a_379_);
lean_inc_ref(v_e_376_);
v___x_438_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(v_lambdas_377_, v_e_376_, v_a_379_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_, v_a_384_);
if (lean_obj_tag(v___x_438_) == 0)
{
lean_object* v_a_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_457_; 
v_a_439_ = lean_ctor_get(v___x_438_, 0);
v_isSharedCheck_457_ = !lean_is_exclusive(v___x_438_);
if (v_isSharedCheck_457_ == 0)
{
v___x_441_ = v___x_438_;
v_isShared_442_ = v_isSharedCheck_457_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_a_439_);
lean_dec(v___x_438_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_457_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v_snd_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_455_; 
v_snd_443_ = lean_ctor_get(v_a_439_, 1);
v_isSharedCheck_455_ = !lean_is_exclusive(v_a_439_);
if (v_isSharedCheck_455_ == 0)
{
lean_object* v_unused_456_; 
v_unused_456_ = lean_ctor_get(v_a_439_, 0);
lean_dec(v_unused_456_);
v___x_445_ = v_a_439_;
v_isShared_446_ = v_isSharedCheck_455_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_snd_443_);
lean_dec(v_a_439_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_455_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_450_; 
v___x_447_ = l_Lean_Expr_getAppNumArgs(v_e_376_);
lean_dec_ref(v_e_376_);
v___x_448_ = lean_alloc_ctor(10, 3, 0);
lean_ctor_set(v___x_448_, 0, v_typeName_436_);
lean_ctor_set(v___x_448_, 1, v_idx_437_);
lean_ctor_set(v___x_448_, 2, v___x_447_);
if (v_isShared_446_ == 0)
{
lean_ctor_set(v___x_445_, 0, v___x_448_);
v___x_450_ = v___x_445_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_454_; 
v_reuseFailAlloc_454_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_454_, 0, v___x_448_);
lean_ctor_set(v_reuseFailAlloc_454_, 1, v_snd_443_);
v___x_450_ = v_reuseFailAlloc_454_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
lean_object* v___x_452_; 
if (v_isShared_442_ == 0)
{
lean_ctor_set(v___x_441_, 0, v___x_450_);
v___x_452_ = v___x_441_;
goto v_reusejp_451_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v___x_450_);
v___x_452_ = v_reuseFailAlloc_453_;
goto v_reusejp_451_;
}
v_reusejp_451_:
{
return v___x_452_;
}
}
}
}
}
else
{
lean_object* v_a_458_; lean_object* v___x_460_; uint8_t v_isShared_461_; uint8_t v_isSharedCheck_465_; 
lean_dec(v_idx_437_);
lean_dec(v_typeName_436_);
lean_dec_ref(v_e_376_);
v_a_458_ = lean_ctor_get(v___x_438_, 0);
v_isSharedCheck_465_ = !lean_is_exclusive(v___x_438_);
if (v_isSharedCheck_465_ == 0)
{
v___x_460_ = v___x_438_;
v_isShared_461_ = v_isSharedCheck_465_;
goto v_resetjp_459_;
}
else
{
lean_inc(v_a_458_);
lean_dec(v___x_438_);
v___x_460_ = lean_box(0);
v_isShared_461_ = v_isSharedCheck_465_;
goto v_resetjp_459_;
}
v_resetjp_459_:
{
lean_object* v___x_463_; 
if (v_isShared_461_ == 0)
{
v___x_463_ = v___x_460_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_464_, 0, v_a_458_);
v___x_463_ = v_reuseFailAlloc_464_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
return v___x_463_;
}
}
}
}
case 1:
{
lean_object* v_fvarId_466_; lean_object* v___x_467_; lean_object* v___y_469_; lean_object* v___x_487_; lean_object* v___x_488_; uint8_t v___x_489_; 
v_fvarId_466_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_fvarId_466_);
lean_dec_ref_known(v___x_386_, 1);
lean_inc(v_a_379_);
lean_inc(v_lambdas_377_);
v___x_467_ = l_List_appendTR___redArg(v_lambdas_377_, v_a_379_);
v___x_487_ = l_Lean_Expr_getAppNumArgs(v_e_376_);
v___x_488_ = lean_unsigned_to_nat(0u);
v___x_489_ = lean_nat_dec_eq(v___x_487_, v___x_488_);
lean_dec(v___x_487_);
if (v___x_489_ == 0)
{
lean_object* v___x_490_; 
lean_inc(v_a_379_);
lean_inc_ref(v_e_376_);
v___x_490_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(v_lambdas_377_, v_e_376_, v_a_379_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_, v_a_384_);
if (lean_obj_tag(v___x_490_) == 0)
{
lean_object* v_a_491_; lean_object* v_snd_492_; 
v_a_491_ = lean_ctor_get(v___x_490_, 0);
lean_inc(v_a_491_);
lean_dec_ref_known(v___x_490_, 1);
v_snd_492_ = lean_ctor_get(v_a_491_, 1);
lean_inc(v_snd_492_);
lean_dec(v_a_491_);
v___y_469_ = v_snd_492_;
goto v___jp_468_;
}
else
{
lean_object* v_a_493_; lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_500_; 
lean_dec(v___x_467_);
lean_dec(v_fvarId_466_);
lean_dec_ref(v_e_376_);
v_a_493_ = lean_ctor_get(v___x_490_, 0);
v_isSharedCheck_500_ = !lean_is_exclusive(v___x_490_);
if (v_isSharedCheck_500_ == 0)
{
v___x_495_ = v___x_490_;
v_isShared_496_ = v_isSharedCheck_500_;
goto v_resetjp_494_;
}
else
{
lean_inc(v_a_493_);
lean_dec(v___x_490_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_500_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
lean_object* v___x_498_; 
if (v_isShared_496_ == 0)
{
v___x_498_ = v___x_495_;
goto v_reusejp_497_;
}
else
{
lean_object* v_reuseFailAlloc_499_; 
v_reuseFailAlloc_499_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_499_, 0, v_a_493_);
v___x_498_ = v_reuseFailAlloc_499_;
goto v_reusejp_497_;
}
v_reusejp_497_:
{
return v___x_498_;
}
}
}
}
else
{
lean_dec(v_lambdas_377_);
v___y_469_ = v_a_380_;
goto v___jp_468_;
}
v___jp_468_:
{
lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_470_ = lean_unsigned_to_nat(0u);
v___x_471_ = lp_mathlib_List_findIdx_x3f_go___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__0(v_fvarId_466_, v___x_467_, v___x_470_);
lean_dec(v___x_467_);
if (lean_obj_tag(v___x_471_) == 1)
{
lean_object* v_val_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_482_; 
lean_dec(v_fvarId_466_);
v_val_472_ = lean_ctor_get(v___x_471_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_471_);
if (v_isSharedCheck_482_ == 0)
{
v___x_474_ = v___x_471_;
v_isShared_475_ = v_isSharedCheck_482_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_val_472_);
lean_dec(v___x_471_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_482_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_480_; 
v___x_476_ = l_Lean_Expr_getAppNumArgs(v_e_376_);
lean_dec_ref(v_e_376_);
v___x_477_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_477_, 0, v_val_472_);
lean_ctor_set(v___x_477_, 1, v___x_476_);
v___x_478_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_478_, 0, v___x_477_);
lean_ctor_set(v___x_478_, 1, v___y_469_);
if (v_isShared_475_ == 0)
{
lean_ctor_set_tag(v___x_474_, 0);
lean_ctor_set(v___x_474_, 0, v___x_478_);
v___x_480_ = v___x_474_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v___x_478_);
v___x_480_ = v_reuseFailAlloc_481_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
return v___x_480_;
}
}
}
else
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; 
lean_dec(v___x_471_);
v___x_483_ = l_Lean_Expr_getAppNumArgs(v_e_376_);
lean_dec_ref(v_e_376_);
v___x_484_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_484_, 0, v_fvarId_466_);
lean_ctor_set(v___x_484_, 1, v___x_483_);
v___x_485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_485_, 0, v___x_484_);
lean_ctor_set(v___x_485_, 1, v___y_469_);
v___x_486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_486_, 0, v___x_485_);
return v___x_486_;
}
}
}
case 2:
{
lean_object* v_mvarId_501_; uint8_t v___x_502_; 
lean_dec(v_lambdas_377_);
v_mvarId_501_ = lean_ctor_get(v___x_386_, 0);
lean_inc(v_mvarId_501_);
lean_dec_ref_known(v___x_386_, 1);
v___x_502_ = l_Lean_Expr_isApp(v_e_376_);
lean_dec_ref(v_e_376_);
if (v___x_502_ == 0)
{
lean_object* v___x_503_; 
v___x_503_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_mkLabelledStar___redArg(v_mvarId_501_, v_a_380_);
return v___x_503_;
}
else
{
lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; 
lean_dec(v_mvarId_501_);
v___x_504_ = lean_box(0);
v___x_505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_505_, 0, v___x_504_);
lean_ctor_set(v___x_505_, 1, v_a_380_);
v___x_506_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_506_, 0, v___x_505_);
return v___x_506_;
}
}
case 7:
{
lean_object* v___x_507_; 
lean_dec_ref_known(v___x_386_, 3);
lean_inc(v_a_379_);
v___x_507_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___lam__0(v_lambdas_377_, v_e_376_, v_a_379_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_, v_a_384_);
if (lean_obj_tag(v___x_507_) == 0)
{
lean_object* v_a_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_525_; 
v_a_508_ = lean_ctor_get(v___x_507_, 0);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_507_);
if (v_isSharedCheck_525_ == 0)
{
v___x_510_ = v___x_507_;
v_isShared_511_ = v_isSharedCheck_525_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_a_508_);
lean_dec(v___x_507_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_525_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v_snd_512_; lean_object* v___x_514_; uint8_t v_isShared_515_; uint8_t v_isSharedCheck_523_; 
v_snd_512_ = lean_ctor_get(v_a_508_, 1);
v_isSharedCheck_523_ = !lean_is_exclusive(v_a_508_);
if (v_isSharedCheck_523_ == 0)
{
lean_object* v_unused_524_; 
v_unused_524_ = lean_ctor_get(v_a_508_, 0);
lean_dec(v_unused_524_);
v___x_514_ = v_a_508_;
v_isShared_515_ = v_isSharedCheck_523_;
goto v_resetjp_513_;
}
else
{
lean_inc(v_snd_512_);
lean_dec(v_a_508_);
v___x_514_ = lean_box(0);
v_isShared_515_ = v_isSharedCheck_523_;
goto v_resetjp_513_;
}
v_resetjp_513_:
{
lean_object* v___x_516_; lean_object* v___x_518_; 
v___x_516_ = lean_box(9);
if (v_isShared_515_ == 0)
{
lean_ctor_set(v___x_514_, 0, v___x_516_);
v___x_518_ = v___x_514_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___x_516_);
lean_ctor_set(v_reuseFailAlloc_522_, 1, v_snd_512_);
v___x_518_ = v_reuseFailAlloc_522_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
lean_object* v___x_520_; 
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 0, v___x_518_);
v___x_520_ = v___x_510_;
goto v_reusejp_519_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v___x_518_);
v___x_520_ = v_reuseFailAlloc_521_;
goto v_reusejp_519_;
}
v_reusejp_519_:
{
return v___x_520_;
}
}
}
}
}
else
{
lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_533_; 
v_a_526_ = lean_ctor_get(v___x_507_, 0);
v_isSharedCheck_533_ = !lean_is_exclusive(v___x_507_);
if (v_isSharedCheck_533_ == 0)
{
v___x_528_ = v___x_507_;
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_507_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_533_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v___x_531_; 
if (v_isShared_529_ == 0)
{
v___x_531_ = v___x_528_;
goto v_reusejp_530_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v_a_526_);
v___x_531_ = v_reuseFailAlloc_532_;
goto v_reusejp_530_;
}
v_reusejp_530_:
{
return v___x_531_;
}
}
}
}
case 9:
{
lean_object* v_a_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; 
lean_dec(v_lambdas_377_);
lean_dec_ref(v_e_376_);
v_a_534_ = lean_ctor_get(v___x_386_, 0);
lean_inc_ref(v_a_534_);
lean_dec_ref_known(v___x_386_, 1);
v___x_535_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v___x_535_, 0, v_a_534_);
v___x_536_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_536_, 0, v___x_535_);
lean_ctor_set(v___x_536_, 1, v_a_380_);
v___x_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_537_, 0, v___x_536_);
return v___x_537_;
}
case 3:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
lean_dec_ref_known(v___x_386_, 1);
lean_dec(v_lambdas_377_);
lean_dec_ref(v_e_376_);
v___x_538_ = lean_box(7);
v___x_539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_538_);
lean_ctor_set(v___x_539_, 1, v_a_380_);
v___x_540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_540_, 0, v___x_539_);
return v___x_540_;
}
case 8:
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; 
lean_dec_ref_known(v___x_386_, 4);
lean_dec(v_lambdas_377_);
lean_dec_ref(v_e_376_);
v___x_541_ = lean_box(2);
v___x_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_542_, 0, v___x_541_);
lean_ctor_set(v___x_542_, 1, v_a_380_);
v___x_543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_543_, 0, v___x_542_);
return v___x_543_;
}
case 6:
{
lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; 
lean_dec_ref_known(v___x_386_, 3);
lean_dec(v_lambdas_377_);
lean_dec_ref(v_e_376_);
v___x_544_ = lean_box(2);
v___x_545_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_545_, 0, v___x_544_);
lean_ctor_set(v___x_545_, 1, v_a_380_);
v___x_546_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_546_, 0, v___x_545_);
return v___x_546_;
}
default: 
{
lean_object* v___x_547_; lean_object* v___x_548_; 
lean_dec_ref(v___x_386_);
lean_dec(v_lambdas_377_);
lean_dec_ref(v_e_376_);
v___x_547_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3, &lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__3);
v___x_548_ = lp_mathlib_panic___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go_spec__1(v___x_547_, v_a_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_, v_a_384_);
return v___x_548_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___boxed(lean_object* v_e_549_, lean_object* v_lambdas_550_, lean_object* v_root_551_, lean_object* v_a_552_, lean_object* v_a_553_, lean_object* v_a_554_, lean_object* v_a_555_, lean_object* v_a_556_, lean_object* v_a_557_, lean_object* v_a_558_){
_start:
{
uint8_t v_root_boxed_559_; lean_object* v_res_560_; 
v_root_boxed_559_ = lean_unbox(v_root_551_);
v_res_560_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(v_e_549_, v_lambdas_550_, v_root_boxed_559_, v_a_552_, v_a_553_, v_a_554_, v_a_555_, v_a_556_, v_a_557_);
lean_dec(v_a_557_);
lean_dec_ref(v_a_556_);
lean_dec(v_a_555_);
lean_dec_ref(v_a_554_);
lean_dec(v_a_552_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux(lean_object* v_e_561_, lean_object* v_lambdas_562_, uint8_t v_root_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_, lean_object* v_a_568_, lean_object* v_a_569_){
_start:
{
lean_object* v___x_571_; 
lean_inc(v_lambdas_562_);
v___x_571_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(v_e_561_, v_lambdas_562_, v_root_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_, v_a_569_);
if (lean_obj_tag(v___x_571_) == 0)
{
lean_object* v_a_572_; 
v_a_572_ = lean_ctor_get(v___x_571_, 0);
lean_inc(v_a_572_);
if (lean_obj_tag(v_lambdas_562_) == 0)
{
lean_dec(v_a_572_);
return v___x_571_;
}
else
{
lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_613_; 
v_isSharedCheck_613_ = !lean_is_exclusive(v___x_571_);
if (v_isSharedCheck_613_ == 0)
{
lean_object* v_unused_614_; 
v_unused_614_ = lean_ctor_get(v___x_571_, 0);
lean_dec(v_unused_614_);
v___x_574_ = v___x_571_;
v_isShared_575_ = v_isSharedCheck_613_;
goto v_resetjp_573_;
}
else
{
lean_dec(v___x_571_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_613_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v_snd_576_; lean_object* v_fst_577_; lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_612_; 
v_snd_576_ = lean_ctor_get(v_a_572_, 1);
v_fst_577_ = lean_ctor_get(v_a_572_, 0);
v_isSharedCheck_612_ = !lean_is_exclusive(v_a_572_);
if (v_isSharedCheck_612_ == 0)
{
v___x_579_ = v_a_572_;
v_isShared_580_ = v_isSharedCheck_612_;
goto v_resetjp_578_;
}
else
{
lean_inc(v_snd_576_);
lean_inc(v_fst_577_);
lean_dec(v_a_572_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_612_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v_tail_581_; lean_object* v___x_583_; uint8_t v_isShared_584_; uint8_t v_isSharedCheck_610_; 
v_tail_581_ = lean_ctor_get(v_lambdas_562_, 1);
v_isSharedCheck_610_ = !lean_is_exclusive(v_lambdas_562_);
if (v_isSharedCheck_610_ == 0)
{
lean_object* v_unused_611_; 
v_unused_611_ = lean_ctor_get(v_lambdas_562_, 0);
lean_dec(v_unused_611_);
v___x_583_ = v_lambdas_562_;
v_isShared_584_ = v_isSharedCheck_610_;
goto v_resetjp_582_;
}
else
{
lean_inc(v_tail_581_);
lean_dec(v_lambdas_562_);
v___x_583_ = lean_box(0);
v_isShared_584_ = v_isSharedCheck_610_;
goto v_resetjp_582_;
}
v_resetjp_582_:
{
lean_object* v_previous_585_; lean_object* v_stack_586_; lean_object* v_mctx_587_; lean_object* v_labelledStars_x3f_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_608_; 
v_previous_585_ = lean_ctor_get(v_snd_576_, 0);
v_stack_586_ = lean_ctor_get(v_snd_576_, 1);
v_mctx_587_ = lean_ctor_get(v_snd_576_, 2);
v_labelledStars_x3f_588_ = lean_ctor_get(v_snd_576_, 3);
v_isSharedCheck_608_ = !lean_is_exclusive(v_snd_576_);
if (v_isSharedCheck_608_ == 0)
{
lean_object* v_unused_609_; 
v_unused_609_ = lean_ctor_get(v_snd_576_, 4);
lean_dec(v_unused_609_);
v___x_590_ = v_snd_576_;
v_isShared_591_ = v_isSharedCheck_608_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_labelledStars_x3f_588_);
lean_inc(v_mctx_587_);
lean_inc(v_stack_586_);
lean_inc(v_previous_585_);
lean_dec(v_snd_576_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_608_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v___f_592_; lean_object* v___x_593_; lean_object* v___x_595_; 
v___f_592_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_withLams___redArg___closed__0));
v___x_593_ = lean_box(0);
if (v_isShared_584_ == 0)
{
lean_ctor_set(v___x_583_, 1, v___x_593_);
lean_ctor_set(v___x_583_, 0, v_fst_577_);
v___x_595_ = v___x_583_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v_fst_577_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v___x_593_);
v___x_595_ = v_reuseFailAlloc_607_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
lean_object* v___x_596_; lean_object* v___x_598_; 
v___x_596_ = l_List_foldl___redArg(v___f_592_, v___x_595_, v_tail_581_);
if (v_isShared_591_ == 0)
{
lean_ctor_set(v___x_590_, 4, v___x_596_);
v___x_598_ = v___x_590_;
goto v_reusejp_597_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v_previous_585_);
lean_ctor_set(v_reuseFailAlloc_606_, 1, v_stack_586_);
lean_ctor_set(v_reuseFailAlloc_606_, 2, v_mctx_587_);
lean_ctor_set(v_reuseFailAlloc_606_, 3, v_labelledStars_x3f_588_);
lean_ctor_set(v_reuseFailAlloc_606_, 4, v___x_596_);
v___x_598_ = v_reuseFailAlloc_606_;
goto v_reusejp_597_;
}
v_reusejp_597_:
{
lean_object* v___x_599_; lean_object* v___x_601_; 
v___x_599_ = lean_box(8);
if (v_isShared_580_ == 0)
{
lean_ctor_set(v___x_579_, 1, v___x_598_);
lean_ctor_set(v___x_579_, 0, v___x_599_);
v___x_601_ = v___x_579_;
goto v_reusejp_600_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v___x_599_);
lean_ctor_set(v_reuseFailAlloc_605_, 1, v___x_598_);
v___x_601_ = v_reuseFailAlloc_605_;
goto v_reusejp_600_;
}
v_reusejp_600_:
{
lean_object* v___x_603_; 
if (v_isShared_575_ == 0)
{
lean_ctor_set(v___x_574_, 0, v___x_601_);
v___x_603_ = v___x_574_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v___x_601_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
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
lean_dec(v_lambdas_562_);
return v___x_571_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux___boxed(lean_object* v_e_615_, lean_object* v_lambdas_616_, lean_object* v_root_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_, lean_object* v_a_623_, lean_object* v_a_624_){
_start:
{
uint8_t v_root_boxed_625_; lean_object* v_res_626_; 
v_root_boxed_625_ = lean_unbox(v_root_617_);
v_res_626_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux(v_e_615_, v_lambdas_616_, v_root_boxed_625_, v_a_618_, v_a_619_, v_a_620_, v_a_621_, v_a_622_, v_a_623_);
lean_dec(v_a_623_);
lean_dec_ref(v_a_622_);
lean_dec(v_a_621_);
lean_dec_ref(v_a_620_);
lean_dec(v_a_618_);
return v_res_626_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(lean_object* v_arg_627_, lean_object* v_a_628_){
_start:
{
if (lean_obj_tag(v_a_628_) == 5)
{
lean_object* v_fn_629_; lean_object* v_arg_630_; uint8_t v___x_631_; 
v_fn_629_ = lean_ctor_get(v_a_628_, 0);
v_arg_630_ = lean_ctor_get(v_a_628_, 1);
v___x_631_ = lean_expr_eqv(v_arg_630_, v_arg_627_);
if (v___x_631_ == 0)
{
v_a_628_ = v_fn_629_;
goto _start;
}
else
{
lean_object* v___x_633_; uint8_t v___x_634_; 
v___x_633_ = l_Lean_Expr_getAppFn(v_fn_629_);
v___x_634_ = l_Lean_Expr_isMVar(v___x_633_);
lean_dec_ref(v___x_633_);
return v___x_634_;
}
}
else
{
uint8_t v___x_635_; 
v___x_635_ = 0;
return v___x_635_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg___boxed(lean_object* v_arg_636_, lean_object* v_a_637_){
_start:
{
uint8_t v_res_638_; lean_object* v_r_639_; 
v_res_638_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(v_arg_636_, v_a_637_);
lean_dec_ref(v_a_637_);
lean_dec_ref(v_arg_636_);
v_r_639_ = lean_box(v_res_638_);
return v_r_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(lean_object* v_x_640_, lean_object* v_x_641_){
_start:
{
if (lean_obj_tag(v_x_641_) == 0)
{
return v_x_640_;
}
else
{
lean_object* v_tail_642_; lean_object* v___x_644_; uint8_t v_isShared_645_; uint8_t v_isSharedCheck_651_; 
v_tail_642_ = lean_ctor_get(v_x_641_, 1);
v_isSharedCheck_651_ = !lean_is_exclusive(v_x_641_);
if (v_isSharedCheck_651_ == 0)
{
lean_object* v_unused_652_; 
v_unused_652_ = lean_ctor_get(v_x_641_, 0);
lean_dec(v_unused_652_);
v___x_644_ = v_x_641_;
v_isShared_645_ = v_isSharedCheck_651_;
goto v_resetjp_643_;
}
else
{
lean_inc(v_tail_642_);
lean_dec(v_x_641_);
v___x_644_ = lean_box(0);
v_isShared_645_ = v_isSharedCheck_651_;
goto v_resetjp_643_;
}
v_resetjp_643_:
{
lean_object* v___x_646_; lean_object* v___x_648_; 
v___x_646_ = lean_box(8);
if (v_isShared_645_ == 0)
{
lean_ctor_set(v___x_644_, 1, v_x_640_);
lean_ctor_set(v___x_644_, 0, v___x_646_);
v___x_648_ = v___x_644_;
goto v_reusejp_647_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_646_);
lean_ctor_set(v_reuseFailAlloc_650_, 1, v_x_640_);
v___x_648_ = v_reuseFailAlloc_650_;
goto v_reusejp_647_;
}
v_reusejp_647_:
{
v_x_640_ = v___x_648_;
v_x_641_ = v_tail_642_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(lean_object* v_e_653_, lean_object* v_lambdas_654_, uint8_t v_root_655_, lean_object* v_entry_656_, lean_object* v_a_657_, lean_object* v_a_658_, lean_object* v_a_659_, lean_object* v_a_660_, lean_object* v_a_661_){
_start:
{
lean_object* v___y_664_; lean_object* v_____do__lift_665_; lean_object* v___y_669_; lean_object* v___y_672_; lean_object* v___y_675_; lean_object* v___y_676_; lean_object* v___y_677_; uint8_t v___y_678_; lean_object* v_a_682_; lean_object* v___y_692_; lean_object* v___x_702_; 
lean_inc_ref(v_entry_656_);
lean_inc(v_lambdas_654_);
lean_inc_ref(v_e_653_);
v___x_702_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(v_e_653_, v_lambdas_654_, v_root_655_, v_a_657_, v_entry_656_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
if (lean_obj_tag(v___x_702_) == 0)
{
lean_object* v_a_703_; 
v_a_703_ = lean_ctor_get(v___x_702_, 0);
lean_inc(v_a_703_);
if (lean_obj_tag(v_lambdas_654_) == 0)
{
lean_dec(v_a_703_);
v___y_692_ = v___x_702_;
goto v___jp_691_;
}
else
{
lean_object* v_snd_704_; lean_object* v_fst_705_; lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_729_; 
lean_dec_ref_known(v___x_702_, 1);
v_snd_704_ = lean_ctor_get(v_a_703_, 1);
v_fst_705_ = lean_ctor_get(v_a_703_, 0);
v_isSharedCheck_729_ = !lean_is_exclusive(v_a_703_);
if (v_isSharedCheck_729_ == 0)
{
v___x_707_ = v_a_703_;
v_isShared_708_ = v_isSharedCheck_729_;
goto v_resetjp_706_;
}
else
{
lean_inc(v_snd_704_);
lean_inc(v_fst_705_);
lean_dec(v_a_703_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_729_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
lean_object* v_tail_709_; lean_object* v_previous_710_; lean_object* v_stack_711_; lean_object* v_mctx_712_; lean_object* v_labelledStars_x3f_713_; lean_object* v___x_715_; uint8_t v_isShared_716_; uint8_t v_isSharedCheck_727_; 
v_tail_709_ = lean_ctor_get(v_lambdas_654_, 1);
v_previous_710_ = lean_ctor_get(v_snd_704_, 0);
v_stack_711_ = lean_ctor_get(v_snd_704_, 1);
v_mctx_712_ = lean_ctor_get(v_snd_704_, 2);
v_labelledStars_x3f_713_ = lean_ctor_get(v_snd_704_, 3);
v_isSharedCheck_727_ = !lean_is_exclusive(v_snd_704_);
if (v_isSharedCheck_727_ == 0)
{
lean_object* v_unused_728_; 
v_unused_728_ = lean_ctor_get(v_snd_704_, 4);
lean_dec(v_unused_728_);
v___x_715_ = v_snd_704_;
v_isShared_716_ = v_isSharedCheck_727_;
goto v_resetjp_714_;
}
else
{
lean_inc(v_labelledStars_x3f_713_);
lean_inc(v_mctx_712_);
lean_inc(v_stack_711_);
lean_inc(v_previous_710_);
lean_dec(v_snd_704_);
v___x_715_ = lean_box(0);
v_isShared_716_ = v_isSharedCheck_727_;
goto v_resetjp_714_;
}
v_resetjp_714_:
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_721_; 
v___x_717_ = lean_box(0);
v___x_718_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_718_, 0, v_fst_705_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
lean_inc(v_tail_709_);
v___x_719_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(v___x_718_, v_tail_709_);
if (v_isShared_716_ == 0)
{
lean_ctor_set(v___x_715_, 4, v___x_719_);
v___x_721_ = v___x_715_;
goto v_reusejp_720_;
}
else
{
lean_object* v_reuseFailAlloc_726_; 
v_reuseFailAlloc_726_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_726_, 0, v_previous_710_);
lean_ctor_set(v_reuseFailAlloc_726_, 1, v_stack_711_);
lean_ctor_set(v_reuseFailAlloc_726_, 2, v_mctx_712_);
lean_ctor_set(v_reuseFailAlloc_726_, 3, v_labelledStars_x3f_713_);
lean_ctor_set(v_reuseFailAlloc_726_, 4, v___x_719_);
v___x_721_ = v_reuseFailAlloc_726_;
goto v_reusejp_720_;
}
v_reusejp_720_:
{
lean_object* v___x_722_; lean_object* v___x_724_; 
v___x_722_ = lean_box(8);
if (v_isShared_708_ == 0)
{
lean_ctor_set(v___x_707_, 1, v___x_721_);
lean_ctor_set(v___x_707_, 0, v___x_722_);
v___x_724_ = v___x_707_;
goto v_reusejp_723_;
}
else
{
lean_object* v_reuseFailAlloc_725_; 
v_reuseFailAlloc_725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_725_, 0, v___x_722_);
lean_ctor_set(v_reuseFailAlloc_725_, 1, v___x_721_);
v___x_724_ = v_reuseFailAlloc_725_;
goto v_reusejp_723_;
}
v_reusejp_723_:
{
v_a_682_ = v___x_724_;
goto v___jp_681_;
}
}
}
}
}
}
else
{
v___y_692_ = v___x_702_;
goto v___jp_691_;
}
v___jp_663_:
{
lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_666_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_666_, 0, v___y_664_);
lean_ctor_set(v___x_666_, 1, v_____do__lift_665_);
v___x_667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_667_, 0, v___x_666_);
return v___x_667_;
}
v___jp_668_:
{
lean_object* v___x_670_; 
v___x_670_ = lean_box(0);
v___y_664_ = v___y_669_;
v_____do__lift_665_ = v___x_670_;
goto v___jp_663_;
}
v___jp_671_:
{
lean_object* v___x_673_; 
v___x_673_ = lean_box(0);
v___y_664_ = v___y_672_;
v_____do__lift_665_ = v___x_673_;
goto v___jp_663_;
}
v___jp_674_:
{
if (v___y_678_ == 0)
{
lean_dec_ref(v___y_676_);
lean_dec(v___y_675_);
lean_dec_ref(v_entry_656_);
v___y_669_ = v___y_677_;
goto v___jp_668_;
}
else
{
lean_object* v___x_679_; 
v___x_679_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(v___y_676_, v___y_675_, v_root_655_, v_entry_656_, v_a_657_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
if (lean_obj_tag(v___x_679_) == 0)
{
lean_object* v_a_680_; 
v_a_680_ = lean_ctor_get(v___x_679_, 0);
lean_inc(v_a_680_);
lean_dec_ref_known(v___x_679_, 1);
v___y_664_ = v___y_677_;
v_____do__lift_665_ = v_a_680_;
goto v___jp_663_;
}
else
{
lean_dec_ref(v___y_677_);
return v___x_679_;
}
}
}
v___jp_681_:
{
if (lean_obj_tag(v_e_653_) == 5)
{
if (lean_obj_tag(v_lambdas_654_) == 1)
{
lean_object* v_fn_683_; lean_object* v_arg_684_; lean_object* v_head_685_; lean_object* v_tail_686_; lean_object* v___x_687_; uint8_t v___x_688_; 
v_fn_683_ = lean_ctor_get(v_e_653_, 0);
lean_inc_ref(v_fn_683_);
v_arg_684_ = lean_ctor_get(v_e_653_, 1);
lean_inc_ref(v_arg_684_);
lean_dec_ref_known(v_e_653_, 2);
v_head_685_ = lean_ctor_get(v_lambdas_654_, 0);
lean_inc(v_head_685_);
v_tail_686_ = lean_ctor_get(v_lambdas_654_, 1);
lean_inc(v_tail_686_);
lean_dec_ref_known(v_lambdas_654_, 2);
v___x_687_ = l_Lean_Expr_fvar___override(v_head_685_);
v___x_688_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_isStarWithArg(v___x_687_, v_arg_684_);
lean_dec_ref(v_arg_684_);
lean_dec_ref(v___x_687_);
if (v___x_688_ == 0)
{
v___y_675_ = v_tail_686_;
v___y_676_ = v_fn_683_;
v___y_677_ = v_a_682_;
v___y_678_ = v___x_688_;
goto v___jp_674_;
}
else
{
lean_object* v___x_689_; uint8_t v___x_690_; 
v___x_689_ = l_Lean_Expr_getAppFn(v_fn_683_);
v___x_690_ = l_Lean_Expr_isMVar(v___x_689_);
lean_dec_ref(v___x_689_);
if (v___x_690_ == 0)
{
v___y_675_ = v_tail_686_;
v___y_676_ = v_fn_683_;
v___y_677_ = v_a_682_;
v___y_678_ = v___x_688_;
goto v___jp_674_;
}
else
{
lean_dec(v_tail_686_);
lean_dec_ref(v_fn_683_);
lean_dec_ref(v_entry_656_);
v___y_669_ = v_a_682_;
goto v___jp_668_;
}
}
}
else
{
lean_dec_ref_known(v_e_653_, 2);
lean_dec_ref(v_entry_656_);
lean_dec(v_lambdas_654_);
v___y_672_ = v_a_682_;
goto v___jp_671_;
}
}
else
{
lean_dec_ref(v_entry_656_);
lean_dec(v_lambdas_654_);
lean_dec_ref(v_e_653_);
v___y_672_ = v_a_682_;
goto v___jp_671_;
}
}
v___jp_691_:
{
if (lean_obj_tag(v___y_692_) == 0)
{
lean_object* v_a_693_; 
v_a_693_ = lean_ctor_get(v___y_692_, 0);
lean_inc(v_a_693_);
lean_dec_ref_known(v___y_692_, 1);
v_a_682_ = v_a_693_;
goto v___jp_681_;
}
else
{
lean_object* v_a_694_; lean_object* v___x_696_; uint8_t v_isShared_697_; uint8_t v_isSharedCheck_701_; 
lean_dec_ref(v_entry_656_);
lean_dec(v_lambdas_654_);
lean_dec_ref(v_e_653_);
v_a_694_ = lean_ctor_get(v___y_692_, 0);
v_isSharedCheck_701_ = !lean_is_exclusive(v___y_692_);
if (v_isSharedCheck_701_ == 0)
{
v___x_696_ = v___y_692_;
v_isShared_697_ = v_isSharedCheck_701_;
goto v_resetjp_695_;
}
else
{
lean_inc(v_a_694_);
lean_dec(v___y_692_);
v___x_696_ = lean_box(0);
v_isShared_697_ = v_isSharedCheck_701_;
goto v_resetjp_695_;
}
v_resetjp_695_:
{
lean_object* v___x_699_; 
if (v_isShared_697_ == 0)
{
v___x_699_ = v___x_696_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v_a_694_);
v___x_699_ = v_reuseFailAlloc_700_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
return v___x_699_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities___boxed(lean_object* v_e_730_, lean_object* v_lambdas_731_, lean_object* v_root_732_, lean_object* v_entry_733_, lean_object* v_a_734_, lean_object* v_a_735_, lean_object* v_a_736_, lean_object* v_a_737_, lean_object* v_a_738_, lean_object* v_a_739_){
_start:
{
uint8_t v_root_boxed_740_; lean_object* v_res_741_; 
v_root_boxed_740_ = lean_unbox(v_root_732_);
v_res_741_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(v_e_730_, v_lambdas_731_, v_root_boxed_740_, v_entry_733_, v_a_734_, v_a_735_, v_a_736_, v_a_737_, v_a_738_);
lean_dec(v_a_738_);
lean_dec_ref(v_a_737_);
lean_dec(v_a_736_);
lean_dec_ref(v_a_735_);
lean_dec(v_a_734_);
return v_res_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0___boxed(lean_object* v_body_742_, lean_object* v_lambdas_743_, lean_object* v_inst_744_, lean_object* v_inst_745_, lean_object* v_inst_746_, lean_object* v_noIndex_747_, lean_object* v_k_748_, lean_object* v_fvar_749_){
_start:
{
lean_object* v_res_750_; 
v_res_750_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0(v_body_742_, v_lambdas_743_, v_inst_744_, v_inst_745_, v_inst_746_, v_noIndex_747_, v_k_748_, v_fvar_749_);
lean_dec_ref(v_fvar_749_);
lean_dec_ref(v_body_742_);
return v_res_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__1(lean_object* v_lambdas_751_, lean_object* v_inst_752_, lean_object* v_inst_753_, lean_object* v_inst_754_, lean_object* v_noIndex_755_, lean_object* v_k_756_, lean_object* v_____do__lift_757_){
_start:
{
if (lean_obj_tag(v_____do__lift_757_) == 6)
{
lean_object* v_binderName_758_; lean_object* v_binderType_759_; lean_object* v_body_760_; uint8_t v_binderInfo_761_; lean_object* v___f_762_; uint8_t v___x_763_; lean_object* v___x_764_; 
v_binderName_758_ = lean_ctor_get(v_____do__lift_757_, 0);
lean_inc(v_binderName_758_);
v_binderType_759_ = lean_ctor_get(v_____do__lift_757_, 1);
lean_inc_ref(v_binderType_759_);
v_body_760_ = lean_ctor_get(v_____do__lift_757_, 2);
lean_inc_ref(v_body_760_);
v_binderInfo_761_ = lean_ctor_get_uint8(v_____do__lift_757_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_____do__lift_757_, 3);
lean_inc_ref(v_inst_754_);
lean_inc_ref(v_inst_752_);
v___f_762_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_762_, 0, v_body_760_);
lean_closure_set(v___f_762_, 1, v_lambdas_751_);
lean_closure_set(v___f_762_, 2, v_inst_752_);
lean_closure_set(v___f_762_, 3, v_inst_753_);
lean_closure_set(v___f_762_, 4, v_inst_754_);
lean_closure_set(v___f_762_, 5, v_noIndex_755_);
lean_closure_set(v___f_762_, 6, v_k_756_);
v___x_763_ = 0;
v___x_764_ = l_Lean_Meta_withLocalDecl___redArg(v_inst_754_, v_inst_752_, v_binderName_758_, v_binderInfo_761_, v_binderType_759_, v___f_762_, v___x_763_);
return v___x_764_;
}
else
{
lean_object* v___x_765_; 
lean_dec(v_noIndex_755_);
lean_dec_ref(v_inst_754_);
lean_dec(v_inst_753_);
lean_dec_ref(v_inst_752_);
v___x_765_ = lean_apply_2(v_k_756_, v_____do__lift_757_, v_lambdas_751_);
return v___x_765_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(lean_object* v_inst_766_, lean_object* v_inst_767_, lean_object* v_inst_768_, lean_object* v_e_769_, lean_object* v_lambdas_770_, lean_object* v_noIndex_771_, lean_object* v_k_772_){
_start:
{
uint8_t v___x_773_; 
v___x_773_ = l_Lean_Meta_DiscrTree_hasNoindexAnnotation(v_e_769_);
if (v___x_773_ == 0)
{
lean_object* v_toBind_774_; lean_object* v___f_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; 
v_toBind_774_ = lean_ctor_get(v_inst_766_, 1);
lean_inc(v_toBind_774_);
lean_inc(v_inst_767_);
v___f_775_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__1), 7, 6);
lean_closure_set(v___f_775_, 0, v_lambdas_770_);
lean_closure_set(v___f_775_, 1, v_inst_766_);
lean_closure_set(v___f_775_, 2, v_inst_767_);
lean_closure_set(v___f_775_, 3, v_inst_768_);
lean_closure_set(v___f_775_, 4, v_noIndex_771_);
lean_closure_set(v___f_775_, 5, v_k_772_);
v___x_776_ = lean_alloc_closure((void*)(l_Lean_Meta_DiscrTree_reduce___boxed), 6, 1);
lean_closure_set(v___x_776_, 0, v_e_769_);
v___x_777_ = lean_apply_2(v_inst_767_, lean_box(0), v___x_776_);
v___x_778_ = lean_apply_4(v_toBind_774_, lean_box(0), lean_box(0), v___x_777_, v___f_775_);
return v___x_778_;
}
else
{
lean_object* v___x_779_; 
lean_dec(v_k_772_);
lean_dec_ref(v_e_769_);
lean_dec_ref(v_inst_768_);
lean_dec(v_inst_767_);
lean_dec_ref(v_inst_766_);
v___x_779_ = lean_apply_1(v_noIndex_771_, v_lambdas_770_);
return v___x_779_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg___lam__0(lean_object* v_body_780_, lean_object* v_lambdas_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_inst_784_, lean_object* v_noIndex_785_, lean_object* v_k_786_, lean_object* v_fvar_787_){
_start:
{
lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; 
v___x_788_ = lean_expr_instantiate1(v_body_780_, v_fvar_787_);
v___x_789_ = l_Lean_Expr_fvarId_x21(v_fvar_787_);
v___x_790_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_790_, 0, v___x_789_);
lean_ctor_set(v___x_790_, 1, v_lambdas_781_);
v___x_791_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(v_inst_782_, v_inst_783_, v_inst_784_, v___x_788_, v___x_790_, v_noIndex_785_, v_k_786_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce(lean_object* v_m_792_, lean_object* v_00_u03b1_793_, lean_object* v_inst_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_inst_797_, lean_object* v_e_798_, lean_object* v_lambdas_799_, lean_object* v_noIndex_800_, lean_object* v_k_801_){
_start:
{
lean_object* v___x_802_; 
v___x_802_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___redArg(v_inst_795_, v_inst_796_, v_inst_797_, v_e_798_, v_lambdas_799_, v_noIndex_800_, v_k_801_);
return v___x_802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0(lean_object* v_k_803_, lean_object* v___y_804_, lean_object* v_b_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_){
_start:
{
lean_object* v___x_811_; 
lean_inc(v___y_809_);
lean_inc_ref(v___y_808_);
lean_inc(v___y_807_);
lean_inc_ref(v___y_806_);
lean_inc(v___y_804_);
v___x_811_ = lean_apply_7(v_k_803_, v_b_805_, v___y_804_, v___y_806_, v___y_807_, v___y_808_, v___y_809_, lean_box(0));
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_k_812_, lean_object* v___y_813_, lean_object* v_b_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_){
_start:
{
lean_object* v_res_820_; 
v_res_820_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0(v_k_812_, v___y_813_, v_b_814_, v___y_815_, v___y_816_, v___y_817_, v___y_818_);
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
lean_dec(v___y_816_);
lean_dec_ref(v___y_815_);
lean_dec(v___y_813_);
return v_res_820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(lean_object* v_name_821_, uint8_t v_bi_822_, lean_object* v_type_823_, lean_object* v_k_824_, uint8_t v_kind_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_){
_start:
{
lean_object* v___f_832_; lean_object* v___x_833_; 
lean_inc(v___y_826_);
v___f_832_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_832_, 0, v_k_824_);
lean_closure_set(v___f_832_, 1, v___y_826_);
v___x_833_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_821_, v_bi_822_, v_type_823_, v___f_832_, v_kind_825_, v___y_827_, v___y_828_, v___y_829_, v___y_830_);
if (lean_obj_tag(v___x_833_) == 0)
{
return v___x_833_;
}
else
{
lean_object* v_a_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_841_; 
v_a_834_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_841_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_841_ == 0)
{
v___x_836_ = v___x_833_;
v_isShared_837_ = v_isSharedCheck_841_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_a_834_);
lean_dec(v___x_833_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_841_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
lean_object* v___x_839_; 
if (v_isShared_837_ == 0)
{
v___x_839_ = v___x_836_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v_a_834_);
v___x_839_ = v_reuseFailAlloc_840_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
return v___x_839_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg___boxed(lean_object* v_name_842_, lean_object* v_bi_843_, lean_object* v_type_844_, lean_object* v_k_845_, lean_object* v_kind_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_){
_start:
{
uint8_t v_bi_boxed_853_; uint8_t v_kind_boxed_854_; lean_object* v_res_855_; 
v_bi_boxed_853_ = lean_unbox(v_bi_843_);
v_kind_boxed_854_ = lean_unbox(v_kind_846_);
v_res_855_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(v_name_842_, v_bi_boxed_853_, v_type_844_, v_k_845_, v_kind_boxed_854_, v___y_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
lean_dec(v___y_851_);
lean_dec_ref(v___y_850_);
lean_dec(v___y_849_);
lean_dec_ref(v___y_848_);
lean_dec(v___y_847_);
return v_res_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0___boxed(lean_object* v_body_856_, lean_object* v_lambdas_857_, lean_object* v_entry_858_, lean_object* v_root_859_, lean_object* v_fvar_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_){
_start:
{
uint8_t v_root_boxed_867_; lean_object* v_res_868_; 
v_root_boxed_867_ = lean_unbox(v_root_859_);
v_res_868_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0(v_body_856_, v_lambdas_857_, v_entry_858_, v_root_boxed_867_, v_fvar_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_, v___y_865_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
lean_dec(v___y_863_);
lean_dec_ref(v___y_862_);
lean_dec(v___y_861_);
lean_dec_ref(v_fvar_860_);
lean_dec_ref(v_body_856_);
return v_res_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(lean_object* v_entry_872_, uint8_t v_root_873_, lean_object* v_e_874_, lean_object* v_lambdas_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_){
_start:
{
lean_object* v_a_883_; uint8_t v___x_887_; 
v___x_887_ = l_Lean_Meta_DiscrTree_hasNoindexAnnotation(v_e_874_);
if (v___x_887_ == 0)
{
lean_object* v___x_888_; 
v___x_888_ = l_Lean_Meta_DiscrTree_reduce(v_e_874_, v___y_877_, v___y_878_, v___y_879_, v___y_880_);
if (lean_obj_tag(v___x_888_) == 0)
{
lean_object* v_a_889_; 
v_a_889_ = lean_ctor_get(v___x_888_, 0);
lean_inc(v_a_889_);
lean_dec_ref_known(v___x_888_, 1);
if (lean_obj_tag(v_a_889_) == 6)
{
lean_object* v_binderName_890_; lean_object* v_binderType_891_; lean_object* v_body_892_; uint8_t v_binderInfo_893_; lean_object* v___x_894_; lean_object* v___f_895_; uint8_t v___x_896_; lean_object* v___x_897_; 
v_binderName_890_ = lean_ctor_get(v_a_889_, 0);
lean_inc(v_binderName_890_);
v_binderType_891_ = lean_ctor_get(v_a_889_, 1);
lean_inc_ref(v_binderType_891_);
v_body_892_ = lean_ctor_get(v_a_889_, 2);
lean_inc_ref(v_body_892_);
v_binderInfo_893_ = lean_ctor_get_uint8(v_a_889_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_a_889_, 3);
v___x_894_ = lean_box(v_root_873_);
v___f_895_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0___boxed), 11, 4);
lean_closure_set(v___f_895_, 0, v_body_892_);
lean_closure_set(v___f_895_, 1, v_lambdas_875_);
lean_closure_set(v___f_895_, 2, v_entry_872_);
lean_closure_set(v___f_895_, 3, v___x_894_);
v___x_896_ = 0;
v___x_897_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(v_binderName_890_, v_binderInfo_893_, v_binderType_891_, v___f_895_, v___x_896_, v___y_876_, v___y_877_, v___y_878_, v___y_879_, v___y_880_);
return v___x_897_;
}
else
{
lean_object* v___x_898_; 
v___x_898_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities(v_a_889_, v_lambdas_875_, v_root_873_, v_entry_872_, v___y_876_, v___y_877_, v___y_878_, v___y_879_, v___y_880_);
return v___x_898_;
}
}
else
{
lean_object* v_a_899_; lean_object* v___x_901_; uint8_t v_isShared_902_; uint8_t v_isSharedCheck_906_; 
lean_dec(v_lambdas_875_);
lean_dec_ref(v_entry_872_);
v_a_899_ = lean_ctor_get(v___x_888_, 0);
v_isSharedCheck_906_ = !lean_is_exclusive(v___x_888_);
if (v_isSharedCheck_906_ == 0)
{
v___x_901_ = v___x_888_;
v_isShared_902_ = v_isSharedCheck_906_;
goto v_resetjp_900_;
}
else
{
lean_inc(v_a_899_);
lean_dec(v___x_888_);
v___x_901_ = lean_box(0);
v_isShared_902_ = v_isSharedCheck_906_;
goto v_resetjp_900_;
}
v_resetjp_900_:
{
lean_object* v___x_904_; 
if (v_isShared_902_ == 0)
{
v___x_904_ = v___x_901_;
goto v_reusejp_903_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v_a_899_);
v___x_904_ = v_reuseFailAlloc_905_;
goto v_reusejp_903_;
}
v_reusejp_903_:
{
return v___x_904_;
}
}
}
}
else
{
lean_object* v___x_907_; 
lean_dec_ref(v_e_874_);
v___x_907_ = lean_box(0);
if (lean_obj_tag(v_lambdas_875_) == 0)
{
lean_object* v___x_908_; 
v___x_908_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
lean_ctor_set(v___x_908_, 1, v_entry_872_);
v_a_883_ = v___x_908_;
goto v___jp_882_;
}
else
{
lean_object* v_tail_909_; lean_object* v___x_911_; uint8_t v_isShared_912_; uint8_t v_isSharedCheck_931_; 
v_tail_909_ = lean_ctor_get(v_lambdas_875_, 1);
v_isSharedCheck_931_ = !lean_is_exclusive(v_lambdas_875_);
if (v_isSharedCheck_931_ == 0)
{
lean_object* v_unused_932_; 
v_unused_932_ = lean_ctor_get(v_lambdas_875_, 0);
lean_dec(v_unused_932_);
v___x_911_ = v_lambdas_875_;
v_isShared_912_ = v_isSharedCheck_931_;
goto v_resetjp_910_;
}
else
{
lean_inc(v_tail_909_);
lean_dec(v_lambdas_875_);
v___x_911_ = lean_box(0);
v_isShared_912_ = v_isSharedCheck_931_;
goto v_resetjp_910_;
}
v_resetjp_910_:
{
lean_object* v_previous_913_; lean_object* v_stack_914_; lean_object* v_mctx_915_; lean_object* v_labelledStars_x3f_916_; lean_object* v___x_918_; uint8_t v_isShared_919_; uint8_t v_isSharedCheck_929_; 
v_previous_913_ = lean_ctor_get(v_entry_872_, 0);
v_stack_914_ = lean_ctor_get(v_entry_872_, 1);
v_mctx_915_ = lean_ctor_get(v_entry_872_, 2);
v_labelledStars_x3f_916_ = lean_ctor_get(v_entry_872_, 3);
v_isSharedCheck_929_ = !lean_is_exclusive(v_entry_872_);
if (v_isSharedCheck_929_ == 0)
{
lean_object* v_unused_930_; 
v_unused_930_ = lean_ctor_get(v_entry_872_, 4);
lean_dec(v_unused_930_);
v___x_918_ = v_entry_872_;
v_isShared_919_ = v_isSharedCheck_929_;
goto v_resetjp_917_;
}
else
{
lean_inc(v_labelledStars_x3f_916_);
lean_inc(v_mctx_915_);
lean_inc(v_stack_914_);
lean_inc(v_previous_913_);
lean_dec(v_entry_872_);
v___x_918_ = lean_box(0);
v_isShared_919_ = v_isSharedCheck_929_;
goto v_resetjp_917_;
}
v_resetjp_917_:
{
lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_923_; 
v___x_920_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0));
v___x_921_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(v___x_920_, v_tail_909_);
if (v_isShared_919_ == 0)
{
lean_ctor_set(v___x_918_, 4, v___x_921_);
v___x_923_ = v___x_918_;
goto v_reusejp_922_;
}
else
{
lean_object* v_reuseFailAlloc_928_; 
v_reuseFailAlloc_928_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_928_, 0, v_previous_913_);
lean_ctor_set(v_reuseFailAlloc_928_, 1, v_stack_914_);
lean_ctor_set(v_reuseFailAlloc_928_, 2, v_mctx_915_);
lean_ctor_set(v_reuseFailAlloc_928_, 3, v_labelledStars_x3f_916_);
lean_ctor_set(v_reuseFailAlloc_928_, 4, v___x_921_);
v___x_923_ = v_reuseFailAlloc_928_;
goto v_reusejp_922_;
}
v_reusejp_922_:
{
lean_object* v___x_924_; lean_object* v___x_926_; 
v___x_924_ = lean_box(8);
if (v_isShared_912_ == 0)
{
lean_ctor_set_tag(v___x_911_, 0);
lean_ctor_set(v___x_911_, 1, v___x_923_);
lean_ctor_set(v___x_911_, 0, v___x_924_);
v___x_926_ = v___x_911_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v___x_924_);
lean_ctor_set(v_reuseFailAlloc_927_, 1, v___x_923_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
v_a_883_ = v___x_926_;
goto v___jp_882_;
}
}
}
}
}
}
v___jp_882_:
{
lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; 
v___x_884_ = lean_box(0);
v___x_885_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_885_, 0, v_a_883_);
lean_ctor_set(v___x_885_, 1, v___x_884_);
v___x_886_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_886_, 0, v___x_885_);
return v___x_886_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___lam__0(lean_object* v_body_933_, lean_object* v_lambdas_934_, lean_object* v_entry_935_, uint8_t v_root_936_, lean_object* v_fvar_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_){
_start:
{
lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; 
v___x_944_ = lean_expr_instantiate1(v_body_933_, v_fvar_937_);
v___x_945_ = l_Lean_Expr_fvarId_x21(v_fvar_937_);
v___x_946_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_946_, 0, v___x_945_);
lean_ctor_set(v___x_946_, 1, v_lambdas_934_);
v___x_947_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(v_entry_935_, v_root_936_, v___x_944_, v___x_946_, v___y_938_, v___y_939_, v___y_940_, v___y_941_, v___y_942_);
return v___x_947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___boxed(lean_object* v_entry_948_, lean_object* v_root_949_, lean_object* v_e_950_, lean_object* v_lambdas_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_){
_start:
{
uint8_t v_root_boxed_958_; lean_object* v_res_959_; 
v_root_boxed_958_ = lean_unbox(v_root_949_);
v_res_959_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(v_entry_948_, v_root_boxed_958_, v_e_950_, v_lambdas_951_, v___y_952_, v___y_953_, v___y_954_, v___y_955_, v___y_956_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
lean_dec(v___y_954_);
lean_dec_ref(v___y_953_);
lean_dec(v___y_952_);
return v_res_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(lean_object* v_e_960_, uint8_t v_root_961_, lean_object* v_entry_962_, lean_object* v_a_963_, lean_object* v_a_964_, lean_object* v_a_965_, lean_object* v_a_966_, lean_object* v_a_967_){
_start:
{
lean_object* v___x_969_; lean_object* v___x_970_; 
v___x_969_ = lean_box(0);
v___x_970_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(v_entry_962_, v_root_961_, v_e_960_, v___x_969_, v_a_963_, v_a_964_, v_a_965_, v_a_966_, v_a_967_);
return v___x_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta___boxed(lean_object* v_e_971_, lean_object* v_root_972_, lean_object* v_entry_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_, lean_object* v_a_978_, lean_object* v_a_979_){
_start:
{
uint8_t v_root_boxed_980_; lean_object* v_res_981_; 
v_root_boxed_980_ = lean_unbox(v_root_972_);
v_res_981_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(v_e_971_, v_root_boxed_980_, v_entry_973_, v_a_974_, v_a_975_, v_a_976_, v_a_977_, v_a_978_);
lean_dec(v_a_978_);
lean_dec_ref(v_a_977_);
lean_dec(v_a_976_);
lean_dec_ref(v_a_975_);
lean_dec(v_a_974_);
return v_res_981_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0(lean_object* v_00_u03b1_982_, lean_object* v_name_983_, uint8_t v_bi_984_, lean_object* v_type_985_, lean_object* v_k_986_, uint8_t v_kind_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_){
_start:
{
lean_object* v___x_994_; 
v___x_994_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___redArg(v_name_983_, v_bi_984_, v_type_985_, v_k_986_, v_kind_987_, v___y_988_, v___y_989_, v___y_990_, v___y_991_, v___y_992_);
return v___x_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0___boxed(lean_object* v_00_u03b1_995_, lean_object* v_name_996_, lean_object* v_bi_997_, lean_object* v_type_998_, lean_object* v_k_999_, lean_object* v_kind_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_){
_start:
{
uint8_t v_bi_boxed_1007_; uint8_t v_kind_boxed_1008_; lean_object* v_res_1009_; 
v_bi_boxed_1007_ = lean_unbox(v_bi_997_);
v_kind_boxed_1008_ = lean_unbox(v_kind_1000_);
v_res_1009_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0_spec__0(v_00_u03b1_995_, v_name_996_, v_bi_boxed_1007_, v_type_998_, v_k_999_, v_kind_boxed_1008_, v___y_1001_, v___y_1002_, v___y_1003_, v___y_1004_, v___y_1005_);
lean_dec(v___y_1005_);
lean_dec_ref(v___y_1004_);
lean_dec(v___y_1003_);
lean_dec_ref(v___y_1002_);
lean_dec(v___y_1001_);
return v_res_1009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0(lean_object* v_entry_1010_, uint8_t v_root_1011_, lean_object* v_inst_1012_, lean_object* v_e_1013_, lean_object* v_lambdas_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_){
_start:
{
lean_object* v___x_1021_; 
v___x_1021_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg(v_entry_1010_, v_root_1011_, v_e_1013_, v_lambdas_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
return v___x_1021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___boxed(lean_object* v_entry_1022_, lean_object* v_root_1023_, lean_object* v_inst_1024_, lean_object* v_e_1025_, lean_object* v_lambdas_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_){
_start:
{
uint8_t v_root_boxed_1033_; lean_object* v_res_1034_; 
v_root_boxed_1033_ = lean_unbox(v_root_1023_);
v_res_1034_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0(v_entry_1022_, v_root_boxed_1033_, v_inst_1024_, v_e_1025_, v_lambdas_1026_, v___y_1027_, v___y_1028_, v___y_1029_, v___y_1030_, v___y_1031_);
lean_dec(v___y_1031_);
lean_dec_ref(v___y_1030_);
lean_dec(v___y_1029_);
lean_dec_ref(v___y_1028_);
lean_dec(v___y_1027_);
return v_res_1034_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0(lean_object* v_k_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v_b_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_){
_start:
{
lean_object* v___x_1044_; 
lean_inc(v___y_1042_);
lean_inc_ref(v___y_1041_);
lean_inc(v___y_1040_);
lean_inc_ref(v___y_1039_);
lean_inc(v___y_1036_);
v___x_1044_ = lean_apply_8(v_k_1035_, v_b_1038_, v___y_1036_, v___y_1037_, v___y_1039_, v___y_1040_, v___y_1041_, v___y_1042_, lean_box(0));
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_k_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v_b_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_){
_start:
{
lean_object* v_res_1054_; 
v_res_1054_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0(v_k_1045_, v___y_1046_, v___y_1047_, v_b_1048_, v___y_1049_, v___y_1050_, v___y_1051_, v___y_1052_);
lean_dec(v___y_1052_);
lean_dec_ref(v___y_1051_);
lean_dec(v___y_1050_);
lean_dec_ref(v___y_1049_);
lean_dec(v___y_1046_);
return v_res_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(lean_object* v_name_1055_, uint8_t v_bi_1056_, lean_object* v_type_1057_, lean_object* v_k_1058_, uint8_t v_kind_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_){
_start:
{
lean_object* v___f_1067_; lean_object* v___x_1068_; 
lean_inc(v___y_1060_);
v___f_1067_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_1067_, 0, v_k_1058_);
lean_closure_set(v___f_1067_, 1, v___y_1060_);
lean_closure_set(v___f_1067_, 2, v___y_1061_);
v___x_1068_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1055_, v_bi_1056_, v_type_1057_, v___f_1067_, v_kind_1059_, v___y_1062_, v___y_1063_, v___y_1064_, v___y_1065_);
if (lean_obj_tag(v___x_1068_) == 0)
{
lean_object* v_a_1069_; lean_object* v___x_1071_; uint8_t v_isShared_1072_; uint8_t v_isSharedCheck_1076_; 
v_a_1069_ = lean_ctor_get(v___x_1068_, 0);
v_isSharedCheck_1076_ = !lean_is_exclusive(v___x_1068_);
if (v_isSharedCheck_1076_ == 0)
{
v___x_1071_ = v___x_1068_;
v_isShared_1072_ = v_isSharedCheck_1076_;
goto v_resetjp_1070_;
}
else
{
lean_inc(v_a_1069_);
lean_dec(v___x_1068_);
v___x_1071_ = lean_box(0);
v_isShared_1072_ = v_isSharedCheck_1076_;
goto v_resetjp_1070_;
}
v_resetjp_1070_:
{
lean_object* v___x_1074_; 
if (v_isShared_1072_ == 0)
{
v___x_1074_ = v___x_1071_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v_a_1069_);
v___x_1074_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
return v___x_1074_;
}
}
}
else
{
lean_object* v_a_1077_; lean_object* v___x_1079_; uint8_t v_isShared_1080_; uint8_t v_isSharedCheck_1084_; 
v_a_1077_ = lean_ctor_get(v___x_1068_, 0);
v_isSharedCheck_1084_ = !lean_is_exclusive(v___x_1068_);
if (v_isSharedCheck_1084_ == 0)
{
v___x_1079_ = v___x_1068_;
v_isShared_1080_ = v_isSharedCheck_1084_;
goto v_resetjp_1078_;
}
else
{
lean_inc(v_a_1077_);
lean_dec(v___x_1068_);
v___x_1079_ = lean_box(0);
v_isShared_1080_ = v_isSharedCheck_1084_;
goto v_resetjp_1078_;
}
v_resetjp_1078_:
{
lean_object* v___x_1082_; 
if (v_isShared_1080_ == 0)
{
v___x_1082_ = v___x_1079_;
goto v_reusejp_1081_;
}
else
{
lean_object* v_reuseFailAlloc_1083_; 
v_reuseFailAlloc_1083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1083_, 0, v_a_1077_);
v___x_1082_ = v_reuseFailAlloc_1083_;
goto v_reusejp_1081_;
}
v_reusejp_1081_:
{
return v___x_1082_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg___boxed(lean_object* v_name_1085_, lean_object* v_bi_1086_, lean_object* v_type_1087_, lean_object* v_k_1088_, lean_object* v_kind_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
uint8_t v_bi_boxed_1097_; uint8_t v_kind_boxed_1098_; lean_object* v_res_1099_; 
v_bi_boxed_1097_ = lean_unbox(v_bi_1086_);
v_kind_boxed_1098_ = lean_unbox(v_kind_1089_);
v_res_1099_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(v_name_1085_, v_bi_boxed_1097_, v_type_1087_, v_k_1088_, v_kind_boxed_1098_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
lean_dec(v___y_1093_);
lean_dec_ref(v___y_1092_);
lean_dec(v___y_1090_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0___boxed(lean_object* v_body_1100_, lean_object* v_lambdas_1101_, lean_object* v_root_1102_, lean_object* v_fvar_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_){
_start:
{
uint8_t v_root_boxed_1111_; lean_object* v_res_1112_; 
v_root_boxed_1111_ = lean_unbox(v_root_1102_);
v_res_1112_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0(v_body_1100_, v_lambdas_1101_, v_root_boxed_1111_, v_fvar_1103_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_, v___y_1108_, v___y_1109_);
lean_dec(v___y_1109_);
lean_dec_ref(v___y_1108_);
lean_dec(v___y_1107_);
lean_dec_ref(v___y_1106_);
lean_dec(v___y_1104_);
lean_dec_ref(v_fvar_1103_);
lean_dec_ref(v_body_1100_);
return v_res_1112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(uint8_t v_root_1113_, lean_object* v_e_1114_, lean_object* v_lambdas_1115_, lean_object* v___y_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_){
_start:
{
uint8_t v___x_1123_; 
v___x_1123_ = l_Lean_Meta_DiscrTree_hasNoindexAnnotation(v_e_1114_);
if (v___x_1123_ == 0)
{
lean_object* v___x_1124_; 
v___x_1124_ = l_Lean_Meta_DiscrTree_reduce(v_e_1114_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_);
if (lean_obj_tag(v___x_1124_) == 0)
{
lean_object* v_a_1125_; 
v_a_1125_ = lean_ctor_get(v___x_1124_, 0);
lean_inc(v_a_1125_);
lean_dec_ref_known(v___x_1124_, 1);
if (lean_obj_tag(v_a_1125_) == 6)
{
lean_object* v_binderName_1126_; lean_object* v_binderType_1127_; lean_object* v_body_1128_; uint8_t v_binderInfo_1129_; lean_object* v___x_1130_; lean_object* v___f_1131_; uint8_t v___x_1132_; lean_object* v___x_1133_; 
v_binderName_1126_ = lean_ctor_get(v_a_1125_, 0);
lean_inc(v_binderName_1126_);
v_binderType_1127_ = lean_ctor_get(v_a_1125_, 1);
lean_inc_ref(v_binderType_1127_);
v_body_1128_ = lean_ctor_get(v_a_1125_, 2);
lean_inc_ref(v_body_1128_);
v_binderInfo_1129_ = lean_ctor_get_uint8(v_a_1125_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_a_1125_, 3);
v___x_1130_ = lean_box(v_root_1113_);
v___f_1131_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0___boxed), 11, 3);
lean_closure_set(v___f_1131_, 0, v_body_1128_);
lean_closure_set(v___f_1131_, 1, v_lambdas_1115_);
lean_closure_set(v___f_1131_, 2, v___x_1130_);
v___x_1132_ = 0;
v___x_1133_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(v_binderName_1126_, v_binderInfo_1129_, v_binderType_1127_, v___f_1131_, v___x_1132_, v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_);
return v___x_1133_;
}
else
{
lean_object* v___x_1134_; 
lean_inc(v_lambdas_1115_);
v___x_1134_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go(v_a_1125_, v_lambdas_1115_, v_root_1113_, v___y_1116_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_);
if (lean_obj_tag(v___x_1134_) == 0)
{
lean_object* v_a_1135_; 
v_a_1135_ = lean_ctor_get(v___x_1134_, 0);
lean_inc(v_a_1135_);
if (lean_obj_tag(v_lambdas_1115_) == 0)
{
lean_dec(v_a_1135_);
return v___x_1134_;
}
else
{
lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1175_; 
v_isSharedCheck_1175_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1175_ == 0)
{
lean_object* v_unused_1176_; 
v_unused_1176_ = lean_ctor_get(v___x_1134_, 0);
lean_dec(v_unused_1176_);
v___x_1137_ = v___x_1134_;
v_isShared_1138_ = v_isSharedCheck_1175_;
goto v_resetjp_1136_;
}
else
{
lean_dec(v___x_1134_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1175_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v_snd_1139_; lean_object* v_fst_1140_; lean_object* v___x_1142_; uint8_t v_isShared_1143_; uint8_t v_isSharedCheck_1174_; 
v_snd_1139_ = lean_ctor_get(v_a_1135_, 1);
v_fst_1140_ = lean_ctor_get(v_a_1135_, 0);
v_isSharedCheck_1174_ = !lean_is_exclusive(v_a_1135_);
if (v_isSharedCheck_1174_ == 0)
{
v___x_1142_ = v_a_1135_;
v_isShared_1143_ = v_isSharedCheck_1174_;
goto v_resetjp_1141_;
}
else
{
lean_inc(v_snd_1139_);
lean_inc(v_fst_1140_);
lean_dec(v_a_1135_);
v___x_1142_ = lean_box(0);
v_isShared_1143_ = v_isSharedCheck_1174_;
goto v_resetjp_1141_;
}
v_resetjp_1141_:
{
lean_object* v_tail_1144_; lean_object* v___x_1146_; uint8_t v_isShared_1147_; uint8_t v_isSharedCheck_1172_; 
v_tail_1144_ = lean_ctor_get(v_lambdas_1115_, 1);
v_isSharedCheck_1172_ = !lean_is_exclusive(v_lambdas_1115_);
if (v_isSharedCheck_1172_ == 0)
{
lean_object* v_unused_1173_; 
v_unused_1173_ = lean_ctor_get(v_lambdas_1115_, 0);
lean_dec(v_unused_1173_);
v___x_1146_ = v_lambdas_1115_;
v_isShared_1147_ = v_isSharedCheck_1172_;
goto v_resetjp_1145_;
}
else
{
lean_inc(v_tail_1144_);
lean_dec(v_lambdas_1115_);
v___x_1146_ = lean_box(0);
v_isShared_1147_ = v_isSharedCheck_1172_;
goto v_resetjp_1145_;
}
v_resetjp_1145_:
{
lean_object* v_previous_1148_; lean_object* v_stack_1149_; lean_object* v_mctx_1150_; lean_object* v_labelledStars_x3f_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1170_; 
v_previous_1148_ = lean_ctor_get(v_snd_1139_, 0);
v_stack_1149_ = lean_ctor_get(v_snd_1139_, 1);
v_mctx_1150_ = lean_ctor_get(v_snd_1139_, 2);
v_labelledStars_x3f_1151_ = lean_ctor_get(v_snd_1139_, 3);
v_isSharedCheck_1170_ = !lean_is_exclusive(v_snd_1139_);
if (v_isSharedCheck_1170_ == 0)
{
lean_object* v_unused_1171_; 
v_unused_1171_ = lean_ctor_get(v_snd_1139_, 4);
lean_dec(v_unused_1171_);
v___x_1153_ = v_snd_1139_;
v_isShared_1154_ = v_isSharedCheck_1170_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_labelledStars_x3f_1151_);
lean_inc(v_mctx_1150_);
lean_inc(v_stack_1149_);
lean_inc(v_previous_1148_);
lean_dec(v_snd_1139_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1170_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1155_; lean_object* v___x_1157_; 
v___x_1155_ = lean_box(0);
if (v_isShared_1147_ == 0)
{
lean_ctor_set(v___x_1146_, 1, v___x_1155_);
lean_ctor_set(v___x_1146_, 0, v_fst_1140_);
v___x_1157_ = v___x_1146_;
goto v_reusejp_1156_;
}
else
{
lean_object* v_reuseFailAlloc_1169_; 
v_reuseFailAlloc_1169_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1169_, 0, v_fst_1140_);
lean_ctor_set(v_reuseFailAlloc_1169_, 1, v___x_1155_);
v___x_1157_ = v_reuseFailAlloc_1169_;
goto v_reusejp_1156_;
}
v_reusejp_1156_:
{
lean_object* v___x_1158_; lean_object* v___x_1160_; 
v___x_1158_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(v___x_1157_, v_tail_1144_);
if (v_isShared_1154_ == 0)
{
lean_ctor_set(v___x_1153_, 4, v___x_1158_);
v___x_1160_ = v___x_1153_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1168_; 
v_reuseFailAlloc_1168_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1168_, 0, v_previous_1148_);
lean_ctor_set(v_reuseFailAlloc_1168_, 1, v_stack_1149_);
lean_ctor_set(v_reuseFailAlloc_1168_, 2, v_mctx_1150_);
lean_ctor_set(v_reuseFailAlloc_1168_, 3, v_labelledStars_x3f_1151_);
lean_ctor_set(v_reuseFailAlloc_1168_, 4, v___x_1158_);
v___x_1160_ = v_reuseFailAlloc_1168_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
lean_object* v___x_1161_; lean_object* v___x_1163_; 
v___x_1161_ = lean_box(8);
if (v_isShared_1143_ == 0)
{
lean_ctor_set(v___x_1142_, 1, v___x_1160_);
lean_ctor_set(v___x_1142_, 0, v___x_1161_);
v___x_1163_ = v___x_1142_;
goto v_reusejp_1162_;
}
else
{
lean_object* v_reuseFailAlloc_1167_; 
v_reuseFailAlloc_1167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1167_, 0, v___x_1161_);
lean_ctor_set(v_reuseFailAlloc_1167_, 1, v___x_1160_);
v___x_1163_ = v_reuseFailAlloc_1167_;
goto v_reusejp_1162_;
}
v_reusejp_1162_:
{
lean_object* v___x_1165_; 
if (v_isShared_1138_ == 0)
{
lean_ctor_set(v___x_1137_, 0, v___x_1163_);
v___x_1165_ = v___x_1137_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v___x_1163_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
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
lean_dec(v_lambdas_1115_);
return v___x_1134_;
}
}
}
else
{
lean_object* v_a_1177_; lean_object* v___x_1179_; uint8_t v_isShared_1180_; uint8_t v_isSharedCheck_1184_; 
lean_dec_ref(v___y_1117_);
lean_dec(v_lambdas_1115_);
v_a_1177_ = lean_ctor_get(v___x_1124_, 0);
v_isSharedCheck_1184_ = !lean_is_exclusive(v___x_1124_);
if (v_isSharedCheck_1184_ == 0)
{
v___x_1179_ = v___x_1124_;
v_isShared_1180_ = v_isSharedCheck_1184_;
goto v_resetjp_1178_;
}
else
{
lean_inc(v_a_1177_);
lean_dec(v___x_1124_);
v___x_1179_ = lean_box(0);
v_isShared_1180_ = v_isSharedCheck_1184_;
goto v_resetjp_1178_;
}
v_resetjp_1178_:
{
lean_object* v___x_1182_; 
if (v_isShared_1180_ == 0)
{
v___x_1182_ = v___x_1179_;
goto v_reusejp_1181_;
}
else
{
lean_object* v_reuseFailAlloc_1183_; 
v_reuseFailAlloc_1183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1183_, 0, v_a_1177_);
v___x_1182_ = v_reuseFailAlloc_1183_;
goto v_reusejp_1181_;
}
v_reusejp_1181_:
{
return v___x_1182_;
}
}
}
}
else
{
lean_object* v___x_1185_; 
lean_dec_ref(v_e_1114_);
v___x_1185_ = lean_box(0);
if (lean_obj_tag(v_lambdas_1115_) == 0)
{
lean_object* v___x_1186_; lean_object* v___x_1187_; 
v___x_1186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1186_, 0, v___x_1185_);
lean_ctor_set(v___x_1186_, 1, v___y_1117_);
v___x_1187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1187_, 0, v___x_1186_);
return v___x_1187_;
}
else
{
lean_object* v_tail_1188_; lean_object* v_previous_1189_; lean_object* v_stack_1190_; lean_object* v_mctx_1191_; lean_object* v_labelledStars_x3f_1192_; lean_object* v___x_1194_; uint8_t v_isShared_1195_; uint8_t v_isSharedCheck_1204_; 
v_tail_1188_ = lean_ctor_get(v_lambdas_1115_, 1);
lean_inc(v_tail_1188_);
lean_dec_ref_known(v_lambdas_1115_, 2);
v_previous_1189_ = lean_ctor_get(v___y_1117_, 0);
v_stack_1190_ = lean_ctor_get(v___y_1117_, 1);
v_mctx_1191_ = lean_ctor_get(v___y_1117_, 2);
v_labelledStars_x3f_1192_ = lean_ctor_get(v___y_1117_, 3);
v_isSharedCheck_1204_ = !lean_is_exclusive(v___y_1117_);
if (v_isSharedCheck_1204_ == 0)
{
lean_object* v_unused_1205_; 
v_unused_1205_ = lean_ctor_get(v___y_1117_, 4);
lean_dec(v_unused_1205_);
v___x_1194_ = v___y_1117_;
v_isShared_1195_ = v_isSharedCheck_1204_;
goto v_resetjp_1193_;
}
else
{
lean_inc(v_labelledStars_x3f_1192_);
lean_inc(v_mctx_1191_);
lean_inc(v_stack_1190_);
lean_inc(v_previous_1189_);
lean_dec(v___y_1117_);
v___x_1194_ = lean_box(0);
v_isShared_1195_ = v_isSharedCheck_1204_;
goto v_resetjp_1193_;
}
v_resetjp_1193_:
{
lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1199_; 
v___x_1196_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta_spec__0___redArg___closed__0));
v___x_1197_ = lp_mathlib_List_foldl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_etaPossibilities_spec__0(v___x_1196_, v_tail_1188_);
if (v_isShared_1195_ == 0)
{
lean_ctor_set(v___x_1194_, 4, v___x_1197_);
v___x_1199_ = v___x_1194_;
goto v_reusejp_1198_;
}
else
{
lean_object* v_reuseFailAlloc_1203_; 
v_reuseFailAlloc_1203_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1203_, 0, v_previous_1189_);
lean_ctor_set(v_reuseFailAlloc_1203_, 1, v_stack_1190_);
lean_ctor_set(v_reuseFailAlloc_1203_, 2, v_mctx_1191_);
lean_ctor_set(v_reuseFailAlloc_1203_, 3, v_labelledStars_x3f_1192_);
lean_ctor_set(v_reuseFailAlloc_1203_, 4, v___x_1197_);
v___x_1199_ = v_reuseFailAlloc_1203_;
goto v_reusejp_1198_;
}
v_reusejp_1198_:
{
lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v___x_1200_ = lean_box(8);
v___x_1201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1201_, 0, v___x_1200_);
lean_ctor_set(v___x_1201_, 1, v___x_1199_);
v___x_1202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1201_);
return v___x_1202_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___lam__0(lean_object* v_body_1206_, lean_object* v_lambdas_1207_, uint8_t v_root_1208_, lean_object* v_fvar_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_){
_start:
{
lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; 
v___x_1217_ = lean_expr_instantiate1(v_body_1206_, v_fvar_1209_);
v___x_1218_ = l_Lean_Expr_fvarId_x21(v_fvar_1209_);
v___x_1219_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1218_);
lean_ctor_set(v___x_1219_, 1, v_lambdas_1207_);
v___x_1220_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(v_root_1208_, v___x_1217_, v___x_1219_, v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_, v___y_1214_, v___y_1215_);
return v___x_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg___boxed(lean_object* v_root_1221_, lean_object* v_e_1222_, lean_object* v_lambdas_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_){
_start:
{
uint8_t v_root_boxed_1231_; lean_object* v_res_1232_; 
v_root_boxed_1231_ = lean_unbox(v_root_1221_);
v_res_1232_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(v_root_boxed_1231_, v_e_1222_, v_lambdas_1223_, v___y_1224_, v___y_1225_, v___y_1226_, v___y_1227_, v___y_1228_, v___y_1229_);
lean_dec(v___y_1229_);
lean_dec_ref(v___y_1228_);
lean_dec(v___y_1227_);
lean_dec_ref(v___y_1226_);
lean_dec(v___y_1224_);
return v_res_1232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(lean_object* v_e_1233_, uint8_t v_root_1234_, lean_object* v_a_1235_, lean_object* v_a_1236_, lean_object* v_a_1237_, lean_object* v_a_1238_, lean_object* v_a_1239_, lean_object* v_a_1240_){
_start:
{
lean_object* v___x_1242_; lean_object* v___x_1243_; 
v___x_1242_ = lean_box(0);
v___x_1243_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(v_root_1234_, v_e_1233_, v___x_1242_, v_a_1235_, v_a_1236_, v_a_1237_, v_a_1238_, v_a_1239_, v_a_1240_);
return v___x_1243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep___boxed(lean_object* v_e_1244_, lean_object* v_root_1245_, lean_object* v_a_1246_, lean_object* v_a_1247_, lean_object* v_a_1248_, lean_object* v_a_1249_, lean_object* v_a_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_){
_start:
{
uint8_t v_root_boxed_1253_; lean_object* v_res_1254_; 
v_root_boxed_1253_ = lean_unbox(v_root_1245_);
v_res_1254_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(v_e_1244_, v_root_boxed_1253_, v_a_1246_, v_a_1247_, v_a_1248_, v_a_1249_, v_a_1250_, v_a_1251_);
lean_dec(v_a_1251_);
lean_dec_ref(v_a_1250_);
lean_dec(v_a_1249_);
lean_dec_ref(v_a_1248_);
lean_dec(v_a_1246_);
return v_res_1254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0(lean_object* v_00_u03b1_1255_, lean_object* v_name_1256_, uint8_t v_bi_1257_, lean_object* v_type_1258_, lean_object* v_k_1259_, uint8_t v_kind_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_){
_start:
{
lean_object* v___x_1268_; 
v___x_1268_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___redArg(v_name_1256_, v_bi_1257_, v_type_1258_, v_k_1259_, v_kind_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_);
return v___x_1268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1269_, lean_object* v_name_1270_, lean_object* v_bi_1271_, lean_object* v_type_1272_, lean_object* v_k_1273_, lean_object* v_kind_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_){
_start:
{
uint8_t v_bi_boxed_1282_; uint8_t v_kind_boxed_1283_; lean_object* v_res_1284_; 
v_bi_boxed_1282_ = lean_unbox(v_bi_1271_);
v_kind_boxed_1283_ = lean_unbox(v_kind_1274_);
v_res_1284_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0_spec__0(v_00_u03b1_1269_, v_name_1270_, v_bi_boxed_1282_, v_type_1272_, v_k_1273_, v_kind_boxed_1283_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_, v___y_1279_, v___y_1280_);
lean_dec(v___y_1280_);
lean_dec_ref(v___y_1279_);
lean_dec(v___y_1278_);
lean_dec_ref(v___y_1277_);
lean_dec(v___y_1275_);
return v_res_1284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0(uint8_t v_root_1285_, lean_object* v_inst_1286_, lean_object* v_e_1287_, lean_object* v_lambdas_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_){
_start:
{
lean_object* v___x_1296_; 
v___x_1296_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___redArg(v_root_1285_, v_e_1287_, v_lambdas_1288_, v___y_1289_, v___y_1290_, v___y_1291_, v___y_1292_, v___y_1293_, v___y_1294_);
return v___x_1296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0___boxed(lean_object* v_root_1297_, lean_object* v_inst_1298_, lean_object* v_e_1299_, lean_object* v_lambdas_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_){
_start:
{
uint8_t v_root_boxed_1308_; lean_object* v_res_1309_; 
v_root_boxed_1308_ = lean_unbox(v_root_1297_);
v_res_1309_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_lambdaTelescopeReduce___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep_spec__0(v_root_boxed_1308_, v_inst_1298_, v_e_1299_, v_lambdas_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_, v___y_1306_);
lean_dec(v___y_1306_);
lean_dec_ref(v___y_1305_);
lean_dec(v___y_1304_);
lean_dec_ref(v___y_1303_);
lean_dec(v___y_1301_);
return v_res_1309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux(lean_object* v_e_1310_, uint8_t v_labelledStars_1311_, lean_object* v_a_1312_, lean_object* v_a_1313_, lean_object* v_a_1314_, lean_object* v_a_1315_){
_start:
{
lean_object* v___x_1317_; 
v___x_1317_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(v_labelledStars_1311_, v_a_1313_);
if (lean_obj_tag(v___x_1317_) == 0)
{
lean_object* v_a_1318_; uint8_t v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; 
v_a_1318_ = lean_ctor_get(v___x_1317_, 0);
lean_inc(v_a_1318_);
lean_dec_ref_known(v___x_1317_, 1);
v___x_1319_ = 1;
v___x_1320_ = lean_box(0);
v___x_1321_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(v_e_1310_, v___x_1319_, v_a_1318_, v___x_1320_, v_a_1312_, v_a_1313_, v_a_1314_, v_a_1315_);
return v___x_1321_;
}
else
{
lean_object* v_a_1322_; lean_object* v___x_1324_; uint8_t v_isShared_1325_; uint8_t v_isSharedCheck_1329_; 
lean_dec_ref(v_e_1310_);
v_a_1322_ = lean_ctor_get(v___x_1317_, 0);
v_isSharedCheck_1329_ = !lean_is_exclusive(v___x_1317_);
if (v_isSharedCheck_1329_ == 0)
{
v___x_1324_ = v___x_1317_;
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
else
{
lean_inc(v_a_1322_);
lean_dec(v___x_1317_);
v___x_1324_ = lean_box(0);
v_isShared_1325_ = v_isSharedCheck_1329_;
goto v_resetjp_1323_;
}
v_resetjp_1323_:
{
lean_object* v___x_1327_; 
if (v_isShared_1325_ == 0)
{
v___x_1327_ = v___x_1324_;
goto v_reusejp_1326_;
}
else
{
lean_object* v_reuseFailAlloc_1328_; 
v_reuseFailAlloc_1328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1328_, 0, v_a_1322_);
v___x_1327_ = v_reuseFailAlloc_1328_;
goto v_reusejp_1326_;
}
v_reusejp_1326_:
{
return v___x_1327_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux___boxed(lean_object* v_e_1330_, lean_object* v_labelledStars_1331_, lean_object* v_a_1332_, lean_object* v_a_1333_, lean_object* v_a_1334_, lean_object* v_a_1335_, lean_object* v_a_1336_){
_start:
{
uint8_t v_labelledStars_boxed_1337_; lean_object* v_res_1338_; 
v_labelledStars_boxed_1337_ = lean_unbox(v_labelledStars_1331_);
v_res_1338_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEtaAux(v_e_1330_, v_labelledStars_boxed_1337_, v_a_1332_, v_a_1333_, v_a_1334_, v_a_1335_);
lean_dec(v_a_1335_);
lean_dec_ref(v_a_1334_);
lean_dec(v_a_1333_);
lean_dec_ref(v_a_1332_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(lean_object* v_e_1339_, uint8_t v_labelledStars_1340_, lean_object* v_a_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_){
_start:
{
lean_object* v_keyedConfig_1346_; uint8_t v_trackZetaDelta_1347_; lean_object* v_zetaDeltaSet_1348_; lean_object* v_lctx_1349_; lean_object* v_localInstances_1350_; lean_object* v_defEqCtx_x3f_1351_; lean_object* v_synthPendingDepth_1352_; lean_object* v_customCanUnfoldPredicate_x3f_1353_; uint8_t v_univApprox_1354_; uint8_t v_inTypeClassResolution_1355_; uint8_t v_cacheInferType_1356_; lean_object* v___x_1357_; 
v_keyedConfig_1346_ = lean_ctor_get(v_a_1341_, 0);
v_trackZetaDelta_1347_ = lean_ctor_get_uint8(v_a_1341_, sizeof(void*)*7);
v_zetaDeltaSet_1348_ = lean_ctor_get(v_a_1341_, 1);
v_lctx_1349_ = lean_ctor_get(v_a_1341_, 2);
v_localInstances_1350_ = lean_ctor_get(v_a_1341_, 3);
v_defEqCtx_x3f_1351_ = lean_ctor_get(v_a_1341_, 4);
v_synthPendingDepth_1352_ = lean_ctor_get(v_a_1341_, 5);
v_customCanUnfoldPredicate_x3f_1353_ = lean_ctor_get(v_a_1341_, 6);
v_univApprox_1354_ = lean_ctor_get_uint8(v_a_1341_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1355_ = lean_ctor_get_uint8(v_a_1341_, sizeof(void*)*7 + 2);
v_cacheInferType_1356_ = lean_ctor_get_uint8(v_a_1341_, sizeof(void*)*7 + 3);
v___x_1357_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(v_labelledStars_1340_, v_a_1342_);
if (lean_obj_tag(v___x_1357_) == 0)
{
lean_object* v_a_1358_; uint8_t v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; uint8_t v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; 
v_a_1358_ = lean_ctor_get(v___x_1357_, 0);
lean_inc(v_a_1358_);
lean_dec_ref_known(v___x_1357_, 1);
v___x_1359_ = 2;
lean_inc_ref(v_keyedConfig_1346_);
v___x_1360_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1359_, v_keyedConfig_1346_);
lean_inc(v_customCanUnfoldPredicate_x3f_1353_);
lean_inc(v_synthPendingDepth_1352_);
lean_inc(v_defEqCtx_x3f_1351_);
lean_inc_ref(v_localInstances_1350_);
lean_inc_ref(v_lctx_1349_);
lean_inc(v_zetaDeltaSet_1348_);
v___x_1361_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1361_, 0, v___x_1360_);
lean_ctor_set(v___x_1361_, 1, v_zetaDeltaSet_1348_);
lean_ctor_set(v___x_1361_, 2, v_lctx_1349_);
lean_ctor_set(v___x_1361_, 3, v_localInstances_1350_);
lean_ctor_set(v___x_1361_, 4, v_defEqCtx_x3f_1351_);
lean_ctor_set(v___x_1361_, 5, v_synthPendingDepth_1352_);
lean_ctor_set(v___x_1361_, 6, v_customCanUnfoldPredicate_x3f_1353_);
lean_ctor_set_uint8(v___x_1361_, sizeof(void*)*7, v_trackZetaDelta_1347_);
lean_ctor_set_uint8(v___x_1361_, sizeof(void*)*7 + 1, v_univApprox_1354_);
lean_ctor_set_uint8(v___x_1361_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1355_);
lean_ctor_set_uint8(v___x_1361_, sizeof(void*)*7 + 3, v_cacheInferType_1356_);
v___x_1362_ = 1;
v___x_1363_ = lean_box(0);
v___x_1364_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(v_e_1339_, v___x_1362_, v_a_1358_, v___x_1363_, v___x_1361_, v_a_1342_, v_a_1343_, v_a_1344_);
lean_dec_ref_known(v___x_1361_, 7);
return v___x_1364_;
}
else
{
lean_object* v_a_1365_; lean_object* v___x_1367_; uint8_t v_isShared_1368_; uint8_t v_isSharedCheck_1372_; 
lean_dec_ref(v_e_1339_);
v_a_1365_ = lean_ctor_get(v___x_1357_, 0);
v_isSharedCheck_1372_ = !lean_is_exclusive(v___x_1357_);
if (v_isSharedCheck_1372_ == 0)
{
v___x_1367_ = v___x_1357_;
v_isShared_1368_ = v_isSharedCheck_1372_;
goto v_resetjp_1366_;
}
else
{
lean_inc(v_a_1365_);
lean_dec(v___x_1357_);
v___x_1367_ = lean_box(0);
v_isShared_1368_ = v_isSharedCheck_1372_;
goto v_resetjp_1366_;
}
v_resetjp_1366_:
{
lean_object* v___x_1370_; 
if (v_isShared_1368_ == 0)
{
v___x_1370_ = v___x_1367_;
goto v_reusejp_1369_;
}
else
{
lean_object* v_reuseFailAlloc_1371_; 
v_reuseFailAlloc_1371_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1371_, 0, v_a_1365_);
v___x_1370_ = v_reuseFailAlloc_1371_;
goto v_reusejp_1369_;
}
v_reusejp_1369_:
{
return v___x_1370_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta___boxed(lean_object* v_e_1373_, lean_object* v_labelledStars_1374_, lean_object* v_a_1375_, lean_object* v_a_1376_, lean_object* v_a_1377_, lean_object* v_a_1378_, lean_object* v_a_1379_){
_start:
{
uint8_t v_labelledStars_boxed_1380_; lean_object* v_res_1381_; 
v_labelledStars_boxed_1380_ = lean_unbox(v_labelledStars_1374_);
v_res_1381_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_initializeLazyEntryWithEta(v_e_1373_, v_labelledStars_boxed_1380_, v_a_1375_, v_a_1376_, v_a_1377_, v_a_1378_);
lean_dec(v_a_1378_);
lean_dec_ref(v_a_1377_);
lean_dec(v_a_1376_);
lean_dec_ref(v_a_1375_);
return v_res_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(lean_object* v_e_1382_, uint8_t v_labelledStars_1383_, lean_object* v_a_1384_, lean_object* v_a_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_){
_start:
{
lean_object* v___x_1389_; 
v___x_1389_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(v_labelledStars_1383_, v_a_1385_);
if (lean_obj_tag(v___x_1389_) == 0)
{
lean_object* v_a_1390_; uint8_t v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; 
v_a_1390_ = lean_ctor_get(v___x_1389_, 0);
lean_inc(v_a_1390_);
lean_dec_ref_known(v___x_1389_, 1);
v___x_1391_ = 1;
v___x_1392_ = lean_box(0);
v___x_1393_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(v_e_1382_, v___x_1391_, v___x_1392_, v_a_1390_, v_a_1384_, v_a_1385_, v_a_1386_, v_a_1387_);
return v___x_1393_;
}
else
{
lean_object* v_a_1394_; lean_object* v___x_1396_; uint8_t v_isShared_1397_; uint8_t v_isSharedCheck_1401_; 
lean_dec_ref(v_e_1382_);
v_a_1394_ = lean_ctor_get(v___x_1389_, 0);
v_isSharedCheck_1401_ = !lean_is_exclusive(v___x_1389_);
if (v_isSharedCheck_1401_ == 0)
{
v___x_1396_ = v___x_1389_;
v_isShared_1397_ = v_isSharedCheck_1401_;
goto v_resetjp_1395_;
}
else
{
lean_inc(v_a_1394_);
lean_dec(v___x_1389_);
v___x_1396_ = lean_box(0);
v_isShared_1397_ = v_isSharedCheck_1401_;
goto v_resetjp_1395_;
}
v_resetjp_1395_:
{
lean_object* v___x_1399_; 
if (v_isShared_1397_ == 0)
{
v___x_1399_ = v___x_1396_;
goto v_reusejp_1398_;
}
else
{
lean_object* v_reuseFailAlloc_1400_; 
v_reuseFailAlloc_1400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1400_, 0, v_a_1394_);
v___x_1399_ = v_reuseFailAlloc_1400_;
goto v_reusejp_1398_;
}
v_reusejp_1398_:
{
return v___x_1399_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry___boxed(lean_object* v_e_1402_, lean_object* v_labelledStars_1403_, lean_object* v_a_1404_, lean_object* v_a_1405_, lean_object* v_a_1406_, lean_object* v_a_1407_, lean_object* v_a_1408_){
_start:
{
uint8_t v_labelledStars_boxed_1409_; lean_object* v_res_1410_; 
v_labelledStars_boxed_1409_ = lean_unbox(v_labelledStars_1403_);
v_res_1410_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(v_e_1402_, v_labelledStars_boxed_1409_, v_a_1404_, v_a_1405_, v_a_1406_, v_a_1407_);
lean_dec(v_a_1407_);
lean_dec_ref(v_a_1406_);
lean_dec(v_a_1405_);
lean_dec_ref(v_a_1404_);
return v_res_1410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(lean_object* v_lctx_1411_, lean_object* v_localInsts_1412_, lean_object* v_x_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_){
_start:
{
lean_object* v___x_1419_; 
v___x_1419_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_1411_, v_localInsts_1412_, v_x_1413_, v___y_1414_, v___y_1415_, v___y_1416_, v___y_1417_);
if (lean_obj_tag(v___x_1419_) == 0)
{
lean_object* v_a_1420_; lean_object* v___x_1422_; uint8_t v_isShared_1423_; uint8_t v_isSharedCheck_1427_; 
v_a_1420_ = lean_ctor_get(v___x_1419_, 0);
v_isSharedCheck_1427_ = !lean_is_exclusive(v___x_1419_);
if (v_isSharedCheck_1427_ == 0)
{
v___x_1422_ = v___x_1419_;
v_isShared_1423_ = v_isSharedCheck_1427_;
goto v_resetjp_1421_;
}
else
{
lean_inc(v_a_1420_);
lean_dec(v___x_1419_);
v___x_1422_ = lean_box(0);
v_isShared_1423_ = v_isSharedCheck_1427_;
goto v_resetjp_1421_;
}
v_resetjp_1421_:
{
lean_object* v___x_1425_; 
if (v_isShared_1423_ == 0)
{
v___x_1425_ = v___x_1422_;
goto v_reusejp_1424_;
}
else
{
lean_object* v_reuseFailAlloc_1426_; 
v_reuseFailAlloc_1426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1426_, 0, v_a_1420_);
v___x_1425_ = v_reuseFailAlloc_1426_;
goto v_reusejp_1424_;
}
v_reusejp_1424_:
{
return v___x_1425_;
}
}
}
else
{
lean_object* v_a_1428_; lean_object* v___x_1430_; uint8_t v_isShared_1431_; uint8_t v_isSharedCheck_1435_; 
v_a_1428_ = lean_ctor_get(v___x_1419_, 0);
v_isSharedCheck_1435_ = !lean_is_exclusive(v___x_1419_);
if (v_isSharedCheck_1435_ == 0)
{
v___x_1430_ = v___x_1419_;
v_isShared_1431_ = v_isSharedCheck_1435_;
goto v_resetjp_1429_;
}
else
{
lean_inc(v_a_1428_);
lean_dec(v___x_1419_);
v___x_1430_ = lean_box(0);
v_isShared_1431_ = v_isSharedCheck_1435_;
goto v_resetjp_1429_;
}
v_resetjp_1429_:
{
lean_object* v___x_1433_; 
if (v_isShared_1431_ == 0)
{
v___x_1433_ = v___x_1430_;
goto v_reusejp_1432_;
}
else
{
lean_object* v_reuseFailAlloc_1434_; 
v_reuseFailAlloc_1434_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1434_, 0, v_a_1428_);
v___x_1433_ = v_reuseFailAlloc_1434_;
goto v_reusejp_1432_;
}
v_reusejp_1432_:
{
return v___x_1433_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg___boxed(lean_object* v_lctx_1436_, lean_object* v_localInsts_1437_, lean_object* v_x_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_){
_start:
{
lean_object* v_res_1444_; 
v_res_1444_ = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(v_lctx_1436_, v_localInsts_1437_, v_x_1438_, v___y_1439_, v___y_1440_, v___y_1441_, v___y_1442_);
lean_dec(v___y_1442_);
lean_dec_ref(v___y_1441_);
lean_dec(v___y_1440_);
lean_dec_ref(v___y_1439_);
return v_res_1444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0(lean_object* v_00_u03b1_1445_, lean_object* v_lctx_1446_, lean_object* v_localInsts_1447_, lean_object* v_x_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_){
_start:
{
lean_object* v___x_1454_; 
v___x_1454_ = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(v_lctx_1446_, v_localInsts_1447_, v_x_1448_, v___y_1449_, v___y_1450_, v___y_1451_, v___y_1452_);
return v___x_1454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___boxed(lean_object* v_00_u03b1_1455_, lean_object* v_lctx_1456_, lean_object* v_localInsts_1457_, lean_object* v_x_1458_, lean_object* v___y_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_){
_start:
{
lean_object* v_res_1464_; 
v_res_1464_ = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0(v_00_u03b1_1455_, v_lctx_1456_, v_localInsts_1457_, v_x_1458_, v___y_1459_, v___y_1460_, v___y_1461_, v___y_1462_);
lean_dec(v___y_1462_);
lean_dec_ref(v___y_1461_);
lean_dec(v___y_1460_);
lean_dec_ref(v___y_1459_);
return v_res_1464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0(lean_object* v_cfg_1465_, uint8_t v_eta_1466_, lean_object* v_expr_1467_, lean_object* v_bvars_1468_, lean_object* v_entry_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_){
_start:
{
uint8_t v_trackZetaDelta_1475_; lean_object* v_zetaDeltaSet_1476_; lean_object* v_lctx_1477_; lean_object* v_localInstances_1478_; lean_object* v_defEqCtx_x3f_1479_; lean_object* v_synthPendingDepth_1480_; lean_object* v_customCanUnfoldPredicate_x3f_1481_; uint8_t v_univApprox_1482_; uint8_t v_inTypeClassResolution_1483_; uint8_t v_cacheInferType_1484_; lean_object* v___x_1486_; uint8_t v_isShared_1487_; uint8_t v_isSharedCheck_1532_; 
v_trackZetaDelta_1475_ = lean_ctor_get_uint8(v___y_1470_, sizeof(void*)*7);
v_zetaDeltaSet_1476_ = lean_ctor_get(v___y_1470_, 1);
v_lctx_1477_ = lean_ctor_get(v___y_1470_, 2);
v_localInstances_1478_ = lean_ctor_get(v___y_1470_, 3);
v_defEqCtx_x3f_1479_ = lean_ctor_get(v___y_1470_, 4);
v_synthPendingDepth_1480_ = lean_ctor_get(v___y_1470_, 5);
v_customCanUnfoldPredicate_x3f_1481_ = lean_ctor_get(v___y_1470_, 6);
v_univApprox_1482_ = lean_ctor_get_uint8(v___y_1470_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1483_ = lean_ctor_get_uint8(v___y_1470_, sizeof(void*)*7 + 2);
v_cacheInferType_1484_ = lean_ctor_get_uint8(v___y_1470_, sizeof(void*)*7 + 3);
v_isSharedCheck_1532_ = !lean_is_exclusive(v___y_1470_);
if (v_isSharedCheck_1532_ == 0)
{
lean_object* v_unused_1533_; 
v_unused_1533_ = lean_ctor_get(v___y_1470_, 0);
lean_dec(v_unused_1533_);
v___x_1486_ = v___y_1470_;
v_isShared_1487_ = v_isSharedCheck_1532_;
goto v_resetjp_1485_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1481_);
lean_inc(v_synthPendingDepth_1480_);
lean_inc(v_defEqCtx_x3f_1479_);
lean_inc(v_localInstances_1478_);
lean_inc(v_lctx_1477_);
lean_inc(v_zetaDeltaSet_1476_);
lean_dec(v___y_1470_);
v___x_1486_ = lean_box(0);
v_isShared_1487_ = v_isSharedCheck_1532_;
goto v_resetjp_1485_;
}
v_resetjp_1485_:
{
uint64_t v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1491_; 
v___x_1488_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v_cfg_1465_);
v___x_1489_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1489_, 0, v_cfg_1465_);
lean_ctor_set_uint64(v___x_1489_, sizeof(void*)*1, v___x_1488_);
if (v_isShared_1487_ == 0)
{
lean_ctor_set(v___x_1486_, 0, v___x_1489_);
v___x_1491_ = v___x_1486_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1531_; 
v_reuseFailAlloc_1531_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1531_, 0, v___x_1489_);
lean_ctor_set(v_reuseFailAlloc_1531_, 1, v_zetaDeltaSet_1476_);
lean_ctor_set(v_reuseFailAlloc_1531_, 2, v_lctx_1477_);
lean_ctor_set(v_reuseFailAlloc_1531_, 3, v_localInstances_1478_);
lean_ctor_set(v_reuseFailAlloc_1531_, 4, v_defEqCtx_x3f_1479_);
lean_ctor_set(v_reuseFailAlloc_1531_, 5, v_synthPendingDepth_1480_);
lean_ctor_set(v_reuseFailAlloc_1531_, 6, v_customCanUnfoldPredicate_x3f_1481_);
lean_ctor_set_uint8(v_reuseFailAlloc_1531_, sizeof(void*)*7, v_trackZetaDelta_1475_);
lean_ctor_set_uint8(v_reuseFailAlloc_1531_, sizeof(void*)*7 + 1, v_univApprox_1482_);
lean_ctor_set_uint8(v_reuseFailAlloc_1531_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1483_);
lean_ctor_set_uint8(v_reuseFailAlloc_1531_, sizeof(void*)*7 + 3, v_cacheInferType_1484_);
v___x_1491_ = v_reuseFailAlloc_1531_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
if (v_eta_1466_ == 0)
{
lean_object* v___x_1492_; 
v___x_1492_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStep(v_expr_1467_, v_eta_1466_, v_bvars_1468_, v_entry_1469_, v___x_1491_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec_ref(v___x_1491_);
if (lean_obj_tag(v___x_1492_) == 0)
{
lean_object* v_a_1493_; lean_object* v___x_1495_; uint8_t v_isShared_1496_; uint8_t v_isSharedCheck_1503_; 
v_a_1493_ = lean_ctor_get(v___x_1492_, 0);
v_isSharedCheck_1503_ = !lean_is_exclusive(v___x_1492_);
if (v_isSharedCheck_1503_ == 0)
{
v___x_1495_ = v___x_1492_;
v_isShared_1496_ = v_isSharedCheck_1503_;
goto v_resetjp_1494_;
}
else
{
lean_inc(v_a_1493_);
lean_dec(v___x_1492_);
v___x_1495_ = lean_box(0);
v_isShared_1496_ = v_isSharedCheck_1503_;
goto v_resetjp_1494_;
}
v_resetjp_1494_:
{
lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1501_; 
v___x_1497_ = lean_box(0);
v___x_1498_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1498_, 0, v_a_1493_);
lean_ctor_set(v___x_1498_, 1, v___x_1497_);
v___x_1499_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1499_, 0, v___x_1498_);
if (v_isShared_1496_ == 0)
{
lean_ctor_set(v___x_1495_, 0, v___x_1499_);
v___x_1501_ = v___x_1495_;
goto v_reusejp_1500_;
}
else
{
lean_object* v_reuseFailAlloc_1502_; 
v_reuseFailAlloc_1502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1502_, 0, v___x_1499_);
v___x_1501_ = v_reuseFailAlloc_1502_;
goto v_reusejp_1500_;
}
v_reusejp_1500_:
{
return v___x_1501_;
}
}
}
else
{
lean_object* v_a_1504_; lean_object* v___x_1506_; uint8_t v_isShared_1507_; uint8_t v_isSharedCheck_1511_; 
v_a_1504_ = lean_ctor_get(v___x_1492_, 0);
v_isSharedCheck_1511_ = !lean_is_exclusive(v___x_1492_);
if (v_isSharedCheck_1511_ == 0)
{
v___x_1506_ = v___x_1492_;
v_isShared_1507_ = v_isSharedCheck_1511_;
goto v_resetjp_1505_;
}
else
{
lean_inc(v_a_1504_);
lean_dec(v___x_1492_);
v___x_1506_ = lean_box(0);
v_isShared_1507_ = v_isSharedCheck_1511_;
goto v_resetjp_1505_;
}
v_resetjp_1505_:
{
lean_object* v___x_1509_; 
if (v_isShared_1507_ == 0)
{
v___x_1509_ = v___x_1506_;
goto v_reusejp_1508_;
}
else
{
lean_object* v_reuseFailAlloc_1510_; 
v_reuseFailAlloc_1510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1510_, 0, v_a_1504_);
v___x_1509_ = v_reuseFailAlloc_1510_;
goto v_reusejp_1508_;
}
v_reusejp_1508_:
{
return v___x_1509_;
}
}
}
}
else
{
uint8_t v___x_1512_; lean_object* v___x_1513_; 
v___x_1512_ = 0;
v___x_1513_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(v_expr_1467_, v___x_1512_, v_entry_1469_, v_bvars_1468_, v___x_1491_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec_ref(v___x_1491_);
if (lean_obj_tag(v___x_1513_) == 0)
{
lean_object* v_a_1514_; lean_object* v___x_1516_; uint8_t v_isShared_1517_; uint8_t v_isSharedCheck_1522_; 
v_a_1514_ = lean_ctor_get(v___x_1513_, 0);
v_isSharedCheck_1522_ = !lean_is_exclusive(v___x_1513_);
if (v_isSharedCheck_1522_ == 0)
{
v___x_1516_ = v___x_1513_;
v_isShared_1517_ = v_isSharedCheck_1522_;
goto v_resetjp_1515_;
}
else
{
lean_inc(v_a_1514_);
lean_dec(v___x_1513_);
v___x_1516_ = lean_box(0);
v_isShared_1517_ = v_isSharedCheck_1522_;
goto v_resetjp_1515_;
}
v_resetjp_1515_:
{
lean_object* v___x_1518_; lean_object* v___x_1520_; 
v___x_1518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1518_, 0, v_a_1514_);
if (v_isShared_1517_ == 0)
{
lean_ctor_set(v___x_1516_, 0, v___x_1518_);
v___x_1520_ = v___x_1516_;
goto v_reusejp_1519_;
}
else
{
lean_object* v_reuseFailAlloc_1521_; 
v_reuseFailAlloc_1521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1521_, 0, v___x_1518_);
v___x_1520_ = v_reuseFailAlloc_1521_;
goto v_reusejp_1519_;
}
v_reusejp_1519_:
{
return v___x_1520_;
}
}
}
else
{
lean_object* v_a_1523_; lean_object* v___x_1525_; uint8_t v_isShared_1526_; uint8_t v_isSharedCheck_1530_; 
v_a_1523_ = lean_ctor_get(v___x_1513_, 0);
v_isSharedCheck_1530_ = !lean_is_exclusive(v___x_1513_);
if (v_isSharedCheck_1530_ == 0)
{
v___x_1525_ = v___x_1513_;
v_isShared_1526_ = v_isSharedCheck_1530_;
goto v_resetjp_1524_;
}
else
{
lean_inc(v_a_1523_);
lean_dec(v___x_1513_);
v___x_1525_ = lean_box(0);
v_isShared_1526_ = v_isSharedCheck_1530_;
goto v_resetjp_1524_;
}
v_resetjp_1524_:
{
lean_object* v___x_1528_; 
if (v_isShared_1526_ == 0)
{
v___x_1528_ = v___x_1525_;
goto v_reusejp_1527_;
}
else
{
lean_object* v_reuseFailAlloc_1529_; 
v_reuseFailAlloc_1529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1529_, 0, v_a_1523_);
v___x_1528_ = v_reuseFailAlloc_1529_;
goto v_reusejp_1527_;
}
v_reusejp_1527_:
{
return v___x_1528_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed(lean_object* v_cfg_1534_, lean_object* v_eta_1535_, lean_object* v_expr_1536_, lean_object* v_bvars_1537_, lean_object* v_entry_1538_, lean_object* v___y_1539_, lean_object* v___y_1540_, lean_object* v___y_1541_, lean_object* v___y_1542_, lean_object* v___y_1543_){
_start:
{
uint8_t v_eta_boxed_1544_; lean_object* v_res_1545_; 
v_eta_boxed_1544_ = lean_unbox(v_eta_1535_);
v_res_1545_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0(v_cfg_1534_, v_eta_boxed_1544_, v_expr_1536_, v_bvars_1537_, v_entry_1538_, v___y_1539_, v___y_1540_, v___y_1541_, v___y_1542_);
lean_dec(v___y_1542_);
lean_dec_ref(v___y_1541_);
lean_dec(v___y_1540_);
lean_dec(v_bvars_1537_);
return v_res_1545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(lean_object* v_entry_1546_, uint8_t v_eta_1547_, lean_object* v_a_1548_, lean_object* v_a_1549_, lean_object* v_a_1550_, lean_object* v_a_1551_){
_start:
{
lean_object* v_stack_1553_; 
v_stack_1553_ = lean_ctor_get(v_entry_1546_, 1);
lean_inc(v_stack_1553_);
if (lean_obj_tag(v_stack_1553_) == 0)
{
lean_object* v___x_1554_; lean_object* v___x_1555_; 
lean_dec_ref(v_entry_1546_);
v___x_1554_ = lean_box(0);
v___x_1555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1555_, 0, v___x_1554_);
return v___x_1555_;
}
else
{
lean_object* v_previous_1556_; lean_object* v_mctx_1557_; lean_object* v_labelledStars_x3f_1558_; lean_object* v_computedKeys_1559_; lean_object* v___x_1561_; uint8_t v_isShared_1562_; uint8_t v_isSharedCheck_1589_; 
v_previous_1556_ = lean_ctor_get(v_entry_1546_, 0);
v_mctx_1557_ = lean_ctor_get(v_entry_1546_, 2);
v_labelledStars_x3f_1558_ = lean_ctor_get(v_entry_1546_, 3);
v_computedKeys_1559_ = lean_ctor_get(v_entry_1546_, 4);
v_isSharedCheck_1589_ = !lean_is_exclusive(v_entry_1546_);
if (v_isSharedCheck_1589_ == 0)
{
lean_object* v_unused_1590_; 
v_unused_1590_ = lean_ctor_get(v_entry_1546_, 1);
lean_dec(v_unused_1590_);
v___x_1561_ = v_entry_1546_;
v_isShared_1562_ = v_isSharedCheck_1589_;
goto v_resetjp_1560_;
}
else
{
lean_inc(v_computedKeys_1559_);
lean_inc(v_labelledStars_x3f_1558_);
lean_inc(v_mctx_1557_);
lean_inc(v_previous_1556_);
lean_dec(v_entry_1546_);
v___x_1561_ = lean_box(0);
v_isShared_1562_ = v_isSharedCheck_1589_;
goto v_resetjp_1560_;
}
v_resetjp_1560_:
{
lean_object* v_head_1563_; lean_object* v_tail_1564_; lean_object* v___x_1566_; uint8_t v_isShared_1567_; uint8_t v_isSharedCheck_1588_; 
v_head_1563_ = lean_ctor_get(v_stack_1553_, 0);
v_tail_1564_ = lean_ctor_get(v_stack_1553_, 1);
v_isSharedCheck_1588_ = !lean_is_exclusive(v_stack_1553_);
if (v_isSharedCheck_1588_ == 0)
{
v___x_1566_ = v_stack_1553_;
v_isShared_1567_ = v_isSharedCheck_1588_;
goto v_resetjp_1565_;
}
else
{
lean_inc(v_tail_1564_);
lean_inc(v_head_1563_);
lean_dec(v_stack_1553_);
v___x_1566_ = lean_box(0);
v_isShared_1567_ = v_isSharedCheck_1588_;
goto v_resetjp_1565_;
}
v_resetjp_1565_:
{
lean_object* v_entry_1569_; 
if (v_isShared_1562_ == 0)
{
lean_ctor_set(v___x_1561_, 1, v_tail_1564_);
v_entry_1569_ = v___x_1561_;
goto v_reusejp_1568_;
}
else
{
lean_object* v_reuseFailAlloc_1587_; 
v_reuseFailAlloc_1587_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1587_, 0, v_previous_1556_);
lean_ctor_set(v_reuseFailAlloc_1587_, 1, v_tail_1564_);
lean_ctor_set(v_reuseFailAlloc_1587_, 2, v_mctx_1557_);
lean_ctor_set(v_reuseFailAlloc_1587_, 3, v_labelledStars_x3f_1558_);
lean_ctor_set(v_reuseFailAlloc_1587_, 4, v_computedKeys_1559_);
v_entry_1569_ = v_reuseFailAlloc_1587_;
goto v_reusejp_1568_;
}
v_reusejp_1568_:
{
if (lean_obj_tag(v_head_1563_) == 0)
{
lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1574_; 
v___x_1570_ = lean_box(0);
v___x_1571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1571_, 0, v___x_1570_);
lean_ctor_set(v___x_1571_, 1, v_entry_1569_);
v___x_1572_ = lean_box(0);
if (v_isShared_1567_ == 0)
{
lean_ctor_set(v___x_1566_, 1, v___x_1572_);
lean_ctor_set(v___x_1566_, 0, v___x_1571_);
v___x_1574_ = v___x_1566_;
goto v_reusejp_1573_;
}
else
{
lean_object* v_reuseFailAlloc_1577_; 
v_reuseFailAlloc_1577_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1577_, 0, v___x_1571_);
lean_ctor_set(v_reuseFailAlloc_1577_, 1, v___x_1572_);
v___x_1574_ = v_reuseFailAlloc_1577_;
goto v_reusejp_1573_;
}
v_reusejp_1573_:
{
lean_object* v___x_1575_; lean_object* v___x_1576_; 
v___x_1575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1575_, 0, v___x_1574_);
v___x_1576_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1576_, 0, v___x_1575_);
return v___x_1576_;
}
}
else
{
lean_object* v_info_1578_; lean_object* v_expr_1579_; lean_object* v_bvars_1580_; lean_object* v_lctx_1581_; lean_object* v_localInsts_1582_; lean_object* v_cfg_1583_; lean_object* v___x_1584_; lean_object* v___f_1585_; lean_object* v___x_1586_; 
lean_del_object(v___x_1566_);
v_info_1578_ = lean_ctor_get(v_head_1563_, 0);
lean_inc_ref(v_info_1578_);
lean_dec_ref_known(v_head_1563_, 1);
v_expr_1579_ = lean_ctor_get(v_info_1578_, 0);
lean_inc_ref(v_expr_1579_);
v_bvars_1580_ = lean_ctor_get(v_info_1578_, 1);
lean_inc(v_bvars_1580_);
v_lctx_1581_ = lean_ctor_get(v_info_1578_, 2);
lean_inc_ref(v_lctx_1581_);
v_localInsts_1582_ = lean_ctor_get(v_info_1578_, 3);
lean_inc_ref(v_localInsts_1582_);
v_cfg_1583_ = lean_ctor_get(v_info_1578_, 4);
lean_inc_ref(v_cfg_1583_);
lean_dec_ref(v_info_1578_);
v___x_1584_ = lean_box(v_eta_1547_);
v___f_1585_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1585_, 0, v_cfg_1583_);
lean_closure_set(v___f_1585_, 1, v___x_1584_);
lean_closure_set(v___f_1585_, 2, v_expr_1579_);
lean_closure_set(v___f_1585_, 3, v_bvars_1580_);
lean_closure_set(v___f_1585_, 4, v_entry_1569_);
v___x_1586_ = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(v_lctx_1581_, v_localInsts_1582_, v___f_1585_, v_a_1548_, v_a_1549_, v_a_1550_, v_a_1551_);
return v___x_1586_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux___boxed(lean_object* v_entry_1591_, lean_object* v_eta_1592_, lean_object* v_a_1593_, lean_object* v_a_1594_, lean_object* v_a_1595_, lean_object* v_a_1596_, lean_object* v_a_1597_){
_start:
{
uint8_t v_eta_boxed_1598_; lean_object* v_res_1599_; 
v_eta_boxed_1598_ = lean_unbox(v_eta_1592_);
v_res_1599_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(v_entry_1591_, v_eta_boxed_1598_, v_a_1593_, v_a_1594_, v_a_1595_, v_a_1596_);
lean_dec(v_a_1596_);
lean_dec_ref(v_a_1595_);
lean_dec(v_a_1594_);
lean_dec_ref(v_a_1593_);
return v_res_1599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(lean_object* v_args_1600_, lean_object* v_fnType_1601_, lean_object* v_i_1602_, lean_object* v_j_1603_, lean_object* v_cont_1604_, lean_object* v_a_1605_, lean_object* v_a_1606_, lean_object* v_a_1607_, lean_object* v_a_1608_){
_start:
{
lean_object* v___x_1610_; lean_object* v___x_1611_; 
v___x_1610_ = lean_expr_instantiate_rev_range(v_fnType_1601_, v_j_1603_, v_i_1602_, v_args_1600_);
v___x_1611_ = l_Lean_Meta_whnfD(v___x_1610_, v_a_1605_, v_a_1606_, v_a_1607_, v_a_1608_);
if (lean_obj_tag(v___x_1611_) == 0)
{
lean_object* v_a_1612_; 
v_a_1612_ = lean_ctor_get(v___x_1611_, 0);
lean_inc(v_a_1612_);
lean_dec_ref_known(v___x_1611_, 1);
if (lean_obj_tag(v_a_1612_) == 7)
{
lean_object* v_binderType_1613_; lean_object* v_body_1614_; uint8_t v_binderInfo_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; 
v_binderType_1613_ = lean_ctor_get(v_a_1612_, 1);
lean_inc_ref(v_binderType_1613_);
v_body_1614_ = lean_ctor_get(v_a_1612_, 2);
lean_inc_ref(v_body_1614_);
v_binderInfo_1615_ = lean_ctor_get_uint8(v_a_1612_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_a_1612_, 3);
v___x_1616_ = lean_box(v_binderInfo_1615_);
lean_inc(v_a_1608_);
lean_inc_ref(v_a_1607_);
lean_inc(v_a_1606_);
lean_inc_ref(v_a_1605_);
v___x_1617_ = lean_apply_9(v_cont_1604_, v_i_1602_, v_binderType_1613_, v_body_1614_, v___x_1616_, v_a_1605_, v_a_1606_, v_a_1607_, v_a_1608_, lean_box(0));
return v___x_1617_;
}
else
{
lean_object* v___x_1618_; 
lean_dec_ref(v_cont_1604_);
lean_dec(v_i_1602_);
v___x_1618_ = l_Lean_Meta_throwFunctionExpected___redArg(v_a_1612_, v_a_1605_, v_a_1606_, v_a_1607_, v_a_1608_);
return v___x_1618_;
}
}
else
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1626_; 
lean_dec_ref(v_cont_1604_);
lean_dec(v_i_1602_);
v_a_1619_ = lean_ctor_get(v___x_1611_, 0);
v_isSharedCheck_1626_ = !lean_is_exclusive(v___x_1611_);
if (v_isSharedCheck_1626_ == 0)
{
v___x_1621_ = v___x_1611_;
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1611_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1624_; 
if (v_isShared_1622_ == 0)
{
v___x_1624_ = v___x_1621_;
goto v_reusejp_1623_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_a_1619_);
v___x_1624_ = v_reuseFailAlloc_1625_;
goto v_reusejp_1623_;
}
v_reusejp_1623_:
{
return v___x_1624_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce___boxed(lean_object* v_args_1627_, lean_object* v_fnType_1628_, lean_object* v_i_1629_, lean_object* v_j_1630_, lean_object* v_cont_1631_, lean_object* v_a_1632_, lean_object* v_a_1633_, lean_object* v_a_1634_, lean_object* v_a_1635_, lean_object* v_a_1636_){
_start:
{
lean_object* v_res_1637_; 
v_res_1637_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(v_args_1627_, v_fnType_1628_, v_i_1629_, v_j_1630_, v_cont_1631_, v_a_1632_, v_a_1633_, v_a_1634_, v_a_1635_);
lean_dec(v_a_1635_);
lean_dec_ref(v_a_1634_);
lean_dec(v_a_1633_);
lean_dec_ref(v_a_1632_);
lean_dec(v_j_1630_);
lean_dec_ref(v_fnType_1628_);
lean_dec_ref(v_args_1627_);
return v_res_1637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(lean_object* v_arg_1638_, lean_object* v_domain_1639_, uint8_t v_binderInfo_1640_, lean_object* v_a_1641_, lean_object* v_a_1642_, lean_object* v_a_1643_, lean_object* v_a_1644_){
_start:
{
uint8_t v___x_1646_; uint8_t v___x_1647_; 
v___x_1646_ = l_Lean_Expr_isOutParam(v_domain_1639_);
v___x_1647_ = 1;
if (v___x_1646_ == 0)
{
switch(v_binderInfo_1640_)
{
case 0:
{
lean_object* v___x_1648_; 
v___x_1648_ = l_Lean_Meta_isProof(v_arg_1638_, v_a_1641_, v_a_1642_, v_a_1643_, v_a_1644_);
return v___x_1648_;
}
case 3:
{
lean_object* v___x_1649_; lean_object* v___x_1650_; 
lean_dec_ref(v_arg_1638_);
v___x_1649_ = lean_box(v___x_1647_);
v___x_1650_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1650_, 0, v___x_1649_);
return v___x_1650_;
}
default: 
{
lean_object* v___x_1651_; 
v___x_1651_ = l_Lean_Meta_isType(v_arg_1638_, v_a_1641_, v_a_1642_, v_a_1643_, v_a_1644_);
if (lean_obj_tag(v___x_1651_) == 0)
{
lean_object* v_a_1652_; lean_object* v___x_1654_; uint8_t v_isShared_1655_; uint8_t v_isSharedCheck_1665_; 
v_a_1652_ = lean_ctor_get(v___x_1651_, 0);
v_isSharedCheck_1665_ = !lean_is_exclusive(v___x_1651_);
if (v_isSharedCheck_1665_ == 0)
{
v___x_1654_ = v___x_1651_;
v_isShared_1655_ = v_isSharedCheck_1665_;
goto v_resetjp_1653_;
}
else
{
lean_inc(v_a_1652_);
lean_dec(v___x_1651_);
v___x_1654_ = lean_box(0);
v_isShared_1655_ = v_isSharedCheck_1665_;
goto v_resetjp_1653_;
}
v_resetjp_1653_:
{
uint8_t v___x_1656_; 
v___x_1656_ = lean_unbox(v_a_1652_);
lean_dec(v_a_1652_);
if (v___x_1656_ == 0)
{
lean_object* v___x_1657_; lean_object* v___x_1659_; 
v___x_1657_ = lean_box(v___x_1647_);
if (v_isShared_1655_ == 0)
{
lean_ctor_set(v___x_1654_, 0, v___x_1657_);
v___x_1659_ = v___x_1654_;
goto v_reusejp_1658_;
}
else
{
lean_object* v_reuseFailAlloc_1660_; 
v_reuseFailAlloc_1660_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1660_, 0, v___x_1657_);
v___x_1659_ = v_reuseFailAlloc_1660_;
goto v_reusejp_1658_;
}
v_reusejp_1658_:
{
return v___x_1659_;
}
}
else
{
lean_object* v___x_1661_; lean_object* v___x_1663_; 
v___x_1661_ = lean_box(v___x_1646_);
if (v_isShared_1655_ == 0)
{
lean_ctor_set(v___x_1654_, 0, v___x_1661_);
v___x_1663_ = v___x_1654_;
goto v_reusejp_1662_;
}
else
{
lean_object* v_reuseFailAlloc_1664_; 
v_reuseFailAlloc_1664_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1664_, 0, v___x_1661_);
v___x_1663_ = v_reuseFailAlloc_1664_;
goto v_reusejp_1662_;
}
v_reusejp_1662_:
{
return v___x_1663_;
}
}
}
}
else
{
return v___x_1651_;
}
}
}
}
else
{
lean_object* v___x_1666_; lean_object* v___x_1667_; 
lean_dec_ref(v_arg_1638_);
v___x_1666_ = lean_box(v___x_1647_);
v___x_1667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1667_, 0, v___x_1666_);
return v___x_1667_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg___boxed(lean_object* v_arg_1668_, lean_object* v_domain_1669_, lean_object* v_binderInfo_1670_, lean_object* v_a_1671_, lean_object* v_a_1672_, lean_object* v_a_1673_, lean_object* v_a_1674_, lean_object* v_a_1675_){
_start:
{
uint8_t v_binderInfo_boxed_1676_; lean_object* v_res_1677_; 
v_binderInfo_boxed_1676_ = lean_unbox(v_binderInfo_1670_);
v_res_1677_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(v_arg_1668_, v_domain_1669_, v_binderInfo_boxed_1676_, v_a_1671_, v_a_1672_, v_a_1673_, v_a_1674_);
lean_dec(v_a_1674_);
lean_dec_ref(v_a_1673_);
lean_dec(v_a_1672_);
lean_dec_ref(v_a_1671_);
lean_dec_ref(v_domain_1669_);
return v_res_1677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(lean_object* v_arg_1678_, lean_object* v_bvars_1679_, lean_object* v_i_1680_, lean_object* v_entries_1681_, lean_object* v_args_1682_, lean_object* v_j_1683_, lean_object* v_d_1684_, lean_object* v_b_1685_, uint8_t v_bi_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_){
_start:
{
lean_object* v___x_1692_; 
lean_inc_ref(v_arg_1678_);
v___x_1692_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_isIgnoredArg(v_arg_1678_, v_d_1684_, v_bi_1686_, v___y_1687_, v___y_1688_, v___y_1689_, v___y_1690_);
if (lean_obj_tag(v___x_1692_) == 0)
{
lean_object* v_a_1693_; uint8_t v___x_1694_; 
v_a_1693_ = lean_ctor_get(v___x_1692_, 0);
lean_inc(v_a_1693_);
lean_dec_ref_known(v___x_1692_, 1);
v___x_1694_ = lean_unbox(v_a_1693_);
lean_dec(v_a_1693_);
if (v___x_1694_ == 0)
{
lean_object* v___x_1695_; 
lean_inc(v_bvars_1679_);
v___x_1695_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v_arg_1678_, v_bvars_1679_, v___y_1687_);
if (lean_obj_tag(v___x_1695_) == 0)
{
lean_object* v_a_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; 
v_a_1696_ = lean_ctor_get(v___x_1695_, 0);
lean_inc(v_a_1696_);
lean_dec_ref_known(v___x_1695_, 1);
v___x_1697_ = lean_unsigned_to_nat(1u);
v___x_1698_ = lean_nat_add(v_i_1680_, v___x_1697_);
v___x_1699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1699_, 0, v_a_1696_);
v___x_1700_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1700_, 0, v___x_1699_);
lean_ctor_set(v___x_1700_, 1, v_entries_1681_);
v___x_1701_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(v_args_1682_, v_bvars_1679_, v_b_1685_, v___x_1698_, v_j_1683_, v___x_1700_, v___y_1687_, v___y_1688_, v___y_1689_, v___y_1690_);
return v___x_1701_;
}
else
{
lean_object* v_a_1702_; lean_object* v___x_1704_; uint8_t v_isShared_1705_; uint8_t v_isSharedCheck_1709_; 
lean_dec_ref(v_args_1682_);
lean_dec(v_entries_1681_);
lean_dec(v_bvars_1679_);
v_a_1702_ = lean_ctor_get(v___x_1695_, 0);
v_isSharedCheck_1709_ = !lean_is_exclusive(v___x_1695_);
if (v_isSharedCheck_1709_ == 0)
{
v___x_1704_ = v___x_1695_;
v_isShared_1705_ = v_isSharedCheck_1709_;
goto v_resetjp_1703_;
}
else
{
lean_inc(v_a_1702_);
lean_dec(v___x_1695_);
v___x_1704_ = lean_box(0);
v_isShared_1705_ = v_isSharedCheck_1709_;
goto v_resetjp_1703_;
}
v_resetjp_1703_:
{
lean_object* v___x_1707_; 
if (v_isShared_1705_ == 0)
{
v___x_1707_ = v___x_1704_;
goto v_reusejp_1706_;
}
else
{
lean_object* v_reuseFailAlloc_1708_; 
v_reuseFailAlloc_1708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1708_, 0, v_a_1702_);
v___x_1707_ = v_reuseFailAlloc_1708_;
goto v_reusejp_1706_;
}
v_reusejp_1706_:
{
return v___x_1707_;
}
}
}
}
else
{
lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; 
lean_dec_ref(v_arg_1678_);
v___x_1710_ = lean_unsigned_to_nat(1u);
v___x_1711_ = lean_nat_add(v_i_1680_, v___x_1710_);
v___x_1712_ = lean_box(0);
v___x_1713_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1713_, 0, v___x_1712_);
lean_ctor_set(v___x_1713_, 1, v_entries_1681_);
v___x_1714_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(v_args_1682_, v_bvars_1679_, v_b_1685_, v___x_1711_, v_j_1683_, v___x_1713_, v___y_1687_, v___y_1688_, v___y_1689_, v___y_1690_);
return v___x_1714_;
}
}
else
{
lean_object* v_a_1715_; lean_object* v___x_1717_; uint8_t v_isShared_1718_; uint8_t v_isSharedCheck_1722_; 
lean_dec_ref(v_args_1682_);
lean_dec(v_entries_1681_);
lean_dec(v_bvars_1679_);
lean_dec_ref(v_arg_1678_);
v_a_1715_ = lean_ctor_get(v___x_1692_, 0);
v_isSharedCheck_1722_ = !lean_is_exclusive(v___x_1692_);
if (v_isSharedCheck_1722_ == 0)
{
v___x_1717_ = v___x_1692_;
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
else
{
lean_inc(v_a_1715_);
lean_dec(v___x_1692_);
v___x_1717_ = lean_box(0);
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
v_resetjp_1716_:
{
lean_object* v___x_1720_; 
if (v_isShared_1718_ == 0)
{
v___x_1720_ = v___x_1717_;
goto v_reusejp_1719_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v_a_1715_);
v___x_1720_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1719_;
}
v_reusejp_1719_:
{
return v___x_1720_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0___boxed(lean_object* v_arg_1723_, lean_object* v_bvars_1724_, lean_object* v_i_1725_, lean_object* v_entries_1726_, lean_object* v_args_1727_, lean_object* v_j_1728_, lean_object* v_d_1729_, lean_object* v_b_1730_, lean_object* v_bi_1731_, lean_object* v___y_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_){
_start:
{
uint8_t v_bi_boxed_1737_; lean_object* v_res_1738_; 
v_bi_boxed_1737_ = lean_unbox(v_bi_1731_);
v_res_1738_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(v_arg_1723_, v_bvars_1724_, v_i_1725_, v_entries_1726_, v_args_1727_, v_j_1728_, v_d_1729_, v_b_1730_, v_bi_boxed_1737_, v___y_1732_, v___y_1733_, v___y_1734_, v___y_1735_);
lean_dec(v___y_1735_);
lean_dec_ref(v___y_1734_);
lean_dec(v___y_1733_);
lean_dec_ref(v___y_1732_);
lean_dec_ref(v_b_1730_);
lean_dec_ref(v_d_1729_);
lean_dec(v_j_1728_);
lean_dec(v_i_1725_);
return v_res_1738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(lean_object* v_args_1739_, lean_object* v_bvars_1740_, lean_object* v_fnType_1741_, lean_object* v_i_1742_, lean_object* v_j_1743_, lean_object* v_entries_1744_, lean_object* v_a_1745_, lean_object* v_a_1746_, lean_object* v_a_1747_, lean_object* v_a_1748_){
_start:
{
lean_object* v___x_1750_; uint8_t v___x_1751_; 
v___x_1750_ = lean_array_get_size(v_args_1739_);
v___x_1751_ = lean_nat_dec_lt(v_i_1742_, v___x_1750_);
if (v___x_1751_ == 0)
{
lean_object* v___x_1752_; 
lean_dec(v_i_1742_);
lean_dec(v_bvars_1740_);
lean_dec_ref(v_args_1739_);
v___x_1752_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1752_, 0, v_entries_1744_);
return v___x_1752_;
}
else
{
lean_object* v_arg_1753_; lean_object* v_cont_1754_; 
v_arg_1753_ = lean_array_fget_borrowed(v_args_1739_, v_i_1742_);
lean_inc_ref(v_args_1739_);
lean_inc(v_entries_1744_);
lean_inc(v_i_1742_);
lean_inc(v_bvars_1740_);
lean_inc(v_arg_1753_);
v_cont_1754_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0___boxed), 14, 5);
lean_closure_set(v_cont_1754_, 0, v_arg_1753_);
lean_closure_set(v_cont_1754_, 1, v_bvars_1740_);
lean_closure_set(v_cont_1754_, 2, v_i_1742_);
lean_closure_set(v_cont_1754_, 3, v_entries_1744_);
lean_closure_set(v_cont_1754_, 4, v_args_1739_);
if (lean_obj_tag(v_fnType_1741_) == 7)
{
lean_object* v_binderType_1755_; lean_object* v_body_1756_; uint8_t v_binderInfo_1757_; lean_object* v___x_1758_; 
lean_inc(v_arg_1753_);
lean_dec_ref(v_cont_1754_);
v_binderType_1755_ = lean_ctor_get(v_fnType_1741_, 1);
v_body_1756_ = lean_ctor_get(v_fnType_1741_, 2);
v_binderInfo_1757_ = lean_ctor_get_uint8(v_fnType_1741_, sizeof(void*)*3 + 8);
v___x_1758_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___lam__0(v_arg_1753_, v_bvars_1740_, v_i_1742_, v_entries_1744_, v_args_1739_, v_j_1743_, v_binderType_1755_, v_body_1756_, v_binderInfo_1757_, v_a_1745_, v_a_1746_, v_a_1747_, v_a_1748_);
lean_dec(v_i_1742_);
return v___x_1758_;
}
else
{
lean_object* v___x_1759_; 
lean_dec(v_entries_1744_);
lean_dec(v_bvars_1740_);
v___x_1759_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop_reduce(v_args_1739_, v_fnType_1741_, v_i_1742_, v_j_1743_, v_cont_1754_, v_a_1745_, v_a_1746_, v_a_1747_, v_a_1748_);
lean_dec_ref(v_args_1739_);
return v___x_1759_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop___boxed(lean_object* v_args_1760_, lean_object* v_bvars_1761_, lean_object* v_fnType_1762_, lean_object* v_i_1763_, lean_object* v_j_1764_, lean_object* v_entries_1765_, lean_object* v_a_1766_, lean_object* v_a_1767_, lean_object* v_a_1768_, lean_object* v_a_1769_, lean_object* v_a_1770_){
_start:
{
lean_object* v_res_1771_; 
v_res_1771_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(v_args_1760_, v_bvars_1761_, v_fnType_1762_, v_i_1763_, v_j_1764_, v_entries_1765_, v_a_1766_, v_a_1767_, v_a_1768_, v_a_1769_);
lean_dec(v_a_1769_);
lean_dec_ref(v_a_1768_);
lean_dec(v_a_1767_);
lean_dec_ref(v_a_1766_);
lean_dec(v_j_1764_);
lean_dec_ref(v_fnType_1762_);
return v_res_1771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(lean_object* v_fn_1772_, lean_object* v_args_1773_, lean_object* v_bvars_1774_, lean_object* v_a_1775_, lean_object* v_a_1776_, lean_object* v_a_1777_, lean_object* v_a_1778_){
_start:
{
lean_object* v___x_1780_; 
lean_inc(v_a_1778_);
lean_inc_ref(v_a_1777_);
lean_inc(v_a_1776_);
lean_inc_ref(v_a_1775_);
v___x_1780_ = lean_infer_type(v_fn_1772_, v_a_1775_, v_a_1776_, v_a_1777_, v_a_1778_);
if (lean_obj_tag(v___x_1780_) == 0)
{
lean_object* v_a_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; 
v_a_1781_ = lean_ctor_get(v___x_1780_, 0);
lean_inc(v_a_1781_);
lean_dec_ref_known(v___x_1780_, 1);
v___x_1782_ = lean_unsigned_to_nat(0u);
v___x_1783_ = lean_box(0);
v___x_1784_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries_loop(v_args_1773_, v_bvars_1774_, v_a_1781_, v___x_1782_, v___x_1782_, v___x_1783_, v_a_1775_, v_a_1776_, v_a_1777_, v_a_1778_);
lean_dec(v_a_1781_);
return v___x_1784_;
}
else
{
lean_object* v_a_1785_; lean_object* v___x_1787_; uint8_t v_isShared_1788_; uint8_t v_isSharedCheck_1792_; 
lean_dec(v_bvars_1774_);
lean_dec_ref(v_args_1773_);
v_a_1785_ = lean_ctor_get(v___x_1780_, 0);
v_isSharedCheck_1792_ = !lean_is_exclusive(v___x_1780_);
if (v_isSharedCheck_1792_ == 0)
{
v___x_1787_ = v___x_1780_;
v_isShared_1788_ = v_isSharedCheck_1792_;
goto v_resetjp_1786_;
}
else
{
lean_inc(v_a_1785_);
lean_dec(v___x_1780_);
v___x_1787_ = lean_box(0);
v_isShared_1788_ = v_isSharedCheck_1792_;
goto v_resetjp_1786_;
}
v_resetjp_1786_:
{
lean_object* v___x_1790_; 
if (v_isShared_1788_ == 0)
{
v___x_1790_ = v___x_1787_;
goto v_reusejp_1789_;
}
else
{
lean_object* v_reuseFailAlloc_1791_; 
v_reuseFailAlloc_1791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1791_, 0, v_a_1785_);
v___x_1790_ = v_reuseFailAlloc_1791_;
goto v_reusejp_1789_;
}
v_reusejp_1789_:
{
return v___x_1790_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries___boxed(lean_object* v_fn_1793_, lean_object* v_args_1794_, lean_object* v_bvars_1795_, lean_object* v_a_1796_, lean_object* v_a_1797_, lean_object* v_a_1798_, lean_object* v_a_1799_, lean_object* v_a_1800_){
_start:
{
lean_object* v_res_1801_; 
v_res_1801_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(v_fn_1793_, v_args_1794_, v_bvars_1795_, v_a_1796_, v_a_1797_, v_a_1798_, v_a_1799_);
lean_dec(v_a_1799_);
lean_dec_ref(v_a_1798_);
lean_dec(v_a_1797_);
lean_dec_ref(v_a_1796_);
return v_res_1801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0(lean_object* v_k_1802_, lean_object* v_b_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_){
_start:
{
lean_object* v___x_1809_; 
lean_inc(v___y_1807_);
lean_inc_ref(v___y_1806_);
lean_inc(v___y_1805_);
lean_inc_ref(v___y_1804_);
v___x_1809_ = lean_apply_6(v_k_1802_, v_b_1803_, v___y_1804_, v___y_1805_, v___y_1806_, v___y_1807_, lean_box(0));
return v___x_1809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0___boxed(lean_object* v_k_1810_, lean_object* v_b_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_){
_start:
{
lean_object* v_res_1817_; 
v_res_1817_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0(v_k_1810_, v_b_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
lean_dec(v___y_1815_);
lean_dec_ref(v___y_1814_);
lean_dec(v___y_1813_);
lean_dec_ref(v___y_1812_);
return v_res_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(lean_object* v_name_1818_, uint8_t v_bi_1819_, lean_object* v_type_1820_, lean_object* v_k_1821_, uint8_t v_kind_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_){
_start:
{
lean_object* v___f_1828_; lean_object* v___x_1829_; 
v___f_1828_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1828_, 0, v_k_1821_);
v___x_1829_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_1818_, v_bi_1819_, v_type_1820_, v___f_1828_, v_kind_1822_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
if (lean_obj_tag(v___x_1829_) == 0)
{
lean_object* v_a_1830_; lean_object* v___x_1832_; uint8_t v_isShared_1833_; uint8_t v_isSharedCheck_1837_; 
v_a_1830_ = lean_ctor_get(v___x_1829_, 0);
v_isSharedCheck_1837_ = !lean_is_exclusive(v___x_1829_);
if (v_isSharedCheck_1837_ == 0)
{
v___x_1832_ = v___x_1829_;
v_isShared_1833_ = v_isSharedCheck_1837_;
goto v_resetjp_1831_;
}
else
{
lean_inc(v_a_1830_);
lean_dec(v___x_1829_);
v___x_1832_ = lean_box(0);
v_isShared_1833_ = v_isSharedCheck_1837_;
goto v_resetjp_1831_;
}
v_resetjp_1831_:
{
lean_object* v___x_1835_; 
if (v_isShared_1833_ == 0)
{
v___x_1835_ = v___x_1832_;
goto v_reusejp_1834_;
}
else
{
lean_object* v_reuseFailAlloc_1836_; 
v_reuseFailAlloc_1836_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1836_, 0, v_a_1830_);
v___x_1835_ = v_reuseFailAlloc_1836_;
goto v_reusejp_1834_;
}
v_reusejp_1834_:
{
return v___x_1835_;
}
}
}
else
{
lean_object* v_a_1838_; lean_object* v___x_1840_; uint8_t v_isShared_1841_; uint8_t v_isSharedCheck_1845_; 
v_a_1838_ = lean_ctor_get(v___x_1829_, 0);
v_isSharedCheck_1845_ = !lean_is_exclusive(v___x_1829_);
if (v_isSharedCheck_1845_ == 0)
{
v___x_1840_ = v___x_1829_;
v_isShared_1841_ = v_isSharedCheck_1845_;
goto v_resetjp_1839_;
}
else
{
lean_inc(v_a_1838_);
lean_dec(v___x_1829_);
v___x_1840_ = lean_box(0);
v_isShared_1841_ = v_isSharedCheck_1845_;
goto v_resetjp_1839_;
}
v_resetjp_1839_:
{
lean_object* v___x_1843_; 
if (v_isShared_1841_ == 0)
{
v___x_1843_ = v___x_1840_;
goto v_reusejp_1842_;
}
else
{
lean_object* v_reuseFailAlloc_1844_; 
v_reuseFailAlloc_1844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1844_, 0, v_a_1838_);
v___x_1843_ = v_reuseFailAlloc_1844_;
goto v_reusejp_1842_;
}
v_reusejp_1842_:
{
return v___x_1843_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg___boxed(lean_object* v_name_1846_, lean_object* v_bi_1847_, lean_object* v_type_1848_, lean_object* v_k_1849_, lean_object* v_kind_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_){
_start:
{
uint8_t v_bi_boxed_1856_; uint8_t v_kind_boxed_1857_; lean_object* v_res_1858_; 
v_bi_boxed_1856_ = lean_unbox(v_bi_1847_);
v_kind_boxed_1857_ = lean_unbox(v_kind_1850_);
v_res_1858_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(v_name_1846_, v_bi_boxed_1856_, v_type_1848_, v_k_1849_, v_kind_boxed_1857_, v___y_1851_, v___y_1852_, v___y_1853_, v___y_1854_);
lean_dec(v___y_1854_);
lean_dec_ref(v___y_1853_);
lean_dec(v___y_1852_);
lean_dec_ref(v___y_1851_);
return v_res_1858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0(lean_object* v_00_u03b1_1859_, lean_object* v_name_1860_, uint8_t v_bi_1861_, lean_object* v_type_1862_, lean_object* v_k_1863_, uint8_t v_kind_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_){
_start:
{
lean_object* v___x_1870_; 
v___x_1870_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(v_name_1860_, v_bi_1861_, v_type_1862_, v_k_1863_, v_kind_1864_, v___y_1865_, v___y_1866_, v___y_1867_, v___y_1868_);
return v___x_1870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___boxed(lean_object* v_00_u03b1_1871_, lean_object* v_name_1872_, lean_object* v_bi_1873_, lean_object* v_type_1874_, lean_object* v_k_1875_, lean_object* v_kind_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_){
_start:
{
uint8_t v_bi_boxed_1882_; uint8_t v_kind_boxed_1883_; lean_object* v_res_1884_; 
v_bi_boxed_1882_ = lean_unbox(v_bi_1873_);
v_kind_boxed_1883_ = lean_unbox(v_kind_1876_);
v_res_1884_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0(v_00_u03b1_1871_, v_name_1872_, v_bi_boxed_1882_, v_type_1874_, v_k_1875_, v_kind_boxed_1883_, v___y_1877_, v___y_1878_, v___y_1879_, v___y_1880_);
lean_dec(v___y_1880_);
lean_dec_ref(v___y_1879_);
lean_dec(v___y_1878_);
lean_dec_ref(v___y_1877_);
return v_res_1884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1(lean_object* v_body_1885_, lean_object* v_bvars_1886_, lean_object* v_fvar_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_){
_start:
{
lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; 
v___x_1893_ = lean_expr_instantiate1(v_body_1885_, v_fvar_1887_);
v___x_1894_ = l_Lean_Expr_fvarId_x21(v_fvar_1887_);
v___x_1895_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1895_, 0, v___x_1894_);
lean_ctor_set(v___x_1895_, 1, v_bvars_1886_);
v___x_1896_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v___x_1893_, v___x_1895_, v___y_1888_);
if (lean_obj_tag(v___x_1896_) == 0)
{
lean_object* v_a_1897_; lean_object* v___x_1899_; uint8_t v_isShared_1900_; uint8_t v_isSharedCheck_1905_; 
v_a_1897_ = lean_ctor_get(v___x_1896_, 0);
v_isSharedCheck_1905_ = !lean_is_exclusive(v___x_1896_);
if (v_isSharedCheck_1905_ == 0)
{
v___x_1899_ = v___x_1896_;
v_isShared_1900_ = v_isSharedCheck_1905_;
goto v_resetjp_1898_;
}
else
{
lean_inc(v_a_1897_);
lean_dec(v___x_1896_);
v___x_1899_ = lean_box(0);
v_isShared_1900_ = v_isSharedCheck_1905_;
goto v_resetjp_1898_;
}
v_resetjp_1898_:
{
lean_object* v___x_1901_; lean_object* v___x_1903_; 
v___x_1901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1901_, 0, v_a_1897_);
if (v_isShared_1900_ == 0)
{
lean_ctor_set(v___x_1899_, 0, v___x_1901_);
v___x_1903_ = v___x_1899_;
goto v_reusejp_1902_;
}
else
{
lean_object* v_reuseFailAlloc_1904_; 
v_reuseFailAlloc_1904_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1904_, 0, v___x_1901_);
v___x_1903_ = v_reuseFailAlloc_1904_;
goto v_reusejp_1902_;
}
v_reusejp_1902_:
{
return v___x_1903_;
}
}
}
else
{
lean_object* v_a_1906_; lean_object* v___x_1908_; uint8_t v_isShared_1909_; uint8_t v_isSharedCheck_1913_; 
v_a_1906_ = lean_ctor_get(v___x_1896_, 0);
v_isSharedCheck_1913_ = !lean_is_exclusive(v___x_1896_);
if (v_isSharedCheck_1913_ == 0)
{
v___x_1908_ = v___x_1896_;
v_isShared_1909_ = v_isSharedCheck_1913_;
goto v_resetjp_1907_;
}
else
{
lean_inc(v_a_1906_);
lean_dec(v___x_1896_);
v___x_1908_ = lean_box(0);
v_isShared_1909_ = v_isSharedCheck_1913_;
goto v_resetjp_1907_;
}
v_resetjp_1907_:
{
lean_object* v___x_1911_; 
if (v_isShared_1909_ == 0)
{
v___x_1911_ = v___x_1908_;
goto v_reusejp_1910_;
}
else
{
lean_object* v_reuseFailAlloc_1912_; 
v_reuseFailAlloc_1912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1912_, 0, v_a_1906_);
v___x_1911_ = v_reuseFailAlloc_1912_;
goto v_reusejp_1910_;
}
v_reusejp_1910_:
{
return v___x_1911_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1___boxed(lean_object* v_body_1914_, lean_object* v_bvars_1915_, lean_object* v_fvar_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_){
_start:
{
lean_object* v_res_1922_; 
v_res_1922_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1(v_body_1914_, v_bvars_1915_, v_fvar_1916_, v___y_1917_, v___y_1918_, v___y_1919_, v___y_1920_);
lean_dec(v___y_1920_);
lean_dec_ref(v___y_1919_);
lean_dec(v___y_1918_);
lean_dec_ref(v___y_1917_);
lean_dec_ref(v_fvar_1916_);
lean_dec_ref(v_body_1914_);
return v_res_1922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(lean_object* v_x_1923_, lean_object* v_x_1924_, lean_object* v_bvars_1925_, lean_object* v_entry_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_){
_start:
{
lean_object* v___x_1932_; 
v___x_1932_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_getStackEntries(v_x_1923_, v_x_1924_, v_bvars_1925_, v___y_1927_, v___y_1928_, v___y_1929_, v___y_1930_);
if (lean_obj_tag(v___x_1932_) == 0)
{
lean_object* v_a_1933_; lean_object* v___x_1935_; uint8_t v_isShared_1936_; uint8_t v_isSharedCheck_1953_; 
v_a_1933_ = lean_ctor_get(v___x_1932_, 0);
v_isSharedCheck_1953_ = !lean_is_exclusive(v___x_1932_);
if (v_isSharedCheck_1953_ == 0)
{
v___x_1935_ = v___x_1932_;
v_isShared_1936_ = v_isSharedCheck_1953_;
goto v_resetjp_1934_;
}
else
{
lean_inc(v_a_1933_);
lean_dec(v___x_1932_);
v___x_1935_ = lean_box(0);
v_isShared_1936_ = v_isSharedCheck_1953_;
goto v_resetjp_1934_;
}
v_resetjp_1934_:
{
lean_object* v_previous_1937_; lean_object* v_stack_1938_; lean_object* v_mctx_1939_; lean_object* v_labelledStars_x3f_1940_; lean_object* v_computedKeys_1941_; lean_object* v___x_1943_; uint8_t v_isShared_1944_; uint8_t v_isSharedCheck_1952_; 
v_previous_1937_ = lean_ctor_get(v_entry_1926_, 0);
v_stack_1938_ = lean_ctor_get(v_entry_1926_, 1);
v_mctx_1939_ = lean_ctor_get(v_entry_1926_, 2);
v_labelledStars_x3f_1940_ = lean_ctor_get(v_entry_1926_, 3);
v_computedKeys_1941_ = lean_ctor_get(v_entry_1926_, 4);
v_isSharedCheck_1952_ = !lean_is_exclusive(v_entry_1926_);
if (v_isSharedCheck_1952_ == 0)
{
v___x_1943_ = v_entry_1926_;
v_isShared_1944_ = v_isSharedCheck_1952_;
goto v_resetjp_1942_;
}
else
{
lean_inc(v_computedKeys_1941_);
lean_inc(v_labelledStars_x3f_1940_);
lean_inc(v_mctx_1939_);
lean_inc(v_stack_1938_);
lean_inc(v_previous_1937_);
lean_dec(v_entry_1926_);
v___x_1943_ = lean_box(0);
v_isShared_1944_ = v_isSharedCheck_1952_;
goto v_resetjp_1942_;
}
v_resetjp_1942_:
{
lean_object* v___x_1945_; lean_object* v___x_1947_; 
v___x_1945_ = l_List_reverseAux___redArg(v_a_1933_, v_stack_1938_);
if (v_isShared_1944_ == 0)
{
lean_ctor_set(v___x_1943_, 1, v___x_1945_);
v___x_1947_ = v___x_1943_;
goto v_reusejp_1946_;
}
else
{
lean_object* v_reuseFailAlloc_1951_; 
v_reuseFailAlloc_1951_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1951_, 0, v_previous_1937_);
lean_ctor_set(v_reuseFailAlloc_1951_, 1, v___x_1945_);
lean_ctor_set(v_reuseFailAlloc_1951_, 2, v_mctx_1939_);
lean_ctor_set(v_reuseFailAlloc_1951_, 3, v_labelledStars_x3f_1940_);
lean_ctor_set(v_reuseFailAlloc_1951_, 4, v_computedKeys_1941_);
v___x_1947_ = v_reuseFailAlloc_1951_;
goto v_reusejp_1946_;
}
v_reusejp_1946_:
{
lean_object* v___x_1949_; 
if (v_isShared_1936_ == 0)
{
lean_ctor_set(v___x_1935_, 0, v___x_1947_);
v___x_1949_ = v___x_1935_;
goto v_reusejp_1948_;
}
else
{
lean_object* v_reuseFailAlloc_1950_; 
v_reuseFailAlloc_1950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1950_, 0, v___x_1947_);
v___x_1949_ = v_reuseFailAlloc_1950_;
goto v_reusejp_1948_;
}
v_reusejp_1948_:
{
return v___x_1949_;
}
}
}
}
}
else
{
lean_object* v_a_1954_; lean_object* v___x_1956_; uint8_t v_isShared_1957_; uint8_t v_isSharedCheck_1961_; 
lean_dec_ref(v_entry_1926_);
v_a_1954_ = lean_ctor_get(v___x_1932_, 0);
v_isSharedCheck_1961_ = !lean_is_exclusive(v___x_1932_);
if (v_isSharedCheck_1961_ == 0)
{
v___x_1956_ = v___x_1932_;
v_isShared_1957_ = v_isSharedCheck_1961_;
goto v_resetjp_1955_;
}
else
{
lean_inc(v_a_1954_);
lean_dec(v___x_1932_);
v___x_1956_ = lean_box(0);
v_isShared_1957_ = v_isSharedCheck_1961_;
goto v_resetjp_1955_;
}
v_resetjp_1955_:
{
lean_object* v___x_1959_; 
if (v_isShared_1957_ == 0)
{
v___x_1959_ = v___x_1956_;
goto v_reusejp_1958_;
}
else
{
lean_object* v_reuseFailAlloc_1960_; 
v_reuseFailAlloc_1960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1960_, 0, v_a_1954_);
v___x_1959_ = v_reuseFailAlloc_1960_;
goto v_reusejp_1958_;
}
v_reusejp_1958_:
{
return v___x_1959_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0___boxed(lean_object* v_x_1962_, lean_object* v_x_1963_, lean_object* v_bvars_1964_, lean_object* v_entry_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_){
_start:
{
lean_object* v_res_1971_; 
v_res_1971_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(v_x_1962_, v_x_1963_, v_bvars_1964_, v_entry_1965_, v___y_1966_, v___y_1967_, v___y_1968_, v___y_1969_);
lean_dec(v___y_1969_);
lean_dec_ref(v___y_1968_);
lean_dec(v___y_1967_);
lean_dec_ref(v___y_1966_);
return v_res_1971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(lean_object* v_bvars_1972_, lean_object* v___x_1973_, lean_object* v___x_1974_, lean_object* v___x_1975_, lean_object* v___x_1976_, lean_object* v_entry_1977_, lean_object* v_x_1978_, lean_object* v_x_1979_, lean_object* v_x_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_){
_start:
{
if (lean_obj_tag(v_x_1978_) == 5)
{
lean_object* v_fn_1986_; lean_object* v_arg_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; 
v_fn_1986_ = lean_ctor_get(v_x_1978_, 0);
lean_inc_ref(v_fn_1986_);
v_arg_1987_ = lean_ctor_get(v_x_1978_, 1);
lean_inc_ref(v_arg_1987_);
lean_dec_ref_known(v_x_1978_, 2);
v___x_1988_ = lean_array_set(v_x_1979_, v_x_1980_, v_arg_1987_);
v___x_1989_ = lean_unsigned_to_nat(1u);
v___x_1990_ = lean_nat_sub(v_x_1980_, v___x_1989_);
lean_dec(v_x_1980_);
v_x_1978_ = v_fn_1986_;
v_x_1979_ = v___x_1988_;
v_x_1980_ = v___x_1990_;
goto _start;
}
else
{
lean_dec(v_x_1980_);
switch(lean_obj_tag(v_x_1978_))
{
case 7:
{
lean_object* v_binderName_1992_; lean_object* v_binderType_1993_; lean_object* v_body_1994_; uint8_t v_binderInfo_1995_; lean_object* v___x_1996_; 
lean_dec_ref(v_x_1979_);
lean_dec_ref(v_entry_1977_);
v_binderName_1992_ = lean_ctor_get(v_x_1978_, 0);
lean_inc(v_binderName_1992_);
v_binderType_1993_ = lean_ctor_get(v_x_1978_, 1);
lean_inc_ref_n(v_binderType_1993_, 2);
v_body_1994_ = lean_ctor_get(v_x_1978_, 2);
lean_inc_ref(v_body_1994_);
v_binderInfo_1995_ = lean_ctor_get_uint8(v_x_1978_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_1978_, 3);
lean_inc(v_bvars_1972_);
v___x_1996_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v_binderType_1993_, v_bvars_1972_, v___y_1981_);
if (lean_obj_tag(v___x_1996_) == 0)
{
lean_object* v_a_1997_; lean_object* v___f_1998_; uint8_t v___x_1999_; lean_object* v___x_2000_; 
v_a_1997_ = lean_ctor_get(v___x_1996_, 0);
lean_inc(v_a_1997_);
lean_dec_ref_known(v___x_1996_, 1);
v___f_1998_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__1___boxed), 8, 2);
lean_closure_set(v___f_1998_, 0, v_body_1994_);
lean_closure_set(v___f_1998_, 1, v_bvars_1972_);
v___x_1999_ = 0;
v___x_2000_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__0___redArg(v_binderName_1992_, v_binderInfo_1995_, v_binderType_1993_, v___f_1998_, v___x_1999_, v___y_1981_, v___y_1982_, v___y_1983_, v___y_1984_);
if (lean_obj_tag(v___x_2000_) == 0)
{
lean_object* v_a_2001_; lean_object* v___x_2003_; uint8_t v_isShared_2004_; uint8_t v_isSharedCheck_2013_; 
v_a_2001_ = lean_ctor_get(v___x_2000_, 0);
v_isSharedCheck_2013_ = !lean_is_exclusive(v___x_2000_);
if (v_isSharedCheck_2013_ == 0)
{
v___x_2003_ = v___x_2000_;
v_isShared_2004_ = v_isSharedCheck_2013_;
goto v_resetjp_2002_;
}
else
{
lean_inc(v_a_2001_);
lean_dec(v___x_2000_);
v___x_2003_ = lean_box(0);
v_isShared_2004_ = v_isSharedCheck_2013_;
goto v_resetjp_2002_;
}
v_resetjp_2002_:
{
lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2011_; 
v___x_2005_ = lean_box(0);
v___x_2006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2006_, 0, v_a_1997_);
v___x_2007_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2007_, 0, v_a_2001_);
lean_ctor_set(v___x_2007_, 1, v___x_1973_);
v___x_2008_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2008_, 0, v___x_2006_);
lean_ctor_set(v___x_2008_, 1, v___x_2007_);
v___x_2009_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2009_, 0, v___x_2005_);
lean_ctor_set(v___x_2009_, 1, v___x_2008_);
lean_ctor_set(v___x_2009_, 2, v___x_1974_);
lean_ctor_set(v___x_2009_, 3, v___x_1975_);
lean_ctor_set(v___x_2009_, 4, v___x_1976_);
if (v_isShared_2004_ == 0)
{
lean_ctor_set(v___x_2003_, 0, v___x_2009_);
v___x_2011_ = v___x_2003_;
goto v_reusejp_2010_;
}
else
{
lean_object* v_reuseFailAlloc_2012_; 
v_reuseFailAlloc_2012_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2012_, 0, v___x_2009_);
v___x_2011_ = v_reuseFailAlloc_2012_;
goto v_reusejp_2010_;
}
v_reusejp_2010_:
{
return v___x_2011_;
}
}
}
else
{
lean_object* v_a_2014_; lean_object* v___x_2016_; uint8_t v_isShared_2017_; uint8_t v_isSharedCheck_2021_; 
lean_dec(v_a_1997_);
lean_dec(v___x_1976_);
lean_dec(v___x_1975_);
lean_dec_ref(v___x_1974_);
lean_dec(v___x_1973_);
v_a_2014_ = lean_ctor_get(v___x_2000_, 0);
v_isSharedCheck_2021_ = !lean_is_exclusive(v___x_2000_);
if (v_isSharedCheck_2021_ == 0)
{
v___x_2016_ = v___x_2000_;
v_isShared_2017_ = v_isSharedCheck_2021_;
goto v_resetjp_2015_;
}
else
{
lean_inc(v_a_2014_);
lean_dec(v___x_2000_);
v___x_2016_ = lean_box(0);
v_isShared_2017_ = v_isSharedCheck_2021_;
goto v_resetjp_2015_;
}
v_resetjp_2015_:
{
lean_object* v___x_2019_; 
if (v_isShared_2017_ == 0)
{
v___x_2019_ = v___x_2016_;
goto v_reusejp_2018_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v_a_2014_);
v___x_2019_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2018_;
}
v_reusejp_2018_:
{
return v___x_2019_;
}
}
}
}
else
{
lean_object* v_a_2022_; lean_object* v___x_2024_; uint8_t v_isShared_2025_; uint8_t v_isSharedCheck_2029_; 
lean_dec_ref(v_body_1994_);
lean_dec_ref(v_binderType_1993_);
lean_dec(v_binderName_1992_);
lean_dec(v___x_1976_);
lean_dec(v___x_1975_);
lean_dec_ref(v___x_1974_);
lean_dec(v___x_1973_);
lean_dec(v_bvars_1972_);
v_a_2022_ = lean_ctor_get(v___x_1996_, 0);
v_isSharedCheck_2029_ = !lean_is_exclusive(v___x_1996_);
if (v_isSharedCheck_2029_ == 0)
{
v___x_2024_ = v___x_1996_;
v_isShared_2025_ = v_isSharedCheck_2029_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_a_2022_);
lean_dec(v___x_1996_);
v___x_2024_ = lean_box(0);
v_isShared_2025_ = v_isSharedCheck_2029_;
goto v_resetjp_2023_;
}
v_resetjp_2023_:
{
lean_object* v___x_2027_; 
if (v_isShared_2025_ == 0)
{
v___x_2027_ = v___x_2024_;
goto v_reusejp_2026_;
}
else
{
lean_object* v_reuseFailAlloc_2028_; 
v_reuseFailAlloc_2028_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2028_, 0, v_a_2022_);
v___x_2027_ = v_reuseFailAlloc_2028_;
goto v_reusejp_2026_;
}
v_reusejp_2026_:
{
return v___x_2027_;
}
}
}
}
case 11:
{
lean_object* v_typeName_2030_; lean_object* v_struct_2031_; lean_object* v___x_2032_; 
lean_dec(v___x_1976_);
lean_dec(v___x_1975_);
lean_dec_ref(v___x_1974_);
lean_dec(v___x_1973_);
v_typeName_2030_ = lean_ctor_get(v_x_1978_, 0);
lean_inc(v_typeName_2030_);
v_struct_2031_ = lean_ctor_get(v_x_1978_, 2);
lean_inc_ref(v_struct_2031_);
lean_inc(v_bvars_1972_);
v___x_2032_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(v_x_1978_, v_x_1979_, v_bvars_1972_, v_entry_1977_, v___y_1981_, v___y_1982_, v___y_1983_, v___y_1984_);
if (lean_obj_tag(v___x_2032_) == 0)
{
lean_object* v_a_2033_; lean_object* v___x_2035_; uint8_t v_isShared_2036_; uint8_t v_isSharedCheck_2088_; 
v_a_2033_ = lean_ctor_get(v___x_2032_, 0);
v_isSharedCheck_2088_ = !lean_is_exclusive(v___x_2032_);
if (v_isSharedCheck_2088_ == 0)
{
v___x_2035_ = v___x_2032_;
v_isShared_2036_ = v_isSharedCheck_2088_;
goto v_resetjp_2034_;
}
else
{
lean_inc(v_a_2033_);
lean_dec(v___x_2032_);
v___x_2035_ = lean_box(0);
v_isShared_2036_ = v_isSharedCheck_2088_;
goto v_resetjp_2034_;
}
v_resetjp_2034_:
{
lean_object* v___x_2037_; lean_object* v_env_2038_; uint8_t v___x_2039_; 
v___x_2037_ = lean_st_ref_get(v___y_1984_);
v_env_2038_ = lean_ctor_get(v___x_2037_, 0);
lean_inc_ref(v_env_2038_);
lean_dec(v___x_2037_);
v___x_2039_ = l_Lean_isClass(v_env_2038_, v_typeName_2030_);
lean_dec(v_typeName_2030_);
if (v___x_2039_ == 0)
{
lean_object* v___x_2040_; 
lean_del_object(v___x_2035_);
v___x_2040_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v_struct_2031_, v_bvars_1972_, v___y_1981_);
if (lean_obj_tag(v___x_2040_) == 0)
{
lean_object* v_a_2041_; lean_object* v___x_2043_; uint8_t v_isShared_2044_; uint8_t v_isSharedCheck_2062_; 
v_a_2041_ = lean_ctor_get(v___x_2040_, 0);
v_isSharedCheck_2062_ = !lean_is_exclusive(v___x_2040_);
if (v_isSharedCheck_2062_ == 0)
{
v___x_2043_ = v___x_2040_;
v_isShared_2044_ = v_isSharedCheck_2062_;
goto v_resetjp_2042_;
}
else
{
lean_inc(v_a_2041_);
lean_dec(v___x_2040_);
v___x_2043_ = lean_box(0);
v_isShared_2044_ = v_isSharedCheck_2062_;
goto v_resetjp_2042_;
}
v_resetjp_2042_:
{
lean_object* v_previous_2045_; lean_object* v_stack_2046_; lean_object* v_mctx_2047_; lean_object* v_labelledStars_x3f_2048_; lean_object* v_computedKeys_2049_; lean_object* v___x_2051_; uint8_t v_isShared_2052_; uint8_t v_isSharedCheck_2061_; 
v_previous_2045_ = lean_ctor_get(v_a_2033_, 0);
v_stack_2046_ = lean_ctor_get(v_a_2033_, 1);
v_mctx_2047_ = lean_ctor_get(v_a_2033_, 2);
v_labelledStars_x3f_2048_ = lean_ctor_get(v_a_2033_, 3);
v_computedKeys_2049_ = lean_ctor_get(v_a_2033_, 4);
v_isSharedCheck_2061_ = !lean_is_exclusive(v_a_2033_);
if (v_isSharedCheck_2061_ == 0)
{
v___x_2051_ = v_a_2033_;
v_isShared_2052_ = v_isSharedCheck_2061_;
goto v_resetjp_2050_;
}
else
{
lean_inc(v_computedKeys_2049_);
lean_inc(v_labelledStars_x3f_2048_);
lean_inc(v_mctx_2047_);
lean_inc(v_stack_2046_);
lean_inc(v_previous_2045_);
lean_dec(v_a_2033_);
v___x_2051_ = lean_box(0);
v_isShared_2052_ = v_isSharedCheck_2061_;
goto v_resetjp_2050_;
}
v_resetjp_2050_:
{
lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2056_; 
v___x_2053_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2053_, 0, v_a_2041_);
v___x_2054_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2054_, 0, v___x_2053_);
lean_ctor_set(v___x_2054_, 1, v_stack_2046_);
if (v_isShared_2052_ == 0)
{
lean_ctor_set(v___x_2051_, 1, v___x_2054_);
v___x_2056_ = v___x_2051_;
goto v_reusejp_2055_;
}
else
{
lean_object* v_reuseFailAlloc_2060_; 
v_reuseFailAlloc_2060_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2060_, 0, v_previous_2045_);
lean_ctor_set(v_reuseFailAlloc_2060_, 1, v___x_2054_);
lean_ctor_set(v_reuseFailAlloc_2060_, 2, v_mctx_2047_);
lean_ctor_set(v_reuseFailAlloc_2060_, 3, v_labelledStars_x3f_2048_);
lean_ctor_set(v_reuseFailAlloc_2060_, 4, v_computedKeys_2049_);
v___x_2056_ = v_reuseFailAlloc_2060_;
goto v_reusejp_2055_;
}
v_reusejp_2055_:
{
lean_object* v___x_2058_; 
if (v_isShared_2044_ == 0)
{
lean_ctor_set(v___x_2043_, 0, v___x_2056_);
v___x_2058_ = v___x_2043_;
goto v_reusejp_2057_;
}
else
{
lean_object* v_reuseFailAlloc_2059_; 
v_reuseFailAlloc_2059_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2059_, 0, v___x_2056_);
v___x_2058_ = v_reuseFailAlloc_2059_;
goto v_reusejp_2057_;
}
v_reusejp_2057_:
{
return v___x_2058_;
}
}
}
}
}
else
{
lean_object* v_a_2063_; lean_object* v___x_2065_; uint8_t v_isShared_2066_; uint8_t v_isSharedCheck_2070_; 
lean_dec(v_a_2033_);
v_a_2063_ = lean_ctor_get(v___x_2040_, 0);
v_isSharedCheck_2070_ = !lean_is_exclusive(v___x_2040_);
if (v_isSharedCheck_2070_ == 0)
{
v___x_2065_ = v___x_2040_;
v_isShared_2066_ = v_isSharedCheck_2070_;
goto v_resetjp_2064_;
}
else
{
lean_inc(v_a_2063_);
lean_dec(v___x_2040_);
v___x_2065_ = lean_box(0);
v_isShared_2066_ = v_isSharedCheck_2070_;
goto v_resetjp_2064_;
}
v_resetjp_2064_:
{
lean_object* v___x_2068_; 
if (v_isShared_2066_ == 0)
{
v___x_2068_ = v___x_2065_;
goto v_reusejp_2067_;
}
else
{
lean_object* v_reuseFailAlloc_2069_; 
v_reuseFailAlloc_2069_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2069_, 0, v_a_2063_);
v___x_2068_ = v_reuseFailAlloc_2069_;
goto v_reusejp_2067_;
}
v_reusejp_2067_:
{
return v___x_2068_;
}
}
}
}
else
{
lean_object* v_previous_2071_; lean_object* v_stack_2072_; lean_object* v_mctx_2073_; lean_object* v_labelledStars_x3f_2074_; lean_object* v_computedKeys_2075_; lean_object* v___x_2077_; uint8_t v_isShared_2078_; uint8_t v_isSharedCheck_2087_; 
lean_dec_ref(v_struct_2031_);
lean_dec(v_bvars_1972_);
v_previous_2071_ = lean_ctor_get(v_a_2033_, 0);
v_stack_2072_ = lean_ctor_get(v_a_2033_, 1);
v_mctx_2073_ = lean_ctor_get(v_a_2033_, 2);
v_labelledStars_x3f_2074_ = lean_ctor_get(v_a_2033_, 3);
v_computedKeys_2075_ = lean_ctor_get(v_a_2033_, 4);
v_isSharedCheck_2087_ = !lean_is_exclusive(v_a_2033_);
if (v_isSharedCheck_2087_ == 0)
{
v___x_2077_ = v_a_2033_;
v_isShared_2078_ = v_isSharedCheck_2087_;
goto v_resetjp_2076_;
}
else
{
lean_inc(v_computedKeys_2075_);
lean_inc(v_labelledStars_x3f_2074_);
lean_inc(v_mctx_2073_);
lean_inc(v_stack_2072_);
lean_inc(v_previous_2071_);
lean_dec(v_a_2033_);
v___x_2077_ = lean_box(0);
v_isShared_2078_ = v_isSharedCheck_2087_;
goto v_resetjp_2076_;
}
v_resetjp_2076_:
{
lean_object* v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2082_; 
v___x_2079_ = lean_box(0);
v___x_2080_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2080_, 0, v___x_2079_);
lean_ctor_set(v___x_2080_, 1, v_stack_2072_);
if (v_isShared_2078_ == 0)
{
lean_ctor_set(v___x_2077_, 1, v___x_2080_);
v___x_2082_ = v___x_2077_;
goto v_reusejp_2081_;
}
else
{
lean_object* v_reuseFailAlloc_2086_; 
v_reuseFailAlloc_2086_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2086_, 0, v_previous_2071_);
lean_ctor_set(v_reuseFailAlloc_2086_, 1, v___x_2080_);
lean_ctor_set(v_reuseFailAlloc_2086_, 2, v_mctx_2073_);
lean_ctor_set(v_reuseFailAlloc_2086_, 3, v_labelledStars_x3f_2074_);
lean_ctor_set(v_reuseFailAlloc_2086_, 4, v_computedKeys_2075_);
v___x_2082_ = v_reuseFailAlloc_2086_;
goto v_reusejp_2081_;
}
v_reusejp_2081_:
{
lean_object* v___x_2084_; 
if (v_isShared_2036_ == 0)
{
lean_ctor_set(v___x_2035_, 0, v___x_2082_);
v___x_2084_ = v___x_2035_;
goto v_reusejp_2083_;
}
else
{
lean_object* v_reuseFailAlloc_2085_; 
v_reuseFailAlloc_2085_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2085_, 0, v___x_2082_);
v___x_2084_ = v_reuseFailAlloc_2085_;
goto v_reusejp_2083_;
}
v_reusejp_2083_:
{
return v___x_2084_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_struct_2031_);
lean_dec(v_typeName_2030_);
lean_dec(v_bvars_1972_);
return v___x_2032_;
}
}
default: 
{
lean_object* v___x_2089_; 
lean_dec(v___x_1976_);
lean_dec(v___x_1975_);
lean_dec_ref(v___x_1974_);
lean_dec(v___x_1973_);
v___x_2089_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___lam__0(v_x_1978_, v_x_1979_, v_bvars_1972_, v_entry_1977_, v___y_1981_, v___y_1982_, v___y_1983_, v___y_1984_);
return v___x_2089_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1___boxed(lean_object* v_bvars_2090_, lean_object* v___x_2091_, lean_object* v___x_2092_, lean_object* v___x_2093_, lean_object* v___x_2094_, lean_object* v_entry_2095_, lean_object* v_x_2096_, lean_object* v_x_2097_, lean_object* v_x_2098_, lean_object* v___y_2099_, lean_object* v___y_2100_, lean_object* v___y_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_){
_start:
{
lean_object* v_res_2104_; 
v_res_2104_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(v_bvars_2090_, v___x_2091_, v___x_2092_, v___x_2093_, v___x_2094_, v_entry_2095_, v_x_2096_, v_x_2097_, v_x_2098_, v___y_2099_, v___y_2100_, v___y_2101_, v___y_2102_);
lean_dec(v___y_2102_);
lean_dec_ref(v___y_2101_);
lean_dec(v___y_2100_);
lean_dec_ref(v___y_2099_);
return v_res_2104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0(lean_object* v_cfg_2105_, lean_object* v_bvars_2106_, lean_object* v_stack_2107_, lean_object* v_mctx_2108_, lean_object* v_labelledStars_x3f_2109_, lean_object* v_computedKeys_2110_, lean_object* v_entry_2111_, lean_object* v_expr_2112_, lean_object* v___x_2113_, lean_object* v___x_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_){
_start:
{
uint8_t v_trackZetaDelta_2120_; lean_object* v_zetaDeltaSet_2121_; lean_object* v_lctx_2122_; lean_object* v_localInstances_2123_; lean_object* v_defEqCtx_x3f_2124_; lean_object* v_synthPendingDepth_2125_; lean_object* v_customCanUnfoldPredicate_x3f_2126_; uint8_t v_univApprox_2127_; uint8_t v_inTypeClassResolution_2128_; uint8_t v_cacheInferType_2129_; lean_object* v___x_2131_; uint8_t v_isShared_2132_; uint8_t v_isSharedCheck_2139_; 
v_trackZetaDelta_2120_ = lean_ctor_get_uint8(v___y_2115_, sizeof(void*)*7);
v_zetaDeltaSet_2121_ = lean_ctor_get(v___y_2115_, 1);
v_lctx_2122_ = lean_ctor_get(v___y_2115_, 2);
v_localInstances_2123_ = lean_ctor_get(v___y_2115_, 3);
v_defEqCtx_x3f_2124_ = lean_ctor_get(v___y_2115_, 4);
v_synthPendingDepth_2125_ = lean_ctor_get(v___y_2115_, 5);
v_customCanUnfoldPredicate_x3f_2126_ = lean_ctor_get(v___y_2115_, 6);
v_univApprox_2127_ = lean_ctor_get_uint8(v___y_2115_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2128_ = lean_ctor_get_uint8(v___y_2115_, sizeof(void*)*7 + 2);
v_cacheInferType_2129_ = lean_ctor_get_uint8(v___y_2115_, sizeof(void*)*7 + 3);
v_isSharedCheck_2139_ = !lean_is_exclusive(v___y_2115_);
if (v_isSharedCheck_2139_ == 0)
{
lean_object* v_unused_2140_; 
v_unused_2140_ = lean_ctor_get(v___y_2115_, 0);
lean_dec(v_unused_2140_);
v___x_2131_ = v___y_2115_;
v_isShared_2132_ = v_isSharedCheck_2139_;
goto v_resetjp_2130_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2126_);
lean_inc(v_synthPendingDepth_2125_);
lean_inc(v_defEqCtx_x3f_2124_);
lean_inc(v_localInstances_2123_);
lean_inc(v_lctx_2122_);
lean_inc(v_zetaDeltaSet_2121_);
lean_dec(v___y_2115_);
v___x_2131_ = lean_box(0);
v_isShared_2132_ = v_isSharedCheck_2139_;
goto v_resetjp_2130_;
}
v_resetjp_2130_:
{
uint64_t v___x_2133_; lean_object* v___x_2134_; lean_object* v___x_2136_; 
v___x_2133_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v_cfg_2105_);
v___x_2134_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_2134_, 0, v_cfg_2105_);
lean_ctor_set_uint64(v___x_2134_, sizeof(void*)*1, v___x_2133_);
if (v_isShared_2132_ == 0)
{
lean_ctor_set(v___x_2131_, 0, v___x_2134_);
v___x_2136_ = v___x_2131_;
goto v_reusejp_2135_;
}
else
{
lean_object* v_reuseFailAlloc_2138_; 
v_reuseFailAlloc_2138_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2138_, 0, v___x_2134_);
lean_ctor_set(v_reuseFailAlloc_2138_, 1, v_zetaDeltaSet_2121_);
lean_ctor_set(v_reuseFailAlloc_2138_, 2, v_lctx_2122_);
lean_ctor_set(v_reuseFailAlloc_2138_, 3, v_localInstances_2123_);
lean_ctor_set(v_reuseFailAlloc_2138_, 4, v_defEqCtx_x3f_2124_);
lean_ctor_set(v_reuseFailAlloc_2138_, 5, v_synthPendingDepth_2125_);
lean_ctor_set(v_reuseFailAlloc_2138_, 6, v_customCanUnfoldPredicate_x3f_2126_);
lean_ctor_set_uint8(v_reuseFailAlloc_2138_, sizeof(void*)*7, v_trackZetaDelta_2120_);
lean_ctor_set_uint8(v_reuseFailAlloc_2138_, sizeof(void*)*7 + 1, v_univApprox_2127_);
lean_ctor_set_uint8(v_reuseFailAlloc_2138_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2128_);
lean_ctor_set_uint8(v_reuseFailAlloc_2138_, sizeof(void*)*7 + 3, v_cacheInferType_2129_);
v___x_2136_ = v_reuseFailAlloc_2138_;
goto v_reusejp_2135_;
}
v_reusejp_2135_:
{
lean_object* v___x_2137_; 
v___x_2137_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious_spec__1(v_bvars_2106_, v_stack_2107_, v_mctx_2108_, v_labelledStars_x3f_2109_, v_computedKeys_2110_, v_entry_2111_, v_expr_2112_, v___x_2113_, v___x_2114_, v___x_2136_, v___y_2116_, v___y_2117_, v___y_2118_);
lean_dec_ref(v___x_2136_);
return v___x_2137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0___boxed(lean_object* v_cfg_2141_, lean_object* v_bvars_2142_, lean_object* v_stack_2143_, lean_object* v_mctx_2144_, lean_object* v_labelledStars_x3f_2145_, lean_object* v_computedKeys_2146_, lean_object* v_entry_2147_, lean_object* v_expr_2148_, lean_object* v___x_2149_, lean_object* v___x_2150_, lean_object* v___y_2151_, lean_object* v___y_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_){
_start:
{
lean_object* v_res_2156_; 
v_res_2156_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0(v_cfg_2141_, v_bvars_2142_, v_stack_2143_, v_mctx_2144_, v_labelledStars_x3f_2145_, v_computedKeys_2146_, v_entry_2147_, v_expr_2148_, v___x_2149_, v___x_2150_, v___y_2151_, v___y_2152_, v___y_2153_, v___y_2154_);
lean_dec(v___y_2154_);
lean_dec_ref(v___y_2153_);
lean_dec(v___y_2152_);
return v_res_2156_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0(void){
_start:
{
lean_object* v___x_2157_; lean_object* v_dummy_2158_; 
v___x_2157_ = lean_box(0);
v_dummy_2158_ = l_Lean_Expr_sort___override(v___x_2157_);
return v_dummy_2158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(lean_object* v_entry_2159_, lean_object* v_a_2160_, lean_object* v_a_2161_, lean_object* v_a_2162_, lean_object* v_a_2163_){
_start:
{
lean_object* v_previous_2165_; 
v_previous_2165_ = lean_ctor_get(v_entry_2159_, 0);
if (lean_obj_tag(v_previous_2165_) == 1)
{
lean_object* v_val_2166_; lean_object* v_stack_2167_; lean_object* v_mctx_2168_; lean_object* v_labelledStars_x3f_2169_; lean_object* v_computedKeys_2170_; lean_object* v___x_2172_; uint8_t v_isShared_2173_; uint8_t v_isSharedCheck_2190_; 
v_val_2166_ = lean_ctor_get(v_previous_2165_, 0);
lean_inc(v_val_2166_);
v_stack_2167_ = lean_ctor_get(v_entry_2159_, 1);
v_mctx_2168_ = lean_ctor_get(v_entry_2159_, 2);
v_labelledStars_x3f_2169_ = lean_ctor_get(v_entry_2159_, 3);
v_computedKeys_2170_ = lean_ctor_get(v_entry_2159_, 4);
v_isSharedCheck_2190_ = !lean_is_exclusive(v_entry_2159_);
if (v_isSharedCheck_2190_ == 0)
{
lean_object* v_unused_2191_; 
v_unused_2191_ = lean_ctor_get(v_entry_2159_, 0);
lean_dec(v_unused_2191_);
v___x_2172_ = v_entry_2159_;
v_isShared_2173_ = v_isSharedCheck_2190_;
goto v_resetjp_2171_;
}
else
{
lean_inc(v_computedKeys_2170_);
lean_inc(v_labelledStars_x3f_2169_);
lean_inc(v_mctx_2168_);
lean_inc(v_stack_2167_);
lean_dec(v_entry_2159_);
v___x_2172_ = lean_box(0);
v_isShared_2173_ = v_isSharedCheck_2190_;
goto v_resetjp_2171_;
}
v_resetjp_2171_:
{
lean_object* v_expr_2174_; lean_object* v_bvars_2175_; lean_object* v_lctx_2176_; lean_object* v_localInsts_2177_; lean_object* v_cfg_2178_; lean_object* v___x_2179_; lean_object* v_entry_2181_; 
v_expr_2174_ = lean_ctor_get(v_val_2166_, 0);
lean_inc_ref(v_expr_2174_);
v_bvars_2175_ = lean_ctor_get(v_val_2166_, 1);
lean_inc(v_bvars_2175_);
v_lctx_2176_ = lean_ctor_get(v_val_2166_, 2);
lean_inc_ref(v_lctx_2176_);
v_localInsts_2177_ = lean_ctor_get(v_val_2166_, 3);
lean_inc_ref(v_localInsts_2177_);
v_cfg_2178_ = lean_ctor_get(v_val_2166_, 4);
lean_inc_ref(v_cfg_2178_);
lean_dec(v_val_2166_);
v___x_2179_ = lean_box(0);
lean_inc(v_computedKeys_2170_);
lean_inc(v_labelledStars_x3f_2169_);
lean_inc_ref(v_mctx_2168_);
lean_inc(v_stack_2167_);
if (v_isShared_2173_ == 0)
{
lean_ctor_set(v___x_2172_, 0, v___x_2179_);
v_entry_2181_ = v___x_2172_;
goto v_reusejp_2180_;
}
else
{
lean_object* v_reuseFailAlloc_2189_; 
v_reuseFailAlloc_2189_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2189_, 0, v___x_2179_);
lean_ctor_set(v_reuseFailAlloc_2189_, 1, v_stack_2167_);
lean_ctor_set(v_reuseFailAlloc_2189_, 2, v_mctx_2168_);
lean_ctor_set(v_reuseFailAlloc_2189_, 3, v_labelledStars_x3f_2169_);
lean_ctor_set(v_reuseFailAlloc_2189_, 4, v_computedKeys_2170_);
v_entry_2181_ = v_reuseFailAlloc_2189_;
goto v_reusejp_2180_;
}
v_reusejp_2180_:
{
lean_object* v_dummy_2182_; lean_object* v_nargs_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; lean_object* v___f_2187_; lean_object* v___x_2188_; 
v_dummy_2182_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0, &lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0_once, _init_lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___closed__0);
v_nargs_2183_ = l_Lean_Expr_getAppNumArgs(v_expr_2174_);
lean_inc(v_nargs_2183_);
v___x_2184_ = lean_mk_array(v_nargs_2183_, v_dummy_2182_);
v___x_2185_ = lean_unsigned_to_nat(1u);
v___x_2186_ = lean_nat_sub(v_nargs_2183_, v___x_2185_);
lean_dec(v_nargs_2183_);
v___f_2187_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___lam__0___boxed), 15, 10);
lean_closure_set(v___f_2187_, 0, v_cfg_2178_);
lean_closure_set(v___f_2187_, 1, v_bvars_2175_);
lean_closure_set(v___f_2187_, 2, v_stack_2167_);
lean_closure_set(v___f_2187_, 3, v_mctx_2168_);
lean_closure_set(v___f_2187_, 4, v_labelledStars_x3f_2169_);
lean_closure_set(v___f_2187_, 5, v_computedKeys_2170_);
lean_closure_set(v___f_2187_, 6, v_entry_2181_);
lean_closure_set(v___f_2187_, 7, v_expr_2174_);
lean_closure_set(v___f_2187_, 8, v___x_2184_);
lean_closure_set(v___f_2187_, 9, v___x_2186_);
v___x_2188_ = lp_mathlib_Lean_Meta_withLCtx___at___00__private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux_spec__0___redArg(v_lctx_2176_, v_localInsts_2177_, v___f_2187_, v_a_2160_, v_a_2161_, v_a_2162_, v_a_2163_);
return v___x_2188_;
}
}
}
else
{
lean_object* v___x_2192_; 
v___x_2192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2192_, 0, v_entry_2159_);
return v___x_2192_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious___boxed(lean_object* v_entry_2193_, lean_object* v_a_2194_, lean_object* v_a_2195_, lean_object* v_a_2196_, lean_object* v_a_2197_, lean_object* v_a_2198_){
_start:
{
lean_object* v_res_2199_; 
v_res_2199_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(v_entry_2193_, v_a_2194_, v_a_2195_, v_a_2196_, v_a_2197_);
lean_dec(v_a_2197_);
lean_dec_ref(v_a_2196_);
lean_dec(v_a_2195_);
lean_dec_ref(v_a_2194_);
return v_res_2199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(lean_object* v_mctx_2200_, lean_object* v_x_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_){
_start:
{
lean_object* v___x_2207_; 
v___x_2207_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMCtxImp(lean_box(0), v_mctx_2200_, v_x_2201_, v___y_2202_, v___y_2203_, v___y_2204_, v___y_2205_);
if (lean_obj_tag(v___x_2207_) == 0)
{
lean_object* v_a_2208_; lean_object* v___x_2210_; uint8_t v_isShared_2211_; uint8_t v_isSharedCheck_2215_; 
v_a_2208_ = lean_ctor_get(v___x_2207_, 0);
v_isSharedCheck_2215_ = !lean_is_exclusive(v___x_2207_);
if (v_isSharedCheck_2215_ == 0)
{
v___x_2210_ = v___x_2207_;
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
else
{
lean_inc(v_a_2208_);
lean_dec(v___x_2207_);
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
v_reuseFailAlloc_2214_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_2216_; lean_object* v___x_2218_; uint8_t v_isShared_2219_; uint8_t v_isSharedCheck_2223_; 
v_a_2216_ = lean_ctor_get(v___x_2207_, 0);
v_isSharedCheck_2223_ = !lean_is_exclusive(v___x_2207_);
if (v_isSharedCheck_2223_ == 0)
{
v___x_2218_ = v___x_2207_;
v_isShared_2219_ = v_isSharedCheck_2223_;
goto v_resetjp_2217_;
}
else
{
lean_inc(v_a_2216_);
lean_dec(v___x_2207_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg___boxed(lean_object* v_mctx_2224_, lean_object* v_x_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_, lean_object* v___y_2230_){
_start:
{
lean_object* v_res_2231_; 
v_res_2231_ = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(v_mctx_2224_, v_x_2225_, v___y_2226_, v___y_2227_, v___y_2228_, v___y_2229_);
lean_dec(v___y_2229_);
lean_dec_ref(v___y_2228_);
lean_dec(v___y_2227_);
lean_dec_ref(v___y_2226_);
return v_res_2231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0(lean_object* v_00_u03b1_2232_, lean_object* v_mctx_2233_, lean_object* v_x_2234_, lean_object* v___y_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_){
_start:
{
lean_object* v___x_2240_; 
v___x_2240_ = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(v_mctx_2233_, v_x_2234_, v___y_2235_, v___y_2236_, v___y_2237_, v___y_2238_);
return v___x_2240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___boxed(lean_object* v_00_u03b1_2241_, lean_object* v_mctx_2242_, lean_object* v_x_2243_, lean_object* v___y_2244_, lean_object* v___y_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_){
_start:
{
lean_object* v_res_2249_; 
v_res_2249_ = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0(v_00_u03b1_2241_, v_mctx_2242_, v_x_2243_, v___y_2244_, v___y_2245_, v___y_2246_, v___y_2247_);
lean_dec(v___y_2247_);
lean_dec_ref(v___y_2246_);
lean_dec(v___y_2245_);
lean_dec_ref(v___y_2244_);
return v_res_2249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0(lean_object* v_entry_2250_, uint8_t v_eta_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_){
_start:
{
lean_object* v___x_2257_; 
v___x_2257_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_processPrevious(v_entry_2250_, v___y_2252_, v___y_2253_, v___y_2254_, v___y_2255_);
if (lean_obj_tag(v___x_2257_) == 0)
{
lean_object* v_a_2258_; lean_object* v___x_2259_; 
v_a_2258_ = lean_ctor_get(v___x_2257_, 0);
lean_inc(v_a_2258_);
lean_dec_ref_known(v___x_2257_, 1);
v___x_2259_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_evalLazyEntryAux(v_a_2258_, v_eta_2251_, v___y_2252_, v___y_2253_, v___y_2254_, v___y_2255_);
return v___x_2259_;
}
else
{
lean_object* v_a_2260_; lean_object* v___x_2262_; uint8_t v_isShared_2263_; uint8_t v_isSharedCheck_2267_; 
v_a_2260_ = lean_ctor_get(v___x_2257_, 0);
v_isSharedCheck_2267_ = !lean_is_exclusive(v___x_2257_);
if (v_isSharedCheck_2267_ == 0)
{
v___x_2262_ = v___x_2257_;
v_isShared_2263_ = v_isSharedCheck_2267_;
goto v_resetjp_2261_;
}
else
{
lean_inc(v_a_2260_);
lean_dec(v___x_2257_);
v___x_2262_ = lean_box(0);
v_isShared_2263_ = v_isSharedCheck_2267_;
goto v_resetjp_2261_;
}
v_resetjp_2261_:
{
lean_object* v___x_2265_; 
if (v_isShared_2263_ == 0)
{
v___x_2265_ = v___x_2262_;
goto v_reusejp_2264_;
}
else
{
lean_object* v_reuseFailAlloc_2266_; 
v_reuseFailAlloc_2266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2266_, 0, v_a_2260_);
v___x_2265_ = v_reuseFailAlloc_2266_;
goto v_reusejp_2264_;
}
v_reusejp_2264_:
{
return v___x_2265_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0___boxed(lean_object* v_entry_2268_, lean_object* v_eta_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_){
_start:
{
uint8_t v_eta_boxed_2275_; lean_object* v_res_2276_; 
v_eta_boxed_2275_ = lean_unbox(v_eta_2269_);
v_res_2276_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0(v_entry_2268_, v_eta_boxed_2275_, v___y_2270_, v___y_2271_, v___y_2272_, v___y_2273_);
lean_dec(v___y_2273_);
lean_dec_ref(v___y_2272_);
lean_dec(v___y_2271_);
lean_dec_ref(v___y_2270_);
return v_res_2276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(lean_object* v_entry_2277_, uint8_t v_eta_2278_, lean_object* v_a_2279_, lean_object* v_a_2280_, lean_object* v_a_2281_, lean_object* v_a_2282_){
_start:
{
lean_object* v_computedKeys_2284_; 
v_computedKeys_2284_ = lean_ctor_get(v_entry_2277_, 4);
lean_inc(v_computedKeys_2284_);
if (lean_obj_tag(v_computedKeys_2284_) == 1)
{
lean_object* v_previous_2285_; lean_object* v_stack_2286_; lean_object* v_mctx_2287_; lean_object* v_labelledStars_x3f_2288_; lean_object* v___x_2290_; uint8_t v_isShared_2291_; uint8_t v_isSharedCheck_2308_; 
v_previous_2285_ = lean_ctor_get(v_entry_2277_, 0);
v_stack_2286_ = lean_ctor_get(v_entry_2277_, 1);
v_mctx_2287_ = lean_ctor_get(v_entry_2277_, 2);
v_labelledStars_x3f_2288_ = lean_ctor_get(v_entry_2277_, 3);
v_isSharedCheck_2308_ = !lean_is_exclusive(v_entry_2277_);
if (v_isSharedCheck_2308_ == 0)
{
lean_object* v_unused_2309_; 
v_unused_2309_ = lean_ctor_get(v_entry_2277_, 4);
lean_dec(v_unused_2309_);
v___x_2290_ = v_entry_2277_;
v_isShared_2291_ = v_isSharedCheck_2308_;
goto v_resetjp_2289_;
}
else
{
lean_inc(v_labelledStars_x3f_2288_);
lean_inc(v_mctx_2287_);
lean_inc(v_stack_2286_);
lean_inc(v_previous_2285_);
lean_dec(v_entry_2277_);
v___x_2290_ = lean_box(0);
v_isShared_2291_ = v_isSharedCheck_2308_;
goto v_resetjp_2289_;
}
v_resetjp_2289_:
{
lean_object* v_head_2292_; lean_object* v_tail_2293_; lean_object* v___x_2295_; uint8_t v_isShared_2296_; uint8_t v_isSharedCheck_2307_; 
v_head_2292_ = lean_ctor_get(v_computedKeys_2284_, 0);
v_tail_2293_ = lean_ctor_get(v_computedKeys_2284_, 1);
v_isSharedCheck_2307_ = !lean_is_exclusive(v_computedKeys_2284_);
if (v_isSharedCheck_2307_ == 0)
{
v___x_2295_ = v_computedKeys_2284_;
v_isShared_2296_ = v_isSharedCheck_2307_;
goto v_resetjp_2294_;
}
else
{
lean_inc(v_tail_2293_);
lean_inc(v_head_2292_);
lean_dec(v_computedKeys_2284_);
v___x_2295_ = lean_box(0);
v_isShared_2296_ = v_isSharedCheck_2307_;
goto v_resetjp_2294_;
}
v_resetjp_2294_:
{
lean_object* v___x_2298_; 
if (v_isShared_2291_ == 0)
{
lean_ctor_set(v___x_2290_, 4, v_tail_2293_);
v___x_2298_ = v___x_2290_;
goto v_reusejp_2297_;
}
else
{
lean_object* v_reuseFailAlloc_2306_; 
v_reuseFailAlloc_2306_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2306_, 0, v_previous_2285_);
lean_ctor_set(v_reuseFailAlloc_2306_, 1, v_stack_2286_);
lean_ctor_set(v_reuseFailAlloc_2306_, 2, v_mctx_2287_);
lean_ctor_set(v_reuseFailAlloc_2306_, 3, v_labelledStars_x3f_2288_);
lean_ctor_set(v_reuseFailAlloc_2306_, 4, v_tail_2293_);
v___x_2298_ = v_reuseFailAlloc_2306_;
goto v_reusejp_2297_;
}
v_reusejp_2297_:
{
lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2302_; 
v___x_2299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2299_, 0, v_head_2292_);
lean_ctor_set(v___x_2299_, 1, v___x_2298_);
v___x_2300_ = lean_box(0);
if (v_isShared_2296_ == 0)
{
lean_ctor_set(v___x_2295_, 1, v___x_2300_);
lean_ctor_set(v___x_2295_, 0, v___x_2299_);
v___x_2302_ = v___x_2295_;
goto v_reusejp_2301_;
}
else
{
lean_object* v_reuseFailAlloc_2305_; 
v_reuseFailAlloc_2305_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2305_, 0, v___x_2299_);
lean_ctor_set(v_reuseFailAlloc_2305_, 1, v___x_2300_);
v___x_2302_ = v_reuseFailAlloc_2305_;
goto v_reusejp_2301_;
}
v_reusejp_2301_:
{
lean_object* v___x_2303_; lean_object* v___x_2304_; 
v___x_2303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2303_, 0, v___x_2302_);
v___x_2304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2304_, 0, v___x_2303_);
return v___x_2304_;
}
}
}
}
}
else
{
lean_object* v_mctx_2310_; lean_object* v___x_2311_; lean_object* v___f_2312_; lean_object* v___x_2313_; 
lean_dec(v_computedKeys_2284_);
v_mctx_2310_ = lean_ctor_get(v_entry_2277_, 2);
lean_inc_ref(v_mctx_2310_);
v___x_2311_ = lean_box(v_eta_2278_);
v___f_2312_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___lam__0___boxed), 7, 2);
lean_closure_set(v___f_2312_, 0, v_entry_2277_);
lean_closure_set(v___f_2312_, 1, v___x_2311_);
v___x_2313_ = lp_mathlib_Lean_Meta_withMCtx___at___00Lean_Meta_RefinedDiscrTree_evalLazyEntry_spec__0___redArg(v_mctx_2310_, v___f_2312_, v_a_2279_, v_a_2280_, v_a_2281_, v_a_2282_);
return v___x_2313_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry___boxed(lean_object* v_entry_2314_, lean_object* v_eta_2315_, lean_object* v_a_2316_, lean_object* v_a_2317_, lean_object* v_a_2318_, lean_object* v_a_2319_, lean_object* v_a_2320_){
_start:
{
uint8_t v_eta_boxed_2321_; lean_object* v_res_2322_; 
v_eta_boxed_2321_ = lean_unbox(v_eta_2315_);
v_res_2322_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(v_entry_2314_, v_eta_boxed_2321_, v_a_2316_, v_a_2317_, v_a_2318_, v_a_2319_);
lean_dec(v_a_2319_);
lean_dec_ref(v_a_2318_);
lean_dec(v_a_2317_);
lean_dec_ref(v_a_2316_);
return v_res_2322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go_fold(lean_object* v_keys_2323_, lean_object* v_xs_2324_, lean_object* v_todo_2325_){
_start:
{
if (lean_obj_tag(v_xs_2324_) == 0)
{
lean_dec_ref(v_keys_2323_);
return v_todo_2325_;
}
else
{
lean_object* v_head_2326_; lean_object* v_tail_2327_; 
v_head_2326_ = lean_ctor_get(v_xs_2324_, 0);
lean_inc(v_head_2326_);
v_tail_2327_ = lean_ctor_get(v_xs_2324_, 1);
lean_inc(v_tail_2327_);
lean_dec_ref_known(v_xs_2324_, 2);
if (lean_obj_tag(v_tail_2327_) == 0)
{
lean_object* v_fst_2328_; lean_object* v_snd_2329_; lean_object* v___x_2331_; uint8_t v_isShared_2332_; uint8_t v_isSharedCheck_2338_; 
v_fst_2328_ = lean_ctor_get(v_head_2326_, 0);
v_snd_2329_ = lean_ctor_get(v_head_2326_, 1);
v_isSharedCheck_2338_ = !lean_is_exclusive(v_head_2326_);
if (v_isSharedCheck_2338_ == 0)
{
v___x_2331_ = v_head_2326_;
v_isShared_2332_ = v_isSharedCheck_2338_;
goto v_resetjp_2330_;
}
else
{
lean_inc(v_snd_2329_);
lean_inc(v_fst_2328_);
lean_dec(v_head_2326_);
v___x_2331_ = lean_box(0);
v_isShared_2332_ = v_isSharedCheck_2338_;
goto v_resetjp_2330_;
}
v_resetjp_2330_:
{
lean_object* v___x_2333_; lean_object* v___x_2335_; 
v___x_2333_ = lean_array_push(v_keys_2323_, v_fst_2328_);
if (v_isShared_2332_ == 0)
{
lean_ctor_set(v___x_2331_, 0, v___x_2333_);
v___x_2335_ = v___x_2331_;
goto v_reusejp_2334_;
}
else
{
lean_object* v_reuseFailAlloc_2337_; 
v_reuseFailAlloc_2337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2337_, 0, v___x_2333_);
lean_ctor_set(v_reuseFailAlloc_2337_, 1, v_snd_2329_);
v___x_2335_ = v_reuseFailAlloc_2337_;
goto v_reusejp_2334_;
}
v_reusejp_2334_:
{
lean_object* v___x_2336_; 
v___x_2336_ = lean_array_push(v_todo_2325_, v___x_2335_);
return v___x_2336_;
}
}
}
else
{
lean_object* v_fst_2339_; lean_object* v_snd_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2350_; 
v_fst_2339_ = lean_ctor_get(v_head_2326_, 0);
v_snd_2340_ = lean_ctor_get(v_head_2326_, 1);
v_isSharedCheck_2350_ = !lean_is_exclusive(v_head_2326_);
if (v_isSharedCheck_2350_ == 0)
{
v___x_2342_ = v_head_2326_;
v_isShared_2343_ = v_isSharedCheck_2350_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_snd_2340_);
lean_inc(v_fst_2339_);
lean_dec(v_head_2326_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2350_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v___x_2344_; lean_object* v___x_2346_; 
lean_inc_ref(v_keys_2323_);
v___x_2344_ = lean_array_push(v_keys_2323_, v_fst_2339_);
if (v_isShared_2343_ == 0)
{
lean_ctor_set(v___x_2342_, 0, v___x_2344_);
v___x_2346_ = v___x_2342_;
goto v_reusejp_2345_;
}
else
{
lean_object* v_reuseFailAlloc_2349_; 
v_reuseFailAlloc_2349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2349_, 0, v___x_2344_);
lean_ctor_set(v_reuseFailAlloc_2349_, 1, v_snd_2340_);
v___x_2346_ = v_reuseFailAlloc_2349_;
goto v_reusejp_2345_;
}
v_reusejp_2345_:
{
lean_object* v___x_2347_; 
v___x_2347_ = lean_array_push(v_todo_2325_, v___x_2346_);
v_xs_2324_ = v_tail_2327_;
v_todo_2325_ = v___x_2347_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(lean_object* v_todo_2351_, lean_object* v_result_2352_, lean_object* v_a_2353_, lean_object* v_a_2354_, lean_object* v_a_2355_, lean_object* v_a_2356_){
_start:
{
lean_object* v___x_2358_; lean_object* v___x_2359_; uint8_t v___x_2360_; 
v___x_2358_ = lean_array_get_size(v_todo_2351_);
v___x_2359_ = lean_unsigned_to_nat(0u);
v___x_2360_ = lean_nat_dec_eq(v___x_2358_, v___x_2359_);
if (v___x_2360_ == 0)
{
lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v_fst_2364_; lean_object* v_snd_2365_; uint8_t v___x_2366_; lean_object* v___x_2367_; 
v___x_2361_ = lean_unsigned_to_nat(1u);
v___x_2362_ = lean_nat_sub(v___x_2358_, v___x_2361_);
v___x_2363_ = lean_array_fget_borrowed(v_todo_2351_, v___x_2362_);
lean_dec(v___x_2362_);
v_fst_2364_ = lean_ctor_get(v___x_2363_, 0);
lean_inc(v_fst_2364_);
v_snd_2365_ = lean_ctor_get(v___x_2363_, 1);
v___x_2366_ = 1;
lean_inc(v_snd_2365_);
v___x_2367_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(v_snd_2365_, v___x_2366_, v_a_2353_, v_a_2354_, v_a_2355_, v_a_2356_);
if (lean_obj_tag(v___x_2367_) == 0)
{
lean_object* v_a_2368_; lean_object* v_todo_2369_; 
v_a_2368_ = lean_ctor_get(v___x_2367_, 0);
lean_inc(v_a_2368_);
lean_dec_ref_known(v___x_2367_, 1);
v_todo_2369_ = lean_array_pop(v_todo_2351_);
if (lean_obj_tag(v_a_2368_) == 0)
{
lean_object* v___x_2370_; 
v___x_2370_ = lean_array_push(v_result_2352_, v_fst_2364_);
v_todo_2351_ = v_todo_2369_;
v_result_2352_ = v___x_2370_;
goto _start;
}
else
{
lean_object* v_val_2372_; lean_object* v___x_2373_; 
v_val_2372_ = lean_ctor_get(v_a_2368_, 0);
lean_inc(v_val_2372_);
lean_dec_ref_known(v_a_2368_, 1);
v___x_2373_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go_fold(v_fst_2364_, v_val_2372_, v_todo_2369_);
v_todo_2351_ = v___x_2373_;
goto _start;
}
}
else
{
lean_object* v_a_2375_; lean_object* v___x_2377_; uint8_t v_isShared_2378_; uint8_t v_isSharedCheck_2382_; 
lean_dec(v_fst_2364_);
lean_dec_ref(v_result_2352_);
lean_dec_ref(v_todo_2351_);
v_a_2375_ = lean_ctor_get(v___x_2367_, 0);
v_isSharedCheck_2382_ = !lean_is_exclusive(v___x_2367_);
if (v_isSharedCheck_2382_ == 0)
{
v___x_2377_ = v___x_2367_;
v_isShared_2378_ = v_isSharedCheck_2382_;
goto v_resetjp_2376_;
}
else
{
lean_inc(v_a_2375_);
lean_dec(v___x_2367_);
v___x_2377_ = lean_box(0);
v_isShared_2378_ = v_isSharedCheck_2382_;
goto v_resetjp_2376_;
}
v_resetjp_2376_:
{
lean_object* v___x_2380_; 
if (v_isShared_2378_ == 0)
{
v___x_2380_ = v___x_2377_;
goto v_reusejp_2379_;
}
else
{
lean_object* v_reuseFailAlloc_2381_; 
v_reuseFailAlloc_2381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2381_, 0, v_a_2375_);
v___x_2380_ = v_reuseFailAlloc_2381_;
goto v_reusejp_2379_;
}
v_reusejp_2379_:
{
return v___x_2380_;
}
}
}
}
else
{
lean_object* v___x_2383_; 
lean_dec_ref(v_todo_2351_);
v___x_2383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2383_, 0, v_result_2352_);
return v___x_2383_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go___boxed(lean_object* v_todo_2384_, lean_object* v_result_2385_, lean_object* v_a_2386_, lean_object* v_a_2387_, lean_object* v_a_2388_, lean_object* v_a_2389_, lean_object* v_a_2390_){
_start:
{
lean_object* v_res_2391_; 
v_res_2391_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(v_todo_2384_, v_result_2385_, v_a_2386_, v_a_2387_, v_a_2388_, v_a_2389_);
lean_dec(v_a_2389_);
lean_dec_ref(v_a_2388_);
lean_dec(v_a_2387_);
lean_dec_ref(v_a_2386_);
return v_res_2391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(lean_object* v_a_2392_, lean_object* v_a_2393_){
_start:
{
if (lean_obj_tag(v_a_2392_) == 0)
{
lean_object* v___x_2394_; 
v___x_2394_ = l_List_reverse___redArg(v_a_2393_);
return v___x_2394_;
}
else
{
lean_object* v_head_2395_; lean_object* v_tail_2396_; lean_object* v___x_2398_; uint8_t v_isShared_2399_; uint8_t v_isSharedCheck_2416_; 
v_head_2395_ = lean_ctor_get(v_a_2392_, 0);
v_tail_2396_ = lean_ctor_get(v_a_2392_, 1);
v_isSharedCheck_2416_ = !lean_is_exclusive(v_a_2392_);
if (v_isSharedCheck_2416_ == 0)
{
v___x_2398_ = v_a_2392_;
v_isShared_2399_ = v_isSharedCheck_2416_;
goto v_resetjp_2397_;
}
else
{
lean_inc(v_tail_2396_);
lean_inc(v_head_2395_);
lean_dec(v_a_2392_);
v___x_2398_ = lean_box(0);
v_isShared_2399_ = v_isSharedCheck_2416_;
goto v_resetjp_2397_;
}
v_resetjp_2397_:
{
lean_object* v_fst_2400_; lean_object* v_snd_2401_; lean_object* v___x_2403_; uint8_t v_isShared_2404_; uint8_t v_isSharedCheck_2415_; 
v_fst_2400_ = lean_ctor_get(v_head_2395_, 0);
v_snd_2401_ = lean_ctor_get(v_head_2395_, 1);
v_isSharedCheck_2415_ = !lean_is_exclusive(v_head_2395_);
if (v_isSharedCheck_2415_ == 0)
{
v___x_2403_ = v_head_2395_;
v_isShared_2404_ = v_isSharedCheck_2415_;
goto v_resetjp_2402_;
}
else
{
lean_inc(v_snd_2401_);
lean_inc(v_fst_2400_);
lean_dec(v_head_2395_);
v___x_2403_ = lean_box(0);
v_isShared_2404_ = v_isSharedCheck_2415_;
goto v_resetjp_2402_;
}
v_resetjp_2402_:
{
lean_object* v___x_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2409_; 
v___x_2405_ = lean_unsigned_to_nat(1u);
v___x_2406_ = lean_mk_empty_array_with_capacity(v___x_2405_);
v___x_2407_ = lean_array_push(v___x_2406_, v_fst_2400_);
if (v_isShared_2404_ == 0)
{
lean_ctor_set(v___x_2403_, 0, v___x_2407_);
v___x_2409_ = v___x_2403_;
goto v_reusejp_2408_;
}
else
{
lean_object* v_reuseFailAlloc_2414_; 
v_reuseFailAlloc_2414_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2414_, 0, v___x_2407_);
lean_ctor_set(v_reuseFailAlloc_2414_, 1, v_snd_2401_);
v___x_2409_ = v_reuseFailAlloc_2414_;
goto v_reusejp_2408_;
}
v_reusejp_2408_:
{
lean_object* v___x_2411_; 
if (v_isShared_2399_ == 0)
{
lean_ctor_set(v___x_2398_, 1, v_a_2393_);
lean_ctor_set(v___x_2398_, 0, v___x_2409_);
v___x_2411_ = v___x_2398_;
goto v_reusejp_2410_;
}
else
{
lean_object* v_reuseFailAlloc_2413_; 
v_reuseFailAlloc_2413_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2413_, 0, v___x_2409_);
lean_ctor_set(v_reuseFailAlloc_2413_, 1, v_a_2393_);
v___x_2411_ = v_reuseFailAlloc_2413_;
goto v_reusejp_2410_;
}
v_reusejp_2410_:
{
v_a_2392_ = v_tail_2396_;
v_a_2393_ = v___x_2411_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta(lean_object* v_e_2419_, uint8_t v_labelledStars_2420_, lean_object* v_a_2421_, lean_object* v_a_2422_, lean_object* v_a_2423_, lean_object* v_a_2424_){
_start:
{
lean_object* v_keyedConfig_2426_; uint8_t v_trackZetaDelta_2427_; lean_object* v_zetaDeltaSet_2428_; lean_object* v_lctx_2429_; lean_object* v_localInstances_2430_; lean_object* v_defEqCtx_x3f_2431_; lean_object* v_synthPendingDepth_2432_; lean_object* v_customCanUnfoldPredicate_x3f_2433_; uint8_t v_univApprox_2434_; uint8_t v_inTypeClassResolution_2435_; uint8_t v_cacheInferType_2436_; lean_object* v___x_2437_; 
v_keyedConfig_2426_ = lean_ctor_get(v_a_2421_, 0);
v_trackZetaDelta_2427_ = lean_ctor_get_uint8(v_a_2421_, sizeof(void*)*7);
v_zetaDeltaSet_2428_ = lean_ctor_get(v_a_2421_, 1);
v_lctx_2429_ = lean_ctor_get(v_a_2421_, 2);
v_localInstances_2430_ = lean_ctor_get(v_a_2421_, 3);
v_defEqCtx_x3f_2431_ = lean_ctor_get(v_a_2421_, 4);
v_synthPendingDepth_2432_ = lean_ctor_get(v_a_2421_, 5);
v_customCanUnfoldPredicate_x3f_2433_ = lean_ctor_get(v_a_2421_, 6);
v_univApprox_2434_ = lean_ctor_get_uint8(v_a_2421_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2435_ = lean_ctor_get_uint8(v_a_2421_, sizeof(void*)*7 + 2);
v_cacheInferType_2436_ = lean_ctor_get_uint8(v_a_2421_, sizeof(void*)*7 + 3);
v___x_2437_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(v_labelledStars_2420_, v_a_2422_);
if (lean_obj_tag(v___x_2437_) == 0)
{
lean_object* v_a_2438_; uint8_t v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; uint8_t v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; 
v_a_2438_ = lean_ctor_get(v___x_2437_, 0);
lean_inc(v_a_2438_);
lean_dec_ref_known(v___x_2437_, 1);
v___x_2439_ = 2;
lean_inc_ref(v_keyedConfig_2426_);
v___x_2440_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2439_, v_keyedConfig_2426_);
lean_inc(v_customCanUnfoldPredicate_x3f_2433_);
lean_inc(v_synthPendingDepth_2432_);
lean_inc(v_defEqCtx_x3f_2431_);
lean_inc_ref(v_localInstances_2430_);
lean_inc_ref(v_lctx_2429_);
lean_inc(v_zetaDeltaSet_2428_);
v___x_2441_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2441_, 0, v___x_2440_);
lean_ctor_set(v___x_2441_, 1, v_zetaDeltaSet_2428_);
lean_ctor_set(v___x_2441_, 2, v_lctx_2429_);
lean_ctor_set(v___x_2441_, 3, v_localInstances_2430_);
lean_ctor_set(v___x_2441_, 4, v_defEqCtx_x3f_2431_);
lean_ctor_set(v___x_2441_, 5, v_synthPendingDepth_2432_);
lean_ctor_set(v___x_2441_, 6, v_customCanUnfoldPredicate_x3f_2433_);
lean_ctor_set_uint8(v___x_2441_, sizeof(void*)*7, v_trackZetaDelta_2427_);
lean_ctor_set_uint8(v___x_2441_, sizeof(void*)*7 + 1, v_univApprox_2434_);
lean_ctor_set_uint8(v___x_2441_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2435_);
lean_ctor_set_uint8(v___x_2441_, sizeof(void*)*7 + 3, v_cacheInferType_2436_);
v___x_2442_ = 1;
v___x_2443_ = lean_box(0);
v___x_2444_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepWithEta(v_e_2419_, v___x_2442_, v_a_2438_, v___x_2443_, v___x_2441_, v_a_2422_, v_a_2423_, v_a_2424_);
if (lean_obj_tag(v___x_2444_) == 0)
{
lean_object* v_a_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; 
v_a_2445_ = lean_ctor_get(v___x_2444_, 0);
lean_inc(v_a_2445_);
lean_dec_ref_known(v___x_2444_, 1);
v___x_2446_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_encodeExprWithEta_spec__0(v_a_2445_, v___x_2443_);
v___x_2447_ = lean_array_mk(v___x_2446_);
v___x_2448_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___closed__0));
v___x_2449_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodeExprWithEta_go(v___x_2447_, v___x_2448_, v___x_2441_, v_a_2422_, v_a_2423_, v_a_2424_);
lean_dec_ref_known(v___x_2441_, 7);
return v___x_2449_;
}
else
{
lean_object* v_a_2450_; lean_object* v___x_2452_; uint8_t v_isShared_2453_; uint8_t v_isSharedCheck_2457_; 
lean_dec_ref_known(v___x_2441_, 7);
v_a_2450_ = lean_ctor_get(v___x_2444_, 0);
v_isSharedCheck_2457_ = !lean_is_exclusive(v___x_2444_);
if (v_isSharedCheck_2457_ == 0)
{
v___x_2452_ = v___x_2444_;
v_isShared_2453_ = v_isSharedCheck_2457_;
goto v_resetjp_2451_;
}
else
{
lean_inc(v_a_2450_);
lean_dec(v___x_2444_);
v___x_2452_ = lean_box(0);
v_isShared_2453_ = v_isSharedCheck_2457_;
goto v_resetjp_2451_;
}
v_resetjp_2451_:
{
lean_object* v___x_2455_; 
if (v_isShared_2453_ == 0)
{
v___x_2455_ = v___x_2452_;
goto v_reusejp_2454_;
}
else
{
lean_object* v_reuseFailAlloc_2456_; 
v_reuseFailAlloc_2456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2456_, 0, v_a_2450_);
v___x_2455_ = v_reuseFailAlloc_2456_;
goto v_reusejp_2454_;
}
v_reusejp_2454_:
{
return v___x_2455_;
}
}
}
}
else
{
lean_object* v_a_2458_; lean_object* v___x_2460_; uint8_t v_isShared_2461_; uint8_t v_isSharedCheck_2465_; 
lean_dec_ref(v_e_2419_);
v_a_2458_ = lean_ctor_get(v___x_2437_, 0);
v_isSharedCheck_2465_ = !lean_is_exclusive(v___x_2437_);
if (v_isSharedCheck_2465_ == 0)
{
v___x_2460_ = v___x_2437_;
v_isShared_2461_ = v_isSharedCheck_2465_;
goto v_resetjp_2459_;
}
else
{
lean_inc(v_a_2458_);
lean_dec(v___x_2437_);
v___x_2460_ = lean_box(0);
v_isShared_2461_ = v_isSharedCheck_2465_;
goto v_resetjp_2459_;
}
v_resetjp_2459_:
{
lean_object* v___x_2463_; 
if (v_isShared_2461_ == 0)
{
v___x_2463_ = v___x_2460_;
goto v_reusejp_2462_;
}
else
{
lean_object* v_reuseFailAlloc_2464_; 
v_reuseFailAlloc_2464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2464_, 0, v_a_2458_);
v___x_2463_ = v_reuseFailAlloc_2464_;
goto v_reusejp_2462_;
}
v_reusejp_2462_:
{
return v___x_2463_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta___boxed(lean_object* v_e_2466_, lean_object* v_labelledStars_2467_, lean_object* v_a_2468_, lean_object* v_a_2469_, lean_object* v_a_2470_, lean_object* v_a_2471_, lean_object* v_a_2472_){
_start:
{
uint8_t v_labelledStars_boxed_2473_; lean_object* v_res_2474_; 
v_labelledStars_boxed_2473_ = lean_unbox(v_labelledStars_2467_);
v_res_2474_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExprWithEta(v_e_2466_, v_labelledStars_boxed_2473_, v_a_2468_, v_a_2469_, v_a_2470_, v_a_2471_);
lean_dec(v_a_2471_);
lean_dec_ref(v_a_2470_);
lean_dec(v_a_2469_);
lean_dec_ref(v_a_2468_);
return v_res_2474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(lean_object* v_msg_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_){
_start:
{
lean_object* v___f_2482_; lean_object* v___x_314__overap_2483_; lean_object* v___x_2484_; 
v___f_2482_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___closed__0));
v___x_314__overap_2483_ = lean_panic_fn_borrowed(v___f_2482_, v_msg_2476_);
lean_inc(v___y_2480_);
lean_inc_ref(v___y_2479_);
lean_inc(v___y_2478_);
lean_inc_ref(v___y_2477_);
v___x_2484_ = lean_apply_5(v___x_314__overap_2483_, v___y_2477_, v___y_2478_, v___y_2479_, v___y_2480_, lean_box(0));
return v___x_2484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0___boxed(lean_object* v_msg_2485_, lean_object* v___y_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_, lean_object* v___y_2489_, lean_object* v___y_2490_){
_start:
{
lean_object* v_res_2491_; 
v_res_2491_ = lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(v_msg_2485_, v___y_2486_, v___y_2487_, v___y_2488_, v___y_2489_);
lean_dec(v___y_2489_);
lean_dec_ref(v___y_2488_);
lean_dec(v___y_2487_);
lean_dec_ref(v___y_2486_);
return v_res_2491_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2(void){
_start:
{
lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; lean_object* v___x_2498_; lean_object* v___x_2499_; 
v___x_2494_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__1));
v___x_2495_ = lean_unsigned_to_nat(14u);
v___x_2496_ = lean_unsigned_to_nat(312u);
v___x_2497_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__0));
v___x_2498_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_encodingStepAux_go___closed__0));
v___x_2499_ = l_mkPanicMessageWithDecl(v___x_2498_, v___x_2497_, v___x_2496_, v___x_2495_, v___x_2494_);
return v___x_2499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(lean_object* v_entry_2500_, lean_object* v_result_2501_, lean_object* v_a_2502_, lean_object* v_a_2503_, lean_object* v_a_2504_, lean_object* v_a_2505_){
_start:
{
lean_object* v___y_2508_; lean_object* v___y_2509_; lean_object* v___y_2510_; lean_object* v___y_2511_; uint8_t v___x_2514_; lean_object* v___x_2515_; 
v___x_2514_ = 0;
v___x_2515_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_evalLazyEntry(v_entry_2500_, v___x_2514_, v_a_2502_, v_a_2503_, v_a_2504_, v_a_2505_);
if (lean_obj_tag(v___x_2515_) == 0)
{
lean_object* v_a_2516_; lean_object* v___x_2518_; uint8_t v_isShared_2519_; uint8_t v_isSharedCheck_2537_; 
v_a_2516_ = lean_ctor_get(v___x_2515_, 0);
v_isSharedCheck_2537_ = !lean_is_exclusive(v___x_2515_);
if (v_isSharedCheck_2537_ == 0)
{
v___x_2518_ = v___x_2515_;
v_isShared_2519_ = v_isSharedCheck_2537_;
goto v_resetjp_2517_;
}
else
{
lean_inc(v_a_2516_);
lean_dec(v___x_2515_);
v___x_2518_ = lean_box(0);
v_isShared_2519_ = v_isSharedCheck_2537_;
goto v_resetjp_2517_;
}
v_resetjp_2517_:
{
if (lean_obj_tag(v_a_2516_) == 0)
{
lean_object* v___x_2520_; lean_object* v___x_2522_; 
v___x_2520_ = l_List_reverse___redArg(v_result_2501_);
if (v_isShared_2519_ == 0)
{
lean_ctor_set(v___x_2518_, 0, v___x_2520_);
v___x_2522_ = v___x_2518_;
goto v_reusejp_2521_;
}
else
{
lean_object* v_reuseFailAlloc_2523_; 
v_reuseFailAlloc_2523_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2523_, 0, v___x_2520_);
v___x_2522_ = v_reuseFailAlloc_2523_;
goto v_reusejp_2521_;
}
v_reusejp_2521_:
{
return v___x_2522_;
}
}
else
{
lean_object* v_val_2524_; 
lean_del_object(v___x_2518_);
v_val_2524_ = lean_ctor_get(v_a_2516_, 0);
lean_inc(v_val_2524_);
lean_dec_ref_known(v_a_2516_, 1);
if (lean_obj_tag(v_val_2524_) == 1)
{
lean_object* v_head_2525_; lean_object* v_tail_2526_; lean_object* v___x_2528_; uint8_t v_isShared_2529_; uint8_t v_isSharedCheck_2536_; 
v_head_2525_ = lean_ctor_get(v_val_2524_, 0);
v_tail_2526_ = lean_ctor_get(v_val_2524_, 1);
v_isSharedCheck_2536_ = !lean_is_exclusive(v_val_2524_);
if (v_isSharedCheck_2536_ == 0)
{
v___x_2528_ = v_val_2524_;
v_isShared_2529_ = v_isSharedCheck_2536_;
goto v_resetjp_2527_;
}
else
{
lean_inc(v_tail_2526_);
lean_inc(v_head_2525_);
lean_dec(v_val_2524_);
v___x_2528_ = lean_box(0);
v_isShared_2529_ = v_isSharedCheck_2536_;
goto v_resetjp_2527_;
}
v_resetjp_2527_:
{
if (lean_obj_tag(v_tail_2526_) == 0)
{
lean_object* v_fst_2530_; lean_object* v_snd_2531_; lean_object* v___x_2533_; 
v_fst_2530_ = lean_ctor_get(v_head_2525_, 0);
lean_inc(v_fst_2530_);
v_snd_2531_ = lean_ctor_get(v_head_2525_, 1);
lean_inc(v_snd_2531_);
lean_dec(v_head_2525_);
if (v_isShared_2529_ == 0)
{
lean_ctor_set(v___x_2528_, 1, v_result_2501_);
lean_ctor_set(v___x_2528_, 0, v_fst_2530_);
v___x_2533_ = v___x_2528_;
goto v_reusejp_2532_;
}
else
{
lean_object* v_reuseFailAlloc_2535_; 
v_reuseFailAlloc_2535_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2535_, 0, v_fst_2530_);
lean_ctor_set(v_reuseFailAlloc_2535_, 1, v_result_2501_);
v___x_2533_ = v_reuseFailAlloc_2535_;
goto v_reusejp_2532_;
}
v_reusejp_2532_:
{
v_entry_2500_ = v_snd_2531_;
v_result_2501_ = v___x_2533_;
goto _start;
}
}
else
{
lean_del_object(v___x_2528_);
lean_dec(v_tail_2526_);
lean_dec(v_head_2525_);
lean_dec(v_result_2501_);
v___y_2508_ = v_a_2502_;
v___y_2509_ = v_a_2503_;
v___y_2510_ = v_a_2504_;
v___y_2511_ = v_a_2505_;
goto v___jp_2507_;
}
}
}
else
{
lean_dec(v_val_2524_);
lean_dec(v_result_2501_);
v___y_2508_ = v_a_2502_;
v___y_2509_ = v_a_2503_;
v___y_2510_ = v_a_2504_;
v___y_2511_ = v_a_2505_;
goto v___jp_2507_;
}
}
}
}
else
{
lean_object* v_a_2538_; lean_object* v___x_2540_; uint8_t v_isShared_2541_; uint8_t v_isSharedCheck_2545_; 
lean_dec(v_result_2501_);
v_a_2538_ = lean_ctor_get(v___x_2515_, 0);
v_isSharedCheck_2545_ = !lean_is_exclusive(v___x_2515_);
if (v_isSharedCheck_2545_ == 0)
{
v___x_2540_ = v___x_2515_;
v_isShared_2541_ = v_isSharedCheck_2545_;
goto v_resetjp_2539_;
}
else
{
lean_inc(v_a_2538_);
lean_dec(v___x_2515_);
v___x_2540_ = lean_box(0);
v_isShared_2541_ = v_isSharedCheck_2545_;
goto v_resetjp_2539_;
}
v_resetjp_2539_:
{
lean_object* v___x_2543_; 
if (v_isShared_2541_ == 0)
{
v___x_2543_ = v___x_2540_;
goto v_reusejp_2542_;
}
else
{
lean_object* v_reuseFailAlloc_2544_; 
v_reuseFailAlloc_2544_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2544_, 0, v_a_2538_);
v___x_2543_ = v_reuseFailAlloc_2544_;
goto v_reusejp_2542_;
}
v_reusejp_2542_:
{
return v___x_2543_;
}
}
}
v___jp_2507_:
{
lean_object* v___x_2512_; lean_object* v___x_2513_; 
v___x_2512_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2, &lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___closed__2);
v___x_2513_ = lp_mathlib_panic___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_toList_spec__0(v___x_2512_, v___y_2508_, v___y_2509_, v___y_2510_, v___y_2511_);
return v___x_2513_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList___boxed(lean_object* v_entry_2546_, lean_object* v_result_2547_, lean_object* v_a_2548_, lean_object* v_a_2549_, lean_object* v_a_2550_, lean_object* v_a_2551_, lean_object* v_a_2552_){
_start:
{
lean_object* v_res_2553_; 
v_res_2553_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(v_entry_2546_, v_result_2547_, v_a_2548_, v_a_2549_, v_a_2550_, v_a_2551_);
lean_dec(v_a_2551_);
lean_dec_ref(v_a_2550_);
lean_dec(v_a_2549_);
lean_dec_ref(v_a_2548_);
return v_res_2553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr(lean_object* v_e_2554_, uint8_t v_labelledStars_2555_, lean_object* v_a_2556_, lean_object* v_a_2557_, lean_object* v_a_2558_, lean_object* v_a_2559_){
_start:
{
lean_object* v_keyedConfig_2561_; uint8_t v_trackZetaDelta_2562_; lean_object* v_zetaDeltaSet_2563_; lean_object* v_lctx_2564_; lean_object* v_localInstances_2565_; lean_object* v_defEqCtx_x3f_2566_; lean_object* v_synthPendingDepth_2567_; lean_object* v_customCanUnfoldPredicate_x3f_2568_; uint8_t v_univApprox_2569_; uint8_t v_inTypeClassResolution_2570_; uint8_t v_cacheInferType_2571_; uint8_t v___x_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; 
v_keyedConfig_2561_ = lean_ctor_get(v_a_2556_, 0);
v_trackZetaDelta_2562_ = lean_ctor_get_uint8(v_a_2556_, sizeof(void*)*7);
v_zetaDeltaSet_2563_ = lean_ctor_get(v_a_2556_, 1);
v_lctx_2564_ = lean_ctor_get(v_a_2556_, 2);
v_localInstances_2565_ = lean_ctor_get(v_a_2556_, 3);
v_defEqCtx_x3f_2566_ = lean_ctor_get(v_a_2556_, 4);
v_synthPendingDepth_2567_ = lean_ctor_get(v_a_2556_, 5);
v_customCanUnfoldPredicate_x3f_2568_ = lean_ctor_get(v_a_2556_, 6);
v_univApprox_2569_ = lean_ctor_get_uint8(v_a_2556_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2570_ = lean_ctor_get_uint8(v_a_2556_, sizeof(void*)*7 + 2);
v_cacheInferType_2571_ = lean_ctor_get_uint8(v_a_2556_, sizeof(void*)*7 + 3);
v___x_2572_ = 2;
lean_inc_ref(v_keyedConfig_2561_);
v___x_2573_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2572_, v_keyedConfig_2561_);
lean_inc(v_customCanUnfoldPredicate_x3f_2568_);
lean_inc(v_synthPendingDepth_2567_);
lean_inc(v_defEqCtx_x3f_2566_);
lean_inc_ref(v_localInstances_2565_);
lean_inc_ref(v_lctx_2564_);
lean_inc(v_zetaDeltaSet_2563_);
v___x_2574_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2574_, 0, v___x_2573_);
lean_ctor_set(v___x_2574_, 1, v_zetaDeltaSet_2563_);
lean_ctor_set(v___x_2574_, 2, v_lctx_2564_);
lean_ctor_set(v___x_2574_, 3, v_localInstances_2565_);
lean_ctor_set(v___x_2574_, 4, v_defEqCtx_x3f_2566_);
lean_ctor_set(v___x_2574_, 5, v_synthPendingDepth_2567_);
lean_ctor_set(v___x_2574_, 6, v_customCanUnfoldPredicate_x3f_2568_);
lean_ctor_set_uint8(v___x_2574_, sizeof(void*)*7, v_trackZetaDelta_2562_);
lean_ctor_set_uint8(v___x_2574_, sizeof(void*)*7 + 1, v_univApprox_2569_);
lean_ctor_set_uint8(v___x_2574_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2570_);
lean_ctor_set_uint8(v___x_2574_, sizeof(void*)*7 + 3, v_cacheInferType_2571_);
v___x_2575_ = lp_mathlib___private_Mathlib_Lean_Meta_RefinedDiscrTree_Encode_0__Lean_Meta_RefinedDiscrTree_initializeLazyEntry(v_e_2554_, v_labelledStars_2555_, v___x_2574_, v_a_2557_, v_a_2558_, v_a_2559_);
if (lean_obj_tag(v___x_2575_) == 0)
{
lean_object* v_a_2576_; lean_object* v_fst_2577_; lean_object* v_snd_2578_; lean_object* v___x_2580_; uint8_t v_isShared_2581_; uint8_t v_isSharedCheck_2603_; 
v_a_2576_ = lean_ctor_get(v___x_2575_, 0);
lean_inc(v_a_2576_);
lean_dec_ref_known(v___x_2575_, 1);
v_fst_2577_ = lean_ctor_get(v_a_2576_, 0);
v_snd_2578_ = lean_ctor_get(v_a_2576_, 1);
v_isSharedCheck_2603_ = !lean_is_exclusive(v_a_2576_);
if (v_isSharedCheck_2603_ == 0)
{
v___x_2580_ = v_a_2576_;
v_isShared_2581_ = v_isSharedCheck_2603_;
goto v_resetjp_2579_;
}
else
{
lean_inc(v_snd_2578_);
lean_inc(v_fst_2577_);
lean_dec(v_a_2576_);
v___x_2580_ = lean_box(0);
v_isShared_2581_ = v_isSharedCheck_2603_;
goto v_resetjp_2579_;
}
v_resetjp_2579_:
{
lean_object* v___x_2582_; lean_object* v___x_2583_; 
v___x_2582_ = lean_box(0);
v___x_2583_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_toList(v_snd_2578_, v___x_2582_, v___x_2574_, v_a_2557_, v_a_2558_, v_a_2559_);
lean_dec_ref_known(v___x_2574_, 7);
if (lean_obj_tag(v___x_2583_) == 0)
{
lean_object* v_a_2584_; lean_object* v___x_2586_; uint8_t v_isShared_2587_; uint8_t v_isSharedCheck_2594_; 
v_a_2584_ = lean_ctor_get(v___x_2583_, 0);
v_isSharedCheck_2594_ = !lean_is_exclusive(v___x_2583_);
if (v_isSharedCheck_2594_ == 0)
{
v___x_2586_ = v___x_2583_;
v_isShared_2587_ = v_isSharedCheck_2594_;
goto v_resetjp_2585_;
}
else
{
lean_inc(v_a_2584_);
lean_dec(v___x_2583_);
v___x_2586_ = lean_box(0);
v_isShared_2587_ = v_isSharedCheck_2594_;
goto v_resetjp_2585_;
}
v_resetjp_2585_:
{
lean_object* v___x_2589_; 
if (v_isShared_2581_ == 0)
{
lean_ctor_set(v___x_2580_, 1, v_a_2584_);
v___x_2589_ = v___x_2580_;
goto v_reusejp_2588_;
}
else
{
lean_object* v_reuseFailAlloc_2593_; 
v_reuseFailAlloc_2593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2593_, 0, v_fst_2577_);
lean_ctor_set(v_reuseFailAlloc_2593_, 1, v_a_2584_);
v___x_2589_ = v_reuseFailAlloc_2593_;
goto v_reusejp_2588_;
}
v_reusejp_2588_:
{
lean_object* v___x_2591_; 
if (v_isShared_2587_ == 0)
{
lean_ctor_set(v___x_2586_, 0, v___x_2589_);
v___x_2591_ = v___x_2586_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2592_; 
v_reuseFailAlloc_2592_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2592_, 0, v___x_2589_);
v___x_2591_ = v_reuseFailAlloc_2592_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
return v___x_2591_;
}
}
}
}
else
{
lean_object* v_a_2595_; lean_object* v___x_2597_; uint8_t v_isShared_2598_; uint8_t v_isSharedCheck_2602_; 
lean_del_object(v___x_2580_);
lean_dec(v_fst_2577_);
v_a_2595_ = lean_ctor_get(v___x_2583_, 0);
v_isSharedCheck_2602_ = !lean_is_exclusive(v___x_2583_);
if (v_isSharedCheck_2602_ == 0)
{
v___x_2597_ = v___x_2583_;
v_isShared_2598_ = v_isSharedCheck_2602_;
goto v_resetjp_2596_;
}
else
{
lean_inc(v_a_2595_);
lean_dec(v___x_2583_);
v___x_2597_ = lean_box(0);
v_isShared_2598_ = v_isSharedCheck_2602_;
goto v_resetjp_2596_;
}
v_resetjp_2596_:
{
lean_object* v___x_2600_; 
if (v_isShared_2598_ == 0)
{
v___x_2600_ = v___x_2597_;
goto v_reusejp_2599_;
}
else
{
lean_object* v_reuseFailAlloc_2601_; 
v_reuseFailAlloc_2601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2601_, 0, v_a_2595_);
v___x_2600_ = v_reuseFailAlloc_2601_;
goto v_reusejp_2599_;
}
v_reusejp_2599_:
{
return v___x_2600_;
}
}
}
}
}
else
{
lean_object* v_a_2604_; lean_object* v___x_2606_; uint8_t v_isShared_2607_; uint8_t v_isSharedCheck_2611_; 
lean_dec_ref_known(v___x_2574_, 7);
v_a_2604_ = lean_ctor_get(v___x_2575_, 0);
v_isSharedCheck_2611_ = !lean_is_exclusive(v___x_2575_);
if (v_isSharedCheck_2611_ == 0)
{
v___x_2606_ = v___x_2575_;
v_isShared_2607_ = v_isSharedCheck_2611_;
goto v_resetjp_2605_;
}
else
{
lean_inc(v_a_2604_);
lean_dec(v___x_2575_);
v___x_2606_ = lean_box(0);
v_isShared_2607_ = v_isSharedCheck_2611_;
goto v_resetjp_2605_;
}
v_resetjp_2605_:
{
lean_object* v___x_2609_; 
if (v_isShared_2607_ == 0)
{
v___x_2609_ = v___x_2606_;
goto v_reusejp_2608_;
}
else
{
lean_object* v_reuseFailAlloc_2610_; 
v_reuseFailAlloc_2610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2610_, 0, v_a_2604_);
v___x_2609_ = v_reuseFailAlloc_2610_;
goto v_reusejp_2608_;
}
v_reusejp_2608_:
{
return v___x_2609_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr___boxed(lean_object* v_e_2612_, lean_object* v_labelledStars_2613_, lean_object* v_a_2614_, lean_object* v_a_2615_, lean_object* v_a_2616_, lean_object* v_a_2617_, lean_object* v_a_2618_){
_start:
{
uint8_t v_labelledStars_boxed_2619_; lean_object* v_res_2620_; 
v_labelledStars_boxed_2619_ = lean_unbox(v_labelledStars_2613_);
v_res_2620_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_encodeExpr(v_e_2612_, v_labelledStars_boxed_2619_, v_a_2614_, v_a_2615_, v_a_2616_, v_a_2617_);
lean_dec(v_a_2617_);
lean_dec_ref(v_a_2616_);
lean_dec(v_a_2615_);
lean_dec_ref(v_a_2614_);
return v_res_2620_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_LazyDiscrTree(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Encode(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_LazyDiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Encode(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree(uint8_t builtin);
lean_object* initialize_Lean_Meta_LazyDiscrTree(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Encode(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_LazyDiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Encode(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Encode(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Encode(builtin);
}
#ifdef __cplusplus
}
#endif
