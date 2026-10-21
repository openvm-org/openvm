// Lean compiler output
// Module: Mathlib.Tactic.FunProp.Mor
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.CoeAttr public import Lean.Meta.CoeAttr
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Meta_getCoeFnInfo_x3f___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs_x27(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getProjFnForField_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getStructureInfo_x3f(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedStructureInfo_default;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Array_reverse___redArg(lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_addZetaDeltaFVarId___redArg(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_unfoldDefinition_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_instDecidableEqCoeFnType(uint8_t, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "loose bvar in expression"};
static const lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Lean.Meta.whnfEasyCases"};
static const lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Lean.Meta.WHNF"};
static const lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_app(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__2(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "Mathlib.Tactic.FunProp.Mor"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 74, .m_capacity = 74, .m_length = 73, .m_data = "_private.Mathlib.Tactic.FunProp.Mor.0.Mathlib.Meta.FunProp.Mor.withApp.go"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "bug in Mor.withApp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "FunProp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Forall"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__8_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(51, 139, 43, 17, 218, 109, 44, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12_value;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_Mor_mkAppN_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_Mor_mkAppN_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg(lean_object* v_name_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = l_Lean_Meta_getCoeFnInfo_x3f___redArg(v_name_1_, v_a_2_);
if (lean_obj_tag(v___x_4_) == 0)
{
lean_object* v_a_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_22_; 
v_a_5_ = lean_ctor_get(v___x_4_, 0);
v_isSharedCheck_22_ = !lean_is_exclusive(v___x_4_);
if (v_isSharedCheck_22_ == 0)
{
v___x_7_ = v___x_4_;
v_isShared_8_ = v_isSharedCheck_22_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_a_5_);
lean_dec(v___x_4_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_22_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
if (lean_obj_tag(v_a_5_) == 1)
{
lean_object* v_val_9_; uint8_t v_type_10_; uint8_t v___x_11_; uint8_t v___x_12_; lean_object* v___x_13_; lean_object* v___x_15_; 
v_val_9_ = lean_ctor_get(v_a_5_, 0);
lean_inc(v_val_9_);
lean_dec_ref_known(v_a_5_, 1);
v_type_10_ = lean_ctor_get_uint8(v_val_9_, sizeof(void*)*2);
lean_dec(v_val_9_);
v___x_11_ = 1;
v___x_12_ = l_Lean_Meta_instDecidableEqCoeFnType(v_type_10_, v___x_11_);
v___x_13_ = lean_box(v___x_12_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 0, v___x_13_);
v___x_15_ = v___x_7_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v___x_13_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
else
{
uint8_t v___x_17_; lean_object* v___x_18_; lean_object* v___x_20_; 
lean_dec(v_a_5_);
v___x_17_ = 0;
v___x_18_ = lean_box(v___x_17_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 0, v___x_18_);
v___x_20_ = v___x_7_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v___x_18_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
}
else
{
lean_object* v_a_23_; lean_object* v___x_25_; uint8_t v_isShared_26_; uint8_t v_isSharedCheck_30_; 
v_a_23_ = lean_ctor_get(v___x_4_, 0);
v_isSharedCheck_30_ = !lean_is_exclusive(v___x_4_);
if (v_isSharedCheck_30_ == 0)
{
v___x_25_ = v___x_4_;
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
else
{
lean_inc(v_a_23_);
lean_dec(v___x_4_);
v___x_25_ = lean_box(0);
v_isShared_26_ = v_isSharedCheck_30_;
goto v_resetjp_24_;
}
v_resetjp_24_:
{
lean_object* v___x_28_; 
if (v_isShared_26_ == 0)
{
v___x_28_ = v___x_25_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v_a_23_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg___boxed(lean_object* v_name_31_, lean_object* v_a_32_, lean_object* v_a_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg(v_name_31_, v_a_32_);
lean_dec(v_a_32_);
lean_dec(v_name_31_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName(lean_object* v_name_35_, lean_object* v_a_36_, lean_object* v_a_37_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___redArg(v_name_35_, v_a_37_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName___boxed(lean_object* v_name_40_, lean_object* v_a_41_, lean_object* v_a_42_, lean_object* v_a_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFunName(v_name_40_, v_a_41_, v_a_42_);
lean_dec(v_a_42_);
lean_dec_ref(v_a_41_);
lean_dec(v_name_40_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg(lean_object* v_e_45_, lean_object* v_a_46_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = l_Lean_Expr_getAppFn(v_e_45_);
if (lean_obj_tag(v___x_48_) == 4)
{
lean_object* v_declName_49_; lean_object* v___x_50_; 
v_declName_49_ = lean_ctor_get(v___x_48_, 0);
lean_inc(v_declName_49_);
lean_dec_ref_known(v___x_48_, 2);
v___x_50_ = l_Lean_Meta_getCoeFnInfo_x3f___redArg(v_declName_49_, v_a_46_);
lean_dec(v_declName_49_);
if (lean_obj_tag(v___x_50_) == 0)
{
lean_object* v_a_51_; lean_object* v___x_53_; uint8_t v_isShared_54_; uint8_t v_isSharedCheck_70_; 
v_a_51_ = lean_ctor_get(v___x_50_, 0);
v_isSharedCheck_70_ = !lean_is_exclusive(v___x_50_);
if (v_isSharedCheck_70_ == 0)
{
v___x_53_ = v___x_50_;
v_isShared_54_ = v_isSharedCheck_70_;
goto v_resetjp_52_;
}
else
{
lean_inc(v_a_51_);
lean_dec(v___x_50_);
v___x_53_ = lean_box(0);
v_isShared_54_ = v_isSharedCheck_70_;
goto v_resetjp_52_;
}
v_resetjp_52_:
{
if (lean_obj_tag(v_a_51_) == 1)
{
lean_object* v_val_55_; lean_object* v_numArgs_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; uint8_t v___x_60_; lean_object* v___x_61_; lean_object* v___x_63_; 
v_val_55_ = lean_ctor_get(v_a_51_, 0);
lean_inc(v_val_55_);
lean_dec_ref_known(v_a_51_, 1);
v_numArgs_56_ = lean_ctor_get(v_val_55_, 0);
lean_inc(v_numArgs_56_);
lean_dec(v_val_55_);
v___x_57_ = l_Lean_Expr_getAppNumArgs_x27(v_e_45_);
v___x_58_ = lean_unsigned_to_nat(1u);
v___x_59_ = lean_nat_add(v___x_57_, v___x_58_);
lean_dec(v___x_57_);
v___x_60_ = lean_nat_dec_eq(v___x_59_, v_numArgs_56_);
lean_dec(v_numArgs_56_);
lean_dec(v___x_59_);
v___x_61_ = lean_box(v___x_60_);
if (v_isShared_54_ == 0)
{
lean_ctor_set(v___x_53_, 0, v___x_61_);
v___x_63_ = v___x_53_;
goto v_reusejp_62_;
}
else
{
lean_object* v_reuseFailAlloc_64_; 
v_reuseFailAlloc_64_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_64_, 0, v___x_61_);
v___x_63_ = v_reuseFailAlloc_64_;
goto v_reusejp_62_;
}
v_reusejp_62_:
{
return v___x_63_;
}
}
else
{
uint8_t v___x_65_; lean_object* v___x_66_; lean_object* v___x_68_; 
lean_dec(v_a_51_);
v___x_65_ = 0;
v___x_66_ = lean_box(v___x_65_);
if (v_isShared_54_ == 0)
{
lean_ctor_set(v___x_53_, 0, v___x_66_);
v___x_68_ = v___x_53_;
goto v_reusejp_67_;
}
else
{
lean_object* v_reuseFailAlloc_69_; 
v_reuseFailAlloc_69_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_69_, 0, v___x_66_);
v___x_68_ = v_reuseFailAlloc_69_;
goto v_reusejp_67_;
}
v_reusejp_67_:
{
return v___x_68_;
}
}
}
}
else
{
lean_object* v_a_71_; lean_object* v___x_73_; uint8_t v_isShared_74_; uint8_t v_isSharedCheck_78_; 
v_a_71_ = lean_ctor_get(v___x_50_, 0);
v_isSharedCheck_78_ = !lean_is_exclusive(v___x_50_);
if (v_isSharedCheck_78_ == 0)
{
v___x_73_ = v___x_50_;
v_isShared_74_ = v_isSharedCheck_78_;
goto v_resetjp_72_;
}
else
{
lean_inc(v_a_71_);
lean_dec(v___x_50_);
v___x_73_ = lean_box(0);
v_isShared_74_ = v_isSharedCheck_78_;
goto v_resetjp_72_;
}
v_resetjp_72_:
{
lean_object* v___x_76_; 
if (v_isShared_74_ == 0)
{
v___x_76_ = v___x_73_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v_a_71_);
v___x_76_ = v_reuseFailAlloc_77_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
return v___x_76_;
}
}
}
}
else
{
uint8_t v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
lean_dec_ref(v___x_48_);
v___x_79_ = 0;
v___x_80_ = lean_box(v___x_79_);
v___x_81_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
return v___x_81_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg___boxed(lean_object* v_e_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg(v_e_82_, v_a_83_);
lean_dec(v_a_83_);
lean_dec_ref(v_e_82_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun(lean_object* v_e_86_, lean_object* v_a_87_, lean_object* v_a_88_, lean_object* v_a_89_, lean_object* v_a_90_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg(v_e_86_, v_a_90_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___boxed(lean_object* v_e_93_, lean_object* v_a_94_, lean_object* v_a_95_, lean_object* v_a_96_, lean_object* v_a_97_, lean_object* v_a_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun(v_e_93_, v_a_94_, v_a_95_, v_a_96_, v_a_97_);
lean_dec(v_a_97_);
lean_dec_ref(v_a_96_);
lean_dec(v_a_95_);
lean_dec_ref(v_a_94_);
lean_dec_ref(v_e_93_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___redArg(lean_object* v_e_100_, lean_object* v_a_101_){
_start:
{
if (lean_obj_tag(v_e_100_) == 5)
{
lean_object* v_fn_106_; 
v_fn_106_ = lean_ctor_get(v_e_100_, 0);
if (lean_obj_tag(v_fn_106_) == 5)
{
lean_object* v_arg_107_; lean_object* v_fn_108_; lean_object* v_arg_109_; lean_object* v___x_110_; 
v_arg_107_ = lean_ctor_get(v_e_100_, 1);
v_fn_108_ = lean_ctor_get(v_fn_106_, 0);
v_arg_109_ = lean_ctor_get(v_fn_106_, 1);
v___x_110_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg(v_fn_108_, v_a_101_);
if (lean_obj_tag(v___x_110_) == 0)
{
lean_object* v_a_111_; lean_object* v___x_113_; uint8_t v_isShared_114_; uint8_t v_isSharedCheck_125_; 
v_a_111_ = lean_ctor_get(v___x_110_, 0);
v_isSharedCheck_125_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_125_ == 0)
{
v___x_113_ = v___x_110_;
v_isShared_114_ = v_isSharedCheck_125_;
goto v_resetjp_112_;
}
else
{
lean_inc(v_a_111_);
lean_dec(v___x_110_);
v___x_113_ = lean_box(0);
v_isShared_114_ = v_isSharedCheck_125_;
goto v_resetjp_112_;
}
v_resetjp_112_:
{
uint8_t v___x_115_; 
v___x_115_ = lean_unbox(v_a_111_);
lean_dec(v_a_111_);
if (v___x_115_ == 0)
{
lean_object* v___x_116_; lean_object* v___x_118_; 
v___x_116_ = lean_box(0);
if (v_isShared_114_ == 0)
{
lean_ctor_set(v___x_113_, 0, v___x_116_);
v___x_118_ = v___x_113_;
goto v_reusejp_117_;
}
else
{
lean_object* v_reuseFailAlloc_119_; 
v_reuseFailAlloc_119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_119_, 0, v___x_116_);
v___x_118_ = v_reuseFailAlloc_119_;
goto v_reusejp_117_;
}
v_reusejp_117_:
{
return v___x_118_;
}
}
else
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_123_; 
lean_inc_ref(v_arg_107_);
lean_inc_ref(v_arg_109_);
lean_inc_ref(v_fn_108_);
v___x_120_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_120_, 0, v_fn_108_);
lean_ctor_set(v___x_120_, 1, v_arg_109_);
lean_ctor_set(v___x_120_, 2, v_arg_107_);
v___x_121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
if (v_isShared_114_ == 0)
{
lean_ctor_set(v___x_113_, 0, v___x_121_);
v___x_123_ = v___x_113_;
goto v_reusejp_122_;
}
else
{
lean_object* v_reuseFailAlloc_124_; 
v_reuseFailAlloc_124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_124_, 0, v___x_121_);
v___x_123_ = v_reuseFailAlloc_124_;
goto v_reusejp_122_;
}
v_reusejp_122_:
{
return v___x_123_;
}
}
}
}
else
{
lean_object* v_a_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_133_; 
v_a_126_ = lean_ctor_get(v___x_110_, 0);
v_isSharedCheck_133_ = !lean_is_exclusive(v___x_110_);
if (v_isSharedCheck_133_ == 0)
{
v___x_128_ = v___x_110_;
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_a_126_);
lean_dec(v___x_110_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_131_; 
if (v_isShared_129_ == 0)
{
v___x_131_ = v___x_128_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v_a_126_);
v___x_131_ = v_reuseFailAlloc_132_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
return v___x_131_;
}
}
}
}
else
{
goto v___jp_103_;
}
}
else
{
goto v___jp_103_;
}
v___jp_103_:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
return v___x_105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___redArg___boxed(lean_object* v_e_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___redArg(v_e_134_, v_a_135_);
lean_dec(v_a_135_);
lean_dec_ref(v_e_134_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f(lean_object* v_e_138_, lean_object* v_a_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___redArg(v_e_138_, v_a_142_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___boxed(lean_object* v_e_145_, lean_object* v_a_146_, lean_object* v_a_147_, lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f(v_e_145_, v_a_146_, v_a_147_, v_a_148_, v_a_149_);
lean_dec(v_a_149_);
lean_dec_ref(v_a_148_);
lean_dec(v_a_147_);
lean_dec_ref(v_a_146_);
lean_dec_ref(v_e_145_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2(lean_object* v_msg_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v___f_159_; lean_object* v___x_2111__overap_160_; lean_object* v___x_161_; 
v___f_159_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2___closed__0));
v___x_2111__overap_160_ = lean_panic_fn_borrowed(v___f_159_, v_msg_153_);
lean_inc(v___y_157_);
lean_inc_ref(v___y_156_);
lean_inc(v___y_155_);
lean_inc_ref(v___y_154_);
v___x_161_ = lean_apply_5(v___x_2111__overap_160_, v___y_154_, v___y_155_, v___y_156_, v___y_157_, lean_box(0));
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2___boxed(lean_object* v_msg_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2(v_msg_162_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
lean_dec(v___y_164_);
lean_dec_ref(v___y_163_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___redArg(lean_object* v_mvarId_169_, lean_object* v___y_170_){
_start:
{
lean_object* v___x_172_; lean_object* v_mctx_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_172_ = lean_st_ref_get(v___y_170_);
v_mctx_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc_ref(v_mctx_173_);
lean_dec(v___x_172_);
v___x_174_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_173_, v_mvarId_169_);
lean_dec_ref(v_mctx_173_);
v___x_175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___redArg___boxed(lean_object* v_mvarId_176_, lean_object* v___y_177_, lean_object* v___y_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___redArg(v_mvarId_176_, v___y_177_);
lean_dec(v___y_177_);
lean_dec(v_mvarId_176_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___lam__0(lean_object* v_coe_180_, lean_object* v_arg_181_, lean_object* v_x_182_, lean_object* v_f_x27_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_189_ = l_Lean_Expr_app___override(v_coe_180_, v_f_x27_183_);
v___x_190_ = l_Lean_Expr_app___override(v___x_189_, v_arg_181_);
v___x_191_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___lam__0___boxed(lean_object* v_coe_192_, lean_object* v_arg_193_, lean_object* v_x_194_, lean_object* v_f_x27_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___lam__0(v_coe_192_, v_arg_193_, v_x_194_, v_f_x27_195_, v___y_196_, v___y_197_, v___y_198_, v___y_199_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
lean_dec_ref(v_x_194_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___lam__0(lean_object* v_k_202_, lean_object* v_b_203_, lean_object* v_c_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_){
_start:
{
lean_object* v___x_210_; 
lean_inc(v___y_208_);
lean_inc_ref(v___y_207_);
lean_inc(v___y_206_);
lean_inc_ref(v___y_205_);
v___x_210_ = lean_apply_7(v_k_202_, v_b_203_, v_c_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_, lean_box(0));
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v_k_211_, lean_object* v_b_212_, lean_object* v_c_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___lam__0(v_k_211_, v_b_212_, v_c_213_, v___y_214_, v___y_215_, v___y_216_, v___y_217_);
lean_dec(v___y_217_);
lean_dec_ref(v___y_216_);
lean_dec(v___y_215_);
lean_dec_ref(v___y_214_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg(lean_object* v_e_220_, lean_object* v_k_221_, uint8_t v_cleanupAnnotations_222_, uint8_t v_preserveNondepLet_223_, uint8_t v_nondepLetOnly_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_){
_start:
{
lean_object* v___f_230_; uint8_t v___x_231_; uint8_t v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___f_230_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_230_, 0, v_k_221_);
v___x_231_ = 0;
v___x_232_ = 1;
v___x_233_ = lean_box(0);
v___x_234_ = l___private_Lean_Meta_Basic_0__Lean_Meta_lambdaTelescopeImp(lean_box(0), v_e_220_, v___x_231_, v___x_232_, v_preserveNondepLet_223_, v_nondepLetOnly_224_, v___x_233_, v___f_230_, v_cleanupAnnotations_222_, v___y_225_, v___y_226_, v___y_227_, v___y_228_);
if (lean_obj_tag(v___x_234_) == 0)
{
lean_object* v_a_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_242_; 
v_a_235_ = lean_ctor_get(v___x_234_, 0);
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_242_ == 0)
{
v___x_237_ = v___x_234_;
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_a_235_);
lean_dec(v___x_234_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_240_; 
if (v_isShared_238_ == 0)
{
v___x_240_ = v___x_237_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v_a_235_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
}
else
{
lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_250_; 
v_a_243_ = lean_ctor_get(v___x_234_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_250_ == 0)
{
v___x_245_ = v___x_234_;
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_234_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_248_; 
if (v_isShared_246_ == 0)
{
v___x_248_ = v___x_245_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_a_243_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg___boxed(lean_object* v_e_251_, lean_object* v_k_252_, lean_object* v_cleanupAnnotations_253_, lean_object* v_preserveNondepLet_254_, lean_object* v_nondepLetOnly_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_261_; uint8_t v_preserveNondepLet_boxed_262_; uint8_t v_nondepLetOnly_boxed_263_; lean_object* v_res_264_; 
v_cleanupAnnotations_boxed_261_ = lean_unbox(v_cleanupAnnotations_253_);
v_preserveNondepLet_boxed_262_ = lean_unbox(v_preserveNondepLet_254_);
v_nondepLetOnly_boxed_263_ = lean_unbox(v_nondepLetOnly_255_);
v_res_264_ = lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg(v_e_251_, v_k_252_, v_cleanupAnnotations_boxed_261_, v_preserveNondepLet_boxed_262_, v_nondepLetOnly_boxed_263_, v___y_256_, v___y_257_, v___y_258_, v___y_259_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec(v___y_257_);
lean_dec_ref(v___y_256_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___lam__0(lean_object* v_k_265_, uint8_t v_usedLetOnly_266_, lean_object* v_xs_267_, lean_object* v_b_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v___x_274_; 
lean_inc(v___y_272_);
lean_inc_ref(v___y_271_);
lean_inc(v___y_270_);
lean_inc_ref(v___y_269_);
lean_inc_ref(v_xs_267_);
v___x_274_ = lean_apply_7(v_k_265_, v_xs_267_, v_b_268_, v___y_269_, v___y_270_, v___y_271_, v___y_272_, lean_box(0));
if (lean_obj_tag(v___x_274_) == 0)
{
lean_object* v_a_275_; uint8_t v___x_276_; uint8_t v___x_277_; lean_object* v___x_278_; 
v_a_275_ = lean_ctor_get(v___x_274_, 0);
lean_inc(v_a_275_);
lean_dec_ref_known(v___x_274_, 1);
v___x_276_ = 0;
v___x_277_ = 1;
v___x_278_ = l_Lean_Meta_mkLetFVars(v_xs_267_, v_a_275_, v_usedLetOnly_266_, v___x_276_, v___x_277_, v___y_269_, v___y_270_, v___y_271_, v___y_272_);
lean_dec_ref(v_xs_267_);
return v___x_278_;
}
else
{
lean_dec_ref(v_xs_267_);
return v___x_274_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___lam__0___boxed(lean_object* v_k_279_, lean_object* v_usedLetOnly_280_, lean_object* v_xs_281_, lean_object* v_b_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_){
_start:
{
uint8_t v_usedLetOnly_boxed_288_; lean_object* v_res_289_; 
v_usedLetOnly_boxed_288_ = lean_unbox(v_usedLetOnly_280_);
v_res_289_ = lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___lam__0(v_k_279_, v_usedLetOnly_boxed_288_, v_xs_281_, v_b_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0(lean_object* v_e_290_, lean_object* v_k_291_, uint8_t v_cleanupAnnotations_292_, uint8_t v_preserveNondepLet_293_, uint8_t v_nondepLetOnly_294_, uint8_t v_usedLetOnly_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v___x_301_; lean_object* v___f_302_; lean_object* v___x_303_; 
v___x_301_ = lean_box(v_usedLetOnly_295_);
v___f_302_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___lam__0___boxed), 9, 2);
lean_closure_set(v___f_302_, 0, v_k_291_);
lean_closure_set(v___f_302_, 1, v___x_301_);
v___x_303_ = lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg(v_e_290_, v___f_302_, v_cleanupAnnotations_292_, v_preserveNondepLet_293_, v_nondepLetOnly_294_, v___y_296_, v___y_297_, v___y_298_, v___y_299_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0___boxed(lean_object* v_e_304_, lean_object* v_k_305_, lean_object* v_cleanupAnnotations_306_, lean_object* v_preserveNondepLet_307_, lean_object* v_nondepLetOnly_308_, lean_object* v_usedLetOnly_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_315_; uint8_t v_preserveNondepLet_boxed_316_; uint8_t v_nondepLetOnly_boxed_317_; uint8_t v_usedLetOnly_boxed_318_; lean_object* v_res_319_; 
v_cleanupAnnotations_boxed_315_ = lean_unbox(v_cleanupAnnotations_306_);
v_preserveNondepLet_boxed_316_ = lean_unbox(v_preserveNondepLet_307_);
v_nondepLetOnly_boxed_317_ = lean_unbox(v_nondepLetOnly_308_);
v_usedLetOnly_boxed_318_ = lean_unbox(v_usedLetOnly_309_);
v_res_319_ = lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0(v_e_304_, v_k_305_, v_cleanupAnnotations_boxed_315_, v_preserveNondepLet_boxed_316_, v_nondepLetOnly_boxed_317_, v_usedLetOnly_boxed_318_, v___y_310_, v___y_311_, v___y_312_, v___y_313_);
lean_dec(v___y_313_);
lean_dec_ref(v___y_312_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
return v_res_319_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___redArg(lean_object* v_k_320_, lean_object* v_t_321_){
_start:
{
if (lean_obj_tag(v_t_321_) == 0)
{
lean_object* v_k_322_; lean_object* v_l_323_; lean_object* v_r_324_; uint8_t v___x_325_; 
v_k_322_ = lean_ctor_get(v_t_321_, 1);
v_l_323_ = lean_ctor_get(v_t_321_, 3);
v_r_324_ = lean_ctor_get(v_t_321_, 4);
v___x_325_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_320_, v_k_322_);
switch(v___x_325_)
{
case 0:
{
v_t_321_ = v_l_323_;
goto _start;
}
case 1:
{
uint8_t v___x_327_; 
v___x_327_ = 1;
return v___x_327_;
}
default: 
{
v_t_321_ = v_r_324_;
goto _start;
}
}
}
else
{
uint8_t v___x_329_; 
v___x_329_ = 0;
return v___x_329_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___redArg___boxed(lean_object* v_k_330_, lean_object* v_t_331_){
_start:
{
uint8_t v_res_332_; lean_object* v_r_333_; 
v_res_332_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___redArg(v_k_330_, v_t_331_);
lean_dec(v_t_331_);
lean_dec(v_k_330_);
v_r_333_ = lean_box(v_res_332_);
return v_r_333_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__3(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__2));
v___x_338_ = lean_unsigned_to_nat(22u);
v___x_339_ = lean_unsigned_to_nat(391u);
v___x_340_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__1));
v___x_341_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__0));
v___x_342_ = l_mkPanicMessageWithDecl(v___x_341_, v___x_340_, v___x_339_, v___x_338_, v___x_337_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1(lean_object* v_pred_343_, lean_object* v_e_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_, lean_object* v_a_348_){
_start:
{
switch(lean_obj_tag(v_e_344_))
{
case 0:
{
lean_object* v___x_350_; lean_object* v___x_351_; 
lean_dec_ref_known(v_e_344_, 1);
lean_dec_ref(v_pred_343_);
v___x_350_ = lean_obj_once(&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__3, &lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__3_once, _init_lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___closed__3);
v___x_351_ = lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2(v___x_350_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
return v___x_351_;
}
case 1:
{
lean_object* v_fvarId_352_; lean_object* v___x_353_; 
v_fvarId_352_ = lean_ctor_get(v_e_344_, 0);
lean_inc(v_fvarId_352_);
v___x_353_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_352_, v_a_345_, v_a_347_, v_a_348_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_398_; 
v_a_354_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_398_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_398_ == 0)
{
v___x_356_ = v___x_353_;
v_isShared_357_ = v_isSharedCheck_398_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_353_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_398_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
if (lean_obj_tag(v_a_354_) == 1)
{
lean_object* v_value_358_; uint8_t v_nondep_359_; lean_object* v___y_361_; uint8_t v_trackZetaDelta_362_; lean_object* v___y_363_; lean_object* v___y_364_; lean_object* v___y_365_; lean_object* v___y_378_; lean_object* v___y_379_; lean_object* v___y_380_; lean_object* v___y_381_; 
v_value_358_ = lean_ctor_get(v_a_354_, 4);
lean_inc_ref(v_value_358_);
v_nondep_359_ = lean_ctor_get_uint8(v_a_354_, sizeof(void*)*5);
if (v_nondep_359_ == 0)
{
uint8_t v___x_383_; 
v___x_383_ = l_Lean_LocalDecl_isImplementationDetail(v_a_354_);
lean_dec_ref_known(v_a_354_, 5);
if (v___x_383_ == 0)
{
lean_object* v___x_384_; uint8_t v_zetaDelta_385_; 
v___x_384_ = l_Lean_Meta_Context_config(v_a_345_);
v_zetaDelta_385_ = lean_ctor_get_uint8(v___x_384_, 16);
lean_dec_ref(v___x_384_);
if (v_zetaDelta_385_ == 0)
{
uint8_t v_trackZetaDelta_386_; lean_object* v_zetaDeltaSet_387_; uint8_t v___x_388_; 
v_trackZetaDelta_386_ = lean_ctor_get_uint8(v_a_345_, sizeof(void*)*7);
v_zetaDeltaSet_387_ = lean_ctor_get(v_a_345_, 1);
v___x_388_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___redArg(v_fvarId_352_, v_zetaDeltaSet_387_);
if (v___x_388_ == 0)
{
lean_object* v___x_390_; 
lean_dec_ref(v_value_358_);
lean_dec_ref(v_pred_343_);
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 0, v_e_344_);
v___x_390_ = v___x_356_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v_e_344_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
}
}
else
{
lean_inc(v_fvarId_352_);
lean_del_object(v___x_356_);
lean_dec_ref_known(v_e_344_, 1);
v___y_361_ = v_a_345_;
v_trackZetaDelta_362_ = v_trackZetaDelta_386_;
v___y_363_ = v_a_346_;
v___y_364_ = v_a_347_;
v___y_365_ = v_a_348_;
goto v___jp_360_;
}
}
else
{
lean_inc(v_fvarId_352_);
lean_del_object(v___x_356_);
lean_dec_ref_known(v_e_344_, 1);
v___y_378_ = v_a_345_;
v___y_379_ = v_a_346_;
v___y_380_ = v_a_347_;
v___y_381_ = v_a_348_;
goto v___jp_377_;
}
}
else
{
lean_inc(v_fvarId_352_);
lean_del_object(v___x_356_);
lean_dec_ref_known(v_e_344_, 1);
v___y_378_ = v_a_345_;
v___y_379_ = v_a_346_;
v___y_380_ = v_a_347_;
v___y_381_ = v_a_348_;
goto v___jp_377_;
}
}
else
{
lean_object* v___x_393_; 
lean_dec_ref(v_value_358_);
lean_dec_ref_known(v_a_354_, 5);
lean_dec_ref(v_pred_343_);
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 0, v_e_344_);
v___x_393_ = v___x_356_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_e_344_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
v___jp_360_:
{
if (v_trackZetaDelta_362_ == 0)
{
lean_dec(v_fvarId_352_);
v_e_344_ = v_value_358_;
v_a_345_ = v___y_361_;
v_a_346_ = v___y_363_;
v_a_347_ = v___y_364_;
v_a_348_ = v___y_365_;
goto _start;
}
else
{
lean_object* v___x_367_; 
v___x_367_ = l_Lean_Meta_addZetaDeltaFVarId___redArg(v_fvarId_352_, v___y_363_);
if (lean_obj_tag(v___x_367_) == 0)
{
lean_dec_ref_known(v___x_367_, 1);
v_e_344_ = v_value_358_;
v_a_345_ = v___y_361_;
v_a_346_ = v___y_363_;
v_a_347_ = v___y_364_;
v_a_348_ = v___y_365_;
goto _start;
}
else
{
lean_object* v_a_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_376_; 
lean_dec_ref(v_value_358_);
lean_dec_ref(v_pred_343_);
v_a_369_ = lean_ctor_get(v___x_367_, 0);
v_isSharedCheck_376_ = !lean_is_exclusive(v___x_367_);
if (v_isSharedCheck_376_ == 0)
{
v___x_371_ = v___x_367_;
v_isShared_372_ = v_isSharedCheck_376_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_a_369_);
lean_dec(v___x_367_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_376_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_374_; 
if (v_isShared_372_ == 0)
{
v___x_374_ = v___x_371_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v_a_369_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
}
}
}
v___jp_377_:
{
uint8_t v_trackZetaDelta_382_; 
v_trackZetaDelta_382_ = lean_ctor_get_uint8(v___y_378_, sizeof(void*)*7);
v___y_361_ = v___y_378_;
v_trackZetaDelta_362_ = v_trackZetaDelta_382_;
v___y_363_ = v___y_379_;
v___y_364_ = v___y_380_;
v___y_365_ = v___y_381_;
goto v___jp_360_;
}
}
else
{
lean_object* v___x_396_; 
lean_dec(v_a_354_);
lean_dec_ref(v_pred_343_);
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 0, v_e_344_);
v___x_396_ = v___x_356_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v_e_344_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
}
else
{
lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_406_; 
lean_dec_ref_known(v_e_344_, 1);
lean_dec_ref(v_pred_343_);
v_a_399_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_406_ == 0)
{
v___x_401_ = v___x_353_;
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_353_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_406_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_404_; 
if (v_isShared_402_ == 0)
{
v___x_404_ = v___x_401_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_a_399_);
v___x_404_ = v_reuseFailAlloc_405_;
goto v_reusejp_403_;
}
v_reusejp_403_:
{
return v___x_404_;
}
}
}
}
case 2:
{
lean_object* v_mvarId_407_; lean_object* v___x_408_; 
v_mvarId_407_ = lean_ctor_get(v_e_344_, 0);
v___x_408_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___redArg(v_mvarId_407_, v_a_346_);
if (lean_obj_tag(v___x_408_) == 0)
{
lean_object* v_a_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_418_; 
v_a_409_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_418_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_418_ == 0)
{
v___x_411_ = v___x_408_;
v_isShared_412_ = v_isSharedCheck_418_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_a_409_);
lean_dec(v___x_408_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_418_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
if (lean_obj_tag(v_a_409_) == 0)
{
lean_object* v___x_414_; 
lean_dec_ref(v_pred_343_);
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 0, v_e_344_);
v___x_414_ = v___x_411_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v_e_344_);
v___x_414_ = v_reuseFailAlloc_415_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
return v___x_414_;
}
}
else
{
lean_object* v_val_416_; 
lean_del_object(v___x_411_);
lean_dec_ref_known(v_e_344_, 1);
v_val_416_ = lean_ctor_get(v_a_409_, 0);
lean_inc(v_val_416_);
lean_dec_ref_known(v_a_409_, 1);
v_e_344_ = v_val_416_;
goto _start;
}
}
}
else
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
lean_dec_ref_known(v_e_344_, 1);
lean_dec_ref(v_pred_343_);
v_a_419_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_408_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_408_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_a_419_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
}
}
case 3:
{
lean_object* v___x_427_; 
lean_dec_ref(v_pred_343_);
v___x_427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_427_, 0, v_e_344_);
return v___x_427_;
}
case 6:
{
lean_object* v___x_428_; 
lean_dec_ref(v_pred_343_);
v___x_428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_428_, 0, v_e_344_);
return v___x_428_;
}
case 7:
{
lean_object* v___x_429_; 
lean_dec_ref(v_pred_343_);
v___x_429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_429_, 0, v_e_344_);
return v___x_429_;
}
case 9:
{
lean_object* v___x_430_; 
lean_dec_ref(v_pred_343_);
v___x_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_430_, 0, v_e_344_);
return v___x_430_;
}
case 10:
{
lean_object* v_expr_431_; 
v_expr_431_ = lean_ctor_get(v_e_344_, 1);
lean_inc_ref(v_expr_431_);
lean_dec_ref_known(v_e_344_, 2);
v_e_344_ = v_expr_431_;
goto _start;
}
default: 
{
lean_object* v___x_433_; 
v___x_433_ = l_Lean_Meta_whnfCore(v_e_344_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
if (lean_obj_tag(v___x_433_) == 0)
{
lean_object* v_a_434_; lean_object* v___x_435_; 
v_a_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_434_);
lean_dec_ref_known(v___x_433_, 1);
v___x_435_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isMorApp_x3f___redArg(v_a_434_, v_a_348_);
if (lean_obj_tag(v___x_435_) == 0)
{
lean_object* v_a_436_; 
v_a_436_ = lean_ctor_get(v___x_435_, 0);
lean_inc(v_a_436_);
lean_dec_ref_known(v___x_435_, 1);
if (lean_obj_tag(v_a_436_) == 1)
{
lean_object* v_val_437_; lean_object* v_coe_438_; lean_object* v_fn_439_; lean_object* v_arg_440_; lean_object* v___x_441_; 
lean_dec(v_a_434_);
v_val_437_ = lean_ctor_get(v_a_436_, 0);
lean_inc(v_val_437_);
lean_dec_ref_known(v_a_436_, 1);
v_coe_438_ = lean_ctor_get(v_val_437_, 0);
lean_inc_ref(v_coe_438_);
v_fn_439_ = lean_ctor_get(v_val_437_, 1);
lean_inc_ref(v_fn_439_);
v_arg_440_ = lean_ctor_get(v_val_437_, 2);
lean_inc_ref(v_arg_440_);
lean_dec(v_val_437_);
v___x_441_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(v_fn_439_, v_pred_343_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
if (lean_obj_tag(v___x_441_) == 0)
{
lean_object* v_a_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_456_; 
v_a_442_ = lean_ctor_get(v___x_441_, 0);
v_isSharedCheck_456_ = !lean_is_exclusive(v___x_441_);
if (v_isSharedCheck_456_ == 0)
{
v___x_444_ = v___x_441_;
v_isShared_445_ = v_isSharedCheck_456_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_a_442_);
lean_dec(v___x_441_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_456_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___x_446_; uint8_t v_zeta_447_; 
v___x_446_ = l_Lean_Meta_Context_config(v_a_345_);
v_zeta_447_ = lean_ctor_get_uint8(v___x_446_, 15);
lean_dec_ref(v___x_446_);
if (v_zeta_447_ == 0)
{
lean_object* v___f_448_; uint8_t v___x_449_; lean_object* v___x_450_; 
lean_del_object(v___x_444_);
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___lam__0___boxed), 9, 2);
lean_closure_set(v___f_448_, 0, v_coe_438_);
lean_closure_set(v___f_448_, 1, v_arg_440_);
v___x_449_ = 1;
v___x_450_ = lp_mathlib_Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0(v_a_442_, v___f_448_, v_zeta_447_, v___x_449_, v_zeta_447_, v___x_449_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
return v___x_450_;
}
else
{
lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_454_; 
v___x_451_ = l_Lean_Expr_app___override(v_coe_438_, v_a_442_);
v___x_452_ = l_Lean_Expr_app___override(v___x_451_, v_arg_440_);
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 0, v___x_452_);
v___x_454_ = v___x_444_;
goto v_reusejp_453_;
}
else
{
lean_object* v_reuseFailAlloc_455_; 
v_reuseFailAlloc_455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_455_, 0, v___x_452_);
v___x_454_ = v_reuseFailAlloc_455_;
goto v_reusejp_453_;
}
v_reusejp_453_:
{
return v___x_454_;
}
}
}
}
else
{
lean_dec_ref(v_arg_440_);
lean_dec_ref(v_coe_438_);
return v___x_441_;
}
}
else
{
lean_object* v___x_457_; 
lean_dec(v_a_436_);
lean_inc_ref(v_pred_343_);
lean_inc(v_a_348_);
lean_inc_ref(v_a_347_);
lean_inc(v_a_346_);
lean_inc_ref(v_a_345_);
lean_inc(v_a_434_);
v___x_457_ = lean_apply_6(v_pred_343_, v_a_434_, v_a_345_, v_a_346_, v_a_347_, v_a_348_, lean_box(0));
if (lean_obj_tag(v___x_457_) == 0)
{
lean_object* v_a_458_; lean_object* v___x_460_; uint8_t v_isShared_461_; uint8_t v_isSharedCheck_486_; 
v_a_458_ = lean_ctor_get(v___x_457_, 0);
v_isSharedCheck_486_ = !lean_is_exclusive(v___x_457_);
if (v_isSharedCheck_486_ == 0)
{
v___x_460_ = v___x_457_;
v_isShared_461_ = v_isSharedCheck_486_;
goto v_resetjp_459_;
}
else
{
lean_inc(v_a_458_);
lean_dec(v___x_457_);
v___x_460_ = lean_box(0);
v_isShared_461_ = v_isSharedCheck_486_;
goto v_resetjp_459_;
}
v_resetjp_459_:
{
uint8_t v___x_462_; 
v___x_462_ = lean_unbox(v_a_458_);
lean_dec(v_a_458_);
if (v___x_462_ == 0)
{
lean_object* v___x_464_; 
lean_dec_ref(v_pred_343_);
if (v_isShared_461_ == 0)
{
lean_ctor_set(v___x_460_, 0, v_a_434_);
v___x_464_ = v___x_460_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v_a_434_);
v___x_464_ = v_reuseFailAlloc_465_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
return v___x_464_;
}
}
else
{
uint8_t v___x_466_; lean_object* v___x_467_; 
lean_del_object(v___x_460_);
v___x_466_ = 0;
lean_inc(v_a_434_);
v___x_467_ = l_Lean_Meta_unfoldDefinition_x3f(v_a_434_, v___x_466_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
if (lean_obj_tag(v___x_467_) == 0)
{
lean_object* v_a_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_477_; 
v_a_468_ = lean_ctor_get(v___x_467_, 0);
v_isSharedCheck_477_ = !lean_is_exclusive(v___x_467_);
if (v_isSharedCheck_477_ == 0)
{
v___x_470_ = v___x_467_;
v_isShared_471_ = v_isSharedCheck_477_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_a_468_);
lean_dec(v___x_467_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_477_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
if (lean_obj_tag(v_a_468_) == 0)
{
lean_object* v___x_473_; 
lean_dec_ref(v_pred_343_);
if (v_isShared_471_ == 0)
{
lean_ctor_set(v___x_470_, 0, v_a_434_);
v___x_473_ = v___x_470_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v_a_434_);
v___x_473_ = v_reuseFailAlloc_474_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
return v___x_473_;
}
}
else
{
lean_object* v_val_475_; lean_object* v___x_476_; 
lean_del_object(v___x_470_);
lean_dec(v_a_434_);
v_val_475_ = lean_ctor_get(v_a_468_, 0);
lean_inc(v_val_475_);
lean_dec_ref_known(v_a_468_, 1);
v___x_476_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(v_val_475_, v_pred_343_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
return v___x_476_;
}
}
}
else
{
lean_object* v_a_478_; lean_object* v___x_480_; uint8_t v_isShared_481_; uint8_t v_isSharedCheck_485_; 
lean_dec(v_a_434_);
lean_dec_ref(v_pred_343_);
v_a_478_ = lean_ctor_get(v___x_467_, 0);
v_isSharedCheck_485_ = !lean_is_exclusive(v___x_467_);
if (v_isSharedCheck_485_ == 0)
{
v___x_480_ = v___x_467_;
v_isShared_481_ = v_isSharedCheck_485_;
goto v_resetjp_479_;
}
else
{
lean_inc(v_a_478_);
lean_dec(v___x_467_);
v___x_480_ = lean_box(0);
v_isShared_481_ = v_isSharedCheck_485_;
goto v_resetjp_479_;
}
v_resetjp_479_:
{
lean_object* v___x_483_; 
if (v_isShared_481_ == 0)
{
v___x_483_ = v___x_480_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_484_; 
v_reuseFailAlloc_484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_484_, 0, v_a_478_);
v___x_483_ = v_reuseFailAlloc_484_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
return v___x_483_;
}
}
}
}
}
}
else
{
lean_object* v_a_487_; lean_object* v___x_489_; uint8_t v_isShared_490_; uint8_t v_isSharedCheck_494_; 
lean_dec(v_a_434_);
lean_dec_ref(v_pred_343_);
v_a_487_ = lean_ctor_get(v___x_457_, 0);
v_isSharedCheck_494_ = !lean_is_exclusive(v___x_457_);
if (v_isSharedCheck_494_ == 0)
{
v___x_489_ = v___x_457_;
v_isShared_490_ = v_isSharedCheck_494_;
goto v_resetjp_488_;
}
else
{
lean_inc(v_a_487_);
lean_dec(v___x_457_);
v___x_489_ = lean_box(0);
v_isShared_490_ = v_isSharedCheck_494_;
goto v_resetjp_488_;
}
v_resetjp_488_:
{
lean_object* v___x_492_; 
if (v_isShared_490_ == 0)
{
v___x_492_ = v___x_489_;
goto v_reusejp_491_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v_a_487_);
v___x_492_ = v_reuseFailAlloc_493_;
goto v_reusejp_491_;
}
v_reusejp_491_:
{
return v___x_492_;
}
}
}
}
}
else
{
lean_object* v_a_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_502_; 
lean_dec(v_a_434_);
lean_dec_ref(v_pred_343_);
v_a_495_ = lean_ctor_get(v___x_435_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v___x_435_);
if (v_isSharedCheck_502_ == 0)
{
v___x_497_ = v___x_435_;
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_a_495_);
lean_dec(v___x_435_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_502_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_500_; 
if (v_isShared_498_ == 0)
{
v___x_500_ = v___x_497_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_a_495_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
else
{
lean_dec_ref(v_pred_343_);
return v___x_433_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(lean_object* v_e_503_, lean_object* v_pred_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1(v_pred_504_, v_e_503_, v_a_505_, v_a_506_, v_a_507_, v_a_508_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred___boxed(lean_object* v_e_511_, lean_object* v_pred_512_, lean_object* v_a_513_, lean_object* v_a_514_, lean_object* v_a_515_, lean_object* v_a_516_, lean_object* v_a_517_){
_start:
{
lean_object* v_res_518_; 
v_res_518_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(v_e_511_, v_pred_512_, v_a_513_, v_a_514_, v_a_515_, v_a_516_);
lean_dec(v_a_516_);
lean_dec_ref(v_a_515_);
lean_dec(v_a_514_);
lean_dec_ref(v_a_513_);
return v_res_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1___boxed(lean_object* v_pred_519_, lean_object* v_e_520_, lean_object* v_a_521_, lean_object* v_a_522_, lean_object* v_a_523_, lean_object* v_a_524_, lean_object* v_a_525_){
_start:
{
lean_object* v_res_526_; 
v_res_526_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1(v_pred_519_, v_e_520_, v_a_521_, v_a_522_, v_a_523_, v_a_524_);
lean_dec(v_a_524_);
lean_dec_ref(v_a_523_);
lean_dec(v_a_522_);
lean_dec_ref(v_a_521_);
return v_res_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0(lean_object* v_00_u03b1_527_, lean_object* v_e_528_, lean_object* v_k_529_, uint8_t v_cleanupAnnotations_530_, uint8_t v_preserveNondepLet_531_, uint8_t v_nondepLetOnly_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___redArg(v_e_528_, v_k_529_, v_cleanupAnnotations_530_, v_preserveNondepLet_531_, v_nondepLetOnly_532_, v___y_533_, v___y_534_, v___y_535_, v___y_536_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0___boxed(lean_object* v_00_u03b1_539_, lean_object* v_e_540_, lean_object* v_k_541_, lean_object* v_cleanupAnnotations_542_, lean_object* v_preserveNondepLet_543_, lean_object* v_nondepLetOnly_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_550_; uint8_t v_preserveNondepLet_boxed_551_; uint8_t v_nondepLetOnly_boxed_552_; lean_object* v_res_553_; 
v_cleanupAnnotations_boxed_550_ = lean_unbox(v_cleanupAnnotations_542_);
v_preserveNondepLet_boxed_551_ = lean_unbox(v_preserveNondepLet_543_);
v_nondepLetOnly_boxed_552_ = lean_unbox(v_nondepLetOnly_544_);
v_res_553_ = lp_mathlib_Lean_Meta_letTelescope___at___00Lean_Meta_mapLetTelescope___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__0_spec__0(v_00_u03b1_539_, v_e_540_, v_k_541_, v_cleanupAnnotations_boxed_550_, v_preserveNondepLet_boxed_551_, v_nondepLetOnly_boxed_552_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
lean_dec(v___y_548_);
lean_dec_ref(v___y_547_);
lean_dec(v___y_546_);
lean_dec_ref(v___y_545_);
return v_res_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4(lean_object* v_mvarId_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___redArg(v_mvarId_554_, v___y_556_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4___boxed(lean_object* v_mvarId_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_){
_start:
{
lean_object* v_res_567_; 
v_res_567_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__4(v_mvarId_561_, v___y_562_, v___y_563_, v___y_564_, v___y_565_);
lean_dec(v___y_565_);
lean_dec_ref(v___y_564_);
lean_dec(v___y_563_);
lean_dec_ref(v___y_562_);
lean_dec(v_mvarId_561_);
return v_res_567_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3(lean_object* v_00_u03b2_568_, lean_object* v_k_569_, lean_object* v_t_570_){
_start:
{
uint8_t v___x_571_; 
v___x_571_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___redArg(v_k_569_, v_t_570_);
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3___boxed(lean_object* v_00_u03b2_572_, lean_object* v_k_573_, lean_object* v_t_574_){
_start:
{
uint8_t v_res_575_; lean_object* v_r_576_; 
v_res_575_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__3(v_00_u03b2_572_, v_k_573_, v_t_574_);
lean_dec(v_t_574_);
lean_dec(v_k_573_);
v_r_576_ = lean_box(v_res_575_);
return v_r_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___lam__0(lean_object* v_x_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
uint8_t v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; 
v___x_583_ = 0;
v___x_584_ = lean_box(v___x_583_);
v___x_585_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_585_, 0, v___x_584_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___lam__0___boxed(lean_object* v_x_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___lam__0(v_x_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_);
lean_dec(v___y_590_);
lean_dec_ref(v___y_589_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec_ref(v_x_586_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf(lean_object* v_e_594_, lean_object* v_a_595_, lean_object* v_a_596_, lean_object* v_a_597_, lean_object* v_a_598_){
_start:
{
lean_object* v___f_600_; lean_object* v___x_601_; 
v___f_600_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___closed__0));
v___x_601_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_whnfPred(v_e_594_, v___f_600_, v_a_595_, v_a_596_, v_a_597_, v_a_598_);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf___boxed(lean_object* v_e_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_, lean_object* v_a_606_, lean_object* v_a_607_){
_start:
{
lean_object* v_res_608_; 
v_res_608_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_whnf(v_e_602_, v_a_603_, v_a_604_, v_a_605_, v_a_606_);
lean_dec(v_a_606_);
lean_dec_ref(v_a_605_);
lean_dec(v_a_604_);
lean_dec_ref(v_a_603_);
return v_res_608_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__2(void){
_start:
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_612_ = lean_box(0);
v___x_613_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__1));
v___x_614_ = l_Lean_Expr_const___override(v___x_613_, v___x_612_);
return v___x_614_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__3(void){
_start:
{
lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_615_ = lean_box(0);
v___x_616_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__2, &lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__2_once, _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__2);
v___x_617_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_617_, 0, v___x_616_);
lean_ctor_set(v___x_617_, 1, v___x_615_);
return v___x_617_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default(void){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__3, &lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__3_once, _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default___closed__3);
return v___x_618_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg(void){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default;
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_app(lean_object* v_f_620_, lean_object* v_arg_621_){
_start:
{
lean_object* v_coe_622_; 
v_coe_622_ = lean_ctor_get(v_arg_621_, 1);
if (lean_obj_tag(v_coe_622_) == 0)
{
lean_object* v_expr_623_; lean_object* v___x_624_; 
v_expr_623_ = lean_ctor_get(v_arg_621_, 0);
lean_inc_ref(v_expr_623_);
lean_dec_ref(v_arg_621_);
v___x_624_ = l_Lean_Expr_app___override(v_f_620_, v_expr_623_);
return v___x_624_;
}
else
{
lean_object* v_expr_625_; lean_object* v_val_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
lean_inc_ref(v_coe_622_);
v_expr_625_ = lean_ctor_get(v_arg_621_, 0);
lean_inc_ref(v_expr_625_);
lean_dec_ref(v_arg_621_);
v_val_626_ = lean_ctor_get(v_coe_622_, 0);
lean_inc(v_val_626_);
lean_dec_ref_known(v_coe_622_, 1);
v___x_627_ = l_Lean_Expr_app___override(v_val_626_, v_f_620_);
v___x_628_ = l_Lean_Expr_app___override(v___x_627_, v_expr_625_);
return v___x_628_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___redArg(lean_object* v_msg_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_){
_start:
{
lean_object* v___f_635_; lean_object* v___x_1114__overap_636_; lean_object* v___x_637_; 
v___f_635_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Mathlib_Meta_FunProp_Mor_whnfPred_spec__1_spec__2___closed__0));
v___x_1114__overap_636_ = lean_panic_fn_borrowed(v___f_635_, v_msg_629_);
lean_inc(v___y_633_);
lean_inc_ref(v___y_632_);
lean_inc(v___y_631_);
lean_inc_ref(v___y_630_);
v___x_637_ = lean_apply_5(v___x_1114__overap_636_, v___y_630_, v___y_631_, v___y_632_, v___y_633_, lean_box(0));
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___redArg___boxed(lean_object* v_msg_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___redArg(v_msg_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_);
lean_dec(v___y_642_);
lean_dec_ref(v___y_641_);
lean_dec(v___y_640_);
lean_dec_ref(v___y_639_);
return v_res_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0(lean_object* v_00_u03b1_645_, lean_object* v_msg_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___redArg(v_msg_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___boxed(lean_object* v_00_u03b1_653_, lean_object* v_msg_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_){
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0(v_00_u03b1_653_, v_msg_654_, v___y_655_, v___y_656_, v___y_657_, v___y_658_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec(v___y_656_);
lean_dec_ref(v___y_655_);
return v_res_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__1(lean_object* v_msg_661_){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_662_ = lean_box(0);
v___x_663_ = lean_panic_fn_borrowed(v___x_662_, v_msg_661_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__2(lean_object* v_msg_664_){
_start:
{
lean_object* v___x_665_; lean_object* v___x_666_; 
v___x_665_ = l_Lean_instInhabitedStructureInfo_default;
v___x_666_ = lean_panic_fn_borrowed(v___x_665_, v_msg_664_);
return v___x_666_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__3(void){
_start:
{
lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; 
v___x_670_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__2));
v___x_671_ = lean_unsigned_to_nat(42u);
v___x_672_ = lean_unsigned_to_nat(137u);
v___x_673_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__1));
v___x_674_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__0));
v___x_675_ = l_mkPanicMessageWithDecl(v___x_674_, v___x_673_, v___x_672_, v___x_671_, v___x_670_);
return v___x_675_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7(void){
_start:
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; 
v___x_679_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__6));
v___x_680_ = lean_unsigned_to_nat(14u);
v___x_681_ = lean_unsigned_to_nat(22u);
v___x_682_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__5));
v___x_683_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__4));
v___x_684_ = l_mkPanicMessageWithDecl(v___x_683_, v___x_682_, v___x_681_, v___x_680_, v___x_679_);
return v___x_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg(lean_object* v_k_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_, lean_object* v_a_702_){
_start:
{
switch(lean_obj_tag(v_a_697_))
{
case 10:
{
lean_object* v_expr_704_; 
v_expr_704_ = lean_ctor_get(v_a_697_, 1);
lean_inc_ref(v_expr_704_);
lean_dec_ref_known(v_a_697_, 2);
v_a_697_ = v_expr_704_;
goto _start;
}
case 5:
{
lean_object* v_fn_706_; 
v_fn_706_ = lean_ctor_get(v_a_697_, 0);
lean_inc_ref(v_fn_706_);
switch(lean_obj_tag(v_fn_706_))
{
case 5:
{
lean_object* v_arg_707_; lean_object* v_fn_708_; lean_object* v_arg_709_; lean_object* v___x_710_; 
v_arg_707_ = lean_ctor_get(v_a_697_, 1);
lean_inc_ref(v_arg_707_);
lean_dec_ref_known(v_a_697_, 2);
v_fn_708_ = lean_ctor_get(v_fn_706_, 0);
v_arg_709_ = lean_ctor_get(v_fn_706_, 1);
v___x_710_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg(v_fn_708_, v_a_702_);
if (lean_obj_tag(v___x_710_) == 0)
{
lean_object* v_a_711_; uint8_t v___x_712_; 
v_a_711_ = lean_ctor_get(v___x_710_, 0);
lean_inc(v_a_711_);
lean_dec_ref_known(v___x_710_, 1);
v___x_712_ = lean_unbox(v_a_711_);
lean_dec(v_a_711_);
if (v___x_712_ == 0)
{
lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; 
v___x_713_ = lean_box(0);
v___x_714_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_714_, 0, v_arg_707_);
lean_ctor_set(v___x_714_, 1, v___x_713_);
v___x_715_ = lean_array_push(v_a_698_, v___x_714_);
v_a_697_ = v_fn_706_;
v_a_698_ = v___x_715_;
goto _start;
}
else
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
lean_inc_ref(v_arg_709_);
lean_inc_ref(v_fn_708_);
lean_dec_ref_known(v_fn_706_, 2);
v___x_717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_717_, 0, v_fn_708_);
v___x_718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_718_, 0, v_arg_707_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
v___x_719_ = lean_array_push(v_a_698_, v___x_718_);
v_a_697_ = v_arg_709_;
v_a_698_ = v___x_719_;
goto _start;
}
}
else
{
lean_object* v_a_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_728_; 
lean_dec_ref(v_arg_707_);
lean_dec_ref_known(v_fn_706_, 2);
lean_dec_ref(v_a_698_);
lean_dec_ref(v_k_696_);
v_a_721_ = lean_ctor_get(v___x_710_, 0);
v_isSharedCheck_728_ = !lean_is_exclusive(v___x_710_);
if (v_isSharedCheck_728_ == 0)
{
v___x_723_ = v___x_710_;
v_isShared_724_ = v_isSharedCheck_728_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_a_721_);
lean_dec(v___x_710_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_728_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v___x_726_; 
if (v_isShared_724_ == 0)
{
v___x_726_ = v___x_723_;
goto v_reusejp_725_;
}
else
{
lean_object* v_reuseFailAlloc_727_; 
v_reuseFailAlloc_727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_727_, 0, v_a_721_);
v___x_726_ = v_reuseFailAlloc_727_;
goto v_reusejp_725_;
}
v_reusejp_725_:
{
return v___x_726_;
}
}
}
}
case 11:
{
lean_object* v_arg_729_; lean_object* v_typeName_730_; lean_object* v_idx_731_; lean_object* v_struct_732_; lean_object* v___x_733_; lean_object* v___y_735_; lean_object* v_env_753_; lean_object* v___x_754_; lean_object* v___y_756_; lean_object* v___x_763_; 
v_arg_729_ = lean_ctor_get(v_a_697_, 1);
lean_inc_ref(v_arg_729_);
lean_dec_ref_known(v_a_697_, 2);
v_typeName_730_ = lean_ctor_get(v_fn_706_, 0);
lean_inc_n(v_typeName_730_, 2);
v_idx_731_ = lean_ctor_get(v_fn_706_, 1);
lean_inc(v_idx_731_);
v_struct_732_ = lean_ctor_get(v_fn_706_, 2);
lean_inc_ref(v_struct_732_);
lean_dec_ref_known(v_fn_706_, 3);
v___x_733_ = lean_st_ref_get(v_a_702_);
v_env_753_ = lean_ctor_get(v___x_733_, 0);
lean_inc_ref_n(v_env_753_, 2);
lean_dec(v___x_733_);
v___x_754_ = lean_box(0);
v___x_763_ = l_Lean_getStructureInfo_x3f(v_env_753_, v_typeName_730_);
if (lean_obj_tag(v___x_763_) == 0)
{
lean_object* v___x_764_; lean_object* v___x_765_; 
v___x_764_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7);
v___x_765_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__2(v___x_764_);
v___y_756_ = v___x_765_;
goto v___jp_755_;
}
else
{
lean_object* v_val_766_; 
v_val_766_ = lean_ctor_get(v___x_763_, 0);
lean_inc(v_val_766_);
lean_dec_ref_known(v___x_763_, 1);
v___y_756_ = v_val_766_;
goto v___jp_755_;
}
v___jp_734_:
{
lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_736_ = lean_unsigned_to_nat(1u);
v___x_737_ = lean_mk_empty_array_with_capacity(v___x_736_);
v___x_738_ = lean_array_push(v___x_737_, v_struct_732_);
v___x_739_ = l_Lean_Meta_mkAppM(v___y_735_, v___x_738_, v_a_699_, v_a_700_, v_a_701_, v_a_702_);
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
lean_inc(v_a_740_);
lean_dec_ref_known(v___x_739_, 1);
if (lean_obj_tag(v_a_740_) == 5)
{
lean_object* v___x_741_; 
v___x_741_ = l_Lean_Expr_app___override(v_a_740_, v_arg_729_);
v_a_697_ = v___x_741_;
goto _start;
}
else
{
lean_object* v___x_743_; lean_object* v___x_744_; 
lean_dec(v_a_740_);
lean_dec_ref(v_arg_729_);
lean_dec_ref(v_a_698_);
lean_dec_ref(v_k_696_);
v___x_743_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__3, &lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__3);
v___x_744_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__0___redArg(v___x_743_, v_a_699_, v_a_700_, v_a_701_, v_a_702_);
return v___x_744_;
}
}
else
{
lean_object* v_a_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_752_; 
lean_dec_ref(v_arg_729_);
lean_dec_ref(v_a_698_);
lean_dec_ref(v_k_696_);
v_a_745_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_752_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_752_ == 0)
{
v___x_747_ = v___x_739_;
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_a_745_);
lean_dec(v___x_739_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_752_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
lean_object* v___x_750_; 
if (v_isShared_748_ == 0)
{
v___x_750_ = v___x_747_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v_a_745_);
v___x_750_ = v_reuseFailAlloc_751_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
return v___x_750_;
}
}
}
}
v___jp_755_:
{
lean_object* v_fieldNames_757_; lean_object* v___x_758_; lean_object* v___x_759_; 
v_fieldNames_757_ = lean_ctor_get(v___y_756_, 1);
lean_inc_ref(v_fieldNames_757_);
lean_dec_ref(v___y_756_);
v___x_758_ = lean_array_get(v___x_754_, v_fieldNames_757_, v_idx_731_);
lean_dec(v_idx_731_);
lean_dec_ref(v_fieldNames_757_);
v___x_759_ = l_Lean_getProjFnForField_x3f(v_env_753_, v_typeName_730_, v___x_758_);
if (lean_obj_tag(v___x_759_) == 0)
{
lean_object* v___x_760_; lean_object* v___x_761_; 
v___x_760_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7, &lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__7);
v___x_761_ = lp_mathlib_panic___at___00__private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go_spec__1(v___x_760_);
v___y_735_ = v___x_761_;
goto v___jp_734_;
}
else
{
lean_object* v_val_762_; 
v_val_762_ = lean_ctor_get(v___x_759_, 0);
lean_inc(v_val_762_);
lean_dec_ref_known(v___x_759_, 1);
v___y_735_ = v_val_762_;
goto v___jp_734_;
}
}
}
default: 
{
lean_object* v_arg_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; 
v_arg_767_ = lean_ctor_get(v_a_697_, 1);
lean_inc_ref(v_arg_767_);
lean_dec_ref_known(v_a_697_, 2);
v___x_768_ = lean_box(0);
v___x_769_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_769_, 0, v_arg_767_);
lean_ctor_set(v___x_769_, 1, v___x_768_);
v___x_770_ = lean_array_push(v_a_698_, v___x_769_);
v_a_697_ = v_fn_706_;
v_a_698_ = v___x_770_;
goto _start;
}
}
}
case 7:
{
lean_object* v_binderName_772_; lean_object* v_binderType_773_; lean_object* v_body_774_; uint8_t v_binderInfo_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; 
lean_dec_ref(v_a_698_);
v_binderName_772_ = lean_ctor_get(v_a_697_, 0);
lean_inc(v_binderName_772_);
v_binderType_773_ = lean_ctor_get(v_a_697_, 1);
lean_inc_ref(v_binderType_773_);
v_body_774_ = lean_ctor_get(v_a_697_, 2);
lean_inc_ref(v_body_774_);
v_binderInfo_775_ = lean_ctor_get_uint8(v_a_697_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_a_697_, 3);
v___x_776_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__12));
v___x_777_ = l_Lean_Expr_lam___override(v_binderName_772_, v_binderType_773_, v_body_774_, v_binderInfo_775_);
v___x_778_ = lean_unsigned_to_nat(1u);
v___x_779_ = lean_mk_empty_array_with_capacity(v___x_778_);
v___x_780_ = lean_array_push(v___x_779_, v___x_777_);
v___x_781_ = l_Lean_Meta_mkAppM(v___x_776_, v___x_780_, v_a_699_, v_a_700_, v_a_701_, v_a_702_);
if (lean_obj_tag(v___x_781_) == 0)
{
lean_object* v_a_782_; lean_object* v___x_783_; 
v_a_782_ = lean_ctor_get(v___x_781_, 0);
lean_inc(v_a_782_);
lean_dec_ref_known(v___x_781_, 1);
v___x_783_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__13));
v_a_697_ = v_a_782_;
v_a_698_ = v___x_783_;
goto _start;
}
else
{
lean_object* v_a_785_; lean_object* v___x_787_; uint8_t v_isShared_788_; uint8_t v_isSharedCheck_792_; 
lean_dec_ref(v_k_696_);
v_a_785_ = lean_ctor_get(v___x_781_, 0);
v_isSharedCheck_792_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_792_ == 0)
{
v___x_787_ = v___x_781_;
v_isShared_788_ = v_isSharedCheck_792_;
goto v_resetjp_786_;
}
else
{
lean_inc(v_a_785_);
lean_dec(v___x_781_);
v___x_787_ = lean_box(0);
v_isShared_788_ = v_isSharedCheck_792_;
goto v_resetjp_786_;
}
v_resetjp_786_:
{
lean_object* v___x_790_; 
if (v_isShared_788_ == 0)
{
v___x_790_ = v___x_787_;
goto v_reusejp_789_;
}
else
{
lean_object* v_reuseFailAlloc_791_; 
v_reuseFailAlloc_791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_791_, 0, v_a_785_);
v___x_790_ = v_reuseFailAlloc_791_;
goto v_reusejp_789_;
}
v_reusejp_789_:
{
return v___x_790_;
}
}
}
}
default: 
{
lean_object* v___x_793_; lean_object* v___x_794_; 
v___x_793_ = l_Array_reverse___redArg(v_a_698_);
lean_inc(v_a_702_);
lean_inc_ref(v_a_701_);
lean_inc(v_a_700_);
lean_inc_ref(v_a_699_);
v___x_794_ = lean_apply_7(v_k_696_, v_a_697_, v___x_793_, v_a_699_, v_a_700_, v_a_701_, v_a_702_, lean_box(0));
return v___x_794_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___boxed(lean_object* v_k_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_, lean_object* v_a_802_){
_start:
{
lean_object* v_res_803_; 
v_res_803_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg(v_k_795_, v_a_796_, v_a_797_, v_a_798_, v_a_799_, v_a_800_, v_a_801_);
lean_dec(v_a_801_);
lean_dec_ref(v_a_800_);
lean_dec(v_a_799_);
lean_dec_ref(v_a_798_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go(lean_object* v_00_u03b1_804_, lean_object* v_k_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_){
_start:
{
lean_object* v___x_813_; 
v___x_813_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg(v_k_805_, v_a_806_, v_a_807_, v_a_808_, v_a_809_, v_a_810_, v_a_811_);
return v___x_813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___boxed(lean_object* v_00_u03b1_814_, lean_object* v_k_815_, lean_object* v_a_816_, lean_object* v_a_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_, lean_object* v_a_822_){
_start:
{
lean_object* v_res_823_; 
v_res_823_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go(v_00_u03b1_814_, v_k_815_, v_a_816_, v_a_817_, v_a_818_, v_a_819_, v_a_820_, v_a_821_);
lean_dec(v_a_821_);
lean_dec_ref(v_a_820_);
lean_dec(v_a_819_);
lean_dec_ref(v_a_818_);
return v_res_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg(lean_object* v_e_824_, lean_object* v_k_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_){
_start:
{
lean_object* v___x_831_; lean_object* v___x_832_; 
v___x_831_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg___closed__13));
v___x_832_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Mor_0__Mathlib_Meta_FunProp_Mor_withApp_go___redArg(v_k_825_, v_e_824_, v___x_831_, v_a_826_, v_a_827_, v_a_828_, v_a_829_);
return v___x_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg___boxed(lean_object* v_e_833_, lean_object* v_k_834_, lean_object* v_a_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_a_838_, lean_object* v_a_839_){
_start:
{
lean_object* v_res_840_; 
v_res_840_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg(v_e_833_, v_k_834_, v_a_835_, v_a_836_, v_a_837_, v_a_838_);
lean_dec(v_a_838_);
lean_dec_ref(v_a_837_);
lean_dec(v_a_836_);
lean_dec_ref(v_a_835_);
return v_res_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp(lean_object* v_00_u03b1_841_, lean_object* v_e_842_, lean_object* v_k_843_, lean_object* v_a_844_, lean_object* v_a_845_, lean_object* v_a_846_, lean_object* v_a_847_){
_start:
{
lean_object* v___x_849_; 
v___x_849_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg(v_e_842_, v_k_843_, v_a_844_, v_a_845_, v_a_846_, v_a_847_);
return v___x_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___boxed(lean_object* v_00_u03b1_850_, lean_object* v_e_851_, lean_object* v_k_852_, lean_object* v_a_853_, lean_object* v_a_854_, lean_object* v_a_855_, lean_object* v_a_856_, lean_object* v_a_857_){
_start:
{
lean_object* v_res_858_; 
v_res_858_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp(v_00_u03b1_850_, v_e_851_, v_k_852_, v_a_853_, v_a_854_, v_a_855_, v_a_856_);
lean_dec(v_a_856_);
lean_dec_ref(v_a_855_);
lean_dec(v_a_854_);
lean_dec_ref(v_a_853_);
return v_res_858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___redArg(lean_object* v_e_859_, lean_object* v_a_860_){
_start:
{
switch(lean_obj_tag(v_e_859_))
{
case 10:
{
lean_object* v_expr_862_; 
v_expr_862_ = lean_ctor_get(v_e_859_, 1);
lean_inc_ref(v_expr_862_);
lean_dec_ref_known(v_e_859_, 2);
v_e_859_ = v_expr_862_;
goto _start;
}
case 5:
{
lean_object* v_fn_864_; 
v_fn_864_ = lean_ctor_get(v_e_859_, 0);
lean_inc_ref(v_fn_864_);
lean_dec_ref_known(v_e_859_, 2);
if (lean_obj_tag(v_fn_864_) == 5)
{
lean_object* v_fn_865_; lean_object* v_arg_866_; lean_object* v___x_867_; 
v_fn_865_ = lean_ctor_get(v_fn_864_, 0);
v_arg_866_ = lean_ctor_get(v_fn_864_, 1);
v___x_867_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_isCoeFun___redArg(v_fn_865_, v_a_860_);
if (lean_obj_tag(v___x_867_) == 0)
{
lean_object* v_a_868_; uint8_t v___x_869_; 
v_a_868_ = lean_ctor_get(v___x_867_, 0);
lean_inc(v_a_868_);
lean_dec_ref_known(v___x_867_, 1);
v___x_869_ = lean_unbox(v_a_868_);
lean_dec(v_a_868_);
if (v___x_869_ == 0)
{
v_e_859_ = v_fn_864_;
goto _start;
}
else
{
lean_inc_ref(v_arg_866_);
lean_dec_ref_known(v_fn_864_, 2);
v_e_859_ = v_arg_866_;
goto _start;
}
}
else
{
lean_object* v_a_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_879_; 
lean_dec_ref_known(v_fn_864_, 2);
v_a_872_ = lean_ctor_get(v___x_867_, 0);
v_isSharedCheck_879_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_879_ == 0)
{
v___x_874_ = v___x_867_;
v_isShared_875_ = v_isSharedCheck_879_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_a_872_);
lean_dec(v___x_867_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_879_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v___x_877_; 
if (v_isShared_875_ == 0)
{
v___x_877_ = v___x_874_;
goto v_reusejp_876_;
}
else
{
lean_object* v_reuseFailAlloc_878_; 
v_reuseFailAlloc_878_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_878_, 0, v_a_872_);
v___x_877_ = v_reuseFailAlloc_878_;
goto v_reusejp_876_;
}
v_reusejp_876_:
{
return v___x_877_;
}
}
}
}
else
{
v_e_859_ = v_fn_864_;
goto _start;
}
}
default: 
{
lean_object* v___x_881_; 
v___x_881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_881_, 0, v_e_859_);
return v___x_881_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___redArg___boxed(lean_object* v_e_882_, lean_object* v_a_883_, lean_object* v_a_884_){
_start:
{
lean_object* v_res_885_; 
v_res_885_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___redArg(v_e_882_, v_a_883_);
lean_dec(v_a_883_);
return v_res_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn(lean_object* v_e_886_, lean_object* v_a_887_, lean_object* v_a_888_, lean_object* v_a_889_, lean_object* v_a_890_){
_start:
{
lean_object* v___x_892_; 
v___x_892_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___redArg(v_e_886_, v_a_890_);
return v___x_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn___boxed(lean_object* v_e_893_, lean_object* v_a_894_, lean_object* v_a_895_, lean_object* v_a_896_, lean_object* v_a_897_, lean_object* v_a_898_){
_start:
{
lean_object* v_res_899_; 
v_res_899_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppFn(v_e_893_, v_a_894_, v_a_895_, v_a_896_, v_a_897_);
lean_dec(v_a_897_);
lean_dec_ref(v_a_896_);
lean_dec(v_a_895_);
lean_dec_ref(v_a_894_);
return v_res_899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___lam__0(lean_object* v_x_900_, lean_object* v_xs_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_){
_start:
{
lean_object* v___x_907_; 
v___x_907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_907_, 0, v_xs_901_);
return v___x_907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___lam__0___boxed(lean_object* v_x_908_, lean_object* v_xs_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___lam__0(v_x_908_, v_xs_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
lean_dec_ref(v_x_908_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs(lean_object* v_e_917_, lean_object* v_a_918_, lean_object* v_a_919_, lean_object* v_a_920_, lean_object* v_a_921_){
_start:
{
lean_object* v___f_923_; lean_object* v___x_924_; 
v___f_923_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___closed__0));
v___x_924_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_withApp___redArg(v_e_917_, v___f_923_, v_a_918_, v_a_919_, v_a_920_, v_a_921_);
return v___x_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs___boxed(lean_object* v_e_925_, lean_object* v_a_926_, lean_object* v_a_927_, lean_object* v_a_928_, lean_object* v_a_929_, lean_object* v_a_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_getAppArgs(v_e_925_, v_a_926_, v_a_927_, v_a_928_, v_a_929_);
lean_dec(v_a_929_);
lean_dec_ref(v_a_928_);
lean_dec(v_a_927_);
lean_dec_ref(v_a_926_);
return v_res_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_Mor_mkAppN_spec__0(lean_object* v_as_932_, size_t v_i_933_, size_t v_stop_934_, lean_object* v_b_935_){
_start:
{
lean_object* v___y_937_; uint8_t v___x_941_; 
v___x_941_ = lean_usize_dec_eq(v_i_933_, v_stop_934_);
if (v___x_941_ == 0)
{
lean_object* v___x_942_; lean_object* v_coe_943_; 
v___x_942_ = lean_array_uget_borrowed(v_as_932_, v_i_933_);
v_coe_943_ = lean_ctor_get(v___x_942_, 1);
if (lean_obj_tag(v_coe_943_) == 0)
{
lean_object* v_expr_944_; lean_object* v___x_945_; 
v_expr_944_ = lean_ctor_get(v___x_942_, 0);
lean_inc_ref(v_expr_944_);
v___x_945_ = l_Lean_Expr_app___override(v_b_935_, v_expr_944_);
v___y_937_ = v___x_945_;
goto v___jp_936_;
}
else
{
lean_object* v_expr_946_; lean_object* v_val_947_; lean_object* v___x_948_; lean_object* v___x_949_; 
v_expr_946_ = lean_ctor_get(v___x_942_, 0);
v_val_947_ = lean_ctor_get(v_coe_943_, 0);
lean_inc(v_val_947_);
v___x_948_ = l_Lean_Expr_app___override(v_val_947_, v_b_935_);
lean_inc_ref(v_expr_946_);
v___x_949_ = l_Lean_Expr_app___override(v___x_948_, v_expr_946_);
v___y_937_ = v___x_949_;
goto v___jp_936_;
}
}
else
{
return v_b_935_;
}
v___jp_936_:
{
size_t v___x_938_; size_t v___x_939_; 
v___x_938_ = ((size_t)1ULL);
v___x_939_ = lean_usize_add(v_i_933_, v___x_938_);
v_i_933_ = v___x_939_;
v_b_935_ = v___y_937_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_Mor_mkAppN_spec__0___boxed(lean_object* v_as_950_, lean_object* v_i_951_, lean_object* v_stop_952_, lean_object* v_b_953_){
_start:
{
size_t v_i_boxed_954_; size_t v_stop_boxed_955_; lean_object* v_res_956_; 
v_i_boxed_954_ = lean_unbox_usize(v_i_951_);
lean_dec(v_i_951_);
v_stop_boxed_955_ = lean_unbox_usize(v_stop_952_);
lean_dec(v_stop_952_);
v_res_956_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_Mor_mkAppN_spec__0(v_as_950_, v_i_boxed_954_, v_stop_boxed_955_, v_b_953_);
lean_dec_ref(v_as_950_);
return v_res_956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(lean_object* v_f_957_, lean_object* v_xs_958_){
_start:
{
lean_object* v___x_959_; lean_object* v___x_960_; uint8_t v___x_961_; 
v___x_959_ = lean_unsigned_to_nat(0u);
v___x_960_ = lean_array_get_size(v_xs_958_);
v___x_961_ = lean_nat_dec_lt(v___x_959_, v___x_960_);
if (v___x_961_ == 0)
{
return v_f_957_;
}
else
{
uint8_t v___x_962_; 
v___x_962_ = lean_nat_dec_le(v___x_960_, v___x_960_);
if (v___x_962_ == 0)
{
if (v___x_961_ == 0)
{
return v_f_957_;
}
else
{
size_t v___x_963_; size_t v___x_964_; lean_object* v___x_965_; 
v___x_963_ = ((size_t)0ULL);
v___x_964_ = lean_usize_of_nat(v___x_960_);
v___x_965_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_Mor_mkAppN_spec__0(v_xs_958_, v___x_963_, v___x_964_, v_f_957_);
return v___x_965_;
}
}
else
{
size_t v___x_966_; size_t v___x_967_; lean_object* v___x_968_; 
v___x_966_ = ((size_t)0ULL);
v___x_967_ = lean_usize_of_nat(v___x_960_);
v___x_968_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Meta_FunProp_Mor_mkAppN_spec__0(v_xs_958_, v___x_966_, v___x_967_, v_f_957_);
return v___x_968_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN___boxed(lean_object* v_f_969_, lean_object* v_xs_970_){
_start:
{
lean_object* v_res_971_; 
v_res_971_ = lp_mathlib_Mathlib_Meta_FunProp_Mor_mkAppN(v_f_969_, v_xs_970_);
lean_dec_ref(v_xs_970_);
return v_res_971_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_CoeAttr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin) {
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
res = runtime_initialize_Lean_Meta_CoeAttr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_CoeAttr(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_CoeAttr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default = _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg_default);
lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg = _init_lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_Mor_instInhabitedArg);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_CoeAttr(uint8_t builtin);
lean_object* initialize_Lean_Meta_CoeAttr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Mor(uint8_t builtin) {
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
res = initialize_Lean_Meta_CoeAttr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_CoeAttr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FunProp_Mor(builtin);
}
#ifdef __cplusplus
}
#endif
