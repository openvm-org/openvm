// Lean compiler output
// Module: Aesop.BuiltinRules.DestructProducts
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Apply public import Aesop.Frontend.Attribute
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lp_aesop_Aesop_getAppUpToDefeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_revert(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* lean_expr_abstract(lean_object*, lean_object*);
lean_object* l_Lean_mkLambda(lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_mkApp4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_check___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MVarId_clear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_getUnusedName(lean_object*, lean_object*);
lean_object* l_Lean_Meta_introNCore(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_get_x21___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_diffGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__1_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "destructProducts: apply did not return exactly one goal"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__3_value;
static lean_once_cell_t lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__4;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prod"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "PProd"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "MProd"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__4_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Subtype"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Sigma"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__6_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "PSigma"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "casesOn"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 171, 149, 177, 120, 131, 37, 223)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__9_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(225, 129, 3, 119, 45, 252, 168, 83)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__9_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 171, 149, 177, 120, 131, 37, 223)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__11_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(248, 249, 30, 71, 49, 108, 60, 175)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__11_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fst"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__12_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__12_value),LEAN_SCALAR_PTR_LITERAL(64, 239, 110, 160, 241, 65, 100, 219)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "snd"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__14 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__14_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__14_value),LEAN_SCALAR_PTR_LITERAL(57, 234, 30, 96, 132, 92, 152, 52)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 250, 144, 56, 109, 24, 162, 237)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__16_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(247, 24, 174, 191, 156, 199, 52, 201)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__16 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__16_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 250, 144, 56, 109, 24, 162, 237)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__17_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(182, 125, 185, 139, 77, 41, 21, 199)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__17 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__17_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(30, 108, 3, 75, 185, 102, 103, 84)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__18_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(191, 115, 230, 188, 12, 193, 46, 63)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__18 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__18_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(30, 108, 3, 75, 185, 102, 103, 84)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__19_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(126, 56, 149, 205, 195, 101, 211, 120)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__19 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__19_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "val"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__20 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__20_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__20_value),LEAN_SCALAR_PTR_LITERAL(228, 28, 19, 111, 76, 58, 44, 203)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__21 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__21_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "property"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__22 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__22_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__22_value),LEAN_SCALAR_PTR_LITERAL(89, 150, 111, 5, 6, 150, 149, 192)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__23 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__23_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__24_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(140, 163, 176, 179, 14, 167, 130, 141)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__24 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__24_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__25 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__25_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__26_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__25_value),LEAN_SCALAR_PTR_LITERAL(74, 55, 158, 60, 144, 34, 77, 172)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__26 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__26_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "w"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__27 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__27_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__27_value),LEAN_SCALAR_PTR_LITERAL(238, 128, 149, 182, 175, 207, 218, 129)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__28 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__28_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(14, 7, 204, 83, 215, 28, 179, 196)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__29_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(15, 134, 18, 17, 232, 169, 214, 30)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__29 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__29_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(14, 7, 204, 83, 215, 28, 179, 196)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__30_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(206, 154, 92, 103, 254, 18, 5, 89)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__30 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__30_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(17, 14, 124, 134, 125, 191, 184, 142)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__31_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(252, 95, 91, 197, 249, 179, 208, 137)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__31 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__31_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(17, 14, 124, 134, 125, 191, 184, 142)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__32_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(61, 171, 224, 173, 195, 175, 128, 27)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__32 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__32_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__33_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 62, 4, 104, 125, 224, 29, 215)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__33 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__33_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__34_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(117, 121, 37, 123, 104, 28, 189, 89)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__34 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__34_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__35_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(156, 172, 111, 8, 9, 170, 133, 121)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__35 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__35_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__36_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__25_value),LEAN_SCALAR_PTR_LITERAL(58, 46, 244, 208, 18, 71, 77, 162)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__36 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__36_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "left"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__37 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__37_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__37_value),LEAN_SCALAR_PTR_LITERAL(14, 26, 230, 200, 188, 33, 106, 9)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__38 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__38_value;
static const lean_string_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "right"};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__39 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__39_value;
static const lean_ctor_object lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__39_value),LEAN_SCALAR_PTR_LITERAL(192, 52, 10, 58, 87, 38, 120, 247)}};
static const lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__40 = (const lean_object*)&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__40_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__0_value;
static const lean_string_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__1_value;
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__2_value_aux_0),((lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__2 = (const lean_object*)&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__3;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__4;
static lean_once_cell_t lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "destructProducts: found no hypothesis with a product-like type"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProductsCore(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProductsCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProducts(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProducts___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Lean_Expr_hasMVar(v_e_1_);
if (v___x_4_ == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_e_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; lean_object* v_mctx_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; lean_object* v_cache_12_; lean_object* v_zetaDeltaFVarIds_13_; lean_object* v_postponed_14_; lean_object* v_diag_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_6_ = lean_st_ref_get(v___y_2_);
v_mctx_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_mctx_7_);
lean_dec(v___x_6_);
v___x_8_ = l_Lean_instantiateMVarsCore(v_mctx_7_, v_e_1_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v___x_8_);
v___x_11_ = lean_st_ref_take(v___y_2_);
v_cache_12_ = lean_ctor_get(v___x_11_, 1);
v_zetaDeltaFVarIds_13_ = lean_ctor_get(v___x_11_, 2);
v_postponed_14_ = lean_ctor_get(v___x_11_, 3);
v_diag_15_ = lean_ctor_get(v___x_11_, 4);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_11_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_diag_15_);
lean_inc(v_postponed_14_);
lean_inc(v_zetaDeltaFVarIds_13_);
lean_inc(v_cache_12_);
lean_dec(v___x_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v_snd_10_);
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_snd_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_cache_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_zetaDeltaFVarIds_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_postponed_14_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_diag_15_);
v___x_20_ = v_reuseFailAlloc_23_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_st_ref_set(v___y_2_, v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_fst_9_);
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1(lean_object* v_e_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___redArg(v_e_30_, v___y_32_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___boxed(lean_object* v_e_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1(v_e_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg(lean_object* v_mvarId_44_, lean_object* v_x_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_44_, v_x_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
if (lean_obj_tag(v___x_51_) == 0)
{
lean_object* v_a_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_59_; 
v_a_52_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_59_ == 0)
{
v___x_54_ = v___x_51_;
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_a_52_);
lean_dec(v___x_51_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_57_; 
if (v_isShared_55_ == 0)
{
v___x_57_ = v___x_54_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_a_52_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
else
{
lean_object* v_a_60_; lean_object* v___x_62_; uint8_t v_isShared_63_; uint8_t v_isSharedCheck_67_; 
v_a_60_ = lean_ctor_get(v___x_51_, 0);
v_isSharedCheck_67_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_67_ == 0)
{
v___x_62_ = v___x_51_;
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
else
{
lean_inc(v_a_60_);
lean_dec(v___x_51_);
v___x_62_ = lean_box(0);
v_isShared_63_ = v_isSharedCheck_67_;
goto v_resetjp_61_;
}
v_resetjp_61_:
{
lean_object* v___x_65_; 
if (v_isShared_63_ == 0)
{
v___x_65_ = v___x_62_;
goto v_reusejp_64_;
}
else
{
lean_object* v_reuseFailAlloc_66_; 
v_reuseFailAlloc_66_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_66_, 0, v_a_60_);
v___x_65_ = v_reuseFailAlloc_66_;
goto v_reusejp_64_;
}
v_reusejp_64_:
{
return v___x_65_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg___boxed(lean_object* v_mvarId_68_, lean_object* v_x_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg(v_mvarId_68_, v_x_69_, v___y_70_, v___y_71_, v___y_72_, v___y_73_);
lean_dec(v___y_73_);
lean_dec_ref(v___y_72_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2(lean_object* v_00_u03b1_76_, lean_object* v_mvarId_77_, lean_object* v_x_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg(v_mvarId_77_, v_x_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___boxed(lean_object* v_00_u03b1_85_, lean_object* v_mvarId_86_, lean_object* v_x_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2(v_00_u03b1_85_, v_mvarId_86_, v_x_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_);
lean_dec(v___y_91_);
lean_dec_ref(v___y_90_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0_spec__0(lean_object* v_msgData_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
lean_object* v___x_100_; lean_object* v_env_101_; lean_object* v___x_102_; lean_object* v_mctx_103_; lean_object* v_lctx_104_; lean_object* v_options_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_100_ = lean_st_ref_get(v___y_98_);
v_env_101_ = lean_ctor_get(v___x_100_, 0);
lean_inc_ref(v_env_101_);
lean_dec(v___x_100_);
v___x_102_ = lean_st_ref_get(v___y_96_);
v_mctx_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc_ref(v_mctx_103_);
lean_dec(v___x_102_);
v_lctx_104_ = lean_ctor_get(v___y_95_, 2);
v_options_105_ = lean_ctor_get(v___y_97_, 2);
lean_inc_ref(v_options_105_);
lean_inc_ref(v_lctx_104_);
v___x_106_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_106_, 0, v_env_101_);
lean_ctor_set(v___x_106_, 1, v_mctx_103_);
lean_ctor_set(v___x_106_, 2, v_lctx_104_);
lean_ctor_set(v___x_106_, 3, v_options_105_);
v___x_107_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v_msgData_94_);
v___x_108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0_spec__0___boxed(lean_object* v_msgData_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0_spec__0(v_msgData_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___redArg(lean_object* v_msg_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v_ref_122_; lean_object* v___x_123_; lean_object* v_a_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_132_; 
v_ref_122_ = lean_ctor_get(v___y_119_, 5);
v___x_123_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0_spec__0(v_msg_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_);
v_a_124_ = lean_ctor_get(v___x_123_, 0);
v_isSharedCheck_132_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_132_ == 0)
{
v___x_126_ = v___x_123_;
v_isShared_127_ = v_isSharedCheck_132_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_a_124_);
lean_dec(v___x_123_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_132_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v___x_128_; lean_object* v___x_130_; 
lean_inc(v_ref_122_);
v___x_128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_128_, 0, v_ref_122_);
lean_ctor_set(v___x_128_, 1, v_a_124_);
if (v_isShared_127_ == 0)
{
lean_ctor_set_tag(v___x_126_, 1);
lean_ctor_set(v___x_126_, 0, v___x_128_);
v___x_130_ = v___x_126_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v___x_128_);
v___x_130_ = v_reuseFailAlloc_131_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
return v___x_130_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___redArg___boxed(lean_object* v_msg_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___redArg(v_msg_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
return v_res_139_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__4(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_148_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__3));
v___x_149_ = l_Lean_stringToMessageData(v___x_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac(lean_object* v_goal_150_, lean_object* v_hyp_151_, lean_object* v_hypType_152_, lean_object* v_rec_153_, lean_object* v_lType_154_, lean_object* v_rType_155_, lean_object* v_lName_156_, lean_object* v_rName_157_, lean_object* v_a_158_, lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; uint8_t v___x_166_; uint8_t v___x_167_; lean_object* v___x_168_; 
v___x_163_ = lean_unsigned_to_nat(1u);
v___x_164_ = lean_mk_empty_array_with_capacity(v___x_163_);
lean_inc_ref(v___x_164_);
v___x_165_ = lean_array_push(v___x_164_, v_hyp_151_);
v___x_166_ = 1;
v___x_167_ = 0;
v___x_168_ = l_Lean_MVarId_revert(v_goal_150_, v___x_165_, v___x_166_, v___x_167_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_168_) == 0)
{
lean_object* v_a_169_; lean_object* v_fst_170_; lean_object* v_snd_171_; lean_object* v___x_172_; 
v_a_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_a_169_);
lean_dec_ref_known(v___x_168_, 1);
v_fst_170_ = lean_ctor_get(v_a_169_, 0);
lean_inc(v_fst_170_);
v_snd_171_ = lean_ctor_get(v_a_169_, 1);
lean_inc(v_snd_171_);
lean_dec(v_a_169_);
v___x_172_ = l_Lean_Meta_intro1Core(v_snd_171_, v___x_166_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_172_) == 0)
{
lean_object* v_a_173_; lean_object* v_fst_174_; lean_object* v_snd_175_; lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_314_; 
v_a_173_ = lean_ctor_get(v___x_172_, 0);
lean_inc(v_a_173_);
lean_dec_ref_known(v___x_172_, 1);
v_fst_174_ = lean_ctor_get(v_a_173_, 0);
v_snd_175_ = lean_ctor_get(v_a_173_, 1);
v_isSharedCheck_314_ = !lean_is_exclusive(v_a_173_);
if (v_isSharedCheck_314_ == 0)
{
v___x_177_ = v_a_173_;
v_isShared_178_ = v_isSharedCheck_314_;
goto v_resetjp_176_;
}
else
{
lean_inc(v_snd_175_);
lean_inc(v_fst_174_);
lean_dec(v_a_173_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_314_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v___x_179_; 
lean_inc(v_snd_175_);
v___x_179_ = l_Lean_MVarId_getType(v_snd_175_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_179_) == 0)
{
lean_object* v_a_180_; lean_object* v___x_181_; lean_object* v_a_182_; lean_object* v___x_183_; lean_object* v___x_184_; uint8_t v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; uint8_t v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v_a_180_ = lean_ctor_get(v___x_179_, 0);
lean_inc(v_a_180_);
lean_dec_ref_known(v___x_179_, 1);
v___x_181_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__1___redArg(v_a_180_, v_a_159_);
v_a_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc(v_a_182_);
lean_dec_ref(v___x_181_);
lean_inc(v_fst_174_);
v___x_183_ = l_Lean_mkFVar(v_fst_174_);
v___x_184_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__1));
v___x_185_ = 0;
lean_inc_ref(v___x_183_);
v___x_186_ = lean_array_push(v___x_164_, v___x_183_);
v___x_187_ = lean_expr_abstract(v_a_182_, v___x_186_);
lean_dec_ref(v___x_186_);
lean_dec(v_a_182_);
v___x_188_ = l_Lean_mkLambda(v___x_184_, v___x_185_, v_hypType_152_, v___x_187_);
v___x_189_ = l_Lean_mkApp4(v_rec_153_, v_lType_154_, v_rType_155_, v___x_188_, v___x_183_);
v___x_190_ = 0;
v___x_191_ = lean_box(v___x_190_);
lean_inc_ref(v___x_189_);
v___x_192_ = lean_alloc_closure((void*)(l_Lean_Meta_check___boxed), 7, 2);
lean_closure_set(v___x_192_, 0, v___x_189_);
lean_closure_set(v___x_192_, 1, v___x_191_);
lean_inc(v_snd_175_);
v___x_193_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg(v_snd_175_, v___x_192_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_193_) == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
lean_dec_ref_known(v___x_193_, 1);
v___x_194_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__2));
v___x_195_ = lean_box(0);
v___x_196_ = l_Lean_MVarId_apply(v_snd_175_, v___x_189_, v___x_194_, v___x_195_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_196_) == 0)
{
lean_object* v_a_197_; lean_object* v___y_199_; lean_object* v___y_200_; lean_object* v___y_201_; lean_object* v___y_202_; 
v_a_197_ = lean_ctor_get(v___x_196_, 0);
lean_inc(v_a_197_);
lean_dec_ref_known(v___x_196_, 1);
if (lean_obj_tag(v_a_197_) == 1)
{
lean_object* v_tail_205_; 
v_tail_205_ = lean_ctor_get(v_a_197_, 1);
if (lean_obj_tag(v_tail_205_) == 0)
{
lean_object* v_head_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_288_; 
v_head_206_ = lean_ctor_get(v_a_197_, 0);
v_isSharedCheck_288_ = !lean_is_exclusive(v_a_197_);
if (v_isSharedCheck_288_ == 0)
{
lean_object* v_unused_289_; 
v_unused_289_ = lean_ctor_get(v_a_197_, 1);
lean_dec(v_unused_289_);
v___x_208_ = v_a_197_;
v_isShared_209_ = v_isSharedCheck_288_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_head_206_);
lean_dec(v_a_197_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_288_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_210_; 
v___x_210_ = l_Lean_MVarId_clear(v_head_206_, v_fst_174_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_object* v_a_211_; lean_object* v___x_212_; 
v_a_211_ = lean_ctor_get(v___x_210_, 0);
lean_inc_n(v_a_211_, 2);
lean_dec_ref_known(v___x_210_, 1);
v___x_212_ = l_Lean_MVarId_getDecl(v_a_211_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_212_) == 0)
{
lean_object* v_a_213_; lean_object* v_lctx_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_220_; 
v_a_213_ = lean_ctor_get(v___x_212_, 0);
lean_inc(v_a_213_);
lean_dec_ref_known(v___x_212_, 1);
v_lctx_214_ = lean_ctor_get(v_a_213_, 1);
lean_inc_ref(v_lctx_214_);
lean_dec(v_a_213_);
v___x_215_ = l_Lean_LocalContext_getUnusedName(v_lctx_214_, v_lName_156_);
v___x_216_ = l_Lean_LocalContext_getUnusedName(v_lctx_214_, v_rName_157_);
lean_dec_ref(v_lctx_214_);
v___x_217_ = lean_unsigned_to_nat(2u);
v___x_218_ = lean_box(0);
lean_inc(v___x_216_);
if (v_isShared_209_ == 0)
{
lean_ctor_set(v___x_208_, 1, v___x_218_);
lean_ctor_set(v___x_208_, 0, v___x_216_);
v___x_220_ = v___x_208_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v___x_216_);
lean_ctor_set(v_reuseFailAlloc_271_, 1, v___x_218_);
v___x_220_ = v_reuseFailAlloc_271_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
lean_object* v___x_222_; 
lean_inc(v___x_215_);
if (v_isShared_178_ == 0)
{
lean_ctor_set_tag(v___x_177_, 1);
lean_ctor_set(v___x_177_, 1, v___x_220_);
lean_ctor_set(v___x_177_, 0, v___x_215_);
v___x_222_ = v___x_177_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v___x_215_);
lean_ctor_set(v_reuseFailAlloc_270_, 1, v___x_220_);
v___x_222_ = v_reuseFailAlloc_270_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
lean_object* v___x_223_; 
v___x_223_ = l_Lean_Meta_introNCore(v_a_211_, v___x_217_, v___x_222_, v___x_167_, v___x_167_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_223_) == 0)
{
lean_object* v_a_224_; lean_object* v_snd_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_260_; 
v_a_224_ = lean_ctor_get(v___x_223_, 0);
lean_inc(v_a_224_);
lean_dec_ref_known(v___x_223_, 1);
v_snd_225_ = lean_ctor_get(v_a_224_, 1);
v_isSharedCheck_260_ = !lean_is_exclusive(v_a_224_);
if (v_isSharedCheck_260_ == 0)
{
lean_object* v_unused_261_; 
v_unused_261_ = lean_ctor_get(v_a_224_, 0);
lean_dec(v_unused_261_);
v___x_227_ = v_a_224_;
v_isShared_228_ = v_isSharedCheck_260_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_snd_225_);
lean_dec(v_a_224_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_260_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_229_ = lean_array_get_size(v_fst_170_);
lean_dec(v_fst_170_);
v___x_230_ = lean_nat_sub(v___x_229_, v___x_163_);
v___x_231_ = l_Lean_Meta_introNCore(v_snd_225_, v___x_230_, v___x_218_, v___x_167_, v___x_166_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
if (lean_obj_tag(v___x_231_) == 0)
{
lean_object* v_a_232_; lean_object* v___x_234_; uint8_t v_isShared_235_; uint8_t v_isSharedCheck_251_; 
v_a_232_ = lean_ctor_get(v___x_231_, 0);
v_isSharedCheck_251_ = !lean_is_exclusive(v___x_231_);
if (v_isSharedCheck_251_ == 0)
{
v___x_234_ = v___x_231_;
v_isShared_235_ = v_isSharedCheck_251_;
goto v_resetjp_233_;
}
else
{
lean_inc(v_a_232_);
lean_dec(v___x_231_);
v___x_234_ = lean_box(0);
v_isShared_235_ = v_isSharedCheck_251_;
goto v_resetjp_233_;
}
v_resetjp_233_:
{
lean_object* v_snd_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_249_; 
v_snd_236_ = lean_ctor_get(v_a_232_, 1);
v_isSharedCheck_249_ = !lean_is_exclusive(v_a_232_);
if (v_isSharedCheck_249_ == 0)
{
lean_object* v_unused_250_; 
v_unused_250_ = lean_ctor_get(v_a_232_, 0);
lean_dec(v_unused_250_);
v___x_238_ = v_a_232_;
v_isShared_239_ = v_isSharedCheck_249_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_snd_236_);
lean_dec(v_a_232_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_249_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_241_; 
if (v_isShared_239_ == 0)
{
lean_ctor_set(v___x_238_, 1, v___x_216_);
lean_ctor_set(v___x_238_, 0, v___x_215_);
v___x_241_ = v___x_238_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v___x_215_);
lean_ctor_set(v_reuseFailAlloc_248_, 1, v___x_216_);
v___x_241_ = v_reuseFailAlloc_248_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
lean_object* v___x_243_; 
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 1, v___x_241_);
lean_ctor_set(v___x_227_, 0, v_snd_236_);
v___x_243_ = v___x_227_;
goto v_reusejp_242_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_snd_236_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v___x_241_);
v___x_243_ = v_reuseFailAlloc_247_;
goto v_reusejp_242_;
}
v_reusejp_242_:
{
lean_object* v___x_245_; 
if (v_isShared_235_ == 0)
{
lean_ctor_set(v___x_234_, 0, v___x_243_);
v___x_245_ = v___x_234_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_243_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
}
}
else
{
lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
lean_del_object(v___x_227_);
lean_dec(v___x_216_);
lean_dec(v___x_215_);
v_a_252_ = lean_ctor_get(v___x_231_, 0);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_231_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_231_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_dec(v___x_231_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
}
else
{
lean_object* v_a_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_269_; 
lean_dec(v___x_216_);
lean_dec(v___x_215_);
lean_dec(v_fst_170_);
v_a_262_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_269_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_269_ == 0)
{
v___x_264_ = v___x_223_;
v_isShared_265_ = v_isSharedCheck_269_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_a_262_);
lean_dec(v___x_223_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_269_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v___x_267_; 
if (v_isShared_265_ == 0)
{
v___x_267_ = v___x_264_;
goto v_reusejp_266_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v_a_262_);
v___x_267_ = v_reuseFailAlloc_268_;
goto v_reusejp_266_;
}
v_reusejp_266_:
{
return v___x_267_;
}
}
}
}
}
}
else
{
lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_279_; 
lean_dec(v_a_211_);
lean_del_object(v___x_208_);
lean_del_object(v___x_177_);
lean_dec(v_fst_170_);
v_a_272_ = lean_ctor_get(v___x_212_, 0);
v_isSharedCheck_279_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_279_ == 0)
{
v___x_274_ = v___x_212_;
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_dec(v___x_212_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_277_; 
if (v_isShared_275_ == 0)
{
v___x_277_ = v___x_274_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v_a_272_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
else
{
lean_object* v_a_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_287_; 
lean_del_object(v___x_208_);
lean_del_object(v___x_177_);
lean_dec(v_fst_170_);
v_a_280_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_287_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_287_ == 0)
{
v___x_282_ = v___x_210_;
v_isShared_283_ = v_isSharedCheck_287_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_a_280_);
lean_dec(v___x_210_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_287_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v___x_285_; 
if (v_isShared_283_ == 0)
{
v___x_285_ = v___x_282_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_286_; 
v_reuseFailAlloc_286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_286_, 0, v_a_280_);
v___x_285_ = v_reuseFailAlloc_286_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
return v___x_285_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_a_197_, 2);
lean_del_object(v___x_177_);
lean_dec(v_fst_174_);
lean_dec(v_fst_170_);
v___y_199_ = v_a_158_;
v___y_200_ = v_a_159_;
v___y_201_ = v_a_160_;
v___y_202_ = v_a_161_;
goto v___jp_198_;
}
}
else
{
lean_dec(v_a_197_);
lean_del_object(v___x_177_);
lean_dec(v_fst_174_);
lean_dec(v_fst_170_);
v___y_199_ = v_a_158_;
v___y_200_ = v_a_159_;
v___y_201_ = v_a_160_;
v___y_202_ = v_a_161_;
goto v___jp_198_;
}
v___jp_198_:
{
lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_203_ = lean_obj_once(&lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__4, &lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__4_once, _init_lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__4);
v___x_204_ = lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___redArg(v___x_203_, v___y_199_, v___y_200_, v___y_201_, v___y_202_);
return v___x_204_;
}
}
else
{
lean_object* v_a_290_; lean_object* v___x_292_; uint8_t v_isShared_293_; uint8_t v_isSharedCheck_297_; 
lean_del_object(v___x_177_);
lean_dec(v_fst_174_);
lean_dec(v_fst_170_);
v_a_290_ = lean_ctor_get(v___x_196_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v___x_196_);
if (v_isSharedCheck_297_ == 0)
{
v___x_292_ = v___x_196_;
v_isShared_293_ = v_isSharedCheck_297_;
goto v_resetjp_291_;
}
else
{
lean_inc(v_a_290_);
lean_dec(v___x_196_);
v___x_292_ = lean_box(0);
v_isShared_293_ = v_isSharedCheck_297_;
goto v_resetjp_291_;
}
v_resetjp_291_:
{
lean_object* v___x_295_; 
if (v_isShared_293_ == 0)
{
v___x_295_ = v___x_292_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v_a_290_);
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
else
{
lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_305_; 
lean_dec_ref(v___x_189_);
lean_del_object(v___x_177_);
lean_dec(v_snd_175_);
lean_dec(v_fst_174_);
lean_dec(v_fst_170_);
v_a_298_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_305_ == 0)
{
v___x_300_ = v___x_193_;
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_193_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_303_; 
if (v_isShared_301_ == 0)
{
v___x_303_ = v___x_300_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_a_298_);
v___x_303_ = v_reuseFailAlloc_304_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
return v___x_303_;
}
}
}
}
else
{
lean_object* v_a_306_; lean_object* v___x_308_; uint8_t v_isShared_309_; uint8_t v_isSharedCheck_313_; 
lean_del_object(v___x_177_);
lean_dec(v_snd_175_);
lean_dec(v_fst_174_);
lean_dec(v_fst_170_);
lean_dec_ref(v___x_164_);
lean_dec_ref(v_rType_155_);
lean_dec_ref(v_lType_154_);
lean_dec_ref(v_rec_153_);
lean_dec_ref(v_hypType_152_);
v_a_306_ = lean_ctor_get(v___x_179_, 0);
v_isSharedCheck_313_ = !lean_is_exclusive(v___x_179_);
if (v_isSharedCheck_313_ == 0)
{
v___x_308_ = v___x_179_;
v_isShared_309_ = v_isSharedCheck_313_;
goto v_resetjp_307_;
}
else
{
lean_inc(v_a_306_);
lean_dec(v___x_179_);
v___x_308_ = lean_box(0);
v_isShared_309_ = v_isSharedCheck_313_;
goto v_resetjp_307_;
}
v_resetjp_307_:
{
lean_object* v___x_311_; 
if (v_isShared_309_ == 0)
{
v___x_311_ = v___x_308_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v_a_306_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
}
}
}
else
{
lean_object* v_a_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_322_; 
lean_dec(v_fst_170_);
lean_dec_ref(v___x_164_);
lean_dec_ref(v_rType_155_);
lean_dec_ref(v_lType_154_);
lean_dec_ref(v_rec_153_);
lean_dec_ref(v_hypType_152_);
v_a_315_ = lean_ctor_get(v___x_172_, 0);
v_isSharedCheck_322_ = !lean_is_exclusive(v___x_172_);
if (v_isSharedCheck_322_ == 0)
{
v___x_317_ = v___x_172_;
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_a_315_);
lean_dec(v___x_172_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_320_; 
if (v_isShared_318_ == 0)
{
v___x_320_ = v___x_317_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v_a_315_);
v___x_320_ = v_reuseFailAlloc_321_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
return v___x_320_;
}
}
}
}
else
{
lean_object* v_a_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_330_; 
lean_dec_ref(v___x_164_);
lean_dec_ref(v_rType_155_);
lean_dec_ref(v_lType_154_);
lean_dec_ref(v_rec_153_);
lean_dec_ref(v_hypType_152_);
v_a_323_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_330_ == 0)
{
v___x_325_ = v___x_168_;
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_a_323_);
lean_dec(v___x_168_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___x_328_; 
if (v_isShared_326_ == 0)
{
v___x_328_ = v___x_325_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_323_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___boxed(lean_object* v_goal_331_, lean_object* v_hyp_332_, lean_object* v_hypType_333_, lean_object* v_rec_334_, lean_object* v_lType_335_, lean_object* v_rType_336_, lean_object* v_lName_337_, lean_object* v_rName_338_, lean_object* v_a_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac(v_goal_331_, v_hyp_332_, v_hypType_333_, v_rec_334_, v_lType_335_, v_rType_336_, v_lName_337_, v_rName_338_, v_a_339_, v_a_340_, v_a_341_, v_a_342_);
lean_dec(v_a_342_);
lean_dec_ref(v_a_341_);
lean_dec(v_a_340_);
lean_dec_ref(v_a_339_);
lean_dec(v_rName_338_);
lean_dec(v_lName_337_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0(lean_object* v_00_u03b1_345_, lean_object* v_msg_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___redArg(v_msg_346_, v___y_347_, v___y_348_, v___y_349_, v___y_350_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0___boxed(lean_object* v_00_u03b1_353_, lean_object* v_msg_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_aesop_Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0(v_00_u03b1_353_, v_msg_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_);
lean_dec(v___y_358_);
lean_dec_ref(v___y_357_);
lean_dec(v___y_356_);
lean_dec_ref(v___y_355_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___lam__0(lean_object* v_snd_361_, lean_object* v_hyp_362_, lean_object* v_ctor_363_, lean_object* v_goal_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_){
_start:
{
lean_object* v_fst_370_; lean_object* v_snd_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; uint8_t v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
v_fst_370_ = lean_ctor_get(v_snd_361_, 0);
lean_inc(v_fst_370_);
v_snd_371_ = lean_ctor_get(v_snd_361_, 1);
lean_inc(v_snd_371_);
lean_dec_ref(v_snd_361_);
v___x_372_ = l_Lean_Expr_fvar___override(v_hyp_362_);
v___x_373_ = lean_unsigned_to_nat(2u);
v___x_374_ = lean_mk_empty_array_with_capacity(v___x_373_);
v___x_375_ = lean_array_push(v___x_374_, v_fst_370_);
v___x_376_ = lean_array_push(v___x_375_, v_snd_371_);
v___x_377_ = 0;
v___x_378_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_378_, 0, v_ctor_363_);
lean_ctor_set(v___x_378_, 1, v___x_376_);
lean_ctor_set_uint8(v___x_378_, sizeof(void*)*2, v___x_377_);
v___x_379_ = lp_aesop_Aesop_Script_TacticBuilder_obtain(v_goal_364_, v___x_372_, v___x_378_, v___y_365_, v___y_366_, v___y_367_, v___y_368_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___lam__0___boxed(lean_object* v_snd_380_, lean_object* v_hyp_381_, lean_object* v_ctor_382_, lean_object* v_goal_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___lam__0(v_snd_380_, v_hyp_381_, v_ctor_382_, v_goal_383_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(lean_object* v_goal_390_, lean_object* v_hyp_391_, lean_object* v_hypType_392_, lean_object* v_rec_393_, lean_object* v_lType_394_, lean_object* v_rType_395_, lean_object* v_ctor_396_, lean_object* v_lName_397_, lean_object* v_rName_398_, lean_object* v_a_399_, lean_object* v_a_400_, lean_object* v_a_401_, lean_object* v_a_402_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = l_Lean_Meta_saveState___redArg(v_a_400_, v_a_402_);
if (lean_obj_tag(v___x_404_) == 0)
{
lean_object* v_a_405_; lean_object* v___x_406_; 
v_a_405_ = lean_ctor_get(v___x_404_, 0);
lean_inc(v_a_405_);
lean_dec_ref_known(v___x_404_, 1);
lean_inc(v_hyp_391_);
lean_inc(v_goal_390_);
v___x_406_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac(v_goal_390_, v_hyp_391_, v_hypType_392_, v_rec_393_, v_lType_394_, v_rType_395_, v_lName_397_, v_rName_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_);
if (lean_obj_tag(v___x_406_) == 0)
{
lean_object* v_a_407_; lean_object* v___x_408_; 
v_a_407_ = lean_ctor_get(v___x_406_, 0);
lean_inc(v_a_407_);
lean_dec_ref_known(v___x_406_, 1);
v___x_408_ = l_Lean_Meta_saveState___redArg(v_a_400_, v_a_402_);
if (lean_obj_tag(v___x_408_) == 0)
{
lean_object* v_a_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_431_; 
v_a_409_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_431_ == 0)
{
v___x_411_ = v___x_408_;
v_isShared_412_ = v_isSharedCheck_431_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_a_409_);
lean_dec(v___x_408_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_431_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v_fst_413_; lean_object* v_snd_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_430_; 
v_fst_413_ = lean_ctor_get(v_a_407_, 0);
v_snd_414_ = lean_ctor_get(v_a_407_, 1);
v_isSharedCheck_430_ = !lean_is_exclusive(v_a_407_);
if (v_isSharedCheck_430_ == 0)
{
v___x_416_ = v_a_407_;
v_isShared_417_ = v_isSharedCheck_430_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_snd_414_);
lean_inc(v_fst_413_);
lean_dec(v_a_407_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_430_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___f_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_425_; 
lean_inc(v_goal_390_);
v___f_418_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___lam__0___boxed), 9, 4);
lean_closure_set(v___f_418_, 0, v_snd_414_);
lean_closure_set(v___f_418_, 1, v_hyp_391_);
lean_closure_set(v___f_418_, 2, v_ctor_396_);
lean_closure_set(v___f_418_, 3, v_goal_390_);
v___x_419_ = lean_unsigned_to_nat(1u);
v___x_420_ = lean_mk_empty_array_with_capacity(v___x_419_);
lean_inc_ref(v___x_420_);
v___x_421_ = lean_array_push(v___x_420_, v___f_418_);
lean_inc(v_fst_413_);
v___x_422_ = lean_array_push(v___x_420_, v_fst_413_);
v___x_423_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_423_, 0, v_a_405_);
lean_ctor_set(v___x_423_, 1, v_goal_390_);
lean_ctor_set(v___x_423_, 2, v___x_421_);
lean_ctor_set(v___x_423_, 3, v_a_409_);
lean_ctor_set(v___x_423_, 4, v___x_422_);
if (v_isShared_417_ == 0)
{
lean_ctor_set(v___x_416_, 1, v_fst_413_);
lean_ctor_set(v___x_416_, 0, v___x_423_);
v___x_425_ = v___x_416_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v___x_423_);
lean_ctor_set(v_reuseFailAlloc_429_, 1, v_fst_413_);
v___x_425_ = v_reuseFailAlloc_429_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
lean_object* v___x_427_; 
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 0, v___x_425_);
v___x_427_ = v___x_411_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_425_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
}
}
}
else
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_439_; 
lean_dec(v_a_407_);
lean_dec(v_a_405_);
lean_dec(v_ctor_396_);
lean_dec(v_hyp_391_);
lean_dec(v_goal_390_);
v_a_432_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_439_ == 0)
{
v___x_434_ = v___x_408_;
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_408_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_437_; 
if (v_isShared_435_ == 0)
{
v___x_437_ = v___x_434_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_a_432_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
else
{
lean_object* v_a_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_447_; 
lean_dec(v_a_405_);
lean_dec(v_ctor_396_);
lean_dec(v_hyp_391_);
lean_dec(v_goal_390_);
v_a_440_ = lean_ctor_get(v___x_406_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v___x_406_);
if (v_isSharedCheck_447_ == 0)
{
v___x_442_ = v___x_406_;
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_a_440_);
lean_dec(v___x_406_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_445_; 
if (v_isShared_443_ == 0)
{
v___x_445_ = v___x_442_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v_a_440_);
v___x_445_ = v_reuseFailAlloc_446_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
return v___x_445_;
}
}
}
}
else
{
lean_object* v_a_448_; lean_object* v___x_450_; uint8_t v_isShared_451_; uint8_t v_isSharedCheck_455_; 
lean_dec(v_ctor_396_);
lean_dec_ref(v_rType_395_);
lean_dec_ref(v_lType_394_);
lean_dec_ref(v_rec_393_);
lean_dec_ref(v_hypType_392_);
lean_dec(v_hyp_391_);
lean_dec(v_goal_390_);
v_a_448_ = lean_ctor_get(v___x_404_, 0);
v_isSharedCheck_455_ = !lean_is_exclusive(v___x_404_);
if (v_isSharedCheck_455_ == 0)
{
v___x_450_ = v___x_404_;
v_isShared_451_ = v_isSharedCheck_455_;
goto v_resetjp_449_;
}
else
{
lean_inc(v_a_448_);
lean_dec(v___x_404_);
v___x_450_ = lean_box(0);
v_isShared_451_ = v_isSharedCheck_455_;
goto v_resetjp_449_;
}
v_resetjp_449_:
{
lean_object* v___x_453_; 
if (v_isShared_451_ == 0)
{
v___x_453_ = v___x_450_;
goto v_reusejp_452_;
}
else
{
lean_object* v_reuseFailAlloc_454_; 
v_reuseFailAlloc_454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_454_, 0, v_a_448_);
v___x_453_ = v_reuseFailAlloc_454_;
goto v_reusejp_452_;
}
v_reusejp_452_:
{
return v___x_453_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go___boxed(lean_object* v_goal_456_, lean_object* v_hyp_457_, lean_object* v_hypType_458_, lean_object* v_rec_459_, lean_object* v_lType_460_, lean_object* v_rType_461_, lean_object* v_ctor_462_, lean_object* v_lName_463_, lean_object* v_rName_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_, lean_object* v_a_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_456_, v_hyp_457_, v_hypType_458_, v_rec_459_, v_lType_460_, v_rType_461_, v_ctor_462_, v_lName_463_, v_rName_464_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
lean_dec(v_a_468_);
lean_dec_ref(v_a_467_);
lean_dec(v_a_466_);
lean_dec_ref(v_a_465_);
lean_dec(v_rName_464_);
lean_dec(v_lName_463_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0(lean_object* v_hyp_551_, uint8_t v_md_552_, lean_object* v_goal_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
lean_object* v___x_562_; 
lean_inc(v_hyp_551_);
v___x_562_ = l_Lean_FVarId_getType___redArg(v_hyp_551_, v___y_554_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_562_) == 0)
{
lean_object* v_a_563_; lean_object* v_keyedConfig_564_; uint8_t v_trackZetaDelta_565_; lean_object* v_zetaDeltaSet_566_; lean_object* v_lctx_567_; lean_object* v_localInstances_568_; lean_object* v_defEqCtx_x3f_569_; lean_object* v_synthPendingDepth_570_; lean_object* v_customCanUnfoldPredicate_x3f_571_; uint8_t v_univApprox_572_; uint8_t v_inTypeClassResolution_573_; uint8_t v_cacheInferType_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; 
v_a_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc_n(v_a_563_, 2);
lean_dec_ref_known(v___x_562_, 1);
v_keyedConfig_564_ = lean_ctor_get(v___y_554_, 0);
v_trackZetaDelta_565_ = lean_ctor_get_uint8(v___y_554_, sizeof(void*)*7);
v_zetaDeltaSet_566_ = lean_ctor_get(v___y_554_, 1);
v_lctx_567_ = lean_ctor_get(v___y_554_, 2);
v_localInstances_568_ = lean_ctor_get(v___y_554_, 3);
v_defEqCtx_x3f_569_ = lean_ctor_get(v___y_554_, 4);
v_synthPendingDepth_570_ = lean_ctor_get(v___y_554_, 5);
v_customCanUnfoldPredicate_x3f_571_ = lean_ctor_get(v___y_554_, 6);
v_univApprox_572_ = lean_ctor_get_uint8(v___y_554_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_573_ = lean_ctor_get_uint8(v___y_554_, sizeof(void*)*7 + 2);
v_cacheInferType_574_ = lean_ctor_get_uint8(v___y_554_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_564_);
v___x_575_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_md_552_, v_keyedConfig_564_);
lean_inc(v_customCanUnfoldPredicate_x3f_571_);
lean_inc(v_synthPendingDepth_570_);
lean_inc(v_defEqCtx_x3f_569_);
lean_inc_ref(v_localInstances_568_);
lean_inc_ref(v_lctx_567_);
lean_inc(v_zetaDeltaSet_566_);
v___x_576_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_576_, 0, v___x_575_);
lean_ctor_set(v___x_576_, 1, v_zetaDeltaSet_566_);
lean_ctor_set(v___x_576_, 2, v_lctx_567_);
lean_ctor_set(v___x_576_, 3, v_localInstances_568_);
lean_ctor_set(v___x_576_, 4, v_defEqCtx_x3f_569_);
lean_ctor_set(v___x_576_, 5, v_synthPendingDepth_570_);
lean_ctor_set(v___x_576_, 6, v_customCanUnfoldPredicate_x3f_571_);
lean_ctor_set_uint8(v___x_576_, sizeof(void*)*7, v_trackZetaDelta_565_);
lean_ctor_set_uint8(v___x_576_, sizeof(void*)*7 + 1, v_univApprox_572_);
lean_ctor_set_uint8(v___x_576_, sizeof(void*)*7 + 2, v_inTypeClassResolution_573_);
lean_ctor_set_uint8(v___x_576_, sizeof(void*)*7 + 3, v_cacheInferType_574_);
v___x_577_ = lp_aesop_Aesop_getAppUpToDefeq(v_a_563_, v___x_576_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref_known(v___x_576_, 7);
if (lean_obj_tag(v___x_577_) == 0)
{
lean_object* v_a_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_895_; 
v_a_578_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_895_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_895_ == 0)
{
v___x_580_ = v___x_577_;
v_isShared_581_ = v_isSharedCheck_895_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_a_578_);
lean_dec(v___x_577_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_895_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
lean_object* v_fst_582_; lean_object* v_snd_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_894_; 
v_fst_582_ = lean_ctor_get(v_a_578_, 0);
v_snd_583_ = lean_ctor_get(v_a_578_, 1);
v_isSharedCheck_894_ = !lean_is_exclusive(v_a_578_);
if (v_isSharedCheck_894_ == 0)
{
v___x_585_ = v_a_578_;
v_isShared_586_ = v_isSharedCheck_894_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_snd_583_);
lean_inc(v_fst_582_);
lean_dec(v_a_578_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_894_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v___x_587_; lean_object* v___x_588_; uint8_t v___x_589_; 
v___x_587_ = lean_array_get_size(v_snd_583_);
v___x_588_ = lean_unsigned_to_nat(2u);
v___x_589_ = lean_nat_dec_eq(v___x_587_, v___x_588_);
if (v___x_589_ == 0)
{
lean_object* v___x_590_; lean_object* v___x_592_; 
lean_del_object(v___x_585_);
lean_dec(v_snd_583_);
lean_dec(v_fst_582_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v___x_590_ = lean_box(0);
if (v_isShared_581_ == 0)
{
lean_ctor_set(v___x_580_, 0, v___x_590_);
v___x_592_ = v___x_580_;
goto v_reusejp_591_;
}
else
{
lean_object* v_reuseFailAlloc_593_; 
v_reuseFailAlloc_593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_593_, 0, v___x_590_);
v___x_592_ = v_reuseFailAlloc_593_;
goto v_reusejp_591_;
}
v_reusejp_591_:
{
return v___x_592_;
}
}
else
{
lean_del_object(v___x_580_);
if (lean_obj_tag(v_fst_582_) == 4)
{
lean_object* v_declName_594_; 
v_declName_594_ = lean_ctor_get(v_fst_582_, 0);
lean_inc(v_declName_594_);
if (lean_obj_tag(v_declName_594_) == 1)
{
lean_object* v_pre_595_; 
v_pre_595_ = lean_ctor_get(v_declName_594_, 0);
if (lean_obj_tag(v_pre_595_) == 0)
{
lean_object* v_us_596_; lean_object* v_str_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; uint8_t v___x_603_; 
v_us_596_ = lean_ctor_get(v_fst_582_, 1);
lean_inc(v_us_596_);
lean_dec_ref_known(v_fst_582_, 2);
v_str_597_ = lean_ctor_get(v_declName_594_, 1);
lean_inc_ref(v_str_597_);
lean_dec_ref_known(v_declName_594_, 2);
v___x_598_ = lean_unsigned_to_nat(0u);
v___x_599_ = lean_array_fget(v_snd_583_, v___x_598_);
v___x_600_ = lean_unsigned_to_nat(1u);
v___x_601_ = lean_array_fget(v_snd_583_, v___x_600_);
lean_dec(v_snd_583_);
v___x_602_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__0));
v___x_603_ = lean_string_dec_eq(v_str_597_, v___x_602_);
if (v___x_603_ == 0)
{
lean_object* v___x_604_; uint8_t v___x_605_; 
v___x_604_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__1));
v___x_605_ = lean_string_dec_eq(v_str_597_, v___x_604_);
if (v___x_605_ == 0)
{
lean_object* v___x_606_; uint8_t v___x_607_; 
v___x_606_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__2));
v___x_607_ = lean_string_dec_eq(v_str_597_, v___x_606_);
if (v___x_607_ == 0)
{
lean_object* v___x_608_; uint8_t v___x_609_; 
v___x_608_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__3));
v___x_609_ = lean_string_dec_eq(v_str_597_, v___x_608_);
if (v___x_609_ == 0)
{
lean_object* v___x_610_; uint8_t v___x_611_; 
v___x_610_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__4));
v___x_611_ = lean_string_dec_eq(v_str_597_, v___x_610_);
if (v___x_611_ == 0)
{
lean_object* v___x_612_; uint8_t v___x_613_; 
v___x_612_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__5));
v___x_613_ = lean_string_dec_eq(v_str_597_, v___x_612_);
if (v___x_613_ == 0)
{
lean_object* v___x_614_; uint8_t v___x_615_; 
v___x_614_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__6));
v___x_615_ = lean_string_dec_eq(v_str_597_, v___x_614_);
if (v___x_615_ == 0)
{
lean_object* v___x_616_; uint8_t v___x_617_; 
v___x_616_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__7));
v___x_617_ = lean_string_dec_eq(v_str_597_, v___x_616_);
lean_dec_ref(v_str_597_);
if (v___x_617_ == 0)
{
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_dec(v_us_596_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
goto v___jp_559_;
}
else
{
lean_object* v___x_618_; 
v___x_618_ = l_Lean_Meta_mkFreshLevelMVar(v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_618_) == 0)
{
lean_object* v_a_619_; lean_object* v___x_620_; lean_object* v___x_622_; 
v_a_619_ = lean_ctor_get(v___x_618_, 0);
lean_inc(v_a_619_);
lean_dec_ref_known(v___x_618_, 1);
v___x_620_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__9));
if (v_isShared_586_ == 0)
{
lean_ctor_set_tag(v___x_585_, 1);
lean_ctor_set(v___x_585_, 1, v_us_596_);
lean_ctor_set(v___x_585_, 0, v_a_619_);
v___x_622_ = v___x_585_;
goto v_reusejp_621_;
}
else
{
lean_object* v_reuseFailAlloc_645_; 
v_reuseFailAlloc_645_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_645_, 0, v_a_619_);
lean_ctor_set(v_reuseFailAlloc_645_, 1, v_us_596_);
v___x_622_ = v_reuseFailAlloc_645_;
goto v_reusejp_621_;
}
v_reusejp_621_:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_623_ = l_Lean_Expr_const___override(v___x_620_, v___x_622_);
v___x_624_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__11));
v___x_625_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13));
v___x_626_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15));
v___x_627_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_623_, v___x_599_, v___x_601_, v___x_624_, v___x_625_, v___x_626_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v_a_628_; lean_object* v___x_630_; uint8_t v_isShared_631_; uint8_t v_isSharedCheck_636_; 
v_a_628_ = lean_ctor_get(v___x_627_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_636_ == 0)
{
v___x_630_ = v___x_627_;
v_isShared_631_ = v_isSharedCheck_636_;
goto v_resetjp_629_;
}
else
{
lean_inc(v_a_628_);
lean_dec(v___x_627_);
v___x_630_ = lean_box(0);
v_isShared_631_ = v_isSharedCheck_636_;
goto v_resetjp_629_;
}
v_resetjp_629_:
{
lean_object* v___x_632_; lean_object* v___x_634_; 
v___x_632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_632_, 0, v_a_628_);
if (v_isShared_631_ == 0)
{
lean_ctor_set(v___x_630_, 0, v___x_632_);
v___x_634_ = v___x_630_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v___x_632_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
else
{
lean_object* v_a_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_644_; 
v_a_637_ = lean_ctor_get(v___x_627_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_644_ == 0)
{
v___x_639_ = v___x_627_;
v_isShared_640_ = v_isSharedCheck_644_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_a_637_);
lean_dec(v___x_627_);
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
}
}
else
{
lean_object* v_a_646_; lean_object* v___x_648_; uint8_t v_isShared_649_; uint8_t v_isSharedCheck_653_; 
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_dec(v_us_596_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_646_ = lean_ctor_get(v___x_618_, 0);
v_isSharedCheck_653_ = !lean_is_exclusive(v___x_618_);
if (v_isSharedCheck_653_ == 0)
{
v___x_648_ = v___x_618_;
v_isShared_649_ = v_isSharedCheck_653_;
goto v_resetjp_647_;
}
else
{
lean_inc(v_a_646_);
lean_dec(v___x_618_);
v___x_648_ = lean_box(0);
v_isShared_649_ = v_isSharedCheck_653_;
goto v_resetjp_647_;
}
v_resetjp_647_:
{
lean_object* v___x_651_; 
if (v_isShared_649_ == 0)
{
v___x_651_ = v___x_648_;
goto v_reusejp_650_;
}
else
{
lean_object* v_reuseFailAlloc_652_; 
v_reuseFailAlloc_652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_652_, 0, v_a_646_);
v___x_651_ = v_reuseFailAlloc_652_;
goto v_reusejp_650_;
}
v_reusejp_650_:
{
return v___x_651_;
}
}
}
}
}
else
{
lean_object* v___x_654_; 
lean_dec_ref(v_str_597_);
v___x_654_ = l_Lean_Meta_mkFreshLevelMVar(v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_654_) == 0)
{
lean_object* v_a_655_; lean_object* v___x_656_; lean_object* v___x_658_; 
v_a_655_ = lean_ctor_get(v___x_654_, 0);
lean_inc(v_a_655_);
lean_dec_ref_known(v___x_654_, 1);
v___x_656_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__16));
if (v_isShared_586_ == 0)
{
lean_ctor_set_tag(v___x_585_, 1);
lean_ctor_set(v___x_585_, 1, v_us_596_);
lean_ctor_set(v___x_585_, 0, v_a_655_);
v___x_658_ = v___x_585_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_681_; 
v_reuseFailAlloc_681_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_681_, 0, v_a_655_);
lean_ctor_set(v_reuseFailAlloc_681_, 1, v_us_596_);
v___x_658_ = v_reuseFailAlloc_681_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_659_ = l_Lean_Expr_const___override(v___x_656_, v___x_658_);
v___x_660_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__17));
v___x_661_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13));
v___x_662_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15));
v___x_663_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_659_, v___x_599_, v___x_601_, v___x_660_, v___x_661_, v___x_662_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_663_) == 0)
{
lean_object* v_a_664_; lean_object* v___x_666_; uint8_t v_isShared_667_; uint8_t v_isSharedCheck_672_; 
v_a_664_ = lean_ctor_get(v___x_663_, 0);
v_isSharedCheck_672_ = !lean_is_exclusive(v___x_663_);
if (v_isSharedCheck_672_ == 0)
{
v___x_666_ = v___x_663_;
v_isShared_667_ = v_isSharedCheck_672_;
goto v_resetjp_665_;
}
else
{
lean_inc(v_a_664_);
lean_dec(v___x_663_);
v___x_666_ = lean_box(0);
v_isShared_667_ = v_isSharedCheck_672_;
goto v_resetjp_665_;
}
v_resetjp_665_:
{
lean_object* v___x_668_; lean_object* v___x_670_; 
v___x_668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_668_, 0, v_a_664_);
if (v_isShared_667_ == 0)
{
lean_ctor_set(v___x_666_, 0, v___x_668_);
v___x_670_ = v___x_666_;
goto v_reusejp_669_;
}
else
{
lean_object* v_reuseFailAlloc_671_; 
v_reuseFailAlloc_671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_671_, 0, v___x_668_);
v___x_670_ = v_reuseFailAlloc_671_;
goto v_reusejp_669_;
}
v_reusejp_669_:
{
return v___x_670_;
}
}
}
else
{
lean_object* v_a_673_; lean_object* v___x_675_; uint8_t v_isShared_676_; uint8_t v_isSharedCheck_680_; 
v_a_673_ = lean_ctor_get(v___x_663_, 0);
v_isSharedCheck_680_ = !lean_is_exclusive(v___x_663_);
if (v_isSharedCheck_680_ == 0)
{
v___x_675_ = v___x_663_;
v_isShared_676_ = v_isSharedCheck_680_;
goto v_resetjp_674_;
}
else
{
lean_inc(v_a_673_);
lean_dec(v___x_663_);
v___x_675_ = lean_box(0);
v_isShared_676_ = v_isSharedCheck_680_;
goto v_resetjp_674_;
}
v_resetjp_674_:
{
lean_object* v___x_678_; 
if (v_isShared_676_ == 0)
{
v___x_678_ = v___x_675_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_679_; 
v_reuseFailAlloc_679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_679_, 0, v_a_673_);
v___x_678_ = v_reuseFailAlloc_679_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
return v___x_678_;
}
}
}
}
}
else
{
lean_object* v_a_682_; lean_object* v___x_684_; uint8_t v_isShared_685_; uint8_t v_isSharedCheck_689_; 
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_dec(v_us_596_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_682_ = lean_ctor_get(v___x_654_, 0);
v_isSharedCheck_689_ = !lean_is_exclusive(v___x_654_);
if (v_isSharedCheck_689_ == 0)
{
v___x_684_ = v___x_654_;
v_isShared_685_ = v_isSharedCheck_689_;
goto v_resetjp_683_;
}
else
{
lean_inc(v_a_682_);
lean_dec(v___x_654_);
v___x_684_ = lean_box(0);
v_isShared_685_ = v_isSharedCheck_689_;
goto v_resetjp_683_;
}
v_resetjp_683_:
{
lean_object* v___x_687_; 
if (v_isShared_685_ == 0)
{
v___x_687_ = v___x_684_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v_a_682_);
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
else
{
lean_object* v___x_690_; 
lean_dec_ref(v_str_597_);
v___x_690_ = l_Lean_Meta_mkFreshLevelMVar(v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_690_) == 0)
{
lean_object* v_a_691_; lean_object* v___x_692_; lean_object* v___x_694_; 
v_a_691_ = lean_ctor_get(v___x_690_, 0);
lean_inc(v_a_691_);
lean_dec_ref_known(v___x_690_, 1);
v___x_692_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__18));
if (v_isShared_586_ == 0)
{
lean_ctor_set_tag(v___x_585_, 1);
lean_ctor_set(v___x_585_, 1, v_us_596_);
lean_ctor_set(v___x_585_, 0, v_a_691_);
v___x_694_ = v___x_585_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_a_691_);
lean_ctor_set(v_reuseFailAlloc_717_, 1, v_us_596_);
v___x_694_ = v_reuseFailAlloc_717_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; 
v___x_695_ = l_Lean_Expr_const___override(v___x_692_, v___x_694_);
v___x_696_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__19));
v___x_697_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__21));
v___x_698_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__23));
v___x_699_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_695_, v___x_599_, v___x_601_, v___x_696_, v___x_697_, v___x_698_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_699_) == 0)
{
lean_object* v_a_700_; lean_object* v___x_702_; uint8_t v_isShared_703_; uint8_t v_isSharedCheck_708_; 
v_a_700_ = lean_ctor_get(v___x_699_, 0);
v_isSharedCheck_708_ = !lean_is_exclusive(v___x_699_);
if (v_isSharedCheck_708_ == 0)
{
v___x_702_ = v___x_699_;
v_isShared_703_ = v_isSharedCheck_708_;
goto v_resetjp_701_;
}
else
{
lean_inc(v_a_700_);
lean_dec(v___x_699_);
v___x_702_ = lean_box(0);
v_isShared_703_ = v_isSharedCheck_708_;
goto v_resetjp_701_;
}
v_resetjp_701_:
{
lean_object* v___x_704_; lean_object* v___x_706_; 
v___x_704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_704_, 0, v_a_700_);
if (v_isShared_703_ == 0)
{
lean_ctor_set(v___x_702_, 0, v___x_704_);
v___x_706_ = v___x_702_;
goto v_reusejp_705_;
}
else
{
lean_object* v_reuseFailAlloc_707_; 
v_reuseFailAlloc_707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_707_, 0, v___x_704_);
v___x_706_ = v_reuseFailAlloc_707_;
goto v_reusejp_705_;
}
v_reusejp_705_:
{
return v___x_706_;
}
}
}
else
{
lean_object* v_a_709_; lean_object* v___x_711_; uint8_t v_isShared_712_; uint8_t v_isSharedCheck_716_; 
v_a_709_ = lean_ctor_get(v___x_699_, 0);
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_699_);
if (v_isSharedCheck_716_ == 0)
{
v___x_711_ = v___x_699_;
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
else
{
lean_inc(v_a_709_);
lean_dec(v___x_699_);
v___x_711_ = lean_box(0);
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
v_resetjp_710_:
{
lean_object* v___x_714_; 
if (v_isShared_712_ == 0)
{
v___x_714_ = v___x_711_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_a_709_);
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
else
{
lean_object* v_a_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_725_; 
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_dec(v_us_596_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_718_ = lean_ctor_get(v___x_690_, 0);
v_isSharedCheck_725_ = !lean_is_exclusive(v___x_690_);
if (v_isSharedCheck_725_ == 0)
{
v___x_720_ = v___x_690_;
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_a_718_);
lean_dec(v___x_690_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_725_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_723_; 
if (v_isShared_721_ == 0)
{
v___x_723_ = v___x_720_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v_a_718_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
}
}
}
}
}
else
{
lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; 
lean_dec_ref(v_str_597_);
lean_del_object(v___x_585_);
v___x_726_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__24));
v___x_727_ = l_Lean_Expr_const___override(v___x_726_, v_us_596_);
v___x_728_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__26));
v___x_729_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__28));
v___x_730_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac___closed__1));
v___x_731_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_727_, v___x_599_, v___x_601_, v___x_728_, v___x_729_, v___x_730_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_731_) == 0)
{
lean_object* v_a_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_740_; 
v_a_732_ = lean_ctor_get(v___x_731_, 0);
v_isSharedCheck_740_ = !lean_is_exclusive(v___x_731_);
if (v_isSharedCheck_740_ == 0)
{
v___x_734_ = v___x_731_;
v_isShared_735_ = v_isSharedCheck_740_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_a_732_);
lean_dec(v___x_731_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_740_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_736_; lean_object* v___x_738_; 
v___x_736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_736_, 0, v_a_732_);
if (v_isShared_735_ == 0)
{
lean_ctor_set(v___x_734_, 0, v___x_736_);
v___x_738_ = v___x_734_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v___x_736_);
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
lean_object* v_a_741_; lean_object* v___x_743_; uint8_t v_isShared_744_; uint8_t v_isSharedCheck_748_; 
v_a_741_ = lean_ctor_get(v___x_731_, 0);
v_isSharedCheck_748_ = !lean_is_exclusive(v___x_731_);
if (v_isSharedCheck_748_ == 0)
{
v___x_743_ = v___x_731_;
v_isShared_744_ = v_isSharedCheck_748_;
goto v_resetjp_742_;
}
else
{
lean_inc(v_a_741_);
lean_dec(v___x_731_);
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
else
{
lean_object* v___x_749_; 
lean_dec_ref(v_str_597_);
v___x_749_ = l_Lean_Meta_mkFreshLevelMVar(v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_749_) == 0)
{
lean_object* v_a_750_; lean_object* v___x_751_; lean_object* v___x_753_; 
v_a_750_ = lean_ctor_get(v___x_749_, 0);
lean_inc(v_a_750_);
lean_dec_ref_known(v___x_749_, 1);
v___x_751_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__29));
if (v_isShared_586_ == 0)
{
lean_ctor_set_tag(v___x_585_, 1);
lean_ctor_set(v___x_585_, 1, v_us_596_);
lean_ctor_set(v___x_585_, 0, v_a_750_);
v___x_753_ = v___x_585_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_a_750_);
lean_ctor_set(v_reuseFailAlloc_776_, 1, v_us_596_);
v___x_753_ = v_reuseFailAlloc_776_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; 
v___x_754_ = l_Lean_Expr_const___override(v___x_751_, v___x_753_);
v___x_755_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__30));
v___x_756_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13));
v___x_757_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15));
v___x_758_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_754_, v___x_599_, v___x_601_, v___x_755_, v___x_756_, v___x_757_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_758_) == 0)
{
lean_object* v_a_759_; lean_object* v___x_761_; uint8_t v_isShared_762_; uint8_t v_isSharedCheck_767_; 
v_a_759_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_767_ == 0)
{
v___x_761_ = v___x_758_;
v_isShared_762_ = v_isSharedCheck_767_;
goto v_resetjp_760_;
}
else
{
lean_inc(v_a_759_);
lean_dec(v___x_758_);
v___x_761_ = lean_box(0);
v_isShared_762_ = v_isSharedCheck_767_;
goto v_resetjp_760_;
}
v_resetjp_760_:
{
lean_object* v___x_763_; lean_object* v___x_765_; 
v___x_763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_763_, 0, v_a_759_);
if (v_isShared_762_ == 0)
{
lean_ctor_set(v___x_761_, 0, v___x_763_);
v___x_765_ = v___x_761_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v___x_763_);
v___x_765_ = v_reuseFailAlloc_766_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
return v___x_765_;
}
}
}
else
{
lean_object* v_a_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_775_; 
v_a_768_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_775_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_775_ == 0)
{
v___x_770_ = v___x_758_;
v_isShared_771_ = v_isSharedCheck_775_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_a_768_);
lean_dec(v___x_758_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_775_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_773_; 
if (v_isShared_771_ == 0)
{
v___x_773_ = v___x_770_;
goto v_reusejp_772_;
}
else
{
lean_object* v_reuseFailAlloc_774_; 
v_reuseFailAlloc_774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_774_, 0, v_a_768_);
v___x_773_ = v_reuseFailAlloc_774_;
goto v_reusejp_772_;
}
v_reusejp_772_:
{
return v___x_773_;
}
}
}
}
}
else
{
lean_object* v_a_777_; lean_object* v___x_779_; uint8_t v_isShared_780_; uint8_t v_isSharedCheck_784_; 
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_dec(v_us_596_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_777_ = lean_ctor_get(v___x_749_, 0);
v_isSharedCheck_784_ = !lean_is_exclusive(v___x_749_);
if (v_isSharedCheck_784_ == 0)
{
v___x_779_ = v___x_749_;
v_isShared_780_ = v_isSharedCheck_784_;
goto v_resetjp_778_;
}
else
{
lean_inc(v_a_777_);
lean_dec(v___x_749_);
v___x_779_ = lean_box(0);
v_isShared_780_ = v_isSharedCheck_784_;
goto v_resetjp_778_;
}
v_resetjp_778_:
{
lean_object* v___x_782_; 
if (v_isShared_780_ == 0)
{
v___x_782_ = v___x_779_;
goto v_reusejp_781_;
}
else
{
lean_object* v_reuseFailAlloc_783_; 
v_reuseFailAlloc_783_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_783_, 0, v_a_777_);
v___x_782_ = v_reuseFailAlloc_783_;
goto v_reusejp_781_;
}
v_reusejp_781_:
{
return v___x_782_;
}
}
}
}
}
else
{
lean_object* v___x_785_; 
lean_dec_ref(v_str_597_);
v___x_785_ = l_Lean_Meta_mkFreshLevelMVar(v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_785_) == 0)
{
lean_object* v_a_786_; lean_object* v___x_787_; lean_object* v___x_789_; 
v_a_786_ = lean_ctor_get(v___x_785_, 0);
lean_inc(v_a_786_);
lean_dec_ref_known(v___x_785_, 1);
v___x_787_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__31));
if (v_isShared_586_ == 0)
{
lean_ctor_set_tag(v___x_585_, 1);
lean_ctor_set(v___x_585_, 1, v_us_596_);
lean_ctor_set(v___x_585_, 0, v_a_786_);
v___x_789_ = v___x_585_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v_a_786_);
lean_ctor_set(v_reuseFailAlloc_812_, 1, v_us_596_);
v___x_789_ = v_reuseFailAlloc_812_;
goto v_reusejp_788_;
}
v_reusejp_788_:
{
lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; 
v___x_790_ = l_Lean_Expr_const___override(v___x_787_, v___x_789_);
v___x_791_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__32));
v___x_792_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13));
v___x_793_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15));
v___x_794_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_790_, v___x_599_, v___x_601_, v___x_791_, v___x_792_, v___x_793_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_794_) == 0)
{
lean_object* v_a_795_; lean_object* v___x_797_; uint8_t v_isShared_798_; uint8_t v_isSharedCheck_803_; 
v_a_795_ = lean_ctor_get(v___x_794_, 0);
v_isSharedCheck_803_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_803_ == 0)
{
v___x_797_ = v___x_794_;
v_isShared_798_ = v_isSharedCheck_803_;
goto v_resetjp_796_;
}
else
{
lean_inc(v_a_795_);
lean_dec(v___x_794_);
v___x_797_ = lean_box(0);
v_isShared_798_ = v_isSharedCheck_803_;
goto v_resetjp_796_;
}
v_resetjp_796_:
{
lean_object* v___x_799_; lean_object* v___x_801_; 
v___x_799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_799_, 0, v_a_795_);
if (v_isShared_798_ == 0)
{
lean_ctor_set(v___x_797_, 0, v___x_799_);
v___x_801_ = v___x_797_;
goto v_reusejp_800_;
}
else
{
lean_object* v_reuseFailAlloc_802_; 
v_reuseFailAlloc_802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_802_, 0, v___x_799_);
v___x_801_ = v_reuseFailAlloc_802_;
goto v_reusejp_800_;
}
v_reusejp_800_:
{
return v___x_801_;
}
}
}
else
{
lean_object* v_a_804_; lean_object* v___x_806_; uint8_t v_isShared_807_; uint8_t v_isSharedCheck_811_; 
v_a_804_ = lean_ctor_get(v___x_794_, 0);
v_isSharedCheck_811_ = !lean_is_exclusive(v___x_794_);
if (v_isSharedCheck_811_ == 0)
{
v___x_806_ = v___x_794_;
v_isShared_807_ = v_isSharedCheck_811_;
goto v_resetjp_805_;
}
else
{
lean_inc(v_a_804_);
lean_dec(v___x_794_);
v___x_806_ = lean_box(0);
v_isShared_807_ = v_isSharedCheck_811_;
goto v_resetjp_805_;
}
v_resetjp_805_:
{
lean_object* v___x_809_; 
if (v_isShared_807_ == 0)
{
v___x_809_ = v___x_806_;
goto v_reusejp_808_;
}
else
{
lean_object* v_reuseFailAlloc_810_; 
v_reuseFailAlloc_810_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_810_, 0, v_a_804_);
v___x_809_ = v_reuseFailAlloc_810_;
goto v_reusejp_808_;
}
v_reusejp_808_:
{
return v___x_809_;
}
}
}
}
}
else
{
lean_object* v_a_813_; lean_object* v___x_815_; uint8_t v_isShared_816_; uint8_t v_isSharedCheck_820_; 
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_dec(v_us_596_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_813_ = lean_ctor_get(v___x_785_, 0);
v_isSharedCheck_820_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_820_ == 0)
{
v___x_815_ = v___x_785_;
v_isShared_816_ = v_isSharedCheck_820_;
goto v_resetjp_814_;
}
else
{
lean_inc(v_a_813_);
lean_dec(v___x_785_);
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
else
{
lean_object* v___x_821_; 
lean_dec_ref(v_str_597_);
v___x_821_ = l_Lean_Meta_mkFreshLevelMVar(v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_821_) == 0)
{
lean_object* v_a_822_; lean_object* v___x_823_; lean_object* v___x_825_; 
v_a_822_ = lean_ctor_get(v___x_821_, 0);
lean_inc(v_a_822_);
lean_dec_ref_known(v___x_821_, 1);
v___x_823_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__33));
if (v_isShared_586_ == 0)
{
lean_ctor_set_tag(v___x_585_, 1);
lean_ctor_set(v___x_585_, 1, v_us_596_);
lean_ctor_set(v___x_585_, 0, v_a_822_);
v___x_825_ = v___x_585_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_848_; 
v_reuseFailAlloc_848_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_848_, 0, v_a_822_);
lean_ctor_set(v_reuseFailAlloc_848_, 1, v_us_596_);
v___x_825_ = v_reuseFailAlloc_848_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_826_ = l_Lean_Expr_const___override(v___x_823_, v___x_825_);
v___x_827_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__34));
v___x_828_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__13));
v___x_829_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__15));
v___x_830_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_826_, v___x_599_, v___x_601_, v___x_827_, v___x_828_, v___x_829_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_830_) == 0)
{
lean_object* v_a_831_; lean_object* v___x_833_; uint8_t v_isShared_834_; uint8_t v_isSharedCheck_839_; 
v_a_831_ = lean_ctor_get(v___x_830_, 0);
v_isSharedCheck_839_ = !lean_is_exclusive(v___x_830_);
if (v_isSharedCheck_839_ == 0)
{
v___x_833_ = v___x_830_;
v_isShared_834_ = v_isSharedCheck_839_;
goto v_resetjp_832_;
}
else
{
lean_inc(v_a_831_);
lean_dec(v___x_830_);
v___x_833_ = lean_box(0);
v_isShared_834_ = v_isSharedCheck_839_;
goto v_resetjp_832_;
}
v_resetjp_832_:
{
lean_object* v___x_835_; lean_object* v___x_837_; 
v___x_835_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_835_, 0, v_a_831_);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 0, v___x_835_);
v___x_837_ = v___x_833_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_838_; 
v_reuseFailAlloc_838_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_838_, 0, v___x_835_);
v___x_837_ = v_reuseFailAlloc_838_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
return v___x_837_;
}
}
}
else
{
lean_object* v_a_840_; lean_object* v___x_842_; uint8_t v_isShared_843_; uint8_t v_isSharedCheck_847_; 
v_a_840_ = lean_ctor_get(v___x_830_, 0);
v_isSharedCheck_847_ = !lean_is_exclusive(v___x_830_);
if (v_isSharedCheck_847_ == 0)
{
v___x_842_ = v___x_830_;
v_isShared_843_ = v_isSharedCheck_847_;
goto v_resetjp_841_;
}
else
{
lean_inc(v_a_840_);
lean_dec(v___x_830_);
v___x_842_ = lean_box(0);
v_isShared_843_ = v_isSharedCheck_847_;
goto v_resetjp_841_;
}
v_resetjp_841_:
{
lean_object* v___x_845_; 
if (v_isShared_843_ == 0)
{
v___x_845_ = v___x_842_;
goto v_reusejp_844_;
}
else
{
lean_object* v_reuseFailAlloc_846_; 
v_reuseFailAlloc_846_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_846_, 0, v_a_840_);
v___x_845_ = v_reuseFailAlloc_846_;
goto v_reusejp_844_;
}
v_reusejp_844_:
{
return v___x_845_;
}
}
}
}
}
else
{
lean_object* v_a_849_; lean_object* v___x_851_; uint8_t v_isShared_852_; uint8_t v_isSharedCheck_856_; 
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_dec(v_us_596_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_849_ = lean_ctor_get(v___x_821_, 0);
v_isSharedCheck_856_ = !lean_is_exclusive(v___x_821_);
if (v_isSharedCheck_856_ == 0)
{
v___x_851_ = v___x_821_;
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
else
{
lean_inc(v_a_849_);
lean_dec(v___x_821_);
v___x_851_ = lean_box(0);
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
v_resetjp_850_:
{
lean_object* v___x_854_; 
if (v_isShared_852_ == 0)
{
v___x_854_ = v___x_851_;
goto v_reusejp_853_;
}
else
{
lean_object* v_reuseFailAlloc_855_; 
v_reuseFailAlloc_855_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_855_, 0, v_a_849_);
v___x_854_ = v_reuseFailAlloc_855_;
goto v_reusejp_853_;
}
v_reusejp_853_:
{
return v___x_854_;
}
}
}
}
}
else
{
lean_object* v___x_857_; 
lean_dec_ref(v_str_597_);
lean_dec(v_us_596_);
v___x_857_ = l_Lean_Meta_mkFreshLevelMVar(v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_857_) == 0)
{
lean_object* v_a_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_862_; 
v_a_858_ = lean_ctor_get(v___x_857_, 0);
lean_inc(v_a_858_);
lean_dec_ref_known(v___x_857_, 1);
v___x_859_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__35));
v___x_860_ = lean_box(0);
if (v_isShared_586_ == 0)
{
lean_ctor_set_tag(v___x_585_, 1);
lean_ctor_set(v___x_585_, 1, v___x_860_);
lean_ctor_set(v___x_585_, 0, v_a_858_);
v___x_862_ = v___x_585_;
goto v_reusejp_861_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v_a_858_);
lean_ctor_set(v_reuseFailAlloc_885_, 1, v___x_860_);
v___x_862_ = v_reuseFailAlloc_885_;
goto v_reusejp_861_;
}
v_reusejp_861_:
{
lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
v___x_863_ = l_Lean_Expr_const___override(v___x_859_, v___x_862_);
v___x_864_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__36));
v___x_865_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__38));
v___x_866_ = ((lean_object*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___closed__40));
v___x_867_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_go(v_goal_553_, v_hyp_551_, v_a_563_, v___x_863_, v___x_599_, v___x_601_, v___x_864_, v___x_865_, v___x_866_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
lean_dec_ref(v___y_554_);
if (lean_obj_tag(v___x_867_) == 0)
{
lean_object* v_a_868_; lean_object* v___x_870_; uint8_t v_isShared_871_; uint8_t v_isSharedCheck_876_; 
v_a_868_ = lean_ctor_get(v___x_867_, 0);
v_isSharedCheck_876_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_876_ == 0)
{
v___x_870_ = v___x_867_;
v_isShared_871_ = v_isSharedCheck_876_;
goto v_resetjp_869_;
}
else
{
lean_inc(v_a_868_);
lean_dec(v___x_867_);
v___x_870_ = lean_box(0);
v_isShared_871_ = v_isSharedCheck_876_;
goto v_resetjp_869_;
}
v_resetjp_869_:
{
lean_object* v___x_872_; lean_object* v___x_874_; 
v___x_872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_872_, 0, v_a_868_);
if (v_isShared_871_ == 0)
{
lean_ctor_set(v___x_870_, 0, v___x_872_);
v___x_874_ = v___x_870_;
goto v_reusejp_873_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v___x_872_);
v___x_874_ = v_reuseFailAlloc_875_;
goto v_reusejp_873_;
}
v_reusejp_873_:
{
return v___x_874_;
}
}
}
else
{
lean_object* v_a_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_884_; 
v_a_877_ = lean_ctor_get(v___x_867_, 0);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_867_);
if (v_isSharedCheck_884_ == 0)
{
v___x_879_ = v___x_867_;
v_isShared_880_ = v_isSharedCheck_884_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_a_877_);
lean_dec(v___x_867_);
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
}
else
{
lean_object* v_a_886_; lean_object* v___x_888_; uint8_t v_isShared_889_; uint8_t v_isSharedCheck_893_; 
lean_dec(v___x_601_);
lean_dec(v___x_599_);
lean_del_object(v___x_585_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_886_ = lean_ctor_get(v___x_857_, 0);
v_isSharedCheck_893_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_893_ == 0)
{
v___x_888_ = v___x_857_;
v_isShared_889_ = v_isSharedCheck_893_;
goto v_resetjp_887_;
}
else
{
lean_inc(v_a_886_);
lean_dec(v___x_857_);
v___x_888_ = lean_box(0);
v_isShared_889_ = v_isSharedCheck_893_;
goto v_resetjp_887_;
}
v_resetjp_887_:
{
lean_object* v___x_891_; 
if (v_isShared_889_ == 0)
{
v___x_891_ = v___x_888_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v_a_886_);
v___x_891_ = v_reuseFailAlloc_892_;
goto v_reusejp_890_;
}
v_reusejp_890_:
{
return v___x_891_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_declName_594_, 2);
lean_dec_ref_known(v_fst_582_, 2);
lean_del_object(v___x_585_);
lean_dec(v_snd_583_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
goto v___jp_559_;
}
}
else
{
lean_dec_ref_known(v_fst_582_, 2);
lean_dec(v_declName_594_);
lean_del_object(v___x_585_);
lean_dec(v_snd_583_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
goto v___jp_559_;
}
}
else
{
lean_del_object(v___x_585_);
lean_dec(v_snd_583_);
lean_dec(v_fst_582_);
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
goto v___jp_559_;
}
}
}
}
}
else
{
lean_object* v_a_896_; lean_object* v___x_898_; uint8_t v_isShared_899_; uint8_t v_isSharedCheck_903_; 
lean_dec(v_a_563_);
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_896_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_903_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_903_ == 0)
{
v___x_898_ = v___x_577_;
v_isShared_899_ = v_isSharedCheck_903_;
goto v_resetjp_897_;
}
else
{
lean_inc(v_a_896_);
lean_dec(v___x_577_);
v___x_898_ = lean_box(0);
v_isShared_899_ = v_isSharedCheck_903_;
goto v_resetjp_897_;
}
v_resetjp_897_:
{
lean_object* v___x_901_; 
if (v_isShared_899_ == 0)
{
v___x_901_ = v___x_898_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_902_; 
v_reuseFailAlloc_902_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_902_, 0, v_a_896_);
v___x_901_ = v_reuseFailAlloc_902_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
return v___x_901_;
}
}
}
}
else
{
lean_object* v_a_904_; lean_object* v___x_906_; uint8_t v_isShared_907_; uint8_t v_isSharedCheck_911_; 
lean_dec_ref(v___y_554_);
lean_dec(v_goal_553_);
lean_dec(v_hyp_551_);
v_a_904_ = lean_ctor_get(v___x_562_, 0);
v_isSharedCheck_911_ = !lean_is_exclusive(v___x_562_);
if (v_isSharedCheck_911_ == 0)
{
v___x_906_ = v___x_562_;
v_isShared_907_ = v_isSharedCheck_911_;
goto v_resetjp_905_;
}
else
{
lean_inc(v_a_904_);
lean_dec(v___x_562_);
v___x_906_ = lean_box(0);
v_isShared_907_ = v_isSharedCheck_911_;
goto v_resetjp_905_;
}
v_resetjp_905_:
{
lean_object* v___x_909_; 
if (v_isShared_907_ == 0)
{
v___x_909_ = v___x_906_;
goto v_reusejp_908_;
}
else
{
lean_object* v_reuseFailAlloc_910_; 
v_reuseFailAlloc_910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_910_, 0, v_a_904_);
v___x_909_ = v_reuseFailAlloc_910_;
goto v_reusejp_908_;
}
v_reusejp_908_:
{
return v___x_909_;
}
}
}
v___jp_559_:
{
lean_object* v___x_560_; lean_object* v___x_561_; 
v___x_560_ = lean_box(0);
v___x_561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_561_, 0, v___x_560_);
return v___x_561_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___boxed(lean_object* v_hyp_912_, lean_object* v_md_913_, lean_object* v_goal_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_){
_start:
{
uint8_t v_md_boxed_920_; lean_object* v_res_921_; 
v_md_boxed_920_ = lean_unbox(v_md_913_);
v_res_921_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0(v_hyp_912_, v_md_boxed_920_, v_goal_914_, v___y_915_, v___y_916_, v___y_917_, v___y_918_);
lean_dec(v___y_918_);
lean_dec_ref(v___y_917_);
lean_dec(v___y_916_);
return v_res_921_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f(lean_object* v_goal_922_, lean_object* v_hyp_923_, uint8_t v_md_924_, lean_object* v_a_925_, lean_object* v_a_926_, lean_object* v_a_927_, lean_object* v_a_928_){
_start:
{
lean_object* v___x_930_; lean_object* v___f_931_; lean_object* v___x_932_; 
v___x_930_ = lean_box(v_md_924_);
lean_inc(v_goal_922_);
v___f_931_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___lam__0___boxed), 8, 3);
lean_closure_set(v___f_931_, 0, v_hyp_923_);
lean_closure_set(v___f_931_, 1, v___x_930_);
lean_closure_set(v___f_931_, 2, v_goal_922_);
v___x_932_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__2___redArg(v_goal_922_, v___f_931_, v_a_925_, v_a_926_, v_a_927_, v_a_928_);
return v___x_932_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f___boxed(lean_object* v_goal_933_, lean_object* v_hyp_934_, lean_object* v_md_935_, lean_object* v_a_936_, lean_object* v_a_937_, lean_object* v_a_938_, lean_object* v_a_939_, lean_object* v_a_940_){
_start:
{
uint8_t v_md_boxed_941_; lean_object* v_res_942_; 
v_md_boxed_941_ = lean_unbox(v_md_935_);
v_res_942_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f(v_goal_933_, v_hyp_934_, v_md_boxed_941_, v_a_936_, v_a_937_, v_a_938_, v_a_939_);
lean_dec(v_a_939_);
lean_dec_ref(v_a_938_);
lean_dec(v_a_937_);
lean_dec_ref(v_a_936_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___redArg(lean_object* v_step_943_, lean_object* v___y_944_){
_start:
{
lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; 
v___x_946_ = lean_st_ref_take(v___y_944_);
v___x_947_ = lean_array_push(v___x_946_, v_step_943_);
v___x_948_ = lean_st_ref_set(v___y_944_, v___x_947_);
v___x_949_ = lean_box(0);
v___x_950_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_950_, 0, v___x_949_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___redArg___boxed(lean_object* v_step_951_, lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
lean_object* v_res_954_; 
v_res_954_ = lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___redArg(v_step_951_, v___y_952_);
lean_dec(v___y_952_);
return v_res_954_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0(lean_object* v_step_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_){
_start:
{
lean_object* v___x_963_; 
v___x_963_ = lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___redArg(v_step_955_, v___y_956_);
return v___x_963_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___boxed(lean_object* v_step_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_){
_start:
{
lean_object* v_res_972_; 
v_res_972_ = lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0(v_step_964_, v___y_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_);
lean_dec(v___y_970_);
lean_dec_ref(v___y_969_);
lean_dec(v___y_968_);
lean_dec_ref(v___y_967_);
lean_dec(v___y_966_);
lean_dec(v___y_965_);
return v_res_972_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___lam__0(lean_object* v_x_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_){
_start:
{
lean_object* v___x_981_; 
lean_inc(v___y_975_);
lean_inc(v___y_974_);
v___x_981_ = lean_apply_7(v_x_973_, v___y_974_, v___y_975_, v___y_976_, v___y_977_, v___y_978_, v___y_979_, lean_box(0));
return v___x_981_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___lam__0___boxed(lean_object* v_x_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
lean_object* v_res_990_; 
v_res_990_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___lam__0(v_x_982_, v___y_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_, v___y_988_);
lean_dec(v___y_984_);
lean_dec(v___y_983_);
return v_res_990_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg(lean_object* v_mvarId_991_, lean_object* v_x_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_){
_start:
{
lean_object* v___f_1000_; lean_object* v___x_1001_; 
lean_inc(v___y_994_);
lean_inc(v___y_993_);
v___f_1000_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_1000_, 0, v_x_992_);
lean_closure_set(v___f_1000_, 1, v___y_993_);
lean_closure_set(v___f_1000_, 2, v___y_994_);
v___x_1001_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_991_, v___f_1000_, v___y_995_, v___y_996_, v___y_997_, v___y_998_);
if (lean_obj_tag(v___x_1001_) == 0)
{
return v___x_1001_;
}
else
{
lean_object* v_a_1002_; lean_object* v___x_1004_; uint8_t v_isShared_1005_; uint8_t v_isSharedCheck_1009_; 
v_a_1002_ = lean_ctor_get(v___x_1001_, 0);
v_isSharedCheck_1009_ = !lean_is_exclusive(v___x_1001_);
if (v_isSharedCheck_1009_ == 0)
{
v___x_1004_ = v___x_1001_;
v_isShared_1005_ = v_isSharedCheck_1009_;
goto v_resetjp_1003_;
}
else
{
lean_inc(v_a_1002_);
lean_dec(v___x_1001_);
v___x_1004_ = lean_box(0);
v_isShared_1005_ = v_isSharedCheck_1009_;
goto v_resetjp_1003_;
}
v_resetjp_1003_:
{
lean_object* v___x_1007_; 
if (v_isShared_1005_ == 0)
{
v___x_1007_ = v___x_1004_;
goto v_reusejp_1006_;
}
else
{
lean_object* v_reuseFailAlloc_1008_; 
v_reuseFailAlloc_1008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1008_, 0, v_a_1002_);
v___x_1007_ = v_reuseFailAlloc_1008_;
goto v_reusejp_1006_;
}
v_reusejp_1006_:
{
return v___x_1007_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg___boxed(lean_object* v_mvarId_1010_, lean_object* v_x_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_){
_start:
{
lean_object* v_res_1019_; 
v_res_1019_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg(v_mvarId_1010_, v_x_1011_, v___y_1012_, v___y_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
lean_dec(v___y_1013_);
lean_dec(v___y_1012_);
return v_res_1019_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1(lean_object* v_00_u03b1_1020_, lean_object* v_mvarId_1021_, lean_object* v_x_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_){
_start:
{
lean_object* v___x_1030_; 
v___x_1030_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg(v_mvarId_1021_, v_x_1022_, v___y_1023_, v___y_1024_, v___y_1025_, v___y_1026_, v___y_1027_, v___y_1028_);
return v___x_1030_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___boxed(lean_object* v_00_u03b1_1031_, lean_object* v_mvarId_1032_, lean_object* v_x_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_){
_start:
{
lean_object* v_res_1041_; 
v_res_1041_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1(v_00_u03b1_1031_, v_mvarId_1032_, v_x_1033_, v___y_1034_, v___y_1035_, v___y_1036_, v___y_1037_, v___y_1038_, v___y_1039_);
lean_dec(v___y_1039_);
lean_dec_ref(v___y_1038_);
lean_dec(v___y_1037_);
lean_dec_ref(v___y_1036_);
lean_dec(v___y_1035_);
lean_dec(v___y_1034_);
return v_res_1041_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_1047_; lean_object* v___x_1048_; 
v___x_1047_ = l_Lean_maxRecDepthErrorMessage;
v___x_1048_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1048_, 0, v___x_1047_);
return v___x_1048_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__4(void){
_start:
{
lean_object* v___x_1049_; lean_object* v___x_1050_; 
v___x_1049_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__3, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__3_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__3);
v___x_1050_ = l_Lean_MessageData_ofFormat(v___x_1049_);
return v___x_1050_;
}
}
static lean_object* _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__5(void){
_start:
{
lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1051_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__4, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__4_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__4);
v___x_1052_ = ((lean_object*)(lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__2));
v___x_1053_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1053_, 0, v___x_1052_);
lean_ctor_set(v___x_1053_, 1, v___x_1051_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg(lean_object* v_ref_1054_){
_start:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1056_ = lean_obj_once(&lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__5, &lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__5_once, _init_lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___closed__5);
v___x_1057_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1057_, 0, v_ref_1054_);
lean_ctor_set(v___x_1057_, 1, v___x_1056_);
v___x_1058_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1058_, 0, v___x_1057_);
return v___x_1058_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg___boxed(lean_object* v_ref_1059_, lean_object* v___y_1060_){
_start:
{
lean_object* v_res_1061_; 
v_res_1061_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg(v_ref_1059_);
return v_res_1061_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2(lean_object* v_00_u03b1_1062_, lean_object* v_ref_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_){
_start:
{
lean_object* v___x_1071_; 
v___x_1071_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg(v_ref_1063_);
return v___x_1071_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___boxed(lean_object* v_00_u03b1_1072_, lean_object* v_ref_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v_res_1081_; 
v_res_1081_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2(v_00_u03b1_1072_, v_ref_1073_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_, v___y_1079_);
lean_dec(v___y_1079_);
lean_dec_ref(v___y_1078_);
lean_dec(v___y_1077_);
lean_dec_ref(v___y_1076_);
lean_dec(v___y_1075_);
lean_dec(v___y_1074_);
return v_res_1081_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___lam__0(lean_object* v_i_1082_, lean_object* v_goal_1083_, uint8_t v_md_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_){
_start:
{
lean_object* v_lctx_1092_; lean_object* v_decls_1093_; lean_object* v_size_1094_; uint8_t v___x_1095_; 
v_lctx_1092_ = lean_ctor_get(v___y_1087_, 2);
v_decls_1093_ = lean_ctor_get(v_lctx_1092_, 1);
v_size_1094_ = lean_ctor_get(v_decls_1093_, 2);
v___x_1095_ = lean_nat_dec_lt(v_i_1082_, v_size_1094_);
if (v___x_1095_ == 0)
{
lean_object* v___x_1096_; 
lean_dec(v_i_1082_);
v___x_1096_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1096_, 0, v_goal_1083_);
return v___x_1096_;
}
else
{
lean_object* v___x_1097_; lean_object* v___x_1098_; 
v___x_1097_ = lean_box(0);
v___x_1098_ = l_Lean_PersistentArray_get_x21___redArg(v___x_1097_, v_decls_1093_, v_i_1082_);
if (lean_obj_tag(v___x_1098_) == 0)
{
lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; 
v___x_1099_ = lean_unsigned_to_nat(1u);
v___x_1100_ = lean_nat_add(v_i_1082_, v___x_1099_);
lean_dec(v_i_1082_);
v___x_1101_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go(v_md_1084_, v___x_1100_, v_goal_1083_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
return v___x_1101_;
}
else
{
lean_object* v_val_1102_; uint8_t v___x_1103_; 
v_val_1102_ = lean_ctor_get(v___x_1098_, 0);
lean_inc(v_val_1102_);
lean_dec_ref_known(v___x_1098_, 1);
v___x_1103_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1102_);
if (v___x_1103_ == 0)
{
lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1104_ = l_Lean_LocalDecl_fvarId(v_val_1102_);
lean_dec(v_val_1102_);
lean_inc(v_goal_1083_);
v___x_1105_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f(v_goal_1083_, v___x_1104_, v_md_1084_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
if (lean_obj_tag(v___x_1105_) == 0)
{
lean_object* v_a_1106_; 
v_a_1106_ = lean_ctor_get(v___x_1105_, 0);
lean_inc(v_a_1106_);
lean_dec_ref_known(v___x_1105_, 1);
if (lean_obj_tag(v_a_1106_) == 1)
{
lean_object* v_val_1107_; lean_object* v_fst_1108_; lean_object* v_snd_1109_; lean_object* v___x_1110_; 
lean_dec(v_goal_1083_);
v_val_1107_ = lean_ctor_get(v_a_1106_, 0);
lean_inc(v_val_1107_);
lean_dec_ref_known(v_a_1106_, 1);
v_fst_1108_ = lean_ctor_get(v_val_1107_, 0);
lean_inc(v_fst_1108_);
v_snd_1109_ = lean_ctor_get(v_val_1107_, 1);
lean_inc(v_snd_1109_);
lean_dec(v_val_1107_);
v___x_1110_ = lp_aesop_Aesop_recordScriptStep___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__0___redArg(v_fst_1108_, v___y_1085_);
if (lean_obj_tag(v___x_1110_) == 0)
{
lean_object* v___x_1111_; 
lean_dec_ref_known(v___x_1110_, 1);
v___x_1111_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go(v_md_1084_, v_i_1082_, v_snd_1109_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
return v___x_1111_;
}
else
{
lean_object* v_a_1112_; lean_object* v___x_1114_; uint8_t v_isShared_1115_; uint8_t v_isSharedCheck_1119_; 
lean_dec(v_snd_1109_);
lean_dec(v_i_1082_);
v_a_1112_ = lean_ctor_get(v___x_1110_, 0);
v_isSharedCheck_1119_ = !lean_is_exclusive(v___x_1110_);
if (v_isSharedCheck_1119_ == 0)
{
v___x_1114_ = v___x_1110_;
v_isShared_1115_ = v_isSharedCheck_1119_;
goto v_resetjp_1113_;
}
else
{
lean_inc(v_a_1112_);
lean_dec(v___x_1110_);
v___x_1114_ = lean_box(0);
v_isShared_1115_ = v_isSharedCheck_1119_;
goto v_resetjp_1113_;
}
v_resetjp_1113_:
{
lean_object* v___x_1117_; 
if (v_isShared_1115_ == 0)
{
v___x_1117_ = v___x_1114_;
goto v_reusejp_1116_;
}
else
{
lean_object* v_reuseFailAlloc_1118_; 
v_reuseFailAlloc_1118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1118_, 0, v_a_1112_);
v___x_1117_ = v_reuseFailAlloc_1118_;
goto v_reusejp_1116_;
}
v_reusejp_1116_:
{
return v___x_1117_;
}
}
}
}
else
{
lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
lean_dec(v_a_1106_);
v___x_1120_ = lean_unsigned_to_nat(1u);
v___x_1121_ = lean_nat_add(v_i_1082_, v___x_1120_);
lean_dec(v_i_1082_);
v___x_1122_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go(v_md_1084_, v___x_1121_, v_goal_1083_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
return v___x_1122_;
}
}
else
{
lean_object* v_a_1123_; lean_object* v___x_1125_; uint8_t v_isShared_1126_; uint8_t v_isSharedCheck_1130_; 
lean_dec(v_goal_1083_);
lean_dec(v_i_1082_);
v_a_1123_ = lean_ctor_get(v___x_1105_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1105_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1125_ = v___x_1105_;
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
else
{
lean_inc(v_a_1123_);
lean_dec(v___x_1105_);
v___x_1125_ = lean_box(0);
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
v_resetjp_1124_:
{
lean_object* v___x_1128_; 
if (v_isShared_1126_ == 0)
{
v___x_1128_ = v___x_1125_;
goto v_reusejp_1127_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v_a_1123_);
v___x_1128_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1127_;
}
v_reusejp_1127_:
{
return v___x_1128_;
}
}
}
}
else
{
lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; 
lean_dec(v_val_1102_);
v___x_1131_ = lean_unsigned_to_nat(1u);
v___x_1132_ = lean_nat_add(v_i_1082_, v___x_1131_);
lean_dec(v_i_1082_);
v___x_1133_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go(v_md_1084_, v___x_1132_, v_goal_1083_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
return v___x_1133_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___lam__0___boxed(lean_object* v_i_1134_, lean_object* v_goal_1135_, lean_object* v_md_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_){
_start:
{
uint8_t v_md_boxed_1144_; lean_object* v_res_1145_; 
v_md_boxed_1144_ = lean_unbox(v_md_1136_);
v_res_1145_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___lam__0(v_i_1134_, v_goal_1135_, v_md_boxed_1144_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_);
lean_dec(v___y_1142_);
lean_dec_ref(v___y_1141_);
lean_dec(v___y_1140_);
lean_dec_ref(v___y_1139_);
lean_dec(v___y_1138_);
lean_dec(v___y_1137_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go(uint8_t v_md_1146_, lean_object* v_i_1147_, lean_object* v_goal_1148_, lean_object* v_a_1149_, lean_object* v_a_1150_, lean_object* v_a_1151_, lean_object* v_a_1152_, lean_object* v_a_1153_, lean_object* v_a_1154_){
_start:
{
lean_object* v_fileName_1156_; lean_object* v_fileMap_1157_; lean_object* v_options_1158_; lean_object* v_currRecDepth_1159_; lean_object* v_maxRecDepth_1160_; lean_object* v_ref_1161_; lean_object* v_currNamespace_1162_; lean_object* v_openDecls_1163_; lean_object* v_initHeartbeats_1164_; lean_object* v_maxHeartbeats_1165_; lean_object* v_quotContext_1166_; lean_object* v_currMacroScope_1167_; uint8_t v_diag_1168_; lean_object* v_cancelTk_x3f_1169_; uint8_t v_suppressElabErrors_1170_; lean_object* v_inheritedTraceOptions_1171_; lean_object* v___x_1172_; lean_object* v___f_1173_; lean_object* v___x_1179_; uint8_t v___x_1180_; 
v_fileName_1156_ = lean_ctor_get(v_a_1153_, 0);
v_fileMap_1157_ = lean_ctor_get(v_a_1153_, 1);
v_options_1158_ = lean_ctor_get(v_a_1153_, 2);
v_currRecDepth_1159_ = lean_ctor_get(v_a_1153_, 3);
v_maxRecDepth_1160_ = lean_ctor_get(v_a_1153_, 4);
v_ref_1161_ = lean_ctor_get(v_a_1153_, 5);
v_currNamespace_1162_ = lean_ctor_get(v_a_1153_, 6);
v_openDecls_1163_ = lean_ctor_get(v_a_1153_, 7);
v_initHeartbeats_1164_ = lean_ctor_get(v_a_1153_, 8);
v_maxHeartbeats_1165_ = lean_ctor_get(v_a_1153_, 9);
v_quotContext_1166_ = lean_ctor_get(v_a_1153_, 10);
v_currMacroScope_1167_ = lean_ctor_get(v_a_1153_, 11);
v_diag_1168_ = lean_ctor_get_uint8(v_a_1153_, sizeof(void*)*14);
v_cancelTk_x3f_1169_ = lean_ctor_get(v_a_1153_, 12);
v_suppressElabErrors_1170_ = lean_ctor_get_uint8(v_a_1153_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1171_ = lean_ctor_get(v_a_1153_, 13);
v___x_1172_ = lean_box(v_md_1146_);
lean_inc(v_goal_1148_);
v___f_1173_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1173_, 0, v_i_1147_);
lean_closure_set(v___f_1173_, 1, v_goal_1148_);
lean_closure_set(v___f_1173_, 2, v___x_1172_);
v___x_1179_ = lean_unsigned_to_nat(0u);
v___x_1180_ = lean_nat_dec_eq(v_maxRecDepth_1160_, v___x_1179_);
if (v___x_1180_ == 0)
{
uint8_t v___x_1181_; 
v___x_1181_ = lean_nat_dec_eq(v_currRecDepth_1159_, v_maxRecDepth_1160_);
if (v___x_1181_ == 0)
{
goto v___jp_1174_;
}
else
{
lean_object* v___x_1182_; 
lean_dec_ref(v___f_1173_);
lean_dec(v_goal_1148_);
lean_inc(v_ref_1161_);
v___x_1182_ = lp_aesop_Lean_throwMaxRecDepthAt___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__2___redArg(v_ref_1161_);
return v___x_1182_;
}
}
else
{
goto v___jp_1174_;
}
v___jp_1174_:
{
lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; 
v___x_1175_ = lean_unsigned_to_nat(1u);
v___x_1176_ = lean_nat_add(v_currRecDepth_1159_, v___x_1175_);
lean_inc_ref(v_inheritedTraceOptions_1171_);
lean_inc(v_cancelTk_x3f_1169_);
lean_inc(v_currMacroScope_1167_);
lean_inc(v_quotContext_1166_);
lean_inc(v_maxHeartbeats_1165_);
lean_inc(v_initHeartbeats_1164_);
lean_inc(v_openDecls_1163_);
lean_inc(v_currNamespace_1162_);
lean_inc(v_ref_1161_);
lean_inc(v_maxRecDepth_1160_);
lean_inc_ref(v_options_1158_);
lean_inc_ref(v_fileMap_1157_);
lean_inc_ref(v_fileName_1156_);
v___x_1177_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1177_, 0, v_fileName_1156_);
lean_ctor_set(v___x_1177_, 1, v_fileMap_1157_);
lean_ctor_set(v___x_1177_, 2, v_options_1158_);
lean_ctor_set(v___x_1177_, 3, v___x_1176_);
lean_ctor_set(v___x_1177_, 4, v_maxRecDepth_1160_);
lean_ctor_set(v___x_1177_, 5, v_ref_1161_);
lean_ctor_set(v___x_1177_, 6, v_currNamespace_1162_);
lean_ctor_set(v___x_1177_, 7, v_openDecls_1163_);
lean_ctor_set(v___x_1177_, 8, v_initHeartbeats_1164_);
lean_ctor_set(v___x_1177_, 9, v_maxHeartbeats_1165_);
lean_ctor_set(v___x_1177_, 10, v_quotContext_1166_);
lean_ctor_set(v___x_1177_, 11, v_currMacroScope_1167_);
lean_ctor_set(v___x_1177_, 12, v_cancelTk_x3f_1169_);
lean_ctor_set(v___x_1177_, 13, v_inheritedTraceOptions_1171_);
lean_ctor_set_uint8(v___x_1177_, sizeof(void*)*14, v_diag_1168_);
lean_ctor_set_uint8(v___x_1177_, sizeof(void*)*14 + 1, v_suppressElabErrors_1170_);
v___x_1178_ = lp_aesop_Lean_MVarId_withContext___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go_spec__1___redArg(v_goal_1148_, v___f_1173_, v_a_1149_, v_a_1150_, v_a_1151_, v_a_1152_, v___x_1177_, v_a_1154_);
lean_dec_ref_known(v___x_1177_, 14);
return v___x_1178_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___boxed(lean_object* v_md_1183_, lean_object* v_i_1184_, lean_object* v_goal_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_, lean_object* v_a_1188_, lean_object* v_a_1189_, lean_object* v_a_1190_, lean_object* v_a_1191_, lean_object* v_a_1192_){
_start:
{
uint8_t v_md_boxed_1193_; lean_object* v_res_1194_; 
v_md_boxed_1193_ = lean_unbox(v_md_1183_);
v_res_1194_ = lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go(v_md_boxed_1193_, v_i_1184_, v_goal_1185_, v_a_1186_, v_a_1187_, v_a_1188_, v_a_1189_, v_a_1190_, v_a_1191_);
lean_dec(v_a_1191_);
lean_dec_ref(v_a_1190_);
lean_dec(v_a_1189_);
lean_dec_ref(v_a_1188_);
lean_dec(v_a_1187_);
lean_dec(v_a_1186_);
return v_res_1194_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg(lean_object* v_x_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_){
_start:
{
lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; 
v___x_1204_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg___closed__0));
v___x_1205_ = lean_st_mk_ref(v___x_1204_);
lean_inc(v___y_1202_);
lean_inc_ref(v___y_1201_);
lean_inc(v___y_1200_);
lean_inc_ref(v___y_1199_);
lean_inc(v___y_1198_);
lean_inc(v___x_1205_);
v___x_1206_ = lean_apply_7(v_x_1197_, v___x_1205_, v___y_1198_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_, lean_box(0));
if (lean_obj_tag(v___x_1206_) == 0)
{
lean_object* v_a_1207_; lean_object* v___x_1209_; uint8_t v_isShared_1210_; uint8_t v_isSharedCheck_1216_; 
v_a_1207_ = lean_ctor_get(v___x_1206_, 0);
v_isSharedCheck_1216_ = !lean_is_exclusive(v___x_1206_);
if (v_isSharedCheck_1216_ == 0)
{
v___x_1209_ = v___x_1206_;
v_isShared_1210_ = v_isSharedCheck_1216_;
goto v_resetjp_1208_;
}
else
{
lean_inc(v_a_1207_);
lean_dec(v___x_1206_);
v___x_1209_ = lean_box(0);
v_isShared_1210_ = v_isSharedCheck_1216_;
goto v_resetjp_1208_;
}
v_resetjp_1208_:
{
lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1214_; 
v___x_1211_ = lean_st_ref_get(v___x_1205_);
lean_dec(v___x_1205_);
v___x_1212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1212_, 0, v_a_1207_);
lean_ctor_set(v___x_1212_, 1, v___x_1211_);
if (v_isShared_1210_ == 0)
{
lean_ctor_set(v___x_1209_, 0, v___x_1212_);
v___x_1214_ = v___x_1209_;
goto v_reusejp_1213_;
}
else
{
lean_object* v_reuseFailAlloc_1215_; 
v_reuseFailAlloc_1215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1215_, 0, v___x_1212_);
v___x_1214_ = v_reuseFailAlloc_1215_;
goto v_reusejp_1213_;
}
v_reusejp_1213_:
{
return v___x_1214_;
}
}
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1224_; 
lean_dec(v___x_1205_);
v_a_1217_ = lean_ctor_get(v___x_1206_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1206_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1219_ = v___x_1206_;
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1206_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1222_; 
if (v_isShared_1220_ == 0)
{
v___x_1222_ = v___x_1219_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_a_1217_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg___boxed(lean_object* v_x_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_){
_start:
{
lean_object* v_res_1232_; 
v_res_1232_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg(v_x_1225_, v___y_1226_, v___y_1227_, v___y_1228_, v___y_1229_, v___y_1230_);
lean_dec(v___y_1230_);
lean_dec_ref(v___y_1229_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
lean_dec(v___y_1226_);
return v_res_1232_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0(lean_object* v_00_u03b1_1233_, lean_object* v_x_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_){
_start:
{
lean_object* v___x_1241_; 
v___x_1241_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg(v_x_1234_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_, v___y_1239_);
return v___x_1241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___boxed(lean_object* v_00_u03b1_1242_, lean_object* v_x_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_){
_start:
{
lean_object* v_res_1250_; 
v_res_1250_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0(v_00_u03b1_1242_, v_x_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_);
lean_dec(v___y_1248_);
lean_dec_ref(v___y_1247_);
lean_dec(v___y_1246_);
lean_dec_ref(v___y_1245_);
lean_dec(v___y_1244_);
return v_res_1250_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___redArg(lean_object* v_msg_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_){
_start:
{
lean_object* v_ref_1257_; lean_object* v___x_1258_; lean_object* v_a_1259_; lean_object* v___x_1261_; uint8_t v_isShared_1262_; uint8_t v_isSharedCheck_1267_; 
v_ref_1257_ = lean_ctor_get(v___y_1254_, 5);
v___x_1258_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductHyp_x3f_tac_spec__0_spec__0(v_msg_1251_, v___y_1252_, v___y_1253_, v___y_1254_, v___y_1255_);
v_a_1259_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1267_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1267_ == 0)
{
v___x_1261_ = v___x_1258_;
v_isShared_1262_ = v_isSharedCheck_1267_;
goto v_resetjp_1260_;
}
else
{
lean_inc(v_a_1259_);
lean_dec(v___x_1258_);
v___x_1261_ = lean_box(0);
v_isShared_1262_ = v_isSharedCheck_1267_;
goto v_resetjp_1260_;
}
v_resetjp_1260_:
{
lean_object* v___x_1263_; lean_object* v___x_1265_; 
lean_inc(v_ref_1257_);
v___x_1263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1263_, 0, v_ref_1257_);
lean_ctor_set(v___x_1263_, 1, v_a_1259_);
if (v_isShared_1262_ == 0)
{
lean_ctor_set_tag(v___x_1261_, 1);
lean_ctor_set(v___x_1261_, 0, v___x_1263_);
v___x_1265_ = v___x_1261_;
goto v_reusejp_1264_;
}
else
{
lean_object* v_reuseFailAlloc_1266_; 
v_reuseFailAlloc_1266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1266_, 0, v___x_1263_);
v___x_1265_ = v_reuseFailAlloc_1266_;
goto v_reusejp_1264_;
}
v_reusejp_1264_:
{
return v___x_1265_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___redArg___boxed(lean_object* v_msg_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
lean_object* v_res_1274_; 
v_res_1274_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___redArg(v_msg_1268_, v___y_1269_, v___y_1270_, v___y_1271_, v___y_1272_);
lean_dec(v___y_1272_);
lean_dec_ref(v___y_1271_);
lean_dec(v___y_1270_);
lean_dec_ref(v___y_1269_);
return v_res_1274_;
}
}
static lean_object* _init_lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__1(void){
_start:
{
lean_object* v___x_1276_; lean_object* v___x_1277_; 
v___x_1276_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__0));
v___x_1277_ = l_Lean_stringToMessageData(v___x_1276_);
return v___x_1277_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProductsCore(lean_object* v_goal_1278_, uint8_t v_md_1279_, lean_object* v_a_1280_, lean_object* v_a_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_, lean_object* v_a_1284_){
_start:
{
lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; 
v___x_1286_ = lean_unsigned_to_nat(0u);
v___x_1287_ = lean_box(v_md_1279_);
lean_inc(v_goal_1278_);
v___x_1288_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_BuiltinRules_DestructProducts_0__Aesop_BuiltinRules_destructProductsCore_go___boxed), 10, 3);
lean_closure_set(v___x_1288_, 0, v___x_1287_);
lean_closure_set(v___x_1288_, 1, v___x_1286_);
lean_closure_set(v___x_1288_, 2, v_goal_1278_);
v___x_1289_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_BuiltinRules_destructProductsCore_spec__0___redArg(v___x_1288_, v_a_1280_, v_a_1281_, v_a_1282_, v_a_1283_, v_a_1284_);
if (lean_obj_tag(v___x_1289_) == 0)
{
lean_object* v_a_1290_; lean_object* v_fst_1291_; uint8_t v___x_1292_; 
v_a_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc(v_a_1290_);
v_fst_1291_ = lean_ctor_get(v_a_1290_, 0);
lean_inc(v_fst_1291_);
lean_dec(v_a_1290_);
v___x_1292_ = l_Lean_instBEqMVarId_beq(v_fst_1291_, v_goal_1278_);
lean_dec(v_goal_1278_);
lean_dec(v_fst_1291_);
if (v___x_1292_ == 0)
{
return v___x_1289_;
}
else
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v_a_1295_; lean_object* v___x_1297_; uint8_t v_isShared_1298_; uint8_t v_isSharedCheck_1302_; 
lean_dec_ref_known(v___x_1289_, 1);
v___x_1293_ = lean_obj_once(&lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__1, &lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__1_once, _init_lp_aesop_Aesop_BuiltinRules_destructProductsCore___closed__1);
v___x_1294_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___redArg(v___x_1293_, v_a_1281_, v_a_1282_, v_a_1283_, v_a_1284_);
v_a_1295_ = lean_ctor_get(v___x_1294_, 0);
v_isSharedCheck_1302_ = !lean_is_exclusive(v___x_1294_);
if (v_isSharedCheck_1302_ == 0)
{
v___x_1297_ = v___x_1294_;
v_isShared_1298_ = v_isSharedCheck_1302_;
goto v_resetjp_1296_;
}
else
{
lean_inc(v_a_1295_);
lean_dec(v___x_1294_);
v___x_1297_ = lean_box(0);
v_isShared_1298_ = v_isSharedCheck_1302_;
goto v_resetjp_1296_;
}
v_resetjp_1296_:
{
lean_object* v___x_1300_; 
if (v_isShared_1298_ == 0)
{
v___x_1300_ = v___x_1297_;
goto v_reusejp_1299_;
}
else
{
lean_object* v_reuseFailAlloc_1301_; 
v_reuseFailAlloc_1301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1301_, 0, v_a_1295_);
v___x_1300_ = v_reuseFailAlloc_1301_;
goto v_reusejp_1299_;
}
v_reusejp_1299_:
{
return v___x_1300_;
}
}
}
}
else
{
lean_dec(v_goal_1278_);
return v___x_1289_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProductsCore___boxed(lean_object* v_goal_1303_, lean_object* v_md_1304_, lean_object* v_a_1305_, lean_object* v_a_1306_, lean_object* v_a_1307_, lean_object* v_a_1308_, lean_object* v_a_1309_, lean_object* v_a_1310_){
_start:
{
uint8_t v_md_boxed_1311_; lean_object* v_res_1312_; 
v_md_boxed_1311_ = lean_unbox(v_md_1304_);
v_res_1312_ = lp_aesop_Aesop_BuiltinRules_destructProductsCore(v_goal_1303_, v_md_boxed_1311_, v_a_1305_, v_a_1306_, v_a_1307_, v_a_1308_, v_a_1309_);
lean_dec(v_a_1309_);
lean_dec_ref(v_a_1308_);
lean_dec(v_a_1307_);
lean_dec_ref(v_a_1306_);
lean_dec(v_a_1305_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1(lean_object* v_00_u03b1_1313_, lean_object* v_msg_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_){
_start:
{
lean_object* v___x_1321_; 
v___x_1321_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___redArg(v_msg_1314_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_);
return v___x_1321_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1___boxed(lean_object* v_00_u03b1_1322_, lean_object* v_msg_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_){
_start:
{
lean_object* v_res_1330_; 
v_res_1330_ = lp_aesop_Lean_throwError___at___00Aesop_BuiltinRules_destructProductsCore_spec__1(v_00_u03b1_1322_, v_msg_1323_, v___y_1324_, v___y_1325_, v___y_1326_, v___y_1327_, v___y_1328_);
lean_dec(v___y_1328_);
lean_dec_ref(v___y_1327_);
lean_dec(v___y_1326_);
lean_dec_ref(v___y_1325_);
lean_dec(v___y_1324_);
return v_res_1330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProducts(lean_object* v_a_1331_, lean_object* v_a_1332_, lean_object* v_a_1333_, lean_object* v_a_1334_, lean_object* v_a_1335_, lean_object* v_a_1336_){
_start:
{
lean_object* v_options_1338_; lean_object* v_toOptions_1339_; lean_object* v_goal_1340_; uint8_t v_destructProductsTransparency_1341_; lean_object* v___x_1342_; 
v_options_1338_ = lean_ctor_get(v_a_1331_, 4);
v_toOptions_1339_ = lean_ctor_get(v_options_1338_, 0);
lean_inc_ref(v_toOptions_1339_);
v_goal_1340_ = lean_ctor_get(v_a_1331_, 0);
lean_inc_n(v_goal_1340_, 2);
lean_dec_ref(v_a_1331_);
v_destructProductsTransparency_1341_ = lean_ctor_get_uint8(v_toOptions_1339_, sizeof(void*)*6 + 3);
lean_dec_ref(v_toOptions_1339_);
v___x_1342_ = lp_aesop_Aesop_BuiltinRules_destructProductsCore(v_goal_1340_, v_destructProductsTransparency_1341_, v_a_1332_, v_a_1333_, v_a_1334_, v_a_1335_, v_a_1336_);
if (lean_obj_tag(v___x_1342_) == 0)
{
lean_object* v_a_1343_; lean_object* v_fst_1344_; lean_object* v_snd_1345_; lean_object* v___x_1346_; 
v_a_1343_ = lean_ctor_get(v___x_1342_, 0);
lean_inc(v_a_1343_);
lean_dec_ref_known(v___x_1342_, 1);
v_fst_1344_ = lean_ctor_get(v_a_1343_, 0);
lean_inc(v_fst_1344_);
v_snd_1345_ = lean_ctor_get(v_a_1343_, 1);
lean_inc(v_snd_1345_);
lean_dec(v_a_1343_);
v___x_1346_ = lp_aesop_Aesop_diffGoals(v_goal_1340_, v_fst_1344_, v_a_1332_, v_a_1333_, v_a_1334_, v_a_1335_, v_a_1336_);
if (lean_obj_tag(v___x_1346_) == 0)
{
lean_object* v_a_1347_; lean_object* v___x_1348_; 
v_a_1347_ = lean_ctor_get(v___x_1346_, 0);
lean_inc(v_a_1347_);
lean_dec_ref_known(v___x_1346_, 1);
v___x_1348_ = l_Lean_Meta_saveState___redArg(v_a_1334_, v_a_1336_);
if (lean_obj_tag(v___x_1348_) == 0)
{
lean_object* v_a_1349_; lean_object* v___x_1351_; uint8_t v_isShared_1352_; uint8_t v_isSharedCheck_1363_; 
v_a_1349_ = lean_ctor_get(v___x_1348_, 0);
v_isSharedCheck_1363_ = !lean_is_exclusive(v___x_1348_);
if (v_isSharedCheck_1363_ == 0)
{
v___x_1351_ = v___x_1348_;
v_isShared_1352_ = v_isSharedCheck_1363_;
goto v_resetjp_1350_;
}
else
{
lean_inc(v_a_1349_);
lean_dec(v___x_1348_);
v___x_1351_ = lean_box(0);
v_isShared_1352_ = v_isSharedCheck_1363_;
goto v_resetjp_1350_;
}
v_resetjp_1350_:
{
lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1361_; 
v___x_1353_ = lean_unsigned_to_nat(1u);
v___x_1354_ = lean_mk_empty_array_with_capacity(v___x_1353_);
lean_inc_ref(v___x_1354_);
v___x_1355_ = lean_array_push(v___x_1354_, v_a_1347_);
v___x_1356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1356_, 0, v_snd_1345_);
v___x_1357_ = lean_box(0);
v___x_1358_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1358_, 0, v___x_1355_);
lean_ctor_set(v___x_1358_, 1, v_a_1349_);
lean_ctor_set(v___x_1358_, 2, v___x_1356_);
lean_ctor_set(v___x_1358_, 3, v___x_1357_);
v___x_1359_ = lean_array_push(v___x_1354_, v___x_1358_);
if (v_isShared_1352_ == 0)
{
lean_ctor_set(v___x_1351_, 0, v___x_1359_);
v___x_1361_ = v___x_1351_;
goto v_reusejp_1360_;
}
else
{
lean_object* v_reuseFailAlloc_1362_; 
v_reuseFailAlloc_1362_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1362_, 0, v___x_1359_);
v___x_1361_ = v_reuseFailAlloc_1362_;
goto v_reusejp_1360_;
}
v_reusejp_1360_:
{
return v___x_1361_;
}
}
}
else
{
lean_object* v_a_1364_; lean_object* v___x_1366_; uint8_t v_isShared_1367_; uint8_t v_isSharedCheck_1371_; 
lean_dec(v_a_1347_);
lean_dec(v_snd_1345_);
v_a_1364_ = lean_ctor_get(v___x_1348_, 0);
v_isSharedCheck_1371_ = !lean_is_exclusive(v___x_1348_);
if (v_isSharedCheck_1371_ == 0)
{
v___x_1366_ = v___x_1348_;
v_isShared_1367_ = v_isSharedCheck_1371_;
goto v_resetjp_1365_;
}
else
{
lean_inc(v_a_1364_);
lean_dec(v___x_1348_);
v___x_1366_ = lean_box(0);
v_isShared_1367_ = v_isSharedCheck_1371_;
goto v_resetjp_1365_;
}
v_resetjp_1365_:
{
lean_object* v___x_1369_; 
if (v_isShared_1367_ == 0)
{
v___x_1369_ = v___x_1366_;
goto v_reusejp_1368_;
}
else
{
lean_object* v_reuseFailAlloc_1370_; 
v_reuseFailAlloc_1370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1370_, 0, v_a_1364_);
v___x_1369_ = v_reuseFailAlloc_1370_;
goto v_reusejp_1368_;
}
v_reusejp_1368_:
{
return v___x_1369_;
}
}
}
}
else
{
lean_object* v_a_1372_; lean_object* v___x_1374_; uint8_t v_isShared_1375_; uint8_t v_isSharedCheck_1379_; 
lean_dec(v_snd_1345_);
v_a_1372_ = lean_ctor_get(v___x_1346_, 0);
v_isSharedCheck_1379_ = !lean_is_exclusive(v___x_1346_);
if (v_isSharedCheck_1379_ == 0)
{
v___x_1374_ = v___x_1346_;
v_isShared_1375_ = v_isSharedCheck_1379_;
goto v_resetjp_1373_;
}
else
{
lean_inc(v_a_1372_);
lean_dec(v___x_1346_);
v___x_1374_ = lean_box(0);
v_isShared_1375_ = v_isSharedCheck_1379_;
goto v_resetjp_1373_;
}
v_resetjp_1373_:
{
lean_object* v___x_1377_; 
if (v_isShared_1375_ == 0)
{
v___x_1377_ = v___x_1374_;
goto v_reusejp_1376_;
}
else
{
lean_object* v_reuseFailAlloc_1378_; 
v_reuseFailAlloc_1378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1378_, 0, v_a_1372_);
v___x_1377_ = v_reuseFailAlloc_1378_;
goto v_reusejp_1376_;
}
v_reusejp_1376_:
{
return v___x_1377_;
}
}
}
}
else
{
lean_object* v_a_1380_; lean_object* v___x_1382_; uint8_t v_isShared_1383_; uint8_t v_isSharedCheck_1387_; 
lean_dec(v_goal_1340_);
v_a_1380_ = lean_ctor_get(v___x_1342_, 0);
v_isSharedCheck_1387_ = !lean_is_exclusive(v___x_1342_);
if (v_isSharedCheck_1387_ == 0)
{
v___x_1382_ = v___x_1342_;
v_isShared_1383_ = v_isSharedCheck_1387_;
goto v_resetjp_1381_;
}
else
{
lean_inc(v_a_1380_);
lean_dec(v___x_1342_);
v___x_1382_ = lean_box(0);
v_isShared_1383_ = v_isSharedCheck_1387_;
goto v_resetjp_1381_;
}
v_resetjp_1381_:
{
lean_object* v___x_1385_; 
if (v_isShared_1383_ == 0)
{
v___x_1385_ = v___x_1382_;
goto v_reusejp_1384_;
}
else
{
lean_object* v_reuseFailAlloc_1386_; 
v_reuseFailAlloc_1386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1386_, 0, v_a_1380_);
v___x_1385_ = v_reuseFailAlloc_1386_;
goto v_reusejp_1384_;
}
v_reusejp_1384_:
{
return v___x_1385_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_destructProducts___boxed(lean_object* v_a_1388_, lean_object* v_a_1389_, lean_object* v_a_1390_, lean_object* v_a_1391_, lean_object* v_a_1392_, lean_object* v_a_1393_, lean_object* v_a_1394_){
_start:
{
lean_object* v_res_1395_; 
v_res_1395_ = lp_aesop_Aesop_BuiltinRules_destructProducts(v_a_1388_, v_a_1389_, v_a_1390_, v_a_1391_, v_a_1392_, v_a_1393_);
lean_dec(v_a_1393_);
lean_dec_ref(v_a_1392_);
lean_dec(v_a_1391_);
lean_dec_ref(v_a_1390_);
lean_dec(v_a_1389_);
return v_res_1395_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_DestructProducts(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_BuiltinRules_DestructProducts(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BuiltinRules_DestructProducts(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_DestructProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_BuiltinRules_DestructProducts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_BuiltinRules_DestructProducts(builtin);
}
#ifdef __cplusplus
}
#endif
