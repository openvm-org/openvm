// Lean compiler output
// Module: Mathlib.Tactic.CancelDenoms.Core
// Imports: public import Init public meta import Init public meta import Mathlib.Algebra.Group.Nat.Defs public meta import Mathlib.Basic.Logic.Basic public meta import Mathlib.Data.Tree.Basic public import Mathlib.Algebra.Field.Basic public import Mathlib.Algebra.Order.Ring.Defs public import Mathlib.Data.Tree.Basic public import Mathlib.Tactic.NormNum.Core public import Mathlib.Util.SynthesizeUsing
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Nat_lcm(lean_object*, lean_object*);
lean_object* l_Lean_Expr_nat_x3f(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_synthesizeUsingTactic_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_neutralConfig;
lean_object* lp_mathlib_Lean_Meta_simpOnlyNames(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_gcd(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_replaceTargetEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqMP(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_replace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CancelDenoms"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(49, 112, 138, 54, 109, 104, 55, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__2_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__3_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__5_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__7_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(221, 9, 44, 65, 161, 89, 128, 124)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Core"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__8_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(181, 225, 71, 106, 112, 177, 106, 83)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__10_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(160, 218, 215, 107, 29, 117, 215, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__11_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__12_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(237, 9, 10, 30, 14, 202, 205, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__13_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__14_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(176, 59, 86, 14, 150, 67, 36, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__15_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(57, 136, 182, 196, 26, 6, 35, 60)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__16_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(16, 115, 196, 165, 249, 193, 222, 57)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__17_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(254, 75, 223, 170, 207, 133, 30, 172)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__19_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__18_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__9_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(170, 46, 181, 140, 217, 171, 147, 51)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__19_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__19_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__20_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__19_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1602764063) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(64, 212, 232, 179, 153, 151, 58, 203)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__20_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__20_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__21_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__21_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__21_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__22_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__20_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__21_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(23, 71, 218, 81, 95, 11, 135, 9)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__22_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__22_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__23_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__23_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__23_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__24_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__22_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__23_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(255, 95, 48, 159, 170, 237, 167, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__24_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__24_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__25_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__24_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(58, 135, 17, 134, 87, 23, 17, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__25_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__25_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2____boxed(lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Inv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "normNum"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__0_value),LEAN_SCALAR_PTR_LITERAL(235, 202, 36, 226, 215, 147, 189, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "norm_num"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__5_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Could not prove "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = " using norm_num. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__13;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__8_value),LEAN_SCALAR_PTR_LITERAL(142, 68, 231, 210, 96, 163, 154, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__9_value),LEAN_SCALAR_PTR_LITERAL(63, 31, 248, 222, 13, 64, 40, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "InvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toInv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(120, 190, 7, 179, 62, 236, 21, 116)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(28, 25, 248, 9, 15, 85, 72, 194)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "DivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "toInvOneClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(162, 155, 123, 0, 237, 243, 28, 65)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(181, 224, 200, 199, 184, 130, 54, 26)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toDivInvOneMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(16, 242, 184, 157, 107, 26, 18, 78)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(60, 63, 43, 77, 240, 6, 89, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "DivisionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toDivisionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 252, 206, 54, 37, 44, 48, 53)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__11_value),LEAN_SCALAR_PTR_LITERAL(149, 21, 120, 191, 172, 81, 156, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "CommGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "toDivisionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(197, 14, 145, 254, 35, 172, 249, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__14_value),LEAN_SCALAR_PTR_LITERAL(94, 143, 233, 228, 60, 239, 1, 188)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "toCommGroupWithZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__7_value),LEAN_SCALAR_PTR_LITERAL(155, 188, 136, 200, 106, 253, 76, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__10_value),LEAN_SCALAR_PTR_LITERAL(32, 63, 208, 57, 56, 184, 164, 144)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(213, 197, 76, 235, 199, 0, 254, 199)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "NPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(39, 79, 240, 225, 164, 207, 253, 237)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(56, 108, 173, 227, 4, 14, 173, 115)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Monoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toNPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(162, 147, 2, 115, 233, 179, 113, 5)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(224, 31, 132, 245, 47, 70, 119, 231)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(86, 172, 133, 187, 121, 84, 206, 170)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__6_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__11_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "NegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toNeg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(156, 44, 233, 53, 1, 106, 24, 217)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(124, 136, 108, 160, 134, 153, 101, 8)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "SubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toNegZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(135, 233, 160, 34, 207, 245, 132, 138)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(107, 179, 145, 12, 37, 42, 18, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "SubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toSubNegZeroMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(203, 24, 17, 79, 61, 156, 198, 150)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__8_value),LEAN_SCALAR_PTR_LITERAL(94, 234, 159, 237, 9, 124, 201, 94)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "SubtractionCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "toSubtractionMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(100, 8, 183, 201, 110, 57, 85, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__11_value),LEAN_SCALAR_PTR_LITERAL(203, 26, 135, 240, 118, 74, 112, 111)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "AddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "toDivisionAddCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__13_value),LEAN_SCALAR_PTR_LITERAL(59, 221, 192, 169, 110, 67, 255, 76)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__14_value),LEAN_SCALAR_PTR_LITERAL(65, 138, 55, 164, 85, 246, 87, 209)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toAddCommGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__5_value),LEAN_SCALAR_PTR_LITERAL(74, 223, 78, 88, 255, 236, 144, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__12_value),LEAN_SCALAR_PTR_LITERAL(26, 183, 188, 240, 156, 118, 170, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(34, 70, 113, 198, 157, 211, 131, 18)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toDiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(231, 106, 236, 89, 112, 21, 122, 113)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(95, 209, 62, 72, 37, 30, 170, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivInvMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__3_value),LEAN_SCALAR_PTR_LITERAL(121, 130, 45, 212, 110, 237, 236, 233)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__14_value),LEAN_SCALAR_PTR_LITERAL(231, 253, 204, 163, 168, 77, 27, 58)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__1_value),LEAN_SCALAR_PTR_LITERAL(32, 225, 92, 14, 170, 61, 170, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "SubNegMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__3_value),LEAN_SCALAR_PTR_LITERAL(161, 3, 69, 109, 235, 35, 121, 64)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(17, 223, 222, 114, 35, 206, 250, 124)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "AddGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toSubNegMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__6_value),LEAN_SCALAR_PTR_LITERAL(211, 76, 74, 39, 69, 162, 229, 135)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__7_value),LEAN_SCALAR_PTR_LITERAL(237, 249, 208, 19, 139, 128, 45, 144)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toAddGroup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__2_value),LEAN_SCALAR_PTR_LITERAL(221, 239, 47, 196, 170, 166, 59, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__15_value),LEAN_SCALAR_PTR_LITERAL(134, 172, 115, 219, 189, 252, 56, 148)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__1_value),LEAN_SCALAR_PTR_LITERAL(229, 81, 239, 34, 203, 244, 36, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "AddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "AddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "toAddMonoidWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__2_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__3_value),LEAN_SCALAR_PTR_LITERAL(226, 82, 90, 134, 221, 253, 108, 55)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Ring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "toAddGroupWithOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__5_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__6_value),LEAN_SCALAR_PTR_LITERAL(99, 161, 243, 168, 232, 89, 236, 229)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "DivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__9_value),LEAN_SCALAR_PTR_LITERAL(196, 15, 37, 9, 106, 139, 236, 93)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferInstance"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__11_value),LEAN_SCALAR_PTR_LITERAL(17, 162, 120, 176, 98, 85, 114, 76)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Field"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toDivisionRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__13_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__14_value),LEAN_SCALAR_PTR_LITERAL(60, 172, 238, 141, 54, 76, 141, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__16_value),LEAN_SCALAR_PTR_LITERAL(161, 247, 70, 70, 118, 145, 235, 92)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "OfNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__18_value),LEAN_SCALAR_PTR_LITERAL(135, 241, 166, 108, 243, 216, 193, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(2, 108, 58, 34, 100, 49, 50, 216)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat0"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__23_value),LEAN_SCALAR_PTR_LITERAL(192, 171, 244, 106, 217, 72, 118, 253)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__24_value),LEAN_SCALAR_PTR_LITERAL(208, 59, 186, 84, 178, 224, 2, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "MulZeroClass"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "toZero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__26_value),LEAN_SCALAR_PTR_LITERAL(232, 169, 101, 213, 120, 247, 80, 71)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__27_value),LEAN_SCALAR_PTR_LITERAL(216, 253, 35, 170, 63, 16, 177, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "instMulZeroClassOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__29_value),LEAN_SCALAR_PTR_LITERAL(31, 133, 13, 57, 152, 228, 72, 248)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "inv_subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__31_value),LEAN_SCALAR_PTR_LITERAL(166, 210, 185, 244, 133, 242, 41, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__33;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "pow_subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__35_value),LEAN_SCALAR_PTR_LITERAL(22, 89, 143, 217, 207, 44, 113, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toCommRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__13_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__37_value),LEAN_SCALAR_PTR_LITERAL(80, 133, 194, 203, 22, 179, 103, 113)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__5_value),LEAN_SCALAR_PTR_LITERAL(151, 9, 120, 97, 235, 184, 251, 227)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__39_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__16_value),LEAN_SCALAR_PTR_LITERAL(121, 151, 225, 139, 113, 68, 25, 156)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "neg_subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__40_value),LEAN_SCALAR_PTR_LITERAL(73, 194, 243, 94, 107, 107, 1, 237)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(34, 214, 17, 155, 7, 71, 232, 190)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__6_value),LEAN_SCALAR_PTR_LITERAL(157, 154, 239, 235, 210, 195, 14, 77)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__43_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__44;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "One"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toOfNat1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__45_value),LEAN_SCALAR_PTR_LITERAL(19, 85, 184, 168, 121, 55, 74, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__46_value),LEAN_SCALAR_PTR_LITERAL(105, 141, 113, 1, 81, 178, 189, 182)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toOne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 54, 100, 45, 135, 24, 207, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__49_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__48_value),LEAN_SCALAR_PTR_LITERAL(52, 219, 71, 246, 148, 114, 208, 126)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__49_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "div_subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__50_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__50_value),LEAN_SCALAR_PTR_LITERAL(60, 118, 115, 233, 107, 152, 215, 215)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "mul_subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__52_value),LEAN_SCALAR_PTR_LITERAL(218, 7, 94, 139, 132, 141, 229, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__54_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__55_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "recursing into mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__57_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__58;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__59_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__2_value),LEAN_SCALAR_PTR_LITERAL(88, 61, 45, 121, 84, 135, 11, 188)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__59_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__9_value),LEAN_SCALAR_PTR_LITERAL(14, 89, 255, 1, 224, 64, 98, 35)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__59_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "sub_subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__60_value),LEAN_SCALAR_PTR_LITERAL(243, 90, 143, 115, 14, 163, 113, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__62_value),LEAN_SCALAR_PTR_LITERAL(155, 25, 183, 66, 31, 85, 84, 65)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__63_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Distrib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__64_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__65_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__66_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__64_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__66_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__65_value),LEAN_SCALAR_PTR_LITERAL(159, 190, 95, 162, 187, 73, 156, 147)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__66_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "instDistribOfSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__67_value),LEAN_SCALAR_PTR_LITERAL(208, 10, 80, 43, 19, 152, 244, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__68_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "CommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__70_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__69_value),LEAN_SCALAR_PTR_LITERAL(22, 69, 197, 205, 197, 50, 81, 124)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__70_value),LEAN_SCALAR_PTR_LITERAL(1, 71, 172, 115, 76, 22, 6, 37)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Semifield"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__72_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toCommSemiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__72_value),LEAN_SCALAR_PTR_LITERAL(205, 214, 159, 28, 31, 185, 116, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__73_value),LEAN_SCALAR_PTR_LITERAL(134, 142, 86, 147, 34, 154, 178, 196)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "toSemifield"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__75_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__13_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__75_value),LEAN_SCALAR_PTR_LITERAL(104, 104, 95, 86, 153, 146, 86, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__77_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__4_value),LEAN_SCALAR_PTR_LITERAL(254, 113, 255, 140, 142, 9, 169, 40)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__77_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__13_value),LEAN_SCALAR_PTR_LITERAL(248, 227, 200, 215, 229, 255, 92, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__77_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instHMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__78_value),LEAN_SCALAR_PTR_LITERAL(177, 107, 107, 59, 202, 230, 169, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__79_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__80_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__80_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__81_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__82_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__82_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__83_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__84_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__64_value),LEAN_SCALAR_PTR_LITERAL(221, 154, 114, 180, 95, 113, 104, 161)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__84_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 149, 205, 214, 52, 248, 155, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__84_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "add_subst"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__85_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__85_value),LEAN_SCALAR_PTR_LITERAL(201, 212, 162, 94, 187, 19, 105, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "mkProdPrf "};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__87_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__88_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__88;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__89_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__90_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__90;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "div_div_eq_mul_div"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__0_value),LEAN_SCALAR_PTR_LITERAL(94, 115, 221, 175, 198, 99, 252, 65)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "div_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__2_value),LEAN_SCALAR_PTR_LITERAL(100, 196, 151, 147, 56, 93, 65, 148)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "derive_trans"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 13, .m_data = "derive_trans₂"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "CancelDenoms.derive failed to normalize "};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ".\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__3;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__13_value),LEAN_SCALAR_PTR_LITERAL(22, 39, 49, 148, 16, 49, 114, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "pf : "};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__6;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 1, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__7_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__13;
static const lean_array_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "e norm_num'd = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "e simplified = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__18;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "e = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "gt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "cancel_factors_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__5_value),LEAN_SCALAR_PTR_LITERAL(160, 63, 23, 205, 210, 79, 122, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "cancel_factors_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__8_value),LEAN_SCALAR_PTR_LITERAL(79, 119, 72, 68, 144, 174, 226, 52)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "cancel_factors_ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__13_value),LEAN_SCALAR_PTR_LITERAL(205, 3, 112, 254, 251, 75, 7, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "cancel_factors_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 158, 170, 169, 114, 137, 98, 186)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__15_value),LEAN_SCALAR_PTR_LITERAL(249, 161, 126, 18, 214, 166, 0, 127)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__3_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "IsStrictOrderedRing"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__5_value),LEAN_SCALAR_PTR_LITERAL(91, 31, 27, 198, 71, 31, 228, 59)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SemilatticeInf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "toPartialOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__7_value),LEAN_SCALAR_PTR_LITERAL(62, 131, 181, 193, 54, 206, 77, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__8_value),LEAN_SCALAR_PTR_LITERAL(232, 130, 7, 36, 8, 188, 120, 72)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Lattice"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "toSemilatticeInf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__10_value),LEAN_SCALAR_PTR_LITERAL(58, 214, 49, 195, 61, 20, 1, 8)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__11_value),LEAN_SCALAR_PTR_LITERAL(164, 130, 80, 139, 133, 157, 146, 245)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "DistribLattice"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "toLattice"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__13_value),LEAN_SCALAR_PTR_LITERAL(211, 66, 65, 127, 78, 58, 2, 133)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__14_value),LEAN_SCALAR_PTR_LITERAL(176, 217, 83, 125, 106, 180, 122, 79)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "instDistribLatticeOfLinearOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__16_value),LEAN_SCALAR_PTR_LITERAL(204, 88, 224, 186, 247, 116, 10, 234)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Preorder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "toLT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__19_value),LEAN_SCALAR_PTR_LITERAL(171, 85, 2, 192, 23, 244, 204, 242)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__20_value),LEAN_SCALAR_PTR_LITERAL(213, 59, 145, 160, 110, 90, 162, 17)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "PartialOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "toPreorder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__22_value),LEAN_SCALAR_PTR_LITERAL(47, 196, 146, 225, 179, 207, 152, 76)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__23_value),LEAN_SCALAR_PTR_LITERAL(3, 6, 195, 109, 53, 169, 118, 52)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "cannot kill factors"};
static const lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__26;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "cancelDenoms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 33, 74, 41, 245, 8, 49, 69)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "cancel_denoms"};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_cancelDenoms;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Failed to cancel any denominators"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "tacticCancel_denoms_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 164, 42, 91, 104, 190, 108, 17)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticCancel__denoms__;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 109, 187, 155, 23, 130, 33, 152)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "simpArgs"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(158, 198, 190, 154, 66, 126, 242, 208)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__3_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "mul_assoc"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__14;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(18, 21, 243, 68, 110, 93, 69, 52)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__16_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__18_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__19_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_61_; uint8_t v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_61_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_62_ = 0;
v___x_63_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__25_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_64_ = l_Lean_registerTraceClass(v___x_61_, v___x_62_, v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2____boxed(lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(lean_object* v_e_87_){
_start:
{
lean_object* v_e1_91_; lean_object* v_e2_92_; lean_object* v___x_108_; lean_object* v_fst_109_; 
v___x_108_ = l_Lean_Expr_getAppFnArgs(v_e_87_);
v_fst_109_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_fst_109_);
if (lean_obj_tag(v_fst_109_) == 1)
{
lean_object* v_pre_110_; 
v_pre_110_ = lean_ctor_get(v_fst_109_, 0);
lean_inc(v_pre_110_);
if (lean_obj_tag(v_pre_110_) == 1)
{
lean_object* v_pre_111_; 
v_pre_111_ = lean_ctor_get(v_pre_110_, 0);
if (lean_obj_tag(v_pre_111_) == 0)
{
lean_object* v_snd_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_250_; 
v_snd_112_ = lean_ctor_get(v___x_108_, 1);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_250_ == 0)
{
lean_object* v_unused_251_; 
v_unused_251_ = lean_ctor_get(v___x_108_, 0);
lean_dec(v_unused_251_);
v___x_114_ = v___x_108_;
v_isShared_115_ = v_isSharedCheck_250_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_snd_112_);
lean_dec(v___x_108_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_250_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v_str_116_; lean_object* v_str_117_; lean_object* v___x_118_; uint8_t v___x_119_; 
v_str_116_ = lean_ctor_get(v_fst_109_, 1);
lean_inc_ref(v_str_116_);
lean_dec_ref_known(v_fst_109_, 2);
v_str_117_ = lean_ctor_get(v_pre_110_, 1);
lean_inc_ref(v_str_117_);
lean_dec_ref_known(v_pre_110_, 2);
v___x_118_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__2));
v___x_119_ = lean_string_dec_eq(v_str_117_, v___x_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_120_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__3));
v___x_121_ = lean_string_dec_eq(v_str_117_, v___x_120_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_122_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__4));
v___x_123_ = lean_string_dec_eq(v_str_117_, v___x_122_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; uint8_t v___x_125_; 
v___x_124_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__5));
v___x_125_ = lean_string_dec_eq(v_str_117_, v___x_124_);
if (v___x_125_ == 0)
{
lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_126_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__6));
v___x_127_ = lean_string_dec_eq(v_str_117_, v___x_126_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; uint8_t v___x_129_; 
v___x_128_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__7));
v___x_129_ = lean_string_dec_eq(v_str_117_, v___x_128_);
if (v___x_129_ == 0)
{
lean_object* v___x_130_; uint8_t v___x_131_; 
v___x_130_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__8));
v___x_131_ = lean_string_dec_eq(v_str_117_, v___x_130_);
lean_dec_ref(v_str_117_);
if (v___x_131_ == 0)
{
lean_dec_ref(v_str_116_);
lean_del_object(v___x_114_);
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_132_; uint8_t v___x_133_; 
v___x_132_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__9));
v___x_133_ = lean_string_dec_eq(v_str_116_, v___x_132_);
lean_dec_ref(v_str_116_);
if (v___x_133_ == 0)
{
lean_del_object(v___x_114_);
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_134_; lean_object* v___x_135_; uint8_t v___x_136_; 
v___x_134_ = lean_array_get_size(v_snd_112_);
v___x_135_ = lean_unsigned_to_nat(3u);
v___x_136_ = lean_nat_dec_eq(v___x_134_, v___x_135_);
if (v___x_136_ == 0)
{
lean_del_object(v___x_114_);
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_137_ = lean_unsigned_to_nat(2u);
v___x_138_ = lean_array_fget(v_snd_112_, v___x_137_);
lean_dec(v_snd_112_);
v___x_139_ = l_Lean_Expr_nat_x3f(v___x_138_);
if (lean_obj_tag(v___x_139_) == 0)
{
lean_object* v___x_140_; 
lean_del_object(v___x_114_);
v___x_140_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__1));
return v___x_140_;
}
else
{
lean_object* v_val_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_146_; 
v_val_141_ = lean_ctor_get(v___x_139_, 0);
lean_inc_n(v_val_141_, 3);
lean_dec_ref_known(v___x_139_, 1);
v___x_142_ = lean_box(0);
v___x_143_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_143_, 0, v_val_141_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
lean_ctor_set(v___x_143_, 2, v___x_142_);
v___x_144_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_144_, 0, v_val_141_);
lean_ctor_set(v___x_144_, 1, v___x_142_);
lean_ctor_set(v___x_144_, 2, v___x_143_);
if (v_isShared_115_ == 0)
{
lean_ctor_set(v___x_114_, 1, v___x_144_);
lean_ctor_set(v___x_114_, 0, v_val_141_);
v___x_146_ = v___x_114_;
goto v_reusejp_145_;
}
else
{
lean_object* v_reuseFailAlloc_147_; 
v_reuseFailAlloc_147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_147_, 0, v_val_141_);
lean_ctor_set(v_reuseFailAlloc_147_, 1, v___x_144_);
v___x_146_ = v_reuseFailAlloc_147_;
goto v_reusejp_145_;
}
v_reusejp_145_:
{
return v___x_146_;
}
}
}
}
}
}
else
{
lean_object* v___x_148_; uint8_t v___x_149_; 
lean_dec_ref(v_str_117_);
lean_del_object(v___x_114_);
v___x_148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__10));
v___x_149_ = lean_string_dec_eq(v_str_116_, v___x_148_);
lean_dec_ref(v_str_116_);
if (v___x_149_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_150_; lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_150_ = lean_array_get_size(v_snd_112_);
v___x_151_ = lean_unsigned_to_nat(6u);
v___x_152_ = lean_nat_dec_eq(v___x_150_, v___x_151_);
if (v___x_152_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_153_ = lean_unsigned_to_nat(5u);
v___x_154_ = lean_array_fget_borrowed(v_snd_112_, v___x_153_);
lean_inc(v___x_154_);
v___x_155_ = l_Lean_Expr_nat_x3f(v___x_154_);
if (lean_obj_tag(v___x_155_) == 0)
{
lean_object* v___x_156_; 
lean_dec(v_snd_112_);
v___x_156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__1));
return v___x_156_;
}
else
{
lean_object* v_val_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v_fst_161_; lean_object* v_snd_162_; lean_object* v___x_164_; uint8_t v_isShared_165_; uint8_t v_isSharedCheck_173_; 
v_val_157_ = lean_ctor_get(v___x_155_, 0);
lean_inc(v_val_157_);
lean_dec_ref_known(v___x_155_, 1);
v___x_158_ = lean_unsigned_to_nat(4u);
v___x_159_ = lean_array_fget(v_snd_112_, v___x_158_);
lean_dec(v_snd_112_);
v___x_160_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(v___x_159_);
v_fst_161_ = lean_ctor_get(v___x_160_, 0);
v_snd_162_ = lean_ctor_get(v___x_160_, 1);
v_isSharedCheck_173_ = !lean_is_exclusive(v___x_160_);
if (v_isSharedCheck_173_ == 0)
{
v___x_164_ = v___x_160_;
v_isShared_165_ = v_isSharedCheck_173_;
goto v_resetjp_163_;
}
else
{
lean_inc(v_snd_162_);
lean_inc(v_fst_161_);
lean_dec(v___x_160_);
v___x_164_ = lean_box(0);
v_isShared_165_ = v_isSharedCheck_173_;
goto v_resetjp_163_;
}
v_resetjp_163_:
{
lean_object* v_n_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_171_; 
v_n_166_ = lean_nat_pow(v_fst_161_, v_val_157_);
lean_dec(v_fst_161_);
v___x_167_ = lean_box(0);
v___x_168_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_168_, 0, v_val_157_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
lean_ctor_set(v___x_168_, 2, v___x_167_);
lean_inc(v_n_166_);
v___x_169_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_169_, 0, v_n_166_);
lean_ctor_set(v___x_169_, 1, v_snd_162_);
lean_ctor_set(v___x_169_, 2, v___x_168_);
if (v_isShared_165_ == 0)
{
lean_ctor_set(v___x_164_, 1, v___x_169_);
lean_ctor_set(v___x_164_, 0, v_n_166_);
v___x_171_ = v___x_164_;
goto v_reusejp_170_;
}
else
{
lean_object* v_reuseFailAlloc_172_; 
v_reuseFailAlloc_172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_172_, 0, v_n_166_);
lean_ctor_set(v_reuseFailAlloc_172_, 1, v___x_169_);
v___x_171_ = v_reuseFailAlloc_172_;
goto v_reusejp_170_;
}
v_reusejp_170_:
{
return v___x_171_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_174_; uint8_t v___x_175_; 
lean_dec_ref(v_str_117_);
lean_del_object(v___x_114_);
v___x_174_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__11));
v___x_175_ = lean_string_dec_eq(v_str_116_, v___x_174_);
lean_dec_ref(v_str_116_);
if (v___x_175_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_176_; lean_object* v___x_177_; uint8_t v___x_178_; 
v___x_176_ = lean_array_get_size(v_snd_112_);
v___x_177_ = lean_unsigned_to_nat(3u);
v___x_178_ = lean_nat_dec_eq(v___x_176_, v___x_177_);
if (v___x_178_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_179_ = lean_unsigned_to_nat(2u);
v___x_180_ = lean_array_fget(v_snd_112_, v___x_179_);
lean_dec(v_snd_112_);
v_e_87_ = v___x_180_;
goto _start;
}
}
}
}
else
{
lean_object* v___x_182_; uint8_t v___x_183_; 
lean_dec_ref(v_str_117_);
lean_del_object(v___x_114_);
v___x_182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__12));
v___x_183_ = lean_string_dec_eq(v_str_116_, v___x_182_);
lean_dec_ref(v_str_116_);
if (v___x_183_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_184_; lean_object* v___x_185_; uint8_t v___x_186_; 
v___x_184_ = lean_array_get_size(v_snd_112_);
v___x_185_ = lean_unsigned_to_nat(6u);
v___x_186_ = lean_nat_dec_eq(v___x_184_, v___x_185_);
if (v___x_186_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_187_ = lean_unsigned_to_nat(5u);
v___x_188_ = lean_array_fget_borrowed(v_snd_112_, v___x_187_);
lean_inc(v___x_188_);
v___x_189_ = l_Lean_Expr_nat_x3f(v___x_188_);
if (lean_obj_tag(v___x_189_) == 0)
{
lean_object* v___x_190_; 
lean_dec(v_snd_112_);
v___x_190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__1));
return v___x_190_;
}
else
{
lean_object* v_val_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v_fst_195_; lean_object* v_snd_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_207_; 
v_val_191_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_val_191_);
lean_dec_ref_known(v___x_189_, 1);
v___x_192_ = lean_unsigned_to_nat(4u);
v___x_193_ = lean_array_fget(v_snd_112_, v___x_192_);
lean_dec(v_snd_112_);
v___x_194_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(v___x_193_);
v_fst_195_ = lean_ctor_get(v___x_194_, 0);
v_snd_196_ = lean_ctor_get(v___x_194_, 1);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_194_);
if (v_isSharedCheck_207_ == 0)
{
v___x_198_ = v___x_194_;
v_isShared_199_ = v_isSharedCheck_207_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_snd_196_);
lean_inc(v_fst_195_);
lean_dec(v___x_194_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_207_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
lean_object* v_n_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_205_; 
v_n_200_ = lean_nat_mul(v_fst_195_, v_val_191_);
lean_dec(v_fst_195_);
v___x_201_ = lean_box(0);
v___x_202_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_202_, 0, v_val_191_);
lean_ctor_set(v___x_202_, 1, v___x_201_);
lean_ctor_set(v___x_202_, 2, v___x_201_);
lean_inc(v_n_200_);
v___x_203_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_203_, 0, v_n_200_);
lean_ctor_set(v___x_203_, 1, v_snd_196_);
lean_ctor_set(v___x_203_, 2, v___x_202_);
if (v_isShared_199_ == 0)
{
lean_ctor_set(v___x_198_, 1, v___x_203_);
lean_ctor_set(v___x_198_, 0, v_n_200_);
v___x_205_ = v___x_198_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_n_200_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v___x_203_);
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
}
else
{
lean_object* v___x_208_; uint8_t v___x_209_; 
lean_dec_ref(v_str_117_);
lean_del_object(v___x_114_);
v___x_208_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__13));
v___x_209_ = lean_string_dec_eq(v_str_116_, v___x_208_);
lean_dec_ref(v_str_116_);
if (v___x_209_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_210_; lean_object* v___x_211_; uint8_t v___x_212_; 
v___x_210_ = lean_array_get_size(v_snd_112_);
v___x_211_ = lean_unsigned_to_nat(6u);
v___x_212_ = lean_nat_dec_eq(v___x_210_, v___x_211_);
if (v___x_212_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v_fst_216_; lean_object* v_snd_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v_fst_221_; lean_object* v_snd_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_231_; 
v___x_213_ = lean_unsigned_to_nat(4u);
v___x_214_ = lean_array_fget_borrowed(v_snd_112_, v___x_213_);
lean_inc(v___x_214_);
v___x_215_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(v___x_214_);
v_fst_216_ = lean_ctor_get(v___x_215_, 0);
lean_inc(v_fst_216_);
v_snd_217_ = lean_ctor_get(v___x_215_, 1);
lean_inc(v_snd_217_);
lean_dec_ref(v___x_215_);
v___x_218_ = lean_unsigned_to_nat(5u);
v___x_219_ = lean_array_fget(v_snd_112_, v___x_218_);
lean_dec(v_snd_112_);
v___x_220_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(v___x_219_);
v_fst_221_ = lean_ctor_get(v___x_220_, 0);
v_snd_222_ = lean_ctor_get(v___x_220_, 1);
v_isSharedCheck_231_ = !lean_is_exclusive(v___x_220_);
if (v_isSharedCheck_231_ == 0)
{
v___x_224_ = v___x_220_;
v_isShared_225_ = v_isSharedCheck_231_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_snd_222_);
lean_inc(v_fst_221_);
lean_dec(v___x_220_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_231_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v_pd_226_; lean_object* v___x_227_; lean_object* v___x_229_; 
v_pd_226_ = lean_nat_mul(v_fst_216_, v_fst_221_);
lean_dec(v_fst_221_);
lean_dec(v_fst_216_);
lean_inc(v_pd_226_);
v___x_227_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_227_, 0, v_pd_226_);
lean_ctor_set(v___x_227_, 1, v_snd_217_);
lean_ctor_set(v___x_227_, 2, v_snd_222_);
if (v_isShared_225_ == 0)
{
lean_ctor_set(v___x_224_, 1, v___x_227_);
lean_ctor_set(v___x_224_, 0, v_pd_226_);
v___x_229_ = v___x_224_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_pd_226_);
lean_ctor_set(v_reuseFailAlloc_230_, 1, v___x_227_);
v___x_229_ = v_reuseFailAlloc_230_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
return v___x_229_;
}
}
}
}
}
}
else
{
lean_object* v___x_232_; uint8_t v___x_233_; 
lean_dec_ref(v_str_117_);
lean_del_object(v___x_114_);
v___x_232_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__14));
v___x_233_ = lean_string_dec_eq(v_str_116_, v___x_232_);
lean_dec_ref(v_str_116_);
if (v___x_233_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_234_; lean_object* v___x_235_; uint8_t v___x_236_; 
v___x_234_ = lean_array_get_size(v_snd_112_);
v___x_235_ = lean_unsigned_to_nat(6u);
v___x_236_ = lean_nat_dec_eq(v___x_234_, v___x_235_);
if (v___x_236_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_237_ = lean_unsigned_to_nat(4u);
v___x_238_ = lean_array_fget(v_snd_112_, v___x_237_);
v___x_239_ = lean_unsigned_to_nat(5u);
v___x_240_ = lean_array_fget(v_snd_112_, v___x_239_);
lean_dec(v_snd_112_);
v_e1_91_ = v___x_238_;
v_e2_92_ = v___x_240_;
goto v___jp_90_;
}
}
}
}
else
{
lean_object* v___x_241_; uint8_t v___x_242_; 
lean_dec_ref(v_str_117_);
lean_del_object(v___x_114_);
v___x_241_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__15));
v___x_242_ = lean_string_dec_eq(v_str_116_, v___x_241_);
lean_dec_ref(v_str_116_);
if (v___x_242_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_243_; lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_243_ = lean_array_get_size(v_snd_112_);
v___x_244_ = lean_unsigned_to_nat(6u);
v___x_245_ = lean_nat_dec_eq(v___x_243_, v___x_244_);
if (v___x_245_ == 0)
{
lean_dec(v_snd_112_);
goto v___jp_88_;
}
else
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_246_ = lean_unsigned_to_nat(4u);
v___x_247_ = lean_array_fget(v_snd_112_, v___x_246_);
v___x_248_ = lean_unsigned_to_nat(5u);
v___x_249_ = lean_array_fget(v_snd_112_, v___x_248_);
lean_dec(v_snd_112_);
v_e1_91_ = v___x_247_;
v_e2_92_ = v___x_249_;
goto v___jp_90_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_110_, 2);
lean_dec_ref_known(v_fst_109_, 2);
lean_dec_ref(v___x_108_);
goto v___jp_88_;
}
}
else
{
lean_dec_ref_known(v_fst_109_, 2);
lean_dec(v_pre_110_);
lean_dec_ref(v___x_108_);
goto v___jp_88_;
}
}
else
{
lean_dec(v_fst_109_);
lean_dec_ref(v___x_108_);
goto v___jp_88_;
}
v___jp_88_:
{
lean_object* v___x_89_; 
v___x_89_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor___closed__1));
return v___x_89_;
}
v___jp_90_:
{
lean_object* v___x_93_; lean_object* v_fst_94_; lean_object* v_snd_95_; lean_object* v___x_96_; lean_object* v_fst_97_; lean_object* v_snd_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_107_; 
v___x_93_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(v_e1_91_);
v_fst_94_ = lean_ctor_get(v___x_93_, 0);
lean_inc(v_fst_94_);
v_snd_95_ = lean_ctor_get(v___x_93_, 1);
lean_inc(v_snd_95_);
lean_dec_ref(v___x_93_);
v___x_96_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(v_e2_92_);
v_fst_97_ = lean_ctor_get(v___x_96_, 0);
v_snd_98_ = lean_ctor_get(v___x_96_, 1);
v_isSharedCheck_107_ = !lean_is_exclusive(v___x_96_);
if (v_isSharedCheck_107_ == 0)
{
v___x_100_ = v___x_96_;
v_isShared_101_ = v_isSharedCheck_107_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_snd_98_);
lean_inc(v_fst_97_);
lean_dec(v___x_96_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_107_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v_lcm_102_; lean_object* v___x_103_; lean_object* v___x_105_; 
v_lcm_102_ = l_Nat_lcm(v_fst_94_, v_fst_97_);
lean_dec(v_fst_97_);
lean_dec(v_fst_94_);
lean_inc(v_lcm_102_);
v___x_103_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_103_, 0, v_lcm_102_);
lean_ctor_set(v___x_103_, 1, v_snd_95_);
lean_ctor_set(v___x_103_, 2, v_snd_98_);
if (v_isShared_101_ == 0)
{
lean_ctor_set(v___x_100_, 1, v___x_103_);
lean_ctor_set(v___x_100_, 0, v_lcm_102_);
v___x_105_ = v___x_100_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v_lcm_102_);
lean_ctor_set(v_reuseFailAlloc_106_, 1, v___x_103_);
v___x_105_ = v_reuseFailAlloc_106_;
goto v_reusejp_104_;
}
v_reusejp_104_:
{
return v___x_105_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(lean_object* v_msgData_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_){
_start:
{
lean_object* v___x_258_; lean_object* v_env_259_; lean_object* v___x_260_; lean_object* v_mctx_261_; lean_object* v_lctx_262_; lean_object* v_options_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_258_ = lean_st_ref_get(v___y_256_);
v_env_259_ = lean_ctor_get(v___x_258_, 0);
lean_inc_ref(v_env_259_);
lean_dec(v___x_258_);
v___x_260_ = lean_st_ref_get(v___y_254_);
v_mctx_261_ = lean_ctor_get(v___x_260_, 0);
lean_inc_ref(v_mctx_261_);
lean_dec(v___x_260_);
v_lctx_262_ = lean_ctor_get(v___y_253_, 2);
v_options_263_ = lean_ctor_get(v___y_255_, 2);
lean_inc_ref(v_options_263_);
lean_inc_ref(v_lctx_262_);
v___x_264_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_264_, 0, v_env_259_);
lean_ctor_set(v___x_264_, 1, v_mctx_261_);
lean_ctor_set(v___x_264_, 2, v_lctx_262_);
lean_ctor_set(v___x_264_, 3, v_options_263_);
v___x_265_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
lean_ctor_set(v___x_265_, 1, v_msgData_252_);
v___x_266_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_266_, 0, v___x_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0___boxed(lean_object* v_msgData_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(v_msgData_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(lean_object* v_msg_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_){
_start:
{
lean_object* v_ref_280_; lean_object* v___x_281_; lean_object* v_a_282_; lean_object* v___x_284_; uint8_t v_isShared_285_; uint8_t v_isSharedCheck_290_; 
v_ref_280_ = lean_ctor_get(v___y_277_, 5);
v___x_281_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(v_msg_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_);
v_a_282_ = lean_ctor_get(v___x_281_, 0);
v_isSharedCheck_290_ = !lean_is_exclusive(v___x_281_);
if (v_isSharedCheck_290_ == 0)
{
v___x_284_ = v___x_281_;
v_isShared_285_ = v_isSharedCheck_290_;
goto v_resetjp_283_;
}
else
{
lean_inc(v_a_282_);
lean_dec(v___x_281_);
v___x_284_ = lean_box(0);
v_isShared_285_ = v_isSharedCheck_290_;
goto v_resetjp_283_;
}
v_resetjp_283_:
{
lean_object* v___x_286_; lean_object* v___x_288_; 
lean_inc(v_ref_280_);
v___x_286_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_286_, 0, v_ref_280_);
lean_ctor_set(v___x_286_, 1, v_a_282_);
if (v_isShared_285_ == 0)
{
lean_ctor_set_tag(v___x_284_, 1);
lean_ctor_set(v___x_284_, 0, v___x_286_);
v___x_288_ = v___x_284_;
goto v_reusejp_287_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v___x_286_);
v___x_288_ = v_reuseFailAlloc_289_;
goto v_reusejp_287_;
}
v_reusejp_287_:
{
return v___x_288_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg___boxed(lean_object* v_msg_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(v_msg_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_);
lean_dec(v___y_295_);
lean_dec_ref(v___y_294_);
lean_dec(v___y_293_);
lean_dec_ref(v___y_292_);
return v_res_297_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9(void){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = l_Array_mkArray0(lean_box(0));
return v___x_315_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__11(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; 
v___x_317_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__10));
v___x_318_ = l_Lean_stringToMessageData(v___x_317_);
return v___x_318_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__13(void){
_start:
{
lean_object* v___x_320_; lean_object* v___x_321_; 
v___x_320_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__12));
v___x_321_ = l_Lean_stringToMessageData(v___x_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(lean_object* v_type_322_, lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_){
_start:
{
lean_object* v_ref_328_; uint8_t v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v_ref_328_ = lean_ctor_get(v_a_325_, 5);
v___x_329_ = 0;
v___x_330_ = l_Lean_SourceInfo_fromRef(v_ref_328_, v___x_329_);
v___x_331_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1));
v___x_332_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__2));
lean_inc_n(v___x_330_, 3);
v___x_333_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_330_);
lean_ctor_set(v___x_333_, 1, v___x_332_);
v___x_334_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6));
v___x_335_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__8));
v___x_336_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9, &lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9);
v___x_337_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_337_, 0, v___x_330_);
lean_ctor_set(v___x_337_, 1, v___x_335_);
lean_ctor_set(v___x_337_, 2, v___x_336_);
lean_inc_ref_n(v___x_337_, 3);
v___x_338_ = l_Lean_Syntax_node1(v___x_330_, v___x_334_, v___x_337_);
v___x_339_ = l_Lean_Syntax_node5(v___x_330_, v___x_331_, v___x_333_, v___x_338_, v___x_337_, v___x_337_, v___x_337_);
lean_inc_ref(v_type_322_);
v___x_340_ = lp_mathlib_synthesizeUsingTactic_x27___redArg(v_type_322_, v___x_339_, v_a_323_, v_a_324_, v_a_325_, v_a_326_);
if (lean_obj_tag(v___x_340_) == 0)
{
lean_dec_ref(v_type_322_);
return v___x_340_;
}
else
{
lean_object* v_a_341_; uint8_t v___y_343_; uint8_t v___x_352_; 
v_a_341_ = lean_ctor_get(v___x_340_, 0);
lean_inc(v_a_341_);
v___x_352_ = l_Lean_Exception_isInterrupt(v_a_341_);
if (v___x_352_ == 0)
{
uint8_t v___x_353_; 
lean_inc(v_a_341_);
v___x_353_ = l_Lean_Exception_isRuntime(v_a_341_);
v___y_343_ = v___x_353_;
goto v___jp_342_;
}
else
{
v___y_343_ = v___x_352_;
goto v___jp_342_;
}
v___jp_342_:
{
if (v___y_343_ == 0)
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
lean_dec_ref_known(v___x_340_, 1);
v___x_344_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__11, &lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__11);
v___x_345_ = l_Lean_MessageData_ofExpr(v_type_322_);
v___x_346_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_344_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__13, &lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__13);
v___x_348_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_346_);
lean_ctor_set(v___x_348_, 1, v___x_347_);
v___x_349_ = l_Lean_Exception_toMessageData(v_a_341_);
v___x_350_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_348_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(v___x_350_, v_a_323_, v_a_324_, v_a_325_, v_a_326_);
return v___x_351_;
}
else
{
lean_dec(v_a_341_);
lean_dec_ref(v_type_322_);
return v___x_340_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___boxed(lean_object* v_type_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_, lean_object* v_a_358_, lean_object* v_a_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v_type_354_, v_a_355_, v_a_356_, v_a_357_, v_a_358_);
lean_dec(v_a_358_);
lean_dec_ref(v_a_357_);
lean_dec(v_a_356_);
lean_dec_ref(v_a_355_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0(lean_object* v_00_u03b1_361_, lean_object* v_msg_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(v_msg_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___boxed(lean_object* v_00_u03b1_369_, lean_object* v_msg_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0(v_00_u03b1_369_, v_msg_370_, v___y_371_, v___y_372_, v___y_373_, v___y_374_);
lean_dec(v___y_374_);
lean_dec_ref(v___y_373_);
lean_dec(v___y_372_);
lean_dec_ref(v___y_371_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(lean_object* v_e_377_, lean_object* v___y_378_){
_start:
{
uint8_t v___x_380_; 
v___x_380_ = l_Lean_Expr_hasMVar(v_e_377_);
if (v___x_380_ == 0)
{
lean_object* v___x_381_; 
v___x_381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_381_, 0, v_e_377_);
return v___x_381_;
}
else
{
lean_object* v___x_382_; lean_object* v_mctx_383_; lean_object* v___x_384_; lean_object* v_fst_385_; lean_object* v_snd_386_; lean_object* v___x_387_; lean_object* v_cache_388_; lean_object* v_zetaDeltaFVarIds_389_; lean_object* v_postponed_390_; lean_object* v_diag_391_; lean_object* v___x_393_; uint8_t v_isShared_394_; uint8_t v_isSharedCheck_400_; 
v___x_382_ = lean_st_ref_get(v___y_378_);
v_mctx_383_ = lean_ctor_get(v___x_382_, 0);
lean_inc_ref(v_mctx_383_);
lean_dec(v___x_382_);
v___x_384_ = l_Lean_instantiateMVarsCore(v_mctx_383_, v_e_377_);
v_fst_385_ = lean_ctor_get(v___x_384_, 0);
lean_inc(v_fst_385_);
v_snd_386_ = lean_ctor_get(v___x_384_, 1);
lean_inc(v_snd_386_);
lean_dec_ref(v___x_384_);
v___x_387_ = lean_st_ref_take(v___y_378_);
v_cache_388_ = lean_ctor_get(v___x_387_, 1);
v_zetaDeltaFVarIds_389_ = lean_ctor_get(v___x_387_, 2);
v_postponed_390_ = lean_ctor_get(v___x_387_, 3);
v_diag_391_ = lean_ctor_get(v___x_387_, 4);
v_isSharedCheck_400_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_400_ == 0)
{
lean_object* v_unused_401_; 
v_unused_401_ = lean_ctor_get(v___x_387_, 0);
lean_dec(v_unused_401_);
v___x_393_ = v___x_387_;
v_isShared_394_ = v_isSharedCheck_400_;
goto v_resetjp_392_;
}
else
{
lean_inc(v_diag_391_);
lean_inc(v_postponed_390_);
lean_inc(v_zetaDeltaFVarIds_389_);
lean_inc(v_cache_388_);
lean_dec(v___x_387_);
v___x_393_ = lean_box(0);
v_isShared_394_ = v_isSharedCheck_400_;
goto v_resetjp_392_;
}
v_resetjp_392_:
{
lean_object* v___x_396_; 
if (v_isShared_394_ == 0)
{
lean_ctor_set(v___x_393_, 0, v_snd_386_);
v___x_396_ = v___x_393_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v_snd_386_);
lean_ctor_set(v_reuseFailAlloc_399_, 1, v_cache_388_);
lean_ctor_set(v_reuseFailAlloc_399_, 2, v_zetaDeltaFVarIds_389_);
lean_ctor_set(v_reuseFailAlloc_399_, 3, v_postponed_390_);
lean_ctor_set(v_reuseFailAlloc_399_, 4, v_diag_391_);
v___x_396_ = v_reuseFailAlloc_399_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
lean_object* v___x_397_; lean_object* v___x_398_; 
v___x_397_ = lean_st_ref_set(v___y_378_, v___x_396_);
v___x_398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_398_, 0, v_fst_385_);
return v___x_398_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg___boxed(lean_object* v_e_402_, lean_object* v___y_403_, lean_object* v___y_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_e_402_, v___y_403_);
lean_dec(v___y_403_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0(lean_object* v_e_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_e_406_, v___y_408_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___boxed(lean_object* v_e_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_){
_start:
{
lean_object* v_res_419_; 
v_res_419_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0(v_e_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_);
lean_dec(v___y_417_);
lean_dec_ref(v___y_416_);
lean_dec(v___y_415_);
lean_dec_ref(v___y_414_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(lean_object* v_k_420_, uint8_t v_allowLevelAssignments_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_421_, v_k_420_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
if (lean_obj_tag(v___x_427_) == 0)
{
lean_object* v_a_428_; lean_object* v___x_430_; uint8_t v_isShared_431_; uint8_t v_isSharedCheck_435_; 
v_a_428_ = lean_ctor_get(v___x_427_, 0);
v_isSharedCheck_435_ = !lean_is_exclusive(v___x_427_);
if (v_isSharedCheck_435_ == 0)
{
v___x_430_ = v___x_427_;
v_isShared_431_ = v_isSharedCheck_435_;
goto v_resetjp_429_;
}
else
{
lean_inc(v_a_428_);
lean_dec(v___x_427_);
v___x_430_ = lean_box(0);
v_isShared_431_ = v_isSharedCheck_435_;
goto v_resetjp_429_;
}
v_resetjp_429_:
{
lean_object* v___x_433_; 
if (v_isShared_431_ == 0)
{
v___x_433_ = v___x_430_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_a_428_);
v___x_433_ = v_reuseFailAlloc_434_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
return v___x_433_;
}
}
}
else
{
lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_443_; 
v_a_436_ = lean_ctor_get(v___x_427_, 0);
v_isSharedCheck_443_ = !lean_is_exclusive(v___x_427_);
if (v_isSharedCheck_443_ == 0)
{
v___x_438_ = v___x_427_;
v_isShared_439_ = v_isSharedCheck_443_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v___x_427_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_443_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_441_; 
if (v_isShared_439_ == 0)
{
v___x_441_ = v___x_438_;
goto v_reusejp_440_;
}
else
{
lean_object* v_reuseFailAlloc_442_; 
v_reuseFailAlloc_442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_442_, 0, v_a_436_);
v___x_441_ = v_reuseFailAlloc_442_;
goto v_reusejp_440_;
}
v_reusejp_440_:
{
return v___x_441_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg___boxed(lean_object* v_k_444_, lean_object* v_allowLevelAssignments_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_451_; lean_object* v_res_452_; 
v_allowLevelAssignments_boxed_451_ = lean_unbox(v_allowLevelAssignments_445_);
v_res_452_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v_k_444_, v_allowLevelAssignments_boxed_451_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1(lean_object* v_00_u03b1_453_, lean_object* v_k_454_, uint8_t v_allowLevelAssignments_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v___x_461_; 
v___x_461_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v_k_454_, v_allowLevelAssignments_455_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___boxed(lean_object* v_00_u03b1_462_, lean_object* v_k_463_, lean_object* v_allowLevelAssignments_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_470_; lean_object* v_res_471_; 
v_allowLevelAssignments_boxed_470_ = lean_unbox(v_allowLevelAssignments_464_);
v_res_471_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1(v_00_u03b1_462_, v_k_463_, v_allowLevelAssignments_boxed_470_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
lean_dec(v___y_468_);
lean_dec_ref(v___y_467_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0(lean_object* v___x_501_, uint8_t v___x_502_, lean_object* v___x_503_, lean_object* v___x_504_, lean_object* v_00_u03b1_505_, lean_object* v___x_506_, lean_object* v___x_507_, lean_object* v_e_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_){
_start:
{
lean_object* v___x_514_; 
v___x_514_ = l_Lean_Meta_mkFreshExprMVar(v___x_501_, v___x_502_, v___x_503_, v___y_509_, v___y_510_, v___y_511_, v___y_512_);
if (lean_obj_tag(v___x_514_) == 0)
{
lean_object* v_a_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v_keyedConfig_525_; uint8_t v_trackZetaDelta_526_; lean_object* v_zetaDeltaSet_527_; lean_object* v_lctx_528_; lean_object* v_localInstances_529_; lean_object* v_defEqCtx_x3f_530_; lean_object* v_synthPendingDepth_531_; lean_object* v_customCanUnfoldPredicate_x3f_532_; uint8_t v_univApprox_533_; uint8_t v_inTypeClassResolution_534_; uint8_t v_cacheInferType_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_602_; 
v_a_515_ = lean_ctor_get(v___x_514_, 0);
lean_inc(v_a_515_);
lean_dec_ref_known(v___x_514_, 1);
v___x_516_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__0));
lean_inc_n(v___x_504_, 3);
v___x_517_ = l_Lean_Expr_const___override(v___x_516_, v___x_504_);
lean_inc_ref_n(v_00_u03b1_505_, 3);
v___x_518_ = l_Lean_Expr_app___override(v___x_517_, v_00_u03b1_505_);
v___x_519_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__3));
v___x_520_ = l_Lean_Expr_const___override(v___x_519_, v___x_504_);
v___x_521_ = l_Lean_Expr_app___override(v___x_520_, v_00_u03b1_505_);
v___x_522_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__6));
v___x_523_ = l_Lean_Expr_const___override(v___x_522_, v___x_504_);
v___x_524_ = l_Lean_Expr_app___override(v___x_523_, v_00_u03b1_505_);
v_keyedConfig_525_ = lean_ctor_get(v___y_509_, 0);
v_trackZetaDelta_526_ = lean_ctor_get_uint8(v___y_509_, sizeof(void*)*7);
v_zetaDeltaSet_527_ = lean_ctor_get(v___y_509_, 1);
v_lctx_528_ = lean_ctor_get(v___y_509_, 2);
v_localInstances_529_ = lean_ctor_get(v___y_509_, 3);
v_defEqCtx_x3f_530_ = lean_ctor_get(v___y_509_, 4);
v_synthPendingDepth_531_ = lean_ctor_get(v___y_509_, 5);
v_customCanUnfoldPredicate_x3f_532_ = lean_ctor_get(v___y_509_, 6);
v_univApprox_533_ = lean_ctor_get_uint8(v___y_509_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_534_ = lean_ctor_get_uint8(v___y_509_, sizeof(void*)*7 + 2);
v_cacheInferType_535_ = lean_ctor_get_uint8(v___y_509_, sizeof(void*)*7 + 3);
v_isSharedCheck_602_ = !lean_is_exclusive(v___y_509_);
if (v_isSharedCheck_602_ == 0)
{
v___x_537_ = v___y_509_;
v_isShared_538_ = v_isSharedCheck_602_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_532_);
lean_inc(v_synthPendingDepth_531_);
lean_inc(v_defEqCtx_x3f_530_);
lean_inc(v_localInstances_529_);
lean_inc(v_lctx_528_);
lean_inc(v_zetaDeltaSet_527_);
lean_inc(v_keyedConfig_525_);
lean_dec(v___y_509_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_602_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; uint8_t v___x_560_; lean_object* v___x_561_; lean_object* v___x_563_; 
v___x_539_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__9));
lean_inc_n(v___x_504_, 3);
v___x_540_ = l_Lean_Expr_const___override(v___x_539_, v___x_504_);
lean_inc_ref_n(v_00_u03b1_505_, 3);
v___x_541_ = l_Lean_Expr_app___override(v___x_540_, v_00_u03b1_505_);
v___x_542_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__12));
v___x_543_ = l_Lean_Expr_const___override(v___x_542_, v___x_504_);
v___x_544_ = l_Lean_Expr_app___override(v___x_543_, v_00_u03b1_505_);
v___x_545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__15));
v___x_546_ = l_Lean_Expr_const___override(v___x_545_, v___x_504_);
v___x_547_ = l_Lean_Expr_app___override(v___x_546_, v_00_u03b1_505_);
v___x_548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___closed__16));
v___x_549_ = l_Lean_Name_mkStr2(v___x_506_, v___x_548_);
v___x_550_ = l_Lean_Expr_const___override(v___x_549_, v___x_504_);
v___x_551_ = l_Lean_Expr_app___override(v___x_550_, v_00_u03b1_505_);
v___x_552_ = l_Lean_Expr_app___override(v___x_551_, v___x_507_);
v___x_553_ = l_Lean_Expr_app___override(v___x_547_, v___x_552_);
v___x_554_ = l_Lean_Expr_app___override(v___x_544_, v___x_553_);
v___x_555_ = l_Lean_Expr_app___override(v___x_541_, v___x_554_);
v___x_556_ = l_Lean_Expr_app___override(v___x_524_, v___x_555_);
v___x_557_ = l_Lean_Expr_app___override(v___x_521_, v___x_556_);
v___x_558_ = l_Lean_Expr_app___override(v___x_518_, v___x_557_);
lean_inc(v_a_515_);
v___x_559_ = l_Lean_Expr_app___override(v___x_558_, v_a_515_);
v___x_560_ = 2;
v___x_561_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_560_, v_keyedConfig_525_);
if (v_isShared_538_ == 0)
{
lean_ctor_set(v___x_537_, 0, v___x_561_);
v___x_563_ = v___x_537_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v___x_561_);
lean_ctor_set(v_reuseFailAlloc_601_, 1, v_zetaDeltaSet_527_);
lean_ctor_set(v_reuseFailAlloc_601_, 2, v_lctx_528_);
lean_ctor_set(v_reuseFailAlloc_601_, 3, v_localInstances_529_);
lean_ctor_set(v_reuseFailAlloc_601_, 4, v_defEqCtx_x3f_530_);
lean_ctor_set(v_reuseFailAlloc_601_, 5, v_synthPendingDepth_531_);
lean_ctor_set(v_reuseFailAlloc_601_, 6, v_customCanUnfoldPredicate_x3f_532_);
lean_ctor_set_uint8(v_reuseFailAlloc_601_, sizeof(void*)*7, v_trackZetaDelta_526_);
lean_ctor_set_uint8(v_reuseFailAlloc_601_, sizeof(void*)*7 + 1, v_univApprox_533_);
lean_ctor_set_uint8(v_reuseFailAlloc_601_, sizeof(void*)*7 + 2, v_inTypeClassResolution_534_);
lean_ctor_set_uint8(v_reuseFailAlloc_601_, sizeof(void*)*7 + 3, v_cacheInferType_535_);
v___x_563_ = v_reuseFailAlloc_601_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
lean_object* v___x_564_; 
v___x_564_ = l_Lean_Meta_isExprDefEq(v___x_559_, v_e_508_, v___x_563_, v___y_510_, v___y_511_, v___y_512_);
lean_dec_ref(v___x_563_);
if (lean_obj_tag(v___x_564_) == 0)
{
lean_object* v_a_565_; lean_object* v___x_567_; uint8_t v_isShared_568_; uint8_t v_isSharedCheck_592_; 
v_a_565_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_592_ == 0)
{
v___x_567_ = v___x_564_;
v_isShared_568_ = v_isSharedCheck_592_;
goto v_resetjp_566_;
}
else
{
lean_inc(v_a_565_);
lean_dec(v___x_564_);
v___x_567_ = lean_box(0);
v_isShared_568_ = v_isSharedCheck_592_;
goto v_resetjp_566_;
}
v_resetjp_566_:
{
uint8_t v___x_569_; 
v___x_569_ = lean_unbox(v_a_565_);
if (v___x_569_ == 0)
{
lean_object* v___x_570_; lean_object* v___x_572_; 
v___x_570_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_570_, 0, v_a_515_);
lean_ctor_set(v___x_570_, 1, v_a_565_);
if (v_isShared_568_ == 0)
{
lean_ctor_set(v___x_567_, 0, v___x_570_);
v___x_572_ = v___x_567_;
goto v_reusejp_571_;
}
else
{
lean_object* v_reuseFailAlloc_573_; 
v_reuseFailAlloc_573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_573_, 0, v___x_570_);
v___x_572_ = v_reuseFailAlloc_573_;
goto v_reusejp_571_;
}
v_reusejp_571_:
{
return v___x_572_;
}
}
else
{
lean_object* v___x_574_; 
lean_del_object(v___x_567_);
v___x_574_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_515_, v___y_510_);
if (lean_obj_tag(v___x_574_) == 0)
{
lean_object* v_a_575_; lean_object* v___x_577_; uint8_t v_isShared_578_; uint8_t v_isSharedCheck_583_; 
v_a_575_ = lean_ctor_get(v___x_574_, 0);
v_isSharedCheck_583_ = !lean_is_exclusive(v___x_574_);
if (v_isSharedCheck_583_ == 0)
{
v___x_577_ = v___x_574_;
v_isShared_578_ = v_isSharedCheck_583_;
goto v_resetjp_576_;
}
else
{
lean_inc(v_a_575_);
lean_dec(v___x_574_);
v___x_577_ = lean_box(0);
v_isShared_578_ = v_isSharedCheck_583_;
goto v_resetjp_576_;
}
v_resetjp_576_:
{
lean_object* v___x_579_; lean_object* v___x_581_; 
v___x_579_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_579_, 0, v_a_575_);
lean_ctor_set(v___x_579_, 1, v_a_565_);
if (v_isShared_578_ == 0)
{
lean_ctor_set(v___x_577_, 0, v___x_579_);
v___x_581_ = v___x_577_;
goto v_reusejp_580_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v___x_579_);
v___x_581_ = v_reuseFailAlloc_582_;
goto v_reusejp_580_;
}
v_reusejp_580_:
{
return v___x_581_;
}
}
}
else
{
lean_object* v_a_584_; lean_object* v___x_586_; uint8_t v_isShared_587_; uint8_t v_isSharedCheck_591_; 
lean_dec(v_a_565_);
v_a_584_ = lean_ctor_get(v___x_574_, 0);
v_isSharedCheck_591_ = !lean_is_exclusive(v___x_574_);
if (v_isSharedCheck_591_ == 0)
{
v___x_586_ = v___x_574_;
v_isShared_587_ = v_isSharedCheck_591_;
goto v_resetjp_585_;
}
else
{
lean_inc(v_a_584_);
lean_dec(v___x_574_);
v___x_586_ = lean_box(0);
v_isShared_587_ = v_isSharedCheck_591_;
goto v_resetjp_585_;
}
v_resetjp_585_:
{
lean_object* v___x_589_; 
if (v_isShared_587_ == 0)
{
v___x_589_ = v___x_586_;
goto v_reusejp_588_;
}
else
{
lean_object* v_reuseFailAlloc_590_; 
v_reuseFailAlloc_590_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_590_, 0, v_a_584_);
v___x_589_ = v_reuseFailAlloc_590_;
goto v_reusejp_588_;
}
v_reusejp_588_:
{
return v___x_589_;
}
}
}
}
}
}
else
{
lean_object* v_a_593_; lean_object* v___x_595_; uint8_t v_isShared_596_; uint8_t v_isSharedCheck_600_; 
lean_dec(v_a_515_);
v_a_593_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_600_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_600_ == 0)
{
v___x_595_ = v___x_564_;
v_isShared_596_ = v_isSharedCheck_600_;
goto v_resetjp_594_;
}
else
{
lean_inc(v_a_593_);
lean_dec(v___x_564_);
v___x_595_ = lean_box(0);
v_isShared_596_ = v_isSharedCheck_600_;
goto v_resetjp_594_;
}
v_resetjp_594_:
{
lean_object* v___x_598_; 
if (v_isShared_596_ == 0)
{
v___x_598_ = v___x_595_;
goto v_reusejp_597_;
}
else
{
lean_object* v_reuseFailAlloc_599_; 
v_reuseFailAlloc_599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_599_, 0, v_a_593_);
v___x_598_ = v_reuseFailAlloc_599_;
goto v_reusejp_597_;
}
v_reusejp_597_:
{
return v___x_598_;
}
}
}
}
}
}
else
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_610_; 
lean_dec_ref(v___y_509_);
lean_dec_ref(v_e_508_);
lean_dec_ref(v___x_507_);
lean_dec_ref(v___x_506_);
lean_dec_ref(v_00_u03b1_505_);
lean_dec(v___x_504_);
v_a_603_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_610_ == 0)
{
v___x_605_ = v___x_514_;
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_514_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_610_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_608_; 
if (v_isShared_606_ == 0)
{
v___x_608_ = v___x_605_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_609_; 
v_reuseFailAlloc_609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_609_, 0, v_a_603_);
v___x_608_ = v_reuseFailAlloc_609_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
return v___x_608_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___boxed(lean_object* v___x_611_, lean_object* v___x_612_, lean_object* v___x_613_, lean_object* v___x_614_, lean_object* v_00_u03b1_615_, lean_object* v___x_616_, lean_object* v___x_617_, lean_object* v_e_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_){
_start:
{
uint8_t v___x_24679__boxed_624_; lean_object* v_res_625_; 
v___x_24679__boxed_624_ = lean_unbox(v___x_612_);
v_res_625_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0(v___x_611_, v___x_24679__boxed_624_, v___x_613_, v___x_614_, v_00_u03b1_615_, v___x_616_, v___x_617_, v_e_618_, v___y_619_, v___y_620_, v___y_621_, v___y_622_);
lean_dec(v___y_622_);
lean_dec_ref(v___y_621_);
lean_dec(v___y_620_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1(lean_object* v___x_650_, uint8_t v___x_651_, lean_object* v___x_652_, lean_object* v___x_653_, lean_object* v___x_654_, lean_object* v_u_655_, lean_object* v_00_u03b1_656_, lean_object* v___x_657_, lean_object* v_e_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_){
_start:
{
lean_object* v___x_664_; 
lean_inc(v___x_652_);
v___x_664_ = l_Lean_Meta_mkFreshExprMVar(v___x_650_, v___x_651_, v___x_652_, v___y_659_, v___y_660_, v___y_661_, v___y_662_);
if (lean_obj_tag(v___x_664_) == 0)
{
lean_object* v_a_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; 
v_a_665_ = lean_ctor_get(v___x_664_, 0);
lean_inc(v_a_665_);
lean_dec_ref_known(v___x_664_, 1);
v___x_666_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__1));
lean_inc(v___x_653_);
v___x_667_ = l_Lean_Expr_const___override(v___x_666_, v___x_653_);
lean_inc_ref(v___x_667_);
v___x_668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_668_, 0, v___x_667_);
v___x_669_ = l_Lean_Meta_mkFreshExprMVar(v___x_668_, v___x_651_, v___x_652_, v___y_659_, v___y_660_, v___y_661_, v___y_662_);
if (lean_obj_tag(v___x_669_) == 0)
{
lean_object* v_a_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v_keyedConfig_679_; uint8_t v_trackZetaDelta_680_; lean_object* v_zetaDeltaSet_681_; lean_object* v_lctx_682_; lean_object* v_localInstances_683_; lean_object* v_defEqCtx_x3f_684_; lean_object* v_synthPendingDepth_685_; lean_object* v_customCanUnfoldPredicate_x3f_686_; uint8_t v_univApprox_687_; uint8_t v_inTypeClassResolution_688_; uint8_t v_cacheInferType_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_769_; 
v_a_670_ = lean_ctor_get(v___x_669_, 0);
lean_inc(v_a_670_);
lean_dec_ref_known(v___x_669_, 1);
v___x_671_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__2));
v___x_672_ = lean_box(0);
lean_inc(v___x_654_);
v___x_673_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_673_, 0, v___x_672_);
lean_ctor_set(v___x_673_, 1, v___x_654_);
lean_inc(v_u_655_);
v___x_674_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_674_, 0, v_u_655_);
lean_ctor_set(v___x_674_, 1, v___x_673_);
v___x_675_ = l_Lean_Expr_const___override(v___x_671_, v___x_674_);
lean_inc_ref_n(v_00_u03b1_656_, 2);
v___x_676_ = l_Lean_Expr_app___override(v___x_675_, v_00_u03b1_656_);
lean_inc_ref(v___x_667_);
v___x_677_ = l_Lean_Expr_app___override(v___x_676_, v___x_667_);
v___x_678_ = l_Lean_Expr_app___override(v___x_677_, v_00_u03b1_656_);
v_keyedConfig_679_ = lean_ctor_get(v___y_659_, 0);
v_trackZetaDelta_680_ = lean_ctor_get_uint8(v___y_659_, sizeof(void*)*7);
v_zetaDeltaSet_681_ = lean_ctor_get(v___y_659_, 1);
v_lctx_682_ = lean_ctor_get(v___y_659_, 2);
v_localInstances_683_ = lean_ctor_get(v___y_659_, 3);
v_defEqCtx_x3f_684_ = lean_ctor_get(v___y_659_, 4);
v_synthPendingDepth_685_ = lean_ctor_get(v___y_659_, 5);
v_customCanUnfoldPredicate_x3f_686_ = lean_ctor_get(v___y_659_, 6);
v_univApprox_687_ = lean_ctor_get_uint8(v___y_659_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_688_ = lean_ctor_get_uint8(v___y_659_, sizeof(void*)*7 + 2);
v_cacheInferType_689_ = lean_ctor_get_uint8(v___y_659_, sizeof(void*)*7 + 3);
v_isSharedCheck_769_ = !lean_is_exclusive(v___y_659_);
if (v_isSharedCheck_769_ == 0)
{
v___x_691_ = v___y_659_;
v_isShared_692_ = v_isSharedCheck_769_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_686_);
lean_inc(v_synthPendingDepth_685_);
lean_inc(v_defEqCtx_x3f_684_);
lean_inc(v_localInstances_683_);
lean_inc(v_lctx_682_);
lean_inc(v_zetaDeltaSet_681_);
lean_inc(v_keyedConfig_679_);
lean_dec(v___y_659_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_769_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; uint8_t v___x_715_; lean_object* v___x_716_; lean_object* v___x_718_; 
v___x_693_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__4));
v___x_694_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_694_, 0, v___x_672_);
lean_ctor_set(v___x_694_, 1, v___x_653_);
v___x_695_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_695_, 0, v_u_655_);
lean_ctor_set(v___x_695_, 1, v___x_694_);
v___x_696_ = l_Lean_Expr_const___override(v___x_693_, v___x_695_);
lean_inc_ref_n(v_00_u03b1_656_, 3);
v___x_697_ = l_Lean_Expr_app___override(v___x_696_, v_00_u03b1_656_);
v___x_698_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__7));
lean_inc_n(v___x_654_, 2);
v___x_699_ = l_Lean_Expr_const___override(v___x_698_, v___x_654_);
v___x_700_ = l_Lean_Expr_app___override(v___x_699_, v_00_u03b1_656_);
v___x_701_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__10));
v___x_702_ = l_Lean_Expr_const___override(v___x_701_, v___x_654_);
v___x_703_ = l_Lean_Expr_app___override(v___x_702_, v_00_u03b1_656_);
v___x_704_ = l_Lean_Expr_app___override(v___x_697_, v___x_667_);
v___x_705_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__13));
v___x_706_ = l_Lean_Expr_const___override(v___x_705_, v___x_654_);
v___x_707_ = l_Lean_Expr_app___override(v___x_706_, v_00_u03b1_656_);
v___x_708_ = l_Lean_Expr_app___override(v___x_707_, v___x_657_);
v___x_709_ = l_Lean_Expr_app___override(v___x_703_, v___x_708_);
v___x_710_ = l_Lean_Expr_app___override(v___x_700_, v___x_709_);
v___x_711_ = l_Lean_Expr_app___override(v___x_704_, v___x_710_);
v___x_712_ = l_Lean_Expr_app___override(v___x_678_, v___x_711_);
lean_inc(v_a_665_);
v___x_713_ = l_Lean_Expr_app___override(v___x_712_, v_a_665_);
lean_inc(v_a_670_);
v___x_714_ = l_Lean_Expr_app___override(v___x_713_, v_a_670_);
v___x_715_ = 2;
v___x_716_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_715_, v_keyedConfig_679_);
if (v_isShared_692_ == 0)
{
lean_ctor_set(v___x_691_, 0, v___x_716_);
v___x_718_ = v___x_691_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v___x_716_);
lean_ctor_set(v_reuseFailAlloc_768_, 1, v_zetaDeltaSet_681_);
lean_ctor_set(v_reuseFailAlloc_768_, 2, v_lctx_682_);
lean_ctor_set(v_reuseFailAlloc_768_, 3, v_localInstances_683_);
lean_ctor_set(v_reuseFailAlloc_768_, 4, v_defEqCtx_x3f_684_);
lean_ctor_set(v_reuseFailAlloc_768_, 5, v_synthPendingDepth_685_);
lean_ctor_set(v_reuseFailAlloc_768_, 6, v_customCanUnfoldPredicate_x3f_686_);
lean_ctor_set_uint8(v_reuseFailAlloc_768_, sizeof(void*)*7, v_trackZetaDelta_680_);
lean_ctor_set_uint8(v_reuseFailAlloc_768_, sizeof(void*)*7 + 1, v_univApprox_687_);
lean_ctor_set_uint8(v_reuseFailAlloc_768_, sizeof(void*)*7 + 2, v_inTypeClassResolution_688_);
lean_ctor_set_uint8(v_reuseFailAlloc_768_, sizeof(void*)*7 + 3, v_cacheInferType_689_);
v___x_718_ = v_reuseFailAlloc_768_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
lean_object* v___x_719_; 
v___x_719_ = l_Lean_Meta_isExprDefEq(v___x_714_, v_e_658_, v___x_718_, v___y_660_, v___y_661_, v___y_662_);
lean_dec_ref(v___x_718_);
if (lean_obj_tag(v___x_719_) == 0)
{
lean_object* v_a_720_; lean_object* v___x_722_; uint8_t v_isShared_723_; uint8_t v_isSharedCheck_759_; 
v_a_720_ = lean_ctor_get(v___x_719_, 0);
v_isSharedCheck_759_ = !lean_is_exclusive(v___x_719_);
if (v_isSharedCheck_759_ == 0)
{
v___x_722_ = v___x_719_;
v_isShared_723_ = v_isSharedCheck_759_;
goto v_resetjp_721_;
}
else
{
lean_inc(v_a_720_);
lean_dec(v___x_719_);
v___x_722_ = lean_box(0);
v_isShared_723_ = v_isSharedCheck_759_;
goto v_resetjp_721_;
}
v_resetjp_721_:
{
uint8_t v___x_724_; 
v___x_724_ = lean_unbox(v_a_720_);
if (v___x_724_ == 0)
{
lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_728_; 
v___x_725_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_725_, 0, v_a_670_);
lean_ctor_set(v___x_725_, 1, v_a_720_);
v___x_726_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_726_, 0, v_a_665_);
lean_ctor_set(v___x_726_, 1, v___x_725_);
if (v_isShared_723_ == 0)
{
lean_ctor_set(v___x_722_, 0, v___x_726_);
v___x_728_ = v___x_722_;
goto v_reusejp_727_;
}
else
{
lean_object* v_reuseFailAlloc_729_; 
v_reuseFailAlloc_729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_729_, 0, v___x_726_);
v___x_728_ = v_reuseFailAlloc_729_;
goto v_reusejp_727_;
}
v_reusejp_727_:
{
return v___x_728_;
}
}
else
{
lean_object* v___x_730_; 
lean_del_object(v___x_722_);
v___x_730_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_665_, v___y_660_);
if (lean_obj_tag(v___x_730_) == 0)
{
lean_object* v_a_731_; lean_object* v___x_732_; 
v_a_731_ = lean_ctor_get(v___x_730_, 0);
lean_inc(v_a_731_);
lean_dec_ref_known(v___x_730_, 1);
v___x_732_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_670_, v___y_660_);
if (lean_obj_tag(v___x_732_) == 0)
{
lean_object* v_a_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_742_; 
v_a_733_ = lean_ctor_get(v___x_732_, 0);
v_isSharedCheck_742_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_742_ == 0)
{
v___x_735_ = v___x_732_;
v_isShared_736_ = v_isSharedCheck_742_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_a_733_);
lean_dec(v___x_732_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_742_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_740_; 
v___x_737_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_737_, 0, v_a_733_);
lean_ctor_set(v___x_737_, 1, v_a_720_);
v___x_738_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_738_, 0, v_a_731_);
lean_ctor_set(v___x_738_, 1, v___x_737_);
if (v_isShared_736_ == 0)
{
lean_ctor_set(v___x_735_, 0, v___x_738_);
v___x_740_ = v___x_735_;
goto v_reusejp_739_;
}
else
{
lean_object* v_reuseFailAlloc_741_; 
v_reuseFailAlloc_741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_741_, 0, v___x_738_);
v___x_740_ = v_reuseFailAlloc_741_;
goto v_reusejp_739_;
}
v_reusejp_739_:
{
return v___x_740_;
}
}
}
else
{
lean_object* v_a_743_; lean_object* v___x_745_; uint8_t v_isShared_746_; uint8_t v_isSharedCheck_750_; 
lean_dec(v_a_731_);
lean_dec(v_a_720_);
v_a_743_ = lean_ctor_get(v___x_732_, 0);
v_isSharedCheck_750_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_750_ == 0)
{
v___x_745_ = v___x_732_;
v_isShared_746_ = v_isSharedCheck_750_;
goto v_resetjp_744_;
}
else
{
lean_inc(v_a_743_);
lean_dec(v___x_732_);
v___x_745_ = lean_box(0);
v_isShared_746_ = v_isSharedCheck_750_;
goto v_resetjp_744_;
}
v_resetjp_744_:
{
lean_object* v___x_748_; 
if (v_isShared_746_ == 0)
{
v___x_748_ = v___x_745_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_749_; 
v_reuseFailAlloc_749_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_749_, 0, v_a_743_);
v___x_748_ = v_reuseFailAlloc_749_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
return v___x_748_;
}
}
}
}
else
{
lean_object* v_a_751_; lean_object* v___x_753_; uint8_t v_isShared_754_; uint8_t v_isSharedCheck_758_; 
lean_dec(v_a_720_);
lean_dec(v_a_670_);
v_a_751_ = lean_ctor_get(v___x_730_, 0);
v_isSharedCheck_758_ = !lean_is_exclusive(v___x_730_);
if (v_isSharedCheck_758_ == 0)
{
v___x_753_ = v___x_730_;
v_isShared_754_ = v_isSharedCheck_758_;
goto v_resetjp_752_;
}
else
{
lean_inc(v_a_751_);
lean_dec(v___x_730_);
v___x_753_ = lean_box(0);
v_isShared_754_ = v_isSharedCheck_758_;
goto v_resetjp_752_;
}
v_resetjp_752_:
{
lean_object* v___x_756_; 
if (v_isShared_754_ == 0)
{
v___x_756_ = v___x_753_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v_a_751_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
}
}
}
}
else
{
lean_object* v_a_760_; lean_object* v___x_762_; uint8_t v_isShared_763_; uint8_t v_isSharedCheck_767_; 
lean_dec(v_a_670_);
lean_dec(v_a_665_);
v_a_760_ = lean_ctor_get(v___x_719_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_719_);
if (v_isSharedCheck_767_ == 0)
{
v___x_762_ = v___x_719_;
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_a_760_);
lean_dec(v___x_719_);
v___x_762_ = lean_box(0);
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
v_resetjp_761_:
{
lean_object* v___x_765_; 
if (v_isShared_763_ == 0)
{
v___x_765_ = v___x_762_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v_a_760_);
v___x_765_ = v_reuseFailAlloc_766_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
return v___x_765_;
}
}
}
}
}
}
else
{
lean_object* v_a_770_; lean_object* v___x_772_; uint8_t v_isShared_773_; uint8_t v_isSharedCheck_777_; 
lean_dec_ref(v___x_667_);
lean_dec(v_a_665_);
lean_dec_ref(v___y_659_);
lean_dec_ref(v_e_658_);
lean_dec_ref(v___x_657_);
lean_dec_ref(v_00_u03b1_656_);
lean_dec(v_u_655_);
lean_dec(v___x_654_);
lean_dec(v___x_653_);
v_a_770_ = lean_ctor_get(v___x_669_, 0);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_669_);
if (v_isSharedCheck_777_ == 0)
{
v___x_772_ = v___x_669_;
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
else
{
lean_inc(v_a_770_);
lean_dec(v___x_669_);
v___x_772_ = lean_box(0);
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
v_resetjp_771_:
{
lean_object* v___x_775_; 
if (v_isShared_773_ == 0)
{
v___x_775_ = v___x_772_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_a_770_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
}
}
else
{
lean_object* v_a_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_785_; 
lean_dec_ref(v___y_659_);
lean_dec_ref(v_e_658_);
lean_dec_ref(v___x_657_);
lean_dec_ref(v_00_u03b1_656_);
lean_dec(v_u_655_);
lean_dec(v___x_654_);
lean_dec(v___x_653_);
lean_dec(v___x_652_);
v_a_778_ = lean_ctor_get(v___x_664_, 0);
v_isSharedCheck_785_ = !lean_is_exclusive(v___x_664_);
if (v_isSharedCheck_785_ == 0)
{
v___x_780_ = v___x_664_;
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_a_778_);
lean_dec(v___x_664_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v___x_783_; 
if (v_isShared_781_ == 0)
{
v___x_783_ = v___x_780_;
goto v_reusejp_782_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v_a_778_);
v___x_783_ = v_reuseFailAlloc_784_;
goto v_reusejp_782_;
}
v_reusejp_782_:
{
return v___x_783_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___boxed(lean_object* v___x_786_, lean_object* v___x_787_, lean_object* v___x_788_, lean_object* v___x_789_, lean_object* v___x_790_, lean_object* v_u_791_, lean_object* v_00_u03b1_792_, lean_object* v___x_793_, lean_object* v_e_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_){
_start:
{
uint8_t v___x_24963__boxed_800_; lean_object* v_res_801_; 
v___x_24963__boxed_800_ = lean_unbox(v___x_787_);
v_res_801_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1(v___x_786_, v___x_24963__boxed_800_, v___x_788_, v___x_789_, v___x_790_, v_u_791_, v_00_u03b1_792_, v___x_793_, v_e_794_, v___y_795_, v___y_796_, v___y_797_, v___y_798_);
lean_dec(v___y_798_);
lean_dec_ref(v___y_797_);
lean_dec(v___y_796_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2(lean_object* v___x_831_, uint8_t v___x_832_, lean_object* v___x_833_, lean_object* v___x_834_, lean_object* v_00_u03b1_835_, lean_object* v___x_836_, lean_object* v___x_837_, lean_object* v_e_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_){
_start:
{
lean_object* v___x_844_; 
v___x_844_ = l_Lean_Meta_mkFreshExprMVar(v___x_831_, v___x_832_, v___x_833_, v___y_839_, v___y_840_, v___y_841_, v___y_842_);
if (lean_obj_tag(v___x_844_) == 0)
{
lean_object* v_a_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v_keyedConfig_855_; uint8_t v_trackZetaDelta_856_; lean_object* v_zetaDeltaSet_857_; lean_object* v_lctx_858_; lean_object* v_localInstances_859_; lean_object* v_defEqCtx_x3f_860_; lean_object* v_synthPendingDepth_861_; lean_object* v_customCanUnfoldPredicate_x3f_862_; uint8_t v_univApprox_863_; uint8_t v_inTypeClassResolution_864_; uint8_t v_cacheInferType_865_; lean_object* v___x_867_; uint8_t v_isShared_868_; uint8_t v_isSharedCheck_932_; 
v_a_845_ = lean_ctor_get(v___x_844_, 0);
lean_inc(v_a_845_);
lean_dec_ref_known(v___x_844_, 1);
v___x_846_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0));
lean_inc_n(v___x_834_, 3);
v___x_847_ = l_Lean_Expr_const___override(v___x_846_, v___x_834_);
lean_inc_ref_n(v_00_u03b1_835_, 3);
v___x_848_ = l_Lean_Expr_app___override(v___x_847_, v_00_u03b1_835_);
v___x_849_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3));
v___x_850_ = l_Lean_Expr_const___override(v___x_849_, v___x_834_);
v___x_851_ = l_Lean_Expr_app___override(v___x_850_, v_00_u03b1_835_);
v___x_852_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6));
v___x_853_ = l_Lean_Expr_const___override(v___x_852_, v___x_834_);
v___x_854_ = l_Lean_Expr_app___override(v___x_853_, v_00_u03b1_835_);
v_keyedConfig_855_ = lean_ctor_get(v___y_839_, 0);
v_trackZetaDelta_856_ = lean_ctor_get_uint8(v___y_839_, sizeof(void*)*7);
v_zetaDeltaSet_857_ = lean_ctor_get(v___y_839_, 1);
v_lctx_858_ = lean_ctor_get(v___y_839_, 2);
v_localInstances_859_ = lean_ctor_get(v___y_839_, 3);
v_defEqCtx_x3f_860_ = lean_ctor_get(v___y_839_, 4);
v_synthPendingDepth_861_ = lean_ctor_get(v___y_839_, 5);
v_customCanUnfoldPredicate_x3f_862_ = lean_ctor_get(v___y_839_, 6);
v_univApprox_863_ = lean_ctor_get_uint8(v___y_839_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_864_ = lean_ctor_get_uint8(v___y_839_, sizeof(void*)*7 + 2);
v_cacheInferType_865_ = lean_ctor_get_uint8(v___y_839_, sizeof(void*)*7 + 3);
v_isSharedCheck_932_ = !lean_is_exclusive(v___y_839_);
if (v_isSharedCheck_932_ == 0)
{
v___x_867_ = v___y_839_;
v_isShared_868_ = v_isSharedCheck_932_;
goto v_resetjp_866_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_862_);
lean_inc(v_synthPendingDepth_861_);
lean_inc(v_defEqCtx_x3f_860_);
lean_inc(v_localInstances_859_);
lean_inc(v_lctx_858_);
lean_inc(v_zetaDeltaSet_857_);
lean_inc(v_keyedConfig_855_);
lean_dec(v___y_839_);
v___x_867_ = lean_box(0);
v_isShared_868_ = v_isSharedCheck_932_;
goto v_resetjp_866_;
}
v_resetjp_866_:
{
lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; uint8_t v___x_890_; lean_object* v___x_891_; lean_object* v___x_893_; 
v___x_869_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9));
lean_inc_n(v___x_834_, 3);
v___x_870_ = l_Lean_Expr_const___override(v___x_869_, v___x_834_);
lean_inc_ref_n(v_00_u03b1_835_, 3);
v___x_871_ = l_Lean_Expr_app___override(v___x_870_, v_00_u03b1_835_);
v___x_872_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12));
v___x_873_ = l_Lean_Expr_const___override(v___x_872_, v___x_834_);
v___x_874_ = l_Lean_Expr_app___override(v___x_873_, v_00_u03b1_835_);
v___x_875_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15));
v___x_876_ = l_Lean_Expr_const___override(v___x_875_, v___x_834_);
v___x_877_ = l_Lean_Expr_app___override(v___x_876_, v_00_u03b1_835_);
v___x_878_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__16));
v___x_879_ = l_Lean_Name_mkStr2(v___x_836_, v___x_878_);
v___x_880_ = l_Lean_Expr_const___override(v___x_879_, v___x_834_);
v___x_881_ = l_Lean_Expr_app___override(v___x_880_, v_00_u03b1_835_);
v___x_882_ = l_Lean_Expr_app___override(v___x_881_, v___x_837_);
v___x_883_ = l_Lean_Expr_app___override(v___x_877_, v___x_882_);
v___x_884_ = l_Lean_Expr_app___override(v___x_874_, v___x_883_);
v___x_885_ = l_Lean_Expr_app___override(v___x_871_, v___x_884_);
v___x_886_ = l_Lean_Expr_app___override(v___x_854_, v___x_885_);
v___x_887_ = l_Lean_Expr_app___override(v___x_851_, v___x_886_);
v___x_888_ = l_Lean_Expr_app___override(v___x_848_, v___x_887_);
lean_inc(v_a_845_);
v___x_889_ = l_Lean_Expr_app___override(v___x_888_, v_a_845_);
v___x_890_ = 2;
v___x_891_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_890_, v_keyedConfig_855_);
if (v_isShared_868_ == 0)
{
lean_ctor_set(v___x_867_, 0, v___x_891_);
v___x_893_ = v___x_867_;
goto v_reusejp_892_;
}
else
{
lean_object* v_reuseFailAlloc_931_; 
v_reuseFailAlloc_931_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_931_, 0, v___x_891_);
lean_ctor_set(v_reuseFailAlloc_931_, 1, v_zetaDeltaSet_857_);
lean_ctor_set(v_reuseFailAlloc_931_, 2, v_lctx_858_);
lean_ctor_set(v_reuseFailAlloc_931_, 3, v_localInstances_859_);
lean_ctor_set(v_reuseFailAlloc_931_, 4, v_defEqCtx_x3f_860_);
lean_ctor_set(v_reuseFailAlloc_931_, 5, v_synthPendingDepth_861_);
lean_ctor_set(v_reuseFailAlloc_931_, 6, v_customCanUnfoldPredicate_x3f_862_);
lean_ctor_set_uint8(v_reuseFailAlloc_931_, sizeof(void*)*7, v_trackZetaDelta_856_);
lean_ctor_set_uint8(v_reuseFailAlloc_931_, sizeof(void*)*7 + 1, v_univApprox_863_);
lean_ctor_set_uint8(v_reuseFailAlloc_931_, sizeof(void*)*7 + 2, v_inTypeClassResolution_864_);
lean_ctor_set_uint8(v_reuseFailAlloc_931_, sizeof(void*)*7 + 3, v_cacheInferType_865_);
v___x_893_ = v_reuseFailAlloc_931_;
goto v_reusejp_892_;
}
v_reusejp_892_:
{
lean_object* v___x_894_; 
v___x_894_ = l_Lean_Meta_isExprDefEq(v___x_889_, v_e_838_, v___x_893_, v___y_840_, v___y_841_, v___y_842_);
lean_dec_ref(v___x_893_);
if (lean_obj_tag(v___x_894_) == 0)
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_922_; 
v_a_895_ = lean_ctor_get(v___x_894_, 0);
v_isSharedCheck_922_ = !lean_is_exclusive(v___x_894_);
if (v_isSharedCheck_922_ == 0)
{
v___x_897_ = v___x_894_;
v_isShared_898_ = v_isSharedCheck_922_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___x_894_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_922_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
uint8_t v___x_899_; 
v___x_899_ = lean_unbox(v_a_895_);
if (v___x_899_ == 0)
{
lean_object* v___x_900_; lean_object* v___x_902_; 
v___x_900_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_900_, 0, v_a_845_);
lean_ctor_set(v___x_900_, 1, v_a_895_);
if (v_isShared_898_ == 0)
{
lean_ctor_set(v___x_897_, 0, v___x_900_);
v___x_902_ = v___x_897_;
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
else
{
lean_object* v___x_904_; 
lean_del_object(v___x_897_);
v___x_904_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_845_, v___y_840_);
if (lean_obj_tag(v___x_904_) == 0)
{
lean_object* v_a_905_; lean_object* v___x_907_; uint8_t v_isShared_908_; uint8_t v_isSharedCheck_913_; 
v_a_905_ = lean_ctor_get(v___x_904_, 0);
v_isSharedCheck_913_ = !lean_is_exclusive(v___x_904_);
if (v_isSharedCheck_913_ == 0)
{
v___x_907_ = v___x_904_;
v_isShared_908_ = v_isSharedCheck_913_;
goto v_resetjp_906_;
}
else
{
lean_inc(v_a_905_);
lean_dec(v___x_904_);
v___x_907_ = lean_box(0);
v_isShared_908_ = v_isSharedCheck_913_;
goto v_resetjp_906_;
}
v_resetjp_906_:
{
lean_object* v___x_909_; lean_object* v___x_911_; 
v___x_909_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_909_, 0, v_a_905_);
lean_ctor_set(v___x_909_, 1, v_a_895_);
if (v_isShared_908_ == 0)
{
lean_ctor_set(v___x_907_, 0, v___x_909_);
v___x_911_ = v___x_907_;
goto v_reusejp_910_;
}
else
{
lean_object* v_reuseFailAlloc_912_; 
v_reuseFailAlloc_912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_912_, 0, v___x_909_);
v___x_911_ = v_reuseFailAlloc_912_;
goto v_reusejp_910_;
}
v_reusejp_910_:
{
return v___x_911_;
}
}
}
else
{
lean_object* v_a_914_; lean_object* v___x_916_; uint8_t v_isShared_917_; uint8_t v_isSharedCheck_921_; 
lean_dec(v_a_895_);
v_a_914_ = lean_ctor_get(v___x_904_, 0);
v_isSharedCheck_921_ = !lean_is_exclusive(v___x_904_);
if (v_isSharedCheck_921_ == 0)
{
v___x_916_ = v___x_904_;
v_isShared_917_ = v_isSharedCheck_921_;
goto v_resetjp_915_;
}
else
{
lean_inc(v_a_914_);
lean_dec(v___x_904_);
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
else
{
lean_object* v_a_923_; lean_object* v___x_925_; uint8_t v_isShared_926_; uint8_t v_isSharedCheck_930_; 
lean_dec(v_a_845_);
v_a_923_ = lean_ctor_get(v___x_894_, 0);
v_isSharedCheck_930_ = !lean_is_exclusive(v___x_894_);
if (v_isSharedCheck_930_ == 0)
{
v___x_925_ = v___x_894_;
v_isShared_926_ = v_isSharedCheck_930_;
goto v_resetjp_924_;
}
else
{
lean_inc(v_a_923_);
lean_dec(v___x_894_);
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
}
else
{
lean_object* v_a_933_; lean_object* v___x_935_; uint8_t v_isShared_936_; uint8_t v_isSharedCheck_940_; 
lean_dec_ref(v___y_839_);
lean_dec_ref(v_e_838_);
lean_dec_ref(v___x_837_);
lean_dec_ref(v___x_836_);
lean_dec_ref(v_00_u03b1_835_);
lean_dec(v___x_834_);
v_a_933_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_940_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_940_ == 0)
{
v___x_935_ = v___x_844_;
v_isShared_936_ = v_isSharedCheck_940_;
goto v_resetjp_934_;
}
else
{
lean_inc(v_a_933_);
lean_dec(v___x_844_);
v___x_935_ = lean_box(0);
v_isShared_936_ = v_isSharedCheck_940_;
goto v_resetjp_934_;
}
v_resetjp_934_:
{
lean_object* v___x_938_; 
if (v_isShared_936_ == 0)
{
v___x_938_ = v___x_935_;
goto v_reusejp_937_;
}
else
{
lean_object* v_reuseFailAlloc_939_; 
v_reuseFailAlloc_939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_939_, 0, v_a_933_);
v___x_938_ = v_reuseFailAlloc_939_;
goto v_reusejp_937_;
}
v_reusejp_937_:
{
return v___x_938_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___boxed(lean_object* v___x_941_, lean_object* v___x_942_, lean_object* v___x_943_, lean_object* v___x_944_, lean_object* v_00_u03b1_945_, lean_object* v___x_946_, lean_object* v___x_947_, lean_object* v_e_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
uint8_t v___x_25302__boxed_954_; lean_object* v_res_955_; 
v___x_25302__boxed_954_ = lean_unbox(v___x_942_);
v_res_955_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2(v___x_941_, v___x_25302__boxed_954_, v___x_943_, v___x_944_, v_00_u03b1_945_, v___x_946_, v___x_947_, v_e_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
lean_dec(v___y_952_);
lean_dec_ref(v___y_951_);
lean_dec(v___y_950_);
return v_res_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3(lean_object* v___x_968_, uint8_t v___x_969_, lean_object* v___x_970_, lean_object* v___x_971_, lean_object* v_00_u03b1_972_, lean_object* v___x_973_, lean_object* v___x_974_, lean_object* v___x_975_, lean_object* v_e_976_, uint8_t v___x_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_){
_start:
{
lean_object* v___x_983_; 
lean_inc(v___x_970_);
lean_inc(v___x_968_);
v___x_983_ = l_Lean_Meta_mkFreshExprMVar(v___x_968_, v___x_969_, v___x_970_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
if (lean_obj_tag(v___x_983_) == 0)
{
lean_object* v_a_984_; lean_object* v___x_985_; 
v_a_984_ = lean_ctor_get(v___x_983_, 0);
lean_inc(v_a_984_);
lean_dec_ref_known(v___x_983_, 1);
v___x_985_ = l_Lean_Meta_mkFreshExprMVar(v___x_968_, v___x_969_, v___x_970_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
if (lean_obj_tag(v___x_985_) == 0)
{
lean_object* v_a_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v_keyedConfig_991_; uint8_t v_trackZetaDelta_992_; lean_object* v_zetaDeltaSet_993_; lean_object* v_lctx_994_; lean_object* v_localInstances_995_; lean_object* v_defEqCtx_x3f_996_; lean_object* v_synthPendingDepth_997_; lean_object* v_customCanUnfoldPredicate_x3f_998_; uint8_t v_univApprox_999_; uint8_t v_inTypeClassResolution_1000_; uint8_t v_cacheInferType_1001_; lean_object* v___x_1003_; uint8_t v_isShared_1004_; uint8_t v_isSharedCheck_1077_; 
v_a_986_ = lean_ctor_get(v___x_985_, 0);
lean_inc(v_a_986_);
lean_dec_ref_known(v___x_985_, 1);
v___x_987_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__0));
v___x_988_ = l_Lean_Expr_const___override(v___x_987_, v___x_971_);
lean_inc_ref_n(v_00_u03b1_972_, 2);
v___x_989_ = l_Lean_Expr_app___override(v___x_988_, v_00_u03b1_972_);
v___x_990_ = l_Lean_Expr_app___override(v___x_989_, v_00_u03b1_972_);
v_keyedConfig_991_ = lean_ctor_get(v___y_978_, 0);
v_trackZetaDelta_992_ = lean_ctor_get_uint8(v___y_978_, sizeof(void*)*7);
v_zetaDeltaSet_993_ = lean_ctor_get(v___y_978_, 1);
v_lctx_994_ = lean_ctor_get(v___y_978_, 2);
v_localInstances_995_ = lean_ctor_get(v___y_978_, 3);
v_defEqCtx_x3f_996_ = lean_ctor_get(v___y_978_, 4);
v_synthPendingDepth_997_ = lean_ctor_get(v___y_978_, 5);
v_customCanUnfoldPredicate_x3f_998_ = lean_ctor_get(v___y_978_, 6);
v_univApprox_999_ = lean_ctor_get_uint8(v___y_978_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1000_ = lean_ctor_get_uint8(v___y_978_, sizeof(void*)*7 + 2);
v_cacheInferType_1001_ = lean_ctor_get_uint8(v___y_978_, sizeof(void*)*7 + 3);
v_isSharedCheck_1077_ = !lean_is_exclusive(v___y_978_);
if (v_isSharedCheck_1077_ == 0)
{
v___x_1003_ = v___y_978_;
v_isShared_1004_ = v_isSharedCheck_1077_;
goto v_resetjp_1002_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_998_);
lean_inc(v_synthPendingDepth_997_);
lean_inc(v_defEqCtx_x3f_996_);
lean_inc(v_localInstances_995_);
lean_inc(v_lctx_994_);
lean_inc(v_zetaDeltaSet_993_);
lean_inc(v_keyedConfig_991_);
lean_dec(v___y_978_);
v___x_1003_ = lean_box(0);
v_isShared_1004_ = v_isSharedCheck_1077_;
goto v_resetjp_1002_;
}
v_resetjp_1002_:
{
lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; uint8_t v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1025_; 
lean_inc_ref_n(v_00_u03b1_972_, 3);
v___x_1005_ = l_Lean_Expr_app___override(v___x_990_, v_00_u03b1_972_);
v___x_1006_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__2));
lean_inc_n(v___x_973_, 2);
v___x_1007_ = l_Lean_Expr_const___override(v___x_1006_, v___x_973_);
v___x_1008_ = l_Lean_Expr_app___override(v___x_1007_, v_00_u03b1_972_);
v___x_1009_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__5));
v___x_1010_ = l_Lean_Expr_const___override(v___x_1009_, v___x_973_);
v___x_1011_ = l_Lean_Expr_app___override(v___x_1010_, v_00_u03b1_972_);
v___x_1012_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__6));
v___x_1013_ = l_Lean_Name_mkStr2(v___x_974_, v___x_1012_);
v___x_1014_ = l_Lean_Expr_const___override(v___x_1013_, v___x_973_);
v___x_1015_ = l_Lean_Expr_app___override(v___x_1014_, v_00_u03b1_972_);
v___x_1016_ = l_Lean_Expr_app___override(v___x_1015_, v___x_975_);
v___x_1017_ = l_Lean_Expr_app___override(v___x_1011_, v___x_1016_);
v___x_1018_ = l_Lean_Expr_app___override(v___x_1008_, v___x_1017_);
v___x_1019_ = l_Lean_Expr_app___override(v___x_1005_, v___x_1018_);
lean_inc(v_a_984_);
v___x_1020_ = l_Lean_Expr_app___override(v___x_1019_, v_a_984_);
lean_inc(v_a_986_);
v___x_1021_ = l_Lean_Expr_app___override(v___x_1020_, v_a_986_);
v___x_1022_ = 2;
v___x_1023_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1022_, v_keyedConfig_991_);
if (v_isShared_1004_ == 0)
{
lean_ctor_set(v___x_1003_, 0, v___x_1023_);
v___x_1025_ = v___x_1003_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v___x_1023_);
lean_ctor_set(v_reuseFailAlloc_1076_, 1, v_zetaDeltaSet_993_);
lean_ctor_set(v_reuseFailAlloc_1076_, 2, v_lctx_994_);
lean_ctor_set(v_reuseFailAlloc_1076_, 3, v_localInstances_995_);
lean_ctor_set(v_reuseFailAlloc_1076_, 4, v_defEqCtx_x3f_996_);
lean_ctor_set(v_reuseFailAlloc_1076_, 5, v_synthPendingDepth_997_);
lean_ctor_set(v_reuseFailAlloc_1076_, 6, v_customCanUnfoldPredicate_x3f_998_);
lean_ctor_set_uint8(v_reuseFailAlloc_1076_, sizeof(void*)*7, v_trackZetaDelta_992_);
lean_ctor_set_uint8(v_reuseFailAlloc_1076_, sizeof(void*)*7 + 1, v_univApprox_999_);
lean_ctor_set_uint8(v_reuseFailAlloc_1076_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1000_);
lean_ctor_set_uint8(v_reuseFailAlloc_1076_, sizeof(void*)*7 + 3, v_cacheInferType_1001_);
v___x_1025_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
lean_object* v___x_1026_; 
v___x_1026_ = l_Lean_Meta_isExprDefEq(v___x_1021_, v_e_976_, v___x_1025_, v___y_979_, v___y_980_, v___y_981_);
lean_dec_ref(v___x_1025_);
if (lean_obj_tag(v___x_1026_) == 0)
{
lean_object* v_a_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1067_; 
v_a_1027_ = lean_ctor_get(v___x_1026_, 0);
v_isSharedCheck_1067_ = !lean_is_exclusive(v___x_1026_);
if (v_isSharedCheck_1067_ == 0)
{
v___x_1029_ = v___x_1026_;
v_isShared_1030_ = v_isSharedCheck_1067_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_a_1027_);
lean_dec(v___x_1026_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1067_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
uint8_t v___x_1031_; 
v___x_1031_ = lean_unbox(v_a_1027_);
if (v___x_1031_ == 0)
{
lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1036_; 
lean_dec(v_a_1027_);
v___x_1032_ = lean_box(v___x_977_);
v___x_1033_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1033_, 0, v_a_986_);
lean_ctor_set(v___x_1033_, 1, v___x_1032_);
v___x_1034_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1034_, 0, v_a_984_);
lean_ctor_set(v___x_1034_, 1, v___x_1033_);
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 0, v___x_1034_);
v___x_1036_ = v___x_1029_;
goto v_reusejp_1035_;
}
else
{
lean_object* v_reuseFailAlloc_1037_; 
v_reuseFailAlloc_1037_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1037_, 0, v___x_1034_);
v___x_1036_ = v_reuseFailAlloc_1037_;
goto v_reusejp_1035_;
}
v_reusejp_1035_:
{
return v___x_1036_;
}
}
else
{
lean_object* v___x_1038_; 
lean_del_object(v___x_1029_);
v___x_1038_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_984_, v___y_979_);
if (lean_obj_tag(v___x_1038_) == 0)
{
lean_object* v_a_1039_; lean_object* v___x_1040_; 
v_a_1039_ = lean_ctor_get(v___x_1038_, 0);
lean_inc(v_a_1039_);
lean_dec_ref_known(v___x_1038_, 1);
v___x_1040_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_986_, v___y_979_);
if (lean_obj_tag(v___x_1040_) == 0)
{
lean_object* v_a_1041_; lean_object* v___x_1043_; uint8_t v_isShared_1044_; uint8_t v_isSharedCheck_1050_; 
v_a_1041_ = lean_ctor_get(v___x_1040_, 0);
v_isSharedCheck_1050_ = !lean_is_exclusive(v___x_1040_);
if (v_isSharedCheck_1050_ == 0)
{
v___x_1043_ = v___x_1040_;
v_isShared_1044_ = v_isSharedCheck_1050_;
goto v_resetjp_1042_;
}
else
{
lean_inc(v_a_1041_);
lean_dec(v___x_1040_);
v___x_1043_ = lean_box(0);
v_isShared_1044_ = v_isSharedCheck_1050_;
goto v_resetjp_1042_;
}
v_resetjp_1042_:
{
lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1048_; 
v___x_1045_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1045_, 0, v_a_1041_);
lean_ctor_set(v___x_1045_, 1, v_a_1027_);
v___x_1046_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1046_, 0, v_a_1039_);
lean_ctor_set(v___x_1046_, 1, v___x_1045_);
if (v_isShared_1044_ == 0)
{
lean_ctor_set(v___x_1043_, 0, v___x_1046_);
v___x_1048_ = v___x_1043_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v___x_1046_);
v___x_1048_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
return v___x_1048_;
}
}
}
else
{
lean_object* v_a_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1058_; 
lean_dec(v_a_1039_);
lean_dec(v_a_1027_);
v_a_1051_ = lean_ctor_get(v___x_1040_, 0);
v_isSharedCheck_1058_ = !lean_is_exclusive(v___x_1040_);
if (v_isSharedCheck_1058_ == 0)
{
v___x_1053_ = v___x_1040_;
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_a_1051_);
lean_dec(v___x_1040_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1056_; 
if (v_isShared_1054_ == 0)
{
v___x_1056_ = v___x_1053_;
goto v_reusejp_1055_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v_a_1051_);
v___x_1056_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1055_;
}
v_reusejp_1055_:
{
return v___x_1056_;
}
}
}
}
else
{
lean_object* v_a_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1066_; 
lean_dec(v_a_1027_);
lean_dec(v_a_986_);
v_a_1059_ = lean_ctor_get(v___x_1038_, 0);
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_1038_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1061_ = v___x_1038_;
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_a_1059_);
lean_dec(v___x_1038_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v___x_1064_; 
if (v_isShared_1062_ == 0)
{
v___x_1064_ = v___x_1061_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v_a_1059_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
}
}
}
}
else
{
lean_object* v_a_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1075_; 
lean_dec(v_a_986_);
lean_dec(v_a_984_);
v_a_1068_ = lean_ctor_get(v___x_1026_, 0);
v_isSharedCheck_1075_ = !lean_is_exclusive(v___x_1026_);
if (v_isSharedCheck_1075_ == 0)
{
v___x_1070_ = v___x_1026_;
v_isShared_1071_ = v_isSharedCheck_1075_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_a_1068_);
lean_dec(v___x_1026_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1075_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v___x_1073_; 
if (v_isShared_1071_ == 0)
{
v___x_1073_ = v___x_1070_;
goto v_reusejp_1072_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v_a_1068_);
v___x_1073_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1072_;
}
v_reusejp_1072_:
{
return v___x_1073_;
}
}
}
}
}
}
else
{
lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1085_; 
lean_dec(v_a_984_);
lean_dec_ref(v___y_978_);
lean_dec_ref(v_e_976_);
lean_dec_ref(v___x_975_);
lean_dec_ref(v___x_974_);
lean_dec(v___x_973_);
lean_dec_ref(v_00_u03b1_972_);
lean_dec(v___x_971_);
v_a_1078_ = lean_ctor_get(v___x_985_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_985_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1080_ = v___x_985_;
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_985_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1083_; 
if (v_isShared_1081_ == 0)
{
v___x_1083_ = v___x_1080_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_a_1078_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
return v___x_1083_;
}
}
}
}
else
{
lean_object* v_a_1086_; lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1093_; 
lean_dec_ref(v___y_978_);
lean_dec_ref(v_e_976_);
lean_dec_ref(v___x_975_);
lean_dec_ref(v___x_974_);
lean_dec(v___x_973_);
lean_dec_ref(v_00_u03b1_972_);
lean_dec(v___x_971_);
lean_dec(v___x_970_);
lean_dec(v___x_968_);
v_a_1086_ = lean_ctor_get(v___x_983_, 0);
v_isSharedCheck_1093_ = !lean_is_exclusive(v___x_983_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1088_ = v___x_983_;
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
else
{
lean_inc(v_a_1086_);
lean_dec(v___x_983_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v___x_1091_; 
if (v_isShared_1089_ == 0)
{
v___x_1091_ = v___x_1088_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1092_; 
v_reuseFailAlloc_1092_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1092_, 0, v_a_1086_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___boxed(lean_object* v___x_1094_, lean_object* v___x_1095_, lean_object* v___x_1096_, lean_object* v___x_1097_, lean_object* v_00_u03b1_1098_, lean_object* v___x_1099_, lean_object* v___x_1100_, lean_object* v___x_1101_, lean_object* v_e_1102_, lean_object* v___x_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_){
_start:
{
uint8_t v___x_25562__boxed_1109_; uint8_t v___x_25568__boxed_1110_; lean_object* v_res_1111_; 
v___x_25562__boxed_1109_ = lean_unbox(v___x_1095_);
v___x_25568__boxed_1110_ = lean_unbox(v___x_1103_);
v_res_1111_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3(v___x_1094_, v___x_25562__boxed_1109_, v___x_1096_, v___x_1097_, v_00_u03b1_1098_, v___x_1099_, v___x_1100_, v___x_1101_, v_e_1102_, v___x_25568__boxed_1110_, v___y_1104_, v___y_1105_, v___y_1106_, v___y_1107_);
lean_dec(v___y_1107_);
lean_dec_ref(v___y_1106_);
lean_dec(v___y_1105_);
return v_res_1111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__4(lean_object* v___x_1112_, uint8_t v___x_1113_, lean_object* v___x_1114_, lean_object* v___x_1115_, lean_object* v_e_1116_, uint8_t v___x_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_){
_start:
{
lean_object* v___x_1123_; 
lean_inc(v___x_1114_);
lean_inc(v___x_1112_);
v___x_1123_ = l_Lean_Meta_mkFreshExprMVar(v___x_1112_, v___x_1113_, v___x_1114_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_);
if (lean_obj_tag(v___x_1123_) == 0)
{
lean_object* v_a_1124_; lean_object* v___x_1125_; 
v_a_1124_ = lean_ctor_get(v___x_1123_, 0);
lean_inc(v_a_1124_);
lean_dec_ref_known(v___x_1123_, 1);
v___x_1125_ = l_Lean_Meta_mkFreshExprMVar(v___x_1112_, v___x_1113_, v___x_1114_, v___y_1118_, v___y_1119_, v___y_1120_, v___y_1121_);
if (lean_obj_tag(v___x_1125_) == 0)
{
lean_object* v_a_1126_; lean_object* v_keyedConfig_1127_; uint8_t v_trackZetaDelta_1128_; lean_object* v_zetaDeltaSet_1129_; lean_object* v_lctx_1130_; lean_object* v_localInstances_1131_; lean_object* v_defEqCtx_x3f_1132_; lean_object* v_synthPendingDepth_1133_; lean_object* v_customCanUnfoldPredicate_x3f_1134_; uint8_t v_univApprox_1135_; uint8_t v_inTypeClassResolution_1136_; uint8_t v_cacheInferType_1137_; lean_object* v___x_1139_; uint8_t v_isShared_1140_; uint8_t v_isSharedCheck_1198_; 
v_a_1126_ = lean_ctor_get(v___x_1125_, 0);
lean_inc(v_a_1126_);
lean_dec_ref_known(v___x_1125_, 1);
v_keyedConfig_1127_ = lean_ctor_get(v___y_1118_, 0);
v_trackZetaDelta_1128_ = lean_ctor_get_uint8(v___y_1118_, sizeof(void*)*7);
v_zetaDeltaSet_1129_ = lean_ctor_get(v___y_1118_, 1);
v_lctx_1130_ = lean_ctor_get(v___y_1118_, 2);
v_localInstances_1131_ = lean_ctor_get(v___y_1118_, 3);
v_defEqCtx_x3f_1132_ = lean_ctor_get(v___y_1118_, 4);
v_synthPendingDepth_1133_ = lean_ctor_get(v___y_1118_, 5);
v_customCanUnfoldPredicate_x3f_1134_ = lean_ctor_get(v___y_1118_, 6);
v_univApprox_1135_ = lean_ctor_get_uint8(v___y_1118_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1136_ = lean_ctor_get_uint8(v___y_1118_, sizeof(void*)*7 + 2);
v_cacheInferType_1137_ = lean_ctor_get_uint8(v___y_1118_, sizeof(void*)*7 + 3);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___y_1118_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1139_ = v___y_1118_;
v_isShared_1140_ = v_isSharedCheck_1198_;
goto v_resetjp_1138_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1134_);
lean_inc(v_synthPendingDepth_1133_);
lean_inc(v_defEqCtx_x3f_1132_);
lean_inc(v_localInstances_1131_);
lean_inc(v_lctx_1130_);
lean_inc(v_zetaDeltaSet_1129_);
lean_inc(v_keyedConfig_1127_);
lean_dec(v___y_1118_);
v___x_1139_ = lean_box(0);
v_isShared_1140_ = v_isSharedCheck_1198_;
goto v_resetjp_1138_;
}
v_resetjp_1138_:
{
lean_object* v___x_1141_; lean_object* v___x_1142_; uint8_t v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1146_; 
lean_inc(v_a_1124_);
v___x_1141_ = l_Lean_Expr_app___override(v___x_1115_, v_a_1124_);
lean_inc(v_a_1126_);
v___x_1142_ = l_Lean_Expr_app___override(v___x_1141_, v_a_1126_);
v___x_1143_ = 2;
v___x_1144_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1143_, v_keyedConfig_1127_);
if (v_isShared_1140_ == 0)
{
lean_ctor_set(v___x_1139_, 0, v___x_1144_);
v___x_1146_ = v___x_1139_;
goto v_reusejp_1145_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v___x_1144_);
lean_ctor_set(v_reuseFailAlloc_1197_, 1, v_zetaDeltaSet_1129_);
lean_ctor_set(v_reuseFailAlloc_1197_, 2, v_lctx_1130_);
lean_ctor_set(v_reuseFailAlloc_1197_, 3, v_localInstances_1131_);
lean_ctor_set(v_reuseFailAlloc_1197_, 4, v_defEqCtx_x3f_1132_);
lean_ctor_set(v_reuseFailAlloc_1197_, 5, v_synthPendingDepth_1133_);
lean_ctor_set(v_reuseFailAlloc_1197_, 6, v_customCanUnfoldPredicate_x3f_1134_);
lean_ctor_set_uint8(v_reuseFailAlloc_1197_, sizeof(void*)*7, v_trackZetaDelta_1128_);
lean_ctor_set_uint8(v_reuseFailAlloc_1197_, sizeof(void*)*7 + 1, v_univApprox_1135_);
lean_ctor_set_uint8(v_reuseFailAlloc_1197_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1136_);
lean_ctor_set_uint8(v_reuseFailAlloc_1197_, sizeof(void*)*7 + 3, v_cacheInferType_1137_);
v___x_1146_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1145_;
}
v_reusejp_1145_:
{
lean_object* v___x_1147_; 
v___x_1147_ = l_Lean_Meta_isExprDefEq(v___x_1142_, v_e_1116_, v___x_1146_, v___y_1119_, v___y_1120_, v___y_1121_);
lean_dec_ref(v___x_1146_);
if (lean_obj_tag(v___x_1147_) == 0)
{
lean_object* v_a_1148_; lean_object* v___x_1150_; uint8_t v_isShared_1151_; uint8_t v_isSharedCheck_1188_; 
v_a_1148_ = lean_ctor_get(v___x_1147_, 0);
v_isSharedCheck_1188_ = !lean_is_exclusive(v___x_1147_);
if (v_isSharedCheck_1188_ == 0)
{
v___x_1150_ = v___x_1147_;
v_isShared_1151_ = v_isSharedCheck_1188_;
goto v_resetjp_1149_;
}
else
{
lean_inc(v_a_1148_);
lean_dec(v___x_1147_);
v___x_1150_ = lean_box(0);
v_isShared_1151_ = v_isSharedCheck_1188_;
goto v_resetjp_1149_;
}
v_resetjp_1149_:
{
uint8_t v___x_1152_; 
v___x_1152_ = lean_unbox(v_a_1148_);
if (v___x_1152_ == 0)
{
lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1157_; 
lean_dec(v_a_1148_);
v___x_1153_ = lean_box(v___x_1117_);
v___x_1154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1154_, 0, v_a_1126_);
lean_ctor_set(v___x_1154_, 1, v___x_1153_);
v___x_1155_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1155_, 0, v_a_1124_);
lean_ctor_set(v___x_1155_, 1, v___x_1154_);
if (v_isShared_1151_ == 0)
{
lean_ctor_set(v___x_1150_, 0, v___x_1155_);
v___x_1157_ = v___x_1150_;
goto v_reusejp_1156_;
}
else
{
lean_object* v_reuseFailAlloc_1158_; 
v_reuseFailAlloc_1158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1158_, 0, v___x_1155_);
v___x_1157_ = v_reuseFailAlloc_1158_;
goto v_reusejp_1156_;
}
v_reusejp_1156_:
{
return v___x_1157_;
}
}
else
{
lean_object* v___x_1159_; 
lean_del_object(v___x_1150_);
v___x_1159_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_1124_, v___y_1119_);
if (lean_obj_tag(v___x_1159_) == 0)
{
lean_object* v_a_1160_; lean_object* v___x_1161_; 
v_a_1160_ = lean_ctor_get(v___x_1159_, 0);
lean_inc(v_a_1160_);
lean_dec_ref_known(v___x_1159_, 1);
v___x_1161_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_1126_, v___y_1119_);
if (lean_obj_tag(v___x_1161_) == 0)
{
lean_object* v_a_1162_; lean_object* v___x_1164_; uint8_t v_isShared_1165_; uint8_t v_isSharedCheck_1171_; 
v_a_1162_ = lean_ctor_get(v___x_1161_, 0);
v_isSharedCheck_1171_ = !lean_is_exclusive(v___x_1161_);
if (v_isSharedCheck_1171_ == 0)
{
v___x_1164_ = v___x_1161_;
v_isShared_1165_ = v_isSharedCheck_1171_;
goto v_resetjp_1163_;
}
else
{
lean_inc(v_a_1162_);
lean_dec(v___x_1161_);
v___x_1164_ = lean_box(0);
v_isShared_1165_ = v_isSharedCheck_1171_;
goto v_resetjp_1163_;
}
v_resetjp_1163_:
{
lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1169_; 
v___x_1166_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1166_, 0, v_a_1162_);
lean_ctor_set(v___x_1166_, 1, v_a_1148_);
v___x_1167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1167_, 0, v_a_1160_);
lean_ctor_set(v___x_1167_, 1, v___x_1166_);
if (v_isShared_1165_ == 0)
{
lean_ctor_set(v___x_1164_, 0, v___x_1167_);
v___x_1169_ = v___x_1164_;
goto v_reusejp_1168_;
}
else
{
lean_object* v_reuseFailAlloc_1170_; 
v_reuseFailAlloc_1170_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1170_, 0, v___x_1167_);
v___x_1169_ = v_reuseFailAlloc_1170_;
goto v_reusejp_1168_;
}
v_reusejp_1168_:
{
return v___x_1169_;
}
}
}
else
{
lean_object* v_a_1172_; lean_object* v___x_1174_; uint8_t v_isShared_1175_; uint8_t v_isSharedCheck_1179_; 
lean_dec(v_a_1160_);
lean_dec(v_a_1148_);
v_a_1172_ = lean_ctor_get(v___x_1161_, 0);
v_isSharedCheck_1179_ = !lean_is_exclusive(v___x_1161_);
if (v_isSharedCheck_1179_ == 0)
{
v___x_1174_ = v___x_1161_;
v_isShared_1175_ = v_isSharedCheck_1179_;
goto v_resetjp_1173_;
}
else
{
lean_inc(v_a_1172_);
lean_dec(v___x_1161_);
v___x_1174_ = lean_box(0);
v_isShared_1175_ = v_isSharedCheck_1179_;
goto v_resetjp_1173_;
}
v_resetjp_1173_:
{
lean_object* v___x_1177_; 
if (v_isShared_1175_ == 0)
{
v___x_1177_ = v___x_1174_;
goto v_reusejp_1176_;
}
else
{
lean_object* v_reuseFailAlloc_1178_; 
v_reuseFailAlloc_1178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1178_, 0, v_a_1172_);
v___x_1177_ = v_reuseFailAlloc_1178_;
goto v_reusejp_1176_;
}
v_reusejp_1176_:
{
return v___x_1177_;
}
}
}
}
else
{
lean_object* v_a_1180_; lean_object* v___x_1182_; uint8_t v_isShared_1183_; uint8_t v_isSharedCheck_1187_; 
lean_dec(v_a_1148_);
lean_dec(v_a_1126_);
v_a_1180_ = lean_ctor_get(v___x_1159_, 0);
v_isSharedCheck_1187_ = !lean_is_exclusive(v___x_1159_);
if (v_isSharedCheck_1187_ == 0)
{
v___x_1182_ = v___x_1159_;
v_isShared_1183_ = v_isSharedCheck_1187_;
goto v_resetjp_1181_;
}
else
{
lean_inc(v_a_1180_);
lean_dec(v___x_1159_);
v___x_1182_ = lean_box(0);
v_isShared_1183_ = v_isSharedCheck_1187_;
goto v_resetjp_1181_;
}
v_resetjp_1181_:
{
lean_object* v___x_1185_; 
if (v_isShared_1183_ == 0)
{
v___x_1185_ = v___x_1182_;
goto v_reusejp_1184_;
}
else
{
lean_object* v_reuseFailAlloc_1186_; 
v_reuseFailAlloc_1186_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1186_, 0, v_a_1180_);
v___x_1185_ = v_reuseFailAlloc_1186_;
goto v_reusejp_1184_;
}
v_reusejp_1184_:
{
return v___x_1185_;
}
}
}
}
}
}
else
{
lean_object* v_a_1189_; lean_object* v___x_1191_; uint8_t v_isShared_1192_; uint8_t v_isSharedCheck_1196_; 
lean_dec(v_a_1126_);
lean_dec(v_a_1124_);
v_a_1189_ = lean_ctor_get(v___x_1147_, 0);
v_isSharedCheck_1196_ = !lean_is_exclusive(v___x_1147_);
if (v_isSharedCheck_1196_ == 0)
{
v___x_1191_ = v___x_1147_;
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
else
{
lean_inc(v_a_1189_);
lean_dec(v___x_1147_);
v___x_1191_ = lean_box(0);
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
v_resetjp_1190_:
{
lean_object* v___x_1194_; 
if (v_isShared_1192_ == 0)
{
v___x_1194_ = v___x_1191_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_a_1189_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
}
}
}
}
else
{
lean_object* v_a_1199_; lean_object* v___x_1201_; uint8_t v_isShared_1202_; uint8_t v_isSharedCheck_1206_; 
lean_dec(v_a_1124_);
lean_dec_ref(v___y_1118_);
lean_dec_ref(v_e_1116_);
lean_dec_ref(v___x_1115_);
v_a_1199_ = lean_ctor_get(v___x_1125_, 0);
v_isSharedCheck_1206_ = !lean_is_exclusive(v___x_1125_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1201_ = v___x_1125_;
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
else
{
lean_inc(v_a_1199_);
lean_dec(v___x_1125_);
v___x_1201_ = lean_box(0);
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
v_resetjp_1200_:
{
lean_object* v___x_1204_; 
if (v_isShared_1202_ == 0)
{
v___x_1204_ = v___x_1201_;
goto v_reusejp_1203_;
}
else
{
lean_object* v_reuseFailAlloc_1205_; 
v_reuseFailAlloc_1205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1205_, 0, v_a_1199_);
v___x_1204_ = v_reuseFailAlloc_1205_;
goto v_reusejp_1203_;
}
v_reusejp_1203_:
{
return v___x_1204_;
}
}
}
}
else
{
lean_object* v_a_1207_; lean_object* v___x_1209_; uint8_t v_isShared_1210_; uint8_t v_isSharedCheck_1214_; 
lean_dec_ref(v___y_1118_);
lean_dec_ref(v_e_1116_);
lean_dec_ref(v___x_1115_);
lean_dec(v___x_1114_);
lean_dec(v___x_1112_);
v_a_1207_ = lean_ctor_get(v___x_1123_, 0);
v_isSharedCheck_1214_ = !lean_is_exclusive(v___x_1123_);
if (v_isSharedCheck_1214_ == 0)
{
v___x_1209_ = v___x_1123_;
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
else
{
lean_inc(v_a_1207_);
lean_dec(v___x_1123_);
v___x_1209_ = lean_box(0);
v_isShared_1210_ = v_isSharedCheck_1214_;
goto v_resetjp_1208_;
}
v_resetjp_1208_:
{
lean_object* v___x_1212_; 
if (v_isShared_1210_ == 0)
{
v___x_1212_ = v___x_1209_;
goto v_reusejp_1211_;
}
else
{
lean_object* v_reuseFailAlloc_1213_; 
v_reuseFailAlloc_1213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1213_, 0, v_a_1207_);
v___x_1212_ = v_reuseFailAlloc_1213_;
goto v_reusejp_1211_;
}
v_reusejp_1211_:
{
return v___x_1212_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__4___boxed(lean_object* v___x_1215_, lean_object* v___x_1216_, lean_object* v___x_1217_, lean_object* v___x_1218_, lean_object* v_e_1219_, lean_object* v___x_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_){
_start:
{
uint8_t v___x_25813__boxed_1226_; uint8_t v___x_25816__boxed_1227_; lean_object* v_res_1228_; 
v___x_25813__boxed_1226_ = lean_unbox(v___x_1216_);
v___x_25816__boxed_1227_ = lean_unbox(v___x_1220_);
v_res_1228_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__4(v___x_1215_, v___x_25813__boxed_1226_, v___x_1217_, v___x_1218_, v_e_1219_, v___x_25816__boxed_1227_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_);
lean_dec(v___y_1224_);
lean_dec_ref(v___y_1223_);
lean_dec(v___y_1222_);
return v_res_1228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5(lean_object* v___x_1246_, uint8_t v___x_1247_, lean_object* v___x_1248_, lean_object* v___x_1249_, lean_object* v_00_u03b1_1250_, lean_object* v___x_1251_, lean_object* v___x_1252_, lean_object* v___x_1253_, lean_object* v_e_1254_, uint8_t v___x_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_){
_start:
{
lean_object* v___x_1261_; 
lean_inc(v___x_1248_);
lean_inc(v___x_1246_);
v___x_1261_ = l_Lean_Meta_mkFreshExprMVar(v___x_1246_, v___x_1247_, v___x_1248_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
if (lean_obj_tag(v___x_1261_) == 0)
{
lean_object* v_a_1262_; lean_object* v___x_1263_; 
v_a_1262_ = lean_ctor_get(v___x_1261_, 0);
lean_inc(v_a_1262_);
lean_dec_ref_known(v___x_1261_, 1);
v___x_1263_ = l_Lean_Meta_mkFreshExprMVar(v___x_1246_, v___x_1247_, v___x_1248_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_);
if (lean_obj_tag(v___x_1263_) == 0)
{
lean_object* v_a_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v_keyedConfig_1270_; uint8_t v_trackZetaDelta_1271_; lean_object* v_zetaDeltaSet_1272_; lean_object* v_lctx_1273_; lean_object* v_localInstances_1274_; lean_object* v_defEqCtx_x3f_1275_; lean_object* v_synthPendingDepth_1276_; lean_object* v_customCanUnfoldPredicate_x3f_1277_; uint8_t v_univApprox_1278_; uint8_t v_inTypeClassResolution_1279_; uint8_t v_cacheInferType_1280_; lean_object* v___x_1282_; uint8_t v_isShared_1283_; uint8_t v_isSharedCheck_1359_; 
v_a_1264_ = lean_ctor_get(v___x_1263_, 0);
lean_inc(v_a_1264_);
lean_dec_ref_known(v___x_1263_, 1);
v___x_1265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__0));
v___x_1266_ = l_Lean_Expr_const___override(v___x_1265_, v___x_1249_);
lean_inc_ref_n(v_00_u03b1_1250_, 3);
v___x_1267_ = l_Lean_Expr_app___override(v___x_1266_, v_00_u03b1_1250_);
v___x_1268_ = l_Lean_Expr_app___override(v___x_1267_, v_00_u03b1_1250_);
v___x_1269_ = l_Lean_Expr_app___override(v___x_1268_, v_00_u03b1_1250_);
v_keyedConfig_1270_ = lean_ctor_get(v___y_1256_, 0);
v_trackZetaDelta_1271_ = lean_ctor_get_uint8(v___y_1256_, sizeof(void*)*7);
v_zetaDeltaSet_1272_ = lean_ctor_get(v___y_1256_, 1);
v_lctx_1273_ = lean_ctor_get(v___y_1256_, 2);
v_localInstances_1274_ = lean_ctor_get(v___y_1256_, 3);
v_defEqCtx_x3f_1275_ = lean_ctor_get(v___y_1256_, 4);
v_synthPendingDepth_1276_ = lean_ctor_get(v___y_1256_, 5);
v_customCanUnfoldPredicate_x3f_1277_ = lean_ctor_get(v___y_1256_, 6);
v_univApprox_1278_ = lean_ctor_get_uint8(v___y_1256_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1279_ = lean_ctor_get_uint8(v___y_1256_, sizeof(void*)*7 + 2);
v_cacheInferType_1280_ = lean_ctor_get_uint8(v___y_1256_, sizeof(void*)*7 + 3);
v_isSharedCheck_1359_ = !lean_is_exclusive(v___y_1256_);
if (v_isSharedCheck_1359_ == 0)
{
v___x_1282_ = v___y_1256_;
v_isShared_1283_ = v_isSharedCheck_1359_;
goto v_resetjp_1281_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1277_);
lean_inc(v_synthPendingDepth_1276_);
lean_inc(v_defEqCtx_x3f_1275_);
lean_inc(v_localInstances_1274_);
lean_inc(v_lctx_1273_);
lean_inc(v_zetaDeltaSet_1272_);
lean_inc(v_keyedConfig_1270_);
lean_dec(v___y_1256_);
v___x_1282_ = lean_box(0);
v_isShared_1283_ = v_isSharedCheck_1359_;
goto v_resetjp_1281_;
}
v_resetjp_1281_:
{
lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; uint8_t v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1307_; 
v___x_1284_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__2));
lean_inc_n(v___x_1251_, 3);
v___x_1285_ = l_Lean_Expr_const___override(v___x_1284_, v___x_1251_);
lean_inc_ref_n(v_00_u03b1_1250_, 3);
v___x_1286_ = l_Lean_Expr_app___override(v___x_1285_, v_00_u03b1_1250_);
v___x_1287_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__5));
v___x_1288_ = l_Lean_Expr_const___override(v___x_1287_, v___x_1251_);
v___x_1289_ = l_Lean_Expr_app___override(v___x_1288_, v_00_u03b1_1250_);
v___x_1290_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__8));
v___x_1291_ = l_Lean_Expr_const___override(v___x_1290_, v___x_1251_);
v___x_1292_ = l_Lean_Expr_app___override(v___x_1291_, v_00_u03b1_1250_);
v___x_1293_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__9));
v___x_1294_ = l_Lean_Name_mkStr2(v___x_1252_, v___x_1293_);
v___x_1295_ = l_Lean_Expr_const___override(v___x_1294_, v___x_1251_);
v___x_1296_ = l_Lean_Expr_app___override(v___x_1295_, v_00_u03b1_1250_);
v___x_1297_ = l_Lean_Expr_app___override(v___x_1296_, v___x_1253_);
v___x_1298_ = l_Lean_Expr_app___override(v___x_1292_, v___x_1297_);
v___x_1299_ = l_Lean_Expr_app___override(v___x_1289_, v___x_1298_);
v___x_1300_ = l_Lean_Expr_app___override(v___x_1286_, v___x_1299_);
v___x_1301_ = l_Lean_Expr_app___override(v___x_1269_, v___x_1300_);
lean_inc(v_a_1262_);
v___x_1302_ = l_Lean_Expr_app___override(v___x_1301_, v_a_1262_);
lean_inc(v_a_1264_);
v___x_1303_ = l_Lean_Expr_app___override(v___x_1302_, v_a_1264_);
v___x_1304_ = 2;
v___x_1305_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1304_, v_keyedConfig_1270_);
if (v_isShared_1283_ == 0)
{
lean_ctor_set(v___x_1282_, 0, v___x_1305_);
v___x_1307_ = v___x_1282_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1358_; 
v_reuseFailAlloc_1358_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1358_, 0, v___x_1305_);
lean_ctor_set(v_reuseFailAlloc_1358_, 1, v_zetaDeltaSet_1272_);
lean_ctor_set(v_reuseFailAlloc_1358_, 2, v_lctx_1273_);
lean_ctor_set(v_reuseFailAlloc_1358_, 3, v_localInstances_1274_);
lean_ctor_set(v_reuseFailAlloc_1358_, 4, v_defEqCtx_x3f_1275_);
lean_ctor_set(v_reuseFailAlloc_1358_, 5, v_synthPendingDepth_1276_);
lean_ctor_set(v_reuseFailAlloc_1358_, 6, v_customCanUnfoldPredicate_x3f_1277_);
lean_ctor_set_uint8(v_reuseFailAlloc_1358_, sizeof(void*)*7, v_trackZetaDelta_1271_);
lean_ctor_set_uint8(v_reuseFailAlloc_1358_, sizeof(void*)*7 + 1, v_univApprox_1278_);
lean_ctor_set_uint8(v_reuseFailAlloc_1358_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1279_);
lean_ctor_set_uint8(v_reuseFailAlloc_1358_, sizeof(void*)*7 + 3, v_cacheInferType_1280_);
v___x_1307_ = v_reuseFailAlloc_1358_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
lean_object* v___x_1308_; 
v___x_1308_ = l_Lean_Meta_isExprDefEq(v___x_1303_, v_e_1254_, v___x_1307_, v___y_1257_, v___y_1258_, v___y_1259_);
lean_dec_ref(v___x_1307_);
if (lean_obj_tag(v___x_1308_) == 0)
{
lean_object* v_a_1309_; lean_object* v___x_1311_; uint8_t v_isShared_1312_; uint8_t v_isSharedCheck_1349_; 
v_a_1309_ = lean_ctor_get(v___x_1308_, 0);
v_isSharedCheck_1349_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1349_ == 0)
{
v___x_1311_ = v___x_1308_;
v_isShared_1312_ = v_isSharedCheck_1349_;
goto v_resetjp_1310_;
}
else
{
lean_inc(v_a_1309_);
lean_dec(v___x_1308_);
v___x_1311_ = lean_box(0);
v_isShared_1312_ = v_isSharedCheck_1349_;
goto v_resetjp_1310_;
}
v_resetjp_1310_:
{
uint8_t v___x_1313_; 
v___x_1313_ = lean_unbox(v_a_1309_);
if (v___x_1313_ == 0)
{
lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1318_; 
lean_dec(v_a_1309_);
v___x_1314_ = lean_box(v___x_1255_);
v___x_1315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1315_, 0, v_a_1264_);
lean_ctor_set(v___x_1315_, 1, v___x_1314_);
v___x_1316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1316_, 0, v_a_1262_);
lean_ctor_set(v___x_1316_, 1, v___x_1315_);
if (v_isShared_1312_ == 0)
{
lean_ctor_set(v___x_1311_, 0, v___x_1316_);
v___x_1318_ = v___x_1311_;
goto v_reusejp_1317_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v___x_1316_);
v___x_1318_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1317_;
}
v_reusejp_1317_:
{
return v___x_1318_;
}
}
else
{
lean_object* v___x_1320_; 
lean_del_object(v___x_1311_);
v___x_1320_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_1262_, v___y_1257_);
if (lean_obj_tag(v___x_1320_) == 0)
{
lean_object* v_a_1321_; lean_object* v___x_1322_; 
v_a_1321_ = lean_ctor_get(v___x_1320_, 0);
lean_inc(v_a_1321_);
lean_dec_ref_known(v___x_1320_, 1);
v___x_1322_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_1264_, v___y_1257_);
if (lean_obj_tag(v___x_1322_) == 0)
{
lean_object* v_a_1323_; lean_object* v___x_1325_; uint8_t v_isShared_1326_; uint8_t v_isSharedCheck_1332_; 
v_a_1323_ = lean_ctor_get(v___x_1322_, 0);
v_isSharedCheck_1332_ = !lean_is_exclusive(v___x_1322_);
if (v_isSharedCheck_1332_ == 0)
{
v___x_1325_ = v___x_1322_;
v_isShared_1326_ = v_isSharedCheck_1332_;
goto v_resetjp_1324_;
}
else
{
lean_inc(v_a_1323_);
lean_dec(v___x_1322_);
v___x_1325_ = lean_box(0);
v_isShared_1326_ = v_isSharedCheck_1332_;
goto v_resetjp_1324_;
}
v_resetjp_1324_:
{
lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1330_; 
v___x_1327_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1327_, 0, v_a_1323_);
lean_ctor_set(v___x_1327_, 1, v_a_1309_);
v___x_1328_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1328_, 0, v_a_1321_);
lean_ctor_set(v___x_1328_, 1, v___x_1327_);
if (v_isShared_1326_ == 0)
{
lean_ctor_set(v___x_1325_, 0, v___x_1328_);
v___x_1330_ = v___x_1325_;
goto v_reusejp_1329_;
}
else
{
lean_object* v_reuseFailAlloc_1331_; 
v_reuseFailAlloc_1331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1331_, 0, v___x_1328_);
v___x_1330_ = v_reuseFailAlloc_1331_;
goto v_reusejp_1329_;
}
v_reusejp_1329_:
{
return v___x_1330_;
}
}
}
else
{
lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1340_; 
lean_dec(v_a_1321_);
lean_dec(v_a_1309_);
v_a_1333_ = lean_ctor_get(v___x_1322_, 0);
v_isSharedCheck_1340_ = !lean_is_exclusive(v___x_1322_);
if (v_isSharedCheck_1340_ == 0)
{
v___x_1335_ = v___x_1322_;
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1322_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1340_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1338_; 
if (v_isShared_1336_ == 0)
{
v___x_1338_ = v___x_1335_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v_a_1333_);
v___x_1338_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
return v___x_1338_;
}
}
}
}
else
{
lean_object* v_a_1341_; lean_object* v___x_1343_; uint8_t v_isShared_1344_; uint8_t v_isSharedCheck_1348_; 
lean_dec(v_a_1309_);
lean_dec(v_a_1264_);
v_a_1341_ = lean_ctor_get(v___x_1320_, 0);
v_isSharedCheck_1348_ = !lean_is_exclusive(v___x_1320_);
if (v_isSharedCheck_1348_ == 0)
{
v___x_1343_ = v___x_1320_;
v_isShared_1344_ = v_isSharedCheck_1348_;
goto v_resetjp_1342_;
}
else
{
lean_inc(v_a_1341_);
lean_dec(v___x_1320_);
v___x_1343_ = lean_box(0);
v_isShared_1344_ = v_isSharedCheck_1348_;
goto v_resetjp_1342_;
}
v_resetjp_1342_:
{
lean_object* v___x_1346_; 
if (v_isShared_1344_ == 0)
{
v___x_1346_ = v___x_1343_;
goto v_reusejp_1345_;
}
else
{
lean_object* v_reuseFailAlloc_1347_; 
v_reuseFailAlloc_1347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1347_, 0, v_a_1341_);
v___x_1346_ = v_reuseFailAlloc_1347_;
goto v_reusejp_1345_;
}
v_reusejp_1345_:
{
return v___x_1346_;
}
}
}
}
}
}
else
{
lean_object* v_a_1350_; lean_object* v___x_1352_; uint8_t v_isShared_1353_; uint8_t v_isSharedCheck_1357_; 
lean_dec(v_a_1264_);
lean_dec(v_a_1262_);
v_a_1350_ = lean_ctor_get(v___x_1308_, 0);
v_isSharedCheck_1357_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1357_ == 0)
{
v___x_1352_ = v___x_1308_;
v_isShared_1353_ = v_isSharedCheck_1357_;
goto v_resetjp_1351_;
}
else
{
lean_inc(v_a_1350_);
lean_dec(v___x_1308_);
v___x_1352_ = lean_box(0);
v_isShared_1353_ = v_isSharedCheck_1357_;
goto v_resetjp_1351_;
}
v_resetjp_1351_:
{
lean_object* v___x_1355_; 
if (v_isShared_1353_ == 0)
{
v___x_1355_ = v___x_1352_;
goto v_reusejp_1354_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v_a_1350_);
v___x_1355_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1354_;
}
v_reusejp_1354_:
{
return v___x_1355_;
}
}
}
}
}
}
else
{
lean_object* v_a_1360_; lean_object* v___x_1362_; uint8_t v_isShared_1363_; uint8_t v_isSharedCheck_1367_; 
lean_dec(v_a_1262_);
lean_dec_ref(v___y_1256_);
lean_dec_ref(v_e_1254_);
lean_dec_ref(v___x_1253_);
lean_dec_ref(v___x_1252_);
lean_dec(v___x_1251_);
lean_dec_ref(v_00_u03b1_1250_);
lean_dec(v___x_1249_);
v_a_1360_ = lean_ctor_get(v___x_1263_, 0);
v_isSharedCheck_1367_ = !lean_is_exclusive(v___x_1263_);
if (v_isSharedCheck_1367_ == 0)
{
v___x_1362_ = v___x_1263_;
v_isShared_1363_ = v_isSharedCheck_1367_;
goto v_resetjp_1361_;
}
else
{
lean_inc(v_a_1360_);
lean_dec(v___x_1263_);
v___x_1362_ = lean_box(0);
v_isShared_1363_ = v_isSharedCheck_1367_;
goto v_resetjp_1361_;
}
v_resetjp_1361_:
{
lean_object* v___x_1365_; 
if (v_isShared_1363_ == 0)
{
v___x_1365_ = v___x_1362_;
goto v_reusejp_1364_;
}
else
{
lean_object* v_reuseFailAlloc_1366_; 
v_reuseFailAlloc_1366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1366_, 0, v_a_1360_);
v___x_1365_ = v_reuseFailAlloc_1366_;
goto v_reusejp_1364_;
}
v_reusejp_1364_:
{
return v___x_1365_;
}
}
}
}
else
{
lean_object* v_a_1368_; lean_object* v___x_1370_; uint8_t v_isShared_1371_; uint8_t v_isSharedCheck_1375_; 
lean_dec_ref(v___y_1256_);
lean_dec_ref(v_e_1254_);
lean_dec_ref(v___x_1253_);
lean_dec_ref(v___x_1252_);
lean_dec(v___x_1251_);
lean_dec_ref(v_00_u03b1_1250_);
lean_dec(v___x_1249_);
lean_dec(v___x_1248_);
lean_dec(v___x_1246_);
v_a_1368_ = lean_ctor_get(v___x_1261_, 0);
v_isSharedCheck_1375_ = !lean_is_exclusive(v___x_1261_);
if (v_isSharedCheck_1375_ == 0)
{
v___x_1370_ = v___x_1261_;
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
else
{
lean_inc(v_a_1368_);
lean_dec(v___x_1261_);
v___x_1370_ = lean_box(0);
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
v_resetjp_1369_:
{
lean_object* v___x_1373_; 
if (v_isShared_1371_ == 0)
{
v___x_1373_ = v___x_1370_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1374_; 
v_reuseFailAlloc_1374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1374_, 0, v_a_1368_);
v___x_1373_ = v_reuseFailAlloc_1374_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
return v___x_1373_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___boxed(lean_object* v___x_1376_, lean_object* v___x_1377_, lean_object* v___x_1378_, lean_object* v___x_1379_, lean_object* v_00_u03b1_1380_, lean_object* v___x_1381_, lean_object* v___x_1382_, lean_object* v___x_1383_, lean_object* v_e_1384_, lean_object* v___x_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_){
_start:
{
uint8_t v___x_26041__boxed_1391_; uint8_t v___x_26047__boxed_1392_; lean_object* v_res_1393_; 
v___x_26041__boxed_1391_ = lean_unbox(v___x_1377_);
v___x_26047__boxed_1392_ = lean_unbox(v___x_1385_);
v_res_1393_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5(v___x_1376_, v___x_26041__boxed_1391_, v___x_1378_, v___x_1379_, v_00_u03b1_1380_, v___x_1381_, v___x_1382_, v___x_1383_, v_e_1384_, v___x_26047__boxed_1392_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_);
lean_dec(v___y_1389_);
lean_dec_ref(v___y_1388_);
lean_dec(v___y_1387_);
return v_res_1393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6(lean_object* v___x_1401_, uint8_t v___x_1402_, lean_object* v___x_1403_, lean_object* v___x_1404_, lean_object* v_00_u03b1_1405_, lean_object* v___x_1406_, lean_object* v___x_1407_, lean_object* v___x_1408_, lean_object* v_e_1409_, uint8_t v___x_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_){
_start:
{
lean_object* v___x_1416_; 
lean_inc(v___x_1403_);
lean_inc(v___x_1401_);
v___x_1416_ = l_Lean_Meta_mkFreshExprMVar(v___x_1401_, v___x_1402_, v___x_1403_, v___y_1411_, v___y_1412_, v___y_1413_, v___y_1414_);
if (lean_obj_tag(v___x_1416_) == 0)
{
lean_object* v_a_1417_; lean_object* v___x_1418_; 
v_a_1417_ = lean_ctor_get(v___x_1416_, 0);
lean_inc(v_a_1417_);
lean_dec_ref_known(v___x_1416_, 1);
v___x_1418_ = l_Lean_Meta_mkFreshExprMVar(v___x_1401_, v___x_1402_, v___x_1403_, v___y_1411_, v___y_1412_, v___y_1413_, v___y_1414_);
if (lean_obj_tag(v___x_1418_) == 0)
{
lean_object* v_a_1419_; lean_object* v_keyedConfig_1420_; uint8_t v_trackZetaDelta_1421_; lean_object* v_zetaDeltaSet_1422_; lean_object* v_lctx_1423_; lean_object* v_localInstances_1424_; lean_object* v_defEqCtx_x3f_1425_; lean_object* v_synthPendingDepth_1426_; lean_object* v_customCanUnfoldPredicate_x3f_1427_; uint8_t v_univApprox_1428_; uint8_t v_inTypeClassResolution_1429_; uint8_t v_cacheInferType_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1506_; 
v_a_1419_ = lean_ctor_get(v___x_1418_, 0);
lean_inc(v_a_1419_);
lean_dec_ref_known(v___x_1418_, 1);
v_keyedConfig_1420_ = lean_ctor_get(v___y_1411_, 0);
v_trackZetaDelta_1421_ = lean_ctor_get_uint8(v___y_1411_, sizeof(void*)*7);
v_zetaDeltaSet_1422_ = lean_ctor_get(v___y_1411_, 1);
v_lctx_1423_ = lean_ctor_get(v___y_1411_, 2);
v_localInstances_1424_ = lean_ctor_get(v___y_1411_, 3);
v_defEqCtx_x3f_1425_ = lean_ctor_get(v___y_1411_, 4);
v_synthPendingDepth_1426_ = lean_ctor_get(v___y_1411_, 5);
v_customCanUnfoldPredicate_x3f_1427_ = lean_ctor_get(v___y_1411_, 6);
v_univApprox_1428_ = lean_ctor_get_uint8(v___y_1411_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1429_ = lean_ctor_get_uint8(v___y_1411_, sizeof(void*)*7 + 2);
v_cacheInferType_1430_ = lean_ctor_get_uint8(v___y_1411_, sizeof(void*)*7 + 3);
v_isSharedCheck_1506_ = !lean_is_exclusive(v___y_1411_);
if (v_isSharedCheck_1506_ == 0)
{
v___x_1432_ = v___y_1411_;
v_isShared_1433_ = v_isSharedCheck_1506_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1427_);
lean_inc(v_synthPendingDepth_1426_);
lean_inc(v_defEqCtx_x3f_1425_);
lean_inc(v_localInstances_1424_);
lean_inc(v_lctx_1423_);
lean_inc(v_zetaDeltaSet_1422_);
lean_inc(v_keyedConfig_1420_);
lean_dec(v___y_1411_);
v___x_1432_ = lean_box(0);
v_isShared_1433_ = v_isSharedCheck_1506_;
goto v_resetjp_1431_;
}
v_resetjp_1431_:
{
lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; uint8_t v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1454_; 
v___x_1434_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__0));
v___x_1435_ = l_Lean_Expr_const___override(v___x_1434_, v___x_1404_);
lean_inc_ref_n(v_00_u03b1_1405_, 4);
v___x_1436_ = l_Lean_Expr_app___override(v___x_1435_, v_00_u03b1_1405_);
v___x_1437_ = l_Lean_Expr_app___override(v___x_1436_, v_00_u03b1_1405_);
v___x_1438_ = l_Lean_Expr_app___override(v___x_1437_, v_00_u03b1_1405_);
v___x_1439_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__2));
lean_inc(v___x_1406_);
v___x_1440_ = l_Lean_Expr_const___override(v___x_1439_, v___x_1406_);
v___x_1441_ = l_Lean_Expr_app___override(v___x_1440_, v_00_u03b1_1405_);
v___x_1442_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__3));
v___x_1443_ = l_Lean_Name_mkStr2(v___x_1407_, v___x_1442_);
v___x_1444_ = l_Lean_Expr_const___override(v___x_1443_, v___x_1406_);
v___x_1445_ = l_Lean_Expr_app___override(v___x_1444_, v_00_u03b1_1405_);
v___x_1446_ = l_Lean_Expr_app___override(v___x_1445_, v___x_1408_);
v___x_1447_ = l_Lean_Expr_app___override(v___x_1441_, v___x_1446_);
v___x_1448_ = l_Lean_Expr_app___override(v___x_1438_, v___x_1447_);
lean_inc(v_a_1417_);
v___x_1449_ = l_Lean_Expr_app___override(v___x_1448_, v_a_1417_);
lean_inc(v_a_1419_);
v___x_1450_ = l_Lean_Expr_app___override(v___x_1449_, v_a_1419_);
v___x_1451_ = 2;
v___x_1452_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1451_, v_keyedConfig_1420_);
if (v_isShared_1433_ == 0)
{
lean_ctor_set(v___x_1432_, 0, v___x_1452_);
v___x_1454_ = v___x_1432_;
goto v_reusejp_1453_;
}
else
{
lean_object* v_reuseFailAlloc_1505_; 
v_reuseFailAlloc_1505_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1505_, 0, v___x_1452_);
lean_ctor_set(v_reuseFailAlloc_1505_, 1, v_zetaDeltaSet_1422_);
lean_ctor_set(v_reuseFailAlloc_1505_, 2, v_lctx_1423_);
lean_ctor_set(v_reuseFailAlloc_1505_, 3, v_localInstances_1424_);
lean_ctor_set(v_reuseFailAlloc_1505_, 4, v_defEqCtx_x3f_1425_);
lean_ctor_set(v_reuseFailAlloc_1505_, 5, v_synthPendingDepth_1426_);
lean_ctor_set(v_reuseFailAlloc_1505_, 6, v_customCanUnfoldPredicate_x3f_1427_);
lean_ctor_set_uint8(v_reuseFailAlloc_1505_, sizeof(void*)*7, v_trackZetaDelta_1421_);
lean_ctor_set_uint8(v_reuseFailAlloc_1505_, sizeof(void*)*7 + 1, v_univApprox_1428_);
lean_ctor_set_uint8(v_reuseFailAlloc_1505_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1429_);
lean_ctor_set_uint8(v_reuseFailAlloc_1505_, sizeof(void*)*7 + 3, v_cacheInferType_1430_);
v___x_1454_ = v_reuseFailAlloc_1505_;
goto v_reusejp_1453_;
}
v_reusejp_1453_:
{
lean_object* v___x_1455_; 
v___x_1455_ = l_Lean_Meta_isExprDefEq(v___x_1450_, v_e_1409_, v___x_1454_, v___y_1412_, v___y_1413_, v___y_1414_);
lean_dec_ref(v___x_1454_);
if (lean_obj_tag(v___x_1455_) == 0)
{
lean_object* v_a_1456_; lean_object* v___x_1458_; uint8_t v_isShared_1459_; uint8_t v_isSharedCheck_1496_; 
v_a_1456_ = lean_ctor_get(v___x_1455_, 0);
v_isSharedCheck_1496_ = !lean_is_exclusive(v___x_1455_);
if (v_isSharedCheck_1496_ == 0)
{
v___x_1458_ = v___x_1455_;
v_isShared_1459_ = v_isSharedCheck_1496_;
goto v_resetjp_1457_;
}
else
{
lean_inc(v_a_1456_);
lean_dec(v___x_1455_);
v___x_1458_ = lean_box(0);
v_isShared_1459_ = v_isSharedCheck_1496_;
goto v_resetjp_1457_;
}
v_resetjp_1457_:
{
uint8_t v___x_1460_; 
v___x_1460_ = lean_unbox(v_a_1456_);
if (v___x_1460_ == 0)
{
lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1465_; 
lean_dec(v_a_1456_);
v___x_1461_ = lean_box(v___x_1410_);
v___x_1462_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1462_, 0, v_a_1419_);
lean_ctor_set(v___x_1462_, 1, v___x_1461_);
v___x_1463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1463_, 0, v_a_1417_);
lean_ctor_set(v___x_1463_, 1, v___x_1462_);
if (v_isShared_1459_ == 0)
{
lean_ctor_set(v___x_1458_, 0, v___x_1463_);
v___x_1465_ = v___x_1458_;
goto v_reusejp_1464_;
}
else
{
lean_object* v_reuseFailAlloc_1466_; 
v_reuseFailAlloc_1466_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1466_, 0, v___x_1463_);
v___x_1465_ = v_reuseFailAlloc_1466_;
goto v_reusejp_1464_;
}
v_reusejp_1464_:
{
return v___x_1465_;
}
}
else
{
lean_object* v___x_1467_; 
lean_del_object(v___x_1458_);
v___x_1467_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_1417_, v___y_1412_);
if (lean_obj_tag(v___x_1467_) == 0)
{
lean_object* v_a_1468_; lean_object* v___x_1469_; 
v_a_1468_ = lean_ctor_get(v___x_1467_, 0);
lean_inc(v_a_1468_);
lean_dec_ref_known(v___x_1467_, 1);
v___x_1469_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__0___redArg(v_a_1419_, v___y_1412_);
if (lean_obj_tag(v___x_1469_) == 0)
{
lean_object* v_a_1470_; lean_object* v___x_1472_; uint8_t v_isShared_1473_; uint8_t v_isSharedCheck_1479_; 
v_a_1470_ = lean_ctor_get(v___x_1469_, 0);
v_isSharedCheck_1479_ = !lean_is_exclusive(v___x_1469_);
if (v_isSharedCheck_1479_ == 0)
{
v___x_1472_ = v___x_1469_;
v_isShared_1473_ = v_isSharedCheck_1479_;
goto v_resetjp_1471_;
}
else
{
lean_inc(v_a_1470_);
lean_dec(v___x_1469_);
v___x_1472_ = lean_box(0);
v_isShared_1473_ = v_isSharedCheck_1479_;
goto v_resetjp_1471_;
}
v_resetjp_1471_:
{
lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1477_; 
v___x_1474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1474_, 0, v_a_1470_);
lean_ctor_set(v___x_1474_, 1, v_a_1456_);
v___x_1475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1475_, 0, v_a_1468_);
lean_ctor_set(v___x_1475_, 1, v___x_1474_);
if (v_isShared_1473_ == 0)
{
lean_ctor_set(v___x_1472_, 0, v___x_1475_);
v___x_1477_ = v___x_1472_;
goto v_reusejp_1476_;
}
else
{
lean_object* v_reuseFailAlloc_1478_; 
v_reuseFailAlloc_1478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1478_, 0, v___x_1475_);
v___x_1477_ = v_reuseFailAlloc_1478_;
goto v_reusejp_1476_;
}
v_reusejp_1476_:
{
return v___x_1477_;
}
}
}
else
{
lean_object* v_a_1480_; lean_object* v___x_1482_; uint8_t v_isShared_1483_; uint8_t v_isSharedCheck_1487_; 
lean_dec(v_a_1468_);
lean_dec(v_a_1456_);
v_a_1480_ = lean_ctor_get(v___x_1469_, 0);
v_isSharedCheck_1487_ = !lean_is_exclusive(v___x_1469_);
if (v_isSharedCheck_1487_ == 0)
{
v___x_1482_ = v___x_1469_;
v_isShared_1483_ = v_isSharedCheck_1487_;
goto v_resetjp_1481_;
}
else
{
lean_inc(v_a_1480_);
lean_dec(v___x_1469_);
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
lean_dec(v_a_1456_);
lean_dec(v_a_1419_);
v_a_1488_ = lean_ctor_get(v___x_1467_, 0);
v_isSharedCheck_1495_ = !lean_is_exclusive(v___x_1467_);
if (v_isSharedCheck_1495_ == 0)
{
v___x_1490_ = v___x_1467_;
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
else
{
lean_inc(v_a_1488_);
lean_dec(v___x_1467_);
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
}
}
else
{
lean_object* v_a_1497_; lean_object* v___x_1499_; uint8_t v_isShared_1500_; uint8_t v_isSharedCheck_1504_; 
lean_dec(v_a_1419_);
lean_dec(v_a_1417_);
v_a_1497_ = lean_ctor_get(v___x_1455_, 0);
v_isSharedCheck_1504_ = !lean_is_exclusive(v___x_1455_);
if (v_isSharedCheck_1504_ == 0)
{
v___x_1499_ = v___x_1455_;
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
else
{
lean_inc(v_a_1497_);
lean_dec(v___x_1455_);
v___x_1499_ = lean_box(0);
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
v_resetjp_1498_:
{
lean_object* v___x_1502_; 
if (v_isShared_1500_ == 0)
{
v___x_1502_ = v___x_1499_;
goto v_reusejp_1501_;
}
else
{
lean_object* v_reuseFailAlloc_1503_; 
v_reuseFailAlloc_1503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1503_, 0, v_a_1497_);
v___x_1502_ = v_reuseFailAlloc_1503_;
goto v_reusejp_1501_;
}
v_reusejp_1501_:
{
return v___x_1502_;
}
}
}
}
}
}
else
{
lean_object* v_a_1507_; lean_object* v___x_1509_; uint8_t v_isShared_1510_; uint8_t v_isSharedCheck_1514_; 
lean_dec(v_a_1417_);
lean_dec_ref(v___y_1411_);
lean_dec_ref(v_e_1409_);
lean_dec_ref(v___x_1408_);
lean_dec_ref(v___x_1407_);
lean_dec(v___x_1406_);
lean_dec_ref(v_00_u03b1_1405_);
lean_dec(v___x_1404_);
v_a_1507_ = lean_ctor_get(v___x_1418_, 0);
v_isSharedCheck_1514_ = !lean_is_exclusive(v___x_1418_);
if (v_isSharedCheck_1514_ == 0)
{
v___x_1509_ = v___x_1418_;
v_isShared_1510_ = v_isSharedCheck_1514_;
goto v_resetjp_1508_;
}
else
{
lean_inc(v_a_1507_);
lean_dec(v___x_1418_);
v___x_1509_ = lean_box(0);
v_isShared_1510_ = v_isSharedCheck_1514_;
goto v_resetjp_1508_;
}
v_resetjp_1508_:
{
lean_object* v___x_1512_; 
if (v_isShared_1510_ == 0)
{
v___x_1512_ = v___x_1509_;
goto v_reusejp_1511_;
}
else
{
lean_object* v_reuseFailAlloc_1513_; 
v_reuseFailAlloc_1513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1513_, 0, v_a_1507_);
v___x_1512_ = v_reuseFailAlloc_1513_;
goto v_reusejp_1511_;
}
v_reusejp_1511_:
{
return v___x_1512_;
}
}
}
}
else
{
lean_object* v_a_1515_; lean_object* v___x_1517_; uint8_t v_isShared_1518_; uint8_t v_isSharedCheck_1522_; 
lean_dec_ref(v___y_1411_);
lean_dec_ref(v_e_1409_);
lean_dec_ref(v___x_1408_);
lean_dec_ref(v___x_1407_);
lean_dec(v___x_1406_);
lean_dec_ref(v_00_u03b1_1405_);
lean_dec(v___x_1404_);
lean_dec(v___x_1403_);
lean_dec(v___x_1401_);
v_a_1515_ = lean_ctor_get(v___x_1416_, 0);
v_isSharedCheck_1522_ = !lean_is_exclusive(v___x_1416_);
if (v_isSharedCheck_1522_ == 0)
{
v___x_1517_ = v___x_1416_;
v_isShared_1518_ = v_isSharedCheck_1522_;
goto v_resetjp_1516_;
}
else
{
lean_inc(v_a_1515_);
lean_dec(v___x_1416_);
v___x_1517_ = lean_box(0);
v_isShared_1518_ = v_isSharedCheck_1522_;
goto v_resetjp_1516_;
}
v_resetjp_1516_:
{
lean_object* v___x_1520_; 
if (v_isShared_1518_ == 0)
{
v___x_1520_ = v___x_1517_;
goto v_reusejp_1519_;
}
else
{
lean_object* v_reuseFailAlloc_1521_; 
v_reuseFailAlloc_1521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1521_, 0, v_a_1515_);
v___x_1520_ = v_reuseFailAlloc_1521_;
goto v_reusejp_1519_;
}
v_reusejp_1519_:
{
return v___x_1520_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___boxed(lean_object* v___x_1523_, lean_object* v___x_1524_, lean_object* v___x_1525_, lean_object* v___x_1526_, lean_object* v_00_u03b1_1527_, lean_object* v___x_1528_, lean_object* v___x_1529_, lean_object* v___x_1530_, lean_object* v_e_1531_, lean_object* v___x_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_, lean_object* v___y_1537_){
_start:
{
uint8_t v___x_26321__boxed_1538_; uint8_t v___x_26327__boxed_1539_; lean_object* v_res_1540_; 
v___x_26321__boxed_1538_ = lean_unbox(v___x_1524_);
v___x_26327__boxed_1539_ = lean_unbox(v___x_1532_);
v_res_1540_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6(v___x_1523_, v___x_26321__boxed_1538_, v___x_1525_, v___x_1526_, v_00_u03b1_1527_, v___x_1528_, v___x_1529_, v___x_1530_, v_e_1531_, v___x_26327__boxed_1539_, v___y_1533_, v___y_1534_, v___y_1535_, v___y_1536_);
lean_dec(v___y_1536_);
lean_dec_ref(v___y_1535_);
lean_dec(v___y_1534_);
return v_res_1540_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1541_; double v___x_1542_; 
v___x_1541_ = lean_unsigned_to_nat(0u);
v___x_1542_ = lean_float_of_nat(v___x_1541_);
return v___x_1542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(lean_object* v_cls_1546_, lean_object* v_msg_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_){
_start:
{
lean_object* v_ref_1553_; lean_object* v___x_1554_; lean_object* v_a_1555_; lean_object* v___x_1557_; uint8_t v_isShared_1558_; uint8_t v_isSharedCheck_1599_; 
v_ref_1553_ = lean_ctor_get(v___y_1550_, 5);
v___x_1554_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(v_msg_1547_, v___y_1548_, v___y_1549_, v___y_1550_, v___y_1551_);
v_a_1555_ = lean_ctor_get(v___x_1554_, 0);
v_isSharedCheck_1599_ = !lean_is_exclusive(v___x_1554_);
if (v_isSharedCheck_1599_ == 0)
{
v___x_1557_ = v___x_1554_;
v_isShared_1558_ = v_isSharedCheck_1599_;
goto v_resetjp_1556_;
}
else
{
lean_inc(v_a_1555_);
lean_dec(v___x_1554_);
v___x_1557_ = lean_box(0);
v_isShared_1558_ = v_isSharedCheck_1599_;
goto v_resetjp_1556_;
}
v_resetjp_1556_:
{
lean_object* v___x_1559_; lean_object* v_traceState_1560_; lean_object* v_env_1561_; lean_object* v_nextMacroScope_1562_; lean_object* v_ngen_1563_; lean_object* v_auxDeclNGen_1564_; lean_object* v_cache_1565_; lean_object* v_messages_1566_; lean_object* v_infoState_1567_; lean_object* v_snapshotTasks_1568_; lean_object* v___x_1570_; uint8_t v_isShared_1571_; uint8_t v_isSharedCheck_1598_; 
v___x_1559_ = lean_st_ref_take(v___y_1551_);
v_traceState_1560_ = lean_ctor_get(v___x_1559_, 4);
v_env_1561_ = lean_ctor_get(v___x_1559_, 0);
v_nextMacroScope_1562_ = lean_ctor_get(v___x_1559_, 1);
v_ngen_1563_ = lean_ctor_get(v___x_1559_, 2);
v_auxDeclNGen_1564_ = lean_ctor_get(v___x_1559_, 3);
v_cache_1565_ = lean_ctor_get(v___x_1559_, 5);
v_messages_1566_ = lean_ctor_get(v___x_1559_, 6);
v_infoState_1567_ = lean_ctor_get(v___x_1559_, 7);
v_snapshotTasks_1568_ = lean_ctor_get(v___x_1559_, 8);
v_isSharedCheck_1598_ = !lean_is_exclusive(v___x_1559_);
if (v_isSharedCheck_1598_ == 0)
{
v___x_1570_ = v___x_1559_;
v_isShared_1571_ = v_isSharedCheck_1598_;
goto v_resetjp_1569_;
}
else
{
lean_inc(v_snapshotTasks_1568_);
lean_inc(v_infoState_1567_);
lean_inc(v_messages_1566_);
lean_inc(v_cache_1565_);
lean_inc(v_traceState_1560_);
lean_inc(v_auxDeclNGen_1564_);
lean_inc(v_ngen_1563_);
lean_inc(v_nextMacroScope_1562_);
lean_inc(v_env_1561_);
lean_dec(v___x_1559_);
v___x_1570_ = lean_box(0);
v_isShared_1571_ = v_isSharedCheck_1598_;
goto v_resetjp_1569_;
}
v_resetjp_1569_:
{
uint64_t v_tid_1572_; lean_object* v_traces_1573_; lean_object* v___x_1575_; uint8_t v_isShared_1576_; uint8_t v_isSharedCheck_1597_; 
v_tid_1572_ = lean_ctor_get_uint64(v_traceState_1560_, sizeof(void*)*1);
v_traces_1573_ = lean_ctor_get(v_traceState_1560_, 0);
v_isSharedCheck_1597_ = !lean_is_exclusive(v_traceState_1560_);
if (v_isSharedCheck_1597_ == 0)
{
v___x_1575_ = v_traceState_1560_;
v_isShared_1576_ = v_isSharedCheck_1597_;
goto v_resetjp_1574_;
}
else
{
lean_inc(v_traces_1573_);
lean_dec(v_traceState_1560_);
v___x_1575_ = lean_box(0);
v_isShared_1576_ = v_isSharedCheck_1597_;
goto v_resetjp_1574_;
}
v_resetjp_1574_:
{
lean_object* v___x_1577_; double v___x_1578_; uint8_t v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1587_; 
v___x_1577_ = lean_box(0);
v___x_1578_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__0);
v___x_1579_ = 0;
v___x_1580_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__1));
v___x_1581_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1581_, 0, v_cls_1546_);
lean_ctor_set(v___x_1581_, 1, v___x_1577_);
lean_ctor_set(v___x_1581_, 2, v___x_1580_);
lean_ctor_set_float(v___x_1581_, sizeof(void*)*3, v___x_1578_);
lean_ctor_set_float(v___x_1581_, sizeof(void*)*3 + 8, v___x_1578_);
lean_ctor_set_uint8(v___x_1581_, sizeof(void*)*3 + 16, v___x_1579_);
v___x_1582_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___closed__2));
v___x_1583_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1583_, 0, v___x_1581_);
lean_ctor_set(v___x_1583_, 1, v_a_1555_);
lean_ctor_set(v___x_1583_, 2, v___x_1582_);
lean_inc(v_ref_1553_);
v___x_1584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1584_, 0, v_ref_1553_);
lean_ctor_set(v___x_1584_, 1, v___x_1583_);
v___x_1585_ = l_Lean_PersistentArray_push___redArg(v_traces_1573_, v___x_1584_);
if (v_isShared_1576_ == 0)
{
lean_ctor_set(v___x_1575_, 0, v___x_1585_);
v___x_1587_ = v___x_1575_;
goto v_reusejp_1586_;
}
else
{
lean_object* v_reuseFailAlloc_1596_; 
v_reuseFailAlloc_1596_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1596_, 0, v___x_1585_);
lean_ctor_set_uint64(v_reuseFailAlloc_1596_, sizeof(void*)*1, v_tid_1572_);
v___x_1587_ = v_reuseFailAlloc_1596_;
goto v_reusejp_1586_;
}
v_reusejp_1586_:
{
lean_object* v___x_1589_; 
if (v_isShared_1571_ == 0)
{
lean_ctor_set(v___x_1570_, 4, v___x_1587_);
v___x_1589_ = v___x_1570_;
goto v_reusejp_1588_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v_env_1561_);
lean_ctor_set(v_reuseFailAlloc_1595_, 1, v_nextMacroScope_1562_);
lean_ctor_set(v_reuseFailAlloc_1595_, 2, v_ngen_1563_);
lean_ctor_set(v_reuseFailAlloc_1595_, 3, v_auxDeclNGen_1564_);
lean_ctor_set(v_reuseFailAlloc_1595_, 4, v___x_1587_);
lean_ctor_set(v_reuseFailAlloc_1595_, 5, v_cache_1565_);
lean_ctor_set(v_reuseFailAlloc_1595_, 6, v_messages_1566_);
lean_ctor_set(v_reuseFailAlloc_1595_, 7, v_infoState_1567_);
lean_ctor_set(v_reuseFailAlloc_1595_, 8, v_snapshotTasks_1568_);
v___x_1589_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1588_;
}
v_reusejp_1588_:
{
lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1593_; 
v___x_1590_ = lean_st_ref_set(v___y_1551_, v___x_1589_);
v___x_1591_ = lean_box(0);
if (v_isShared_1558_ == 0)
{
lean_ctor_set(v___x_1557_, 0, v___x_1591_);
v___x_1593_ = v___x_1557_;
goto v_reusejp_1592_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v___x_1591_);
v___x_1593_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1592_;
}
v_reusejp_1592_:
{
return v___x_1593_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2___boxed(lean_object* v_cls_1600_, lean_object* v_msg_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_){
_start:
{
lean_object* v_res_1607_; 
v_res_1607_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(v_cls_1600_, v_msg_1601_, v___y_1602_, v___y_1603_, v___y_1604_, v___y_1605_);
lean_dec(v___y_1605_);
lean_dec_ref(v___y_1604_);
lean_dec(v___y_1603_);
lean_dec_ref(v___y_1602_);
return v_res_1607_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22(void){
_start:
{
lean_object* v___x_1644_; lean_object* v___x_1645_; 
v___x_1644_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__21));
v___x_1645_ = l_Lean_Expr_lit___override(v___x_1644_);
return v___x_1645_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__33(void){
_start:
{
lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; 
v___x_1665_ = lean_box(0);
v___x_1666_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__1));
v___x_1667_ = l_Lean_Expr_const___override(v___x_1666_, v___x_1665_);
return v___x_1667_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__44(void){
_start:
{
lean_object* v___x_1695_; lean_object* v___x_1696_; 
v___x_1695_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__43));
v___x_1696_ = l_Lean_Expr_lit___override(v___x_1695_);
return v___x_1696_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56(void){
_start:
{
lean_object* v_cls_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; 
v_cls_1721_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_1722_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__55));
v___x_1723_ = l_Lean_Name_append(v___x_1722_, v_cls_1721_);
return v___x_1723_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__58(void){
_start:
{
lean_object* v___x_1725_; lean_object* v___x_1726_; 
v___x_1725_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__57));
v___x_1726_ = l_Lean_stringToMessageData(v___x_1725_);
return v___x_1726_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__88(void){
_start:
{
lean_object* v___x_1783_; lean_object* v___x_1784_; 
v___x_1783_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__87));
v___x_1784_ = l_Lean_stringToMessageData(v___x_1783_);
return v___x_1784_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__90(void){
_start:
{
lean_object* v___x_1786_; lean_object* v___x_1787_; 
v___x_1786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__89));
v___x_1787_ = l_Lean_stringToMessageData(v___x_1786_);
return v___x_1787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(lean_object* v_u_1788_, lean_object* v_00_u03b1_1789_, lean_object* v_s_u03b1_1790_, lean_object* v_v_1791_, lean_object* v_v_x27_1792_, lean_object* v_t_1793_, lean_object* v_e_1794_, lean_object* v_a_1795_, lean_object* v_a_1796_, lean_object* v_a_1797_, lean_object* v_a_1798_){
_start:
{
lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v_options_1813_; lean_object* v_inheritedTraceOptions_1814_; uint8_t v_hasTrace_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v_amwo_1832_; lean_object* v___y_1834_; lean_object* v___y_1835_; lean_object* v___y_1836_; lean_object* v___y_1837_; lean_object* v___y_1838_; lean_object* v___y_1839_; lean_object* v_rn_1840_; lean_object* v___y_1841_; lean_object* v___y_1842_; lean_object* v___y_1843_; lean_object* v___y_1844_; lean_object* v___y_1968_; lean_object* v___y_1969_; lean_object* v___y_1970_; lean_object* v___y_1971_; lean_object* v___y_1972_; lean_object* v___y_1973_; lean_object* v___y_1974_; lean_object* v_lhs_1975_; lean_object* v_k1_1976_; lean_object* v_k2_1977_; lean_object* v___y_1978_; lean_object* v___y_1979_; lean_object* v___y_1980_; lean_object* v___y_1981_; lean_object* v___y_2149_; lean_object* v___y_2150_; lean_object* v___y_2151_; lean_object* v___y_2152_; lean_object* v___y_2153_; lean_object* v___y_2154_; lean_object* v___y_2155_; lean_object* v___y_2156_; lean_object* v___y_2157_; lean_object* v___y_2158_; lean_object* v___y_2166_; lean_object* v___y_2167_; lean_object* v___y_2168_; lean_object* v___y_2169_; lean_object* v___y_2170_; uint8_t v___y_2171_; lean_object* v___y_2172_; lean_object* v___y_2173_; lean_object* v___y_2174_; lean_object* v___y_2175_; lean_object* v___y_2176_; lean_object* v_lhs_2177_; lean_object* v_rn_2178_; lean_object* v___y_2179_; lean_object* v___y_2180_; lean_object* v___y_2181_; lean_object* v___y_2182_; lean_object* v___y_2389_; lean_object* v___y_2390_; lean_object* v___y_2391_; lean_object* v___y_2392_; lean_object* v___y_2393_; uint8_t v___y_2394_; lean_object* v___y_2395_; lean_object* v___y_2396_; lean_object* v___y_2397_; lean_object* v___y_2398_; lean_object* v___y_2399_; lean_object* v___y_2400_; lean_object* v___y_2401_; lean_object* v___y_2477_; lean_object* v___y_2478_; lean_object* v___y_2479_; lean_object* v___y_2480_; uint8_t v___y_2481_; lean_object* v___y_2482_; lean_object* v___y_2483_; lean_object* v___y_2484_; lean_object* v___y_2485_; lean_object* v___y_2486_; lean_object* v___y_2487_; lean_object* v___y_2488_; lean_object* v___y_2489_; lean_object* v___y_2490_; lean_object* v___y_2491_; lean_object* v___y_2496_; lean_object* v___y_2497_; lean_object* v___y_2498_; lean_object* v___y_2499_; lean_object* v___y_2500_; lean_object* v___y_2501_; lean_object* v___y_2502_; lean_object* v___y_2503_; lean_object* v___y_2504_; lean_object* v___y_2505_; lean_object* v___y_2506_; lean_object* v_cls_2588_; lean_object* v___y_2590_; lean_object* v___y_2591_; lean_object* v___y_2592_; lean_object* v___y_2593_; lean_object* v___y_2594_; uint8_t v___y_2595_; lean_object* v___y_2596_; lean_object* v___y_2597_; lean_object* v___y_2598_; lean_object* v___y_2599_; lean_object* v___y_2600_; lean_object* v___y_2601_; lean_object* v_lhs_2602_; lean_object* v_ln_2603_; lean_object* v_rhs_2604_; lean_object* v___y_2605_; lean_object* v___y_2606_; lean_object* v___y_2607_; lean_object* v___y_2608_; lean_object* v___y_2645_; lean_object* v___y_2646_; lean_object* v___y_2647_; lean_object* v___y_2648_; lean_object* v___y_2649_; lean_object* v___y_2650_; lean_object* v___y_2651_; lean_object* v___y_2652_; lean_object* v___y_2653_; uint8_t v___y_2654_; lean_object* v___y_2655_; lean_object* v___y_2656_; lean_object* v___y_2657_; lean_object* v___y_2658_; lean_object* v___y_2659_; lean_object* v___y_2660_; lean_object* v_lhs_2661_; lean_object* v_rhs_2662_; lean_object* v___y_2739_; lean_object* v___y_2740_; lean_object* v___y_2741_; lean_object* v___y_2742_; 
v___x_1800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__1));
v___x_1801_ = lean_box(0);
lean_inc_n(v_u_1788_, 2);
v___x_1802_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1802_, 0, v_u_1788_);
lean_ctor_set(v___x_1802_, 1, v___x_1801_);
lean_inc_ref_n(v___x_1802_, 5);
v___x_1803_ = l_Lean_Expr_const___override(v___x_1800_, v___x_1802_);
lean_inc_ref_n(v_00_u03b1_1789_, 5);
v___x_1804_ = l_Lean_Expr_app___override(v___x_1803_, v_00_u03b1_1789_);
v___x_1805_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__2));
v___x_1806_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__4));
v___x_1807_ = l_Lean_Expr_const___override(v___x_1806_, v___x_1802_);
v___x_1808_ = l_Lean_Expr_app___override(v___x_1807_, v_00_u03b1_1789_);
v___x_1809_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__5));
v___x_1810_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__7));
v___x_1811_ = l_Lean_Expr_const___override(v___x_1810_, v___x_1802_);
v___x_1812_ = l_Lean_Expr_app___override(v___x_1811_, v_00_u03b1_1789_);
v_options_1813_ = lean_ctor_get(v_a_1797_, 2);
v_inheritedTraceOptions_1814_ = lean_ctor_get(v_a_1797_, 13);
v_hasTrace_1815_ = lean_ctor_get_uint8(v_options_1813_, sizeof(void*)*1);
v___x_1816_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__8));
v___x_1817_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__10));
v___x_1818_ = l_Lean_Expr_const___override(v___x_1817_, v___x_1802_);
v___x_1819_ = l_Lean_Expr_app___override(v___x_1818_, v_00_u03b1_1789_);
v___x_1820_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__12));
v___x_1821_ = l_Lean_Level_succ___override(v_u_1788_);
v___x_1822_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1822_, 0, v___x_1821_);
lean_ctor_set(v___x_1822_, 1, v___x_1801_);
lean_inc_ref(v___x_1822_);
v___x_1823_ = l_Lean_Expr_const___override(v___x_1820_, v___x_1822_);
lean_inc_ref(v___x_1823_);
v___x_1824_ = l_Lean_Expr_app___override(v___x_1823_, v___x_1804_);
v___x_1825_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__15));
v___x_1826_ = l_Lean_Expr_const___override(v___x_1825_, v___x_1802_);
v___x_1827_ = l_Lean_Expr_app___override(v___x_1826_, v_00_u03b1_1789_);
lean_inc_ref(v_s_u03b1_1790_);
v___x_1828_ = l_Lean_Expr_app___override(v___x_1827_, v_s_u03b1_1790_);
lean_inc_ref(v___x_1828_);
v___x_1829_ = l_Lean_Expr_app___override(v___x_1819_, v___x_1828_);
lean_inc_ref(v___x_1829_);
v___x_1830_ = l_Lean_Expr_app___override(v___x_1812_, v___x_1829_);
lean_inc_ref(v___x_1830_);
v___x_1831_ = l_Lean_Expr_app___override(v___x_1808_, v___x_1830_);
v_amwo_1832_ = l_Lean_Expr_app___override(v___x_1824_, v___x_1831_);
v_cls_2588_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
if (v_hasTrace_1815_ == 0)
{
v___y_2739_ = v_a_1795_;
v___y_2740_ = v_a_1796_;
v___y_2741_ = v_a_1797_;
v___y_2742_ = v_a_1798_;
goto v___jp_2738_;
}
else
{
lean_object* v___x_2885_; uint8_t v___x_2886_; 
v___x_2885_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56);
v___x_2886_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1814_, v_options_1813_, v___x_2885_);
if (v___x_2886_ == 0)
{
v___y_2739_ = v_a_1795_;
v___y_2740_ = v_a_1796_;
v___y_2741_ = v_a_1797_;
v___y_2742_ = v_a_1798_;
goto v___jp_2738_;
}
else
{
lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; 
v___x_2887_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__88, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__88_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__88);
lean_inc_ref(v_e_1794_);
v___x_2888_ = l_Lean_MessageData_ofExpr(v_e_1794_);
v___x_2889_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2889_, 0, v___x_2887_);
lean_ctor_set(v___x_2889_, 1, v___x_2888_);
v___x_2890_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__90, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__90_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__90);
v___x_2891_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2891_, 0, v___x_2889_);
lean_ctor_set(v___x_2891_, 1, v___x_2890_);
lean_inc(v_v_1791_);
v___x_2892_ = l_Nat_reprFast(v_v_1791_);
v___x_2893_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2893_, 0, v___x_2892_);
v___x_2894_ = l_Lean_MessageData_ofFormat(v___x_2893_);
v___x_2895_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2895_, 0, v___x_2891_);
lean_ctor_set(v___x_2895_, 1, v___x_2894_);
v___x_2896_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(v_cls_2588_, v___x_2895_, v_a_1795_, v_a_1796_, v_a_1797_, v_a_1798_);
if (lean_obj_tag(v___x_2896_) == 0)
{
lean_dec_ref_known(v___x_2896_, 1);
v___y_2739_ = v_a_1795_;
v___y_2740_ = v_a_1796_;
v___y_2741_ = v_a_1797_;
v___y_2742_ = v_a_1798_;
goto v___jp_2738_;
}
else
{
lean_object* v_a_2897_; lean_object* v___x_2899_; uint8_t v_isShared_2900_; uint8_t v_isSharedCheck_2904_; 
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1830_);
lean_dec_ref(v___x_1829_);
lean_dec_ref(v___x_1828_);
lean_dec_ref(v___x_1823_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2897_ = lean_ctor_get(v___x_2896_, 0);
v_isSharedCheck_2904_ = !lean_is_exclusive(v___x_2896_);
if (v_isSharedCheck_2904_ == 0)
{
v___x_2899_ = v___x_2896_;
v_isShared_2900_ = v_isSharedCheck_2904_;
goto v_resetjp_2898_;
}
else
{
lean_inc(v_a_2897_);
lean_dec(v___x_2896_);
v___x_2899_ = lean_box(0);
v_isShared_2900_ = v_isSharedCheck_2904_;
goto v_resetjp_2898_;
}
v_resetjp_2898_:
{
lean_object* v___x_2902_; 
if (v_isShared_2900_ == 0)
{
v___x_2902_ = v___x_2899_;
goto v_reusejp_2901_;
}
else
{
lean_object* v_reuseFailAlloc_2903_; 
v_reuseFailAlloc_2903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2903_, 0, v_a_2897_);
v___x_2902_ = v_reuseFailAlloc_2903_;
goto v_reusejp_2901_;
}
v_reusejp_2901_:
{
return v___x_2902_;
}
}
}
}
}
v___jp_1833_:
{
lean_object* v___x_1845_; uint8_t v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___f_1849_; uint8_t v___x_1850_; lean_object* v___x_1851_; 
lean_inc_ref_n(v_00_u03b1_1789_, 2);
v___x_1845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1845_, 0, v_00_u03b1_1789_);
v___x_1846_ = 0;
v___x_1847_ = lean_box(0);
v___x_1848_ = lean_box(v___x_1846_);
lean_inc_ref(v___y_1834_);
lean_inc_ref(v___x_1802_);
v___f_1849_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__0___boxed), 13, 8);
lean_closure_set(v___f_1849_, 0, v___x_1845_);
lean_closure_set(v___f_1849_, 1, v___x_1848_);
lean_closure_set(v___f_1849_, 2, v___x_1847_);
lean_closure_set(v___f_1849_, 3, v___x_1802_);
lean_closure_set(v___f_1849_, 4, v_00_u03b1_1789_);
lean_closure_set(v___f_1849_, 5, v___y_1834_);
lean_closure_set(v___f_1849_, 6, v___y_1835_);
lean_closure_set(v___f_1849_, 7, v_e_1794_);
v___x_1850_ = 0;
v___x_1851_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___f_1849_, v___x_1850_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_);
if (lean_obj_tag(v___x_1851_) == 0)
{
lean_object* v_a_1852_; lean_object* v___x_1854_; uint8_t v_isShared_1855_; uint8_t v_isSharedCheck_1958_; 
v_a_1852_ = lean_ctor_get(v___x_1851_, 0);
v_isSharedCheck_1958_ = !lean_is_exclusive(v___x_1851_);
if (v_isSharedCheck_1958_ == 0)
{
v___x_1854_ = v___x_1851_;
v_isShared_1855_ = v_isSharedCheck_1958_;
goto v_resetjp_1853_;
}
else
{
lean_inc(v_a_1852_);
lean_dec(v___x_1851_);
v___x_1854_ = lean_box(0);
v_isShared_1855_ = v_isSharedCheck_1958_;
goto v_resetjp_1853_;
}
v_resetjp_1853_:
{
lean_object* v_snd_1856_; uint8_t v___x_1857_; 
v_snd_1856_ = lean_ctor_get(v_a_1852_, 1);
v___x_1857_ = lean_unbox(v_snd_1856_);
if (v___x_1857_ == 0)
{
lean_object* v___x_1859_; 
lean_dec(v_a_1852_);
lean_dec(v_rn_1840_);
lean_dec_ref(v___y_1839_);
lean_dec_ref(v___y_1838_);
lean_dec_ref(v___y_1837_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
if (v_isShared_1855_ == 0)
{
lean_ctor_set(v___x_1854_, 0, v___y_1836_);
v___x_1859_ = v___x_1854_;
goto v_reusejp_1858_;
}
else
{
lean_object* v_reuseFailAlloc_1860_; 
v_reuseFailAlloc_1860_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1860_, 0, v___y_1836_);
v___x_1859_ = v_reuseFailAlloc_1860_;
goto v_reusejp_1858_;
}
v_reusejp_1858_:
{
return v___x_1859_;
}
}
else
{
lean_object* v_fst_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; 
lean_del_object(v___x_1854_);
lean_dec_ref(v___y_1836_);
v_fst_1861_ = lean_ctor_get(v_a_1852_, 0);
lean_inc(v_fst_1861_);
lean_dec(v_a_1852_);
lean_inc(v_rn_1840_);
v___x_1862_ = l_Lean_mkRawNatLit(v_rn_1840_);
lean_inc_ref(v_amwo_1832_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_1863_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_1862_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_);
if (lean_obj_tag(v___x_1863_) == 0)
{
lean_object* v_a_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; 
v_a_1864_ = lean_ctor_get(v___x_1863_, 0);
lean_inc(v_a_1864_);
lean_dec_ref_known(v___x_1863_, 1);
v___x_1865_ = lean_nat_div(v_v_1791_, v_rn_1840_);
lean_dec(v_rn_1840_);
lean_dec(v_v_1791_);
v___x_1866_ = l_Lean_mkRawNatLit(v___x_1865_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_1867_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_1866_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_);
if (lean_obj_tag(v___x_1867_) == 0)
{
lean_object* v_a_1868_; lean_object* v_fst_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; 
v_a_1868_ = lean_ctor_get(v___x_1867_, 0);
lean_inc(v_a_1868_);
lean_dec_ref_known(v___x_1867_, 1);
v_fst_1869_ = lean_ctor_get(v_a_1864_, 0);
lean_inc_n(v_fst_1869_, 2);
lean_dec(v_a_1864_);
v___x_1870_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__17));
v___x_1871_ = l_Lean_Expr_const___override(v___x_1870_, v___x_1822_);
lean_inc_ref_n(v_00_u03b1_1789_, 5);
v___x_1872_ = l_Lean_Expr_app___override(v___x_1871_, v_00_u03b1_1789_);
v___x_1873_ = l_Lean_Expr_app___override(v___x_1872_, v_fst_1869_);
v___x_1874_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20));
lean_inc_ref_n(v___x_1802_, 4);
v___x_1875_ = l_Lean_Expr_const___override(v___x_1874_, v___x_1802_);
v___x_1876_ = l_Lean_Expr_app___override(v___x_1875_, v_00_u03b1_1789_);
v___x_1877_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22);
v___x_1878_ = l_Lean_Expr_app___override(v___x_1876_, v___x_1877_);
v___x_1879_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25));
v___x_1880_ = l_Lean_Expr_const___override(v___x_1879_, v___x_1802_);
v___x_1881_ = l_Lean_Expr_app___override(v___x_1880_, v_00_u03b1_1789_);
v___x_1882_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28));
v___x_1883_ = l_Lean_Expr_const___override(v___x_1882_, v___x_1802_);
v___x_1884_ = l_Lean_Expr_app___override(v___x_1883_, v_00_u03b1_1789_);
v___x_1885_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__30));
v___x_1886_ = l_Lean_Expr_const___override(v___x_1885_, v___x_1802_);
v___x_1887_ = l_Lean_Expr_app___override(v___x_1886_, v_00_u03b1_1789_);
v___x_1888_ = l_Lean_Expr_app___override(v___x_1887_, v___y_1839_);
v___x_1889_ = l_Lean_Expr_app___override(v___x_1884_, v___x_1888_);
v___x_1890_ = l_Lean_Expr_app___override(v___x_1881_, v___x_1889_);
v___x_1891_ = l_Lean_Expr_app___override(v___x_1878_, v___x_1890_);
v___x_1892_ = l_Lean_Expr_app___override(v___x_1873_, v___x_1891_);
v___x_1893_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v___x_1892_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_);
if (lean_obj_tag(v___x_1893_) == 0)
{
lean_object* v_a_1894_; lean_object* v_fst_1895_; lean_object* v___x_1897_; uint8_t v_isShared_1898_; uint8_t v_isSharedCheck_1932_; 
v_a_1894_ = lean_ctor_get(v___x_1893_, 0);
lean_inc(v_a_1894_);
lean_dec_ref_known(v___x_1893_, 1);
v_fst_1895_ = lean_ctor_get(v_a_1868_, 0);
v_isSharedCheck_1932_ = !lean_is_exclusive(v_a_1868_);
if (v_isSharedCheck_1932_ == 0)
{
lean_object* v_unused_1933_; 
v_unused_1933_ = lean_ctor_get(v_a_1868_, 1);
lean_dec(v_unused_1933_);
v___x_1897_ = v_a_1868_;
v_isShared_1898_ = v_isSharedCheck_1932_;
goto v_resetjp_1896_;
}
else
{
lean_inc(v_fst_1895_);
lean_dec(v_a_1868_);
v___x_1897_ = lean_box(0);
v_isShared_1898_ = v_isSharedCheck_1932_;
goto v_resetjp_1896_;
}
v_resetjp_1896_:
{
lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; 
lean_inc(v_fst_1895_);
v___x_1899_ = l_Lean_Expr_app___override(v___y_1837_, v_fst_1895_);
v___x_1900_ = l_Lean_Expr_app___override(v___x_1899_, v_fst_1869_);
v___x_1901_ = l_Lean_Expr_app___override(v___y_1838_, v___x_1900_);
lean_inc_ref(v_v_x27_1792_);
v___x_1902_ = l_Lean_Expr_app___override(v___x_1901_, v_v_x27_1792_);
v___x_1903_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v___x_1902_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_);
if (lean_obj_tag(v___x_1903_) == 0)
{
lean_object* v_a_1904_; lean_object* v___x_1906_; uint8_t v_isShared_1907_; uint8_t v_isSharedCheck_1923_; 
v_a_1904_ = lean_ctor_get(v___x_1903_, 0);
v_isSharedCheck_1923_ = !lean_is_exclusive(v___x_1903_);
if (v_isSharedCheck_1923_ == 0)
{
v___x_1906_ = v___x_1903_;
v_isShared_1907_ = v_isSharedCheck_1923_;
goto v_resetjp_1905_;
}
else
{
lean_inc(v_a_1904_);
lean_dec(v___x_1903_);
v___x_1906_ = lean_box(0);
v_isShared_1907_ = v_isSharedCheck_1923_;
goto v_resetjp_1905_;
}
v_resetjp_1905_:
{
lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1918_; 
v___x_1908_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__32));
v___x_1909_ = l_Lean_Expr_const___override(v___x_1908_, v___x_1802_);
v___x_1910_ = l_Lean_Expr_app___override(v___x_1909_, v_00_u03b1_1789_);
v___x_1911_ = l_Lean_Expr_app___override(v___x_1910_, v_s_u03b1_1790_);
lean_inc(v_fst_1895_);
v___x_1912_ = l_Lean_Expr_app___override(v___x_1911_, v_fst_1895_);
v___x_1913_ = l_Lean_Expr_app___override(v___x_1912_, v_v_x27_1792_);
v___x_1914_ = l_Lean_Expr_app___override(v___x_1913_, v_fst_1861_);
v___x_1915_ = l_Lean_Expr_app___override(v___x_1914_, v_a_1894_);
v___x_1916_ = l_Lean_Expr_app___override(v___x_1915_, v_a_1904_);
if (v_isShared_1898_ == 0)
{
lean_ctor_set(v___x_1897_, 1, v___x_1916_);
v___x_1918_ = v___x_1897_;
goto v_reusejp_1917_;
}
else
{
lean_object* v_reuseFailAlloc_1922_; 
v_reuseFailAlloc_1922_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1922_, 0, v_fst_1895_);
lean_ctor_set(v_reuseFailAlloc_1922_, 1, v___x_1916_);
v___x_1918_ = v_reuseFailAlloc_1922_;
goto v_reusejp_1917_;
}
v_reusejp_1917_:
{
lean_object* v___x_1920_; 
if (v_isShared_1907_ == 0)
{
lean_ctor_set(v___x_1906_, 0, v___x_1918_);
v___x_1920_ = v___x_1906_;
goto v_reusejp_1919_;
}
else
{
lean_object* v_reuseFailAlloc_1921_; 
v_reuseFailAlloc_1921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1921_, 0, v___x_1918_);
v___x_1920_ = v_reuseFailAlloc_1921_;
goto v_reusejp_1919_;
}
v_reusejp_1919_:
{
return v___x_1920_;
}
}
}
}
else
{
lean_object* v_a_1924_; lean_object* v___x_1926_; uint8_t v_isShared_1927_; uint8_t v_isSharedCheck_1931_; 
lean_del_object(v___x_1897_);
lean_dec(v_fst_1895_);
lean_dec(v_a_1894_);
lean_dec(v_fst_1861_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
v_a_1924_ = lean_ctor_get(v___x_1903_, 0);
v_isSharedCheck_1931_ = !lean_is_exclusive(v___x_1903_);
if (v_isSharedCheck_1931_ == 0)
{
v___x_1926_ = v___x_1903_;
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
else
{
lean_inc(v_a_1924_);
lean_dec(v___x_1903_);
v___x_1926_ = lean_box(0);
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
v_resetjp_1925_:
{
lean_object* v___x_1929_; 
if (v_isShared_1927_ == 0)
{
v___x_1929_ = v___x_1926_;
goto v_reusejp_1928_;
}
else
{
lean_object* v_reuseFailAlloc_1930_; 
v_reuseFailAlloc_1930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1930_, 0, v_a_1924_);
v___x_1929_ = v_reuseFailAlloc_1930_;
goto v_reusejp_1928_;
}
v_reusejp_1928_:
{
return v___x_1929_;
}
}
}
}
}
else
{
lean_object* v_a_1934_; lean_object* v___x_1936_; uint8_t v_isShared_1937_; uint8_t v_isSharedCheck_1941_; 
lean_dec(v_fst_1869_);
lean_dec(v_a_1868_);
lean_dec(v_fst_1861_);
lean_dec_ref(v___y_1838_);
lean_dec_ref(v___y_1837_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
v_a_1934_ = lean_ctor_get(v___x_1893_, 0);
v_isSharedCheck_1941_ = !lean_is_exclusive(v___x_1893_);
if (v_isSharedCheck_1941_ == 0)
{
v___x_1936_ = v___x_1893_;
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
else
{
lean_inc(v_a_1934_);
lean_dec(v___x_1893_);
v___x_1936_ = lean_box(0);
v_isShared_1937_ = v_isSharedCheck_1941_;
goto v_resetjp_1935_;
}
v_resetjp_1935_:
{
lean_object* v___x_1939_; 
if (v_isShared_1937_ == 0)
{
v___x_1939_ = v___x_1936_;
goto v_reusejp_1938_;
}
else
{
lean_object* v_reuseFailAlloc_1940_; 
v_reuseFailAlloc_1940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1940_, 0, v_a_1934_);
v___x_1939_ = v_reuseFailAlloc_1940_;
goto v_reusejp_1938_;
}
v_reusejp_1938_:
{
return v___x_1939_;
}
}
}
}
else
{
lean_object* v_a_1942_; lean_object* v___x_1944_; uint8_t v_isShared_1945_; uint8_t v_isSharedCheck_1949_; 
lean_dec(v_a_1864_);
lean_dec(v_fst_1861_);
lean_dec_ref(v___y_1839_);
lean_dec_ref(v___y_1838_);
lean_dec_ref(v___y_1837_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
v_a_1942_ = lean_ctor_get(v___x_1867_, 0);
v_isSharedCheck_1949_ = !lean_is_exclusive(v___x_1867_);
if (v_isSharedCheck_1949_ == 0)
{
v___x_1944_ = v___x_1867_;
v_isShared_1945_ = v_isSharedCheck_1949_;
goto v_resetjp_1943_;
}
else
{
lean_inc(v_a_1942_);
lean_dec(v___x_1867_);
v___x_1944_ = lean_box(0);
v_isShared_1945_ = v_isSharedCheck_1949_;
goto v_resetjp_1943_;
}
v_resetjp_1943_:
{
lean_object* v___x_1947_; 
if (v_isShared_1945_ == 0)
{
v___x_1947_ = v___x_1944_;
goto v_reusejp_1946_;
}
else
{
lean_object* v_reuseFailAlloc_1948_; 
v_reuseFailAlloc_1948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1948_, 0, v_a_1942_);
v___x_1947_ = v_reuseFailAlloc_1948_;
goto v_reusejp_1946_;
}
v_reusejp_1946_:
{
return v___x_1947_;
}
}
}
}
else
{
lean_object* v_a_1950_; lean_object* v___x_1952_; uint8_t v_isShared_1953_; uint8_t v_isSharedCheck_1957_; 
lean_dec(v_fst_1861_);
lean_dec(v_rn_1840_);
lean_dec_ref(v___y_1839_);
lean_dec_ref(v___y_1838_);
lean_dec_ref(v___y_1837_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_1950_ = lean_ctor_get(v___x_1863_, 0);
v_isSharedCheck_1957_ = !lean_is_exclusive(v___x_1863_);
if (v_isSharedCheck_1957_ == 0)
{
v___x_1952_ = v___x_1863_;
v_isShared_1953_ = v_isSharedCheck_1957_;
goto v_resetjp_1951_;
}
else
{
lean_inc(v_a_1950_);
lean_dec(v___x_1863_);
v___x_1952_ = lean_box(0);
v_isShared_1953_ = v_isSharedCheck_1957_;
goto v_resetjp_1951_;
}
v_resetjp_1951_:
{
lean_object* v___x_1955_; 
if (v_isShared_1953_ == 0)
{
v___x_1955_ = v___x_1952_;
goto v_reusejp_1954_;
}
else
{
lean_object* v_reuseFailAlloc_1956_; 
v_reuseFailAlloc_1956_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1956_, 0, v_a_1950_);
v___x_1955_ = v_reuseFailAlloc_1956_;
goto v_reusejp_1954_;
}
v_reusejp_1954_:
{
return v___x_1955_;
}
}
}
}
}
}
else
{
lean_object* v_a_1959_; lean_object* v___x_1961_; uint8_t v_isShared_1962_; uint8_t v_isSharedCheck_1966_; 
lean_dec(v_rn_1840_);
lean_dec_ref(v___y_1839_);
lean_dec_ref(v___y_1838_);
lean_dec_ref(v___y_1837_);
lean_dec_ref(v___y_1836_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_1959_ = lean_ctor_get(v___x_1851_, 0);
v_isSharedCheck_1966_ = !lean_is_exclusive(v___x_1851_);
if (v_isSharedCheck_1966_ == 0)
{
v___x_1961_ = v___x_1851_;
v_isShared_1962_ = v_isSharedCheck_1966_;
goto v_resetjp_1960_;
}
else
{
lean_inc(v_a_1959_);
lean_dec(v___x_1851_);
v___x_1961_ = lean_box(0);
v_isShared_1962_ = v_isSharedCheck_1966_;
goto v_resetjp_1960_;
}
v_resetjp_1960_:
{
lean_object* v___x_1964_; 
if (v_isShared_1962_ == 0)
{
v___x_1964_ = v___x_1961_;
goto v_reusejp_1963_;
}
else
{
lean_object* v_reuseFailAlloc_1965_; 
v_reuseFailAlloc_1965_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1965_, 0, v_a_1959_);
v___x_1964_ = v_reuseFailAlloc_1965_;
goto v_reusejp_1963_;
}
v_reusejp_1963_:
{
return v___x_1964_;
}
}
}
}
v___jp_1967_:
{
lean_object* v___x_1982_; uint8_t v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___f_1986_; uint8_t v___x_1987_; lean_object* v___x_1988_; 
lean_inc_ref_n(v_00_u03b1_1789_, 2);
v___x_1982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1982_, 0, v_00_u03b1_1789_);
v___x_1983_ = 0;
v___x_1984_ = lean_box(0);
v___x_1985_ = lean_box(v___x_1983_);
lean_inc_ref(v_e_1794_);
lean_inc(v_u_1788_);
lean_inc_ref(v___x_1802_);
v___f_1986_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___boxed), 14, 9);
lean_closure_set(v___f_1986_, 0, v___x_1982_);
lean_closure_set(v___f_1986_, 1, v___x_1985_);
lean_closure_set(v___f_1986_, 2, v___x_1984_);
lean_closure_set(v___f_1986_, 3, v___x_1801_);
lean_closure_set(v___f_1986_, 4, v___x_1802_);
lean_closure_set(v___f_1986_, 5, v_u_1788_);
lean_closure_set(v___f_1986_, 6, v_00_u03b1_1789_);
lean_closure_set(v___f_1986_, 7, v___y_1969_);
lean_closure_set(v___f_1986_, 8, v_e_1794_);
v___x_1987_ = 0;
v___x_1988_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___f_1986_, v___x_1987_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_);
if (lean_obj_tag(v___x_1988_) == 0)
{
lean_object* v_a_1989_; lean_object* v___x_1991_; uint8_t v_isShared_1992_; uint8_t v_isSharedCheck_2139_; 
v_a_1989_ = lean_ctor_get(v___x_1988_, 0);
v_isSharedCheck_2139_ = !lean_is_exclusive(v___x_1988_);
if (v_isSharedCheck_2139_ == 0)
{
v___x_1991_ = v___x_1988_;
v_isShared_1992_ = v_isSharedCheck_2139_;
goto v_resetjp_1990_;
}
else
{
lean_inc(v_a_1989_);
lean_dec(v___x_1988_);
v___x_1991_ = lean_box(0);
v_isShared_1992_ = v_isSharedCheck_2139_;
goto v_resetjp_1990_;
}
v_resetjp_1990_:
{
lean_object* v_snd_1993_; lean_object* v_snd_1994_; uint8_t v___x_1995_; 
v_snd_1993_ = lean_ctor_get(v_a_1989_, 1);
lean_inc(v_snd_1993_);
v_snd_1994_ = lean_ctor_get(v_snd_1993_, 1);
v___x_1995_ = lean_unbox(v_snd_1994_);
if (v___x_1995_ == 0)
{
lean_dec(v_snd_1993_);
lean_dec(v_a_1989_);
lean_dec(v_k2_1977_);
lean_dec(v_k1_1976_);
lean_dec(v_lhs_1975_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_1996_; 
v_left_1996_ = lean_ctor_get(v_t_1793_, 1);
if (lean_obj_tag(v_left_1996_) == 0)
{
lean_object* v_right_1997_; 
v_right_1997_ = lean_ctor_get(v_t_1793_, 2);
lean_inc(v_right_1997_);
lean_dec_ref_known(v_t_1793_, 3);
if (lean_obj_tag(v_right_1997_) == 1)
{
lean_object* v_value_1998_; 
lean_del_object(v___x_1991_);
v_value_1998_ = lean_ctor_get(v_right_1997_, 0);
lean_inc(v_value_1998_);
lean_dec_ref_known(v_right_1997_, 3);
v___y_1834_ = v___y_1968_;
v___y_1835_ = v___y_1970_;
v___y_1836_ = v___y_1971_;
v___y_1837_ = v___y_1972_;
v___y_1838_ = v___y_1973_;
v___y_1839_ = v___y_1974_;
v_rn_1840_ = v_value_1998_;
v___y_1841_ = v___y_1978_;
v___y_1842_ = v___y_1979_;
v___y_1843_ = v___y_1980_;
v___y_1844_ = v___y_1981_;
goto v___jp_1833_;
}
else
{
lean_object* v___x_2000_; 
lean_dec(v_right_1997_);
lean_dec_ref(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref(v___y_1970_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
if (v_isShared_1992_ == 0)
{
lean_ctor_set(v___x_1991_, 0, v___y_1971_);
v___x_2000_ = v___x_1991_;
goto v_reusejp_1999_;
}
else
{
lean_object* v_reuseFailAlloc_2001_; 
v_reuseFailAlloc_2001_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2001_, 0, v___y_1971_);
v___x_2000_ = v_reuseFailAlloc_2001_;
goto v_reusejp_1999_;
}
v_reusejp_1999_:
{
return v___x_2000_;
}
}
}
else
{
lean_object* v___x_2003_; 
lean_dec_ref_known(v_t_1793_, 3);
lean_dec_ref(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref(v___y_1970_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
if (v_isShared_1992_ == 0)
{
lean_ctor_set(v___x_1991_, 0, v___y_1971_);
v___x_2003_ = v___x_1991_;
goto v_reusejp_2002_;
}
else
{
lean_object* v_reuseFailAlloc_2004_; 
v_reuseFailAlloc_2004_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2004_, 0, v___y_1971_);
v___x_2003_ = v_reuseFailAlloc_2004_;
goto v_reusejp_2002_;
}
v_reusejp_2002_:
{
return v___x_2003_;
}
}
}
else
{
lean_object* v___x_2006_; 
lean_dec_ref(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref(v___y_1970_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
if (v_isShared_1992_ == 0)
{
lean_ctor_set(v___x_1991_, 0, v___y_1971_);
v___x_2006_ = v___x_1991_;
goto v_reusejp_2005_;
}
else
{
lean_object* v_reuseFailAlloc_2007_; 
v_reuseFailAlloc_2007_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2007_, 0, v___y_1971_);
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
lean_object* v_fst_2008_; lean_object* v_fst_2009_; lean_object* v___x_2011_; uint8_t v_isShared_2012_; uint8_t v_isSharedCheck_2137_; 
lean_del_object(v___x_1991_);
lean_dec_ref(v___y_1971_);
lean_dec_ref(v___y_1970_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
v_fst_2008_ = lean_ctor_get(v_a_1989_, 0);
lean_inc(v_fst_2008_);
lean_dec(v_a_1989_);
v_fst_2009_ = lean_ctor_get(v_snd_1993_, 0);
v_isSharedCheck_2137_ = !lean_is_exclusive(v_snd_1993_);
if (v_isSharedCheck_2137_ == 0)
{
lean_object* v_unused_2138_; 
v_unused_2138_ = lean_ctor_get(v_snd_1993_, 1);
lean_dec(v_unused_2138_);
v___x_2011_ = v_snd_1993_;
v_isShared_2012_ = v_isSharedCheck_2137_;
goto v_resetjp_2010_;
}
else
{
lean_inc(v_fst_2009_);
lean_dec(v_snd_1993_);
v___x_2011_ = lean_box(0);
v_isShared_2012_ = v_isSharedCheck_2137_;
goto v_resetjp_2010_;
}
v_resetjp_2010_:
{
lean_object* v___x_2013_; lean_object* v___x_2014_; 
lean_inc(v_k1_1976_);
v___x_2013_ = l_Lean_mkRawNatLit(v_k1_1976_);
lean_inc_ref(v_amwo_1832_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2014_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_2013_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_);
if (lean_obj_tag(v___x_2014_) == 0)
{
lean_object* v_a_2015_; lean_object* v_fst_2016_; lean_object* v___x_2018_; uint8_t v_isShared_2019_; uint8_t v_isSharedCheck_2127_; 
v_a_2015_ = lean_ctor_get(v___x_2014_, 0);
lean_inc(v_a_2015_);
lean_dec_ref_known(v___x_2014_, 1);
v_fst_2016_ = lean_ctor_get(v_a_2015_, 0);
v_isSharedCheck_2127_ = !lean_is_exclusive(v_a_2015_);
if (v_isSharedCheck_2127_ == 0)
{
lean_object* v_unused_2128_; 
v_unused_2128_ = lean_ctor_get(v_a_2015_, 1);
lean_dec(v_unused_2128_);
v___x_2018_ = v_a_2015_;
v_isShared_2019_ = v_isSharedCheck_2127_;
goto v_resetjp_2017_;
}
else
{
lean_inc(v_fst_2016_);
lean_dec(v_a_2015_);
v___x_2018_ = lean_box(0);
v_isShared_2019_ = v_isSharedCheck_2127_;
goto v_resetjp_2017_;
}
v_resetjp_2017_:
{
lean_object* v___x_2020_; 
lean_inc(v_fst_2008_);
lean_inc(v_fst_2016_);
lean_inc(v_k1_1976_);
lean_inc_ref(v_s_u03b1_1790_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2020_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v_k1_1976_, v_fst_2016_, v_lhs_1975_, v_fst_2008_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_);
if (lean_obj_tag(v___x_2020_) == 0)
{
lean_object* v_a_2021_; lean_object* v_cancelled_2022_; lean_object* v_pf_2023_; lean_object* v___x_2025_; uint8_t v_isShared_2026_; uint8_t v_isSharedCheck_2126_; 
v_a_2021_ = lean_ctor_get(v___x_2020_, 0);
lean_inc(v_a_2021_);
lean_dec_ref_known(v___x_2020_, 1);
v_cancelled_2022_ = lean_ctor_get(v_a_2021_, 0);
v_pf_2023_ = lean_ctor_get(v_a_2021_, 1);
v_isSharedCheck_2126_ = !lean_is_exclusive(v_a_2021_);
if (v_isSharedCheck_2126_ == 0)
{
v___x_2025_ = v_a_2021_;
v_isShared_2026_ = v_isSharedCheck_2126_;
goto v_resetjp_2024_;
}
else
{
lean_inc(v_pf_2023_);
lean_inc(v_cancelled_2022_);
lean_dec(v_a_2021_);
v___x_2025_ = lean_box(0);
v_isShared_2026_ = v_isSharedCheck_2126_;
goto v_resetjp_2024_;
}
v_resetjp_2024_:
{
lean_object* v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; 
v___x_2027_ = lean_nat_pow(v_k1_1976_, v_k2_1977_);
lean_dec(v_k2_1977_);
lean_dec(v_k1_1976_);
v___x_2028_ = lean_nat_div(v_v_1791_, v___x_2027_);
lean_dec(v___x_2027_);
lean_dec(v_v_1791_);
v___x_2029_ = l_Lean_mkRawNatLit(v___x_2028_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2030_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_2029_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_);
if (lean_obj_tag(v___x_2030_) == 0)
{
lean_object* v_a_2031_; lean_object* v_fst_2032_; lean_object* v___x_2034_; uint8_t v_isShared_2035_; uint8_t v_isSharedCheck_2116_; 
v_a_2031_ = lean_ctor_get(v___x_2030_, 0);
lean_inc(v_a_2031_);
lean_dec_ref_known(v___x_2030_, 1);
v_fst_2032_ = lean_ctor_get(v_a_2031_, 0);
v_isSharedCheck_2116_ = !lean_is_exclusive(v_a_2031_);
if (v_isSharedCheck_2116_ == 0)
{
lean_object* v_unused_2117_; 
v_unused_2117_ = lean_ctor_get(v_a_2031_, 1);
lean_dec(v_unused_2117_);
v___x_2034_ = v_a_2031_;
v_isShared_2035_ = v_isSharedCheck_2116_;
goto v_resetjp_2033_;
}
else
{
lean_inc(v_fst_2032_);
lean_dec(v_a_2031_);
v___x_2034_ = lean_box(0);
v_isShared_2035_ = v_isSharedCheck_2116_;
goto v_resetjp_2033_;
}
v_resetjp_2033_:
{
lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2040_; 
v___x_2036_ = lean_box(0);
lean_inc(v_fst_2032_);
v___x_2037_ = l_Lean_Expr_app___override(v___y_1972_, v_fst_2032_);
v___x_2038_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__2));
lean_inc_ref(v___x_1802_);
if (v_isShared_2035_ == 0)
{
lean_ctor_set_tag(v___x_2034_, 1);
lean_ctor_set(v___x_2034_, 1, v___x_1802_);
lean_ctor_set(v___x_2034_, 0, v___x_2036_);
v___x_2040_ = v___x_2034_;
goto v_reusejp_2039_;
}
else
{
lean_object* v_reuseFailAlloc_2115_; 
v_reuseFailAlloc_2115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2115_, 0, v___x_2036_);
lean_ctor_set(v_reuseFailAlloc_2115_, 1, v___x_1802_);
v___x_2040_ = v_reuseFailAlloc_2115_;
goto v_reusejp_2039_;
}
v_reusejp_2039_:
{
lean_object* v___x_2042_; 
lean_inc(v_u_1788_);
if (v_isShared_2019_ == 0)
{
lean_ctor_set_tag(v___x_2018_, 1);
lean_ctor_set(v___x_2018_, 1, v___x_2040_);
lean_ctor_set(v___x_2018_, 0, v_u_1788_);
v___x_2042_ = v___x_2018_;
goto v_reusejp_2041_;
}
else
{
lean_object* v_reuseFailAlloc_2114_; 
v_reuseFailAlloc_2114_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2114_, 0, v_u_1788_);
lean_ctor_set(v_reuseFailAlloc_2114_, 1, v___x_2040_);
v___x_2042_ = v_reuseFailAlloc_2114_;
goto v_reusejp_2041_;
}
v_reusejp_2041_:
{
lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; lean_object* v___x_2051_; 
v___x_2043_ = l_Lean_Expr_const___override(v___x_2038_, v___x_2042_);
lean_inc_ref_n(v_00_u03b1_1789_, 2);
v___x_2044_ = l_Lean_Expr_app___override(v___x_2043_, v_00_u03b1_1789_);
v___x_2045_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__33, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__33);
v___x_2046_ = l_Lean_Expr_app___override(v___x_2044_, v___x_2045_);
v___x_2047_ = l_Lean_Expr_app___override(v___x_2046_, v_00_u03b1_1789_);
v___x_2048_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__4));
v___x_2049_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__34));
if (v_isShared_2012_ == 0)
{
lean_ctor_set_tag(v___x_2011_, 1);
lean_ctor_set(v___x_2011_, 1, v___x_2049_);
lean_ctor_set(v___x_2011_, 0, v_u_1788_);
v___x_2051_ = v___x_2011_;
goto v_reusejp_2050_;
}
else
{
lean_object* v_reuseFailAlloc_2113_; 
v_reuseFailAlloc_2113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2113_, 0, v_u_1788_);
lean_ctor_set(v_reuseFailAlloc_2113_, 1, v___x_2049_);
v___x_2051_ = v_reuseFailAlloc_2113_;
goto v_reusejp_2050_;
}
v_reusejp_2050_:
{
lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; lean_object* v___x_2074_; 
v___x_2052_ = l_Lean_Expr_const___override(v___x_2048_, v___x_2051_);
lean_inc_ref_n(v_00_u03b1_1789_, 4);
v___x_2053_ = l_Lean_Expr_app___override(v___x_2052_, v_00_u03b1_1789_);
v___x_2054_ = l_Lean_Expr_app___override(v___x_2053_, v___x_2045_);
v___x_2055_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__7));
lean_inc_ref_n(v___x_1802_, 3);
v___x_2056_ = l_Lean_Expr_const___override(v___x_2055_, v___x_1802_);
v___x_2057_ = l_Lean_Expr_app___override(v___x_2056_, v_00_u03b1_1789_);
v___x_2058_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__10));
v___x_2059_ = l_Lean_Expr_const___override(v___x_2058_, v___x_1802_);
v___x_2060_ = l_Lean_Expr_app___override(v___x_2059_, v_00_u03b1_1789_);
v___x_2061_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__1___closed__13));
v___x_2062_ = l_Lean_Expr_const___override(v___x_2061_, v___x_1802_);
v___x_2063_ = l_Lean_Expr_app___override(v___x_2062_, v_00_u03b1_1789_);
v___x_2064_ = l_Lean_Expr_app___override(v___x_2063_, v___y_1974_);
v___x_2065_ = l_Lean_Expr_app___override(v___x_2060_, v___x_2064_);
v___x_2066_ = l_Lean_Expr_app___override(v___x_2057_, v___x_2065_);
v___x_2067_ = l_Lean_Expr_app___override(v___x_2054_, v___x_2066_);
v___x_2068_ = l_Lean_Expr_app___override(v___x_2047_, v___x_2067_);
lean_inc(v_fst_2016_);
lean_inc_ref(v___x_2068_);
v___x_2069_ = l_Lean_Expr_app___override(v___x_2068_, v_fst_2016_);
lean_inc(v_fst_2009_);
v___x_2070_ = l_Lean_Expr_app___override(v___x_2069_, v_fst_2009_);
lean_inc_ref(v___x_2037_);
v___x_2071_ = l_Lean_Expr_app___override(v___x_2037_, v___x_2070_);
v___x_2072_ = l_Lean_Expr_app___override(v___y_1973_, v___x_2071_);
lean_inc_ref(v_v_x27_1792_);
v___x_2073_ = l_Lean_Expr_app___override(v___x_2072_, v_v_x27_1792_);
v___x_2074_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v___x_2073_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_);
if (lean_obj_tag(v___x_2074_) == 0)
{
lean_object* v_a_2075_; lean_object* v___x_2077_; uint8_t v_isShared_2078_; uint8_t v_isSharedCheck_2104_; 
v_a_2075_ = lean_ctor_get(v___x_2074_, 0);
v_isSharedCheck_2104_ = !lean_is_exclusive(v___x_2074_);
if (v_isSharedCheck_2104_ == 0)
{
v___x_2077_ = v___x_2074_;
v_isShared_2078_ = v_isSharedCheck_2104_;
goto v_resetjp_2076_;
}
else
{
lean_inc(v_a_2075_);
lean_dec(v___x_2074_);
v___x_2077_ = lean_box(0);
v_isShared_2078_ = v_isSharedCheck_2104_;
goto v_resetjp_2076_;
}
v_resetjp_2076_:
{
lean_object* v___x_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2099_; 
lean_inc_ref(v_cancelled_2022_);
v___x_2079_ = l_Lean_Expr_app___override(v___x_2068_, v_cancelled_2022_);
lean_inc(v_fst_2009_);
v___x_2080_ = l_Lean_Expr_app___override(v___x_2079_, v_fst_2009_);
v___x_2081_ = l_Lean_Expr_app___override(v___x_2037_, v___x_2080_);
v___x_2082_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__36));
lean_inc_ref(v___x_1802_);
v___x_2083_ = l_Lean_Expr_const___override(v___x_2082_, v___x_1802_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2084_ = l_Lean_Expr_app___override(v___x_2083_, v_00_u03b1_1789_);
v___x_2085_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__38));
v___x_2086_ = l_Lean_Expr_const___override(v___x_2085_, v___x_1802_);
v___x_2087_ = l_Lean_Expr_app___override(v___x_2086_, v_00_u03b1_1789_);
v___x_2088_ = l_Lean_Expr_app___override(v___x_2087_, v_s_u03b1_1790_);
v___x_2089_ = l_Lean_Expr_app___override(v___x_2084_, v___x_2088_);
v___x_2090_ = l_Lean_Expr_app___override(v___x_2089_, v_fst_2016_);
v___x_2091_ = l_Lean_Expr_app___override(v___x_2090_, v_fst_2008_);
v___x_2092_ = l_Lean_Expr_app___override(v___x_2091_, v_cancelled_2022_);
v___x_2093_ = l_Lean_Expr_app___override(v___x_2092_, v_v_x27_1792_);
v___x_2094_ = l_Lean_Expr_app___override(v___x_2093_, v_fst_2032_);
v___x_2095_ = l_Lean_Expr_app___override(v___x_2094_, v_fst_2009_);
v___x_2096_ = l_Lean_Expr_app___override(v___x_2095_, v_pf_2023_);
v___x_2097_ = l_Lean_Expr_app___override(v___x_2096_, v_a_2075_);
if (v_isShared_2026_ == 0)
{
lean_ctor_set(v___x_2025_, 1, v___x_2097_);
lean_ctor_set(v___x_2025_, 0, v___x_2081_);
v___x_2099_ = v___x_2025_;
goto v_reusejp_2098_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v___x_2081_);
lean_ctor_set(v_reuseFailAlloc_2103_, 1, v___x_2097_);
v___x_2099_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2098_;
}
v_reusejp_2098_:
{
lean_object* v___x_2101_; 
if (v_isShared_2078_ == 0)
{
lean_ctor_set(v___x_2077_, 0, v___x_2099_);
v___x_2101_ = v___x_2077_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2102_; 
v_reuseFailAlloc_2102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2102_, 0, v___x_2099_);
v___x_2101_ = v_reuseFailAlloc_2102_;
goto v_reusejp_2100_;
}
v_reusejp_2100_:
{
return v___x_2101_;
}
}
}
}
else
{
lean_object* v_a_2105_; lean_object* v___x_2107_; uint8_t v_isShared_2108_; uint8_t v_isSharedCheck_2112_; 
lean_dec_ref(v___x_2068_);
lean_dec_ref(v___x_2037_);
lean_dec(v_fst_2032_);
lean_del_object(v___x_2025_);
lean_dec_ref(v_pf_2023_);
lean_dec_ref(v_cancelled_2022_);
lean_dec(v_fst_2016_);
lean_dec(v_fst_2009_);
lean_dec(v_fst_2008_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
v_a_2105_ = lean_ctor_get(v___x_2074_, 0);
v_isSharedCheck_2112_ = !lean_is_exclusive(v___x_2074_);
if (v_isSharedCheck_2112_ == 0)
{
v___x_2107_ = v___x_2074_;
v_isShared_2108_ = v_isSharedCheck_2112_;
goto v_resetjp_2106_;
}
else
{
lean_inc(v_a_2105_);
lean_dec(v___x_2074_);
v___x_2107_ = lean_box(0);
v_isShared_2108_ = v_isSharedCheck_2112_;
goto v_resetjp_2106_;
}
v_resetjp_2106_:
{
lean_object* v___x_2110_; 
if (v_isShared_2108_ == 0)
{
v___x_2110_ = v___x_2107_;
goto v_reusejp_2109_;
}
else
{
lean_object* v_reuseFailAlloc_2111_; 
v_reuseFailAlloc_2111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2111_, 0, v_a_2105_);
v___x_2110_ = v_reuseFailAlloc_2111_;
goto v_reusejp_2109_;
}
v_reusejp_2109_:
{
return v___x_2110_;
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
lean_object* v_a_2118_; lean_object* v___x_2120_; uint8_t v_isShared_2121_; uint8_t v_isSharedCheck_2125_; 
lean_del_object(v___x_2025_);
lean_dec_ref(v_pf_2023_);
lean_dec_ref(v_cancelled_2022_);
lean_del_object(v___x_2018_);
lean_dec(v_fst_2016_);
lean_del_object(v___x_2011_);
lean_dec(v_fst_2009_);
lean_dec(v_fst_2008_);
lean_dec_ref(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2118_ = lean_ctor_get(v___x_2030_, 0);
v_isSharedCheck_2125_ = !lean_is_exclusive(v___x_2030_);
if (v_isSharedCheck_2125_ == 0)
{
v___x_2120_ = v___x_2030_;
v_isShared_2121_ = v_isSharedCheck_2125_;
goto v_resetjp_2119_;
}
else
{
lean_inc(v_a_2118_);
lean_dec(v___x_2030_);
v___x_2120_ = lean_box(0);
v_isShared_2121_ = v_isSharedCheck_2125_;
goto v_resetjp_2119_;
}
v_resetjp_2119_:
{
lean_object* v___x_2123_; 
if (v_isShared_2121_ == 0)
{
v___x_2123_ = v___x_2120_;
goto v_reusejp_2122_;
}
else
{
lean_object* v_reuseFailAlloc_2124_; 
v_reuseFailAlloc_2124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2124_, 0, v_a_2118_);
v___x_2123_ = v_reuseFailAlloc_2124_;
goto v_reusejp_2122_;
}
v_reusejp_2122_:
{
return v___x_2123_;
}
}
}
}
}
else
{
lean_del_object(v___x_2018_);
lean_dec(v_fst_2016_);
lean_del_object(v___x_2011_);
lean_dec(v_fst_2009_);
lean_dec(v_fst_2008_);
lean_dec(v_k2_1977_);
lean_dec(v_k1_1976_);
lean_dec_ref(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
return v___x_2020_;
}
}
}
else
{
lean_object* v_a_2129_; lean_object* v___x_2131_; uint8_t v_isShared_2132_; uint8_t v_isSharedCheck_2136_; 
lean_del_object(v___x_2011_);
lean_dec(v_fst_2009_);
lean_dec(v_fst_2008_);
lean_dec(v_k2_1977_);
lean_dec(v_k1_1976_);
lean_dec(v_lhs_1975_);
lean_dec_ref(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2129_ = lean_ctor_get(v___x_2014_, 0);
v_isSharedCheck_2136_ = !lean_is_exclusive(v___x_2014_);
if (v_isSharedCheck_2136_ == 0)
{
v___x_2131_ = v___x_2014_;
v_isShared_2132_ = v_isSharedCheck_2136_;
goto v_resetjp_2130_;
}
else
{
lean_inc(v_a_2129_);
lean_dec(v___x_2014_);
v___x_2131_ = lean_box(0);
v_isShared_2132_ = v_isSharedCheck_2136_;
goto v_resetjp_2130_;
}
v_resetjp_2130_:
{
lean_object* v___x_2134_; 
if (v_isShared_2132_ == 0)
{
v___x_2134_ = v___x_2131_;
goto v_reusejp_2133_;
}
else
{
lean_object* v_reuseFailAlloc_2135_; 
v_reuseFailAlloc_2135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2135_, 0, v_a_2129_);
v___x_2134_ = v_reuseFailAlloc_2135_;
goto v_reusejp_2133_;
}
v_reusejp_2133_:
{
return v___x_2134_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2140_; lean_object* v___x_2142_; uint8_t v_isShared_2143_; uint8_t v_isSharedCheck_2147_; 
lean_dec(v_k2_1977_);
lean_dec(v_k1_1976_);
lean_dec(v_lhs_1975_);
lean_dec_ref(v___y_1974_);
lean_dec_ref(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec_ref(v___y_1971_);
lean_dec_ref(v___y_1970_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2140_ = lean_ctor_get(v___x_1988_, 0);
v_isSharedCheck_2147_ = !lean_is_exclusive(v___x_1988_);
if (v_isSharedCheck_2147_ == 0)
{
v___x_2142_ = v___x_1988_;
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
else
{
lean_inc(v_a_2140_);
lean_dec(v___x_1988_);
v___x_2142_ = lean_box(0);
v_isShared_2143_ = v_isSharedCheck_2147_;
goto v_resetjp_2141_;
}
v_resetjp_2141_:
{
lean_object* v___x_2145_; 
if (v_isShared_2143_ == 0)
{
v___x_2145_ = v___x_2142_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2146_; 
v_reuseFailAlloc_2146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2146_, 0, v_a_2140_);
v___x_2145_ = v_reuseFailAlloc_2146_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
return v___x_2145_;
}
}
}
}
v___jp_2148_:
{
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_2159_; 
v_left_2159_ = lean_ctor_get(v_t_1793_, 1);
if (lean_obj_tag(v_left_2159_) == 0)
{
lean_object* v_right_2160_; 
v_right_2160_ = lean_ctor_get(v_t_1793_, 2);
lean_inc(v_right_2160_);
lean_dec_ref_known(v_t_1793_, 3);
if (lean_obj_tag(v_right_2160_) == 1)
{
lean_object* v_value_2161_; 
v_value_2161_ = lean_ctor_get(v_right_2160_, 0);
lean_inc(v_value_2161_);
lean_dec_ref_known(v_right_2160_, 3);
v___y_1834_ = v___y_2149_;
v___y_1835_ = v___y_2150_;
v___y_1836_ = v___y_2151_;
v___y_1837_ = v___y_2152_;
v___y_1838_ = v___y_2153_;
v___y_1839_ = v___y_2154_;
v_rn_1840_ = v_value_2161_;
v___y_1841_ = v___y_2155_;
v___y_1842_ = v___y_2156_;
v___y_1843_ = v___y_2157_;
v___y_1844_ = v___y_2158_;
goto v___jp_1833_;
}
else
{
lean_object* v___x_2162_; 
lean_dec(v_right_2160_);
lean_dec_ref(v___y_2154_);
lean_dec_ref(v___y_2153_);
lean_dec_ref(v___y_2152_);
lean_dec_ref(v___y_2150_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v___x_2162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2162_, 0, v___y_2151_);
return v___x_2162_;
}
}
else
{
lean_object* v___x_2163_; 
lean_dec_ref_known(v_t_1793_, 3);
lean_dec_ref(v___y_2154_);
lean_dec_ref(v___y_2153_);
lean_dec_ref(v___y_2152_);
lean_dec_ref(v___y_2150_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v___x_2163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2163_, 0, v___y_2151_);
return v___x_2163_;
}
}
else
{
lean_object* v___x_2164_; 
lean_dec_ref(v___y_2154_);
lean_dec_ref(v___y_2153_);
lean_dec_ref(v___y_2152_);
lean_dec_ref(v___y_2150_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v___x_2164_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2164_, 0, v___y_2151_);
return v___x_2164_;
}
}
v___jp_2165_:
{
lean_object* v___x_2183_; 
v___x_2183_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___y_2175_, v___y_2171_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2183_) == 0)
{
lean_object* v_a_2184_; lean_object* v_snd_2185_; lean_object* v_snd_2186_; uint8_t v___x_2187_; 
v_a_2184_ = lean_ctor_get(v___x_2183_, 0);
lean_inc(v_a_2184_);
lean_dec_ref_known(v___x_2183_, 1);
v_snd_2185_ = lean_ctor_get(v_a_2184_, 1);
lean_inc(v_snd_2185_);
v_snd_2186_ = lean_ctor_get(v_snd_2185_, 1);
v___x_2187_ = lean_unbox(v_snd_2186_);
if (v___x_2187_ == 0)
{
lean_object* v___x_2188_; 
lean_dec(v_snd_2185_);
lean_dec(v_a_2184_);
lean_dec(v_rn_2178_);
lean_dec(v_lhs_2177_);
lean_dec(v___y_2172_);
lean_dec_ref(v___x_1828_);
v___x_2188_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___y_2176_, v___y_2171_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2188_) == 0)
{
lean_object* v_a_2189_; lean_object* v_snd_2190_; uint8_t v___x_2191_; 
v_a_2189_ = lean_ctor_get(v___x_2188_, 0);
lean_inc(v_a_2189_);
lean_dec_ref_known(v___x_2188_, 1);
v_snd_2190_ = lean_ctor_get(v_a_2189_, 1);
v___x_2191_ = lean_unbox(v_snd_2190_);
if (v___x_2191_ == 0)
{
lean_dec(v_a_2189_);
lean_dec_ref(v___x_1829_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_2192_; 
v_left_2192_ = lean_ctor_get(v_t_1793_, 1);
if (lean_obj_tag(v_left_2192_) == 1)
{
lean_object* v_right_2193_; 
v_right_2193_ = lean_ctor_get(v_t_1793_, 2);
if (lean_obj_tag(v_right_2193_) == 1)
{
lean_object* v_left_2194_; 
v_left_2194_ = lean_ctor_get(v_right_2193_, 1);
if (lean_obj_tag(v_left_2194_) == 0)
{
lean_object* v_right_2195_; 
v_right_2195_ = lean_ctor_get(v_right_2193_, 2);
if (lean_obj_tag(v_right_2195_) == 0)
{
lean_object* v_value_2196_; lean_object* v_value_2197_; 
v_value_2196_ = lean_ctor_get(v_left_2192_, 0);
v_value_2197_ = lean_ctor_get(v_right_2193_, 0);
lean_inc(v_value_2197_);
lean_inc(v_value_2196_);
lean_inc_ref(v_left_2192_);
v___y_1968_ = v___y_2166_;
v___y_1969_ = v___y_2167_;
v___y_1970_ = v___y_2168_;
v___y_1971_ = v___y_2169_;
v___y_1972_ = v___y_2170_;
v___y_1973_ = v___y_2173_;
v___y_1974_ = v___y_2174_;
v_lhs_1975_ = v_left_2192_;
v_k1_1976_ = v_value_2196_;
v_k2_1977_ = v_value_2197_;
v___y_1978_ = v___y_2179_;
v___y_1979_ = v___y_2180_;
v___y_1980_ = v___y_2181_;
v___y_1981_ = v___y_2182_;
goto v___jp_1967_;
}
else
{
lean_dec_ref(v___y_2167_);
v___y_2149_ = v___y_2166_;
v___y_2150_ = v___y_2168_;
v___y_2151_ = v___y_2169_;
v___y_2152_ = v___y_2170_;
v___y_2153_ = v___y_2173_;
v___y_2154_ = v___y_2174_;
v___y_2155_ = v___y_2179_;
v___y_2156_ = v___y_2180_;
v___y_2157_ = v___y_2181_;
v___y_2158_ = v___y_2182_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2167_);
v___y_2149_ = v___y_2166_;
v___y_2150_ = v___y_2168_;
v___y_2151_ = v___y_2169_;
v___y_2152_ = v___y_2170_;
v___y_2153_ = v___y_2173_;
v___y_2154_ = v___y_2174_;
v___y_2155_ = v___y_2179_;
v___y_2156_ = v___y_2180_;
v___y_2157_ = v___y_2181_;
v___y_2158_ = v___y_2182_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2167_);
v___y_2149_ = v___y_2166_;
v___y_2150_ = v___y_2168_;
v___y_2151_ = v___y_2169_;
v___y_2152_ = v___y_2170_;
v___y_2153_ = v___y_2173_;
v___y_2154_ = v___y_2174_;
v___y_2155_ = v___y_2179_;
v___y_2156_ = v___y_2180_;
v___y_2157_ = v___y_2181_;
v___y_2158_ = v___y_2182_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2167_);
v___y_2149_ = v___y_2166_;
v___y_2150_ = v___y_2168_;
v___y_2151_ = v___y_2169_;
v___y_2152_ = v___y_2170_;
v___y_2153_ = v___y_2173_;
v___y_2154_ = v___y_2174_;
v___y_2155_ = v___y_2179_;
v___y_2156_ = v___y_2180_;
v___y_2157_ = v___y_2181_;
v___y_2158_ = v___y_2182_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2167_);
v___y_2149_ = v___y_2166_;
v___y_2150_ = v___y_2168_;
v___y_2151_ = v___y_2169_;
v___y_2152_ = v___y_2170_;
v___y_2153_ = v___y_2173_;
v___y_2154_ = v___y_2174_;
v___y_2155_ = v___y_2179_;
v___y_2156_ = v___y_2180_;
v___y_2157_ = v___y_2181_;
v___y_2158_ = v___y_2182_;
goto v___jp_2148_;
}
}
else
{
lean_object* v_fst_2198_; lean_object* v___x_2199_; 
lean_dec_ref(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec_ref(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec_ref(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref(v_e_1794_);
v_fst_2198_ = lean_ctor_get(v_a_2189_, 0);
lean_inc_n(v_fst_2198_, 2);
lean_dec(v_a_2189_);
lean_inc_ref(v_v_x27_1792_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2199_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v_v_1791_, v_v_x27_1792_, v_t_1793_, v_fst_2198_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2199_) == 0)
{
lean_object* v_a_2200_; lean_object* v___x_2202_; uint8_t v_isShared_2203_; uint8_t v_isSharedCheck_2253_; 
v_a_2200_ = lean_ctor_get(v___x_2199_, 0);
v_isSharedCheck_2253_ = !lean_is_exclusive(v___x_2199_);
if (v_isSharedCheck_2253_ == 0)
{
v___x_2202_ = v___x_2199_;
v_isShared_2203_ = v_isSharedCheck_2253_;
goto v_resetjp_2201_;
}
else
{
lean_inc(v_a_2200_);
lean_dec(v___x_2199_);
v___x_2202_ = lean_box(0);
v_isShared_2203_ = v_isSharedCheck_2253_;
goto v_resetjp_2201_;
}
v_resetjp_2201_:
{
lean_object* v_cancelled_2204_; lean_object* v_pf_2205_; lean_object* v___x_2207_; uint8_t v_isShared_2208_; uint8_t v_isSharedCheck_2252_; 
v_cancelled_2204_ = lean_ctor_get(v_a_2200_, 0);
v_pf_2205_ = lean_ctor_get(v_a_2200_, 1);
v_isSharedCheck_2252_ = !lean_is_exclusive(v_a_2200_);
if (v_isSharedCheck_2252_ == 0)
{
v___x_2207_ = v_a_2200_;
v_isShared_2208_ = v_isSharedCheck_2252_;
goto v_resetjp_2206_;
}
else
{
lean_inc(v_pf_2205_);
lean_inc(v_cancelled_2204_);
lean_dec(v_a_2200_);
v___x_2207_ = lean_box(0);
v_isShared_2208_ = v_isSharedCheck_2252_;
goto v_resetjp_2206_;
}
v_resetjp_2206_:
{
lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; lean_object* v___x_2229_; lean_object* v___x_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2247_; 
v___x_2209_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0));
lean_inc_ref_n(v___x_1802_, 7);
v___x_2210_ = l_Lean_Expr_const___override(v___x_2209_, v___x_1802_);
lean_inc_ref_n(v_00_u03b1_1789_, 7);
v___x_2211_ = l_Lean_Expr_app___override(v___x_2210_, v_00_u03b1_1789_);
v___x_2212_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3));
v___x_2213_ = l_Lean_Expr_const___override(v___x_2212_, v___x_1802_);
v___x_2214_ = l_Lean_Expr_app___override(v___x_2213_, v_00_u03b1_1789_);
v___x_2215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6));
v___x_2216_ = l_Lean_Expr_const___override(v___x_2215_, v___x_1802_);
v___x_2217_ = l_Lean_Expr_app___override(v___x_2216_, v_00_u03b1_1789_);
v___x_2218_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9));
v___x_2219_ = l_Lean_Expr_const___override(v___x_2218_, v___x_1802_);
v___x_2220_ = l_Lean_Expr_app___override(v___x_2219_, v_00_u03b1_1789_);
v___x_2221_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12));
v___x_2222_ = l_Lean_Expr_const___override(v___x_2221_, v___x_1802_);
v___x_2223_ = l_Lean_Expr_app___override(v___x_2222_, v_00_u03b1_1789_);
v___x_2224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15));
v___x_2225_ = l_Lean_Expr_const___override(v___x_2224_, v___x_1802_);
v___x_2226_ = l_Lean_Expr_app___override(v___x_2225_, v_00_u03b1_1789_);
v___x_2227_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__39));
v___x_2228_ = l_Lean_Expr_const___override(v___x_2227_, v___x_1802_);
v___x_2229_ = l_Lean_Expr_app___override(v___x_2228_, v_00_u03b1_1789_);
lean_inc_ref(v___x_1829_);
v___x_2230_ = l_Lean_Expr_app___override(v___x_2229_, v___x_1829_);
v___x_2231_ = l_Lean_Expr_app___override(v___x_2226_, v___x_2230_);
v___x_2232_ = l_Lean_Expr_app___override(v___x_2223_, v___x_2231_);
v___x_2233_ = l_Lean_Expr_app___override(v___x_2220_, v___x_2232_);
v___x_2234_ = l_Lean_Expr_app___override(v___x_2217_, v___x_2233_);
v___x_2235_ = l_Lean_Expr_app___override(v___x_2214_, v___x_2234_);
v___x_2236_ = l_Lean_Expr_app___override(v___x_2211_, v___x_2235_);
lean_inc_ref(v_cancelled_2204_);
v___x_2237_ = l_Lean_Expr_app___override(v___x_2236_, v_cancelled_2204_);
v___x_2238_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41));
v___x_2239_ = l_Lean_Expr_const___override(v___x_2238_, v___x_1802_);
v___x_2240_ = l_Lean_Expr_app___override(v___x_2239_, v_00_u03b1_1789_);
v___x_2241_ = l_Lean_Expr_app___override(v___x_2240_, v___x_1829_);
v___x_2242_ = l_Lean_Expr_app___override(v___x_2241_, v_v_x27_1792_);
v___x_2243_ = l_Lean_Expr_app___override(v___x_2242_, v_fst_2198_);
v___x_2244_ = l_Lean_Expr_app___override(v___x_2243_, v_cancelled_2204_);
v___x_2245_ = l_Lean_Expr_app___override(v___x_2244_, v_pf_2205_);
if (v_isShared_2208_ == 0)
{
lean_ctor_set(v___x_2207_, 1, v___x_2245_);
lean_ctor_set(v___x_2207_, 0, v___x_2237_);
v___x_2247_ = v___x_2207_;
goto v_reusejp_2246_;
}
else
{
lean_object* v_reuseFailAlloc_2251_; 
v_reuseFailAlloc_2251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2251_, 0, v___x_2237_);
lean_ctor_set(v_reuseFailAlloc_2251_, 1, v___x_2245_);
v___x_2247_ = v_reuseFailAlloc_2251_;
goto v_reusejp_2246_;
}
v_reusejp_2246_:
{
lean_object* v___x_2249_; 
if (v_isShared_2203_ == 0)
{
lean_ctor_set(v___x_2202_, 0, v___x_2247_);
v___x_2249_ = v___x_2202_;
goto v_reusejp_2248_;
}
else
{
lean_object* v_reuseFailAlloc_2250_; 
v_reuseFailAlloc_2250_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2250_, 0, v___x_2247_);
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
else
{
lean_dec(v_fst_2198_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_00_u03b1_1789_);
return v___x_2199_;
}
}
}
else
{
lean_object* v_a_2254_; lean_object* v___x_2256_; uint8_t v_isShared_2257_; uint8_t v_isSharedCheck_2261_; 
lean_dec_ref(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec_ref(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec_ref(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2254_ = lean_ctor_get(v___x_2188_, 0);
v_isSharedCheck_2261_ = !lean_is_exclusive(v___x_2188_);
if (v_isSharedCheck_2261_ == 0)
{
v___x_2256_ = v___x_2188_;
v_isShared_2257_ = v_isSharedCheck_2261_;
goto v_resetjp_2255_;
}
else
{
lean_inc(v_a_2254_);
lean_dec(v___x_2188_);
v___x_2256_ = lean_box(0);
v_isShared_2257_ = v_isSharedCheck_2261_;
goto v_resetjp_2255_;
}
v_resetjp_2255_:
{
lean_object* v___x_2259_; 
if (v_isShared_2257_ == 0)
{
v___x_2259_ = v___x_2256_;
goto v_reusejp_2258_;
}
else
{
lean_object* v_reuseFailAlloc_2260_; 
v_reuseFailAlloc_2260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2260_, 0, v_a_2254_);
v___x_2259_ = v_reuseFailAlloc_2260_;
goto v_reusejp_2258_;
}
v_reusejp_2258_:
{
return v___x_2259_;
}
}
}
}
else
{
lean_object* v_fst_2262_; lean_object* v_fst_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; 
lean_dec_ref(v___y_2176_);
lean_dec_ref(v___y_2174_);
lean_dec_ref(v___y_2169_);
lean_dec_ref(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
v_fst_2262_ = lean_ctor_get(v_a_2184_, 0);
lean_inc(v_fst_2262_);
lean_dec(v_a_2184_);
v_fst_2263_ = lean_ctor_get(v_snd_2185_, 0);
lean_inc(v_fst_2263_);
lean_dec(v_snd_2185_);
lean_inc(v_rn_2178_);
v___x_2264_ = l_Lean_mkRawNatLit(v_rn_2178_);
lean_inc_ref(v_amwo_1832_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2265_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_2264_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2265_) == 0)
{
lean_object* v_a_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; 
v_a_2266_ = lean_ctor_get(v___x_2265_, 0);
lean_inc(v_a_2266_);
lean_dec_ref_known(v___x_2265_, 1);
v___x_2267_ = lean_nat_div(v_v_1791_, v_rn_2178_);
lean_dec(v_rn_2178_);
lean_dec(v_v_1791_);
lean_inc(v___x_2267_);
v___x_2268_ = l_Lean_mkRawNatLit(v___x_2267_);
lean_inc_ref(v_amwo_1832_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2269_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_2268_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2269_) == 0)
{
lean_object* v_a_2270_; lean_object* v_fst_2271_; lean_object* v___x_2272_; 
v_a_2270_ = lean_ctor_get(v___x_2269_, 0);
lean_inc(v_a_2270_);
lean_dec_ref_known(v___x_2269_, 1);
v_fst_2271_ = lean_ctor_get(v_a_2270_, 0);
lean_inc_n(v_fst_2271_, 2);
lean_dec(v_a_2270_);
lean_inc(v_fst_2262_);
lean_inc_ref(v_s_u03b1_1790_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2272_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v___x_2267_, v_fst_2271_, v_lhs_2177_, v_fst_2262_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2272_) == 0)
{
lean_object* v_a_2273_; lean_object* v_cancelled_2274_; lean_object* v_pf_2275_; lean_object* v___x_2277_; uint8_t v_isShared_2278_; uint8_t v_isSharedCheck_2363_; 
v_a_2273_ = lean_ctor_get(v___x_2272_, 0);
lean_inc(v_a_2273_);
lean_dec_ref_known(v___x_2272_, 1);
v_cancelled_2274_ = lean_ctor_get(v_a_2273_, 0);
v_pf_2275_ = lean_ctor_get(v_a_2273_, 1);
v_isSharedCheck_2363_ = !lean_is_exclusive(v_a_2273_);
if (v_isSharedCheck_2363_ == 0)
{
v___x_2277_ = v_a_2273_;
v_isShared_2278_ = v_isSharedCheck_2363_;
goto v_resetjp_2276_;
}
else
{
lean_inc(v_pf_2275_);
lean_inc(v_cancelled_2274_);
lean_dec(v_a_2273_);
v___x_2277_ = lean_box(0);
v_isShared_2278_ = v_isSharedCheck_2363_;
goto v_resetjp_2276_;
}
v_resetjp_2276_:
{
lean_object* v_fst_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; 
v_fst_2279_ = lean_ctor_get(v_a_2266_, 0);
lean_inc_n(v_fst_2279_, 2);
lean_dec(v_a_2266_);
v___x_2280_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__0));
v___x_2281_ = l_Lean_Expr_const___override(v___x_2280_, v___y_2172_);
lean_inc_ref_n(v_00_u03b1_1789_, 9);
v___x_2282_ = l_Lean_Expr_app___override(v___x_2281_, v_00_u03b1_1789_);
v___x_2283_ = l_Lean_Expr_app___override(v___x_2282_, v_00_u03b1_1789_);
v___x_2284_ = l_Lean_Expr_app___override(v___x_2283_, v_00_u03b1_1789_);
v___x_2285_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__2));
lean_inc_ref_n(v___x_1802_, 6);
v___x_2286_ = l_Lean_Expr_const___override(v___x_2285_, v___x_1802_);
v___x_2287_ = l_Lean_Expr_app___override(v___x_2286_, v_00_u03b1_1789_);
v___x_2288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___closed__5));
v___x_2289_ = l_Lean_Expr_const___override(v___x_2288_, v___x_1802_);
v___x_2290_ = l_Lean_Expr_app___override(v___x_2289_, v_00_u03b1_1789_);
v___x_2291_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__42));
v___x_2292_ = l_Lean_Expr_const___override(v___x_2291_, v___x_1802_);
v___x_2293_ = l_Lean_Expr_app___override(v___x_2292_, v_00_u03b1_1789_);
v___x_2294_ = l_Lean_Expr_app___override(v___x_2293_, v___x_1828_);
v___x_2295_ = l_Lean_Expr_app___override(v___x_2290_, v___x_2294_);
v___x_2296_ = l_Lean_Expr_app___override(v___x_2287_, v___x_2295_);
v___x_2297_ = l_Lean_Expr_app___override(v___x_2284_, v___x_2296_);
v___x_2298_ = l_Lean_Expr_app___override(v___x_2297_, v_fst_2279_);
lean_inc(v_fst_2263_);
v___x_2299_ = l_Lean_Expr_app___override(v___x_2298_, v_fst_2263_);
lean_inc_ref(v___y_2173_);
v___x_2300_ = l_Lean_Expr_app___override(v___y_2173_, v___x_2299_);
v___x_2301_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20));
v___x_2302_ = l_Lean_Expr_const___override(v___x_2301_, v___x_1802_);
v___x_2303_ = l_Lean_Expr_app___override(v___x_2302_, v_00_u03b1_1789_);
v___x_2304_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__44, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__44_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__44);
v___x_2305_ = l_Lean_Expr_app___override(v___x_2303_, v___x_2304_);
v___x_2306_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__47));
v___x_2307_ = l_Lean_Expr_const___override(v___x_2306_, v___x_1802_);
v___x_2308_ = l_Lean_Expr_app___override(v___x_2307_, v_00_u03b1_1789_);
v___x_2309_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__49));
v___x_2310_ = l_Lean_Expr_const___override(v___x_2309_, v___x_1802_);
v___x_2311_ = l_Lean_Expr_app___override(v___x_2310_, v_00_u03b1_1789_);
v___x_2312_ = l_Lean_Expr_app___override(v___x_2311_, v_amwo_1832_);
v___x_2313_ = l_Lean_Expr_app___override(v___x_2308_, v___x_2312_);
v___x_2314_ = l_Lean_Expr_app___override(v___x_2305_, v___x_2313_);
v___x_2315_ = l_Lean_Expr_app___override(v___x_2300_, v___x_2314_);
v___x_2316_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v___x_2315_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2316_) == 0)
{
lean_object* v_a_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2320_; lean_object* v___x_2321_; lean_object* v___x_2322_; 
v_a_2317_ = lean_ctor_get(v___x_2316_, 0);
lean_inc(v_a_2317_);
lean_dec_ref_known(v___x_2316_, 1);
lean_inc(v_fst_2271_);
v___x_2318_ = l_Lean_Expr_app___override(v___y_2170_, v_fst_2271_);
lean_inc(v_fst_2279_);
v___x_2319_ = l_Lean_Expr_app___override(v___x_2318_, v_fst_2279_);
v___x_2320_ = l_Lean_Expr_app___override(v___y_2173_, v___x_2319_);
lean_inc_ref(v_v_x27_1792_);
v___x_2321_ = l_Lean_Expr_app___override(v___x_2320_, v_v_x27_1792_);
v___x_2322_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v___x_2321_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_);
if (lean_obj_tag(v___x_2322_) == 0)
{
lean_object* v_a_2323_; lean_object* v___x_2325_; uint8_t v_isShared_2326_; uint8_t v_isSharedCheck_2346_; 
v_a_2323_ = lean_ctor_get(v___x_2322_, 0);
v_isSharedCheck_2346_ = !lean_is_exclusive(v___x_2322_);
if (v_isSharedCheck_2346_ == 0)
{
v___x_2325_ = v___x_2322_;
v_isShared_2326_ = v_isSharedCheck_2346_;
goto v_resetjp_2324_;
}
else
{
lean_inc(v_a_2323_);
lean_dec(v___x_2322_);
v___x_2325_ = lean_box(0);
v_isShared_2326_ = v_isSharedCheck_2346_;
goto v_resetjp_2324_;
}
v_resetjp_2324_:
{
lean_object* v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; lean_object* v___x_2341_; 
v___x_2327_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__51));
v___x_2328_ = l_Lean_Expr_const___override(v___x_2327_, v___x_1802_);
v___x_2329_ = l_Lean_Expr_app___override(v___x_2328_, v_00_u03b1_1789_);
v___x_2330_ = l_Lean_Expr_app___override(v___x_2329_, v_s_u03b1_1790_);
v___x_2331_ = l_Lean_Expr_app___override(v___x_2330_, v_fst_2271_);
v___x_2332_ = l_Lean_Expr_app___override(v___x_2331_, v_fst_2279_);
v___x_2333_ = l_Lean_Expr_app___override(v___x_2332_, v_v_x27_1792_);
v___x_2334_ = l_Lean_Expr_app___override(v___x_2333_, v_fst_2262_);
v___x_2335_ = l_Lean_Expr_app___override(v___x_2334_, v_fst_2263_);
lean_inc_ref(v_cancelled_2274_);
v___x_2336_ = l_Lean_Expr_app___override(v___x_2335_, v_cancelled_2274_);
v___x_2337_ = l_Lean_Expr_app___override(v___x_2336_, v_pf_2275_);
v___x_2338_ = l_Lean_Expr_app___override(v___x_2337_, v_a_2317_);
v___x_2339_ = l_Lean_Expr_app___override(v___x_2338_, v_a_2323_);
if (v_isShared_2278_ == 0)
{
lean_ctor_set(v___x_2277_, 1, v___x_2339_);
v___x_2341_ = v___x_2277_;
goto v_reusejp_2340_;
}
else
{
lean_object* v_reuseFailAlloc_2345_; 
v_reuseFailAlloc_2345_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2345_, 0, v_cancelled_2274_);
lean_ctor_set(v_reuseFailAlloc_2345_, 1, v___x_2339_);
v___x_2341_ = v_reuseFailAlloc_2345_;
goto v_reusejp_2340_;
}
v_reusejp_2340_:
{
lean_object* v___x_2343_; 
if (v_isShared_2326_ == 0)
{
lean_ctor_set(v___x_2325_, 0, v___x_2341_);
v___x_2343_ = v___x_2325_;
goto v_reusejp_2342_;
}
else
{
lean_object* v_reuseFailAlloc_2344_; 
v_reuseFailAlloc_2344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2344_, 0, v___x_2341_);
v___x_2343_ = v_reuseFailAlloc_2344_;
goto v_reusejp_2342_;
}
v_reusejp_2342_:
{
return v___x_2343_;
}
}
}
}
else
{
lean_object* v_a_2347_; lean_object* v___x_2349_; uint8_t v_isShared_2350_; uint8_t v_isSharedCheck_2354_; 
lean_dec(v_a_2317_);
lean_dec(v_fst_2279_);
lean_del_object(v___x_2277_);
lean_dec_ref(v_pf_2275_);
lean_dec_ref(v_cancelled_2274_);
lean_dec(v_fst_2271_);
lean_dec(v_fst_2263_);
lean_dec(v_fst_2262_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
v_a_2347_ = lean_ctor_get(v___x_2322_, 0);
v_isSharedCheck_2354_ = !lean_is_exclusive(v___x_2322_);
if (v_isSharedCheck_2354_ == 0)
{
v___x_2349_ = v___x_2322_;
v_isShared_2350_ = v_isSharedCheck_2354_;
goto v_resetjp_2348_;
}
else
{
lean_inc(v_a_2347_);
lean_dec(v___x_2322_);
v___x_2349_ = lean_box(0);
v_isShared_2350_ = v_isSharedCheck_2354_;
goto v_resetjp_2348_;
}
v_resetjp_2348_:
{
lean_object* v___x_2352_; 
if (v_isShared_2350_ == 0)
{
v___x_2352_ = v___x_2349_;
goto v_reusejp_2351_;
}
else
{
lean_object* v_reuseFailAlloc_2353_; 
v_reuseFailAlloc_2353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2353_, 0, v_a_2347_);
v___x_2352_ = v_reuseFailAlloc_2353_;
goto v_reusejp_2351_;
}
v_reusejp_2351_:
{
return v___x_2352_;
}
}
}
}
else
{
lean_object* v_a_2355_; lean_object* v___x_2357_; uint8_t v_isShared_2358_; uint8_t v_isSharedCheck_2362_; 
lean_dec(v_fst_2279_);
lean_del_object(v___x_2277_);
lean_dec_ref(v_pf_2275_);
lean_dec_ref(v_cancelled_2274_);
lean_dec(v_fst_2271_);
lean_dec(v_fst_2263_);
lean_dec(v_fst_2262_);
lean_dec_ref(v___y_2173_);
lean_dec_ref(v___y_2170_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
v_a_2355_ = lean_ctor_get(v___x_2316_, 0);
v_isSharedCheck_2362_ = !lean_is_exclusive(v___x_2316_);
if (v_isSharedCheck_2362_ == 0)
{
v___x_2357_ = v___x_2316_;
v_isShared_2358_ = v_isSharedCheck_2362_;
goto v_resetjp_2356_;
}
else
{
lean_inc(v_a_2355_);
lean_dec(v___x_2316_);
v___x_2357_ = lean_box(0);
v_isShared_2358_ = v_isSharedCheck_2362_;
goto v_resetjp_2356_;
}
v_resetjp_2356_:
{
lean_object* v___x_2360_; 
if (v_isShared_2358_ == 0)
{
v___x_2360_ = v___x_2357_;
goto v_reusejp_2359_;
}
else
{
lean_object* v_reuseFailAlloc_2361_; 
v_reuseFailAlloc_2361_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2361_, 0, v_a_2355_);
v___x_2360_ = v_reuseFailAlloc_2361_;
goto v_reusejp_2359_;
}
v_reusejp_2359_:
{
return v___x_2360_;
}
}
}
}
}
else
{
lean_dec(v_fst_2271_);
lean_dec(v_a_2266_);
lean_dec(v_fst_2263_);
lean_dec(v_fst_2262_);
lean_dec_ref(v___y_2173_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2170_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
return v___x_2272_;
}
}
else
{
lean_object* v_a_2364_; lean_object* v___x_2366_; uint8_t v_isShared_2367_; uint8_t v_isSharedCheck_2371_; 
lean_dec(v___x_2267_);
lean_dec(v_a_2266_);
lean_dec(v_fst_2263_);
lean_dec(v_fst_2262_);
lean_dec(v_lhs_2177_);
lean_dec_ref(v___y_2173_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2170_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2364_ = lean_ctor_get(v___x_2269_, 0);
v_isSharedCheck_2371_ = !lean_is_exclusive(v___x_2269_);
if (v_isSharedCheck_2371_ == 0)
{
v___x_2366_ = v___x_2269_;
v_isShared_2367_ = v_isSharedCheck_2371_;
goto v_resetjp_2365_;
}
else
{
lean_inc(v_a_2364_);
lean_dec(v___x_2269_);
v___x_2366_ = lean_box(0);
v_isShared_2367_ = v_isSharedCheck_2371_;
goto v_resetjp_2365_;
}
v_resetjp_2365_:
{
lean_object* v___x_2369_; 
if (v_isShared_2367_ == 0)
{
v___x_2369_ = v___x_2366_;
goto v_reusejp_2368_;
}
else
{
lean_object* v_reuseFailAlloc_2370_; 
v_reuseFailAlloc_2370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2370_, 0, v_a_2364_);
v___x_2369_ = v_reuseFailAlloc_2370_;
goto v_reusejp_2368_;
}
v_reusejp_2368_:
{
return v___x_2369_;
}
}
}
}
else
{
lean_object* v_a_2372_; lean_object* v___x_2374_; uint8_t v_isShared_2375_; uint8_t v_isSharedCheck_2379_; 
lean_dec(v_fst_2263_);
lean_dec(v_fst_2262_);
lean_dec(v_rn_2178_);
lean_dec(v_lhs_2177_);
lean_dec_ref(v___y_2173_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2170_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2372_ = lean_ctor_get(v___x_2265_, 0);
v_isSharedCheck_2379_ = !lean_is_exclusive(v___x_2265_);
if (v_isSharedCheck_2379_ == 0)
{
v___x_2374_ = v___x_2265_;
v_isShared_2375_ = v_isSharedCheck_2379_;
goto v_resetjp_2373_;
}
else
{
lean_inc(v_a_2372_);
lean_dec(v___x_2265_);
v___x_2374_ = lean_box(0);
v_isShared_2375_ = v_isSharedCheck_2379_;
goto v_resetjp_2373_;
}
v_resetjp_2373_:
{
lean_object* v___x_2377_; 
if (v_isShared_2375_ == 0)
{
v___x_2377_ = v___x_2374_;
goto v_reusejp_2376_;
}
else
{
lean_object* v_reuseFailAlloc_2378_; 
v_reuseFailAlloc_2378_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2378_, 0, v_a_2372_);
v___x_2377_ = v_reuseFailAlloc_2378_;
goto v_reusejp_2376_;
}
v_reusejp_2376_:
{
return v___x_2377_;
}
}
}
}
}
else
{
lean_object* v_a_2380_; lean_object* v___x_2382_; uint8_t v_isShared_2383_; uint8_t v_isSharedCheck_2387_; 
lean_dec(v_rn_2178_);
lean_dec(v_lhs_2177_);
lean_dec_ref(v___y_2176_);
lean_dec_ref(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2170_);
lean_dec_ref(v___y_2169_);
lean_dec_ref(v___y_2168_);
lean_dec_ref(v___y_2167_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1829_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2380_ = lean_ctor_get(v___x_2183_, 0);
v_isSharedCheck_2387_ = !lean_is_exclusive(v___x_2183_);
if (v_isSharedCheck_2387_ == 0)
{
v___x_2382_ = v___x_2183_;
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
else
{
lean_inc(v_a_2380_);
lean_dec(v___x_2183_);
v___x_2382_ = lean_box(0);
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
v_resetjp_2381_:
{
lean_object* v___x_2385_; 
if (v_isShared_2383_ == 0)
{
v___x_2385_ = v___x_2382_;
goto v_reusejp_2384_;
}
else
{
lean_object* v_reuseFailAlloc_2386_; 
v_reuseFailAlloc_2386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2386_, 0, v_a_2380_);
v___x_2385_ = v_reuseFailAlloc_2386_;
goto v_reusejp_2384_;
}
v_reusejp_2384_:
{
return v___x_2385_;
}
}
}
}
v___jp_2388_:
{
lean_object* v___x_2402_; 
v___x_2402_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___y_2397_, v___y_2394_, v___y_2398_, v___y_2399_, v___y_2400_, v___y_2401_);
if (lean_obj_tag(v___x_2402_) == 0)
{
lean_object* v_a_2403_; lean_object* v_snd_2404_; uint8_t v___x_2405_; 
v_a_2403_ = lean_ctor_get(v___x_2402_, 0);
lean_inc(v_a_2403_);
lean_dec_ref_known(v___x_2402_, 1);
v_snd_2404_ = lean_ctor_get(v_a_2403_, 1);
v___x_2405_ = lean_unbox(v_snd_2404_);
if (v___x_2405_ == 0)
{
lean_dec(v_a_2403_);
lean_dec_ref(v___x_1829_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_2406_; 
v_left_2406_ = lean_ctor_get(v_t_1793_, 1);
if (lean_obj_tag(v_left_2406_) == 1)
{
lean_object* v_right_2407_; 
v_right_2407_ = lean_ctor_get(v_t_1793_, 2);
if (lean_obj_tag(v_right_2407_) == 1)
{
lean_object* v_left_2408_; 
v_left_2408_ = lean_ctor_get(v_right_2407_, 1);
if (lean_obj_tag(v_left_2408_) == 0)
{
lean_object* v_right_2409_; 
v_right_2409_ = lean_ctor_get(v_right_2407_, 2);
if (lean_obj_tag(v_right_2409_) == 0)
{
lean_object* v_value_2410_; lean_object* v_value_2411_; 
v_value_2410_ = lean_ctor_get(v_left_2406_, 0);
v_value_2411_ = lean_ctor_get(v_right_2407_, 0);
lean_inc(v_value_2411_);
lean_inc(v_value_2410_);
lean_inc_ref(v_left_2406_);
v___y_1968_ = v___y_2389_;
v___y_1969_ = v___y_2390_;
v___y_1970_ = v___y_2391_;
v___y_1971_ = v___y_2392_;
v___y_1972_ = v___y_2393_;
v___y_1973_ = v___y_2395_;
v___y_1974_ = v___y_2396_;
v_lhs_1975_ = v_left_2406_;
v_k1_1976_ = v_value_2410_;
v_k2_1977_ = v_value_2411_;
v___y_1978_ = v___y_2398_;
v___y_1979_ = v___y_2399_;
v___y_1980_ = v___y_2400_;
v___y_1981_ = v___y_2401_;
goto v___jp_1967_;
}
else
{
lean_dec_ref(v___y_2390_);
v___y_2149_ = v___y_2389_;
v___y_2150_ = v___y_2391_;
v___y_2151_ = v___y_2392_;
v___y_2152_ = v___y_2393_;
v___y_2153_ = v___y_2395_;
v___y_2154_ = v___y_2396_;
v___y_2155_ = v___y_2398_;
v___y_2156_ = v___y_2399_;
v___y_2157_ = v___y_2400_;
v___y_2158_ = v___y_2401_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2390_);
v___y_2149_ = v___y_2389_;
v___y_2150_ = v___y_2391_;
v___y_2151_ = v___y_2392_;
v___y_2152_ = v___y_2393_;
v___y_2153_ = v___y_2395_;
v___y_2154_ = v___y_2396_;
v___y_2155_ = v___y_2398_;
v___y_2156_ = v___y_2399_;
v___y_2157_ = v___y_2400_;
v___y_2158_ = v___y_2401_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2390_);
v___y_2149_ = v___y_2389_;
v___y_2150_ = v___y_2391_;
v___y_2151_ = v___y_2392_;
v___y_2152_ = v___y_2393_;
v___y_2153_ = v___y_2395_;
v___y_2154_ = v___y_2396_;
v___y_2155_ = v___y_2398_;
v___y_2156_ = v___y_2399_;
v___y_2157_ = v___y_2400_;
v___y_2158_ = v___y_2401_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2390_);
v___y_2149_ = v___y_2389_;
v___y_2150_ = v___y_2391_;
v___y_2151_ = v___y_2392_;
v___y_2152_ = v___y_2393_;
v___y_2153_ = v___y_2395_;
v___y_2154_ = v___y_2396_;
v___y_2155_ = v___y_2398_;
v___y_2156_ = v___y_2399_;
v___y_2157_ = v___y_2400_;
v___y_2158_ = v___y_2401_;
goto v___jp_2148_;
}
}
else
{
lean_dec_ref(v___y_2390_);
v___y_2149_ = v___y_2389_;
v___y_2150_ = v___y_2391_;
v___y_2151_ = v___y_2392_;
v___y_2152_ = v___y_2393_;
v___y_2153_ = v___y_2395_;
v___y_2154_ = v___y_2396_;
v___y_2155_ = v___y_2398_;
v___y_2156_ = v___y_2399_;
v___y_2157_ = v___y_2400_;
v___y_2158_ = v___y_2401_;
goto v___jp_2148_;
}
}
else
{
lean_object* v_fst_2412_; lean_object* v___x_2413_; 
lean_dec_ref(v___y_2396_);
lean_dec_ref(v___y_2395_);
lean_dec_ref(v___y_2393_);
lean_dec_ref(v___y_2392_);
lean_dec_ref(v___y_2391_);
lean_dec_ref(v___y_2390_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref(v_e_1794_);
v_fst_2412_ = lean_ctor_get(v_a_2403_, 0);
lean_inc_n(v_fst_2412_, 2);
lean_dec(v_a_2403_);
lean_inc_ref(v_v_x27_1792_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2413_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v_v_1791_, v_v_x27_1792_, v_t_1793_, v_fst_2412_, v___y_2398_, v___y_2399_, v___y_2400_, v___y_2401_);
if (lean_obj_tag(v___x_2413_) == 0)
{
lean_object* v_a_2414_; lean_object* v___x_2416_; uint8_t v_isShared_2417_; uint8_t v_isSharedCheck_2467_; 
v_a_2414_ = lean_ctor_get(v___x_2413_, 0);
v_isSharedCheck_2467_ = !lean_is_exclusive(v___x_2413_);
if (v_isSharedCheck_2467_ == 0)
{
v___x_2416_ = v___x_2413_;
v_isShared_2417_ = v_isSharedCheck_2467_;
goto v_resetjp_2415_;
}
else
{
lean_inc(v_a_2414_);
lean_dec(v___x_2413_);
v___x_2416_ = lean_box(0);
v_isShared_2417_ = v_isSharedCheck_2467_;
goto v_resetjp_2415_;
}
v_resetjp_2415_:
{
lean_object* v_cancelled_2418_; lean_object* v_pf_2419_; lean_object* v___x_2421_; uint8_t v_isShared_2422_; uint8_t v_isSharedCheck_2466_; 
v_cancelled_2418_ = lean_ctor_get(v_a_2414_, 0);
v_pf_2419_ = lean_ctor_get(v_a_2414_, 1);
v_isSharedCheck_2466_ = !lean_is_exclusive(v_a_2414_);
if (v_isSharedCheck_2466_ == 0)
{
v___x_2421_ = v_a_2414_;
v_isShared_2422_ = v_isSharedCheck_2466_;
goto v_resetjp_2420_;
}
else
{
lean_inc(v_pf_2419_);
lean_inc(v_cancelled_2418_);
lean_dec(v_a_2414_);
v___x_2421_ = lean_box(0);
v_isShared_2422_ = v_isSharedCheck_2466_;
goto v_resetjp_2420_;
}
v_resetjp_2420_:
{
lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; lean_object* v___x_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v___x_2459_; lean_object* v___x_2461_; 
v___x_2423_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__0));
lean_inc_ref_n(v___x_1802_, 7);
v___x_2424_ = l_Lean_Expr_const___override(v___x_2423_, v___x_1802_);
lean_inc_ref_n(v_00_u03b1_1789_, 7);
v___x_2425_ = l_Lean_Expr_app___override(v___x_2424_, v_00_u03b1_1789_);
v___x_2426_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__3));
v___x_2427_ = l_Lean_Expr_const___override(v___x_2426_, v___x_1802_);
v___x_2428_ = l_Lean_Expr_app___override(v___x_2427_, v_00_u03b1_1789_);
v___x_2429_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__6));
v___x_2430_ = l_Lean_Expr_const___override(v___x_2429_, v___x_1802_);
v___x_2431_ = l_Lean_Expr_app___override(v___x_2430_, v_00_u03b1_1789_);
v___x_2432_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__9));
v___x_2433_ = l_Lean_Expr_const___override(v___x_2432_, v___x_1802_);
v___x_2434_ = l_Lean_Expr_app___override(v___x_2433_, v_00_u03b1_1789_);
v___x_2435_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__12));
v___x_2436_ = l_Lean_Expr_const___override(v___x_2435_, v___x_1802_);
v___x_2437_ = l_Lean_Expr_app___override(v___x_2436_, v_00_u03b1_1789_);
v___x_2438_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___closed__15));
v___x_2439_ = l_Lean_Expr_const___override(v___x_2438_, v___x_1802_);
v___x_2440_ = l_Lean_Expr_app___override(v___x_2439_, v_00_u03b1_1789_);
v___x_2441_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__39));
v___x_2442_ = l_Lean_Expr_const___override(v___x_2441_, v___x_1802_);
v___x_2443_ = l_Lean_Expr_app___override(v___x_2442_, v_00_u03b1_1789_);
lean_inc_ref(v___x_1829_);
v___x_2444_ = l_Lean_Expr_app___override(v___x_2443_, v___x_1829_);
v___x_2445_ = l_Lean_Expr_app___override(v___x_2440_, v___x_2444_);
v___x_2446_ = l_Lean_Expr_app___override(v___x_2437_, v___x_2445_);
v___x_2447_ = l_Lean_Expr_app___override(v___x_2434_, v___x_2446_);
v___x_2448_ = l_Lean_Expr_app___override(v___x_2431_, v___x_2447_);
v___x_2449_ = l_Lean_Expr_app___override(v___x_2428_, v___x_2448_);
v___x_2450_ = l_Lean_Expr_app___override(v___x_2425_, v___x_2449_);
lean_inc_ref(v_cancelled_2418_);
v___x_2451_ = l_Lean_Expr_app___override(v___x_2450_, v_cancelled_2418_);
v___x_2452_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__41));
v___x_2453_ = l_Lean_Expr_const___override(v___x_2452_, v___x_1802_);
v___x_2454_ = l_Lean_Expr_app___override(v___x_2453_, v_00_u03b1_1789_);
v___x_2455_ = l_Lean_Expr_app___override(v___x_2454_, v___x_1829_);
v___x_2456_ = l_Lean_Expr_app___override(v___x_2455_, v_v_x27_1792_);
v___x_2457_ = l_Lean_Expr_app___override(v___x_2456_, v_fst_2412_);
v___x_2458_ = l_Lean_Expr_app___override(v___x_2457_, v_cancelled_2418_);
v___x_2459_ = l_Lean_Expr_app___override(v___x_2458_, v_pf_2419_);
if (v_isShared_2422_ == 0)
{
lean_ctor_set(v___x_2421_, 1, v___x_2459_);
lean_ctor_set(v___x_2421_, 0, v___x_2451_);
v___x_2461_ = v___x_2421_;
goto v_reusejp_2460_;
}
else
{
lean_object* v_reuseFailAlloc_2465_; 
v_reuseFailAlloc_2465_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2465_, 0, v___x_2451_);
lean_ctor_set(v_reuseFailAlloc_2465_, 1, v___x_2459_);
v___x_2461_ = v_reuseFailAlloc_2465_;
goto v_reusejp_2460_;
}
v_reusejp_2460_:
{
lean_object* v___x_2463_; 
if (v_isShared_2417_ == 0)
{
lean_ctor_set(v___x_2416_, 0, v___x_2461_);
v___x_2463_ = v___x_2416_;
goto v_reusejp_2462_;
}
else
{
lean_object* v_reuseFailAlloc_2464_; 
v_reuseFailAlloc_2464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2464_, 0, v___x_2461_);
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
else
{
lean_dec(v_fst_2412_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_00_u03b1_1789_);
return v___x_2413_;
}
}
}
else
{
lean_object* v_a_2468_; lean_object* v___x_2470_; uint8_t v_isShared_2471_; uint8_t v_isSharedCheck_2475_; 
lean_dec_ref(v___y_2396_);
lean_dec_ref(v___y_2395_);
lean_dec_ref(v___y_2393_);
lean_dec_ref(v___y_2392_);
lean_dec_ref(v___y_2391_);
lean_dec_ref(v___y_2390_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2468_ = lean_ctor_get(v___x_2402_, 0);
v_isSharedCheck_2475_ = !lean_is_exclusive(v___x_2402_);
if (v_isSharedCheck_2475_ == 0)
{
v___x_2470_ = v___x_2402_;
v_isShared_2471_ = v_isSharedCheck_2475_;
goto v_resetjp_2469_;
}
else
{
lean_inc(v_a_2468_);
lean_dec(v___x_2402_);
v___x_2470_ = lean_box(0);
v_isShared_2471_ = v_isSharedCheck_2475_;
goto v_resetjp_2469_;
}
v_resetjp_2469_:
{
lean_object* v___x_2473_; 
if (v_isShared_2471_ == 0)
{
v___x_2473_ = v___x_2470_;
goto v_reusejp_2472_;
}
else
{
lean_object* v_reuseFailAlloc_2474_; 
v_reuseFailAlloc_2474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2474_, 0, v_a_2468_);
v___x_2473_ = v_reuseFailAlloc_2474_;
goto v_reusejp_2472_;
}
v_reusejp_2472_:
{
return v___x_2473_;
}
}
}
}
v___jp_2476_:
{
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_right_2492_; 
v_right_2492_ = lean_ctor_get(v_t_1793_, 2);
if (lean_obj_tag(v_right_2492_) == 1)
{
lean_object* v_left_2493_; lean_object* v_value_2494_; 
v_left_2493_ = lean_ctor_get(v_t_1793_, 1);
v_value_2494_ = lean_ctor_get(v_right_2492_, 0);
lean_inc(v_value_2494_);
lean_inc(v_left_2493_);
v___y_2166_ = v___y_2477_;
v___y_2167_ = v___y_2478_;
v___y_2168_ = v___y_2479_;
v___y_2169_ = v___y_2480_;
v___y_2170_ = v___y_2482_;
v___y_2171_ = v___y_2481_;
v___y_2172_ = v___y_2483_;
v___y_2173_ = v___y_2484_;
v___y_2174_ = v___y_2485_;
v___y_2175_ = v___y_2486_;
v___y_2176_ = v___y_2487_;
v_lhs_2177_ = v_left_2493_;
v_rn_2178_ = v_value_2494_;
v___y_2179_ = v___y_2488_;
v___y_2180_ = v___y_2489_;
v___y_2181_ = v___y_2490_;
v___y_2182_ = v___y_2491_;
goto v___jp_2165_;
}
else
{
lean_dec_ref(v___y_2486_);
lean_dec(v___y_2483_);
lean_dec_ref(v___x_1828_);
v___y_2389_ = v___y_2477_;
v___y_2390_ = v___y_2478_;
v___y_2391_ = v___y_2479_;
v___y_2392_ = v___y_2480_;
v___y_2393_ = v___y_2482_;
v___y_2394_ = v___y_2481_;
v___y_2395_ = v___y_2484_;
v___y_2396_ = v___y_2485_;
v___y_2397_ = v___y_2487_;
v___y_2398_ = v___y_2488_;
v___y_2399_ = v___y_2489_;
v___y_2400_ = v___y_2490_;
v___y_2401_ = v___y_2491_;
goto v___jp_2388_;
}
}
else
{
lean_dec_ref(v___y_2486_);
lean_dec(v___y_2483_);
lean_dec_ref(v___x_1828_);
v___y_2389_ = v___y_2477_;
v___y_2390_ = v___y_2478_;
v___y_2391_ = v___y_2479_;
v___y_2392_ = v___y_2480_;
v___y_2393_ = v___y_2482_;
v___y_2394_ = v___y_2481_;
v___y_2395_ = v___y_2484_;
v___y_2396_ = v___y_2485_;
v___y_2397_ = v___y_2487_;
v___y_2398_ = v___y_2488_;
v___y_2399_ = v___y_2489_;
v___y_2400_ = v___y_2490_;
v___y_2401_ = v___y_2491_;
goto v___jp_2388_;
}
}
v___jp_2495_:
{
lean_object* v___x_2507_; lean_object* v___x_2508_; 
lean_inc(v___y_2502_);
v___x_2507_ = l_Lean_mkRawNatLit(v___y_2502_);
lean_inc_ref(v_amwo_1832_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2508_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_2507_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_);
if (lean_obj_tag(v___x_2508_) == 0)
{
lean_object* v_a_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; 
v_a_2509_ = lean_ctor_get(v___x_2508_, 0);
lean_inc(v_a_2509_);
lean_dec_ref_known(v___x_2508_, 1);
v___x_2510_ = lean_nat_div(v_v_1791_, v___y_2502_);
lean_dec(v_v_1791_);
lean_inc(v___x_2510_);
v___x_2511_ = l_Lean_mkRawNatLit(v___x_2510_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2512_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_u_1788_, v_00_u03b1_1789_, v_amwo_1832_, v___x_2511_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_);
if (lean_obj_tag(v___x_2512_) == 0)
{
lean_object* v_a_2513_; lean_object* v_fst_2514_; lean_object* v___x_2515_; 
v_a_2513_ = lean_ctor_get(v___x_2512_, 0);
lean_inc(v_a_2513_);
lean_dec_ref_known(v___x_2512_, 1);
v_fst_2514_ = lean_ctor_get(v_a_2509_, 0);
lean_inc_n(v_fst_2514_, 2);
lean_dec(v_a_2509_);
lean_inc(v___y_2499_);
lean_inc_ref(v_s_u03b1_1790_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2515_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v___y_2502_, v_fst_2514_, v___y_2496_, v___y_2499_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_);
if (lean_obj_tag(v___x_2515_) == 0)
{
lean_object* v_a_2516_; lean_object* v_cancelled_2517_; lean_object* v_pf_2518_; lean_object* v_fst_2519_; lean_object* v___x_2520_; 
v_a_2516_ = lean_ctor_get(v___x_2515_, 0);
lean_inc(v_a_2516_);
lean_dec_ref_known(v___x_2515_, 1);
v_cancelled_2517_ = lean_ctor_get(v_a_2516_, 0);
lean_inc_ref(v_cancelled_2517_);
v_pf_2518_ = lean_ctor_get(v_a_2516_, 1);
lean_inc_ref(v_pf_2518_);
lean_dec(v_a_2516_);
v_fst_2519_ = lean_ctor_get(v_a_2513_, 0);
lean_inc_n(v_fst_2519_, 2);
lean_dec(v_a_2513_);
lean_inc(v___y_2498_);
lean_inc_ref(v_s_u03b1_1790_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2520_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v___x_2510_, v_fst_2519_, v___y_2501_, v___y_2498_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_);
if (lean_obj_tag(v___x_2520_) == 0)
{
lean_object* v_a_2521_; lean_object* v_cancelled_2522_; lean_object* v_pf_2523_; lean_object* v___x_2525_; uint8_t v_isShared_2526_; uint8_t v_isSharedCheck_2571_; 
v_a_2521_ = lean_ctor_get(v___x_2520_, 0);
lean_inc(v_a_2521_);
lean_dec_ref_known(v___x_2520_, 1);
v_cancelled_2522_ = lean_ctor_get(v_a_2521_, 0);
v_pf_2523_ = lean_ctor_get(v_a_2521_, 1);
v_isSharedCheck_2571_ = !lean_is_exclusive(v_a_2521_);
if (v_isSharedCheck_2571_ == 0)
{
v___x_2525_ = v_a_2521_;
v_isShared_2526_ = v_isSharedCheck_2571_;
goto v_resetjp_2524_;
}
else
{
lean_inc(v_pf_2523_);
lean_inc(v_cancelled_2522_);
lean_dec(v_a_2521_);
v___x_2525_ = lean_box(0);
v_isShared_2526_ = v_isSharedCheck_2571_;
goto v_resetjp_2524_;
}
v_resetjp_2524_:
{
lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; 
lean_inc(v_fst_2514_);
lean_inc_ref(v___y_2497_);
v___x_2527_ = l_Lean_Expr_app___override(v___y_2497_, v_fst_2514_);
lean_inc(v_fst_2519_);
v___x_2528_ = l_Lean_Expr_app___override(v___x_2527_, v_fst_2519_);
v___x_2529_ = l_Lean_Expr_app___override(v___y_2500_, v___x_2528_);
lean_inc_ref(v_v_x27_1792_);
v___x_2530_ = l_Lean_Expr_app___override(v___x_2529_, v_v_x27_1792_);
v___x_2531_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v___x_2530_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_);
if (lean_obj_tag(v___x_2531_) == 0)
{
lean_object* v_a_2532_; lean_object* v___x_2534_; uint8_t v_isShared_2535_; uint8_t v_isSharedCheck_2562_; 
v_a_2532_ = lean_ctor_get(v___x_2531_, 0);
v_isSharedCheck_2562_ = !lean_is_exclusive(v___x_2531_);
if (v_isSharedCheck_2562_ == 0)
{
v___x_2534_ = v___x_2531_;
v_isShared_2535_ = v_isSharedCheck_2562_;
goto v_resetjp_2533_;
}
else
{
lean_inc(v_a_2532_);
lean_dec(v___x_2531_);
v___x_2534_ = lean_box(0);
v_isShared_2535_ = v_isSharedCheck_2562_;
goto v_resetjp_2533_;
}
v_resetjp_2533_:
{
lean_object* v___x_2536_; lean_object* v___x_2537_; lean_object* v___x_2538_; lean_object* v___x_2539_; lean_object* v___x_2540_; lean_object* v___x_2541_; lean_object* v___x_2542_; lean_object* v___x_2543_; lean_object* v___x_2544_; lean_object* v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2557_; 
lean_inc_ref(v_cancelled_2517_);
v___x_2536_ = l_Lean_Expr_app___override(v___y_2497_, v_cancelled_2517_);
lean_inc_ref(v_cancelled_2522_);
v___x_2537_ = l_Lean_Expr_app___override(v___x_2536_, v_cancelled_2522_);
v___x_2538_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__53));
lean_inc_ref(v___x_1802_);
v___x_2539_ = l_Lean_Expr_const___override(v___x_2538_, v___x_1802_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2540_ = l_Lean_Expr_app___override(v___x_2539_, v_00_u03b1_1789_);
v___x_2541_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__38));
v___x_2542_ = l_Lean_Expr_const___override(v___x_2541_, v___x_1802_);
v___x_2543_ = l_Lean_Expr_app___override(v___x_2542_, v_00_u03b1_1789_);
v___x_2544_ = l_Lean_Expr_app___override(v___x_2543_, v_s_u03b1_1790_);
v___x_2545_ = l_Lean_Expr_app___override(v___x_2540_, v___x_2544_);
v___x_2546_ = l_Lean_Expr_app___override(v___x_2545_, v_fst_2514_);
v___x_2547_ = l_Lean_Expr_app___override(v___x_2546_, v_fst_2519_);
v___x_2548_ = l_Lean_Expr_app___override(v___x_2547_, v_v_x27_1792_);
v___x_2549_ = l_Lean_Expr_app___override(v___x_2548_, v___y_2499_);
v___x_2550_ = l_Lean_Expr_app___override(v___x_2549_, v___y_2498_);
v___x_2551_ = l_Lean_Expr_app___override(v___x_2550_, v_cancelled_2517_);
v___x_2552_ = l_Lean_Expr_app___override(v___x_2551_, v_cancelled_2522_);
v___x_2553_ = l_Lean_Expr_app___override(v___x_2552_, v_pf_2518_);
v___x_2554_ = l_Lean_Expr_app___override(v___x_2553_, v_pf_2523_);
v___x_2555_ = l_Lean_Expr_app___override(v___x_2554_, v_a_2532_);
if (v_isShared_2526_ == 0)
{
lean_ctor_set(v___x_2525_, 1, v___x_2555_);
lean_ctor_set(v___x_2525_, 0, v___x_2537_);
v___x_2557_ = v___x_2525_;
goto v_reusejp_2556_;
}
else
{
lean_object* v_reuseFailAlloc_2561_; 
v_reuseFailAlloc_2561_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2561_, 0, v___x_2537_);
lean_ctor_set(v_reuseFailAlloc_2561_, 1, v___x_2555_);
v___x_2557_ = v_reuseFailAlloc_2561_;
goto v_reusejp_2556_;
}
v_reusejp_2556_:
{
lean_object* v___x_2559_; 
if (v_isShared_2535_ == 0)
{
lean_ctor_set(v___x_2534_, 0, v___x_2557_);
v___x_2559_ = v___x_2534_;
goto v_reusejp_2558_;
}
else
{
lean_object* v_reuseFailAlloc_2560_; 
v_reuseFailAlloc_2560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2560_, 0, v___x_2557_);
v___x_2559_ = v_reuseFailAlloc_2560_;
goto v_reusejp_2558_;
}
v_reusejp_2558_:
{
return v___x_2559_;
}
}
}
}
else
{
lean_object* v_a_2563_; lean_object* v___x_2565_; uint8_t v_isShared_2566_; uint8_t v_isSharedCheck_2570_; 
lean_del_object(v___x_2525_);
lean_dec_ref(v_pf_2523_);
lean_dec_ref(v_cancelled_2522_);
lean_dec(v_fst_2519_);
lean_dec_ref(v_pf_2518_);
lean_dec_ref(v_cancelled_2517_);
lean_dec(v_fst_2514_);
lean_dec(v___y_2499_);
lean_dec(v___y_2498_);
lean_dec_ref(v___y_2497_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
v_a_2563_ = lean_ctor_get(v___x_2531_, 0);
v_isSharedCheck_2570_ = !lean_is_exclusive(v___x_2531_);
if (v_isSharedCheck_2570_ == 0)
{
v___x_2565_ = v___x_2531_;
v_isShared_2566_ = v_isSharedCheck_2570_;
goto v_resetjp_2564_;
}
else
{
lean_inc(v_a_2563_);
lean_dec(v___x_2531_);
v___x_2565_ = lean_box(0);
v_isShared_2566_ = v_isSharedCheck_2570_;
goto v_resetjp_2564_;
}
v_resetjp_2564_:
{
lean_object* v___x_2568_; 
if (v_isShared_2566_ == 0)
{
v___x_2568_ = v___x_2565_;
goto v_reusejp_2567_;
}
else
{
lean_object* v_reuseFailAlloc_2569_; 
v_reuseFailAlloc_2569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2569_, 0, v_a_2563_);
v___x_2568_ = v_reuseFailAlloc_2569_;
goto v_reusejp_2567_;
}
v_reusejp_2567_:
{
return v___x_2568_;
}
}
}
}
}
else
{
lean_dec(v_fst_2519_);
lean_dec_ref(v_pf_2518_);
lean_dec_ref(v_cancelled_2517_);
lean_dec(v_fst_2514_);
lean_dec_ref(v___y_2500_);
lean_dec(v___y_2499_);
lean_dec(v___y_2498_);
lean_dec_ref(v___y_2497_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
return v___x_2520_;
}
}
else
{
lean_dec(v_fst_2514_);
lean_dec(v_a_2513_);
lean_dec(v___x_2510_);
lean_dec(v___y_2501_);
lean_dec_ref(v___y_2500_);
lean_dec(v___y_2499_);
lean_dec(v___y_2498_);
lean_dec_ref(v___y_2497_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
return v___x_2515_;
}
}
else
{
lean_object* v_a_2572_; lean_object* v___x_2574_; uint8_t v_isShared_2575_; uint8_t v_isSharedCheck_2579_; 
lean_dec(v___x_2510_);
lean_dec(v_a_2509_);
lean_dec(v___y_2502_);
lean_dec(v___y_2501_);
lean_dec_ref(v___y_2500_);
lean_dec(v___y_2499_);
lean_dec(v___y_2498_);
lean_dec_ref(v___y_2497_);
lean_dec(v___y_2496_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2572_ = lean_ctor_get(v___x_2512_, 0);
v_isSharedCheck_2579_ = !lean_is_exclusive(v___x_2512_);
if (v_isSharedCheck_2579_ == 0)
{
v___x_2574_ = v___x_2512_;
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
else
{
lean_inc(v_a_2572_);
lean_dec(v___x_2512_);
v___x_2574_ = lean_box(0);
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
v_resetjp_2573_:
{
lean_object* v___x_2577_; 
if (v_isShared_2575_ == 0)
{
v___x_2577_ = v___x_2574_;
goto v_reusejp_2576_;
}
else
{
lean_object* v_reuseFailAlloc_2578_; 
v_reuseFailAlloc_2578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2578_, 0, v_a_2572_);
v___x_2577_ = v_reuseFailAlloc_2578_;
goto v_reusejp_2576_;
}
v_reusejp_2576_:
{
return v___x_2577_;
}
}
}
}
else
{
lean_object* v_a_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2587_; 
lean_dec(v___y_2502_);
lean_dec(v___y_2501_);
lean_dec_ref(v___y_2500_);
lean_dec(v___y_2499_);
lean_dec(v___y_2498_);
lean_dec_ref(v___y_2497_);
lean_dec(v___y_2496_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2580_ = lean_ctor_get(v___x_2508_, 0);
v_isSharedCheck_2587_ = !lean_is_exclusive(v___x_2508_);
if (v_isSharedCheck_2587_ == 0)
{
v___x_2582_ = v___x_2508_;
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_a_2580_);
lean_dec(v___x_2508_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
lean_object* v___x_2585_; 
if (v_isShared_2583_ == 0)
{
v___x_2585_ = v___x_2582_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2586_; 
v_reuseFailAlloc_2586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2586_, 0, v_a_2580_);
v___x_2585_ = v_reuseFailAlloc_2586_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
return v___x_2585_;
}
}
}
}
v___jp_2589_:
{
lean_object* v___x_2609_; 
v___x_2609_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___y_2598_, v___y_2595_, v___y_2605_, v___y_2606_, v___y_2607_, v___y_2608_);
if (lean_obj_tag(v___x_2609_) == 0)
{
lean_object* v_a_2610_; lean_object* v_snd_2611_; lean_object* v_snd_2612_; uint8_t v___x_2613_; 
v_a_2610_ = lean_ctor_get(v___x_2609_, 0);
lean_inc(v_a_2610_);
lean_dec_ref_known(v___x_2609_, 1);
v_snd_2611_ = lean_ctor_get(v_a_2610_, 1);
lean_inc(v_snd_2611_);
v_snd_2612_ = lean_ctor_get(v_snd_2611_, 1);
v___x_2613_ = lean_unbox(v_snd_2612_);
if (v___x_2613_ == 0)
{
lean_dec(v_snd_2611_);
lean_dec(v_a_2610_);
lean_dec(v_rhs_2604_);
lean_dec(v_ln_2603_);
lean_dec(v_lhs_2602_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_right_2614_; 
v_right_2614_ = lean_ctor_get(v_t_1793_, 2);
if (lean_obj_tag(v_right_2614_) == 1)
{
lean_object* v_left_2615_; lean_object* v_value_2616_; 
v_left_2615_ = lean_ctor_get(v_t_1793_, 1);
v_value_2616_ = lean_ctor_get(v_right_2614_, 0);
lean_inc(v_value_2616_);
lean_inc(v_left_2615_);
v___y_2166_ = v___y_2590_;
v___y_2167_ = v___y_2591_;
v___y_2168_ = v___y_2592_;
v___y_2169_ = v___y_2593_;
v___y_2170_ = v___y_2594_;
v___y_2171_ = v___y_2595_;
v___y_2172_ = v___y_2596_;
v___y_2173_ = v___y_2597_;
v___y_2174_ = v___y_2599_;
v___y_2175_ = v___y_2600_;
v___y_2176_ = v___y_2601_;
v_lhs_2177_ = v_left_2615_;
v_rn_2178_ = v_value_2616_;
v___y_2179_ = v___y_2605_;
v___y_2180_ = v___y_2606_;
v___y_2181_ = v___y_2607_;
v___y_2182_ = v___y_2608_;
goto v___jp_2165_;
}
else
{
lean_dec_ref(v___y_2600_);
lean_dec(v___y_2596_);
lean_dec_ref(v___x_1828_);
v___y_2389_ = v___y_2590_;
v___y_2390_ = v___y_2591_;
v___y_2391_ = v___y_2592_;
v___y_2392_ = v___y_2593_;
v___y_2393_ = v___y_2594_;
v___y_2394_ = v___y_2595_;
v___y_2395_ = v___y_2597_;
v___y_2396_ = v___y_2599_;
v___y_2397_ = v___y_2601_;
v___y_2398_ = v___y_2605_;
v___y_2399_ = v___y_2606_;
v___y_2400_ = v___y_2607_;
v___y_2401_ = v___y_2608_;
goto v___jp_2388_;
}
}
else
{
lean_dec_ref(v___y_2600_);
lean_dec(v___y_2596_);
lean_dec_ref(v___x_1828_);
v___y_2389_ = v___y_2590_;
v___y_2390_ = v___y_2591_;
v___y_2391_ = v___y_2592_;
v___y_2392_ = v___y_2593_;
v___y_2393_ = v___y_2594_;
v___y_2394_ = v___y_2595_;
v___y_2395_ = v___y_2597_;
v___y_2396_ = v___y_2599_;
v___y_2397_ = v___y_2601_;
v___y_2398_ = v___y_2605_;
v___y_2399_ = v___y_2606_;
v___y_2400_ = v___y_2607_;
v___y_2401_ = v___y_2608_;
goto v___jp_2388_;
}
}
else
{
lean_object* v_options_2617_; uint8_t v_hasTrace_2618_; 
lean_dec_ref(v___y_2601_);
lean_dec_ref(v___y_2600_);
lean_dec_ref(v___y_2599_);
lean_dec(v___y_2596_);
lean_dec_ref(v___y_2593_);
lean_dec_ref(v___y_2592_);
lean_dec_ref(v___y_2591_);
lean_dec_ref(v___x_1829_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
v_options_2617_ = lean_ctor_get(v___y_2607_, 2);
v_hasTrace_2618_ = lean_ctor_get_uint8(v_options_2617_, sizeof(void*)*1);
if (v_hasTrace_2618_ == 0)
{
lean_object* v_fst_2619_; lean_object* v_fst_2620_; 
v_fst_2619_ = lean_ctor_get(v_a_2610_, 0);
lean_inc(v_fst_2619_);
lean_dec(v_a_2610_);
v_fst_2620_ = lean_ctor_get(v_snd_2611_, 0);
lean_inc(v_fst_2620_);
lean_dec(v_snd_2611_);
v___y_2496_ = v_lhs_2602_;
v___y_2497_ = v___y_2594_;
v___y_2498_ = v_fst_2620_;
v___y_2499_ = v_fst_2619_;
v___y_2500_ = v___y_2597_;
v___y_2501_ = v_rhs_2604_;
v___y_2502_ = v_ln_2603_;
v___y_2503_ = v___y_2605_;
v___y_2504_ = v___y_2606_;
v___y_2505_ = v___y_2607_;
v___y_2506_ = v___y_2608_;
goto v___jp_2495_;
}
else
{
lean_object* v_fst_2621_; lean_object* v_fst_2622_; lean_object* v_inheritedTraceOptions_2623_; lean_object* v___x_2624_; uint8_t v___x_2625_; 
v_fst_2621_ = lean_ctor_get(v_a_2610_, 0);
lean_inc(v_fst_2621_);
lean_dec(v_a_2610_);
v_fst_2622_ = lean_ctor_get(v_snd_2611_, 0);
lean_inc(v_fst_2622_);
lean_dec(v_snd_2611_);
v_inheritedTraceOptions_2623_ = lean_ctor_get(v___y_2607_, 13);
v___x_2624_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56);
v___x_2625_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2623_, v_options_2617_, v___x_2624_);
if (v___x_2625_ == 0)
{
v___y_2496_ = v_lhs_2602_;
v___y_2497_ = v___y_2594_;
v___y_2498_ = v_fst_2622_;
v___y_2499_ = v_fst_2621_;
v___y_2500_ = v___y_2597_;
v___y_2501_ = v_rhs_2604_;
v___y_2502_ = v_ln_2603_;
v___y_2503_ = v___y_2605_;
v___y_2504_ = v___y_2606_;
v___y_2505_ = v___y_2607_;
v___y_2506_ = v___y_2608_;
goto v___jp_2495_;
}
else
{
lean_object* v___x_2626_; lean_object* v___x_2627_; 
v___x_2626_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__58, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__58_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__58);
v___x_2627_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(v_cls_2588_, v___x_2626_, v___y_2605_, v___y_2606_, v___y_2607_, v___y_2608_);
if (lean_obj_tag(v___x_2627_) == 0)
{
lean_dec_ref_known(v___x_2627_, 1);
v___y_2496_ = v_lhs_2602_;
v___y_2497_ = v___y_2594_;
v___y_2498_ = v_fst_2622_;
v___y_2499_ = v_fst_2621_;
v___y_2500_ = v___y_2597_;
v___y_2501_ = v_rhs_2604_;
v___y_2502_ = v_ln_2603_;
v___y_2503_ = v___y_2605_;
v___y_2504_ = v___y_2606_;
v___y_2505_ = v___y_2607_;
v___y_2506_ = v___y_2608_;
goto v___jp_2495_;
}
else
{
lean_object* v_a_2628_; lean_object* v___x_2630_; uint8_t v_isShared_2631_; uint8_t v_isSharedCheck_2635_; 
lean_dec(v_fst_2622_);
lean_dec(v_fst_2621_);
lean_dec(v_rhs_2604_);
lean_dec(v_ln_2603_);
lean_dec(v_lhs_2602_);
lean_dec_ref(v___y_2597_);
lean_dec_ref(v___y_2594_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2628_ = lean_ctor_get(v___x_2627_, 0);
v_isSharedCheck_2635_ = !lean_is_exclusive(v___x_2627_);
if (v_isSharedCheck_2635_ == 0)
{
v___x_2630_ = v___x_2627_;
v_isShared_2631_ = v_isSharedCheck_2635_;
goto v_resetjp_2629_;
}
else
{
lean_inc(v_a_2628_);
lean_dec(v___x_2627_);
v___x_2630_ = lean_box(0);
v_isShared_2631_ = v_isSharedCheck_2635_;
goto v_resetjp_2629_;
}
v_resetjp_2629_:
{
lean_object* v___x_2633_; 
if (v_isShared_2631_ == 0)
{
v___x_2633_ = v___x_2630_;
goto v_reusejp_2632_;
}
else
{
lean_object* v_reuseFailAlloc_2634_; 
v_reuseFailAlloc_2634_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2634_, 0, v_a_2628_);
v___x_2633_ = v_reuseFailAlloc_2634_;
goto v_reusejp_2632_;
}
v_reusejp_2632_:
{
return v___x_2633_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2636_; lean_object* v___x_2638_; uint8_t v_isShared_2639_; uint8_t v_isSharedCheck_2643_; 
lean_dec(v_rhs_2604_);
lean_dec(v_ln_2603_);
lean_dec(v_lhs_2602_);
lean_dec_ref(v___y_2601_);
lean_dec_ref(v___y_2600_);
lean_dec_ref(v___y_2599_);
lean_dec_ref(v___y_2597_);
lean_dec(v___y_2596_);
lean_dec_ref(v___y_2594_);
lean_dec_ref(v___y_2593_);
lean_dec_ref(v___y_2592_);
lean_dec_ref(v___y_2591_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1829_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2636_ = lean_ctor_get(v___x_2609_, 0);
v_isSharedCheck_2643_ = !lean_is_exclusive(v___x_2609_);
if (v_isSharedCheck_2643_ == 0)
{
v___x_2638_ = v___x_2609_;
v_isShared_2639_ = v_isSharedCheck_2643_;
goto v_resetjp_2637_;
}
else
{
lean_inc(v_a_2636_);
lean_dec(v___x_2609_);
v___x_2638_ = lean_box(0);
v_isShared_2639_ = v_isSharedCheck_2643_;
goto v_resetjp_2637_;
}
v_resetjp_2637_:
{
lean_object* v___x_2641_; 
if (v_isShared_2639_ == 0)
{
v___x_2641_ = v___x_2638_;
goto v_reusejp_2640_;
}
else
{
lean_object* v_reuseFailAlloc_2642_; 
v_reuseFailAlloc_2642_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2642_, 0, v_a_2636_);
v___x_2641_ = v_reuseFailAlloc_2642_;
goto v_reusejp_2640_;
}
v_reusejp_2640_:
{
return v___x_2641_;
}
}
}
}
v___jp_2644_:
{
lean_object* v___x_2663_; 
v___x_2663_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___y_2660_, v___y_2654_, v___y_2650_, v___y_2648_, v___y_2649_, v___y_2646_);
if (lean_obj_tag(v___x_2663_) == 0)
{
lean_object* v_a_2664_; lean_object* v_snd_2665_; lean_object* v_snd_2666_; uint8_t v___x_2667_; 
v_a_2664_ = lean_ctor_get(v___x_2663_, 0);
lean_inc(v_a_2664_);
lean_dec_ref_known(v___x_2663_, 1);
v_snd_2665_ = lean_ctor_get(v_a_2664_, 1);
lean_inc(v_snd_2665_);
v_snd_2666_ = lean_ctor_get(v_snd_2665_, 1);
v___x_2667_ = lean_unbox(v_snd_2666_);
if (v___x_2667_ == 0)
{
lean_dec(v_snd_2665_);
lean_dec(v_a_2664_);
lean_dec(v_rhs_2662_);
lean_dec(v_lhs_2661_);
lean_dec_ref(v___x_1830_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_2668_; 
v_left_2668_ = lean_ctor_get(v_t_1793_, 1);
if (lean_obj_tag(v_left_2668_) == 1)
{
lean_object* v_right_2669_; lean_object* v_value_2670_; 
v_right_2669_ = lean_ctor_get(v_t_1793_, 2);
v_value_2670_ = lean_ctor_get(v_left_2668_, 0);
lean_inc(v_right_2669_);
lean_inc(v_value_2670_);
lean_inc_ref(v_left_2668_);
lean_inc_ref(v___y_2647_);
v___y_2590_ = v___y_2645_;
v___y_2591_ = v___y_2647_;
v___y_2592_ = v___y_2651_;
v___y_2593_ = v___y_2652_;
v___y_2594_ = v___y_2653_;
v___y_2595_ = v___y_2654_;
v___y_2596_ = v___y_2655_;
v___y_2597_ = v___y_2656_;
v___y_2598_ = v___y_2657_;
v___y_2599_ = v___y_2647_;
v___y_2600_ = v___y_2658_;
v___y_2601_ = v___y_2659_;
v_lhs_2602_ = v_left_2668_;
v_ln_2603_ = v_value_2670_;
v_rhs_2604_ = v_right_2669_;
v___y_2605_ = v___y_2650_;
v___y_2606_ = v___y_2648_;
v___y_2607_ = v___y_2649_;
v___y_2608_ = v___y_2646_;
goto v___jp_2589_;
}
else
{
lean_dec_ref(v___y_2657_);
lean_inc_ref(v___y_2647_);
v___y_2477_ = v___y_2645_;
v___y_2478_ = v___y_2647_;
v___y_2479_ = v___y_2651_;
v___y_2480_ = v___y_2652_;
v___y_2481_ = v___y_2654_;
v___y_2482_ = v___y_2653_;
v___y_2483_ = v___y_2655_;
v___y_2484_ = v___y_2656_;
v___y_2485_ = v___y_2647_;
v___y_2486_ = v___y_2658_;
v___y_2487_ = v___y_2659_;
v___y_2488_ = v___y_2650_;
v___y_2489_ = v___y_2648_;
v___y_2490_ = v___y_2649_;
v___y_2491_ = v___y_2646_;
goto v___jp_2476_;
}
}
else
{
lean_dec_ref(v___y_2657_);
lean_inc_ref(v___y_2647_);
v___y_2477_ = v___y_2645_;
v___y_2478_ = v___y_2647_;
v___y_2479_ = v___y_2651_;
v___y_2480_ = v___y_2652_;
v___y_2481_ = v___y_2654_;
v___y_2482_ = v___y_2653_;
v___y_2483_ = v___y_2655_;
v___y_2484_ = v___y_2656_;
v___y_2485_ = v___y_2647_;
v___y_2486_ = v___y_2658_;
v___y_2487_ = v___y_2659_;
v___y_2488_ = v___y_2650_;
v___y_2489_ = v___y_2648_;
v___y_2490_ = v___y_2649_;
v___y_2491_ = v___y_2646_;
goto v___jp_2476_;
}
}
else
{
lean_object* v_fst_2671_; lean_object* v_fst_2672_; lean_object* v___x_2673_; 
lean_dec_ref(v___y_2659_);
lean_dec_ref(v___y_2658_);
lean_dec_ref(v___y_2657_);
lean_dec_ref(v___y_2656_);
lean_dec_ref(v___y_2653_);
lean_dec_ref(v___y_2652_);
lean_dec_ref(v___y_2651_);
lean_dec_ref(v___y_2647_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
v_fst_2671_ = lean_ctor_get(v_a_2664_, 0);
lean_inc_n(v_fst_2671_, 2);
lean_dec(v_a_2664_);
v_fst_2672_ = lean_ctor_get(v_snd_2665_, 0);
lean_inc(v_fst_2672_);
lean_dec(v_snd_2665_);
lean_inc_ref(v_v_x27_1792_);
lean_inc(v_v_1791_);
lean_inc_ref(v_s_u03b1_1790_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2673_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v_v_1791_, v_v_x27_1792_, v_lhs_2661_, v_fst_2671_, v___y_2650_, v___y_2648_, v___y_2649_, v___y_2646_);
if (lean_obj_tag(v___x_2673_) == 0)
{
lean_object* v_a_2674_; lean_object* v_cancelled_2675_; lean_object* v_pf_2676_; lean_object* v___x_2677_; 
v_a_2674_ = lean_ctor_get(v___x_2673_, 0);
lean_inc(v_a_2674_);
lean_dec_ref_known(v___x_2673_, 1);
v_cancelled_2675_ = lean_ctor_get(v_a_2674_, 0);
lean_inc_ref(v_cancelled_2675_);
v_pf_2676_ = lean_ctor_get(v_a_2674_, 1);
lean_inc_ref(v_pf_2676_);
lean_dec(v_a_2674_);
lean_inc(v_fst_2672_);
lean_inc_ref(v_v_x27_1792_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2677_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v_v_1791_, v_v_x27_1792_, v_rhs_2662_, v_fst_2672_, v___y_2650_, v___y_2648_, v___y_2649_, v___y_2646_);
if (lean_obj_tag(v___x_2677_) == 0)
{
lean_object* v_a_2678_; lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_2729_; 
v_a_2678_ = lean_ctor_get(v___x_2677_, 0);
v_isSharedCheck_2729_ = !lean_is_exclusive(v___x_2677_);
if (v_isSharedCheck_2729_ == 0)
{
v___x_2680_ = v___x_2677_;
v_isShared_2681_ = v_isSharedCheck_2729_;
goto v_resetjp_2679_;
}
else
{
lean_inc(v_a_2678_);
lean_dec(v___x_2677_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_2729_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v_cancelled_2682_; lean_object* v_pf_2683_; lean_object* v___x_2685_; uint8_t v_isShared_2686_; uint8_t v_isSharedCheck_2728_; 
v_cancelled_2682_ = lean_ctor_get(v_a_2678_, 0);
v_pf_2683_ = lean_ctor_get(v_a_2678_, 1);
v_isSharedCheck_2728_ = !lean_is_exclusive(v_a_2678_);
if (v_isSharedCheck_2728_ == 0)
{
v___x_2685_ = v_a_2678_;
v_isShared_2686_ = v_isSharedCheck_2728_;
goto v_resetjp_2684_;
}
else
{
lean_inc(v_pf_2683_);
lean_inc(v_cancelled_2682_);
lean_dec(v_a_2678_);
v___x_2685_ = lean_box(0);
v_isShared_2686_ = v_isSharedCheck_2728_;
goto v_resetjp_2684_;
}
v_resetjp_2684_:
{
lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2723_; 
v___x_2687_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__0));
v___x_2688_ = l_Lean_Expr_const___override(v___x_2687_, v___y_2655_);
lean_inc_ref_n(v_00_u03b1_1789_, 7);
v___x_2689_ = l_Lean_Expr_app___override(v___x_2688_, v_00_u03b1_1789_);
v___x_2690_ = l_Lean_Expr_app___override(v___x_2689_, v_00_u03b1_1789_);
v___x_2691_ = l_Lean_Expr_app___override(v___x_2690_, v_00_u03b1_1789_);
v___x_2692_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__2));
lean_inc_ref_n(v___x_1802_, 4);
v___x_2693_ = l_Lean_Expr_const___override(v___x_2692_, v___x_1802_);
v___x_2694_ = l_Lean_Expr_app___override(v___x_2693_, v_00_u03b1_1789_);
v___x_2695_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__5));
v___x_2696_ = l_Lean_Expr_const___override(v___x_2695_, v___x_1802_);
v___x_2697_ = l_Lean_Expr_app___override(v___x_2696_, v_00_u03b1_1789_);
v___x_2698_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___closed__8));
v___x_2699_ = l_Lean_Expr_const___override(v___x_2698_, v___x_1802_);
v___x_2700_ = l_Lean_Expr_app___override(v___x_2699_, v_00_u03b1_1789_);
v___x_2701_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__59));
v___x_2702_ = l_Lean_Expr_const___override(v___x_2701_, v___x_1802_);
v___x_2703_ = l_Lean_Expr_app___override(v___x_2702_, v_00_u03b1_1789_);
v___x_2704_ = l_Lean_Expr_app___override(v___x_2703_, v___x_1830_);
v___x_2705_ = l_Lean_Expr_app___override(v___x_2700_, v___x_2704_);
v___x_2706_ = l_Lean_Expr_app___override(v___x_2697_, v___x_2705_);
v___x_2707_ = l_Lean_Expr_app___override(v___x_2694_, v___x_2706_);
v___x_2708_ = l_Lean_Expr_app___override(v___x_2691_, v___x_2707_);
lean_inc_ref(v_cancelled_2675_);
v___x_2709_ = l_Lean_Expr_app___override(v___x_2708_, v_cancelled_2675_);
lean_inc_ref(v_cancelled_2682_);
v___x_2710_ = l_Lean_Expr_app___override(v___x_2709_, v_cancelled_2682_);
v___x_2711_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__61));
v___x_2712_ = l_Lean_Expr_const___override(v___x_2711_, v___x_1802_);
v___x_2713_ = l_Lean_Expr_app___override(v___x_2712_, v_00_u03b1_1789_);
v___x_2714_ = l_Lean_Expr_app___override(v___x_2713_, v___x_1829_);
v___x_2715_ = l_Lean_Expr_app___override(v___x_2714_, v_v_x27_1792_);
v___x_2716_ = l_Lean_Expr_app___override(v___x_2715_, v_fst_2671_);
v___x_2717_ = l_Lean_Expr_app___override(v___x_2716_, v_fst_2672_);
v___x_2718_ = l_Lean_Expr_app___override(v___x_2717_, v_cancelled_2675_);
v___x_2719_ = l_Lean_Expr_app___override(v___x_2718_, v_cancelled_2682_);
v___x_2720_ = l_Lean_Expr_app___override(v___x_2719_, v_pf_2676_);
v___x_2721_ = l_Lean_Expr_app___override(v___x_2720_, v_pf_2683_);
if (v_isShared_2686_ == 0)
{
lean_ctor_set(v___x_2685_, 1, v___x_2721_);
lean_ctor_set(v___x_2685_, 0, v___x_2710_);
v___x_2723_ = v___x_2685_;
goto v_reusejp_2722_;
}
else
{
lean_object* v_reuseFailAlloc_2727_; 
v_reuseFailAlloc_2727_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2727_, 0, v___x_2710_);
lean_ctor_set(v_reuseFailAlloc_2727_, 1, v___x_2721_);
v___x_2723_ = v_reuseFailAlloc_2727_;
goto v_reusejp_2722_;
}
v_reusejp_2722_:
{
lean_object* v___x_2725_; 
if (v_isShared_2681_ == 0)
{
lean_ctor_set(v___x_2680_, 0, v___x_2723_);
v___x_2725_ = v___x_2680_;
goto v_reusejp_2724_;
}
else
{
lean_object* v_reuseFailAlloc_2726_; 
v_reuseFailAlloc_2726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2726_, 0, v___x_2723_);
v___x_2725_ = v_reuseFailAlloc_2726_;
goto v_reusejp_2724_;
}
v_reusejp_2724_:
{
return v___x_2725_;
}
}
}
}
}
else
{
lean_dec_ref(v_pf_2676_);
lean_dec_ref(v_cancelled_2675_);
lean_dec(v_fst_2672_);
lean_dec(v_fst_2671_);
lean_dec(v___y_2655_);
lean_dec_ref(v___x_1830_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_00_u03b1_1789_);
return v___x_2677_;
}
}
else
{
lean_dec(v_fst_2672_);
lean_dec(v_fst_2671_);
lean_dec(v_rhs_2662_);
lean_dec(v___y_2655_);
lean_dec_ref(v___x_1830_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
return v___x_2673_;
}
}
}
else
{
lean_object* v_a_2730_; lean_object* v___x_2732_; uint8_t v_isShared_2733_; uint8_t v_isSharedCheck_2737_; 
lean_dec(v_rhs_2662_);
lean_dec(v_lhs_2661_);
lean_dec_ref(v___y_2659_);
lean_dec_ref(v___y_2658_);
lean_dec_ref(v___y_2657_);
lean_dec_ref(v___y_2656_);
lean_dec(v___y_2655_);
lean_dec_ref(v___y_2653_);
lean_dec_ref(v___y_2652_);
lean_dec_ref(v___y_2651_);
lean_dec_ref(v___y_2647_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1830_);
lean_dec_ref(v___x_1829_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec(v_t_1793_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2730_ = lean_ctor_get(v___x_2663_, 0);
v_isSharedCheck_2737_ = !lean_is_exclusive(v___x_2663_);
if (v_isSharedCheck_2737_ == 0)
{
v___x_2732_ = v___x_2663_;
v_isShared_2733_ = v_isSharedCheck_2737_;
goto v_resetjp_2731_;
}
else
{
lean_inc(v_a_2730_);
lean_dec(v___x_2663_);
v___x_2732_ = lean_box(0);
v_isShared_2733_ = v_isSharedCheck_2737_;
goto v_resetjp_2731_;
}
v_resetjp_2731_:
{
lean_object* v___x_2735_; 
if (v_isShared_2733_ == 0)
{
v___x_2735_ = v___x_2732_;
goto v_reusejp_2734_;
}
else
{
lean_object* v_reuseFailAlloc_2736_; 
v_reuseFailAlloc_2736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2736_, 0, v_a_2730_);
v___x_2735_ = v_reuseFailAlloc_2736_;
goto v_reusejp_2734_;
}
v_reusejp_2734_:
{
return v___x_2735_;
}
}
}
}
v___jp_2738_:
{
lean_object* v___x_2743_; lean_object* v___x_2744_; lean_object* v___x_2745_; lean_object* v___x_2746_; lean_object* v___x_2747_; lean_object* v___x_2748_; lean_object* v___x_2749_; lean_object* v___x_2750_; lean_object* v___x_2751_; lean_object* v___x_2752_; lean_object* v___x_2753_; lean_object* v___x_2754_; lean_object* v___x_2755_; lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___x_2761_; lean_object* v___x_2762_; lean_object* v___x_2763_; lean_object* v___x_2764_; lean_object* v___x_2765_; lean_object* v___x_2766_; lean_object* v___x_2767_; lean_object* v___x_2768_; lean_object* v___x_2769_; lean_object* v___x_2770_; lean_object* v___x_2771_; lean_object* v___x_2772_; lean_object* v___x_2773_; lean_object* v___x_2774_; lean_object* v___x_2775_; lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v___x_2792_; lean_object* v___x_2793_; lean_object* v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; uint8_t v___x_2797_; lean_object* v___x_2798_; lean_object* v___x_2799_; lean_object* v___f_2800_; uint8_t v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_2803_; lean_object* v___f_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; lean_object* v___f_2807_; lean_object* v___x_2808_; lean_object* v___x_2809_; lean_object* v___f_2810_; 
v___x_2743_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__63));
lean_inc_ref_n(v___x_1802_, 11);
v___x_2744_ = l_Lean_Expr_const___override(v___x_2743_, v___x_1802_);
lean_inc_ref_n(v_00_u03b1_1789_, 16);
v___x_2745_ = l_Lean_Expr_app___override(v___x_2744_, v_00_u03b1_1789_);
v___x_2746_ = l_Lean_Expr_app___override(v___x_1823_, v___x_2745_);
v___x_2747_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__64));
v___x_2748_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__66));
v___x_2749_ = l_Lean_Expr_const___override(v___x_2748_, v___x_1802_);
v___x_2750_ = l_Lean_Expr_app___override(v___x_2749_, v_00_u03b1_1789_);
v___x_2751_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__68));
v___x_2752_ = l_Lean_Expr_const___override(v___x_2751_, v___x_1802_);
v___x_2753_ = l_Lean_Expr_app___override(v___x_2752_, v_00_u03b1_1789_);
v___x_2754_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71));
v___x_2755_ = l_Lean_Expr_const___override(v___x_2754_, v___x_1802_);
v___x_2756_ = l_Lean_Expr_app___override(v___x_2755_, v_00_u03b1_1789_);
v___x_2757_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__72));
v___x_2758_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74));
v___x_2759_ = l_Lean_Expr_const___override(v___x_2758_, v___x_1802_);
v___x_2760_ = l_Lean_Expr_app___override(v___x_2759_, v_00_u03b1_1789_);
v___x_2761_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76));
v___x_2762_ = l_Lean_Expr_const___override(v___x_2761_, v___x_1802_);
v___x_2763_ = l_Lean_Expr_app___override(v___x_2762_, v_00_u03b1_1789_);
lean_inc_ref(v_s_u03b1_1790_);
v___x_2764_ = l_Lean_Expr_app___override(v___x_2763_, v_s_u03b1_1790_);
lean_inc_ref(v___x_2764_);
v___x_2765_ = l_Lean_Expr_app___override(v___x_2760_, v___x_2764_);
v___x_2766_ = l_Lean_Expr_app___override(v___x_2756_, v___x_2765_);
lean_inc_ref(v___x_2766_);
v___x_2767_ = l_Lean_Expr_app___override(v___x_2753_, v___x_2766_);
lean_inc_ref(v___x_2767_);
v___x_2768_ = l_Lean_Expr_app___override(v___x_2750_, v___x_2767_);
lean_inc_ref(v___x_2768_);
v___x_2769_ = l_Lean_Expr_app___override(v___x_2746_, v___x_2768_);
v___x_2770_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__77));
lean_inc_n(v_u_1788_, 2);
v___x_2771_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2771_, 0, v_u_1788_);
lean_ctor_set(v___x_2771_, 1, v___x_1802_);
v___x_2772_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2772_, 0, v_u_1788_);
lean_ctor_set(v___x_2772_, 1, v___x_2771_);
lean_inc_ref_n(v___x_2772_, 3);
v___x_2773_ = l_Lean_Expr_const___override(v___x_2770_, v___x_2772_);
v___x_2774_ = l_Lean_Expr_app___override(v___x_2773_, v_00_u03b1_1789_);
v___x_2775_ = l_Lean_Expr_app___override(v___x_2774_, v_00_u03b1_1789_);
v___x_2776_ = l_Lean_Expr_app___override(v___x_2775_, v_00_u03b1_1789_);
v___x_2777_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__79));
v___x_2778_ = l_Lean_Expr_const___override(v___x_2777_, v___x_1802_);
v___x_2779_ = l_Lean_Expr_app___override(v___x_2778_, v_00_u03b1_1789_);
lean_inc_ref(v___x_2779_);
v___x_2780_ = l_Lean_Expr_app___override(v___x_2779_, v___x_2768_);
lean_inc_ref(v___x_2776_);
v___x_2781_ = l_Lean_Expr_app___override(v___x_2776_, v___x_2780_);
lean_inc_ref_n(v_v_x27_1792_, 2);
lean_inc_ref_n(v___x_2781_, 2);
v___x_2782_ = l_Lean_Expr_app___override(v___x_2781_, v_v_x27_1792_);
lean_inc_ref_n(v_e_1794_, 6);
v___x_2783_ = l_Lean_Expr_app___override(v___x_2782_, v_e_1794_);
v___x_2784_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__81));
lean_inc_ref_n(v___x_1822_, 2);
v___x_2785_ = l_Lean_Expr_const___override(v___x_2784_, v___x_1822_);
v___x_2786_ = l_Lean_Expr_app___override(v___x_2785_, v_00_u03b1_1789_);
v___x_2787_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__83));
v___x_2788_ = l_Lean_Expr_const___override(v___x_2787_, v___x_1822_);
v___x_2789_ = l_Lean_Expr_app___override(v___x_2788_, v_00_u03b1_1789_);
v___x_2790_ = l_Lean_Expr_app___override(v___x_2779_, v___x_2769_);
v___x_2791_ = l_Lean_Expr_app___override(v___x_2776_, v___x_2790_);
v___x_2792_ = l_Lean_Expr_app___override(v___x_2791_, v_v_x27_1792_);
v___x_2793_ = l_Lean_Expr_app___override(v___x_2792_, v_e_1794_);
v___x_2794_ = l_Lean_Expr_app___override(v___x_2789_, v___x_2793_);
v___x_2795_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2795_, 0, v___x_2783_);
lean_ctor_set(v___x_2795_, 1, v___x_2794_);
v___x_2796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2796_, 0, v_00_u03b1_1789_);
v___x_2797_ = 0;
v___x_2798_ = lean_box(0);
v___x_2799_ = lean_box(v___x_2797_);
lean_inc_ref(v___x_1829_);
lean_inc_ref_n(v___x_2796_, 4);
v___f_2800_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__2___boxed), 13, 8);
lean_closure_set(v___f_2800_, 0, v___x_2796_);
lean_closure_set(v___f_2800_, 1, v___x_2799_);
lean_closure_set(v___f_2800_, 2, v___x_2798_);
lean_closure_set(v___f_2800_, 3, v___x_1802_);
lean_closure_set(v___f_2800_, 4, v_00_u03b1_1789_);
lean_closure_set(v___f_2800_, 5, v___x_1809_);
lean_closure_set(v___f_2800_, 6, v___x_1829_);
lean_closure_set(v___f_2800_, 7, v_e_1794_);
v___x_2801_ = 0;
v___x_2802_ = lean_box(v___x_2797_);
v___x_2803_ = lean_box(v___x_2801_);
lean_inc_ref(v___x_1828_);
v___f_2804_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__3___boxed), 15, 10);
lean_closure_set(v___f_2804_, 0, v___x_2796_);
lean_closure_set(v___f_2804_, 1, v___x_2802_);
lean_closure_set(v___f_2804_, 2, v___x_2798_);
lean_closure_set(v___f_2804_, 3, v___x_2772_);
lean_closure_set(v___f_2804_, 4, v_00_u03b1_1789_);
lean_closure_set(v___f_2804_, 5, v___x_1802_);
lean_closure_set(v___f_2804_, 6, v___x_1816_);
lean_closure_set(v___f_2804_, 7, v___x_1828_);
lean_closure_set(v___f_2804_, 8, v_e_1794_);
lean_closure_set(v___f_2804_, 9, v___x_2803_);
v___x_2805_ = lean_box(v___x_2797_);
v___x_2806_ = lean_box(v___x_2801_);
v___f_2807_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__4___boxed), 11, 6);
lean_closure_set(v___f_2807_, 0, v___x_2796_);
lean_closure_set(v___f_2807_, 1, v___x_2805_);
lean_closure_set(v___f_2807_, 2, v___x_2798_);
lean_closure_set(v___f_2807_, 3, v___x_2781_);
lean_closure_set(v___f_2807_, 4, v_e_1794_);
lean_closure_set(v___f_2807_, 5, v___x_2806_);
v___x_2808_ = lean_box(v___x_2797_);
v___x_2809_ = lean_box(v___x_2801_);
lean_inc_ref(v___x_1830_);
v___f_2810_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__5___boxed), 15, 10);
lean_closure_set(v___f_2810_, 0, v___x_2796_);
lean_closure_set(v___f_2810_, 1, v___x_2808_);
lean_closure_set(v___f_2810_, 2, v___x_2798_);
lean_closure_set(v___f_2810_, 3, v___x_2772_);
lean_closure_set(v___f_2810_, 4, v_00_u03b1_1789_);
lean_closure_set(v___f_2810_, 5, v___x_1802_);
lean_closure_set(v___f_2810_, 6, v___x_1805_);
lean_closure_set(v___f_2810_, 7, v___x_1830_);
lean_closure_set(v___f_2810_, 8, v_e_1794_);
lean_closure_set(v___f_2810_, 9, v___x_2809_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_2811_; lean_object* v_right_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; lean_object* v___f_2815_; lean_object* v___x_2816_; 
v_left_2811_ = lean_ctor_get(v_t_1793_, 1);
v_right_2812_ = lean_ctor_get(v_t_1793_, 2);
v___x_2813_ = lean_box(v___x_2797_);
v___x_2814_ = lean_box(v___x_2801_);
lean_inc_ref(v_e_1794_);
lean_inc_ref(v___x_2767_);
lean_inc_ref(v___x_1802_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc_ref(v___x_2772_);
v___f_2815_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___boxed), 15, 10);
lean_closure_set(v___f_2815_, 0, v___x_2796_);
lean_closure_set(v___f_2815_, 1, v___x_2813_);
lean_closure_set(v___f_2815_, 2, v___x_2798_);
lean_closure_set(v___f_2815_, 3, v___x_2772_);
lean_closure_set(v___f_2815_, 4, v_00_u03b1_1789_);
lean_closure_set(v___f_2815_, 5, v___x_1802_);
lean_closure_set(v___f_2815_, 6, v___x_2747_);
lean_closure_set(v___f_2815_, 7, v___x_2767_);
lean_closure_set(v___f_2815_, 8, v_e_1794_);
lean_closure_set(v___f_2815_, 9, v___x_2814_);
v___x_2816_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__1___redArg(v___f_2815_, v___x_2801_, v___y_2739_, v___y_2740_, v___y_2741_, v___y_2742_);
if (lean_obj_tag(v___x_2816_) == 0)
{
lean_object* v_a_2817_; lean_object* v_snd_2818_; lean_object* v_snd_2819_; uint8_t v___x_2820_; 
v_a_2817_ = lean_ctor_get(v___x_2816_, 0);
lean_inc(v_a_2817_);
lean_dec_ref_known(v___x_2816_, 1);
v_snd_2818_ = lean_ctor_get(v_a_2817_, 1);
lean_inc(v_snd_2818_);
v_snd_2819_ = lean_ctor_get(v_snd_2818_, 1);
v___x_2820_ = lean_unbox(v_snd_2819_);
if (v___x_2820_ == 0)
{
lean_dec(v_snd_2818_);
lean_dec(v_a_2817_);
lean_dec_ref(v___x_2767_);
lean_inc(v_right_2812_);
lean_inc(v_left_2811_);
v___y_2645_ = v___x_2757_;
v___y_2646_ = v___y_2742_;
v___y_2647_ = v___x_2766_;
v___y_2648_ = v___y_2740_;
v___y_2649_ = v___y_2741_;
v___y_2650_ = v___y_2739_;
v___y_2651_ = v___x_2764_;
v___y_2652_ = v___x_2795_;
v___y_2653_ = v___x_2781_;
v___y_2654_ = v___x_2801_;
v___y_2655_ = v___x_2772_;
v___y_2656_ = v___x_2786_;
v___y_2657_ = v___f_2807_;
v___y_2658_ = v___f_2804_;
v___y_2659_ = v___f_2800_;
v___y_2660_ = v___f_2810_;
v_lhs_2661_ = v_left_2811_;
v_rhs_2662_ = v_right_2812_;
goto v___jp_2644_;
}
else
{
lean_object* v_fst_2821_; lean_object* v_fst_2822_; lean_object* v___x_2823_; 
lean_inc(v_right_2812_);
lean_inc(v_left_2811_);
lean_dec_ref_known(v_t_1793_, 3);
lean_dec_ref(v___f_2810_);
lean_dec_ref(v___f_2807_);
lean_dec_ref(v___f_2804_);
lean_dec_ref(v___f_2800_);
lean_dec_ref_known(v___x_2795_, 2);
lean_dec_ref(v___x_2786_);
lean_dec_ref(v___x_2781_);
lean_dec_ref(v___x_2766_);
lean_dec_ref(v___x_2764_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1830_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref(v_e_1794_);
v_fst_2821_ = lean_ctor_get(v_a_2817_, 0);
lean_inc_n(v_fst_2821_, 2);
lean_dec(v_a_2817_);
v_fst_2822_ = lean_ctor_get(v_snd_2818_, 0);
lean_inc(v_fst_2822_);
lean_dec(v_snd_2818_);
lean_inc_ref(v_v_x27_1792_);
lean_inc(v_v_1791_);
lean_inc_ref(v_s_u03b1_1790_);
lean_inc_ref(v_00_u03b1_1789_);
lean_inc(v_u_1788_);
v___x_2823_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v_v_1791_, v_v_x27_1792_, v_left_2811_, v_fst_2821_, v___y_2739_, v___y_2740_, v___y_2741_, v___y_2742_);
if (lean_obj_tag(v___x_2823_) == 0)
{
lean_object* v_a_2824_; lean_object* v_cancelled_2825_; lean_object* v_pf_2826_; lean_object* v___x_2827_; 
v_a_2824_ = lean_ctor_get(v___x_2823_, 0);
lean_inc(v_a_2824_);
lean_dec_ref_known(v___x_2823_, 1);
v_cancelled_2825_ = lean_ctor_get(v_a_2824_, 0);
lean_inc_ref(v_cancelled_2825_);
v_pf_2826_ = lean_ctor_get(v_a_2824_, 1);
lean_inc_ref(v_pf_2826_);
lean_dec(v_a_2824_);
lean_inc(v_fst_2822_);
lean_inc_ref(v_v_x27_1792_);
lean_inc_ref(v_00_u03b1_1789_);
v___x_2827_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_1788_, v_00_u03b1_1789_, v_s_u03b1_1790_, v_v_1791_, v_v_x27_1792_, v_right_2812_, v_fst_2822_, v___y_2739_, v___y_2740_, v___y_2741_, v___y_2742_);
if (lean_obj_tag(v___x_2827_) == 0)
{
lean_object* v_a_2828_; lean_object* v___x_2830_; uint8_t v_isShared_2831_; uint8_t v_isSharedCheck_2871_; 
v_a_2828_ = lean_ctor_get(v___x_2827_, 0);
v_isSharedCheck_2871_ = !lean_is_exclusive(v___x_2827_);
if (v_isSharedCheck_2871_ == 0)
{
v___x_2830_ = v___x_2827_;
v_isShared_2831_ = v_isSharedCheck_2871_;
goto v_resetjp_2829_;
}
else
{
lean_inc(v_a_2828_);
lean_dec(v___x_2827_);
v___x_2830_ = lean_box(0);
v_isShared_2831_ = v_isSharedCheck_2871_;
goto v_resetjp_2829_;
}
v_resetjp_2829_:
{
lean_object* v_cancelled_2832_; lean_object* v_pf_2833_; lean_object* v___x_2835_; uint8_t v_isShared_2836_; uint8_t v_isSharedCheck_2870_; 
v_cancelled_2832_ = lean_ctor_get(v_a_2828_, 0);
v_pf_2833_ = lean_ctor_get(v_a_2828_, 1);
v_isSharedCheck_2870_ = !lean_is_exclusive(v_a_2828_);
if (v_isSharedCheck_2870_ == 0)
{
v___x_2835_ = v_a_2828_;
v_isShared_2836_ = v_isSharedCheck_2870_;
goto v_resetjp_2834_;
}
else
{
lean_inc(v_pf_2833_);
lean_inc(v_cancelled_2832_);
lean_dec(v_a_2828_);
v___x_2835_ = lean_box(0);
v_isShared_2836_ = v_isSharedCheck_2870_;
goto v_resetjp_2834_;
}
v_resetjp_2834_:
{
lean_object* v___x_2837_; lean_object* v___x_2838_; lean_object* v___x_2839_; lean_object* v___x_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; lean_object* v___x_2865_; 
v___x_2837_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__0));
v___x_2838_ = l_Lean_Expr_const___override(v___x_2837_, v___x_2772_);
lean_inc_ref_n(v_00_u03b1_1789_, 5);
v___x_2839_ = l_Lean_Expr_app___override(v___x_2838_, v_00_u03b1_1789_);
v___x_2840_ = l_Lean_Expr_app___override(v___x_2839_, v_00_u03b1_1789_);
v___x_2841_ = l_Lean_Expr_app___override(v___x_2840_, v_00_u03b1_1789_);
v___x_2842_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___lam__6___closed__2));
lean_inc_ref_n(v___x_1802_, 2);
v___x_2843_ = l_Lean_Expr_const___override(v___x_2842_, v___x_1802_);
v___x_2844_ = l_Lean_Expr_app___override(v___x_2843_, v_00_u03b1_1789_);
v___x_2845_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__84));
v___x_2846_ = l_Lean_Expr_const___override(v___x_2845_, v___x_1802_);
v___x_2847_ = l_Lean_Expr_app___override(v___x_2846_, v_00_u03b1_1789_);
v___x_2848_ = l_Lean_Expr_app___override(v___x_2847_, v___x_2767_);
v___x_2849_ = l_Lean_Expr_app___override(v___x_2844_, v___x_2848_);
v___x_2850_ = l_Lean_Expr_app___override(v___x_2841_, v___x_2849_);
lean_inc_ref(v_cancelled_2825_);
v___x_2851_ = l_Lean_Expr_app___override(v___x_2850_, v_cancelled_2825_);
lean_inc_ref(v_cancelled_2832_);
v___x_2852_ = l_Lean_Expr_app___override(v___x_2851_, v_cancelled_2832_);
v___x_2853_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__86));
v___x_2854_ = l_Lean_Expr_const___override(v___x_2853_, v___x_1802_);
v___x_2855_ = l_Lean_Expr_app___override(v___x_2854_, v_00_u03b1_1789_);
v___x_2856_ = l_Lean_Expr_app___override(v___x_2855_, v___x_1829_);
v___x_2857_ = l_Lean_Expr_app___override(v___x_2856_, v_v_x27_1792_);
v___x_2858_ = l_Lean_Expr_app___override(v___x_2857_, v_fst_2821_);
v___x_2859_ = l_Lean_Expr_app___override(v___x_2858_, v_fst_2822_);
v___x_2860_ = l_Lean_Expr_app___override(v___x_2859_, v_cancelled_2825_);
v___x_2861_ = l_Lean_Expr_app___override(v___x_2860_, v_cancelled_2832_);
v___x_2862_ = l_Lean_Expr_app___override(v___x_2861_, v_pf_2826_);
v___x_2863_ = l_Lean_Expr_app___override(v___x_2862_, v_pf_2833_);
if (v_isShared_2836_ == 0)
{
lean_ctor_set(v___x_2835_, 1, v___x_2863_);
lean_ctor_set(v___x_2835_, 0, v___x_2852_);
v___x_2865_ = v___x_2835_;
goto v_reusejp_2864_;
}
else
{
lean_object* v_reuseFailAlloc_2869_; 
v_reuseFailAlloc_2869_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2869_, 0, v___x_2852_);
lean_ctor_set(v_reuseFailAlloc_2869_, 1, v___x_2863_);
v___x_2865_ = v_reuseFailAlloc_2869_;
goto v_reusejp_2864_;
}
v_reusejp_2864_:
{
lean_object* v___x_2867_; 
if (v_isShared_2831_ == 0)
{
lean_ctor_set(v___x_2830_, 0, v___x_2865_);
v___x_2867_ = v___x_2830_;
goto v_reusejp_2866_;
}
else
{
lean_object* v_reuseFailAlloc_2868_; 
v_reuseFailAlloc_2868_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2868_, 0, v___x_2865_);
v___x_2867_ = v_reuseFailAlloc_2868_;
goto v_reusejp_2866_;
}
v_reusejp_2866_:
{
return v___x_2867_;
}
}
}
}
}
else
{
lean_dec_ref(v_pf_2826_);
lean_dec_ref(v_cancelled_2825_);
lean_dec(v_fst_2822_);
lean_dec(v_fst_2821_);
lean_dec_ref_known(v___x_2772_, 2);
lean_dec_ref(v___x_2767_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec_ref(v_00_u03b1_1789_);
return v___x_2827_;
}
}
else
{
lean_dec(v_fst_2822_);
lean_dec(v_fst_2821_);
lean_dec(v_right_2812_);
lean_dec_ref_known(v___x_2772_, 2);
lean_dec_ref(v___x_2767_);
lean_dec_ref(v___x_1829_);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
return v___x_2823_;
}
}
}
else
{
lean_object* v_a_2872_; lean_object* v___x_2874_; uint8_t v_isShared_2875_; uint8_t v_isSharedCheck_2879_; 
lean_dec_ref_known(v_t_1793_, 3);
lean_dec_ref(v___f_2810_);
lean_dec_ref(v___f_2807_);
lean_dec_ref(v___f_2804_);
lean_dec_ref(v___f_2800_);
lean_dec_ref_known(v___x_2795_, 2);
lean_dec_ref(v___x_2786_);
lean_dec_ref(v___x_2781_);
lean_dec_ref_known(v___x_2772_, 2);
lean_dec_ref(v___x_2767_);
lean_dec_ref(v___x_2766_);
lean_dec_ref(v___x_2764_);
lean_dec_ref(v_amwo_1832_);
lean_dec_ref(v___x_1830_);
lean_dec_ref(v___x_1829_);
lean_dec_ref(v___x_1828_);
lean_dec_ref_known(v___x_1822_, 2);
lean_dec_ref_known(v___x_1802_, 2);
lean_dec_ref(v_e_1794_);
lean_dec_ref(v_v_x27_1792_);
lean_dec(v_v_1791_);
lean_dec_ref(v_s_u03b1_1790_);
lean_dec_ref(v_00_u03b1_1789_);
lean_dec(v_u_1788_);
v_a_2872_ = lean_ctor_get(v___x_2816_, 0);
v_isSharedCheck_2879_ = !lean_is_exclusive(v___x_2816_);
if (v_isSharedCheck_2879_ == 0)
{
v___x_2874_ = v___x_2816_;
v_isShared_2875_ = v_isSharedCheck_2879_;
goto v_resetjp_2873_;
}
else
{
lean_inc(v_a_2872_);
lean_dec(v___x_2816_);
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
}
else
{
lean_dec_ref_known(v___x_2796_, 1);
lean_dec_ref(v___x_2767_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_2880_; lean_object* v_right_2881_; 
v_left_2880_ = lean_ctor_get(v_t_1793_, 1);
v_right_2881_ = lean_ctor_get(v_t_1793_, 2);
lean_inc(v_right_2881_);
lean_inc(v_left_2880_);
v___y_2645_ = v___x_2757_;
v___y_2646_ = v___y_2742_;
v___y_2647_ = v___x_2766_;
v___y_2648_ = v___y_2740_;
v___y_2649_ = v___y_2741_;
v___y_2650_ = v___y_2739_;
v___y_2651_ = v___x_2764_;
v___y_2652_ = v___x_2795_;
v___y_2653_ = v___x_2781_;
v___y_2654_ = v___x_2801_;
v___y_2655_ = v___x_2772_;
v___y_2656_ = v___x_2786_;
v___y_2657_ = v___f_2807_;
v___y_2658_ = v___f_2804_;
v___y_2659_ = v___f_2800_;
v___y_2660_ = v___f_2810_;
v_lhs_2661_ = v_left_2880_;
v_rhs_2662_ = v_right_2881_;
goto v___jp_2644_;
}
else
{
lean_dec_ref(v___f_2810_);
lean_dec_ref(v___x_1830_);
if (lean_obj_tag(v_t_1793_) == 1)
{
lean_object* v_left_2882_; 
v_left_2882_ = lean_ctor_get(v_t_1793_, 1);
if (lean_obj_tag(v_left_2882_) == 1)
{
lean_object* v_right_2883_; lean_object* v_value_2884_; 
v_right_2883_ = lean_ctor_get(v_t_1793_, 2);
v_value_2884_ = lean_ctor_get(v_left_2882_, 0);
lean_inc(v_right_2883_);
lean_inc(v_value_2884_);
lean_inc_ref(v_left_2882_);
lean_inc_ref(v___x_2766_);
v___y_2590_ = v___x_2757_;
v___y_2591_ = v___x_2766_;
v___y_2592_ = v___x_2764_;
v___y_2593_ = v___x_2795_;
v___y_2594_ = v___x_2781_;
v___y_2595_ = v___x_2801_;
v___y_2596_ = v___x_2772_;
v___y_2597_ = v___x_2786_;
v___y_2598_ = v___f_2807_;
v___y_2599_ = v___x_2766_;
v___y_2600_ = v___f_2804_;
v___y_2601_ = v___f_2800_;
v_lhs_2602_ = v_left_2882_;
v_ln_2603_ = v_value_2884_;
v_rhs_2604_ = v_right_2883_;
v___y_2605_ = v___y_2739_;
v___y_2606_ = v___y_2740_;
v___y_2607_ = v___y_2741_;
v___y_2608_ = v___y_2742_;
goto v___jp_2589_;
}
else
{
lean_dec_ref(v___f_2807_);
lean_inc_ref(v___x_2766_);
v___y_2477_ = v___x_2757_;
v___y_2478_ = v___x_2766_;
v___y_2479_ = v___x_2764_;
v___y_2480_ = v___x_2795_;
v___y_2481_ = v___x_2801_;
v___y_2482_ = v___x_2781_;
v___y_2483_ = v___x_2772_;
v___y_2484_ = v___x_2786_;
v___y_2485_ = v___x_2766_;
v___y_2486_ = v___f_2804_;
v___y_2487_ = v___f_2800_;
v___y_2488_ = v___y_2739_;
v___y_2489_ = v___y_2740_;
v___y_2490_ = v___y_2741_;
v___y_2491_ = v___y_2742_;
goto v___jp_2476_;
}
}
else
{
lean_dec_ref(v___f_2807_);
lean_inc_ref(v___x_2766_);
v___y_2477_ = v___x_2757_;
v___y_2478_ = v___x_2766_;
v___y_2479_ = v___x_2764_;
v___y_2480_ = v___x_2795_;
v___y_2481_ = v___x_2801_;
v___y_2482_ = v___x_2781_;
v___y_2483_ = v___x_2772_;
v___y_2484_ = v___x_2786_;
v___y_2485_ = v___x_2766_;
v___y_2486_ = v___f_2804_;
v___y_2487_ = v___f_2800_;
v___y_2488_ = v___y_2739_;
v___y_2489_ = v___y_2740_;
v___y_2490_ = v___y_2741_;
v___y_2491_ = v___y_2742_;
goto v___jp_2476_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___boxed(lean_object* v_u_2905_, lean_object* v_00_u03b1_2906_, lean_object* v_s_u03b1_2907_, lean_object* v_v_2908_, lean_object* v_v_x27_2909_, lean_object* v_t_2910_, lean_object* v_e_2911_, lean_object* v_a_2912_, lean_object* v_a_2913_, lean_object* v_a_2914_, lean_object* v_a_2915_, lean_object* v_a_2916_){
_start:
{
lean_object* v_res_2917_; 
v_res_2917_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_u_2905_, v_00_u03b1_2906_, v_s_u03b1_2907_, v_v_2908_, v_v_x27_2909_, v_t_2910_, v_e_2911_, v_a_2912_, v_a_2913_, v_a_2914_, v_a_2915_);
lean_dec(v_a_2915_);
lean_dec_ref(v_a_2914_);
lean_dec(v_a_2913_);
lean_dec_ref(v_a_2912_);
return v_res_2917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0(lean_object* v_cls_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_, lean_object* v___y_2934_, lean_object* v___y_2935_){
_start:
{
lean_object* v_options_2937_; uint8_t v_hasTrace_2938_; 
v_options_2937_ = lean_ctor_get(v___y_2934_, 2);
v_hasTrace_2938_ = lean_ctor_get_uint8(v_options_2937_, sizeof(void*)*1);
if (v_hasTrace_2938_ == 0)
{
lean_object* v___x_2939_; lean_object* v___x_2940_; 
lean_dec(v_cls_2931_);
v___x_2939_ = lean_box(v_hasTrace_2938_);
v___x_2940_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2940_, 0, v___x_2939_);
return v___x_2940_;
}
else
{
lean_object* v_inheritedTraceOptions_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; uint8_t v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; 
v_inheritedTraceOptions_2941_ = lean_ctor_get(v___y_2934_, 13);
v___x_2942_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__55));
v___x_2943_ = l_Lean_Name_append(v___x_2942_, v_cls_2931_);
v___x_2944_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2941_, v_options_2937_, v___x_2943_);
lean_dec(v___x_2943_);
v___x_2945_ = lean_box(v___x_2944_);
v___x_2946_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2946_, 0, v___x_2945_);
return v___x_2946_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0___boxed(lean_object* v_cls_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_, lean_object* v___y_2951_, lean_object* v___y_2952_){
_start:
{
lean_object* v_res_2953_; 
v_res_2953_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0(v_cls_2947_, v___y_2948_, v___y_2949_, v___y_2950_, v___y_2951_);
lean_dec(v___y_2951_);
lean_dec_ref(v___y_2950_);
lean_dec(v___y_2949_);
lean_dec_ref(v___y_2948_);
return v_res_2953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1(lean_object* v_fst_2956_, lean_object* v_a_2957_, lean_object* v___x_2958_, lean_object* v___x_2959_, lean_object* v_proof_x3f_2960_, lean_object* v_proof_x3f_2961_, lean_object* v_____r_2962_, lean_object* v___y_2963_, lean_object* v___y_2964_, lean_object* v___y_2965_, lean_object* v___y_2966_){
_start:
{
lean_object* v_pf_x27_2969_; lean_object* v_pfSimp_2974_; 
if (lean_obj_tag(v_proof_x3f_2960_) == 0)
{
if (lean_obj_tag(v_proof_x3f_2961_) == 0)
{
lean_object* v_pf_2993_; 
lean_dec_ref(v___x_2958_);
v_pf_2993_ = lean_ctor_get(v_a_2957_, 1);
lean_inc_ref(v_pf_2993_);
lean_dec_ref(v_a_2957_);
v_pf_x27_2969_ = v_pf_2993_;
goto v___jp_2968_;
}
else
{
lean_object* v_val_2994_; 
v_val_2994_ = lean_ctor_get(v_proof_x3f_2961_, 0);
lean_inc(v_val_2994_);
lean_dec_ref_known(v_proof_x3f_2961_, 1);
v_pfSimp_2974_ = v_val_2994_;
goto v___jp_2973_;
}
}
else
{
if (lean_obj_tag(v_proof_x3f_2961_) == 0)
{
lean_object* v_val_2995_; 
v_val_2995_ = lean_ctor_get(v_proof_x3f_2960_, 0);
lean_inc(v_val_2995_);
lean_dec_ref_known(v_proof_x3f_2960_, 1);
v_pfSimp_2974_ = v_val_2995_;
goto v___jp_2973_;
}
else
{
lean_object* v_val_2996_; lean_object* v_val_2997_; lean_object* v_pf_2998_; lean_object* v___x_2999_; lean_object* v___x_3000_; lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; lean_object* v___x_3005_; lean_object* v___x_3006_; lean_object* v___x_3007_; lean_object* v___x_3008_; 
v_val_2996_ = lean_ctor_get(v_proof_x3f_2960_, 0);
lean_inc(v_val_2996_);
lean_dec_ref_known(v_proof_x3f_2960_, 1);
v_val_2997_ = lean_ctor_get(v_proof_x3f_2961_, 0);
lean_inc(v_val_2997_);
lean_dec_ref_known(v_proof_x3f_2961_, 1);
v_pf_2998_ = lean_ctor_get(v_a_2957_, 1);
lean_inc_ref(v_pf_2998_);
lean_dec_ref(v_a_2957_);
v___x_2999_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_3000_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_3001_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__1));
v___x_3002_ = l_Lean_Name_mkStr4(v___x_2999_, v___x_3000_, v___x_2958_, v___x_3001_);
v___x_3003_ = lean_unsigned_to_nat(3u);
v___x_3004_ = lean_mk_empty_array_with_capacity(v___x_3003_);
v___x_3005_ = lean_array_push(v___x_3004_, v_val_2996_);
v___x_3006_ = lean_array_push(v___x_3005_, v_val_2997_);
v___x_3007_ = lean_array_push(v___x_3006_, v_pf_2998_);
v___x_3008_ = l_Lean_Meta_mkAppM(v___x_3002_, v___x_3007_, v___y_2963_, v___y_2964_, v___y_2965_, v___y_2966_);
if (lean_obj_tag(v___x_3008_) == 0)
{
lean_object* v_a_3009_; 
v_a_3009_ = lean_ctor_get(v___x_3008_, 0);
lean_inc(v_a_3009_);
lean_dec_ref_known(v___x_3008_, 1);
v_pf_x27_2969_ = v_a_3009_;
goto v___jp_2968_;
}
else
{
lean_object* v_a_3010_; lean_object* v___x_3012_; uint8_t v_isShared_3013_; uint8_t v_isSharedCheck_3017_; 
lean_dec(v_fst_2956_);
v_a_3010_ = lean_ctor_get(v___x_3008_, 0);
v_isSharedCheck_3017_ = !lean_is_exclusive(v___x_3008_);
if (v_isSharedCheck_3017_ == 0)
{
v___x_3012_ = v___x_3008_;
v_isShared_3013_ = v_isSharedCheck_3017_;
goto v_resetjp_3011_;
}
else
{
lean_inc(v_a_3010_);
lean_dec(v___x_3008_);
v___x_3012_ = lean_box(0);
v_isShared_3013_ = v_isSharedCheck_3017_;
goto v_resetjp_3011_;
}
v_resetjp_3011_:
{
lean_object* v___x_3015_; 
if (v_isShared_3013_ == 0)
{
v___x_3015_ = v___x_3012_;
goto v_reusejp_3014_;
}
else
{
lean_object* v_reuseFailAlloc_3016_; 
v_reuseFailAlloc_3016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3016_, 0, v_a_3010_);
v___x_3015_ = v_reuseFailAlloc_3016_;
goto v_reusejp_3014_;
}
v_reusejp_3014_:
{
return v___x_3015_;
}
}
}
}
}
v___jp_2968_:
{
lean_object* v___x_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; 
v___x_2970_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2970_, 0, v_fst_2956_);
lean_ctor_set(v___x_2970_, 1, v_pf_x27_2969_);
v___x_2971_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2971_, 0, v___x_2970_);
v___x_2972_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2972_, 0, v___x_2971_);
return v___x_2972_;
}
v___jp_2973_:
{
lean_object* v_pf_2975_; lean_object* v___x_2976_; lean_object* v___x_2977_; lean_object* v___x_2978_; lean_object* v___x_2979_; lean_object* v___x_2980_; lean_object* v___x_2981_; lean_object* v___x_2982_; lean_object* v___x_2983_; 
v_pf_2975_ = lean_ctor_get(v_a_2957_, 1);
lean_inc_ref(v_pf_2975_);
lean_dec_ref(v_a_2957_);
v___x_2976_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__4_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_2977_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__6_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_2978_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___closed__0));
v___x_2979_ = l_Lean_Name_mkStr4(v___x_2976_, v___x_2977_, v___x_2958_, v___x_2978_);
v___x_2980_ = lean_mk_empty_array_with_capacity(v___x_2959_);
v___x_2981_ = lean_array_push(v___x_2980_, v_pfSimp_2974_);
v___x_2982_ = lean_array_push(v___x_2981_, v_pf_2975_);
v___x_2983_ = l_Lean_Meta_mkAppM(v___x_2979_, v___x_2982_, v___y_2963_, v___y_2964_, v___y_2965_, v___y_2966_);
if (lean_obj_tag(v___x_2983_) == 0)
{
lean_object* v_a_2984_; 
v_a_2984_ = lean_ctor_get(v___x_2983_, 0);
lean_inc(v_a_2984_);
lean_dec_ref_known(v___x_2983_, 1);
v_pf_x27_2969_ = v_a_2984_;
goto v___jp_2968_;
}
else
{
lean_object* v_a_2985_; lean_object* v___x_2987_; uint8_t v_isShared_2988_; uint8_t v_isSharedCheck_2992_; 
lean_dec(v_fst_2956_);
v_a_2985_ = lean_ctor_get(v___x_2983_, 0);
v_isSharedCheck_2992_ = !lean_is_exclusive(v___x_2983_);
if (v_isSharedCheck_2992_ == 0)
{
v___x_2987_ = v___x_2983_;
v_isShared_2988_ = v_isSharedCheck_2992_;
goto v_resetjp_2986_;
}
else
{
lean_inc(v_a_2985_);
lean_dec(v___x_2983_);
v___x_2987_ = lean_box(0);
v_isShared_2988_ = v_isSharedCheck_2992_;
goto v_resetjp_2986_;
}
v_resetjp_2986_:
{
lean_object* v___x_2990_; 
if (v_isShared_2988_ == 0)
{
v___x_2990_ = v___x_2987_;
goto v_reusejp_2989_;
}
else
{
lean_object* v_reuseFailAlloc_2991_; 
v_reuseFailAlloc_2991_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2991_, 0, v_a_2985_);
v___x_2990_ = v_reuseFailAlloc_2991_;
goto v_reusejp_2989_;
}
v_reusejp_2989_:
{
return v___x_2990_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___boxed(lean_object* v_fst_3018_, lean_object* v_a_3019_, lean_object* v___x_3020_, lean_object* v___x_3021_, lean_object* v_proof_x3f_3022_, lean_object* v_proof_x3f_3023_, lean_object* v_____r_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_){
_start:
{
lean_object* v_res_3030_; 
v_res_3030_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1(v_fst_3018_, v_a_3019_, v___x_3020_, v___x_3021_, v_proof_x3f_3022_, v_proof_x3f_3023_, v_____r_3024_, v___y_3025_, v___y_3026_, v___y_3027_, v___y_3028_);
lean_dec(v___y_3028_);
lean_dec_ref(v___y_3027_);
lean_dec(v___y_3026_);
lean_dec_ref(v___y_3025_);
lean_dec(v___x_3021_);
return v_res_3030_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__1(void){
_start:
{
lean_object* v___x_3032_; lean_object* v___x_3033_; 
v___x_3032_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__0));
v___x_3033_ = l_Lean_stringToMessageData(v___x_3032_);
return v___x_3033_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__3(void){
_start:
{
lean_object* v___x_3035_; lean_object* v___x_3036_; 
v___x_3035_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__2));
v___x_3036_ = l_Lean_stringToMessageData(v___x_3035_);
return v___x_3036_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__6(void){
_start:
{
lean_object* v___x_3040_; lean_object* v___x_3041_; 
v___x_3040_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__5));
v___x_3041_ = l_Lean_stringToMessageData(v___x_3040_);
return v___x_3041_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__9(void){
_start:
{
lean_object* v___x_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; 
v___x_3051_ = lean_box(0);
v___x_3052_ = lean_unsigned_to_nat(16u);
v___x_3053_ = lean_mk_array(v___x_3052_, v___x_3051_);
return v___x_3053_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__10(void){
_start:
{
lean_object* v___x_3054_; lean_object* v___x_3055_; lean_object* v___x_3056_; 
v___x_3054_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__9, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__9);
v___x_3055_ = lean_unsigned_to_nat(0u);
v___x_3056_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3056_, 0, v___x_3055_);
lean_ctor_set(v___x_3056_, 1, v___x_3054_);
return v___x_3056_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__11(void){
_start:
{
lean_object* v___x_3057_; 
v___x_3057_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3057_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__12(void){
_start:
{
lean_object* v___x_3058_; lean_object* v___x_3059_; 
v___x_3058_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__11, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__11);
v___x_3059_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3059_, 0, v___x_3058_);
return v___x_3059_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__13(void){
_start:
{
lean_object* v___x_3060_; lean_object* v___x_3061_; uint8_t v___x_3062_; lean_object* v___x_3063_; 
v___x_3060_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__12, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__12);
v___x_3061_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__10, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__10);
v___x_3062_ = 1;
v___x_3063_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_3063_, 0, v___x_3061_);
lean_ctor_set(v___x_3063_, 1, v___x_3060_);
lean_ctor_set_uint8(v___x_3063_, sizeof(void*)*2, v___x_3062_);
return v___x_3063_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__16(void){
_start:
{
lean_object* v___x_3067_; lean_object* v___x_3068_; 
v___x_3067_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__15));
v___x_3068_ = l_Lean_stringToMessageData(v___x_3067_);
return v___x_3068_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__18(void){
_start:
{
lean_object* v___x_3070_; lean_object* v___x_3071_; 
v___x_3070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__17));
v___x_3071_ = l_Lean_stringToMessageData(v___x_3070_);
return v___x_3071_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__20(void){
_start:
{
lean_object* v___x_3073_; lean_object* v___x_3074_; 
v___x_3073_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__19));
v___x_3074_ = l_Lean_stringToMessageData(v___x_3073_);
return v___x_3074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive(lean_object* v_e_3075_, lean_object* v_a_3076_, lean_object* v_a_3077_, lean_object* v_a_3078_, lean_object* v_a_3079_){
_start:
{
lean_object* v___y_3082_; lean_object* v___y_3083_; lean_object* v___y_3084_; lean_object* v___y_3085_; lean_object* v___y_3086_; lean_object* v___y_3087_; uint8_t v___y_3088_; lean_object* v___y_3099_; lean_object* v___y_3100_; lean_object* v___y_3101_; lean_object* v___y_3102_; lean_object* v___y_3103_; lean_object* v_a_3104_; lean_object* v___y_3108_; lean_object* v___y_3109_; lean_object* v___y_3110_; lean_object* v___y_3111_; lean_object* v___y_3112_; lean_object* v___y_3113_; lean_object* v___y_3125_; lean_object* v___y_3126_; lean_object* v___y_3127_; lean_object* v___y_3128_; lean_object* v___y_3129_; lean_object* v___y_3130_; lean_object* v___x_3133_; lean_object* v_cls_3134_; lean_object* v___y_3136_; lean_object* v___y_3137_; lean_object* v_expr_3138_; lean_object* v_proof_x3f_3139_; lean_object* v___y_3140_; lean_object* v___y_3141_; lean_object* v___y_3142_; lean_object* v___y_3143_; lean_object* v___y_3246_; lean_object* v___y_3247_; lean_object* v___y_3248_; lean_object* v___y_3249_; lean_object* v___y_3250_; lean_object* v___y_3300_; lean_object* v___y_3301_; lean_object* v___y_3302_; lean_object* v___y_3303_; lean_object* v___x_3332_; lean_object* v_a_3333_; uint8_t v___x_3334_; 
v___x_3133_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__0_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v_cls_3134_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn___closed__1_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_));
v___x_3332_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0(v_cls_3134_, v_a_3076_, v_a_3077_, v_a_3078_, v_a_3079_);
v_a_3333_ = lean_ctor_get(v___x_3332_, 0);
lean_inc(v_a_3333_);
lean_dec_ref(v___x_3332_);
v___x_3334_ = lean_unbox(v_a_3333_);
lean_dec(v_a_3333_);
if (v___x_3334_ == 0)
{
v___y_3300_ = v_a_3076_;
v___y_3301_ = v_a_3077_;
v___y_3302_ = v_a_3078_;
v___y_3303_ = v_a_3079_;
goto v___jp_3299_;
}
else
{
lean_object* v___x_3335_; lean_object* v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; 
v___x_3335_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__20, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__20);
lean_inc_ref(v_e_3075_);
v___x_3336_ = l_Lean_MessageData_ofExpr(v_e_3075_);
v___x_3337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3337_, 0, v___x_3335_);
lean_ctor_set(v___x_3337_, 1, v___x_3336_);
v___x_3338_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(v_cls_3134_, v___x_3337_, v_a_3076_, v_a_3077_, v_a_3078_, v_a_3079_);
if (lean_obj_tag(v___x_3338_) == 0)
{
lean_dec_ref_known(v___x_3338_, 1);
v___y_3300_ = v_a_3076_;
v___y_3301_ = v_a_3077_;
v___y_3302_ = v_a_3078_;
v___y_3303_ = v_a_3079_;
goto v___jp_3299_;
}
else
{
lean_object* v_a_3339_; lean_object* v___x_3341_; uint8_t v_isShared_3342_; uint8_t v_isSharedCheck_3346_; 
lean_dec_ref(v_e_3075_);
v_a_3339_ = lean_ctor_get(v___x_3338_, 0);
v_isSharedCheck_3346_ = !lean_is_exclusive(v___x_3338_);
if (v_isSharedCheck_3346_ == 0)
{
v___x_3341_ = v___x_3338_;
v_isShared_3342_ = v_isSharedCheck_3346_;
goto v_resetjp_3340_;
}
else
{
lean_inc(v_a_3339_);
lean_dec(v___x_3338_);
v___x_3341_ = lean_box(0);
v_isShared_3342_ = v_isSharedCheck_3346_;
goto v_resetjp_3340_;
}
v_resetjp_3340_:
{
lean_object* v___x_3344_; 
if (v_isShared_3342_ == 0)
{
v___x_3344_ = v___x_3341_;
goto v_reusejp_3343_;
}
else
{
lean_object* v_reuseFailAlloc_3345_; 
v_reuseFailAlloc_3345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3345_, 0, v_a_3339_);
v___x_3344_ = v_reuseFailAlloc_3345_;
goto v_reusejp_3343_;
}
v_reusejp_3343_:
{
return v___x_3344_;
}
}
}
}
v___jp_3081_:
{
if (v___y_3088_ == 0)
{
lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_3091_; lean_object* v___x_3092_; lean_object* v___x_3093_; lean_object* v___x_3094_; lean_object* v___x_3095_; lean_object* v___x_3096_; 
v___x_3089_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__1, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__1);
v___x_3090_ = l_Lean_MessageData_ofExpr(v___y_3085_);
v___x_3091_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3091_, 0, v___x_3089_);
lean_ctor_set(v___x_3091_, 1, v___x_3090_);
v___x_3092_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__3, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__3);
v___x_3093_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3093_, 0, v___x_3091_);
lean_ctor_set(v___x_3093_, 1, v___x_3092_);
v___x_3094_ = l_Lean_Exception_toMessageData(v___y_3086_);
v___x_3095_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3095_, 0, v___x_3093_);
lean_ctor_set(v___x_3095_, 1, v___x_3094_);
v___x_3096_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(v___x_3095_, v___y_3084_, v___y_3087_, v___y_3082_, v___y_3083_);
return v___x_3096_;
}
else
{
lean_object* v___x_3097_; 
lean_dec_ref(v___y_3085_);
v___x_3097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3097_, 0, v___y_3086_);
return v___x_3097_;
}
}
v___jp_3098_:
{
uint8_t v___x_3105_; 
v___x_3105_ = l_Lean_Exception_isInterrupt(v_a_3104_);
if (v___x_3105_ == 0)
{
uint8_t v___x_3106_; 
lean_inc_ref(v_a_3104_);
v___x_3106_ = l_Lean_Exception_isRuntime(v_a_3104_);
v___y_3082_ = v___y_3099_;
v___y_3083_ = v___y_3100_;
v___y_3084_ = v___y_3101_;
v___y_3085_ = v___y_3102_;
v___y_3086_ = v_a_3104_;
v___y_3087_ = v___y_3103_;
v___y_3088_ = v___x_3106_;
goto v___jp_3081_;
}
else
{
v___y_3082_ = v___y_3099_;
v___y_3083_ = v___y_3100_;
v___y_3084_ = v___y_3101_;
v___y_3085_ = v___y_3102_;
v___y_3086_ = v_a_3104_;
v___y_3087_ = v___y_3103_;
v___y_3088_ = v___x_3105_;
goto v___jp_3081_;
}
}
v___jp_3107_:
{
if (lean_obj_tag(v___y_3113_) == 0)
{
lean_object* v_a_3114_; lean_object* v___x_3116_; uint8_t v_isShared_3117_; uint8_t v_isSharedCheck_3122_; 
lean_dec_ref(v___y_3111_);
v_a_3114_ = lean_ctor_get(v___y_3113_, 0);
v_isSharedCheck_3122_ = !lean_is_exclusive(v___y_3113_);
if (v_isSharedCheck_3122_ == 0)
{
v___x_3116_ = v___y_3113_;
v_isShared_3117_ = v_isSharedCheck_3122_;
goto v_resetjp_3115_;
}
else
{
lean_inc(v_a_3114_);
lean_dec(v___y_3113_);
v___x_3116_ = lean_box(0);
v_isShared_3117_ = v_isSharedCheck_3122_;
goto v_resetjp_3115_;
}
v_resetjp_3115_:
{
lean_object* v_a_3118_; lean_object* v___x_3120_; 
v_a_3118_ = lean_ctor_get(v_a_3114_, 0);
lean_inc(v_a_3118_);
lean_dec(v_a_3114_);
if (v_isShared_3117_ == 0)
{
lean_ctor_set(v___x_3116_, 0, v_a_3118_);
v___x_3120_ = v___x_3116_;
goto v_reusejp_3119_;
}
else
{
lean_object* v_reuseFailAlloc_3121_; 
v_reuseFailAlloc_3121_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3121_, 0, v_a_3118_);
v___x_3120_ = v_reuseFailAlloc_3121_;
goto v_reusejp_3119_;
}
v_reusejp_3119_:
{
return v___x_3120_;
}
}
}
else
{
lean_object* v_a_3123_; 
v_a_3123_ = lean_ctor_get(v___y_3113_, 0);
lean_inc(v_a_3123_);
lean_dec_ref_known(v___y_3113_, 1);
v___y_3099_ = v___y_3108_;
v___y_3100_ = v___y_3109_;
v___y_3101_ = v___y_3110_;
v___y_3102_ = v___y_3111_;
v___y_3103_ = v___y_3112_;
v_a_3104_ = v_a_3123_;
goto v___jp_3098_;
}
}
v___jp_3124_:
{
lean_object* v___x_3131_; lean_object* v___x_3132_; 
v___x_3131_ = lean_box(0);
lean_inc(v___y_3126_);
lean_inc_ref(v___y_3125_);
lean_inc(v___y_3129_);
lean_inc_ref(v___y_3127_);
v___x_3132_ = lean_apply_6(v___y_3130_, v___x_3131_, v___y_3127_, v___y_3129_, v___y_3125_, v___y_3126_, lean_box(0));
v___y_3108_ = v___y_3125_;
v___y_3109_ = v___y_3126_;
v___y_3110_ = v___y_3127_;
v___y_3111_ = v___y_3128_;
v___y_3112_ = v___y_3129_;
v___y_3113_ = v___x_3132_;
goto v___jp_3107_;
}
v___jp_3135_:
{
lean_object* v___x_3144_; lean_object* v_fst_3145_; lean_object* v_snd_3146_; lean_object* v___x_3147_; 
lean_inc_ref(v_expr_3138_);
v___x_3144_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCancelFactor(v_expr_3138_);
v_fst_3145_ = lean_ctor_get(v___x_3144_, 0);
lean_inc(v_fst_3145_);
v_snd_3146_ = lean_ctor_get(v___x_3144_, 1);
lean_inc(v_snd_3146_);
lean_dec_ref(v___x_3144_);
v___x_3147_ = lp_mathlib_Qq_inferTypeQ_x27(v_expr_3138_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_);
if (lean_obj_tag(v___x_3147_) == 0)
{
lean_object* v_a_3148_; lean_object* v_snd_3149_; lean_object* v_fst_3150_; lean_object* v___x_3152_; uint8_t v_isShared_3153_; uint8_t v_isSharedCheck_3236_; 
v_a_3148_ = lean_ctor_get(v___x_3147_, 0);
lean_inc(v_a_3148_);
lean_dec_ref_known(v___x_3147_, 1);
v_snd_3149_ = lean_ctor_get(v_a_3148_, 1);
v_fst_3150_ = lean_ctor_get(v_a_3148_, 0);
v_isSharedCheck_3236_ = !lean_is_exclusive(v_a_3148_);
if (v_isSharedCheck_3236_ == 0)
{
v___x_3152_ = v_a_3148_;
v_isShared_3153_ = v_isSharedCheck_3236_;
goto v_resetjp_3151_;
}
else
{
lean_inc(v_snd_3149_);
lean_inc(v_fst_3150_);
lean_dec(v_a_3148_);
v___x_3152_ = lean_box(0);
v_isShared_3153_ = v_isSharedCheck_3236_;
goto v_resetjp_3151_;
}
v_resetjp_3151_:
{
lean_object* v_fst_3154_; lean_object* v_snd_3155_; lean_object* v___x_3157_; uint8_t v_isShared_3158_; uint8_t v_isSharedCheck_3235_; 
v_fst_3154_ = lean_ctor_get(v_snd_3149_, 0);
v_snd_3155_ = lean_ctor_get(v_snd_3149_, 1);
v_isSharedCheck_3235_ = !lean_is_exclusive(v_snd_3149_);
if (v_isSharedCheck_3235_ == 0)
{
v___x_3157_ = v_snd_3149_;
v_isShared_3158_ = v_isSharedCheck_3235_;
goto v_resetjp_3156_;
}
else
{
lean_inc(v_snd_3155_);
lean_inc(v_fst_3154_);
lean_dec(v_snd_3149_);
v___x_3157_ = lean_box(0);
v_isShared_3158_ = v_isSharedCheck_3235_;
goto v_resetjp_3156_;
}
v_resetjp_3156_:
{
lean_object* v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; lean_object* v___x_3163_; 
lean_inc_n(v_fst_3150_, 2);
v___x_3159_ = l_Lean_Level_succ___override(v_fst_3150_);
v___x_3160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__4));
v___x_3161_ = lean_box(0);
if (v_isShared_3158_ == 0)
{
lean_ctor_set_tag(v___x_3157_, 1);
lean_ctor_set(v___x_3157_, 1, v___x_3161_);
lean_ctor_set(v___x_3157_, 0, v_fst_3150_);
v___x_3163_ = v___x_3157_;
goto v_reusejp_3162_;
}
else
{
lean_object* v_reuseFailAlloc_3234_; 
v_reuseFailAlloc_3234_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3234_, 0, v_fst_3150_);
lean_ctor_set(v_reuseFailAlloc_3234_, 1, v___x_3161_);
v___x_3163_ = v_reuseFailAlloc_3234_;
goto v_reusejp_3162_;
}
v_reusejp_3162_:
{
lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; 
lean_inc_ref(v___x_3163_);
v___x_3164_ = l_Lean_Expr_const___override(v___x_3160_, v___x_3163_);
lean_inc(v_fst_3154_);
v___x_3165_ = l_Lean_Expr_app___override(v___x_3164_, v_fst_3154_);
v___x_3166_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_3165_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_);
if (lean_obj_tag(v___x_3166_) == 0)
{
lean_object* v_a_3167_; lean_object* v___x_3168_; lean_object* v___x_3169_; lean_object* v___x_3170_; lean_object* v___x_3171_; lean_object* v___x_3173_; 
v_a_3167_ = lean_ctor_get(v___x_3166_, 0);
lean_inc(v_a_3167_);
lean_dec_ref_known(v___x_3166_, 1);
v___x_3168_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__1));
lean_inc_ref(v___x_3163_);
v___x_3169_ = l_Lean_Expr_const___override(v___x_3168_, v___x_3163_);
lean_inc(v_fst_3154_);
v___x_3170_ = l_Lean_Expr_app___override(v___x_3169_, v_fst_3154_);
v___x_3171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__12));
if (v_isShared_3153_ == 0)
{
lean_ctor_set_tag(v___x_3152_, 1);
lean_ctor_set(v___x_3152_, 1, v___x_3161_);
lean_ctor_set(v___x_3152_, 0, v___x_3159_);
v___x_3173_ = v___x_3152_;
goto v_reusejp_3172_;
}
else
{
lean_object* v_reuseFailAlloc_3225_; 
v_reuseFailAlloc_3225_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3225_, 0, v___x_3159_);
lean_ctor_set(v_reuseFailAlloc_3225_, 1, v___x_3161_);
v___x_3173_ = v_reuseFailAlloc_3225_;
goto v_reusejp_3172_;
}
v_reusejp_3172_:
{
lean_object* v___x_3174_; lean_object* v___x_3175_; lean_object* v___x_3176_; lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; lean_object* v___x_3180_; lean_object* v___x_3181_; lean_object* v___x_3182_; lean_object* v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; 
v___x_3174_ = l_Lean_Expr_const___override(v___x_3171_, v___x_3173_);
v___x_3175_ = l_Lean_Expr_app___override(v___x_3174_, v___x_3170_);
v___x_3176_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__4));
lean_inc_ref_n(v___x_3163_, 3);
v___x_3177_ = l_Lean_Expr_const___override(v___x_3176_, v___x_3163_);
lean_inc_n(v_fst_3154_, 5);
v___x_3178_ = l_Lean_Expr_app___override(v___x_3177_, v_fst_3154_);
v___x_3179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__7));
v___x_3180_ = l_Lean_Expr_const___override(v___x_3179_, v___x_3163_);
v___x_3181_ = l_Lean_Expr_app___override(v___x_3180_, v_fst_3154_);
v___x_3182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__10));
v___x_3183_ = l_Lean_Expr_const___override(v___x_3182_, v___x_3163_);
v___x_3184_ = l_Lean_Expr_app___override(v___x_3183_, v_fst_3154_);
v___x_3185_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__15));
v___x_3186_ = l_Lean_Expr_const___override(v___x_3185_, v___x_3163_);
v___x_3187_ = l_Lean_Expr_app___override(v___x_3186_, v_fst_3154_);
lean_inc(v_a_3167_);
v___x_3188_ = l_Lean_Expr_app___override(v___x_3187_, v_a_3167_);
v___x_3189_ = l_Lean_Expr_app___override(v___x_3184_, v___x_3188_);
v___x_3190_ = l_Lean_Expr_app___override(v___x_3181_, v___x_3189_);
v___x_3191_ = l_Lean_Expr_app___override(v___x_3178_, v___x_3190_);
v___x_3192_ = l_Lean_Expr_app___override(v___x_3175_, v___x_3191_);
lean_inc(v_fst_3145_);
v___x_3193_ = l_Lean_mkRawNatLit(v_fst_3145_);
lean_inc(v_fst_3150_);
v___x_3194_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_fst_3150_, v_fst_3154_, v___x_3192_, v___x_3193_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_);
if (lean_obj_tag(v___x_3194_) == 0)
{
lean_object* v_a_3195_; lean_object* v_fst_3196_; lean_object* v___x_3198_; uint8_t v_isShared_3199_; uint8_t v_isSharedCheck_3222_; 
v_a_3195_ = lean_ctor_get(v___x_3194_, 0);
lean_inc(v_a_3195_);
lean_dec_ref_known(v___x_3194_, 1);
v_fst_3196_ = lean_ctor_get(v_a_3195_, 0);
v_isSharedCheck_3222_ = !lean_is_exclusive(v_a_3195_);
if (v_isSharedCheck_3222_ == 0)
{
lean_object* v_unused_3223_; 
v_unused_3223_ = lean_ctor_get(v_a_3195_, 1);
lean_dec(v_unused_3223_);
v___x_3198_ = v_a_3195_;
v_isShared_3199_ = v_isSharedCheck_3222_;
goto v_resetjp_3197_;
}
else
{
lean_inc(v_fst_3196_);
lean_dec(v_a_3195_);
v___x_3198_ = lean_box(0);
v_isShared_3199_ = v_isSharedCheck_3222_;
goto v_resetjp_3197_;
}
v_resetjp_3197_:
{
lean_object* v___x_3200_; 
lean_inc(v_snd_3155_);
lean_inc(v_fst_3145_);
v___x_3200_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf(v_fst_3150_, v_fst_3154_, v_a_3167_, v_fst_3145_, v_fst_3196_, v_snd_3146_, v_snd_3155_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_);
if (lean_obj_tag(v___x_3200_) == 0)
{
lean_object* v_options_3201_; lean_object* v_a_3202_; lean_object* v_inheritedTraceOptions_3203_; uint8_t v_hasTrace_3204_; lean_object* v___f_3205_; 
v_options_3201_ = lean_ctor_get(v___y_3142_, 2);
v_a_3202_ = lean_ctor_get(v___x_3200_, 0);
lean_inc_n(v_a_3202_, 2);
lean_dec_ref_known(v___x_3200_, 1);
v_inheritedTraceOptions_3203_ = lean_ctor_get(v___y_3142_, 13);
v_hasTrace_3204_ = lean_ctor_get_uint8(v_options_3201_, sizeof(void*)*1);
lean_inc(v_proof_x3f_3139_);
lean_inc(v___y_3136_);
lean_inc(v___y_3137_);
lean_inc(v_fst_3145_);
v___f_3205_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1___boxed), 12, 6);
lean_closure_set(v___f_3205_, 0, v_fst_3145_);
lean_closure_set(v___f_3205_, 1, v_a_3202_);
lean_closure_set(v___f_3205_, 2, v___x_3133_);
lean_closure_set(v___f_3205_, 3, v___y_3137_);
lean_closure_set(v___f_3205_, 4, v___y_3136_);
lean_closure_set(v___f_3205_, 5, v_proof_x3f_3139_);
if (v_hasTrace_3204_ == 0)
{
lean_dec(v_a_3202_);
lean_del_object(v___x_3198_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v___y_3125_ = v___y_3142_;
v___y_3126_ = v___y_3143_;
v___y_3127_ = v___y_3140_;
v___y_3128_ = v_snd_3155_;
v___y_3129_ = v___y_3141_;
v___y_3130_ = v___f_3205_;
goto v___jp_3124_;
}
else
{
lean_object* v___x_3206_; uint8_t v___x_3207_; 
v___x_3206_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__56);
v___x_3207_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3203_, v_options_3201_, v___x_3206_);
if (v___x_3207_ == 0)
{
lean_dec(v_a_3202_);
lean_del_object(v___x_3198_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v___y_3125_ = v___y_3142_;
v___y_3126_ = v___y_3143_;
v___y_3127_ = v___y_3140_;
v___y_3128_ = v_snd_3155_;
v___y_3129_ = v___y_3141_;
v___y_3130_ = v___f_3205_;
goto v___jp_3124_;
}
else
{
lean_object* v_pf_3208_; lean_object* v___x_3209_; 
lean_dec_ref(v___f_3205_);
v_pf_3208_ = lean_ctor_get(v_a_3202_, 1);
lean_inc(v___y_3143_);
lean_inc_ref(v___y_3142_);
lean_inc(v___y_3141_);
lean_inc_ref(v___y_3140_);
lean_inc_ref(v_pf_3208_);
v___x_3209_ = lean_infer_type(v_pf_3208_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_);
if (lean_obj_tag(v___x_3209_) == 0)
{
lean_object* v_a_3210_; lean_object* v___x_3211_; lean_object* v___x_3212_; lean_object* v___x_3214_; 
v_a_3210_ = lean_ctor_get(v___x_3209_, 0);
lean_inc(v_a_3210_);
lean_dec_ref_known(v___x_3209_, 1);
v___x_3211_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__6, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__6);
v___x_3212_ = l_Lean_MessageData_ofExpr(v_a_3210_);
if (v_isShared_3199_ == 0)
{
lean_ctor_set_tag(v___x_3198_, 7);
lean_ctor_set(v___x_3198_, 1, v___x_3212_);
lean_ctor_set(v___x_3198_, 0, v___x_3211_);
v___x_3214_ = v___x_3198_;
goto v_reusejp_3213_;
}
else
{
lean_object* v_reuseFailAlloc_3219_; 
v_reuseFailAlloc_3219_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3219_, 0, v___x_3211_);
lean_ctor_set(v_reuseFailAlloc_3219_, 1, v___x_3212_);
v___x_3214_ = v_reuseFailAlloc_3219_;
goto v_reusejp_3213_;
}
v_reusejp_3213_:
{
lean_object* v___x_3215_; 
v___x_3215_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(v_cls_3134_, v___x_3214_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_);
if (lean_obj_tag(v___x_3215_) == 0)
{
lean_object* v_a_3216_; lean_object* v___x_3217_; 
v_a_3216_ = lean_ctor_get(v___x_3215_, 0);
lean_inc(v_a_3216_);
lean_dec_ref_known(v___x_3215_, 1);
v___x_3217_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__1(v_fst_3145_, v_a_3202_, v___x_3133_, v___y_3137_, v___y_3136_, v_proof_x3f_3139_, v_a_3216_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_);
lean_dec(v___y_3137_);
v___y_3108_ = v___y_3142_;
v___y_3109_ = v___y_3143_;
v___y_3110_ = v___y_3140_;
v___y_3111_ = v_snd_3155_;
v___y_3112_ = v___y_3141_;
v___y_3113_ = v___x_3217_;
goto v___jp_3107_;
}
else
{
lean_object* v_a_3218_; 
lean_dec(v_a_3202_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v_a_3218_ = lean_ctor_get(v___x_3215_, 0);
lean_inc(v_a_3218_);
lean_dec_ref_known(v___x_3215_, 1);
v___y_3099_ = v___y_3142_;
v___y_3100_ = v___y_3143_;
v___y_3101_ = v___y_3140_;
v___y_3102_ = v_snd_3155_;
v___y_3103_ = v___y_3141_;
v_a_3104_ = v_a_3218_;
goto v___jp_3098_;
}
}
}
else
{
lean_object* v_a_3220_; 
lean_dec(v_a_3202_);
lean_del_object(v___x_3198_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v_a_3220_ = lean_ctor_get(v___x_3209_, 0);
lean_inc(v_a_3220_);
lean_dec_ref_known(v___x_3209_, 1);
v___y_3099_ = v___y_3142_;
v___y_3100_ = v___y_3143_;
v___y_3101_ = v___y_3140_;
v___y_3102_ = v_snd_3155_;
v___y_3103_ = v___y_3141_;
v_a_3104_ = v_a_3220_;
goto v___jp_3098_;
}
}
}
}
else
{
lean_object* v_a_3221_; 
lean_del_object(v___x_3198_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v_a_3221_ = lean_ctor_get(v___x_3200_, 0);
lean_inc(v_a_3221_);
lean_dec_ref_known(v___x_3200_, 1);
v___y_3099_ = v___y_3142_;
v___y_3100_ = v___y_3143_;
v___y_3101_ = v___y_3140_;
v___y_3102_ = v_snd_3155_;
v___y_3103_ = v___y_3141_;
v_a_3104_ = v_a_3221_;
goto v___jp_3098_;
}
}
}
else
{
lean_object* v_a_3224_; 
lean_dec(v_a_3167_);
lean_dec(v_fst_3154_);
lean_dec(v_fst_3150_);
lean_dec(v_snd_3146_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v_a_3224_ = lean_ctor_get(v___x_3194_, 0);
lean_inc(v_a_3224_);
lean_dec_ref_known(v___x_3194_, 1);
v___y_3099_ = v___y_3142_;
v___y_3100_ = v___y_3143_;
v___y_3101_ = v___y_3140_;
v___y_3102_ = v_snd_3155_;
v___y_3103_ = v___y_3141_;
v_a_3104_ = v_a_3224_;
goto v___jp_3098_;
}
}
}
else
{
lean_object* v_a_3226_; lean_object* v___x_3228_; uint8_t v_isShared_3229_; uint8_t v_isSharedCheck_3233_; 
lean_dec_ref(v___x_3163_);
lean_dec(v___x_3159_);
lean_dec(v_snd_3155_);
lean_dec(v_fst_3154_);
lean_del_object(v___x_3152_);
lean_dec(v_fst_3150_);
lean_dec(v_snd_3146_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v_a_3226_ = lean_ctor_get(v___x_3166_, 0);
v_isSharedCheck_3233_ = !lean_is_exclusive(v___x_3166_);
if (v_isSharedCheck_3233_ == 0)
{
v___x_3228_ = v___x_3166_;
v_isShared_3229_ = v_isSharedCheck_3233_;
goto v_resetjp_3227_;
}
else
{
lean_inc(v_a_3226_);
lean_dec(v___x_3166_);
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
}
}
else
{
lean_object* v_a_3237_; lean_object* v___x_3239_; uint8_t v_isShared_3240_; uint8_t v_isSharedCheck_3244_; 
lean_dec(v_snd_3146_);
lean_dec(v_fst_3145_);
lean_dec(v_proof_x3f_3139_);
lean_dec(v___y_3137_);
lean_dec(v___y_3136_);
v_a_3237_ = lean_ctor_get(v___x_3147_, 0);
v_isSharedCheck_3244_ = !lean_is_exclusive(v___x_3147_);
if (v_isSharedCheck_3244_ == 0)
{
v___x_3239_ = v___x_3147_;
v_isShared_3240_ = v_isSharedCheck_3244_;
goto v_resetjp_3238_;
}
else
{
lean_inc(v_a_3237_);
lean_dec(v___x_3147_);
v___x_3239_ = lean_box(0);
v_isShared_3240_ = v_isSharedCheck_3244_;
goto v_resetjp_3238_;
}
v_resetjp_3238_:
{
lean_object* v___x_3242_; 
if (v_isShared_3240_ == 0)
{
v___x_3242_ = v___x_3239_;
goto v_reusejp_3241_;
}
else
{
lean_object* v_reuseFailAlloc_3243_; 
v_reuseFailAlloc_3243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3243_, 0, v_a_3237_);
v___x_3242_ = v_reuseFailAlloc_3243_;
goto v_reusejp_3241_;
}
v_reusejp_3241_:
{
return v___x_3242_;
}
}
}
}
v___jp_3245_:
{
lean_object* v___x_3251_; uint8_t v___x_3252_; lean_object* v___x_3253_; lean_object* v___x_3254_; lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3257_; 
v___x_3251_ = lean_unsigned_to_nat(2u);
v___x_3252_ = 0;
v___x_3253_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__7));
v___x_3254_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__8));
v___x_3255_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__13, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__13);
v___x_3256_ = l_Lean_Options_empty;
v___x_3257_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_3253_, v___x_3254_, v___x_3255_, v___x_3256_, v___y_3247_, v___y_3249_, v___y_3250_);
if (lean_obj_tag(v___x_3257_) == 0)
{
lean_object* v_a_3258_; lean_object* v_expr_3259_; lean_object* v_proof_x3f_3260_; lean_object* v___x_3261_; lean_object* v___x_3262_; 
v_a_3258_ = lean_ctor_get(v___x_3257_, 0);
lean_inc(v_a_3258_);
lean_dec_ref_known(v___x_3257_, 1);
v_expr_3259_ = lean_ctor_get(v___y_3246_, 0);
lean_inc_ref(v_expr_3259_);
v_proof_x3f_3260_ = lean_ctor_get(v___y_3246_, 1);
lean_inc(v_proof_x3f_3260_);
lean_dec_ref(v___y_3246_);
v___x_3261_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__14));
v___x_3262_ = lp_mathlib_Mathlib_Meta_NormNum_deriveSimp(v_a_3258_, v___x_3261_, v___x_3252_, v_expr_3259_, v___y_3247_, v___y_3248_, v___y_3249_, v___y_3250_);
if (lean_obj_tag(v___x_3262_) == 0)
{
lean_object* v_a_3263_; lean_object* v___x_3264_; lean_object* v_a_3265_; uint8_t v___x_3266_; 
v_a_3263_ = lean_ctor_get(v___x_3262_, 0);
lean_inc(v_a_3263_);
lean_dec_ref_known(v___x_3262_, 1);
v___x_3264_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0(v_cls_3134_, v___y_3247_, v___y_3248_, v___y_3249_, v___y_3250_);
v_a_3265_ = lean_ctor_get(v___x_3264_, 0);
lean_inc(v_a_3265_);
lean_dec_ref(v___x_3264_);
v___x_3266_ = lean_unbox(v_a_3265_);
lean_dec(v_a_3265_);
if (v___x_3266_ == 0)
{
lean_object* v_expr_3267_; lean_object* v_proof_x3f_3268_; 
v_expr_3267_ = lean_ctor_get(v_a_3263_, 0);
lean_inc_ref(v_expr_3267_);
v_proof_x3f_3268_ = lean_ctor_get(v_a_3263_, 1);
lean_inc(v_proof_x3f_3268_);
lean_dec(v_a_3263_);
v___y_3136_ = v_proof_x3f_3260_;
v___y_3137_ = v___x_3251_;
v_expr_3138_ = v_expr_3267_;
v_proof_x3f_3139_ = v_proof_x3f_3268_;
v___y_3140_ = v___y_3247_;
v___y_3141_ = v___y_3248_;
v___y_3142_ = v___y_3249_;
v___y_3143_ = v___y_3250_;
goto v___jp_3135_;
}
else
{
lean_object* v_expr_3269_; lean_object* v_proof_x3f_3270_; lean_object* v___x_3271_; lean_object* v___x_3272_; lean_object* v___x_3273_; lean_object* v___x_3274_; 
v_expr_3269_ = lean_ctor_get(v_a_3263_, 0);
lean_inc_ref_n(v_expr_3269_, 2);
v_proof_x3f_3270_ = lean_ctor_get(v_a_3263_, 1);
lean_inc(v_proof_x3f_3270_);
lean_dec(v_a_3263_);
v___x_3271_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__16, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__16);
v___x_3272_ = l_Lean_MessageData_ofExpr(v_expr_3269_);
v___x_3273_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3273_, 0, v___x_3271_);
lean_ctor_set(v___x_3273_, 1, v___x_3272_);
v___x_3274_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(v_cls_3134_, v___x_3273_, v___y_3247_, v___y_3248_, v___y_3249_, v___y_3250_);
if (lean_obj_tag(v___x_3274_) == 0)
{
lean_dec_ref_known(v___x_3274_, 1);
v___y_3136_ = v_proof_x3f_3260_;
v___y_3137_ = v___x_3251_;
v_expr_3138_ = v_expr_3269_;
v_proof_x3f_3139_ = v_proof_x3f_3270_;
v___y_3140_ = v___y_3247_;
v___y_3141_ = v___y_3248_;
v___y_3142_ = v___y_3249_;
v___y_3143_ = v___y_3250_;
goto v___jp_3135_;
}
else
{
lean_object* v_a_3275_; lean_object* v___x_3277_; uint8_t v_isShared_3278_; uint8_t v_isSharedCheck_3282_; 
lean_dec(v_proof_x3f_3270_);
lean_dec_ref(v_expr_3269_);
lean_dec(v_proof_x3f_3260_);
v_a_3275_ = lean_ctor_get(v___x_3274_, 0);
v_isSharedCheck_3282_ = !lean_is_exclusive(v___x_3274_);
if (v_isSharedCheck_3282_ == 0)
{
v___x_3277_ = v___x_3274_;
v_isShared_3278_ = v_isSharedCheck_3282_;
goto v_resetjp_3276_;
}
else
{
lean_inc(v_a_3275_);
lean_dec(v___x_3274_);
v___x_3277_ = lean_box(0);
v_isShared_3278_ = v_isSharedCheck_3282_;
goto v_resetjp_3276_;
}
v_resetjp_3276_:
{
lean_object* v___x_3280_; 
if (v_isShared_3278_ == 0)
{
v___x_3280_ = v___x_3277_;
goto v_reusejp_3279_;
}
else
{
lean_object* v_reuseFailAlloc_3281_; 
v_reuseFailAlloc_3281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3281_, 0, v_a_3275_);
v___x_3280_ = v_reuseFailAlloc_3281_;
goto v_reusejp_3279_;
}
v_reusejp_3279_:
{
return v___x_3280_;
}
}
}
}
}
else
{
lean_object* v_a_3283_; lean_object* v___x_3285_; uint8_t v_isShared_3286_; uint8_t v_isSharedCheck_3290_; 
lean_dec(v_proof_x3f_3260_);
v_a_3283_ = lean_ctor_get(v___x_3262_, 0);
v_isSharedCheck_3290_ = !lean_is_exclusive(v___x_3262_);
if (v_isSharedCheck_3290_ == 0)
{
v___x_3285_ = v___x_3262_;
v_isShared_3286_ = v_isSharedCheck_3290_;
goto v_resetjp_3284_;
}
else
{
lean_inc(v_a_3283_);
lean_dec(v___x_3262_);
v___x_3285_ = lean_box(0);
v_isShared_3286_ = v_isSharedCheck_3290_;
goto v_resetjp_3284_;
}
v_resetjp_3284_:
{
lean_object* v___x_3288_; 
if (v_isShared_3286_ == 0)
{
v___x_3288_ = v___x_3285_;
goto v_reusejp_3287_;
}
else
{
lean_object* v_reuseFailAlloc_3289_; 
v_reuseFailAlloc_3289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3289_, 0, v_a_3283_);
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
lean_dec_ref(v___y_3246_);
v_a_3291_ = lean_ctor_get(v___x_3257_, 0);
v_isSharedCheck_3298_ = !lean_is_exclusive(v___x_3257_);
if (v_isSharedCheck_3298_ == 0)
{
v___x_3293_ = v___x_3257_;
v_isShared_3294_ = v_isSharedCheck_3298_;
goto v_resetjp_3292_;
}
else
{
lean_inc(v_a_3291_);
lean_dec(v___x_3257_);
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
v___jp_3299_:
{
lean_object* v___x_3304_; lean_object* v___x_3305_; lean_object* v___x_3306_; 
v___x_3304_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_deriveThms));
v___x_3305_ = l_Lean_Meta_Simp_neutralConfig;
v___x_3306_ = lp_mathlib_Lean_Meta_simpOnlyNames(v___x_3304_, v_e_3075_, v___x_3305_, v___y_3300_, v___y_3301_, v___y_3302_, v___y_3303_);
if (lean_obj_tag(v___x_3306_) == 0)
{
lean_object* v_a_3307_; lean_object* v___x_3308_; lean_object* v_a_3309_; uint8_t v___x_3310_; 
v_a_3307_ = lean_ctor_get(v___x_3306_, 0);
lean_inc(v_a_3307_);
lean_dec_ref_known(v___x_3306_, 1);
v___x_3308_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___lam__0(v_cls_3134_, v___y_3300_, v___y_3301_, v___y_3302_, v___y_3303_);
v_a_3309_ = lean_ctor_get(v___x_3308_, 0);
lean_inc(v_a_3309_);
lean_dec_ref(v___x_3308_);
v___x_3310_ = lean_unbox(v_a_3309_);
lean_dec(v_a_3309_);
if (v___x_3310_ == 0)
{
v___y_3246_ = v_a_3307_;
v___y_3247_ = v___y_3300_;
v___y_3248_ = v___y_3301_;
v___y_3249_ = v___y_3302_;
v___y_3250_ = v___y_3303_;
goto v___jp_3245_;
}
else
{
lean_object* v_expr_3311_; lean_object* v___x_3312_; lean_object* v___x_3313_; lean_object* v___x_3314_; lean_object* v___x_3315_; 
v_expr_3311_ = lean_ctor_get(v_a_3307_, 0);
v___x_3312_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__18, &lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__18);
lean_inc_ref(v_expr_3311_);
v___x_3313_ = l_Lean_MessageData_ofExpr(v_expr_3311_);
v___x_3314_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3314_, 0, v___x_3312_);
lean_ctor_set(v___x_3314_, 1, v___x_3313_);
v___x_3315_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_CancelDenoms_mkProdPrf_spec__2(v_cls_3134_, v___x_3314_, v___y_3300_, v___y_3301_, v___y_3302_, v___y_3303_);
if (lean_obj_tag(v___x_3315_) == 0)
{
lean_dec_ref_known(v___x_3315_, 1);
v___y_3246_ = v_a_3307_;
v___y_3247_ = v___y_3300_;
v___y_3248_ = v___y_3301_;
v___y_3249_ = v___y_3302_;
v___y_3250_ = v___y_3303_;
goto v___jp_3245_;
}
else
{
lean_object* v_a_3316_; lean_object* v___x_3318_; uint8_t v_isShared_3319_; uint8_t v_isSharedCheck_3323_; 
lean_dec(v_a_3307_);
v_a_3316_ = lean_ctor_get(v___x_3315_, 0);
v_isSharedCheck_3323_ = !lean_is_exclusive(v___x_3315_);
if (v_isSharedCheck_3323_ == 0)
{
v___x_3318_ = v___x_3315_;
v_isShared_3319_ = v_isSharedCheck_3323_;
goto v_resetjp_3317_;
}
else
{
lean_inc(v_a_3316_);
lean_dec(v___x_3315_);
v___x_3318_ = lean_box(0);
v_isShared_3319_ = v_isSharedCheck_3323_;
goto v_resetjp_3317_;
}
v_resetjp_3317_:
{
lean_object* v___x_3321_; 
if (v_isShared_3319_ == 0)
{
v___x_3321_ = v___x_3318_;
goto v_reusejp_3320_;
}
else
{
lean_object* v_reuseFailAlloc_3322_; 
v_reuseFailAlloc_3322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3322_, 0, v_a_3316_);
v___x_3321_ = v_reuseFailAlloc_3322_;
goto v_reusejp_3320_;
}
v_reusejp_3320_:
{
return v___x_3321_;
}
}
}
}
}
else
{
lean_object* v_a_3324_; lean_object* v___x_3326_; uint8_t v_isShared_3327_; uint8_t v_isSharedCheck_3331_; 
v_a_3324_ = lean_ctor_get(v___x_3306_, 0);
v_isSharedCheck_3331_ = !lean_is_exclusive(v___x_3306_);
if (v_isSharedCheck_3331_ == 0)
{
v___x_3326_ = v___x_3306_;
v_isShared_3327_ = v_isSharedCheck_3331_;
goto v_resetjp_3325_;
}
else
{
lean_inc(v_a_3324_);
lean_dec(v___x_3306_);
v___x_3326_ = lean_box(0);
v_isShared_3327_ = v_isSharedCheck_3331_;
goto v_resetjp_3325_;
}
v_resetjp_3325_:
{
lean_object* v___x_3329_; 
if (v_isShared_3327_ == 0)
{
v___x_3329_ = v___x_3326_;
goto v_reusejp_3328_;
}
else
{
lean_object* v_reuseFailAlloc_3330_; 
v_reuseFailAlloc_3330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3330_, 0, v_a_3324_);
v___x_3329_ = v_reuseFailAlloc_3330_;
goto v_reusejp_3328_;
}
v_reusejp_3328_:
{
return v___x_3329_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___boxed(lean_object* v_e_3347_, lean_object* v_a_3348_, lean_object* v_a_3349_, lean_object* v_a_3350_, lean_object* v_a_3351_, lean_object* v_a_3352_){
_start:
{
lean_object* v_res_3353_; 
v_res_3353_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive(v_e_3347_, v_a_3348_, v_a_3349_, v_a_3350_, v_a_3351_);
lean_dec(v_a_3351_);
lean_dec_ref(v_a_3350_);
lean_dec(v_a_3349_);
lean_dec_ref(v_a_3348_);
return v_res_3353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma(lean_object* v_e_3391_, lean_object* v_a_3392_, lean_object* v_a_3393_, lean_object* v_a_3394_, lean_object* v_a_3395_){
_start:
{
lean_object* v___x_3403_; 
v___x_3403_ = l_Lean_Meta_whnfR(v_e_3391_, v_a_3392_, v_a_3393_, v_a_3394_, v_a_3395_);
if (lean_obj_tag(v___x_3403_) == 0)
{
lean_object* v_a_3404_; lean_object* v___x_3406_; uint8_t v_isShared_3407_; uint8_t v_isSharedCheck_3584_; 
v_a_3404_ = lean_ctor_get(v___x_3403_, 0);
v_isSharedCheck_3584_ = !lean_is_exclusive(v___x_3403_);
if (v_isSharedCheck_3584_ == 0)
{
v___x_3406_ = v___x_3403_;
v_isShared_3407_ = v_isSharedCheck_3584_;
goto v_resetjp_3405_;
}
else
{
lean_inc(v_a_3404_);
lean_dec(v___x_3403_);
v___x_3406_ = lean_box(0);
v_isShared_3407_ = v_isSharedCheck_3584_;
goto v_resetjp_3405_;
}
v_resetjp_3405_:
{
lean_object* v___x_3408_; lean_object* v_fst_3409_; 
v___x_3408_ = l_Lean_Expr_getAppFnArgs(v_a_3404_);
v_fst_3409_ = lean_ctor_get(v___x_3408_, 0);
lean_inc(v_fst_3409_);
if (lean_obj_tag(v_fst_3409_) == 1)
{
lean_object* v_pre_3410_; 
v_pre_3410_ = lean_ctor_get(v_fst_3409_, 0);
switch(lean_obj_tag(v_pre_3410_))
{
case 1:
{
lean_object* v_pre_3411_; 
lean_inc_ref(v_pre_3410_);
v_pre_3411_ = lean_ctor_get(v_pre_3410_, 0);
if (lean_obj_tag(v_pre_3411_) == 0)
{
lean_object* v_snd_3412_; lean_object* v___x_3414_; uint8_t v_isShared_3415_; uint8_t v_isSharedCheck_3506_; 
v_snd_3412_ = lean_ctor_get(v___x_3408_, 1);
v_isSharedCheck_3506_ = !lean_is_exclusive(v___x_3408_);
if (v_isSharedCheck_3506_ == 0)
{
lean_object* v_unused_3507_; 
v_unused_3507_ = lean_ctor_get(v___x_3408_, 0);
lean_dec(v_unused_3507_);
v___x_3414_ = v___x_3408_;
v_isShared_3415_ = v_isSharedCheck_3506_;
goto v_resetjp_3413_;
}
else
{
lean_inc(v_snd_3412_);
lean_dec(v___x_3408_);
v___x_3414_ = lean_box(0);
v_isShared_3415_ = v_isSharedCheck_3506_;
goto v_resetjp_3413_;
}
v_resetjp_3413_:
{
lean_object* v_str_3416_; lean_object* v_str_3417_; lean_object* v___x_3418_; uint8_t v___x_3419_; 
v_str_3416_ = lean_ctor_get(v_fst_3409_, 1);
lean_inc_ref(v_str_3416_);
lean_dec_ref_known(v_fst_3409_, 2);
v_str_3417_ = lean_ctor_get(v_pre_3410_, 1);
lean_inc_ref(v_str_3417_);
lean_dec_ref_known(v_pre_3410_, 2);
v___x_3418_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__0));
v___x_3419_ = lean_string_dec_eq(v_str_3417_, v___x_3418_);
if (v___x_3419_ == 0)
{
lean_object* v___x_3420_; uint8_t v___x_3421_; 
v___x_3420_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__1));
v___x_3421_ = lean_string_dec_eq(v_str_3417_, v___x_3420_);
if (v___x_3421_ == 0)
{
lean_object* v___x_3422_; uint8_t v___x_3423_; 
v___x_3422_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__2));
v___x_3423_ = lean_string_dec_eq(v_str_3417_, v___x_3422_);
if (v___x_3423_ == 0)
{
lean_object* v___x_3424_; uint8_t v___x_3425_; 
v___x_3424_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__3));
v___x_3425_ = lean_string_dec_eq(v_str_3417_, v___x_3424_);
lean_dec_ref(v_str_3417_);
if (v___x_3425_ == 0)
{
lean_dec_ref(v_str_3416_);
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3426_; uint8_t v___x_3427_; 
v___x_3426_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__4));
v___x_3427_ = lean_string_dec_eq(v_str_3416_, v___x_3426_);
lean_dec_ref(v_str_3416_);
if (v___x_3427_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3428_; lean_object* v___x_3429_; uint8_t v___x_3430_; 
v___x_3428_ = lean_array_get_size(v_snd_3412_);
v___x_3429_ = lean_unsigned_to_nat(4u);
v___x_3430_ = lean_nat_dec_eq(v___x_3428_, v___x_3429_);
if (v___x_3430_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3438_; 
v___x_3431_ = lean_unsigned_to_nat(2u);
v___x_3432_ = lean_array_fget(v_snd_3412_, v___x_3431_);
v___x_3433_ = lean_unsigned_to_nat(3u);
v___x_3434_ = lean_array_fget(v_snd_3412_, v___x_3433_);
lean_dec(v_snd_3412_);
v___x_3435_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6));
v___x_3436_ = lean_box(v___x_3430_);
if (v_isShared_3415_ == 0)
{
lean_ctor_set(v___x_3414_, 1, v___x_3436_);
lean_ctor_set(v___x_3414_, 0, v___x_3435_);
v___x_3438_ = v___x_3414_;
goto v_reusejp_3437_;
}
else
{
lean_object* v_reuseFailAlloc_3445_; 
v_reuseFailAlloc_3445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3445_, 0, v___x_3435_);
lean_ctor_set(v_reuseFailAlloc_3445_, 1, v___x_3436_);
v___x_3438_ = v_reuseFailAlloc_3445_;
goto v_reusejp_3437_;
}
v_reusejp_3437_:
{
lean_object* v___x_3439_; lean_object* v___x_3440_; lean_object* v___x_3441_; lean_object* v___x_3443_; 
v___x_3439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3439_, 0, v___x_3432_);
lean_ctor_set(v___x_3439_, 1, v___x_3438_);
v___x_3440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3440_, 0, v___x_3434_);
lean_ctor_set(v___x_3440_, 1, v___x_3439_);
v___x_3441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3441_, 0, v___x_3440_);
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 0, v___x_3441_);
v___x_3443_ = v___x_3406_;
goto v_reusejp_3442_;
}
else
{
lean_object* v_reuseFailAlloc_3444_; 
v_reuseFailAlloc_3444_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3444_, 0, v___x_3441_);
v___x_3443_ = v_reuseFailAlloc_3444_;
goto v_reusejp_3442_;
}
v_reusejp_3442_:
{
return v___x_3443_;
}
}
}
}
}
}
else
{
lean_object* v___x_3446_; uint8_t v___x_3447_; 
lean_dec_ref(v_str_3417_);
v___x_3446_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__7));
v___x_3447_ = lean_string_dec_eq(v_str_3416_, v___x_3446_);
lean_dec_ref(v_str_3416_);
if (v___x_3447_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3448_; lean_object* v___x_3449_; uint8_t v___x_3450_; 
v___x_3448_ = lean_array_get_size(v_snd_3412_);
v___x_3449_ = lean_unsigned_to_nat(4u);
v___x_3450_ = lean_nat_dec_eq(v___x_3448_, v___x_3449_);
if (v___x_3450_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3451_; lean_object* v___x_3452_; lean_object* v___x_3453_; lean_object* v___x_3454_; lean_object* v___x_3455_; lean_object* v___x_3456_; lean_object* v___x_3458_; 
v___x_3451_ = lean_unsigned_to_nat(2u);
v___x_3452_ = lean_array_fget(v_snd_3412_, v___x_3451_);
v___x_3453_ = lean_unsigned_to_nat(3u);
v___x_3454_ = lean_array_fget(v_snd_3412_, v___x_3453_);
lean_dec(v_snd_3412_);
v___x_3455_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9));
v___x_3456_ = lean_box(v___x_3450_);
if (v_isShared_3415_ == 0)
{
lean_ctor_set(v___x_3414_, 1, v___x_3456_);
lean_ctor_set(v___x_3414_, 0, v___x_3455_);
v___x_3458_ = v___x_3414_;
goto v_reusejp_3457_;
}
else
{
lean_object* v_reuseFailAlloc_3465_; 
v_reuseFailAlloc_3465_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3465_, 0, v___x_3455_);
lean_ctor_set(v_reuseFailAlloc_3465_, 1, v___x_3456_);
v___x_3458_ = v_reuseFailAlloc_3465_;
goto v_reusejp_3457_;
}
v_reusejp_3457_:
{
lean_object* v___x_3459_; lean_object* v___x_3460_; lean_object* v___x_3461_; lean_object* v___x_3463_; 
v___x_3459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3459_, 0, v___x_3452_);
lean_ctor_set(v___x_3459_, 1, v___x_3458_);
v___x_3460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3460_, 0, v___x_3454_);
lean_ctor_set(v___x_3460_, 1, v___x_3459_);
v___x_3461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3461_, 0, v___x_3460_);
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 0, v___x_3461_);
v___x_3463_ = v___x_3406_;
goto v_reusejp_3462_;
}
else
{
lean_object* v_reuseFailAlloc_3464_; 
v_reuseFailAlloc_3464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3464_, 0, v___x_3461_);
v___x_3463_ = v_reuseFailAlloc_3464_;
goto v_reusejp_3462_;
}
v_reusejp_3462_:
{
return v___x_3463_;
}
}
}
}
}
}
else
{
lean_object* v___x_3466_; uint8_t v___x_3467_; 
lean_dec_ref(v_str_3417_);
v___x_3466_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__10));
v___x_3467_ = lean_string_dec_eq(v_str_3416_, v___x_3466_);
lean_dec_ref(v_str_3416_);
if (v___x_3467_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3468_; lean_object* v___x_3469_; uint8_t v___x_3470_; 
v___x_3468_ = lean_array_get_size(v_snd_3412_);
v___x_3469_ = lean_unsigned_to_nat(4u);
v___x_3470_ = lean_nat_dec_eq(v___x_3468_, v___x_3469_);
if (v___x_3470_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3471_; lean_object* v___x_3472_; lean_object* v___x_3473_; lean_object* v___x_3474_; lean_object* v___x_3475_; lean_object* v___x_3476_; lean_object* v___x_3478_; 
v___x_3471_ = lean_unsigned_to_nat(2u);
v___x_3472_ = lean_array_fget(v_snd_3412_, v___x_3471_);
v___x_3473_ = lean_unsigned_to_nat(3u);
v___x_3474_ = lean_array_fget(v_snd_3412_, v___x_3473_);
lean_dec(v_snd_3412_);
v___x_3475_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__9));
v___x_3476_ = lean_box(v___x_3470_);
if (v_isShared_3415_ == 0)
{
lean_ctor_set(v___x_3414_, 1, v___x_3476_);
lean_ctor_set(v___x_3414_, 0, v___x_3475_);
v___x_3478_ = v___x_3414_;
goto v_reusejp_3477_;
}
else
{
lean_object* v_reuseFailAlloc_3485_; 
v_reuseFailAlloc_3485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3485_, 0, v___x_3475_);
lean_ctor_set(v_reuseFailAlloc_3485_, 1, v___x_3476_);
v___x_3478_ = v_reuseFailAlloc_3485_;
goto v_reusejp_3477_;
}
v_reusejp_3477_:
{
lean_object* v___x_3479_; lean_object* v___x_3480_; lean_object* v___x_3481_; lean_object* v___x_3483_; 
v___x_3479_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3479_, 0, v___x_3474_);
lean_ctor_set(v___x_3479_, 1, v___x_3478_);
v___x_3480_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3480_, 0, v___x_3472_);
lean_ctor_set(v___x_3480_, 1, v___x_3479_);
v___x_3481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3481_, 0, v___x_3480_);
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 0, v___x_3481_);
v___x_3483_ = v___x_3406_;
goto v_reusejp_3482_;
}
else
{
lean_object* v_reuseFailAlloc_3484_; 
v_reuseFailAlloc_3484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3484_, 0, v___x_3481_);
v___x_3483_ = v_reuseFailAlloc_3484_;
goto v_reusejp_3482_;
}
v_reusejp_3482_:
{
return v___x_3483_;
}
}
}
}
}
}
else
{
lean_object* v___x_3486_; uint8_t v___x_3487_; 
lean_dec_ref(v_str_3417_);
v___x_3486_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__11));
v___x_3487_ = lean_string_dec_eq(v_str_3416_, v___x_3486_);
lean_dec_ref(v_str_3416_);
if (v___x_3487_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3488_; lean_object* v___x_3489_; uint8_t v___x_3490_; 
v___x_3488_ = lean_array_get_size(v_snd_3412_);
v___x_3489_ = lean_unsigned_to_nat(4u);
v___x_3490_ = lean_nat_dec_eq(v___x_3488_, v___x_3489_);
if (v___x_3490_ == 0)
{
lean_del_object(v___x_3414_);
lean_dec(v_snd_3412_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3491_; lean_object* v___x_3492_; lean_object* v___x_3493_; lean_object* v___x_3494_; lean_object* v___x_3495_; lean_object* v___x_3496_; lean_object* v___x_3498_; 
v___x_3491_ = lean_unsigned_to_nat(2u);
v___x_3492_ = lean_array_fget(v_snd_3412_, v___x_3491_);
v___x_3493_ = lean_unsigned_to_nat(3u);
v___x_3494_ = lean_array_fget(v_snd_3412_, v___x_3493_);
lean_dec(v_snd_3412_);
v___x_3495_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__6));
v___x_3496_ = lean_box(v___x_3490_);
if (v_isShared_3415_ == 0)
{
lean_ctor_set(v___x_3414_, 1, v___x_3496_);
lean_ctor_set(v___x_3414_, 0, v___x_3495_);
v___x_3498_ = v___x_3414_;
goto v_reusejp_3497_;
}
else
{
lean_object* v_reuseFailAlloc_3505_; 
v_reuseFailAlloc_3505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3505_, 0, v___x_3495_);
lean_ctor_set(v_reuseFailAlloc_3505_, 1, v___x_3496_);
v___x_3498_ = v_reuseFailAlloc_3505_;
goto v_reusejp_3497_;
}
v_reusejp_3497_:
{
lean_object* v___x_3499_; lean_object* v___x_3500_; lean_object* v___x_3501_; lean_object* v___x_3503_; 
v___x_3499_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3499_, 0, v___x_3494_);
lean_ctor_set(v___x_3499_, 1, v___x_3498_);
v___x_3500_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3500_, 0, v___x_3492_);
lean_ctor_set(v___x_3500_, 1, v___x_3499_);
v___x_3501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3501_, 0, v___x_3500_);
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 0, v___x_3501_);
v___x_3503_ = v___x_3406_;
goto v_reusejp_3502_;
}
else
{
lean_object* v_reuseFailAlloc_3504_; 
v_reuseFailAlloc_3504_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3504_, 0, v___x_3501_);
v___x_3503_ = v_reuseFailAlloc_3504_;
goto v_reusejp_3502_;
}
v_reusejp_3502_:
{
return v___x_3503_;
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_3410_, 2);
lean_dec_ref_known(v_fst_3409_, 2);
lean_dec_ref(v___x_3408_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
}
case 0:
{
lean_object* v_snd_3508_; lean_object* v___x_3510_; uint8_t v_isShared_3511_; uint8_t v_isSharedCheck_3582_; 
v_snd_3508_ = lean_ctor_get(v___x_3408_, 1);
v_isSharedCheck_3582_ = !lean_is_exclusive(v___x_3408_);
if (v_isSharedCheck_3582_ == 0)
{
lean_object* v_unused_3583_; 
v_unused_3583_ = lean_ctor_get(v___x_3408_, 0);
lean_dec(v_unused_3583_);
v___x_3510_ = v___x_3408_;
v_isShared_3511_ = v_isSharedCheck_3582_;
goto v_resetjp_3509_;
}
else
{
lean_inc(v_snd_3508_);
lean_dec(v___x_3408_);
v___x_3510_ = lean_box(0);
v_isShared_3511_ = v_isSharedCheck_3582_;
goto v_resetjp_3509_;
}
v_resetjp_3509_:
{
lean_object* v_str_3512_; lean_object* v___x_3513_; uint8_t v___x_3514_; 
v_str_3512_ = lean_ctor_get(v_fst_3409_, 1);
lean_inc_ref(v_str_3512_);
lean_dec_ref_known(v_fst_3409_, 2);
v___x_3513_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__80));
v___x_3514_ = lean_string_dec_eq(v_str_3512_, v___x_3513_);
if (v___x_3514_ == 0)
{
lean_object* v___x_3515_; uint8_t v___x_3516_; 
lean_del_object(v___x_3406_);
v___x_3515_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__12));
v___x_3516_ = lean_string_dec_eq(v_str_3512_, v___x_3515_);
lean_dec_ref(v_str_3512_);
if (v___x_3516_ == 0)
{
lean_del_object(v___x_3510_);
lean_dec(v_snd_3508_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3517_; lean_object* v___x_3518_; uint8_t v___x_3519_; 
v___x_3517_ = lean_array_get_size(v_snd_3508_);
v___x_3518_ = lean_unsigned_to_nat(1u);
v___x_3519_ = lean_nat_dec_eq(v___x_3517_, v___x_3518_);
if (v___x_3519_ == 0)
{
lean_del_object(v___x_3510_);
lean_dec(v_snd_3508_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3520_; lean_object* v___x_3521_; lean_object* v___x_3522_; 
v___x_3520_ = lean_unsigned_to_nat(0u);
v___x_3521_ = lean_array_fget(v_snd_3508_, v___x_3520_);
lean_dec(v_snd_3508_);
v___x_3522_ = l_Lean_Meta_whnfR(v___x_3521_, v_a_3392_, v_a_3393_, v_a_3394_, v_a_3395_);
if (lean_obj_tag(v___x_3522_) == 0)
{
lean_object* v_a_3523_; lean_object* v___x_3525_; uint8_t v_isShared_3526_; uint8_t v_isSharedCheck_3557_; 
v_a_3523_ = lean_ctor_get(v___x_3522_, 0);
v_isSharedCheck_3557_ = !lean_is_exclusive(v___x_3522_);
if (v_isSharedCheck_3557_ == 0)
{
v___x_3525_ = v___x_3522_;
v_isShared_3526_ = v_isSharedCheck_3557_;
goto v_resetjp_3524_;
}
else
{
lean_inc(v_a_3523_);
lean_dec(v___x_3522_);
v___x_3525_ = lean_box(0);
v_isShared_3526_ = v_isSharedCheck_3557_;
goto v_resetjp_3524_;
}
v_resetjp_3524_:
{
lean_object* v___x_3527_; lean_object* v_fst_3528_; 
v___x_3527_ = l_Lean_Expr_getAppFnArgs(v_a_3523_);
v_fst_3528_ = lean_ctor_get(v___x_3527_, 0);
lean_inc(v_fst_3528_);
if (lean_obj_tag(v_fst_3528_) == 1)
{
lean_object* v_pre_3529_; 
v_pre_3529_ = lean_ctor_get(v_fst_3528_, 0);
if (lean_obj_tag(v_pre_3529_) == 0)
{
lean_object* v_snd_3530_; lean_object* v___x_3532_; uint8_t v_isShared_3533_; uint8_t v_isSharedCheck_3555_; 
v_snd_3530_ = lean_ctor_get(v___x_3527_, 1);
v_isSharedCheck_3555_ = !lean_is_exclusive(v___x_3527_);
if (v_isSharedCheck_3555_ == 0)
{
lean_object* v_unused_3556_; 
v_unused_3556_ = lean_ctor_get(v___x_3527_, 0);
lean_dec(v_unused_3556_);
v___x_3532_ = v___x_3527_;
v_isShared_3533_ = v_isSharedCheck_3555_;
goto v_resetjp_3531_;
}
else
{
lean_inc(v_snd_3530_);
lean_dec(v___x_3527_);
v___x_3532_ = lean_box(0);
v_isShared_3533_ = v_isSharedCheck_3555_;
goto v_resetjp_3531_;
}
v_resetjp_3531_:
{
lean_object* v_str_3534_; uint8_t v___x_3535_; 
v_str_3534_ = lean_ctor_get(v_fst_3528_, 1);
lean_inc_ref(v_str_3534_);
lean_dec_ref_known(v_fst_3528_, 2);
v___x_3535_ = lean_string_dec_eq(v_str_3534_, v___x_3513_);
lean_dec_ref(v_str_3534_);
if (v___x_3535_ == 0)
{
lean_del_object(v___x_3532_);
lean_dec(v_snd_3530_);
lean_del_object(v___x_3525_);
lean_del_object(v___x_3510_);
goto v___jp_3400_;
}
else
{
lean_object* v___x_3536_; lean_object* v___x_3537_; uint8_t v___x_3538_; 
v___x_3536_ = lean_array_get_size(v_snd_3530_);
v___x_3537_ = lean_unsigned_to_nat(3u);
v___x_3538_ = lean_nat_dec_eq(v___x_3536_, v___x_3537_);
if (v___x_3538_ == 0)
{
lean_del_object(v___x_3532_);
lean_dec(v_snd_3530_);
lean_del_object(v___x_3525_);
lean_del_object(v___x_3510_);
goto v___jp_3400_;
}
else
{
lean_object* v___x_3539_; lean_object* v___x_3540_; lean_object* v___x_3541_; lean_object* v___x_3542_; lean_object* v___x_3543_; lean_object* v___x_3545_; 
v___x_3539_ = lean_array_fget(v_snd_3530_, v___x_3518_);
v___x_3540_ = lean_unsigned_to_nat(2u);
v___x_3541_ = lean_array_fget(v_snd_3530_, v___x_3540_);
lean_dec(v_snd_3530_);
v___x_3542_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__14));
v___x_3543_ = lean_box(v___x_3514_);
if (v_isShared_3533_ == 0)
{
lean_ctor_set(v___x_3532_, 1, v___x_3543_);
lean_ctor_set(v___x_3532_, 0, v___x_3542_);
v___x_3545_ = v___x_3532_;
goto v_reusejp_3544_;
}
else
{
lean_object* v_reuseFailAlloc_3554_; 
v_reuseFailAlloc_3554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3554_, 0, v___x_3542_);
lean_ctor_set(v_reuseFailAlloc_3554_, 1, v___x_3543_);
v___x_3545_ = v_reuseFailAlloc_3554_;
goto v_reusejp_3544_;
}
v_reusejp_3544_:
{
lean_object* v___x_3547_; 
if (v_isShared_3511_ == 0)
{
lean_ctor_set(v___x_3510_, 1, v___x_3545_);
lean_ctor_set(v___x_3510_, 0, v___x_3541_);
v___x_3547_ = v___x_3510_;
goto v_reusejp_3546_;
}
else
{
lean_object* v_reuseFailAlloc_3553_; 
v_reuseFailAlloc_3553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3553_, 0, v___x_3541_);
lean_ctor_set(v_reuseFailAlloc_3553_, 1, v___x_3545_);
v___x_3547_ = v_reuseFailAlloc_3553_;
goto v_reusejp_3546_;
}
v_reusejp_3546_:
{
lean_object* v___x_3548_; lean_object* v___x_3549_; lean_object* v___x_3551_; 
v___x_3548_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3548_, 0, v___x_3539_);
lean_ctor_set(v___x_3548_, 1, v___x_3547_);
v___x_3549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3549_, 0, v___x_3548_);
if (v_isShared_3526_ == 0)
{
lean_ctor_set(v___x_3525_, 0, v___x_3549_);
v___x_3551_ = v___x_3525_;
goto v_reusejp_3550_;
}
else
{
lean_object* v_reuseFailAlloc_3552_; 
v_reuseFailAlloc_3552_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3552_, 0, v___x_3549_);
v___x_3551_ = v_reuseFailAlloc_3552_;
goto v_reusejp_3550_;
}
v_reusejp_3550_:
{
return v___x_3551_;
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_fst_3528_, 2);
lean_dec_ref(v___x_3527_);
lean_del_object(v___x_3525_);
lean_del_object(v___x_3510_);
goto v___jp_3400_;
}
}
else
{
lean_dec(v_fst_3528_);
lean_dec_ref(v___x_3527_);
lean_del_object(v___x_3525_);
lean_del_object(v___x_3510_);
goto v___jp_3400_;
}
}
}
else
{
lean_object* v_a_3558_; lean_object* v___x_3560_; uint8_t v_isShared_3561_; uint8_t v_isSharedCheck_3565_; 
lean_del_object(v___x_3510_);
v_a_3558_ = lean_ctor_get(v___x_3522_, 0);
v_isSharedCheck_3565_ = !lean_is_exclusive(v___x_3522_);
if (v_isSharedCheck_3565_ == 0)
{
v___x_3560_ = v___x_3522_;
v_isShared_3561_ = v_isSharedCheck_3565_;
goto v_resetjp_3559_;
}
else
{
lean_inc(v_a_3558_);
lean_dec(v___x_3522_);
v___x_3560_ = lean_box(0);
v_isShared_3561_ = v_isSharedCheck_3565_;
goto v_resetjp_3559_;
}
v_resetjp_3559_:
{
lean_object* v___x_3563_; 
if (v_isShared_3561_ == 0)
{
v___x_3563_ = v___x_3560_;
goto v_reusejp_3562_;
}
else
{
lean_object* v_reuseFailAlloc_3564_; 
v_reuseFailAlloc_3564_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3564_, 0, v_a_3558_);
v___x_3563_ = v_reuseFailAlloc_3564_;
goto v_reusejp_3562_;
}
v_reusejp_3562_:
{
return v___x_3563_;
}
}
}
}
}
}
else
{
lean_object* v___x_3566_; lean_object* v___x_3567_; uint8_t v___x_3568_; 
lean_dec_ref(v_str_3512_);
v___x_3566_ = lean_array_get_size(v_snd_3508_);
v___x_3567_ = lean_unsigned_to_nat(3u);
v___x_3568_ = lean_nat_dec_eq(v___x_3566_, v___x_3567_);
if (v___x_3568_ == 0)
{
lean_del_object(v___x_3510_);
lean_dec(v_snd_3508_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
else
{
lean_object* v___x_3569_; lean_object* v___x_3570_; lean_object* v___x_3571_; lean_object* v___x_3572_; lean_object* v___x_3573_; lean_object* v___x_3575_; 
v___x_3569_ = lean_unsigned_to_nat(1u);
v___x_3570_ = lean_array_fget(v_snd_3508_, v___x_3569_);
v___x_3571_ = lean_unsigned_to_nat(2u);
v___x_3572_ = lean_array_fget(v_snd_3508_, v___x_3571_);
lean_dec(v_snd_3508_);
v___x_3573_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___closed__17));
if (v_isShared_3511_ == 0)
{
lean_ctor_set(v___x_3510_, 1, v___x_3573_);
lean_ctor_set(v___x_3510_, 0, v___x_3572_);
v___x_3575_ = v___x_3510_;
goto v_reusejp_3574_;
}
else
{
lean_object* v_reuseFailAlloc_3581_; 
v_reuseFailAlloc_3581_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3581_, 0, v___x_3572_);
lean_ctor_set(v_reuseFailAlloc_3581_, 1, v___x_3573_);
v___x_3575_ = v_reuseFailAlloc_3581_;
goto v_reusejp_3574_;
}
v_reusejp_3574_:
{
lean_object* v___x_3576_; lean_object* v___x_3577_; lean_object* v___x_3579_; 
v___x_3576_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3576_, 0, v___x_3570_);
lean_ctor_set(v___x_3576_, 1, v___x_3575_);
v___x_3577_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3577_, 0, v___x_3576_);
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 0, v___x_3577_);
v___x_3579_ = v___x_3406_;
goto v_reusejp_3578_;
}
else
{
lean_object* v_reuseFailAlloc_3580_; 
v_reuseFailAlloc_3580_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3580_, 0, v___x_3577_);
v___x_3579_ = v_reuseFailAlloc_3580_;
goto v_reusejp_3578_;
}
v_reusejp_3578_:
{
return v___x_3579_;
}
}
}
}
}
}
default: 
{
lean_dec_ref_known(v_fst_3409_, 2);
lean_dec_ref(v___x_3408_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
}
}
else
{
lean_dec(v_fst_3409_);
lean_dec_ref(v___x_3408_);
lean_del_object(v___x_3406_);
goto v___jp_3397_;
}
}
}
else
{
lean_object* v_a_3585_; lean_object* v___x_3587_; uint8_t v_isShared_3588_; uint8_t v_isSharedCheck_3592_; 
v_a_3585_ = lean_ctor_get(v___x_3403_, 0);
v_isSharedCheck_3592_ = !lean_is_exclusive(v___x_3403_);
if (v_isSharedCheck_3592_ == 0)
{
v___x_3587_ = v___x_3403_;
v_isShared_3588_ = v_isSharedCheck_3592_;
goto v_resetjp_3586_;
}
else
{
lean_inc(v_a_3585_);
lean_dec(v___x_3403_);
v___x_3587_ = lean_box(0);
v_isShared_3588_ = v_isSharedCheck_3592_;
goto v_resetjp_3586_;
}
v_resetjp_3586_:
{
lean_object* v___x_3590_; 
if (v_isShared_3588_ == 0)
{
v___x_3590_ = v___x_3587_;
goto v_reusejp_3589_;
}
else
{
lean_object* v_reuseFailAlloc_3591_; 
v_reuseFailAlloc_3591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3591_, 0, v_a_3585_);
v___x_3590_ = v_reuseFailAlloc_3591_;
goto v_reusejp_3589_;
}
v_reusejp_3589_:
{
return v___x_3590_;
}
}
}
v___jp_3397_:
{
lean_object* v___x_3398_; lean_object* v___x_3399_; 
v___x_3398_ = lean_box(0);
v___x_3399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3399_, 0, v___x_3398_);
return v___x_3399_;
}
v___jp_3400_:
{
lean_object* v___x_3401_; lean_object* v___x_3402_; 
v___x_3401_ = lean_box(0);
v___x_3402_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3402_, 0, v___x_3401_);
return v___x_3402_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma___boxed(lean_object* v_e_3593_, lean_object* v_a_3594_, lean_object* v_a_3595_, lean_object* v_a_3596_, lean_object* v_a_3597_, lean_object* v_a_3598_){
_start:
{
lean_object* v_res_3599_; 
v_res_3599_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma(v_e_3593_, v_a_3594_, v_a_3595_, v_a_3596_, v_a_3597_);
lean_dec(v_a_3597_);
lean_dec_ref(v_a_3596_);
lean_dec(v_a_3595_);
lean_dec_ref(v_a_3594_);
return v_res_3599_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__2(void){
_start:
{
lean_object* v___x_3603_; lean_object* v___x_3604_; lean_object* v___x_3605_; 
v___x_3603_ = lean_box(0);
v___x_3604_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__1));
v___x_3605_ = l_Lean_Expr_const___override(v___x_3604_, v___x_3603_);
return v___x_3605_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__26(void){
_start:
{
lean_object* v___x_3644_; lean_object* v___x_3645_; 
v___x_3644_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__25));
v___x_3645_ = l_Lean_stringToMessageData(v___x_3644_);
return v___x_3645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType(lean_object* v_h_3646_, lean_object* v_a_3647_, lean_object* v_a_3648_, lean_object* v_a_3649_, lean_object* v_a_3650_){
_start:
{
lean_object* v___y_3653_; lean_object* v___y_3654_; lean_object* v___x_3657_; 
v___x_3657_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma(v_h_3646_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3657_) == 0)
{
lean_object* v_a_3658_; 
v_a_3658_ = lean_ctor_get(v___x_3657_, 0);
lean_inc(v_a_3658_);
lean_dec_ref_known(v___x_3657_, 1);
if (lean_obj_tag(v_a_3658_) == 1)
{
lean_object* v_val_3659_; lean_object* v_snd_3660_; lean_object* v_snd_3661_; lean_object* v_fst_3662_; lean_object* v_fst_3663_; lean_object* v_fst_3664_; lean_object* v_snd_3665_; lean_object* v___x_3666_; 
v_val_3659_ = lean_ctor_get(v_a_3658_, 0);
lean_inc(v_val_3659_);
lean_dec_ref_known(v_a_3658_, 1);
v_snd_3660_ = lean_ctor_get(v_val_3659_, 1);
lean_inc(v_snd_3660_);
v_snd_3661_ = lean_ctor_get(v_snd_3660_, 1);
lean_inc(v_snd_3661_);
v_fst_3662_ = lean_ctor_get(v_val_3659_, 0);
lean_inc_n(v_fst_3662_, 2);
lean_dec(v_val_3659_);
v_fst_3663_ = lean_ctor_get(v_snd_3660_, 0);
lean_inc(v_fst_3663_);
lean_dec(v_snd_3660_);
v_fst_3664_ = lean_ctor_get(v_snd_3661_, 0);
lean_inc(v_fst_3664_);
v_snd_3665_ = lean_ctor_get(v_snd_3661_, 1);
lean_inc(v_snd_3665_);
lean_dec(v_snd_3661_);
v___x_3666_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive(v_fst_3662_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3666_) == 0)
{
lean_object* v_a_3667_; lean_object* v_fst_3668_; lean_object* v_snd_3669_; lean_object* v___x_3670_; 
v_a_3667_ = lean_ctor_get(v___x_3666_, 0);
lean_inc(v_a_3667_);
lean_dec_ref_known(v___x_3666_, 1);
v_fst_3668_ = lean_ctor_get(v_a_3667_, 0);
lean_inc(v_fst_3668_);
v_snd_3669_ = lean_ctor_get(v_a_3667_, 1);
lean_inc(v_snd_3669_);
lean_dec(v_a_3667_);
v___x_3670_ = lp_mathlib_Qq_inferTypeQ_x27(v_fst_3662_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3670_) == 0)
{
lean_object* v_a_3671_; lean_object* v_snd_3672_; lean_object* v_fst_3673_; lean_object* v_fst_3674_; lean_object* v___x_3676_; uint8_t v_isShared_3677_; uint8_t v_isSharedCheck_3990_; 
v_a_3671_ = lean_ctor_get(v___x_3670_, 0);
lean_inc(v_a_3671_);
lean_dec_ref_known(v___x_3670_, 1);
v_snd_3672_ = lean_ctor_get(v_a_3671_, 1);
lean_inc(v_snd_3672_);
v_fst_3673_ = lean_ctor_get(v_a_3671_, 0);
lean_inc(v_fst_3673_);
lean_dec(v_a_3671_);
v_fst_3674_ = lean_ctor_get(v_snd_3672_, 0);
v_isSharedCheck_3990_ = !lean_is_exclusive(v_snd_3672_);
if (v_isSharedCheck_3990_ == 0)
{
lean_object* v_unused_3991_; 
v_unused_3991_ = lean_ctor_get(v_snd_3672_, 1);
lean_dec(v_unused_3991_);
v___x_3676_ = v_snd_3672_;
v_isShared_3677_ = v_isSharedCheck_3990_;
goto v_resetjp_3675_;
}
else
{
lean_inc(v_fst_3674_);
lean_dec(v_snd_3672_);
v___x_3676_ = lean_box(0);
v_isShared_3677_ = v_isSharedCheck_3990_;
goto v_resetjp_3675_;
}
v_resetjp_3675_:
{
lean_object* v___x_3678_; lean_object* v___x_3679_; lean_object* v___x_3680_; lean_object* v___x_3682_; 
lean_inc_n(v_fst_3673_, 2);
v___x_3678_ = l_Lean_Level_succ___override(v_fst_3673_);
v___x_3679_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__1));
v___x_3680_ = lean_box(0);
if (v_isShared_3677_ == 0)
{
lean_ctor_set_tag(v___x_3676_, 1);
lean_ctor_set(v___x_3676_, 1, v___x_3680_);
lean_ctor_set(v___x_3676_, 0, v_fst_3673_);
v___x_3682_ = v___x_3676_;
goto v_reusejp_3681_;
}
else
{
lean_object* v_reuseFailAlloc_3989_; 
v_reuseFailAlloc_3989_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3989_, 0, v_fst_3673_);
lean_ctor_set(v_reuseFailAlloc_3989_, 1, v___x_3680_);
v___x_3682_ = v_reuseFailAlloc_3989_;
goto v_reusejp_3681_;
}
v_reusejp_3681_:
{
lean_object* v___x_3683_; lean_object* v___x_3684_; lean_object* v___x_3685_; 
lean_inc_ref(v___x_3682_);
v___x_3683_ = l_Lean_Expr_const___override(v___x_3679_, v___x_3682_);
lean_inc(v_fst_3674_);
v___x_3684_ = l_Lean_Expr_app___override(v___x_3683_, v_fst_3674_);
v___x_3685_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_3684_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3685_) == 0)
{
lean_object* v_a_3686_; lean_object* v___x_3687_; 
v_a_3686_ = lean_ctor_get(v___x_3685_, 0);
lean_inc(v_a_3686_);
lean_dec_ref_known(v___x_3685_, 1);
v___x_3687_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_derive(v_fst_3663_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3687_) == 0)
{
lean_object* v_a_3688_; lean_object* v_fst_3689_; lean_object* v_snd_3690_; lean_object* v_fst_3692_; lean_object* v_fst_3693_; lean_object* v_snd_3694_; lean_object* v___y_3695_; lean_object* v___y_3696_; lean_object* v___y_3697_; lean_object* v___y_3698_; lean_object* v___x_3770_; lean_object* v___x_3771_; lean_object* v___x_3772_; 
v_a_3688_ = lean_ctor_get(v___x_3687_, 0);
lean_inc(v_a_3688_);
lean_dec_ref_known(v___x_3687_, 1);
v_fst_3689_ = lean_ctor_get(v_a_3688_, 0);
lean_inc(v_fst_3689_);
v_snd_3690_ = lean_ctor_get(v_a_3688_, 1);
lean_inc(v_snd_3690_);
lean_dec(v_a_3688_);
v___x_3770_ = lean_nat_gcd(v_fst_3668_, v_fst_3689_);
v___x_3771_ = l_Lean_mkRawNatLit(v_fst_3668_);
lean_inc(v_a_3686_);
lean_inc(v_fst_3674_);
lean_inc(v_fst_3673_);
v___x_3772_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_fst_3673_, v_fst_3674_, v_a_3686_, v___x_3771_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3772_) == 0)
{
lean_object* v_a_3773_; lean_object* v___x_3774_; lean_object* v___x_3775_; 
v_a_3773_ = lean_ctor_get(v___x_3772_, 0);
lean_inc(v_a_3773_);
lean_dec_ref_known(v___x_3772_, 1);
v___x_3774_ = l_Lean_mkRawNatLit(v_fst_3689_);
lean_inc(v_a_3686_);
lean_inc(v_fst_3674_);
lean_inc(v_fst_3673_);
v___x_3775_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_fst_3673_, v_fst_3674_, v_a_3686_, v___x_3774_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3775_) == 0)
{
lean_object* v_a_3776_; lean_object* v___x_3777_; lean_object* v___x_3778_; 
v_a_3776_ = lean_ctor_get(v___x_3775_, 0);
lean_inc(v_a_3776_);
lean_dec_ref_known(v___x_3775_, 1);
v___x_3777_ = l_Lean_mkRawNatLit(v___x_3770_);
lean_inc(v_fst_3674_);
v___x_3778_ = lp_mathlib_Mathlib_Meta_NormNum_mkOfNat(v_fst_3673_, v_fst_3674_, v_a_3686_, v___x_3777_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3778_) == 0)
{
lean_object* v_a_3779_; uint8_t v___x_3780_; 
v_a_3779_ = lean_ctor_get(v___x_3778_, 0);
lean_inc(v_a_3779_);
lean_dec_ref_known(v___x_3778_, 1);
v___x_3780_ = lean_unbox(v_snd_3665_);
lean_dec(v_snd_3665_);
if (v___x_3780_ == 0)
{
lean_object* v_fst_3781_; lean_object* v_fst_3782_; lean_object* v_fst_3783_; lean_object* v___x_3785_; uint8_t v_isShared_3786_; uint8_t v_isSharedCheck_3842_; 
v_fst_3781_ = lean_ctor_get(v_a_3773_, 0);
lean_inc(v_fst_3781_);
lean_dec(v_a_3773_);
v_fst_3782_ = lean_ctor_get(v_a_3776_, 0);
lean_inc(v_fst_3782_);
lean_dec(v_a_3776_);
v_fst_3783_ = lean_ctor_get(v_a_3779_, 0);
v_isSharedCheck_3842_ = !lean_is_exclusive(v_a_3779_);
if (v_isSharedCheck_3842_ == 0)
{
lean_object* v_unused_3843_; 
v_unused_3843_ = lean_ctor_get(v_a_3779_, 1);
lean_dec(v_unused_3843_);
v___x_3785_ = v_a_3779_;
v_isShared_3786_ = v_isSharedCheck_3842_;
goto v_resetjp_3784_;
}
else
{
lean_inc(v_fst_3783_);
lean_dec(v_a_3779_);
v___x_3785_ = lean_box(0);
v_isShared_3786_ = v_isSharedCheck_3842_;
goto v_resetjp_3784_;
}
v_resetjp_3784_:
{
lean_object* v___x_3787_; lean_object* v___x_3788_; lean_object* v___x_3789_; lean_object* v___x_3790_; 
v___x_3787_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__4));
lean_inc_ref(v___x_3682_);
v___x_3788_ = l_Lean_Expr_const___override(v___x_3787_, v___x_3682_);
lean_inc(v_fst_3674_);
v___x_3789_ = l_Lean_Expr_app___override(v___x_3788_, v_fst_3674_);
v___x_3790_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_3789_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3790_) == 0)
{
lean_object* v_a_3791_; lean_object* v___x_3792_; lean_object* v___x_3794_; 
v_a_3791_ = lean_ctor_get(v___x_3790_, 0);
lean_inc(v_a_3791_);
lean_dec_ref_known(v___x_3790_, 1);
v___x_3792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__17));
if (v_isShared_3786_ == 0)
{
lean_ctor_set_tag(v___x_3785_, 1);
lean_ctor_set(v___x_3785_, 1, v___x_3680_);
lean_ctor_set(v___x_3785_, 0, v___x_3678_);
v___x_3794_ = v___x_3785_;
goto v_reusejp_3793_;
}
else
{
lean_object* v_reuseFailAlloc_3833_; 
v_reuseFailAlloc_3833_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3833_, 0, v___x_3678_);
lean_ctor_set(v_reuseFailAlloc_3833_, 1, v___x_3680_);
v___x_3794_ = v_reuseFailAlloc_3833_;
goto v_reusejp_3793_;
}
v_reusejp_3793_:
{
lean_object* v___x_3795_; lean_object* v___x_3796_; lean_object* v___x_3797_; lean_object* v___x_3798_; lean_object* v___x_3799_; lean_object* v___x_3800_; lean_object* v___x_3801_; lean_object* v___x_3802_; lean_object* v___x_3803_; lean_object* v___x_3804_; lean_object* v___x_3805_; lean_object* v___x_3806_; lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; lean_object* v___x_3811_; lean_object* v___x_3812_; lean_object* v___x_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; lean_object* v___x_3817_; lean_object* v___x_3818_; lean_object* v___x_3819_; lean_object* v___x_3820_; lean_object* v___x_3821_; lean_object* v___x_3822_; lean_object* v___x_3823_; lean_object* v___x_3824_; lean_object* v___x_3825_; lean_object* v___x_3826_; lean_object* v___x_3827_; lean_object* v___x_3828_; lean_object* v___x_3829_; lean_object* v___x_3830_; lean_object* v___x_3831_; lean_object* v___x_3832_; 
v___x_3795_ = l_Lean_Expr_const___override(v___x_3792_, v___x_3794_);
lean_inc_n(v_fst_3674_, 7);
v___x_3796_ = l_Lean_Expr_app___override(v___x_3795_, v_fst_3674_);
lean_inc_ref_n(v___x_3796_, 2);
v___x_3797_ = l_Lean_Expr_app___override(v___x_3796_, v_fst_3781_);
v___x_3798_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20));
lean_inc_ref_n(v___x_3682_, 6);
v___x_3799_ = l_Lean_Expr_const___override(v___x_3798_, v___x_3682_);
v___x_3800_ = l_Lean_Expr_app___override(v___x_3799_, v_fst_3674_);
v___x_3801_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22);
v___x_3802_ = l_Lean_Expr_app___override(v___x_3800_, v___x_3801_);
v___x_3803_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25));
v___x_3804_ = l_Lean_Expr_const___override(v___x_3803_, v___x_3682_);
v___x_3805_ = l_Lean_Expr_app___override(v___x_3804_, v_fst_3674_);
v___x_3806_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28));
v___x_3807_ = l_Lean_Expr_const___override(v___x_3806_, v___x_3682_);
v___x_3808_ = l_Lean_Expr_app___override(v___x_3807_, v_fst_3674_);
v___x_3809_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__30));
v___x_3810_ = l_Lean_Expr_const___override(v___x_3809_, v___x_3682_);
v___x_3811_ = l_Lean_Expr_app___override(v___x_3810_, v_fst_3674_);
v___x_3812_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71));
v___x_3813_ = l_Lean_Expr_const___override(v___x_3812_, v___x_3682_);
v___x_3814_ = l_Lean_Expr_app___override(v___x_3813_, v_fst_3674_);
v___x_3815_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74));
v___x_3816_ = l_Lean_Expr_const___override(v___x_3815_, v___x_3682_);
v___x_3817_ = l_Lean_Expr_app___override(v___x_3816_, v_fst_3674_);
v___x_3818_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76));
v___x_3819_ = l_Lean_Expr_const___override(v___x_3818_, v___x_3682_);
v___x_3820_ = l_Lean_Expr_app___override(v___x_3819_, v_fst_3674_);
v___x_3821_ = l_Lean_Expr_app___override(v___x_3820_, v_a_3791_);
v___x_3822_ = l_Lean_Expr_app___override(v___x_3817_, v___x_3821_);
v___x_3823_ = l_Lean_Expr_app___override(v___x_3814_, v___x_3822_);
v___x_3824_ = l_Lean_Expr_app___override(v___x_3811_, v___x_3823_);
v___x_3825_ = l_Lean_Expr_app___override(v___x_3808_, v___x_3824_);
v___x_3826_ = l_Lean_Expr_app___override(v___x_3805_, v___x_3825_);
v___x_3827_ = l_Lean_Expr_app___override(v___x_3802_, v___x_3826_);
lean_inc_ref_n(v___x_3827_, 2);
v___x_3828_ = l_Lean_Expr_app___override(v___x_3797_, v___x_3827_);
v___x_3829_ = l_Lean_Expr_app___override(v___x_3796_, v_fst_3782_);
v___x_3830_ = l_Lean_Expr_app___override(v___x_3829_, v___x_3827_);
v___x_3831_ = l_Lean_Expr_app___override(v___x_3796_, v_fst_3783_);
v___x_3832_ = l_Lean_Expr_app___override(v___x_3831_, v___x_3827_);
v_fst_3692_ = v___x_3828_;
v_fst_3693_ = v___x_3830_;
v_snd_3694_ = v___x_3832_;
v___y_3695_ = v_a_3647_;
v___y_3696_ = v_a_3648_;
v___y_3697_ = v_a_3649_;
v___y_3698_ = v_a_3650_;
goto v___jp_3691_;
}
}
else
{
lean_object* v_a_3834_; lean_object* v___x_3836_; uint8_t v_isShared_3837_; uint8_t v_isSharedCheck_3841_; 
lean_del_object(v___x_3785_);
lean_dec(v_fst_3783_);
lean_dec(v_fst_3782_);
lean_dec(v_fst_3781_);
lean_dec(v_snd_3690_);
lean_dec_ref(v___x_3682_);
lean_dec(v___x_3678_);
lean_dec(v_fst_3674_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3664_);
v_a_3834_ = lean_ctor_get(v___x_3790_, 0);
v_isSharedCheck_3841_ = !lean_is_exclusive(v___x_3790_);
if (v_isSharedCheck_3841_ == 0)
{
v___x_3836_ = v___x_3790_;
v_isShared_3837_ = v_isSharedCheck_3841_;
goto v_resetjp_3835_;
}
else
{
lean_inc(v_a_3834_);
lean_dec(v___x_3790_);
v___x_3836_ = lean_box(0);
v_isShared_3837_ = v_isSharedCheck_3841_;
goto v_resetjp_3835_;
}
v_resetjp_3835_:
{
lean_object* v___x_3839_; 
if (v_isShared_3837_ == 0)
{
v___x_3839_ = v___x_3836_;
goto v_reusejp_3838_;
}
else
{
lean_object* v_reuseFailAlloc_3840_; 
v_reuseFailAlloc_3840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3840_, 0, v_a_3834_);
v___x_3839_ = v_reuseFailAlloc_3840_;
goto v_reusejp_3838_;
}
v_reusejp_3838_:
{
return v___x_3839_;
}
}
}
}
}
else
{
lean_object* v_fst_3844_; lean_object* v_fst_3845_; lean_object* v_fst_3846_; lean_object* v___x_3847_; lean_object* v___x_3848_; lean_object* v___x_3849_; lean_object* v___x_3850_; 
lean_dec(v___x_3678_);
v_fst_3844_ = lean_ctor_get(v_a_3773_, 0);
lean_inc(v_fst_3844_);
lean_dec(v_a_3773_);
v_fst_3845_ = lean_ctor_get(v_a_3776_, 0);
lean_inc(v_fst_3845_);
lean_dec(v_a_3776_);
v_fst_3846_ = lean_ctor_get(v_a_3779_, 0);
lean_inc(v_fst_3846_);
lean_dec(v_a_3779_);
v___x_3847_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_derive___closed__4));
lean_inc_ref(v___x_3682_);
v___x_3848_ = l_Lean_Expr_const___override(v___x_3847_, v___x_3682_);
lean_inc(v_fst_3674_);
v___x_3849_ = l_Lean_Expr_app___override(v___x_3848_, v_fst_3674_);
v___x_3850_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_3849_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3850_) == 0)
{
lean_object* v_a_3851_; lean_object* v___x_3852_; lean_object* v___x_3853_; lean_object* v___x_3854_; lean_object* v___x_3855_; 
v_a_3851_ = lean_ctor_get(v___x_3850_, 0);
lean_inc(v_a_3851_);
lean_dec_ref_known(v___x_3850_, 1);
v___x_3852_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__4));
lean_inc_ref(v___x_3682_);
v___x_3853_ = l_Lean_Expr_const___override(v___x_3852_, v___x_3682_);
lean_inc(v_fst_3674_);
v___x_3854_ = l_Lean_Expr_app___override(v___x_3853_, v_fst_3674_);
v___x_3855_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_3854_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3855_) == 0)
{
lean_object* v_a_3856_; lean_object* v___x_3857_; lean_object* v___x_3858_; lean_object* v___x_3859_; lean_object* v___x_3860_; lean_object* v___x_3861_; lean_object* v___x_3862_; lean_object* v___x_3863_; lean_object* v___x_3864_; lean_object* v___x_3865_; lean_object* v___x_3866_; lean_object* v___x_3867_; lean_object* v___x_3868_; lean_object* v___x_3869_; lean_object* v___x_3870_; lean_object* v___x_3871_; lean_object* v___x_3872_; lean_object* v___x_3873_; lean_object* v___x_3874_; lean_object* v___x_3875_; lean_object* v___x_3876_; lean_object* v___x_3877_; lean_object* v___x_3878_; lean_object* v___x_3879_; lean_object* v___x_3880_; lean_object* v___x_3881_; lean_object* v___x_3882_; lean_object* v___x_3883_; lean_object* v___x_3884_; lean_object* v___x_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; 
v_a_3856_ = lean_ctor_get(v___x_3855_, 0);
lean_inc(v_a_3856_);
lean_dec_ref_known(v___x_3855_, 1);
v___x_3857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__6));
lean_inc_ref_n(v___x_3682_, 8);
v___x_3858_ = l_Lean_Expr_const___override(v___x_3857_, v___x_3682_);
lean_inc_n(v_fst_3674_, 8);
v___x_3859_ = l_Lean_Expr_app___override(v___x_3858_, v_fst_3674_);
v___x_3860_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__71));
v___x_3861_ = l_Lean_Expr_const___override(v___x_3860_, v___x_3682_);
v___x_3862_ = l_Lean_Expr_app___override(v___x_3861_, v_fst_3674_);
v___x_3863_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__74));
v___x_3864_ = l_Lean_Expr_const___override(v___x_3863_, v___x_3682_);
v___x_3865_ = l_Lean_Expr_app___override(v___x_3864_, v_fst_3674_);
v___x_3866_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__76));
v___x_3867_ = l_Lean_Expr_const___override(v___x_3866_, v___x_3682_);
v___x_3868_ = l_Lean_Expr_app___override(v___x_3867_, v_fst_3674_);
v___x_3869_ = l_Lean_Expr_app___override(v___x_3868_, v_a_3851_);
v___x_3870_ = l_Lean_Expr_app___override(v___x_3865_, v___x_3869_);
v___x_3871_ = l_Lean_Expr_app___override(v___x_3862_, v___x_3870_);
lean_inc_ref(v___x_3871_);
v___x_3872_ = l_Lean_Expr_app___override(v___x_3859_, v___x_3871_);
v___x_3873_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__9));
v___x_3874_ = l_Lean_Expr_const___override(v___x_3873_, v___x_3682_);
v___x_3875_ = l_Lean_Expr_app___override(v___x_3874_, v_fst_3674_);
v___x_3876_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__12));
v___x_3877_ = l_Lean_Expr_const___override(v___x_3876_, v___x_3682_);
v___x_3878_ = l_Lean_Expr_app___override(v___x_3877_, v_fst_3674_);
v___x_3879_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__15));
v___x_3880_ = l_Lean_Expr_const___override(v___x_3879_, v___x_3682_);
v___x_3881_ = l_Lean_Expr_app___override(v___x_3880_, v_fst_3674_);
v___x_3882_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__17));
v___x_3883_ = l_Lean_Expr_const___override(v___x_3882_, v___x_3682_);
v___x_3884_ = l_Lean_Expr_app___override(v___x_3883_, v_fst_3674_);
v___x_3885_ = l_Lean_Expr_app___override(v___x_3884_, v_a_3856_);
v___x_3886_ = l_Lean_Expr_app___override(v___x_3881_, v___x_3885_);
v___x_3887_ = l_Lean_Expr_app___override(v___x_3878_, v___x_3886_);
v___x_3888_ = l_Lean_Expr_app___override(v___x_3875_, v___x_3887_);
lean_inc_ref(v___x_3888_);
v___x_3889_ = l_Lean_Expr_app___override(v___x_3872_, v___x_3888_);
v___x_3890_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_3889_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
if (lean_obj_tag(v___x_3890_) == 0)
{
lean_object* v___x_3891_; lean_object* v___x_3892_; lean_object* v___x_3893_; lean_object* v___x_3894_; lean_object* v___x_3895_; lean_object* v___x_3896_; lean_object* v___x_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; lean_object* v___x_3901_; lean_object* v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; lean_object* v___x_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; lean_object* v___x_3916_; lean_object* v___x_3917_; lean_object* v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; lean_object* v___x_3923_; lean_object* v___x_3924_; 
lean_dec_ref_known(v___x_3890_, 1);
v___x_3891_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__18));
lean_inc_ref_n(v___x_3682_, 6);
v___x_3892_ = l_Lean_Expr_const___override(v___x_3891_, v___x_3682_);
lean_inc_n(v_fst_3674_, 6);
v___x_3893_ = l_Lean_Expr_app___override(v___x_3892_, v_fst_3674_);
v___x_3894_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__21));
v___x_3895_ = l_Lean_Expr_const___override(v___x_3894_, v___x_3682_);
v___x_3896_ = l_Lean_Expr_app___override(v___x_3895_, v_fst_3674_);
v___x_3897_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__24));
v___x_3898_ = l_Lean_Expr_const___override(v___x_3897_, v___x_3682_);
v___x_3899_ = l_Lean_Expr_app___override(v___x_3898_, v_fst_3674_);
v___x_3900_ = l_Lean_Expr_app___override(v___x_3899_, v___x_3888_);
v___x_3901_ = l_Lean_Expr_app___override(v___x_3896_, v___x_3900_);
v___x_3902_ = l_Lean_Expr_app___override(v___x_3893_, v___x_3901_);
v___x_3903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__20));
v___x_3904_ = l_Lean_Expr_const___override(v___x_3903_, v___x_3682_);
v___x_3905_ = l_Lean_Expr_app___override(v___x_3904_, v_fst_3674_);
v___x_3906_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22, &lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__22);
v___x_3907_ = l_Lean_Expr_app___override(v___x_3905_, v___x_3906_);
v___x_3908_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__25));
v___x_3909_ = l_Lean_Expr_const___override(v___x_3908_, v___x_3682_);
v___x_3910_ = l_Lean_Expr_app___override(v___x_3909_, v_fst_3674_);
v___x_3911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__28));
v___x_3912_ = l_Lean_Expr_const___override(v___x_3911_, v___x_3682_);
v___x_3913_ = l_Lean_Expr_app___override(v___x_3912_, v_fst_3674_);
v___x_3914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_CancelDenoms_mkProdPrf___closed__30));
v___x_3915_ = l_Lean_Expr_const___override(v___x_3914_, v___x_3682_);
v___x_3916_ = l_Lean_Expr_app___override(v___x_3915_, v_fst_3674_);
v___x_3917_ = l_Lean_Expr_app___override(v___x_3916_, v___x_3871_);
v___x_3918_ = l_Lean_Expr_app___override(v___x_3913_, v___x_3917_);
v___x_3919_ = l_Lean_Expr_app___override(v___x_3910_, v___x_3918_);
v___x_3920_ = l_Lean_Expr_app___override(v___x_3907_, v___x_3919_);
v___x_3921_ = l_Lean_Expr_app___override(v___x_3902_, v___x_3920_);
lean_inc_ref_n(v___x_3921_, 2);
v___x_3922_ = l_Lean_Expr_app___override(v___x_3921_, v_fst_3844_);
v___x_3923_ = l_Lean_Expr_app___override(v___x_3921_, v_fst_3845_);
v___x_3924_ = l_Lean_Expr_app___override(v___x_3921_, v_fst_3846_);
v_fst_3692_ = v___x_3922_;
v_fst_3693_ = v___x_3923_;
v_snd_3694_ = v___x_3924_;
v___y_3695_ = v_a_3647_;
v___y_3696_ = v_a_3648_;
v___y_3697_ = v_a_3649_;
v___y_3698_ = v_a_3650_;
goto v___jp_3691_;
}
else
{
lean_object* v_a_3925_; lean_object* v___x_3927_; uint8_t v_isShared_3928_; uint8_t v_isSharedCheck_3932_; 
lean_dec_ref(v___x_3888_);
lean_dec_ref(v___x_3871_);
lean_dec(v_fst_3846_);
lean_dec(v_fst_3845_);
lean_dec(v_fst_3844_);
lean_dec(v_snd_3690_);
lean_dec_ref(v___x_3682_);
lean_dec(v_fst_3674_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3664_);
v_a_3925_ = lean_ctor_get(v___x_3890_, 0);
v_isSharedCheck_3932_ = !lean_is_exclusive(v___x_3890_);
if (v_isSharedCheck_3932_ == 0)
{
v___x_3927_ = v___x_3890_;
v_isShared_3928_ = v_isSharedCheck_3932_;
goto v_resetjp_3926_;
}
else
{
lean_inc(v_a_3925_);
lean_dec(v___x_3890_);
v___x_3927_ = lean_box(0);
v_isShared_3928_ = v_isSharedCheck_3932_;
goto v_resetjp_3926_;
}
v_resetjp_3926_:
{
lean_object* v___x_3930_; 
if (v_isShared_3928_ == 0)
{
v___x_3930_ = v___x_3927_;
goto v_reusejp_3929_;
}
else
{
lean_object* v_reuseFailAlloc_3931_; 
v_reuseFailAlloc_3931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3931_, 0, v_a_3925_);
v___x_3930_ = v_reuseFailAlloc_3931_;
goto v_reusejp_3929_;
}
v_reusejp_3929_:
{
return v___x_3930_;
}
}
}
}
else
{
lean_object* v_a_3933_; lean_object* v___x_3935_; uint8_t v_isShared_3936_; uint8_t v_isSharedCheck_3940_; 
lean_dec(v_a_3851_);
lean_dec(v_fst_3846_);
lean_dec(v_fst_3845_);
lean_dec(v_fst_3844_);
lean_dec(v_snd_3690_);
lean_dec_ref(v___x_3682_);
lean_dec(v_fst_3674_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3664_);
v_a_3933_ = lean_ctor_get(v___x_3855_, 0);
v_isSharedCheck_3940_ = !lean_is_exclusive(v___x_3855_);
if (v_isSharedCheck_3940_ == 0)
{
v___x_3935_ = v___x_3855_;
v_isShared_3936_ = v_isSharedCheck_3940_;
goto v_resetjp_3934_;
}
else
{
lean_inc(v_a_3933_);
lean_dec(v___x_3855_);
v___x_3935_ = lean_box(0);
v_isShared_3936_ = v_isSharedCheck_3940_;
goto v_resetjp_3934_;
}
v_resetjp_3934_:
{
lean_object* v___x_3938_; 
if (v_isShared_3936_ == 0)
{
v___x_3938_ = v___x_3935_;
goto v_reusejp_3937_;
}
else
{
lean_object* v_reuseFailAlloc_3939_; 
v_reuseFailAlloc_3939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3939_, 0, v_a_3933_);
v___x_3938_ = v_reuseFailAlloc_3939_;
goto v_reusejp_3937_;
}
v_reusejp_3937_:
{
return v___x_3938_;
}
}
}
}
else
{
lean_object* v_a_3941_; lean_object* v___x_3943_; uint8_t v_isShared_3944_; uint8_t v_isSharedCheck_3948_; 
lean_dec(v_fst_3846_);
lean_dec(v_fst_3845_);
lean_dec(v_fst_3844_);
lean_dec(v_snd_3690_);
lean_dec_ref(v___x_3682_);
lean_dec(v_fst_3674_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3664_);
v_a_3941_ = lean_ctor_get(v___x_3850_, 0);
v_isSharedCheck_3948_ = !lean_is_exclusive(v___x_3850_);
if (v_isSharedCheck_3948_ == 0)
{
v___x_3943_ = v___x_3850_;
v_isShared_3944_ = v_isSharedCheck_3948_;
goto v_resetjp_3942_;
}
else
{
lean_inc(v_a_3941_);
lean_dec(v___x_3850_);
v___x_3943_ = lean_box(0);
v_isShared_3944_ = v_isSharedCheck_3948_;
goto v_resetjp_3942_;
}
v_resetjp_3942_:
{
lean_object* v___x_3946_; 
if (v_isShared_3944_ == 0)
{
v___x_3946_ = v___x_3943_;
goto v_reusejp_3945_;
}
else
{
lean_object* v_reuseFailAlloc_3947_; 
v_reuseFailAlloc_3947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3947_, 0, v_a_3941_);
v___x_3946_ = v_reuseFailAlloc_3947_;
goto v_reusejp_3945_;
}
v_reusejp_3945_:
{
return v___x_3946_;
}
}
}
}
}
else
{
lean_object* v_a_3949_; lean_object* v___x_3951_; uint8_t v_isShared_3952_; uint8_t v_isSharedCheck_3956_; 
lean_dec(v_a_3776_);
lean_dec(v_a_3773_);
lean_dec(v_snd_3690_);
lean_dec_ref(v___x_3682_);
lean_dec(v___x_3678_);
lean_dec(v_fst_3674_);
lean_dec(v_snd_3669_);
lean_dec(v_snd_3665_);
lean_dec(v_fst_3664_);
v_a_3949_ = lean_ctor_get(v___x_3778_, 0);
v_isSharedCheck_3956_ = !lean_is_exclusive(v___x_3778_);
if (v_isSharedCheck_3956_ == 0)
{
v___x_3951_ = v___x_3778_;
v_isShared_3952_ = v_isSharedCheck_3956_;
goto v_resetjp_3950_;
}
else
{
lean_inc(v_a_3949_);
lean_dec(v___x_3778_);
v___x_3951_ = lean_box(0);
v_isShared_3952_ = v_isSharedCheck_3956_;
goto v_resetjp_3950_;
}
v_resetjp_3950_:
{
lean_object* v___x_3954_; 
if (v_isShared_3952_ == 0)
{
v___x_3954_ = v___x_3951_;
goto v_reusejp_3953_;
}
else
{
lean_object* v_reuseFailAlloc_3955_; 
v_reuseFailAlloc_3955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3955_, 0, v_a_3949_);
v___x_3954_ = v_reuseFailAlloc_3955_;
goto v_reusejp_3953_;
}
v_reusejp_3953_:
{
return v___x_3954_;
}
}
}
}
else
{
lean_object* v_a_3957_; lean_object* v___x_3959_; uint8_t v_isShared_3960_; uint8_t v_isSharedCheck_3964_; 
lean_dec(v_a_3773_);
lean_dec(v___x_3770_);
lean_dec(v_snd_3690_);
lean_dec(v_a_3686_);
lean_dec_ref(v___x_3682_);
lean_dec(v___x_3678_);
lean_dec(v_fst_3674_);
lean_dec(v_fst_3673_);
lean_dec(v_snd_3669_);
lean_dec(v_snd_3665_);
lean_dec(v_fst_3664_);
v_a_3957_ = lean_ctor_get(v___x_3775_, 0);
v_isSharedCheck_3964_ = !lean_is_exclusive(v___x_3775_);
if (v_isSharedCheck_3964_ == 0)
{
v___x_3959_ = v___x_3775_;
v_isShared_3960_ = v_isSharedCheck_3964_;
goto v_resetjp_3958_;
}
else
{
lean_inc(v_a_3957_);
lean_dec(v___x_3775_);
v___x_3959_ = lean_box(0);
v_isShared_3960_ = v_isSharedCheck_3964_;
goto v_resetjp_3958_;
}
v_resetjp_3958_:
{
lean_object* v___x_3962_; 
if (v_isShared_3960_ == 0)
{
v___x_3962_ = v___x_3959_;
goto v_reusejp_3961_;
}
else
{
lean_object* v_reuseFailAlloc_3963_; 
v_reuseFailAlloc_3963_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3963_, 0, v_a_3957_);
v___x_3962_ = v_reuseFailAlloc_3963_;
goto v_reusejp_3961_;
}
v_reusejp_3961_:
{
return v___x_3962_;
}
}
}
}
else
{
lean_object* v_a_3965_; lean_object* v___x_3967_; uint8_t v_isShared_3968_; uint8_t v_isSharedCheck_3972_; 
lean_dec(v___x_3770_);
lean_dec(v_snd_3690_);
lean_dec(v_fst_3689_);
lean_dec(v_a_3686_);
lean_dec_ref(v___x_3682_);
lean_dec(v___x_3678_);
lean_dec(v_fst_3674_);
lean_dec(v_fst_3673_);
lean_dec(v_snd_3669_);
lean_dec(v_snd_3665_);
lean_dec(v_fst_3664_);
v_a_3965_ = lean_ctor_get(v___x_3772_, 0);
v_isSharedCheck_3972_ = !lean_is_exclusive(v___x_3772_);
if (v_isSharedCheck_3972_ == 0)
{
v___x_3967_ = v___x_3772_;
v_isShared_3968_ = v_isSharedCheck_3972_;
goto v_resetjp_3966_;
}
else
{
lean_inc(v_a_3965_);
lean_dec(v___x_3772_);
v___x_3967_ = lean_box(0);
v_isShared_3968_ = v_isSharedCheck_3972_;
goto v_resetjp_3966_;
}
v_resetjp_3966_:
{
lean_object* v___x_3970_; 
if (v_isShared_3968_ == 0)
{
v___x_3970_ = v___x_3967_;
goto v_reusejp_3969_;
}
else
{
lean_object* v_reuseFailAlloc_3971_; 
v_reuseFailAlloc_3971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3971_, 0, v_a_3965_);
v___x_3970_ = v_reuseFailAlloc_3971_;
goto v_reusejp_3969_;
}
v_reusejp_3969_:
{
return v___x_3970_;
}
}
}
v___jp_3691_:
{
lean_object* v___x_3699_; 
v___x_3699_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v_fst_3692_, v___y_3695_, v___y_3696_, v___y_3697_, v___y_3698_);
if (lean_obj_tag(v___x_3699_) == 0)
{
lean_object* v_a_3700_; lean_object* v___x_3701_; 
v_a_3700_ = lean_ctor_get(v___x_3699_, 0);
lean_inc(v_a_3700_);
lean_dec_ref_known(v___x_3699_, 1);
v___x_3701_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v_fst_3693_, v___y_3695_, v___y_3696_, v___y_3697_, v___y_3698_);
if (lean_obj_tag(v___x_3701_) == 0)
{
lean_object* v_a_3702_; lean_object* v___x_3703_; 
v_a_3702_ = lean_ctor_get(v___x_3701_, 0);
lean_inc(v_a_3702_);
lean_dec_ref_known(v___x_3701_, 1);
v___x_3703_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum(v_snd_3694_, v___y_3695_, v___y_3696_, v___y_3697_, v___y_3698_);
if (lean_obj_tag(v___x_3703_) == 0)
{
lean_object* v_a_3704_; lean_object* v___x_3705_; lean_object* v___x_3706_; lean_object* v___x_3707_; lean_object* v___x_3708_; lean_object* v___x_3709_; lean_object* v___x_3710_; lean_object* v___x_3711_; lean_object* v___x_3712_; 
v_a_3704_ = lean_ctor_get(v___x_3703_, 0);
lean_inc(v_a_3704_);
lean_dec_ref_known(v___x_3703_, 1);
v___x_3705_ = lean_unsigned_to_nat(5u);
v___x_3706_ = lean_mk_empty_array_with_capacity(v___x_3705_);
v___x_3707_ = lean_array_push(v___x_3706_, v_snd_3669_);
v___x_3708_ = lean_array_push(v___x_3707_, v_snd_3690_);
v___x_3709_ = lean_array_push(v___x_3708_, v_a_3700_);
v___x_3710_ = lean_array_push(v___x_3709_, v_a_3702_);
v___x_3711_ = lean_array_push(v___x_3710_, v_a_3704_);
v___x_3712_ = l_Lean_Meta_mkAppM(v_fst_3664_, v___x_3711_, v___y_3695_, v___y_3696_, v___y_3697_, v___y_3698_);
if (lean_obj_tag(v___x_3712_) == 0)
{
lean_object* v_a_3713_; lean_object* v___x_3714_; 
v_a_3713_ = lean_ctor_get(v___x_3712_, 0);
lean_inc_n(v_a_3713_, 2);
lean_dec_ref_known(v___x_3712_, 1);
lean_inc(v___y_3698_);
lean_inc_ref(v___y_3697_);
lean_inc(v___y_3696_);
lean_inc_ref(v___y_3695_);
v___x_3714_ = lean_infer_type(v_a_3713_, v___y_3695_, v___y_3696_, v___y_3697_, v___y_3698_);
if (lean_obj_tag(v___x_3714_) == 0)
{
lean_object* v_a_3715_; lean_object* v___x_3716_; 
v_a_3715_ = lean_ctor_get(v___x_3714_, 0);
lean_inc(v_a_3715_);
lean_dec_ref_known(v___x_3714_, 1);
v___x_3716_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_findCompLemma(v_a_3715_, v___y_3695_, v___y_3696_, v___y_3697_, v___y_3698_);
if (lean_obj_tag(v___x_3716_) == 0)
{
lean_object* v_a_3717_; 
v_a_3717_ = lean_ctor_get(v___x_3716_, 0);
lean_inc(v_a_3717_);
lean_dec_ref_known(v___x_3716_, 1);
if (lean_obj_tag(v_a_3717_) == 0)
{
lean_object* v___x_3718_; 
v___x_3718_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__2, &lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__2);
v___y_3653_ = v_a_3713_;
v___y_3654_ = v___x_3718_;
goto v___jp_3652_;
}
else
{
lean_object* v_val_3719_; lean_object* v_snd_3720_; lean_object* v_fst_3721_; 
v_val_3719_ = lean_ctor_get(v_a_3717_, 0);
lean_inc(v_val_3719_);
lean_dec_ref_known(v_a_3717_, 1);
v_snd_3720_ = lean_ctor_get(v_val_3719_, 1);
lean_inc(v_snd_3720_);
lean_dec(v_val_3719_);
v_fst_3721_ = lean_ctor_get(v_snd_3720_, 0);
lean_inc(v_fst_3721_);
lean_dec(v_snd_3720_);
v___y_3653_ = v_a_3713_;
v___y_3654_ = v_fst_3721_;
goto v___jp_3652_;
}
}
else
{
lean_object* v_a_3722_; lean_object* v___x_3724_; uint8_t v_isShared_3725_; uint8_t v_isSharedCheck_3729_; 
lean_dec(v_a_3713_);
v_a_3722_ = lean_ctor_get(v___x_3716_, 0);
v_isSharedCheck_3729_ = !lean_is_exclusive(v___x_3716_);
if (v_isSharedCheck_3729_ == 0)
{
v___x_3724_ = v___x_3716_;
v_isShared_3725_ = v_isSharedCheck_3729_;
goto v_resetjp_3723_;
}
else
{
lean_inc(v_a_3722_);
lean_dec(v___x_3716_);
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
lean_object* v_a_3730_; lean_object* v___x_3732_; uint8_t v_isShared_3733_; uint8_t v_isSharedCheck_3737_; 
lean_dec(v_a_3713_);
v_a_3730_ = lean_ctor_get(v___x_3714_, 0);
v_isSharedCheck_3737_ = !lean_is_exclusive(v___x_3714_);
if (v_isSharedCheck_3737_ == 0)
{
v___x_3732_ = v___x_3714_;
v_isShared_3733_ = v_isSharedCheck_3737_;
goto v_resetjp_3731_;
}
else
{
lean_inc(v_a_3730_);
lean_dec(v___x_3714_);
v___x_3732_ = lean_box(0);
v_isShared_3733_ = v_isSharedCheck_3737_;
goto v_resetjp_3731_;
}
v_resetjp_3731_:
{
lean_object* v___x_3735_; 
if (v_isShared_3733_ == 0)
{
v___x_3735_ = v___x_3732_;
goto v_reusejp_3734_;
}
else
{
lean_object* v_reuseFailAlloc_3736_; 
v_reuseFailAlloc_3736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3736_, 0, v_a_3730_);
v___x_3735_ = v_reuseFailAlloc_3736_;
goto v_reusejp_3734_;
}
v_reusejp_3734_:
{
return v___x_3735_;
}
}
}
}
else
{
lean_object* v_a_3738_; lean_object* v___x_3740_; uint8_t v_isShared_3741_; uint8_t v_isSharedCheck_3745_; 
v_a_3738_ = lean_ctor_get(v___x_3712_, 0);
v_isSharedCheck_3745_ = !lean_is_exclusive(v___x_3712_);
if (v_isSharedCheck_3745_ == 0)
{
v___x_3740_ = v___x_3712_;
v_isShared_3741_ = v_isSharedCheck_3745_;
goto v_resetjp_3739_;
}
else
{
lean_inc(v_a_3738_);
lean_dec(v___x_3712_);
v___x_3740_ = lean_box(0);
v_isShared_3741_ = v_isSharedCheck_3745_;
goto v_resetjp_3739_;
}
v_resetjp_3739_:
{
lean_object* v___x_3743_; 
if (v_isShared_3741_ == 0)
{
v___x_3743_ = v___x_3740_;
goto v_reusejp_3742_;
}
else
{
lean_object* v_reuseFailAlloc_3744_; 
v_reuseFailAlloc_3744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3744_, 0, v_a_3738_);
v___x_3743_ = v_reuseFailAlloc_3744_;
goto v_reusejp_3742_;
}
v_reusejp_3742_:
{
return v___x_3743_;
}
}
}
}
else
{
lean_object* v_a_3746_; lean_object* v___x_3748_; uint8_t v_isShared_3749_; uint8_t v_isSharedCheck_3753_; 
lean_dec(v_a_3702_);
lean_dec(v_a_3700_);
lean_dec(v_snd_3690_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3664_);
v_a_3746_ = lean_ctor_get(v___x_3703_, 0);
v_isSharedCheck_3753_ = !lean_is_exclusive(v___x_3703_);
if (v_isSharedCheck_3753_ == 0)
{
v___x_3748_ = v___x_3703_;
v_isShared_3749_ = v_isSharedCheck_3753_;
goto v_resetjp_3747_;
}
else
{
lean_inc(v_a_3746_);
lean_dec(v___x_3703_);
v___x_3748_ = lean_box(0);
v_isShared_3749_ = v_isSharedCheck_3753_;
goto v_resetjp_3747_;
}
v_resetjp_3747_:
{
lean_object* v___x_3751_; 
if (v_isShared_3749_ == 0)
{
v___x_3751_ = v___x_3748_;
goto v_reusejp_3750_;
}
else
{
lean_object* v_reuseFailAlloc_3752_; 
v_reuseFailAlloc_3752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3752_, 0, v_a_3746_);
v___x_3751_ = v_reuseFailAlloc_3752_;
goto v_reusejp_3750_;
}
v_reusejp_3750_:
{
return v___x_3751_;
}
}
}
}
else
{
lean_object* v_a_3754_; lean_object* v___x_3756_; uint8_t v_isShared_3757_; uint8_t v_isSharedCheck_3761_; 
lean_dec(v_a_3700_);
lean_dec_ref(v_snd_3694_);
lean_dec(v_snd_3690_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3664_);
v_a_3754_ = lean_ctor_get(v___x_3701_, 0);
v_isSharedCheck_3761_ = !lean_is_exclusive(v___x_3701_);
if (v_isSharedCheck_3761_ == 0)
{
v___x_3756_ = v___x_3701_;
v_isShared_3757_ = v_isSharedCheck_3761_;
goto v_resetjp_3755_;
}
else
{
lean_inc(v_a_3754_);
lean_dec(v___x_3701_);
v___x_3756_ = lean_box(0);
v_isShared_3757_ = v_isSharedCheck_3761_;
goto v_resetjp_3755_;
}
v_resetjp_3755_:
{
lean_object* v___x_3759_; 
if (v_isShared_3757_ == 0)
{
v___x_3759_ = v___x_3756_;
goto v_reusejp_3758_;
}
else
{
lean_object* v_reuseFailAlloc_3760_; 
v_reuseFailAlloc_3760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3760_, 0, v_a_3754_);
v___x_3759_ = v_reuseFailAlloc_3760_;
goto v_reusejp_3758_;
}
v_reusejp_3758_:
{
return v___x_3759_;
}
}
}
}
else
{
lean_object* v_a_3762_; lean_object* v___x_3764_; uint8_t v_isShared_3765_; uint8_t v_isSharedCheck_3769_; 
lean_dec_ref(v_snd_3694_);
lean_dec_ref(v_fst_3693_);
lean_dec(v_snd_3690_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3664_);
v_a_3762_ = lean_ctor_get(v___x_3699_, 0);
v_isSharedCheck_3769_ = !lean_is_exclusive(v___x_3699_);
if (v_isSharedCheck_3769_ == 0)
{
v___x_3764_ = v___x_3699_;
v_isShared_3765_ = v_isSharedCheck_3769_;
goto v_resetjp_3763_;
}
else
{
lean_inc(v_a_3762_);
lean_dec(v___x_3699_);
v___x_3764_ = lean_box(0);
v_isShared_3765_ = v_isSharedCheck_3769_;
goto v_resetjp_3763_;
}
v_resetjp_3763_:
{
lean_object* v___x_3767_; 
if (v_isShared_3765_ == 0)
{
v___x_3767_ = v___x_3764_;
goto v_reusejp_3766_;
}
else
{
lean_object* v_reuseFailAlloc_3768_; 
v_reuseFailAlloc_3768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3768_, 0, v_a_3762_);
v___x_3767_ = v_reuseFailAlloc_3768_;
goto v_reusejp_3766_;
}
v_reusejp_3766_:
{
return v___x_3767_;
}
}
}
}
}
else
{
lean_object* v_a_3973_; lean_object* v___x_3975_; uint8_t v_isShared_3976_; uint8_t v_isSharedCheck_3980_; 
lean_dec(v_a_3686_);
lean_dec_ref(v___x_3682_);
lean_dec(v___x_3678_);
lean_dec(v_fst_3674_);
lean_dec(v_fst_3673_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3668_);
lean_dec(v_snd_3665_);
lean_dec(v_fst_3664_);
v_a_3973_ = lean_ctor_get(v___x_3687_, 0);
v_isSharedCheck_3980_ = !lean_is_exclusive(v___x_3687_);
if (v_isSharedCheck_3980_ == 0)
{
v___x_3975_ = v___x_3687_;
v_isShared_3976_ = v_isSharedCheck_3980_;
goto v_resetjp_3974_;
}
else
{
lean_inc(v_a_3973_);
lean_dec(v___x_3687_);
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
lean_object* v_a_3981_; lean_object* v___x_3983_; uint8_t v_isShared_3984_; uint8_t v_isSharedCheck_3988_; 
lean_dec_ref(v___x_3682_);
lean_dec(v___x_3678_);
lean_dec(v_fst_3674_);
lean_dec(v_fst_3673_);
lean_dec(v_snd_3669_);
lean_dec(v_fst_3668_);
lean_dec(v_snd_3665_);
lean_dec(v_fst_3664_);
lean_dec(v_fst_3663_);
v_a_3981_ = lean_ctor_get(v___x_3685_, 0);
v_isSharedCheck_3988_ = !lean_is_exclusive(v___x_3685_);
if (v_isSharedCheck_3988_ == 0)
{
v___x_3983_ = v___x_3685_;
v_isShared_3984_ = v_isSharedCheck_3988_;
goto v_resetjp_3982_;
}
else
{
lean_inc(v_a_3981_);
lean_dec(v___x_3685_);
v___x_3983_ = lean_box(0);
v_isShared_3984_ = v_isSharedCheck_3988_;
goto v_resetjp_3982_;
}
v_resetjp_3982_:
{
lean_object* v___x_3986_; 
if (v_isShared_3984_ == 0)
{
v___x_3986_ = v___x_3983_;
goto v_reusejp_3985_;
}
else
{
lean_object* v_reuseFailAlloc_3987_; 
v_reuseFailAlloc_3987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3987_, 0, v_a_3981_);
v___x_3986_ = v_reuseFailAlloc_3987_;
goto v_reusejp_3985_;
}
v_reusejp_3985_:
{
return v___x_3986_;
}
}
}
}
}
}
else
{
lean_object* v_a_3992_; lean_object* v___x_3994_; uint8_t v_isShared_3995_; uint8_t v_isSharedCheck_3999_; 
lean_dec(v_snd_3669_);
lean_dec(v_fst_3668_);
lean_dec(v_snd_3665_);
lean_dec(v_fst_3664_);
lean_dec(v_fst_3663_);
v_a_3992_ = lean_ctor_get(v___x_3670_, 0);
v_isSharedCheck_3999_ = !lean_is_exclusive(v___x_3670_);
if (v_isSharedCheck_3999_ == 0)
{
v___x_3994_ = v___x_3670_;
v_isShared_3995_ = v_isSharedCheck_3999_;
goto v_resetjp_3993_;
}
else
{
lean_inc(v_a_3992_);
lean_dec(v___x_3670_);
v___x_3994_ = lean_box(0);
v_isShared_3995_ = v_isSharedCheck_3999_;
goto v_resetjp_3993_;
}
v_resetjp_3993_:
{
lean_object* v___x_3997_; 
if (v_isShared_3995_ == 0)
{
v___x_3997_ = v___x_3994_;
goto v_reusejp_3996_;
}
else
{
lean_object* v_reuseFailAlloc_3998_; 
v_reuseFailAlloc_3998_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3998_, 0, v_a_3992_);
v___x_3997_ = v_reuseFailAlloc_3998_;
goto v_reusejp_3996_;
}
v_reusejp_3996_:
{
return v___x_3997_;
}
}
}
}
else
{
lean_object* v_a_4000_; lean_object* v___x_4002_; uint8_t v_isShared_4003_; uint8_t v_isSharedCheck_4007_; 
lean_dec(v_snd_3665_);
lean_dec(v_fst_3664_);
lean_dec(v_fst_3663_);
lean_dec(v_fst_3662_);
v_a_4000_ = lean_ctor_get(v___x_3666_, 0);
v_isSharedCheck_4007_ = !lean_is_exclusive(v___x_3666_);
if (v_isSharedCheck_4007_ == 0)
{
v___x_4002_ = v___x_3666_;
v_isShared_4003_ = v_isSharedCheck_4007_;
goto v_resetjp_4001_;
}
else
{
lean_inc(v_a_4000_);
lean_dec(v___x_3666_);
v___x_4002_ = lean_box(0);
v_isShared_4003_ = v_isSharedCheck_4007_;
goto v_resetjp_4001_;
}
v_resetjp_4001_:
{
lean_object* v___x_4005_; 
if (v_isShared_4003_ == 0)
{
v___x_4005_ = v___x_4002_;
goto v_reusejp_4004_;
}
else
{
lean_object* v_reuseFailAlloc_4006_; 
v_reuseFailAlloc_4006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4006_, 0, v_a_4000_);
v___x_4005_ = v_reuseFailAlloc_4006_;
goto v_reusejp_4004_;
}
v_reusejp_4004_:
{
return v___x_4005_;
}
}
}
}
else
{
lean_object* v___x_4008_; lean_object* v___x_4009_; 
lean_dec(v_a_3658_);
v___x_4008_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__26, &lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___closed__26);
v___x_4009_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0___redArg(v___x_4008_, v_a_3647_, v_a_3648_, v_a_3649_, v_a_3650_);
return v___x_4009_;
}
}
else
{
lean_object* v_a_4010_; lean_object* v___x_4012_; uint8_t v_isShared_4013_; uint8_t v_isSharedCheck_4017_; 
v_a_4010_ = lean_ctor_get(v___x_3657_, 0);
v_isSharedCheck_4017_ = !lean_is_exclusive(v___x_3657_);
if (v_isSharedCheck_4017_ == 0)
{
v___x_4012_ = v___x_3657_;
v_isShared_4013_ = v_isSharedCheck_4017_;
goto v_resetjp_4011_;
}
else
{
lean_inc(v_a_4010_);
lean_dec(v___x_3657_);
v___x_4012_ = lean_box(0);
v_isShared_4013_ = v_isSharedCheck_4017_;
goto v_resetjp_4011_;
}
v_resetjp_4011_:
{
lean_object* v___x_4015_; 
if (v_isShared_4013_ == 0)
{
v___x_4015_ = v___x_4012_;
goto v_reusejp_4014_;
}
else
{
lean_object* v_reuseFailAlloc_4016_; 
v_reuseFailAlloc_4016_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4016_, 0, v_a_4010_);
v___x_4015_ = v_reuseFailAlloc_4016_;
goto v_reusejp_4014_;
}
v_reusejp_4014_:
{
return v___x_4015_;
}
}
}
v___jp_3652_:
{
lean_object* v___x_3655_; lean_object* v___x_3656_; 
v___x_3655_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3655_, 0, v___y_3654_);
lean_ctor_set(v___x_3655_, 1, v___y_3653_);
v___x_3656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3656_, 0, v___x_3655_);
return v___x_3656_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType___boxed(lean_object* v_h_4018_, lean_object* v_a_4019_, lean_object* v_a_4020_, lean_object* v_a_4021_, lean_object* v_a_4022_, lean_object* v_a_4023_){
_start:
{
lean_object* v_res_4024_; 
v_res_4024_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType(v_h_4018_, v_a_4019_, v_a_4020_, v_a_4021_, v_a_4022_);
lean_dec(v_a_4022_);
lean_dec_ref(v_a_4021_);
lean_dec(v_a_4020_);
lean_dec_ref(v_a_4019_);
return v_res_4024_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__8(void){
_start:
{
lean_object* v___x_4040_; lean_object* v___x_4041_; lean_object* v___x_4042_; 
v___x_4040_ = l_Lean_Parser_Tactic_location;
v___x_4041_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__7));
v___x_4042_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4042_, 0, v___x_4041_);
lean_ctor_set(v___x_4042_, 1, v___x_4040_);
return v___x_4042_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9(void){
_start:
{
lean_object* v___x_4043_; lean_object* v___x_4044_; lean_object* v___x_4045_; lean_object* v___x_4046_; 
v___x_4043_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__8, &lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__8);
v___x_4044_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__5));
v___x_4045_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__3));
v___x_4046_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_4046_, 0, v___x_4045_);
lean_ctor_set(v___x_4046_, 1, v___x_4044_);
lean_ctor_set(v___x_4046_, 2, v___x_4043_);
return v___x_4046_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__10(void){
_start:
{
lean_object* v___x_4047_; lean_object* v___x_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; 
v___x_4047_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9, &lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9);
v___x_4048_ = lean_unsigned_to_nat(1022u);
v___x_4049_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__1));
v___x_4050_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4050_, 0, v___x_4049_);
lean_ctor_set(v___x_4050_, 1, v___x_4048_);
lean_ctor_set(v___x_4050_, 2, v___x_4047_);
return v___x_4050_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_cancelDenoms(void){
_start:
{
lean_object* v___x_4051_; 
v___x_4051_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__10, &lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__10);
return v___x_4051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___redArg(lean_object* v_e_4052_, lean_object* v___y_4053_){
_start:
{
uint8_t v___x_4055_; 
v___x_4055_ = l_Lean_Expr_hasMVar(v_e_4052_);
if (v___x_4055_ == 0)
{
lean_object* v___x_4056_; 
v___x_4056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4056_, 0, v_e_4052_);
return v___x_4056_;
}
else
{
lean_object* v___x_4057_; lean_object* v_mctx_4058_; lean_object* v___x_4059_; lean_object* v_fst_4060_; lean_object* v_snd_4061_; lean_object* v___x_4062_; lean_object* v_cache_4063_; lean_object* v_zetaDeltaFVarIds_4064_; lean_object* v_postponed_4065_; lean_object* v_diag_4066_; lean_object* v___x_4068_; uint8_t v_isShared_4069_; uint8_t v_isSharedCheck_4075_; 
v___x_4057_ = lean_st_ref_get(v___y_4053_);
v_mctx_4058_ = lean_ctor_get(v___x_4057_, 0);
lean_inc_ref(v_mctx_4058_);
lean_dec(v___x_4057_);
v___x_4059_ = l_Lean_instantiateMVarsCore(v_mctx_4058_, v_e_4052_);
v_fst_4060_ = lean_ctor_get(v___x_4059_, 0);
lean_inc(v_fst_4060_);
v_snd_4061_ = lean_ctor_get(v___x_4059_, 1);
lean_inc(v_snd_4061_);
lean_dec_ref(v___x_4059_);
v___x_4062_ = lean_st_ref_take(v___y_4053_);
v_cache_4063_ = lean_ctor_get(v___x_4062_, 1);
v_zetaDeltaFVarIds_4064_ = lean_ctor_get(v___x_4062_, 2);
v_postponed_4065_ = lean_ctor_get(v___x_4062_, 3);
v_diag_4066_ = lean_ctor_get(v___x_4062_, 4);
v_isSharedCheck_4075_ = !lean_is_exclusive(v___x_4062_);
if (v_isSharedCheck_4075_ == 0)
{
lean_object* v_unused_4076_; 
v_unused_4076_ = lean_ctor_get(v___x_4062_, 0);
lean_dec(v_unused_4076_);
v___x_4068_ = v___x_4062_;
v_isShared_4069_ = v_isSharedCheck_4075_;
goto v_resetjp_4067_;
}
else
{
lean_inc(v_diag_4066_);
lean_inc(v_postponed_4065_);
lean_inc(v_zetaDeltaFVarIds_4064_);
lean_inc(v_cache_4063_);
lean_dec(v___x_4062_);
v___x_4068_ = lean_box(0);
v_isShared_4069_ = v_isSharedCheck_4075_;
goto v_resetjp_4067_;
}
v_resetjp_4067_:
{
lean_object* v___x_4071_; 
if (v_isShared_4069_ == 0)
{
lean_ctor_set(v___x_4068_, 0, v_snd_4061_);
v___x_4071_ = v___x_4068_;
goto v_reusejp_4070_;
}
else
{
lean_object* v_reuseFailAlloc_4074_; 
v_reuseFailAlloc_4074_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4074_, 0, v_snd_4061_);
lean_ctor_set(v_reuseFailAlloc_4074_, 1, v_cache_4063_);
lean_ctor_set(v_reuseFailAlloc_4074_, 2, v_zetaDeltaFVarIds_4064_);
lean_ctor_set(v_reuseFailAlloc_4074_, 3, v_postponed_4065_);
lean_ctor_set(v_reuseFailAlloc_4074_, 4, v_diag_4066_);
v___x_4071_ = v_reuseFailAlloc_4074_;
goto v_reusejp_4070_;
}
v_reusejp_4070_:
{
lean_object* v___x_4072_; lean_object* v___x_4073_; 
v___x_4072_ = lean_st_ref_set(v___y_4053_, v___x_4071_);
v___x_4073_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4073_, 0, v_fst_4060_);
return v___x_4073_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___redArg___boxed(lean_object* v_e_4077_, lean_object* v___y_4078_, lean_object* v___y_4079_){
_start:
{
lean_object* v_res_4080_; 
v_res_4080_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___redArg(v_e_4077_, v___y_4078_);
lean_dec(v___y_4078_);
return v_res_4080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0(lean_object* v_e_4081_, lean_object* v___y_4082_, lean_object* v___y_4083_, lean_object* v___y_4084_, lean_object* v___y_4085_, lean_object* v___y_4086_, lean_object* v___y_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_){
_start:
{
lean_object* v___x_4091_; 
v___x_4091_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___redArg(v_e_4081_, v___y_4087_);
return v___x_4091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___boxed(lean_object* v_e_4092_, lean_object* v___y_4093_, lean_object* v___y_4094_, lean_object* v___y_4095_, lean_object* v___y_4096_, lean_object* v___y_4097_, lean_object* v___y_4098_, lean_object* v___y_4099_, lean_object* v___y_4100_, lean_object* v___y_4101_){
_start:
{
lean_object* v_res_4102_; 
v_res_4102_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0(v_e_4092_, v___y_4093_, v___y_4094_, v___y_4095_, v___y_4096_, v___y_4097_, v___y_4098_, v___y_4099_, v___y_4100_);
lean_dec(v___y_4100_);
lean_dec_ref(v___y_4099_);
lean_dec(v___y_4098_);
lean_dec_ref(v___y_4097_);
lean_dec(v___y_4096_);
lean_dec_ref(v___y_4095_);
lean_dec(v___y_4094_);
lean_dec_ref(v___y_4093_);
return v_res_4102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___redArg(lean_object* v_mvarId_4103_, lean_object* v_x_4104_, lean_object* v___y_4105_, lean_object* v___y_4106_, lean_object* v___y_4107_, lean_object* v___y_4108_){
_start:
{
lean_object* v___x_4110_; 
v___x_4110_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_4103_, v_x_4104_, v___y_4105_, v___y_4106_, v___y_4107_, v___y_4108_);
if (lean_obj_tag(v___x_4110_) == 0)
{
lean_object* v_a_4111_; lean_object* v___x_4113_; uint8_t v_isShared_4114_; uint8_t v_isSharedCheck_4118_; 
v_a_4111_ = lean_ctor_get(v___x_4110_, 0);
v_isSharedCheck_4118_ = !lean_is_exclusive(v___x_4110_);
if (v_isSharedCheck_4118_ == 0)
{
v___x_4113_ = v___x_4110_;
v_isShared_4114_ = v_isSharedCheck_4118_;
goto v_resetjp_4112_;
}
else
{
lean_inc(v_a_4111_);
lean_dec(v___x_4110_);
v___x_4113_ = lean_box(0);
v_isShared_4114_ = v_isSharedCheck_4118_;
goto v_resetjp_4112_;
}
v_resetjp_4112_:
{
lean_object* v___x_4116_; 
if (v_isShared_4114_ == 0)
{
v___x_4116_ = v___x_4113_;
goto v_reusejp_4115_;
}
else
{
lean_object* v_reuseFailAlloc_4117_; 
v_reuseFailAlloc_4117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4117_, 0, v_a_4111_);
v___x_4116_ = v_reuseFailAlloc_4117_;
goto v_reusejp_4115_;
}
v_reusejp_4115_:
{
return v___x_4116_;
}
}
}
else
{
lean_object* v_a_4119_; lean_object* v___x_4121_; uint8_t v_isShared_4122_; uint8_t v_isSharedCheck_4126_; 
v_a_4119_ = lean_ctor_get(v___x_4110_, 0);
v_isSharedCheck_4126_ = !lean_is_exclusive(v___x_4110_);
if (v_isSharedCheck_4126_ == 0)
{
v___x_4121_ = v___x_4110_;
v_isShared_4122_ = v_isSharedCheck_4126_;
goto v_resetjp_4120_;
}
else
{
lean_inc(v_a_4119_);
lean_dec(v___x_4110_);
v___x_4121_ = lean_box(0);
v_isShared_4122_ = v_isSharedCheck_4126_;
goto v_resetjp_4120_;
}
v_resetjp_4120_:
{
lean_object* v___x_4124_; 
if (v_isShared_4122_ == 0)
{
v___x_4124_ = v___x_4121_;
goto v_reusejp_4123_;
}
else
{
lean_object* v_reuseFailAlloc_4125_; 
v_reuseFailAlloc_4125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4125_, 0, v_a_4119_);
v___x_4124_ = v_reuseFailAlloc_4125_;
goto v_reusejp_4123_;
}
v_reusejp_4123_:
{
return v___x_4124_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___redArg___boxed(lean_object* v_mvarId_4127_, lean_object* v_x_4128_, lean_object* v___y_4129_, lean_object* v___y_4130_, lean_object* v___y_4131_, lean_object* v___y_4132_, lean_object* v___y_4133_){
_start:
{
lean_object* v_res_4134_; 
v_res_4134_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___redArg(v_mvarId_4127_, v_x_4128_, v___y_4129_, v___y_4130_, v___y_4131_, v___y_4132_);
lean_dec(v___y_4132_);
lean_dec_ref(v___y_4131_);
lean_dec(v___y_4130_);
lean_dec_ref(v___y_4129_);
return v_res_4134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1(lean_object* v_00_u03b1_4135_, lean_object* v_mvarId_4136_, lean_object* v_x_4137_, lean_object* v___y_4138_, lean_object* v___y_4139_, lean_object* v___y_4140_, lean_object* v___y_4141_){
_start:
{
lean_object* v___x_4143_; 
v___x_4143_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___redArg(v_mvarId_4136_, v_x_4137_, v___y_4138_, v___y_4139_, v___y_4140_, v___y_4141_);
return v___x_4143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___boxed(lean_object* v_00_u03b1_4144_, lean_object* v_mvarId_4145_, lean_object* v_x_4146_, lean_object* v___y_4147_, lean_object* v___y_4148_, lean_object* v___y_4149_, lean_object* v___y_4150_, lean_object* v___y_4151_){
_start:
{
lean_object* v_res_4152_; 
v_res_4152_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1(v_00_u03b1_4144_, v_mvarId_4145_, v_x_4146_, v___y_4147_, v___y_4148_, v___y_4149_, v___y_4150_);
lean_dec(v___y_4150_);
lean_dec_ref(v___y_4149_);
lean_dec(v___y_4148_);
lean_dec_ref(v___y_4147_);
return v_res_4152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__0(lean_object* v_snd_4153_, lean_object* v___x_4154_, lean_object* v_fst_4155_, lean_object* v_g_4156_, lean_object* v_fvar_4157_, lean_object* v___y_4158_, lean_object* v___y_4159_, lean_object* v___y_4160_, lean_object* v___y_4161_){
_start:
{
lean_object* v___x_4163_; 
v___x_4163_ = l_Lean_Meta_mkEqMP(v_snd_4153_, v___x_4154_, v___y_4158_, v___y_4159_, v___y_4160_, v___y_4161_);
if (lean_obj_tag(v___x_4163_) == 0)
{
lean_object* v_a_4164_; lean_object* v___x_4165_; lean_object* v___x_4166_; lean_object* v___x_4167_; 
v_a_4164_ = lean_ctor_get(v___x_4163_, 0);
lean_inc(v_a_4164_);
lean_dec_ref_known(v___x_4163_, 1);
v___x_4165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4165_, 0, v_fst_4155_);
v___x_4166_ = lean_box(0);
v___x_4167_ = l_Lean_MVarId_replace(v_g_4156_, v_fvar_4157_, v_a_4164_, v___x_4165_, v___x_4166_, v___y_4158_, v___y_4159_, v___y_4160_, v___y_4161_);
return v___x_4167_;
}
else
{
lean_object* v_a_4168_; lean_object* v___x_4170_; uint8_t v_isShared_4171_; uint8_t v_isSharedCheck_4175_; 
lean_dec(v_fvar_4157_);
lean_dec(v_g_4156_);
lean_dec_ref(v_fst_4155_);
v_a_4168_ = lean_ctor_get(v___x_4163_, 0);
v_isSharedCheck_4175_ = !lean_is_exclusive(v___x_4163_);
if (v_isSharedCheck_4175_ == 0)
{
v___x_4170_ = v___x_4163_;
v_isShared_4171_ = v_isSharedCheck_4175_;
goto v_resetjp_4169_;
}
else
{
lean_inc(v_a_4168_);
lean_dec(v___x_4163_);
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
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__0___boxed(lean_object* v_snd_4176_, lean_object* v___x_4177_, lean_object* v_fst_4178_, lean_object* v_g_4179_, lean_object* v_fvar_4180_, lean_object* v___y_4181_, lean_object* v___y_4182_, lean_object* v___y_4183_, lean_object* v___y_4184_, lean_object* v___y_4185_){
_start:
{
lean_object* v_res_4186_; 
v_res_4186_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__0(v_snd_4176_, v___x_4177_, v_fst_4178_, v_g_4179_, v_fvar_4180_, v___y_4181_, v___y_4182_, v___y_4183_, v___y_4184_);
lean_dec(v___y_4184_);
lean_dec_ref(v___y_4183_);
lean_dec(v___y_4182_);
lean_dec_ref(v___y_4181_);
return v_res_4186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__1(lean_object* v_fvar_4187_, lean_object* v_snd_4188_, lean_object* v_fst_4189_, lean_object* v_g_4190_, lean_object* v___y_4191_, lean_object* v___y_4192_, lean_object* v___y_4193_, lean_object* v___y_4194_){
_start:
{
lean_object* v___x_4196_; lean_object* v___f_4197_; lean_object* v___x_4198_; 
lean_inc(v_fvar_4187_);
v___x_4196_ = l_Lean_mkFVar(v_fvar_4187_);
lean_inc(v_g_4190_);
v___f_4197_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__0___boxed), 10, 5);
lean_closure_set(v___f_4197_, 0, v_snd_4188_);
lean_closure_set(v___f_4197_, 1, v___x_4196_);
lean_closure_set(v___f_4197_, 2, v_fst_4189_);
lean_closure_set(v___f_4197_, 3, v_g_4190_);
lean_closure_set(v___f_4197_, 4, v_fvar_4187_);
v___x_4198_ = lp_mathlib_Lean_MVarId_withContext___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__1___redArg(v_g_4190_, v___f_4197_, v___y_4191_, v___y_4192_, v___y_4193_, v___y_4194_);
if (lean_obj_tag(v___x_4198_) == 0)
{
lean_object* v_a_4199_; lean_object* v___x_4201_; uint8_t v_isShared_4202_; uint8_t v_isSharedCheck_4207_; 
v_a_4199_ = lean_ctor_get(v___x_4198_, 0);
v_isSharedCheck_4207_ = !lean_is_exclusive(v___x_4198_);
if (v_isSharedCheck_4207_ == 0)
{
v___x_4201_ = v___x_4198_;
v_isShared_4202_ = v_isSharedCheck_4207_;
goto v_resetjp_4200_;
}
else
{
lean_inc(v_a_4199_);
lean_dec(v___x_4198_);
v___x_4201_ = lean_box(0);
v_isShared_4202_ = v_isSharedCheck_4207_;
goto v_resetjp_4200_;
}
v_resetjp_4200_:
{
lean_object* v_mvarId_4203_; lean_object* v___x_4205_; 
v_mvarId_4203_ = lean_ctor_get(v_a_4199_, 1);
lean_inc(v_mvarId_4203_);
lean_dec(v_a_4199_);
if (v_isShared_4202_ == 0)
{
lean_ctor_set(v___x_4201_, 0, v_mvarId_4203_);
v___x_4205_ = v___x_4201_;
goto v_reusejp_4204_;
}
else
{
lean_object* v_reuseFailAlloc_4206_; 
v_reuseFailAlloc_4206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4206_, 0, v_mvarId_4203_);
v___x_4205_ = v_reuseFailAlloc_4206_;
goto v_reusejp_4204_;
}
v_reusejp_4204_:
{
return v___x_4205_;
}
}
}
else
{
lean_object* v_a_4208_; lean_object* v___x_4210_; uint8_t v_isShared_4211_; uint8_t v_isSharedCheck_4215_; 
v_a_4208_ = lean_ctor_get(v___x_4198_, 0);
v_isSharedCheck_4215_ = !lean_is_exclusive(v___x_4198_);
if (v_isSharedCheck_4215_ == 0)
{
v___x_4210_ = v___x_4198_;
v_isShared_4211_ = v_isSharedCheck_4215_;
goto v_resetjp_4209_;
}
else
{
lean_inc(v_a_4208_);
lean_dec(v___x_4198_);
v___x_4210_ = lean_box(0);
v_isShared_4211_ = v_isSharedCheck_4215_;
goto v_resetjp_4209_;
}
v_resetjp_4209_:
{
lean_object* v___x_4213_; 
if (v_isShared_4211_ == 0)
{
v___x_4213_ = v___x_4210_;
goto v_reusejp_4212_;
}
else
{
lean_object* v_reuseFailAlloc_4214_; 
v_reuseFailAlloc_4214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4214_, 0, v_a_4208_);
v___x_4213_ = v_reuseFailAlloc_4214_;
goto v_reusejp_4212_;
}
v_reusejp_4212_:
{
return v___x_4213_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__1___boxed(lean_object* v_fvar_4216_, lean_object* v_snd_4217_, lean_object* v_fst_4218_, lean_object* v_g_4219_, lean_object* v___y_4220_, lean_object* v___y_4221_, lean_object* v___y_4222_, lean_object* v___y_4223_, lean_object* v___y_4224_){
_start:
{
lean_object* v_res_4225_; 
v_res_4225_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__1(v_fvar_4216_, v_snd_4217_, v_fst_4218_, v_g_4219_, v___y_4220_, v___y_4221_, v___y_4222_, v___y_4223_);
lean_dec(v___y_4223_);
lean_dec_ref(v___y_4222_);
lean_dec(v___y_4221_);
lean_dec_ref(v___y_4220_);
return v_res_4225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt(lean_object* v_fvar_4226_, lean_object* v_a_4227_, lean_object* v_a_4228_, lean_object* v_a_4229_, lean_object* v_a_4230_, lean_object* v_a_4231_, lean_object* v_a_4232_, lean_object* v_a_4233_, lean_object* v_a_4234_){
_start:
{
lean_object* v___x_4236_; 
lean_inc(v_fvar_4226_);
v___x_4236_ = l_Lean_FVarId_getDecl___redArg(v_fvar_4226_, v_a_4231_, v_a_4233_, v_a_4234_);
if (lean_obj_tag(v___x_4236_) == 0)
{
lean_object* v_a_4237_; lean_object* v___x_4238_; lean_object* v___x_4239_; lean_object* v_a_4240_; lean_object* v___x_4241_; 
v_a_4237_ = lean_ctor_get(v___x_4236_, 0);
lean_inc(v_a_4237_);
lean_dec_ref_known(v___x_4236_, 1);
v___x_4238_ = l_Lean_LocalDecl_type(v_a_4237_);
lean_dec(v_a_4237_);
v___x_4239_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt_spec__0___redArg(v___x_4238_, v_a_4232_);
v_a_4240_ = lean_ctor_get(v___x_4239_, 0);
lean_inc(v_a_4240_);
lean_dec_ref(v___x_4239_);
v___x_4241_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType(v_a_4240_, v_a_4231_, v_a_4232_, v_a_4233_, v_a_4234_);
if (lean_obj_tag(v___x_4241_) == 0)
{
lean_object* v_a_4242_; lean_object* v_fst_4243_; lean_object* v_snd_4244_; lean_object* v___f_4245_; lean_object* v___x_4246_; 
v_a_4242_ = lean_ctor_get(v___x_4241_, 0);
lean_inc(v_a_4242_);
lean_dec_ref_known(v___x_4241_, 1);
v_fst_4243_ = lean_ctor_get(v_a_4242_, 0);
lean_inc(v_fst_4243_);
v_snd_4244_ = lean_ctor_get(v_a_4242_, 1);
lean_inc(v_snd_4244_);
lean_dec(v_a_4242_);
v___f_4245_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___lam__1___boxed), 9, 3);
lean_closure_set(v___f_4245_, 0, v_fvar_4226_);
lean_closure_set(v___f_4245_, 1, v_snd_4244_);
lean_closure_set(v___f_4245_, 2, v_fst_4243_);
v___x_4246_ = lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(v___f_4245_, v_a_4227_, v_a_4228_, v_a_4229_, v_a_4230_, v_a_4231_, v_a_4232_, v_a_4233_, v_a_4234_);
return v___x_4246_;
}
else
{
lean_object* v_a_4247_; lean_object* v___x_4249_; uint8_t v_isShared_4250_; uint8_t v_isSharedCheck_4254_; 
lean_dec(v_fvar_4226_);
v_a_4247_ = lean_ctor_get(v___x_4241_, 0);
v_isSharedCheck_4254_ = !lean_is_exclusive(v___x_4241_);
if (v_isSharedCheck_4254_ == 0)
{
v___x_4249_ = v___x_4241_;
v_isShared_4250_ = v_isSharedCheck_4254_;
goto v_resetjp_4248_;
}
else
{
lean_inc(v_a_4247_);
lean_dec(v___x_4241_);
v___x_4249_ = lean_box(0);
v_isShared_4250_ = v_isSharedCheck_4254_;
goto v_resetjp_4248_;
}
v_resetjp_4248_:
{
lean_object* v___x_4252_; 
if (v_isShared_4250_ == 0)
{
v___x_4252_ = v___x_4249_;
goto v_reusejp_4251_;
}
else
{
lean_object* v_reuseFailAlloc_4253_; 
v_reuseFailAlloc_4253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4253_, 0, v_a_4247_);
v___x_4252_ = v_reuseFailAlloc_4253_;
goto v_reusejp_4251_;
}
v_reusejp_4251_:
{
return v___x_4252_;
}
}
}
}
else
{
lean_object* v_a_4255_; lean_object* v___x_4257_; uint8_t v_isShared_4258_; uint8_t v_isSharedCheck_4262_; 
lean_dec(v_fvar_4226_);
v_a_4255_ = lean_ctor_get(v___x_4236_, 0);
v_isSharedCheck_4262_ = !lean_is_exclusive(v___x_4236_);
if (v_isSharedCheck_4262_ == 0)
{
v___x_4257_ = v___x_4236_;
v_isShared_4258_ = v_isSharedCheck_4262_;
goto v_resetjp_4256_;
}
else
{
lean_inc(v_a_4255_);
lean_dec(v___x_4236_);
v___x_4257_ = lean_box(0);
v_isShared_4258_ = v_isSharedCheck_4262_;
goto v_resetjp_4256_;
}
v_resetjp_4256_:
{
lean_object* v___x_4260_; 
if (v_isShared_4258_ == 0)
{
v___x_4260_ = v___x_4257_;
goto v_reusejp_4259_;
}
else
{
lean_object* v_reuseFailAlloc_4261_; 
v_reuseFailAlloc_4261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4261_, 0, v_a_4255_);
v___x_4260_ = v_reuseFailAlloc_4261_;
goto v_reusejp_4259_;
}
v_reusejp_4259_:
{
return v___x_4260_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt___boxed(lean_object* v_fvar_4263_, lean_object* v_a_4264_, lean_object* v_a_4265_, lean_object* v_a_4266_, lean_object* v_a_4267_, lean_object* v_a_4268_, lean_object* v_a_4269_, lean_object* v_a_4270_, lean_object* v_a_4271_, lean_object* v_a_4272_){
_start:
{
lean_object* v_res_4273_; 
v_res_4273_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsAt(v_fvar_4263_, v_a_4264_, v_a_4265_, v_a_4266_, v_a_4267_, v_a_4268_, v_a_4269_, v_a_4270_, v_a_4271_);
lean_dec(v_a_4271_);
lean_dec_ref(v_a_4270_);
lean_dec(v_a_4269_);
lean_dec_ref(v_a_4268_);
lean_dec(v_a_4267_);
lean_dec_ref(v_a_4266_);
lean_dec(v_a_4265_);
lean_dec_ref(v_a_4264_);
return v_res_4273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___lam__0(lean_object* v_fst_4274_, lean_object* v_snd_4275_, lean_object* v_g_4276_, lean_object* v___y_4277_, lean_object* v___y_4278_, lean_object* v___y_4279_, lean_object* v___y_4280_){
_start:
{
lean_object* v___x_4282_; 
v___x_4282_ = l_Lean_MVarId_replaceTargetEq(v_g_4276_, v_fst_4274_, v_snd_4275_, v___y_4277_, v___y_4278_, v___y_4279_, v___y_4280_);
return v___x_4282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___lam__0___boxed(lean_object* v_fst_4283_, lean_object* v_snd_4284_, lean_object* v_g_4285_, lean_object* v___y_4286_, lean_object* v___y_4287_, lean_object* v___y_4288_, lean_object* v___y_4289_, lean_object* v___y_4290_){
_start:
{
lean_object* v_res_4291_; 
v_res_4291_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___lam__0(v_fst_4283_, v_snd_4284_, v_g_4285_, v___y_4286_, v___y_4287_, v___y_4288_, v___y_4289_);
lean_dec(v___y_4289_);
lean_dec_ref(v___y_4288_);
lean_dec(v___y_4287_);
lean_dec_ref(v___y_4286_);
return v_res_4291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget(lean_object* v_a_4292_, lean_object* v_a_4293_, lean_object* v_a_4294_, lean_object* v_a_4295_, lean_object* v_a_4296_, lean_object* v_a_4297_, lean_object* v_a_4298_, lean_object* v_a_4299_){
_start:
{
lean_object* v___x_4301_; 
v___x_4301_ = l_Lean_Elab_Tactic_getMainTarget(v_a_4292_, v_a_4293_, v_a_4294_, v_a_4295_, v_a_4296_, v_a_4297_, v_a_4298_, v_a_4299_);
if (lean_obj_tag(v___x_4301_) == 0)
{
lean_object* v_a_4302_; lean_object* v___x_4303_; 
v_a_4302_ = lean_ctor_get(v___x_4301_, 0);
lean_inc(v_a_4302_);
lean_dec_ref_known(v___x_4301_, 1);
v___x_4303_ = lp_mathlib_Mathlib_Tactic_CancelDenoms_cancelDenominatorsInType(v_a_4302_, v_a_4296_, v_a_4297_, v_a_4298_, v_a_4299_);
if (lean_obj_tag(v___x_4303_) == 0)
{
lean_object* v_a_4304_; lean_object* v_fst_4305_; lean_object* v_snd_4306_; lean_object* v___f_4307_; lean_object* v___x_4308_; 
v_a_4304_ = lean_ctor_get(v___x_4303_, 0);
lean_inc(v_a_4304_);
lean_dec_ref_known(v___x_4303_, 1);
v_fst_4305_ = lean_ctor_get(v_a_4304_, 0);
lean_inc(v_fst_4305_);
v_snd_4306_ = lean_ctor_get(v_a_4304_, 1);
lean_inc(v_snd_4306_);
lean_dec(v_a_4304_);
v___f_4307_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___lam__0___boxed), 8, 2);
lean_closure_set(v___f_4307_, 0, v_fst_4305_);
lean_closure_set(v___f_4307_, 1, v_snd_4306_);
v___x_4308_ = lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(v___f_4307_, v_a_4292_, v_a_4293_, v_a_4294_, v_a_4295_, v_a_4296_, v_a_4297_, v_a_4298_, v_a_4299_);
return v___x_4308_;
}
else
{
lean_object* v_a_4309_; lean_object* v___x_4311_; uint8_t v_isShared_4312_; uint8_t v_isSharedCheck_4316_; 
v_a_4309_ = lean_ctor_get(v___x_4303_, 0);
v_isSharedCheck_4316_ = !lean_is_exclusive(v___x_4303_);
if (v_isSharedCheck_4316_ == 0)
{
v___x_4311_ = v___x_4303_;
v_isShared_4312_ = v_isSharedCheck_4316_;
goto v_resetjp_4310_;
}
else
{
lean_inc(v_a_4309_);
lean_dec(v___x_4303_);
v___x_4311_ = lean_box(0);
v_isShared_4312_ = v_isSharedCheck_4316_;
goto v_resetjp_4310_;
}
v_resetjp_4310_:
{
lean_object* v___x_4314_; 
if (v_isShared_4312_ == 0)
{
v___x_4314_ = v___x_4311_;
goto v_reusejp_4313_;
}
else
{
lean_object* v_reuseFailAlloc_4315_; 
v_reuseFailAlloc_4315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4315_, 0, v_a_4309_);
v___x_4314_ = v_reuseFailAlloc_4315_;
goto v_reusejp_4313_;
}
v_reusejp_4313_:
{
return v___x_4314_;
}
}
}
}
else
{
lean_object* v_a_4317_; lean_object* v___x_4319_; uint8_t v_isShared_4320_; uint8_t v_isSharedCheck_4324_; 
v_a_4317_ = lean_ctor_get(v___x_4301_, 0);
v_isSharedCheck_4324_ = !lean_is_exclusive(v___x_4301_);
if (v_isSharedCheck_4324_ == 0)
{
v___x_4319_ = v___x_4301_;
v_isShared_4320_ = v_isSharedCheck_4324_;
goto v_resetjp_4318_;
}
else
{
lean_inc(v_a_4317_);
lean_dec(v___x_4301_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___boxed(lean_object* v_a_4325_, lean_object* v_a_4326_, lean_object* v_a_4327_, lean_object* v_a_4328_, lean_object* v_a_4329_, lean_object* v_a_4330_, lean_object* v_a_4331_, lean_object* v_a_4332_, lean_object* v_a_4333_){
_start:
{
lean_object* v_res_4334_; 
v_res_4334_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget(v_a_4325_, v_a_4326_, v_a_4327_, v_a_4328_, v_a_4329_, v_a_4330_, v_a_4331_, v_a_4332_);
lean_dec(v_a_4332_);
lean_dec_ref(v_a_4331_);
lean_dec(v_a_4330_);
lean_dec_ref(v_a_4329_);
lean_dec(v_a_4328_);
lean_dec_ref(v_a_4327_);
lean_dec(v_a_4326_);
lean_dec_ref(v_a_4325_);
return v_res_4334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___redArg(lean_object* v_msg_4335_, lean_object* v___y_4336_, lean_object* v___y_4337_, lean_object* v___y_4338_, lean_object* v___y_4339_){
_start:
{
lean_object* v_ref_4341_; lean_object* v___x_4342_; lean_object* v_a_4343_; lean_object* v___x_4345_; uint8_t v_isShared_4346_; uint8_t v_isSharedCheck_4351_; 
v_ref_4341_ = lean_ctor_get(v___y_4338_, 5);
v___x_4342_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum_spec__0_spec__0(v_msg_4335_, v___y_4336_, v___y_4337_, v___y_4338_, v___y_4339_);
v_a_4343_ = lean_ctor_get(v___x_4342_, 0);
v_isSharedCheck_4351_ = !lean_is_exclusive(v___x_4342_);
if (v_isSharedCheck_4351_ == 0)
{
v___x_4345_ = v___x_4342_;
v_isShared_4346_ = v_isSharedCheck_4351_;
goto v_resetjp_4344_;
}
else
{
lean_inc(v_a_4343_);
lean_dec(v___x_4342_);
v___x_4345_ = lean_box(0);
v_isShared_4346_ = v_isSharedCheck_4351_;
goto v_resetjp_4344_;
}
v_resetjp_4344_:
{
lean_object* v___x_4347_; lean_object* v___x_4349_; 
lean_inc(v_ref_4341_);
v___x_4347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4347_, 0, v_ref_4341_);
lean_ctor_set(v___x_4347_, 1, v_a_4343_);
if (v_isShared_4346_ == 0)
{
lean_ctor_set_tag(v___x_4345_, 1);
lean_ctor_set(v___x_4345_, 0, v___x_4347_);
v___x_4349_ = v___x_4345_;
goto v_reusejp_4348_;
}
else
{
lean_object* v_reuseFailAlloc_4350_; 
v_reuseFailAlloc_4350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4350_, 0, v___x_4347_);
v___x_4349_ = v_reuseFailAlloc_4350_;
goto v_reusejp_4348_;
}
v_reusejp_4348_:
{
return v___x_4349_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___redArg___boxed(lean_object* v_msg_4352_, lean_object* v___y_4353_, lean_object* v___y_4354_, lean_object* v___y_4355_, lean_object* v___y_4356_, lean_object* v___y_4357_){
_start:
{
lean_object* v_res_4358_; 
v_res_4358_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___redArg(v_msg_4352_, v___y_4353_, v___y_4354_, v___y_4355_, v___y_4356_);
lean_dec(v___y_4356_);
lean_dec_ref(v___y_4355_);
lean_dec(v___y_4354_);
lean_dec_ref(v___y_4353_);
return v_res_4358_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__1(void){
_start:
{
lean_object* v___x_4360_; lean_object* v___x_4361_; 
v___x_4360_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__0));
v___x_4361_ = l_Lean_stringToMessageData(v___x_4360_);
return v___x_4361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0(lean_object* v_x_4362_, lean_object* v___y_4363_, lean_object* v___y_4364_, lean_object* v___y_4365_, lean_object* v___y_4366_, lean_object* v___y_4367_, lean_object* v___y_4368_, lean_object* v___y_4369_, lean_object* v___y_4370_){
_start:
{
lean_object* v___x_4372_; lean_object* v___x_4373_; 
v___x_4372_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___closed__1);
v___x_4373_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___redArg(v___x_4372_, v___y_4367_, v___y_4368_, v___y_4369_, v___y_4370_);
return v___x_4373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0___boxed(lean_object* v_x_4374_, lean_object* v___y_4375_, lean_object* v___y_4376_, lean_object* v___y_4377_, lean_object* v___y_4378_, lean_object* v___y_4379_, lean_object* v___y_4380_, lean_object* v___y_4381_, lean_object* v___y_4382_, lean_object* v___y_4383_){
_start:
{
lean_object* v_res_4384_; 
v_res_4384_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___lam__0(v_x_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_, v___y_4379_, v___y_4380_, v___y_4381_, v___y_4382_);
lean_dec(v___y_4382_);
lean_dec_ref(v___y_4381_);
lean_dec(v___y_4380_);
lean_dec_ref(v___y_4379_);
lean_dec(v___y_4378_);
lean_dec_ref(v___y_4377_);
lean_dec(v___y_4376_);
lean_dec_ref(v___y_4375_);
lean_dec(v_x_4374_);
return v_res_4384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators(lean_object* v_loc_4387_, lean_object* v_a_4388_, lean_object* v_a_4389_, lean_object* v_a_4390_, lean_object* v_a_4391_, lean_object* v_a_4392_, lean_object* v_a_4393_, lean_object* v_a_4394_, lean_object* v_a_4395_){
_start:
{
lean_object* v___f_4397_; lean_object* v___x_4398_; lean_object* v___x_4399_; lean_object* v___x_4400_; 
v___f_4397_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__0));
v___x_4398_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___closed__1));
v___x_4399_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominatorsTarget___boxed), 9, 0);
v___x_4400_ = l_Lean_Elab_Tactic_withLocation(v_loc_4387_, v___x_4398_, v___x_4399_, v___f_4397_, v_a_4388_, v_a_4389_, v_a_4390_, v_a_4391_, v_a_4392_, v_a_4393_, v_a_4394_, v_a_4395_);
return v___x_4400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators___boxed(lean_object* v_loc_4401_, lean_object* v_a_4402_, lean_object* v_a_4403_, lean_object* v_a_4404_, lean_object* v_a_4405_, lean_object* v_a_4406_, lean_object* v_a_4407_, lean_object* v_a_4408_, lean_object* v_a_4409_, lean_object* v_a_4410_){
_start:
{
lean_object* v_res_4411_; 
v_res_4411_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators(v_loc_4401_, v_a_4402_, v_a_4403_, v_a_4404_, v_a_4405_, v_a_4406_, v_a_4407_, v_a_4408_, v_a_4409_);
lean_dec(v_a_4409_);
lean_dec_ref(v_a_4408_);
lean_dec(v_a_4407_);
lean_dec_ref(v_a_4406_);
lean_dec(v_a_4405_);
lean_dec_ref(v_a_4404_);
lean_dec(v_a_4403_);
lean_dec_ref(v_a_4402_);
lean_dec(v_loc_4401_);
return v_res_4411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0(lean_object* v_00_u03b1_4412_, lean_object* v_msg_4413_, lean_object* v___y_4414_, lean_object* v___y_4415_, lean_object* v___y_4416_, lean_object* v___y_4417_, lean_object* v___y_4418_, lean_object* v___y_4419_, lean_object* v___y_4420_, lean_object* v___y_4421_){
_start:
{
lean_object* v___x_4423_; 
v___x_4423_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___redArg(v_msg_4413_, v___y_4418_, v___y_4419_, v___y_4420_, v___y_4421_);
return v___x_4423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0___boxed(lean_object* v_00_u03b1_4424_, lean_object* v_msg_4425_, lean_object* v___y_4426_, lean_object* v___y_4427_, lean_object* v___y_4428_, lean_object* v___y_4429_, lean_object* v___y_4430_, lean_object* v___y_4431_, lean_object* v___y_4432_, lean_object* v___y_4433_, lean_object* v___y_4434_){
_start:
{
lean_object* v_res_4435_; 
v_res_4435_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators_spec__0(v_00_u03b1_4424_, v_msg_4425_, v___y_4426_, v___y_4427_, v___y_4428_, v___y_4429_, v___y_4430_, v___y_4431_, v___y_4432_, v___y_4433_);
lean_dec(v___y_4433_);
lean_dec_ref(v___y_4432_);
lean_dec(v___y_4431_);
lean_dec_ref(v___y_4430_);
lean_dec(v___y_4429_);
lean_dec_ref(v___y_4428_);
lean_dec(v___y_4427_);
lean_dec_ref(v___y_4426_);
return v_res_4435_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__2(void){
_start:
{
lean_object* v___x_4441_; lean_object* v___x_4442_; lean_object* v___x_4443_; lean_object* v___x_4444_; 
v___x_4441_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9, &lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_cancelDenoms___closed__9);
v___x_4442_ = lean_unsigned_to_nat(1022u);
v___x_4443_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1));
v___x_4444_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_4444_, 0, v___x_4443_);
lean_ctor_set(v___x_4444_, 1, v___x_4442_);
lean_ctor_set(v___x_4444_, 2, v___x_4441_);
return v___x_4444_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticCancel__denoms__(void){
_start:
{
lean_object* v___x_4445_; 
v___x_4445_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__2, &lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__2_once, _init_lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__2);
return v___x_4445_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_4446_; lean_object* v___x_4447_; lean_object* v___x_4448_; 
v___x_4446_ = lean_box(0);
v___x_4447_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_4448_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4448_, 0, v___x_4447_);
lean_ctor_set(v___x_4448_, 1, v___x_4446_);
return v___x_4448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg(){
_start:
{
lean_object* v___x_4450_; lean_object* v___x_4451_; 
v___x_4450_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___closed__0);
v___x_4451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4451_, 0, v___x_4450_);
return v___x_4451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg___boxed(lean_object* v___y_4452_){
_start:
{
lean_object* v_res_4453_; 
v_res_4453_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg();
return v_res_4453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0(lean_object* v_00_u03b1_4454_, lean_object* v___y_4455_, lean_object* v___y_4456_, lean_object* v___y_4457_, lean_object* v___y_4458_, lean_object* v___y_4459_, lean_object* v___y_4460_, lean_object* v___y_4461_, lean_object* v___y_4462_){
_start:
{
lean_object* v___x_4464_; 
v___x_4464_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg();
return v___x_4464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___boxed(lean_object* v_00_u03b1_4465_, lean_object* v___y_4466_, lean_object* v___y_4467_, lean_object* v___y_4468_, lean_object* v___y_4469_, lean_object* v___y_4470_, lean_object* v___y_4471_, lean_object* v___y_4472_, lean_object* v___y_4473_, lean_object* v___y_4474_){
_start:
{
lean_object* v_res_4475_; 
v_res_4475_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0(v_00_u03b1_4465_, v___y_4466_, v___y_4467_, v___y_4468_, v___y_4469_, v___y_4470_, v___y_4471_, v___y_4472_, v___y_4473_);
lean_dec(v___y_4473_);
lean_dec_ref(v___y_4472_);
lean_dec(v___y_4471_);
lean_dec_ref(v___y_4470_);
lean_dec(v___y_4469_);
lean_dec_ref(v___y_4468_);
lean_dec(v___y_4467_);
lean_dec_ref(v___y_4466_);
return v_res_4475_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__14(void){
_start:
{
lean_object* v___x_4510_; lean_object* v___x_4511_; 
v___x_4510_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__13));
v___x_4511_ = l_String_toRawSubstring_x27(v___x_4510_);
return v___x_4511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1(lean_object* v_x_4523_, lean_object* v_a_4524_, lean_object* v_a_4525_, lean_object* v_a_4526_, lean_object* v_a_4527_, lean_object* v_a_4528_, lean_object* v_a_4529_, lean_object* v_a_4530_, lean_object* v_a_4531_){
_start:
{
lean_object* v___y_4534_; lean_object* v___y_4535_; lean_object* v___y_4536_; lean_object* v___y_4537_; lean_object* v___y_4538_; lean_object* v___y_4539_; lean_object* v___y_4540_; lean_object* v___y_4541_; lean_object* v___y_4542_; lean_object* v___y_4543_; lean_object* v___y_4544_; lean_object* v___y_4545_; lean_object* v___y_4546_; lean_object* v___y_4556_; lean_object* v___y_4557_; lean_object* v___x_4600_; uint8_t v___x_4601_; 
v___x_4600_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticCancel__denoms___00__closed__1));
lean_inc(v_x_4523_);
v___x_4601_ = l_Lean_Syntax_isOfKind(v_x_4523_, v___x_4600_);
if (v___x_4601_ == 0)
{
lean_object* v___x_4602_; 
lean_dec(v_x_4523_);
v___x_4602_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1_spec__0___redArg();
return v___x_4602_;
}
else
{
lean_object* v___x_4603_; lean_object* v___x_4604_; lean_object* v___x_4605_; 
v___x_4603_ = lean_unsigned_to_nat(1u);
v___x_4604_ = l_Lean_Syntax_getArg(v_x_4523_, v___x_4603_);
lean_dec(v_x_4523_);
v___x_4605_ = l_Lean_Syntax_getOptional_x3f(v___x_4604_);
lean_dec(v___x_4604_);
if (lean_obj_tag(v___x_4605_) == 0)
{
lean_object* v___x_4606_; 
v___x_4606_ = lean_box(0);
v___y_4556_ = v___x_4606_;
v___y_4557_ = v___x_4606_;
goto v___jp_4555_;
}
else
{
lean_object* v_val_4607_; lean_object* v___x_4609_; uint8_t v_isShared_4610_; uint8_t v_isSharedCheck_4614_; 
v_val_4607_ = lean_ctor_get(v___x_4605_, 0);
v_isSharedCheck_4614_ = !lean_is_exclusive(v___x_4605_);
if (v_isSharedCheck_4614_ == 0)
{
v___x_4609_ = v___x_4605_;
v_isShared_4610_ = v_isSharedCheck_4614_;
goto v_resetjp_4608_;
}
else
{
lean_inc(v_val_4607_);
lean_dec(v___x_4605_);
v___x_4609_ = lean_box(0);
v_isShared_4610_ = v_isSharedCheck_4614_;
goto v_resetjp_4608_;
}
v_resetjp_4608_:
{
lean_object* v___x_4612_; 
if (v_isShared_4610_ == 0)
{
v___x_4612_ = v___x_4609_;
goto v_reusejp_4611_;
}
else
{
lean_object* v_reuseFailAlloc_4613_; 
v_reuseFailAlloc_4613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4613_, 0, v_val_4607_);
v___x_4612_ = v_reuseFailAlloc_4613_;
goto v_reusejp_4611_;
}
v_reusejp_4611_:
{
lean_inc_ref(v___x_4612_);
v___y_4556_ = v___x_4612_;
v___y_4557_ = v___x_4612_;
goto v___jp_4555_;
}
}
}
}
v___jp_4533_:
{
lean_object* v___x_4547_; lean_object* v___x_4548_; lean_object* v___x_4549_; lean_object* v___x_4550_; lean_object* v___x_4551_; lean_object* v___x_4552_; lean_object* v___x_4553_; lean_object* v___x_4554_; 
lean_inc_ref(v___y_4539_);
v___x_4547_ = l_Array_append___redArg(v___y_4539_, v___y_4546_);
lean_dec_ref(v___y_4546_);
lean_inc(v___y_4536_);
lean_inc_n(v___y_4542_, 5);
v___x_4548_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4548_, 0, v___y_4542_);
lean_ctor_set(v___x_4548_, 1, v___y_4536_);
lean_ctor_set(v___x_4548_, 2, v___x_4547_);
lean_inc(v___y_4534_);
v___x_4549_ = l_Lean_Syntax_node5(v___y_4542_, v___y_4534_, v___y_4545_, v___y_4544_, v___y_4535_, v___y_4540_, v___x_4548_);
v___x_4550_ = l_Lean_Syntax_node1(v___y_4542_, v___y_4536_, v___x_4549_);
lean_inc(v___y_4537_);
v___x_4551_ = l_Lean_Syntax_node1(v___y_4542_, v___y_4537_, v___x_4550_);
lean_inc(v___y_4543_);
v___x_4552_ = l_Lean_Syntax_node1(v___y_4542_, v___y_4543_, v___x_4551_);
lean_inc(v___y_4538_);
v___x_4553_ = l_Lean_Syntax_node2(v___y_4542_, v___y_4538_, v___y_4541_, v___x_4552_);
v___x_4554_ = l_Lean_Elab_Tactic_evalTactic(v___x_4553_, v_a_4524_, v_a_4525_, v_a_4526_, v_a_4527_, v_a_4528_, v_a_4529_, v_a_4530_, v_a_4531_);
return v___x_4554_;
}
v___jp_4555_:
{
lean_object* v___x_4558_; lean_object* v___x_4559_; lean_object* v___x_4560_; 
v___x_4558_ = l_Lean_mkOptionalNode(v___y_4557_);
v___x_4559_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_4558_);
lean_dec(v___x_4558_);
v___x_4560_ = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_cancelDenominators(v___x_4559_, v_a_4524_, v_a_4525_, v_a_4526_, v_a_4527_, v_a_4528_, v_a_4529_, v_a_4530_, v_a_4531_);
lean_dec(v___x_4559_);
if (lean_obj_tag(v___x_4560_) == 0)
{
lean_object* v_ref_4561_; lean_object* v_quotContext_4562_; lean_object* v_currMacroScope_4563_; uint8_t v___x_4564_; lean_object* v___x_4565_; lean_object* v___x_4566_; lean_object* v___x_4567_; lean_object* v___x_4568_; lean_object* v___x_4569_; lean_object* v___x_4570_; lean_object* v___x_4571_; lean_object* v___x_4572_; lean_object* v___x_4573_; lean_object* v___x_4574_; lean_object* v___x_4575_; lean_object* v___x_4576_; lean_object* v___x_4577_; lean_object* v___x_4578_; lean_object* v___x_4579_; lean_object* v___x_4580_; lean_object* v___x_4581_; lean_object* v___x_4582_; lean_object* v___x_4583_; lean_object* v___x_4584_; lean_object* v___x_4585_; lean_object* v___x_4586_; lean_object* v___x_4587_; lean_object* v___x_4588_; lean_object* v___x_4589_; lean_object* v___x_4590_; lean_object* v___x_4591_; lean_object* v___x_4592_; lean_object* v___x_4593_; lean_object* v___x_4594_; lean_object* v___x_4595_; lean_object* v___x_4596_; 
lean_dec_ref_known(v___x_4560_, 1);
v_ref_4561_ = lean_ctor_get(v_a_4530_, 5);
v_quotContext_4562_ = lean_ctor_get(v_a_4530_, 10);
v_currMacroScope_4563_ = lean_ctor_get(v_a_4530_, 11);
v___x_4564_ = 0;
v___x_4565_ = l_Lean_SourceInfo_fromRef(v_ref_4561_, v___x_4564_);
v___x_4566_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__1));
v___x_4567_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__2));
lean_inc_n(v___x_4565_, 13);
v___x_4568_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4568_, 0, v___x_4565_);
lean_ctor_set(v___x_4568_, 1, v___x_4567_);
v___x_4569_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__4));
v___x_4570_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__6));
v___x_4571_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__8));
v___x_4572_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__1));
v___x_4573_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__2));
v___x_4574_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4574_, 0, v___x_4565_);
lean_ctor_set(v___x_4574_, 1, v___x_4573_);
v___x_4575_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__6));
v___x_4576_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9, &lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__Mathlib_Tactic_CancelDenoms_synthesizeUsingNormNum___closed__9);
v___x_4577_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4577_, 0, v___x_4565_);
lean_ctor_set(v___x_4577_, 1, v___x_4571_);
lean_ctor_set(v___x_4577_, 2, v___x_4576_);
lean_inc_ref_n(v___x_4577_, 2);
v___x_4578_ = l_Lean_Syntax_node1(v___x_4565_, v___x_4575_, v___x_4577_);
v___x_4579_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__8));
v___x_4580_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__9));
v___x_4581_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4581_, 0, v___x_4565_);
lean_ctor_set(v___x_4581_, 1, v___x_4580_);
v___x_4582_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__11));
v___x_4583_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__12));
v___x_4584_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4584_, 0, v___x_4565_);
lean_ctor_set(v___x_4584_, 1, v___x_4583_);
v___x_4585_ = l_Lean_Syntax_node1(v___x_4565_, v___x_4571_, v___x_4584_);
v___x_4586_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__14, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__14);
v___x_4587_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__15));
lean_inc(v_currMacroScope_4563_);
lean_inc(v_quotContext_4562_);
v___x_4588_ = l_Lean_addMacroScope(v_quotContext_4562_, v___x_4587_, v_currMacroScope_4563_);
v___x_4589_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__17));
v___x_4590_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4590_, 0, v___x_4565_);
lean_ctor_set(v___x_4590_, 1, v___x_4586_);
lean_ctor_set(v___x_4590_, 2, v___x_4588_);
lean_ctor_set(v___x_4590_, 3, v___x_4589_);
v___x_4591_ = l_Lean_Syntax_node3(v___x_4565_, v___x_4582_, v___x_4577_, v___x_4585_, v___x_4590_);
v___x_4592_ = l_Lean_Syntax_node1(v___x_4565_, v___x_4571_, v___x_4591_);
v___x_4593_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__18));
v___x_4594_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4594_, 0, v___x_4565_);
lean_ctor_set(v___x_4594_, 1, v___x_4593_);
v___x_4595_ = l_Lean_Syntax_node3(v___x_4565_, v___x_4579_, v___x_4581_, v___x_4592_, v___x_4594_);
v___x_4596_ = l_Lean_Syntax_node1(v___x_4565_, v___x_4571_, v___x_4595_);
if (lean_obj_tag(v___y_4556_) == 1)
{
lean_object* v_val_4597_; lean_object* v___x_4598_; 
v_val_4597_ = lean_ctor_get(v___y_4556_, 0);
lean_inc(v_val_4597_);
lean_dec_ref_known(v___y_4556_, 1);
v___x_4598_ = l_Array_mkArray1___redArg(v_val_4597_);
v___y_4534_ = v___x_4572_;
v___y_4535_ = v___x_4577_;
v___y_4536_ = v___x_4571_;
v___y_4537_ = v___x_4570_;
v___y_4538_ = v___x_4566_;
v___y_4539_ = v___x_4576_;
v___y_4540_ = v___x_4596_;
v___y_4541_ = v___x_4568_;
v___y_4542_ = v___x_4565_;
v___y_4543_ = v___x_4569_;
v___y_4544_ = v___x_4578_;
v___y_4545_ = v___x_4574_;
v___y_4546_ = v___x_4598_;
goto v___jp_4533_;
}
else
{
lean_object* v___x_4599_; 
lean_dec(v___y_4556_);
v___x_4599_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___closed__19));
v___y_4534_ = v___x_4572_;
v___y_4535_ = v___x_4577_;
v___y_4536_ = v___x_4571_;
v___y_4537_ = v___x_4570_;
v___y_4538_ = v___x_4566_;
v___y_4539_ = v___x_4576_;
v___y_4540_ = v___x_4596_;
v___y_4541_ = v___x_4568_;
v___y_4542_ = v___x_4565_;
v___y_4543_ = v___x_4569_;
v___y_4544_ = v___x_4578_;
v___y_4545_ = v___x_4574_;
v___y_4546_ = v___x_4599_;
goto v___jp_4533_;
}
}
else
{
lean_dec(v___y_4556_);
return v___x_4560_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1___boxed(lean_object* v_x_4615_, lean_object* v_a_4616_, lean_object* v_a_4617_, lean_object* v_a_4618_, lean_object* v_a_4619_, lean_object* v_a_4620_, lean_object* v_a_4621_, lean_object* v_a_4622_, lean_object* v_a_4623_, lean_object* v_a_4624_){
_start:
{
lean_object* v_res_4625_; 
v_res_4625_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CancelDenoms__Core______elabRules__Mathlib__Tactic__tacticCancel__denoms____1(v_x_4615_, v_a_4616_, v_a_4617_, v_a_4618_, v_a_4619_, v_a_4620_, v_a_4621_, v_a_4622_, v_a_4623_);
lean_dec(v_a_4623_);
lean_dec_ref(v_a_4622_);
lean_dec(v_a_4621_);
lean_dec_ref(v_a_4620_);
lean_dec(v_a_4619_);
lean_dec_ref(v_a_4618_);
lean_dec(v_a_4617_);
lean_dec_ref(v_a_4616_);
return v_res_4625_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_NormNum_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_CancelDenoms_Core_0__initFn_00___x40_Mathlib_Tactic_CancelDenoms_Core_1602764063____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_cancelDenoms = _init_lp_mathlib_Mathlib_Tactic_cancelDenoms();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_cancelDenoms);
lp_mathlib_Mathlib_Tactic_tacticCancel__denoms__ = _init_lp_mathlib_Mathlib_Tactic_tacticCancel__denoms__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticCancel__denoms__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_NormNum_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_SynthesizeUsing(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_NormNum_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_SynthesizeUsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_CancelDenoms_Core(builtin);
}
#ifdef __cplusplus
}
#endif
