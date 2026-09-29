// Lean compiler output
// Module: Mathlib.Tactic.Linarith.Parsing
// Imports: public import Init public meta import Init public meta import Mathlib.Algebra.GroupWithZero.Nat public meta import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Tactic.Linarith.Datatypes
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
lean_object* l_Nat_decLt___boxed(lean_object*, lean_object*);
uint8_t l_List_decidableLex___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Std_DTreeMap_Internal_Impl_Const_beq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_alter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_filter___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_link___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_link2___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_balance___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_int_add(lean_object*, lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_numeral_x3f(lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_paren(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00List_findDefeq_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00List_findDefeq_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_findDefeq___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_List_findDefeq___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_findDefeq___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_List_findDefeq___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_findDefeq___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Monom_one;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_decLt___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_one;
LEAN_EXPORT uint8_t lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_scaleByMonom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_pow(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_pow___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SumOfMonom(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_one;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_scalar(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_var(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_Linarith_elimMonom_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_elimMonom_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_elimMonom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_Linarith_elimMonom_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_toComp_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_toComp_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toComp(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toComp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toCompFold(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toCompFold___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__0 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__0_value)}};
static const lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__1 = (const lean_object*)&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__1_value;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2;
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "linarith"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "detail"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 239, 24, 66, 70, 17, 119, 33)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__1_value),LEAN_SCALAR_PTR_LITERAL(29, 12, 183, 160, 66, 250, 13, 227)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "monomial map: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___redArg(uint8_t v_red_1_, lean_object* v_e_2_, lean_object* v_x_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
if (lean_obj_tag(v_x_3_) == 0)
{
lean_object* v___x_9_; lean_object* v___x_10_; 
lean_dec_ref(v_e_2_);
v___x_9_ = lean_box(0);
v___x_10_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_10_, 0, v___x_9_);
return v___x_10_;
}
else
{
lean_object* v_head_11_; lean_object* v_tail_12_; lean_object* v_fst_13_; lean_object* v_keyedConfig_14_; uint8_t v_trackZetaDelta_15_; lean_object* v_zetaDeltaSet_16_; lean_object* v_lctx_17_; lean_object* v_localInstances_18_; lean_object* v_defEqCtx_x3f_19_; lean_object* v_synthPendingDepth_20_; lean_object* v_customCanUnfoldPredicate_x3f_21_; uint8_t v_univApprox_22_; uint8_t v_inTypeClassResolution_23_; uint8_t v_cacheInferType_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v_head_11_ = lean_ctor_get(v_x_3_, 0);
lean_inc(v_head_11_);
v_tail_12_ = lean_ctor_get(v_x_3_, 1);
lean_inc(v_tail_12_);
lean_dec_ref_known(v_x_3_, 2);
v_fst_13_ = lean_ctor_get(v_head_11_, 0);
v_keyedConfig_14_ = lean_ctor_get(v___y_4_, 0);
v_trackZetaDelta_15_ = lean_ctor_get_uint8(v___y_4_, sizeof(void*)*7);
v_zetaDeltaSet_16_ = lean_ctor_get(v___y_4_, 1);
v_lctx_17_ = lean_ctor_get(v___y_4_, 2);
v_localInstances_18_ = lean_ctor_get(v___y_4_, 3);
v_defEqCtx_x3f_19_ = lean_ctor_get(v___y_4_, 4);
v_synthPendingDepth_20_ = lean_ctor_get(v___y_4_, 5);
v_customCanUnfoldPredicate_x3f_21_ = lean_ctor_get(v___y_4_, 6);
v_univApprox_22_ = lean_ctor_get_uint8(v___y_4_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_23_ = lean_ctor_get_uint8(v___y_4_, sizeof(void*)*7 + 2);
v_cacheInferType_24_ = lean_ctor_get_uint8(v___y_4_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_14_);
v___x_25_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_red_1_, v_keyedConfig_14_);
lean_inc(v_customCanUnfoldPredicate_x3f_21_);
lean_inc(v_synthPendingDepth_20_);
lean_inc(v_defEqCtx_x3f_19_);
lean_inc_ref(v_localInstances_18_);
lean_inc_ref(v_lctx_17_);
lean_inc(v_zetaDeltaSet_16_);
v___x_26_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_26_, 0, v___x_25_);
lean_ctor_set(v___x_26_, 1, v_zetaDeltaSet_16_);
lean_ctor_set(v___x_26_, 2, v_lctx_17_);
lean_ctor_set(v___x_26_, 3, v_localInstances_18_);
lean_ctor_set(v___x_26_, 4, v_defEqCtx_x3f_19_);
lean_ctor_set(v___x_26_, 5, v_synthPendingDepth_20_);
lean_ctor_set(v___x_26_, 6, v_customCanUnfoldPredicate_x3f_21_);
lean_ctor_set_uint8(v___x_26_, sizeof(void*)*7, v_trackZetaDelta_15_);
lean_ctor_set_uint8(v___x_26_, sizeof(void*)*7 + 1, v_univApprox_22_);
lean_ctor_set_uint8(v___x_26_, sizeof(void*)*7 + 2, v_inTypeClassResolution_23_);
lean_ctor_set_uint8(v___x_26_, sizeof(void*)*7 + 3, v_cacheInferType_24_);
lean_inc(v_fst_13_);
lean_inc_ref(v_e_2_);
v___x_27_ = l_Lean_Meta_isExprDefEq(v_e_2_, v_fst_13_, v___x_26_, v___y_5_, v___y_6_, v___y_7_);
lean_dec_ref_known(v___x_26_, 7);
if (lean_obj_tag(v___x_27_) == 0)
{
lean_object* v_a_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_38_; 
v_a_28_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_38_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_38_ == 0)
{
v___x_30_ = v___x_27_;
v_isShared_31_ = v_isSharedCheck_38_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_a_28_);
lean_dec(v___x_27_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_38_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
uint8_t v___x_32_; 
v___x_32_ = lean_unbox(v_a_28_);
lean_dec(v_a_28_);
if (v___x_32_ == 0)
{
lean_del_object(v___x_30_);
lean_dec(v_head_11_);
v_x_3_ = v_tail_12_;
goto _start;
}
else
{
lean_object* v___x_34_; lean_object* v___x_36_; 
lean_dec(v_tail_12_);
lean_dec_ref(v_e_2_);
v___x_34_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_34_, 0, v_head_11_);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 0, v___x_34_);
v___x_36_ = v___x_30_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_37_; 
v_reuseFailAlloc_37_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_37_, 0, v___x_34_);
v___x_36_ = v_reuseFailAlloc_37_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
return v___x_36_;
}
}
}
}
else
{
lean_object* v_a_39_; lean_object* v___x_41_; uint8_t v_isShared_42_; uint8_t v_isSharedCheck_46_; 
lean_dec(v_tail_12_);
lean_dec(v_head_11_);
lean_dec_ref(v_e_2_);
v_a_39_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_46_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_46_ == 0)
{
v___x_41_ = v___x_27_;
v_isShared_42_ = v_isSharedCheck_46_;
goto v_resetjp_40_;
}
else
{
lean_inc(v_a_39_);
lean_dec(v___x_27_);
v___x_41_ = lean_box(0);
v_isShared_42_ = v_isSharedCheck_46_;
goto v_resetjp_40_;
}
v_resetjp_40_:
{
lean_object* v___x_44_; 
if (v_isShared_42_ == 0)
{
v___x_44_ = v___x_41_;
goto v_reusejp_43_;
}
else
{
lean_object* v_reuseFailAlloc_45_; 
v_reuseFailAlloc_45_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_45_, 0, v_a_39_);
v___x_44_ = v_reuseFailAlloc_45_;
goto v_reusejp_43_;
}
v_reusejp_43_:
{
return v___x_44_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___redArg___boxed(lean_object* v_red_47_, lean_object* v_e_48_, lean_object* v_x_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_){
_start:
{
uint8_t v_red_boxed_55_; lean_object* v_res_56_; 
v_red_boxed_55_ = lean_unbox(v_red_47_);
v_res_56_ = lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___redArg(v_red_boxed_55_, v_e_48_, v_x_49_, v___y_50_, v___y_51_, v___y_52_, v___y_53_);
lean_dec(v___y_53_);
lean_dec_ref(v___y_52_);
lean_dec(v___y_51_);
lean_dec_ref(v___y_50_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00List_findDefeq_spec__1_spec__1(lean_object* v_msgData_57_, lean_object* v___y_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_){
_start:
{
lean_object* v___x_63_; lean_object* v_env_64_; lean_object* v___x_65_; lean_object* v_mctx_66_; lean_object* v_lctx_67_; lean_object* v_options_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_63_ = lean_st_ref_get(v___y_61_);
v_env_64_ = lean_ctor_get(v___x_63_, 0);
lean_inc_ref(v_env_64_);
lean_dec(v___x_63_);
v___x_65_ = lean_st_ref_get(v___y_59_);
v_mctx_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc_ref(v_mctx_66_);
lean_dec(v___x_65_);
v_lctx_67_ = lean_ctor_get(v___y_58_, 2);
v_options_68_ = lean_ctor_get(v___y_60_, 2);
lean_inc_ref(v_options_68_);
lean_inc_ref(v_lctx_67_);
v___x_69_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_69_, 0, v_env_64_);
lean_ctor_set(v___x_69_, 1, v_mctx_66_);
lean_ctor_set(v___x_69_, 2, v_lctx_67_);
lean_ctor_set(v___x_69_, 3, v_options_68_);
v___x_70_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v_msgData_57_);
v___x_71_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00List_findDefeq_spec__1_spec__1___boxed(lean_object* v_msgData_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00List_findDefeq_spec__1_spec__1(v_msgData_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
lean_dec(v___y_74_);
lean_dec_ref(v___y_73_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___redArg(lean_object* v_msg_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_ref_85_; lean_object* v___x_86_; lean_object* v_a_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_95_; 
v_ref_85_ = lean_ctor_get(v___y_82_, 5);
v___x_86_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00List_findDefeq_spec__1_spec__1(v_msg_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
v_a_87_ = lean_ctor_get(v___x_86_, 0);
v_isSharedCheck_95_ = !lean_is_exclusive(v___x_86_);
if (v_isSharedCheck_95_ == 0)
{
v___x_89_ = v___x_86_;
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_a_87_);
lean_dec(v___x_86_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_91_; lean_object* v___x_93_; 
lean_inc(v_ref_85_);
v___x_91_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_91_, 0, v_ref_85_);
lean_ctor_set(v___x_91_, 1, v_a_87_);
if (v_isShared_90_ == 0)
{
lean_ctor_set_tag(v___x_89_, 1);
lean_ctor_set(v___x_89_, 0, v___x_91_);
v___x_93_ = v___x_89_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v___x_91_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___redArg___boxed(lean_object* v_msg_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___redArg(v_msg_96_, v___y_97_, v___y_98_, v___y_99_, v___y_100_);
lean_dec(v___y_100_);
lean_dec_ref(v___y_99_);
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
return v_res_102_;
}
}
static lean_object* _init_lp_mathlib_List_findDefeq___redArg___closed__1(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib_List_findDefeq___redArg___closed__0));
v___x_105_ = l_Lean_stringToMessageData(v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq___redArg(uint8_t v_red_106_, lean_object* v_m_107_, lean_object* v_e_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___redArg(v_red_106_, v_e_108_, v_m_107_, v_a_109_, v_a_110_, v_a_111_, v_a_112_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_126_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_126_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_126_ == 0)
{
v___x_117_ = v___x_114_;
v_isShared_118_ = v_isSharedCheck_126_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_a_115_);
lean_dec(v___x_114_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_126_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
if (lean_obj_tag(v_a_115_) == 1)
{
lean_object* v_val_119_; lean_object* v_snd_120_; lean_object* v___x_122_; 
v_val_119_ = lean_ctor_get(v_a_115_, 0);
lean_inc(v_val_119_);
lean_dec_ref_known(v_a_115_, 1);
v_snd_120_ = lean_ctor_get(v_val_119_, 1);
lean_inc(v_snd_120_);
lean_dec(v_val_119_);
if (v_isShared_118_ == 0)
{
lean_ctor_set(v___x_117_, 0, v_snd_120_);
v___x_122_ = v___x_117_;
goto v_reusejp_121_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v_snd_120_);
v___x_122_ = v_reuseFailAlloc_123_;
goto v_reusejp_121_;
}
v_reusejp_121_:
{
return v___x_122_;
}
}
else
{
lean_object* v___x_124_; lean_object* v___x_125_; 
lean_del_object(v___x_117_);
lean_dec(v_a_115_);
v___x_124_ = lean_obj_once(&lp_mathlib_List_findDefeq___redArg___closed__1, &lp_mathlib_List_findDefeq___redArg___closed__1_once, _init_lp_mathlib_List_findDefeq___redArg___closed__1);
v___x_125_ = lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___redArg(v___x_124_, v_a_109_, v_a_110_, v_a_111_, v_a_112_);
return v___x_125_;
}
}
}
else
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_134_; 
v_a_127_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_134_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_134_ == 0)
{
v___x_129_ = v___x_114_;
v_isShared_130_ = v_isSharedCheck_134_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_114_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_134_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v___x_132_; 
if (v_isShared_130_ == 0)
{
v___x_132_ = v___x_129_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v_a_127_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq___redArg___boxed(lean_object* v_red_135_, lean_object* v_m_136_, lean_object* v_e_137_, lean_object* v_a_138_, lean_object* v_a_139_, lean_object* v_a_140_, lean_object* v_a_141_, lean_object* v_a_142_){
_start:
{
uint8_t v_red_boxed_143_; lean_object* v_res_144_; 
v_red_boxed_143_ = lean_unbox(v_red_135_);
v_res_144_ = lp_mathlib_List_findDefeq___redArg(v_red_boxed_143_, v_m_136_, v_e_137_, v_a_138_, v_a_139_, v_a_140_, v_a_141_);
lean_dec(v_a_141_);
lean_dec_ref(v_a_140_);
lean_dec(v_a_139_);
lean_dec_ref(v_a_138_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq(lean_object* v_v_145_, uint8_t v_red_146_, lean_object* v_m_147_, lean_object* v_e_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_List_findDefeq___redArg(v_red_146_, v_m_147_, v_e_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findDefeq___boxed(lean_object* v_v_155_, lean_object* v_red_156_, lean_object* v_m_157_, lean_object* v_e_158_, lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_, lean_object* v_a_163_){
_start:
{
uint8_t v_red_boxed_164_; lean_object* v_res_165_; 
v_red_boxed_164_ = lean_unbox(v_red_156_);
v_res_165_ = lp_mathlib_List_findDefeq(v_v_155_, v_red_boxed_164_, v_m_157_, v_e_158_, v_a_159_, v_a_160_, v_a_161_, v_a_162_);
lean_dec(v_a_162_);
lean_dec_ref(v_a_161_);
lean_dec(v_a_160_);
lean_dec_ref(v_a_159_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0(lean_object* v_v_166_, uint8_t v_red_167_, lean_object* v_e_168_, lean_object* v_x_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___redArg(v_red_167_, v_e_168_, v_x_169_, v___y_170_, v___y_171_, v___y_172_, v___y_173_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0___boxed(lean_object* v_v_176_, lean_object* v_red_177_, lean_object* v_e_178_, lean_object* v_x_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
uint8_t v_red_boxed_185_; lean_object* v_res_186_; 
v_red_boxed_185_ = lean_unbox(v_red_177_);
v_res_186_ = lp_mathlib_List_findM_x3f___at___00List_findDefeq_spec__0(v_v_176_, v_red_boxed_185_, v_e_178_, v_x_179_, v___y_180_, v___y_181_, v___y_182_, v___y_183_);
lean_dec(v___y_183_);
lean_dec_ref(v___y_182_);
lean_dec(v___y_181_);
lean_dec_ref(v___y_180_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1(lean_object* v_00_u03b1_187_, lean_object* v_msg_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___redArg(v_msg_188_, v___y_189_, v___y_190_, v___y_191_, v___y_192_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1___boxed(lean_object* v_00_u03b1_195_, lean_object* v_msg_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_Lean_throwError___at___00List_findDefeq_spec__1(v_00_u03b1_195_, v_msg_196_, v___y_197_, v___y_198_, v___y_199_, v___y_200_);
lean_dec(v___y_200_);
lean_dec_ref(v___y_199_);
lean_dec(v___y_198_);
lean_dec_ref(v___y_197_);
return v_res_202_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__0(lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_x_205_, lean_object* v_b_206_){
_start:
{
lean_object* v___x_207_; uint8_t v___x_208_; 
v___x_207_ = lean_apply_2(v_inst_203_, v_b_206_, v_inst_204_);
v___x_208_ = lean_unbox(v___x_207_);
if (v___x_208_ == 0)
{
uint8_t v___x_209_; 
v___x_209_ = 1;
return v___x_209_;
}
else
{
uint8_t v___x_210_; 
v___x_210_ = 0;
return v___x_210_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__0___boxed(lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_x_213_, lean_object* v_b_214_){
_start:
{
uint8_t v_res_215_; lean_object* v_r_216_; 
v_res_215_ = lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__0(v_inst_211_, v_inst_212_, v_x_213_, v_b_214_);
lean_dec(v_x_213_);
v_r_216_ = lean_box(v_res_215_);
return v_r_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__1(lean_object* v_b_u2082_217_, lean_object* v_inst_218_, lean_object* v_x_219_){
_start:
{
if (lean_obj_tag(v_x_219_) == 0)
{
lean_object* v___x_220_; 
lean_dec(v_inst_218_);
v___x_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_220_, 0, v_b_u2082_217_);
return v___x_220_;
}
else
{
lean_object* v_val_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_229_; 
v_val_221_ = lean_ctor_get(v_x_219_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v_x_219_);
if (v_isSharedCheck_229_ == 0)
{
v___x_223_ = v_x_219_;
v_isShared_224_ = v_isSharedCheck_229_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_val_221_);
lean_dec(v_x_219_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_229_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_227_; 
v___x_225_ = lean_apply_2(v_inst_218_, v_val_221_, v_b_u2082_217_);
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 0, v___x_225_);
v___x_227_ = v___x_223_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v___x_225_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
return v___x_227_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__2(lean_object* v_inst_230_, lean_object* v_c_231_, lean_object* v_t_232_, lean_object* v_a_233_, lean_object* v_b_u2082_234_){
_start:
{
lean_object* v___f_235_; lean_object* v___x_236_; 
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__1), 3, 2);
lean_closure_set(v___f_235_, 0, v_b_u2082_234_);
lean_closure_set(v___f_235_, 1, v_inst_230_);
v___x_236_ = l_Std_DTreeMap_Internal_Impl_Const_alter___redArg(v_c_231_, v_a_233_, v___f_235_, v_t_232_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__3(lean_object* v___f_237_, lean_object* v___f_238_, lean_object* v_f_239_, lean_object* v_g_240_){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_241_ = l_Std_DTreeMap_Internal_Impl_foldl___redArg(v___f_237_, v_f_239_, v_g_240_);
v___x_242_ = l_Std_DTreeMap_Internal_Impl_filter___redArg(v___f_238_, v___x_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg(lean_object* v_c_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_inst_246_){
_start:
{
lean_object* v___f_247_; lean_object* v___f_248_; lean_object* v___f_249_; 
v___f_247_ = lean_alloc_closure((void*)(lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_247_, 0, v_inst_246_);
lean_closure_set(v___f_247_, 1, v_inst_245_);
v___f_248_ = lean_alloc_closure((void*)(lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__2), 5, 2);
lean_closure_set(v___f_248_, 0, v_inst_244_);
lean_closure_set(v___f_248_, 1, v_c_243_);
v___f_249_ = lean_alloc_closure((void*)(lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg___lam__3), 4, 2);
lean_closure_set(v___f_249_, 0, v___f_248_);
lean_closure_set(v___f_249_, 1, v___f_247_);
return v___f_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib(lean_object* v_00_u03b1_250_, lean_object* v_00_u03b2_251_, lean_object* v_c_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lp_mathlib_instAddTreeMapOfZeroOfDecidableEq__mathlib___redArg(v_c_252_, v_inst_253_, v_inst_254_, v_inst_255_);
return v___x_256_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Monom_one(void){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lean_box(1);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1(lean_object* v_init_258_, lean_object* v_x_259_){
_start:
{
if (lean_obj_tag(v_x_259_) == 0)
{
lean_object* v_v_260_; lean_object* v_l_261_; lean_object* v_r_262_; lean_object* v___x_263_; lean_object* v___x_264_; 
v_v_260_ = lean_ctor_get(v_x_259_, 2);
v_l_261_ = lean_ctor_get(v_x_259_, 3);
v_r_262_ = lean_ctor_get(v_x_259_, 4);
v___x_263_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1(v_init_258_, v_r_262_);
lean_inc(v_v_260_);
v___x_264_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_264_, 0, v_v_260_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v_init_258_ = v___x_264_;
v_x_259_ = v_l_261_;
goto _start;
}
else
{
return v_init_258_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1___boxed(lean_object* v_init_266_, lean_object* v_x_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1(v_init_266_, v_x_267_);
lean_dec(v_x_267_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0(lean_object* v_init_269_, lean_object* v_x_270_){
_start:
{
if (lean_obj_tag(v_x_270_) == 0)
{
lean_object* v_k_271_; lean_object* v_l_272_; lean_object* v_r_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v_k_271_ = lean_ctor_get(v_x_270_, 1);
v_l_272_ = lean_ctor_get(v_x_270_, 3);
v_r_273_ = lean_ctor_get(v_x_270_, 4);
v___x_274_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0(v_init_269_, v_r_273_);
lean_inc(v_k_271_);
v___x_275_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_275_, 0, v_k_271_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v_init_269_ = v___x_275_;
v_x_270_ = v_l_272_;
goto _start;
}
else
{
return v_init_269_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0___boxed(lean_object* v_init_277_, lean_object* v_x_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0(v_init_277_, v_x_278_);
lean_dec(v_x_278_);
return v_res_279_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt(lean_object* v_a_281_, lean_object* v_b_282_){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; uint8_t v___x_288_; 
v___x_283_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___x_284_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt___closed__0));
v___x_285_ = lean_box(0);
v___x_286_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0(v___x_285_, v_a_281_);
v___x_287_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__0(v___x_285_, v_b_282_);
lean_inc(v___x_287_);
lean_inc(v___x_286_);
lean_inc_ref(v___x_283_);
v___x_288_ = l_List_decidableLex___redArg(v___x_283_, v___x_284_, v___x_286_, v___x_287_);
if (v___x_288_ == 0)
{
uint8_t v___x_289_; 
lean_inc_ref(v___x_283_);
v___x_289_ = l_instDecidableEqList___redArg(v___x_283_, v___x_286_, v___x_287_);
if (v___x_289_ == 0)
{
lean_dec_ref(v___x_283_);
return v___x_289_;
}
else
{
lean_object* v___x_290_; lean_object* v___x_291_; uint8_t v___x_292_; 
v___x_290_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1(v___x_285_, v_a_281_);
v___x_291_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Monom_lt_spec__1(v___x_285_, v_b_282_);
v___x_292_ = l_List_decidableLex___redArg(v___x_283_, v___x_284_, v___x_290_, v___x_291_);
return v___x_292_;
}
}
else
{
lean_dec(v___x_287_);
lean_dec(v___x_286_);
lean_dec_ref(v___x_283_);
return v___x_288_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt___boxed(lean_object* v_a_293_, lean_object* v_b_294_){
_start:
{
uint8_t v_res_295_; lean_object* v_r_296_; 
v_res_295_ = lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt(v_a_293_, v_b_294_);
lean_dec(v_b_294_);
lean_dec(v_a_293_);
v_r_296_ = lean_box(v_res_295_);
return v_r_296_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__0(lean_object* v_x_297_, lean_object* v_y_298_){
_start:
{
uint8_t v___x_299_; 
v___x_299_ = lean_nat_dec_lt(v_x_297_, v_y_298_);
if (v___x_299_ == 0)
{
uint8_t v___x_300_; 
v___x_300_ = lean_nat_dec_eq(v_x_297_, v_y_298_);
if (v___x_300_ == 0)
{
uint8_t v___x_301_; 
v___x_301_ = 2;
return v___x_301_;
}
else
{
uint8_t v___x_302_; 
v___x_302_ = 1;
return v___x_302_;
}
}
else
{
uint8_t v___x_303_; 
v___x_303_ = 0;
return v___x_303_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__0___boxed(lean_object* v_x_304_, lean_object* v_y_305_){
_start:
{
uint8_t v_res_306_; lean_object* v_r_307_; 
v_res_306_ = lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__0(v_x_304_, v_y_305_);
lean_dec(v_y_305_);
lean_dec(v_x_304_);
v_r_307_ = lean_box(v_res_306_);
return v_r_307_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___closed__0(void){
_start:
{
lean_object* v___x_308_; lean_object* v___f_309_; 
v___x_308_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___f_309_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_309_, 0, v___x_308_);
return v___f_309_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1(lean_object* v___f_310_, lean_object* v_x_311_, lean_object* v_y_312_){
_start:
{
uint8_t v___x_313_; 
v___x_313_ = lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt(v_x_311_, v_y_312_);
if (v___x_313_ == 0)
{
lean_object* v___f_314_; uint8_t v___x_315_; 
v___f_314_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___closed__0);
v___x_315_ = l_Std_DTreeMap_Internal_Impl_Const_beq___redArg(v___f_310_, v___f_314_, v_x_311_, v_y_312_);
if (v___x_315_ == 0)
{
uint8_t v___x_316_; 
v___x_316_ = 2;
return v___x_316_;
}
else
{
uint8_t v___x_317_; 
v___x_317_ = 1;
return v___x_317_;
}
}
else
{
uint8_t v___x_318_; 
lean_dec(v_y_312_);
lean_dec(v_x_311_);
lean_dec_ref(v___f_310_);
v___x_318_ = 0;
return v___x_318_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1___boxed(lean_object* v___f_319_, lean_object* v_x_320_, lean_object* v_y_321_){
_start:
{
uint8_t v_res_322_; lean_object* v_r_323_; 
v_res_322_ = lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___lam__1(v___f_319_, v_x_320_, v_y_321_);
v_r_323_ = lean_box(v_res_322_);
return v_r_323_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__4(lean_object* v_x_328_, lean_object* v_x_329_){
_start:
{
if (lean_obj_tag(v_x_328_) == 0)
{
if (lean_obj_tag(v_x_329_) == 0)
{
uint8_t v___x_330_; 
v___x_330_ = 1;
return v___x_330_;
}
else
{
uint8_t v___x_331_; 
v___x_331_ = 0;
return v___x_331_;
}
}
else
{
if (lean_obj_tag(v_x_329_) == 0)
{
uint8_t v___x_332_; 
v___x_332_ = 0;
return v___x_332_;
}
else
{
lean_object* v_val_333_; lean_object* v_val_334_; uint8_t v___x_335_; 
v_val_333_ = lean_ctor_get(v_x_328_, 0);
v_val_334_ = lean_ctor_get(v_x_329_, 0);
v___x_335_ = lean_nat_dec_eq(v_val_333_, v_val_334_);
return v___x_335_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_x_336_, lean_object* v_x_337_){
_start:
{
uint8_t v_res_338_; lean_object* v_r_339_; 
v_res_338_ = lp_mathlib_Option_instBEq_beq___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__4(v_x_336_, v_x_337_);
lean_dec(v_x_337_);
lean_dec(v_x_336_);
v_r_339_ = lean_box(v_res_338_);
return v_r_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_cmp_340_, lean_object* v_t_341_, lean_object* v_k_342_){
_start:
{
if (lean_obj_tag(v_t_341_) == 0)
{
lean_object* v_k_343_; lean_object* v_v_344_; lean_object* v_l_345_; lean_object* v_r_346_; lean_object* v___x_347_; uint8_t v___x_348_; 
v_k_343_ = lean_ctor_get(v_t_341_, 1);
lean_inc(v_k_343_);
v_v_344_ = lean_ctor_get(v_t_341_, 2);
lean_inc(v_v_344_);
v_l_345_ = lean_ctor_get(v_t_341_, 3);
lean_inc(v_l_345_);
v_r_346_ = lean_ctor_get(v_t_341_, 4);
lean_inc(v_r_346_);
lean_dec_ref_known(v_t_341_, 5);
lean_inc_ref(v_cmp_340_);
lean_inc(v_k_342_);
v___x_347_ = lean_apply_2(v_cmp_340_, v_k_342_, v_k_343_);
v___x_348_ = lean_unbox(v___x_347_);
switch(v___x_348_)
{
case 0:
{
lean_dec(v_r_346_);
lean_dec(v_v_344_);
v_t_341_ = v_l_345_;
goto _start;
}
case 1:
{
lean_object* v___x_350_; 
lean_dec(v_r_346_);
lean_dec(v_l_345_);
lean_dec(v_k_342_);
lean_dec_ref(v_cmp_340_);
v___x_350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_350_, 0, v_v_344_);
return v___x_350_;
}
default: 
{
lean_dec(v_l_345_);
lean_dec(v_v_344_);
v_t_341_ = v_r_346_;
goto _start;
}
}
}
else
{
lean_object* v___x_352_; 
lean_dec(v_k_342_);
lean_dec_ref(v_cmp_340_);
v___x_352_ = lean_box(0);
return v___x_352_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg(lean_object* v_cmp_356_, lean_object* v_t_u2082_357_, lean_object* v_init_358_, lean_object* v_x_359_){
_start:
{
if (lean_obj_tag(v_x_359_) == 0)
{
lean_object* v_k_360_; lean_object* v_v_361_; lean_object* v_l_362_; lean_object* v_r_363_; lean_object* v___x_364_; 
v_k_360_ = lean_ctor_get(v_x_359_, 1);
lean_inc(v_k_360_);
v_v_361_ = lean_ctor_get(v_x_359_, 2);
lean_inc(v_v_361_);
v_l_362_ = lean_ctor_get(v_x_359_, 3);
lean_inc(v_l_362_);
v_r_363_ = lean_ctor_get(v_x_359_, 4);
lean_inc(v_r_363_);
lean_dec_ref_known(v_x_359_, 5);
lean_inc(v_t_u2082_357_);
lean_inc_ref(v_cmp_356_);
v___x_364_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg(v_cmp_356_, v_t_u2082_357_, v_init_358_, v_l_362_);
if (lean_obj_tag(v___x_364_) == 0)
{
lean_dec(v_r_363_);
lean_dec(v_v_361_);
lean_dec(v_k_360_);
lean_dec(v_t_u2082_357_);
lean_dec_ref(v_cmp_356_);
return v___x_364_;
}
else
{
lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_380_; 
v_isSharedCheck_380_ = !lean_is_exclusive(v___x_364_);
if (v_isSharedCheck_380_ == 0)
{
lean_object* v_unused_381_; 
v_unused_381_ = lean_ctor_get(v___x_364_, 0);
lean_dec(v_unused_381_);
v___x_366_ = v___x_364_;
v_isShared_367_ = v_isSharedCheck_380_;
goto v_resetjp_365_;
}
else
{
lean_dec(v___x_364_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_380_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; uint8_t v___x_371_; 
v___x_368_ = lean_box(0);
lean_inc(v_t_u2082_357_);
lean_inc_ref(v_cmp_356_);
v___x_369_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__3___redArg(v_cmp_356_, v_t_u2082_357_, v_k_360_);
v___x_370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_370_, 0, v_v_361_);
v___x_371_ = lp_mathlib_Option_instBEq_beq___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__4(v___x_369_, v___x_370_);
lean_dec_ref_known(v___x_370_, 1);
lean_dec(v___x_369_);
if (v___x_371_ == 0)
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_376_; 
lean_dec(v_r_363_);
lean_dec(v_t_u2082_357_);
lean_dec_ref(v_cmp_356_);
v___x_372_ = lean_box(v___x_371_);
v___x_373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_373_, 0, v___x_372_);
v___x_374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_373_);
lean_ctor_set(v___x_374_, 1, v___x_368_);
if (v_isShared_367_ == 0)
{
lean_ctor_set_tag(v___x_366_, 0);
lean_ctor_set(v___x_366_, 0, v___x_374_);
v___x_376_ = v___x_366_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v___x_374_);
v___x_376_ = v_reuseFailAlloc_377_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
return v___x_376_;
}
}
else
{
lean_object* v___x_378_; 
lean_del_object(v___x_366_);
v___x_378_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg___closed__0));
v_init_358_ = v___x_378_;
v_x_359_ = v_r_363_;
goto _start;
}
}
}
}
else
{
lean_object* v___x_382_; 
lean_dec(v_t_u2082_357_);
lean_dec_ref(v_cmp_356_);
v___x_382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_382_, 0, v_init_358_);
return v___x_382_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(lean_object* v_cmp_383_, lean_object* v_t_u2081_384_, lean_object* v_t_u2082_385_){
_start:
{
lean_object* v___y_387_; lean_object* v___y_393_; lean_object* v___y_394_; lean_object* v___y_400_; 
if (lean_obj_tag(v_t_u2081_384_) == 0)
{
lean_object* v_size_403_; 
v_size_403_ = lean_ctor_get(v_t_u2081_384_, 0);
lean_inc(v_size_403_);
v___y_400_ = v_size_403_;
goto v___jp_399_;
}
else
{
lean_object* v___x_404_; 
v___x_404_ = lean_unsigned_to_nat(0u);
v___y_400_ = v___x_404_;
goto v___jp_399_;
}
v___jp_386_:
{
lean_object* v_fst_388_; 
v_fst_388_ = lean_ctor_get(v___y_387_, 0);
lean_inc(v_fst_388_);
lean_dec_ref(v___y_387_);
if (lean_obj_tag(v_fst_388_) == 0)
{
uint8_t v___x_389_; 
v___x_389_ = 1;
return v___x_389_;
}
else
{
lean_object* v_val_390_; uint8_t v___x_391_; 
v_val_390_ = lean_ctor_get(v_fst_388_, 0);
lean_inc(v_val_390_);
lean_dec_ref_known(v_fst_388_, 1);
v___x_391_ = lean_unbox(v_val_390_);
lean_dec(v_val_390_);
return v___x_391_;
}
}
v___jp_392_:
{
uint8_t v___x_395_; 
v___x_395_ = lean_nat_dec_eq(v___y_393_, v___y_394_);
lean_dec(v___y_394_);
lean_dec(v___y_393_);
if (v___x_395_ == 0)
{
lean_dec(v_t_u2082_385_);
lean_dec(v_t_u2081_384_);
lean_dec_ref(v_cmp_383_);
return v___x_395_;
}
else
{
lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v_a_398_; 
v___x_396_ = ((lean_object*)(lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg___closed__0));
v___x_397_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg(v_cmp_383_, v_t_u2082_385_, v___x_396_, v_t_u2081_384_);
v_a_398_ = lean_ctor_get(v___x_397_, 0);
lean_inc(v_a_398_);
lean_dec_ref(v___x_397_);
v___y_387_ = v_a_398_;
goto v___jp_386_;
}
}
v___jp_399_:
{
if (lean_obj_tag(v_t_u2082_385_) == 0)
{
lean_object* v_size_401_; 
v_size_401_ = lean_ctor_get(v_t_u2082_385_, 0);
lean_inc(v_size_401_);
v___y_393_ = v___y_400_;
v___y_394_ = v_size_401_;
goto v___jp_392_;
}
else
{
lean_object* v___x_402_; 
v___x_402_ = lean_unsigned_to_nat(0u);
v___y_393_ = v___y_400_;
v___y_394_ = v___x_402_;
goto v___jp_392_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_cmp_405_, lean_object* v_t_u2081_406_, lean_object* v_t_u2082_407_){
_start:
{
uint8_t v_res_408_; lean_object* v_r_409_; 
v_res_408_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v_cmp_405_, v_t_u2081_406_, v_t_u2082_407_);
v_r_409_ = lean_box(v_res_408_);
return v_r_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(lean_object* v_k_410_, lean_object* v_v_411_, lean_object* v_t_412_){
_start:
{
if (lean_obj_tag(v_t_412_) == 0)
{
lean_object* v_size_413_; lean_object* v_k_414_; lean_object* v_v_415_; lean_object* v_l_416_; lean_object* v_r_417_; lean_object* v___x_419_; uint8_t v_isShared_420_; uint8_t v_isSharedCheck_699_; 
v_size_413_ = lean_ctor_get(v_t_412_, 0);
v_k_414_ = lean_ctor_get(v_t_412_, 1);
v_v_415_ = lean_ctor_get(v_t_412_, 2);
v_l_416_ = lean_ctor_get(v_t_412_, 3);
v_r_417_ = lean_ctor_get(v_t_412_, 4);
v_isSharedCheck_699_ = !lean_is_exclusive(v_t_412_);
if (v_isSharedCheck_699_ == 0)
{
v___x_419_ = v_t_412_;
v_isShared_420_ = v_isSharedCheck_699_;
goto v_resetjp_418_;
}
else
{
lean_inc(v_r_417_);
lean_inc(v_l_416_);
lean_inc(v_v_415_);
lean_inc(v_k_414_);
lean_inc(v_size_413_);
lean_dec(v_t_412_);
v___x_419_ = lean_box(0);
v_isShared_420_ = v_isSharedCheck_699_;
goto v_resetjp_418_;
}
v_resetjp_418_:
{
uint8_t v___x_421_; 
v___x_421_ = lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt(v_k_410_, v_k_414_);
if (v___x_421_ == 0)
{
lean_object* v___f_422_; uint8_t v___x_423_; 
v___f_422_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__0));
lean_inc(v_k_414_);
lean_inc(v_k_410_);
v___x_423_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v___f_422_, v_k_410_, v_k_414_);
if (v___x_423_ == 0)
{
lean_object* v_impl_424_; lean_object* v___x_425_; 
lean_dec(v_size_413_);
v_impl_424_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v_k_410_, v_v_411_, v_r_417_);
v___x_425_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_416_) == 0)
{
lean_object* v_size_426_; lean_object* v_size_427_; lean_object* v_k_428_; lean_object* v_v_429_; lean_object* v_l_430_; lean_object* v_r_431_; lean_object* v___x_432_; lean_object* v___x_433_; uint8_t v___x_434_; 
v_size_426_ = lean_ctor_get(v_l_416_, 0);
v_size_427_ = lean_ctor_get(v_impl_424_, 0);
lean_inc(v_size_427_);
v_k_428_ = lean_ctor_get(v_impl_424_, 1);
lean_inc(v_k_428_);
v_v_429_ = lean_ctor_get(v_impl_424_, 2);
lean_inc(v_v_429_);
v_l_430_ = lean_ctor_get(v_impl_424_, 3);
lean_inc(v_l_430_);
v_r_431_ = lean_ctor_get(v_impl_424_, 4);
lean_inc(v_r_431_);
v___x_432_ = lean_unsigned_to_nat(3u);
v___x_433_ = lean_nat_mul(v___x_432_, v_size_426_);
v___x_434_ = lean_nat_dec_lt(v___x_433_, v_size_427_);
lean_dec(v___x_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_438_; 
lean_dec(v_r_431_);
lean_dec(v_l_430_);
lean_dec(v_v_429_);
lean_dec(v_k_428_);
v___x_435_ = lean_nat_add(v___x_425_, v_size_426_);
v___x_436_ = lean_nat_add(v___x_435_, v_size_427_);
lean_dec(v_size_427_);
lean_dec(v___x_435_);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v_impl_424_);
lean_ctor_set(v___x_419_, 0, v___x_436_);
v___x_438_ = v___x_419_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v___x_436_);
lean_ctor_set(v_reuseFailAlloc_439_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_439_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_439_, 3, v_l_416_);
lean_ctor_set(v_reuseFailAlloc_439_, 4, v_impl_424_);
v___x_438_ = v_reuseFailAlloc_439_;
goto v_reusejp_437_;
}
v_reusejp_437_:
{
return v___x_438_;
}
}
else
{
lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_503_; 
v_isSharedCheck_503_ = !lean_is_exclusive(v_impl_424_);
if (v_isSharedCheck_503_ == 0)
{
lean_object* v_unused_504_; lean_object* v_unused_505_; lean_object* v_unused_506_; lean_object* v_unused_507_; lean_object* v_unused_508_; 
v_unused_504_ = lean_ctor_get(v_impl_424_, 4);
lean_dec(v_unused_504_);
v_unused_505_ = lean_ctor_get(v_impl_424_, 3);
lean_dec(v_unused_505_);
v_unused_506_ = lean_ctor_get(v_impl_424_, 2);
lean_dec(v_unused_506_);
v_unused_507_ = lean_ctor_get(v_impl_424_, 1);
lean_dec(v_unused_507_);
v_unused_508_ = lean_ctor_get(v_impl_424_, 0);
lean_dec(v_unused_508_);
v___x_441_ = v_impl_424_;
v_isShared_442_ = v_isSharedCheck_503_;
goto v_resetjp_440_;
}
else
{
lean_dec(v_impl_424_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_503_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v_size_443_; lean_object* v_k_444_; lean_object* v_v_445_; lean_object* v_l_446_; lean_object* v_r_447_; lean_object* v_size_448_; lean_object* v___x_449_; lean_object* v___x_450_; uint8_t v___x_451_; 
v_size_443_ = lean_ctor_get(v_l_430_, 0);
v_k_444_ = lean_ctor_get(v_l_430_, 1);
v_v_445_ = lean_ctor_get(v_l_430_, 2);
v_l_446_ = lean_ctor_get(v_l_430_, 3);
v_r_447_ = lean_ctor_get(v_l_430_, 4);
v_size_448_ = lean_ctor_get(v_r_431_, 0);
v___x_449_ = lean_unsigned_to_nat(2u);
v___x_450_ = lean_nat_mul(v___x_449_, v_size_448_);
v___x_451_ = lean_nat_dec_lt(v_size_443_, v___x_450_);
lean_dec(v___x_450_);
if (v___x_451_ == 0)
{
lean_object* v___x_453_; uint8_t v_isShared_454_; uint8_t v_isSharedCheck_479_; 
lean_inc(v_r_447_);
lean_inc(v_l_446_);
lean_inc(v_v_445_);
lean_inc(v_k_444_);
v_isSharedCheck_479_ = !lean_is_exclusive(v_l_430_);
if (v_isSharedCheck_479_ == 0)
{
lean_object* v_unused_480_; lean_object* v_unused_481_; lean_object* v_unused_482_; lean_object* v_unused_483_; lean_object* v_unused_484_; 
v_unused_480_ = lean_ctor_get(v_l_430_, 4);
lean_dec(v_unused_480_);
v_unused_481_ = lean_ctor_get(v_l_430_, 3);
lean_dec(v_unused_481_);
v_unused_482_ = lean_ctor_get(v_l_430_, 2);
lean_dec(v_unused_482_);
v_unused_483_ = lean_ctor_get(v_l_430_, 1);
lean_dec(v_unused_483_);
v_unused_484_ = lean_ctor_get(v_l_430_, 0);
lean_dec(v_unused_484_);
v___x_453_ = v_l_430_;
v_isShared_454_ = v_isSharedCheck_479_;
goto v_resetjp_452_;
}
else
{
lean_dec(v_l_430_);
v___x_453_ = lean_box(0);
v_isShared_454_ = v_isSharedCheck_479_;
goto v_resetjp_452_;
}
v_resetjp_452_:
{
lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___y_458_; lean_object* v___y_459_; lean_object* v___y_460_; lean_object* v___y_469_; 
v___x_455_ = lean_nat_add(v___x_425_, v_size_426_);
v___x_456_ = lean_nat_add(v___x_455_, v_size_427_);
lean_dec(v_size_427_);
if (lean_obj_tag(v_l_446_) == 0)
{
lean_object* v_size_477_; 
v_size_477_ = lean_ctor_get(v_l_446_, 0);
lean_inc(v_size_477_);
v___y_469_ = v_size_477_;
goto v___jp_468_;
}
else
{
lean_object* v___x_478_; 
v___x_478_ = lean_unsigned_to_nat(0u);
v___y_469_ = v___x_478_;
goto v___jp_468_;
}
v___jp_457_:
{
lean_object* v___x_461_; lean_object* v___x_463_; 
v___x_461_ = lean_nat_add(v___y_458_, v___y_460_);
lean_dec(v___y_460_);
lean_dec(v___y_458_);
if (v_isShared_454_ == 0)
{
lean_ctor_set(v___x_453_, 4, v_r_431_);
lean_ctor_set(v___x_453_, 3, v_r_447_);
lean_ctor_set(v___x_453_, 2, v_v_429_);
lean_ctor_set(v___x_453_, 1, v_k_428_);
lean_ctor_set(v___x_453_, 0, v___x_461_);
v___x_463_ = v___x_453_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_467_; 
v_reuseFailAlloc_467_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_467_, 0, v___x_461_);
lean_ctor_set(v_reuseFailAlloc_467_, 1, v_k_428_);
lean_ctor_set(v_reuseFailAlloc_467_, 2, v_v_429_);
lean_ctor_set(v_reuseFailAlloc_467_, 3, v_r_447_);
lean_ctor_set(v_reuseFailAlloc_467_, 4, v_r_431_);
v___x_463_ = v_reuseFailAlloc_467_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
lean_object* v___x_465_; 
if (v_isShared_442_ == 0)
{
lean_ctor_set(v___x_441_, 4, v___x_463_);
lean_ctor_set(v___x_441_, 3, v___y_459_);
lean_ctor_set(v___x_441_, 2, v_v_445_);
lean_ctor_set(v___x_441_, 1, v_k_444_);
lean_ctor_set(v___x_441_, 0, v___x_456_);
v___x_465_ = v___x_441_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_466_; 
v_reuseFailAlloc_466_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_466_, 0, v___x_456_);
lean_ctor_set(v_reuseFailAlloc_466_, 1, v_k_444_);
lean_ctor_set(v_reuseFailAlloc_466_, 2, v_v_445_);
lean_ctor_set(v_reuseFailAlloc_466_, 3, v___y_459_);
lean_ctor_set(v_reuseFailAlloc_466_, 4, v___x_463_);
v___x_465_ = v_reuseFailAlloc_466_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
return v___x_465_;
}
}
}
v___jp_468_:
{
lean_object* v___x_470_; lean_object* v___x_472_; 
v___x_470_ = lean_nat_add(v___x_455_, v___y_469_);
lean_dec(v___y_469_);
lean_dec(v___x_455_);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v_l_446_);
lean_ctor_set(v___x_419_, 0, v___x_470_);
v___x_472_ = v___x_419_;
goto v_reusejp_471_;
}
else
{
lean_object* v_reuseFailAlloc_476_; 
v_reuseFailAlloc_476_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_476_, 0, v___x_470_);
lean_ctor_set(v_reuseFailAlloc_476_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_476_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_476_, 3, v_l_416_);
lean_ctor_set(v_reuseFailAlloc_476_, 4, v_l_446_);
v___x_472_ = v_reuseFailAlloc_476_;
goto v_reusejp_471_;
}
v_reusejp_471_:
{
lean_object* v___x_473_; 
v___x_473_ = lean_nat_add(v___x_425_, v_size_448_);
if (lean_obj_tag(v_r_447_) == 0)
{
lean_object* v_size_474_; 
v_size_474_ = lean_ctor_get(v_r_447_, 0);
lean_inc(v_size_474_);
v___y_458_ = v___x_473_;
v___y_459_ = v___x_472_;
v___y_460_ = v_size_474_;
goto v___jp_457_;
}
else
{
lean_object* v___x_475_; 
v___x_475_ = lean_unsigned_to_nat(0u);
v___y_458_ = v___x_473_;
v___y_459_ = v___x_472_;
v___y_460_ = v___x_475_;
goto v___jp_457_;
}
}
}
}
}
else
{
lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_489_; 
lean_del_object(v___x_419_);
v___x_485_ = lean_nat_add(v___x_425_, v_size_426_);
v___x_486_ = lean_nat_add(v___x_485_, v_size_427_);
lean_dec(v_size_427_);
v___x_487_ = lean_nat_add(v___x_485_, v_size_443_);
lean_dec(v___x_485_);
lean_inc_ref(v_l_416_);
if (v_isShared_442_ == 0)
{
lean_ctor_set(v___x_441_, 4, v_l_430_);
lean_ctor_set(v___x_441_, 3, v_l_416_);
lean_ctor_set(v___x_441_, 2, v_v_415_);
lean_ctor_set(v___x_441_, 1, v_k_414_);
lean_ctor_set(v___x_441_, 0, v___x_487_);
v___x_489_ = v___x_441_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v___x_487_);
lean_ctor_set(v_reuseFailAlloc_502_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_502_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_502_, 3, v_l_416_);
lean_ctor_set(v_reuseFailAlloc_502_, 4, v_l_430_);
v___x_489_ = v_reuseFailAlloc_502_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_496_; 
v_isSharedCheck_496_ = !lean_is_exclusive(v_l_416_);
if (v_isSharedCheck_496_ == 0)
{
lean_object* v_unused_497_; lean_object* v_unused_498_; lean_object* v_unused_499_; lean_object* v_unused_500_; lean_object* v_unused_501_; 
v_unused_497_ = lean_ctor_get(v_l_416_, 4);
lean_dec(v_unused_497_);
v_unused_498_ = lean_ctor_get(v_l_416_, 3);
lean_dec(v_unused_498_);
v_unused_499_ = lean_ctor_get(v_l_416_, 2);
lean_dec(v_unused_499_);
v_unused_500_ = lean_ctor_get(v_l_416_, 1);
lean_dec(v_unused_500_);
v_unused_501_ = lean_ctor_get(v_l_416_, 0);
lean_dec(v_unused_501_);
v___x_491_ = v_l_416_;
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
else
{
lean_dec(v_l_416_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v___x_494_; 
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 4, v_r_431_);
lean_ctor_set(v___x_491_, 3, v___x_489_);
lean_ctor_set(v___x_491_, 2, v_v_429_);
lean_ctor_set(v___x_491_, 1, v_k_428_);
lean_ctor_set(v___x_491_, 0, v___x_486_);
v___x_494_ = v___x_491_;
goto v_reusejp_493_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v___x_486_);
lean_ctor_set(v_reuseFailAlloc_495_, 1, v_k_428_);
lean_ctor_set(v_reuseFailAlloc_495_, 2, v_v_429_);
lean_ctor_set(v_reuseFailAlloc_495_, 3, v___x_489_);
lean_ctor_set(v_reuseFailAlloc_495_, 4, v_r_431_);
v___x_494_ = v_reuseFailAlloc_495_;
goto v_reusejp_493_;
}
v_reusejp_493_:
{
return v___x_494_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_509_; 
v_l_509_ = lean_ctor_get(v_impl_424_, 3);
lean_inc(v_l_509_);
if (lean_obj_tag(v_l_509_) == 0)
{
lean_object* v_r_510_; lean_object* v_k_511_; lean_object* v_v_512_; lean_object* v___x_514_; uint8_t v_isShared_515_; uint8_t v_isSharedCheck_535_; 
v_r_510_ = lean_ctor_get(v_impl_424_, 4);
v_k_511_ = lean_ctor_get(v_impl_424_, 1);
v_v_512_ = lean_ctor_get(v_impl_424_, 2);
v_isSharedCheck_535_ = !lean_is_exclusive(v_impl_424_);
if (v_isSharedCheck_535_ == 0)
{
lean_object* v_unused_536_; lean_object* v_unused_537_; 
v_unused_536_ = lean_ctor_get(v_impl_424_, 3);
lean_dec(v_unused_536_);
v_unused_537_ = lean_ctor_get(v_impl_424_, 0);
lean_dec(v_unused_537_);
v___x_514_ = v_impl_424_;
v_isShared_515_ = v_isSharedCheck_535_;
goto v_resetjp_513_;
}
else
{
lean_inc(v_r_510_);
lean_inc(v_v_512_);
lean_inc(v_k_511_);
lean_dec(v_impl_424_);
v___x_514_ = lean_box(0);
v_isShared_515_ = v_isSharedCheck_535_;
goto v_resetjp_513_;
}
v_resetjp_513_:
{
lean_object* v_k_516_; lean_object* v_v_517_; lean_object* v___x_519_; uint8_t v_isShared_520_; uint8_t v_isSharedCheck_531_; 
v_k_516_ = lean_ctor_get(v_l_509_, 1);
v_v_517_ = lean_ctor_get(v_l_509_, 2);
v_isSharedCheck_531_ = !lean_is_exclusive(v_l_509_);
if (v_isSharedCheck_531_ == 0)
{
lean_object* v_unused_532_; lean_object* v_unused_533_; lean_object* v_unused_534_; 
v_unused_532_ = lean_ctor_get(v_l_509_, 4);
lean_dec(v_unused_532_);
v_unused_533_ = lean_ctor_get(v_l_509_, 3);
lean_dec(v_unused_533_);
v_unused_534_ = lean_ctor_get(v_l_509_, 0);
lean_dec(v_unused_534_);
v___x_519_ = v_l_509_;
v_isShared_520_ = v_isSharedCheck_531_;
goto v_resetjp_518_;
}
else
{
lean_inc(v_v_517_);
lean_inc(v_k_516_);
lean_dec(v_l_509_);
v___x_519_ = lean_box(0);
v_isShared_520_ = v_isSharedCheck_531_;
goto v_resetjp_518_;
}
v_resetjp_518_:
{
lean_object* v___x_521_; lean_object* v___x_523_; 
v___x_521_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_510_, 2);
if (v_isShared_520_ == 0)
{
lean_ctor_set(v___x_519_, 4, v_r_510_);
lean_ctor_set(v___x_519_, 3, v_r_510_);
lean_ctor_set(v___x_519_, 2, v_v_415_);
lean_ctor_set(v___x_519_, 1, v_k_414_);
lean_ctor_set(v___x_519_, 0, v___x_425_);
v___x_523_ = v___x_519_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v___x_425_);
lean_ctor_set(v_reuseFailAlloc_530_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_530_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_530_, 3, v_r_510_);
lean_ctor_set(v_reuseFailAlloc_530_, 4, v_r_510_);
v___x_523_ = v_reuseFailAlloc_530_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
lean_object* v___x_525_; 
lean_inc(v_r_510_);
if (v_isShared_515_ == 0)
{
lean_ctor_set(v___x_514_, 3, v_r_510_);
lean_ctor_set(v___x_514_, 0, v___x_425_);
v___x_525_ = v___x_514_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v___x_425_);
lean_ctor_set(v_reuseFailAlloc_529_, 1, v_k_511_);
lean_ctor_set(v_reuseFailAlloc_529_, 2, v_v_512_);
lean_ctor_set(v_reuseFailAlloc_529_, 3, v_r_510_);
lean_ctor_set(v_reuseFailAlloc_529_, 4, v_r_510_);
v___x_525_ = v_reuseFailAlloc_529_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
lean_object* v___x_527_; 
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v___x_525_);
lean_ctor_set(v___x_419_, 3, v___x_523_);
lean_ctor_set(v___x_419_, 2, v_v_517_);
lean_ctor_set(v___x_419_, 1, v_k_516_);
lean_ctor_set(v___x_419_, 0, v___x_521_);
v___x_527_ = v___x_419_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v___x_521_);
lean_ctor_set(v_reuseFailAlloc_528_, 1, v_k_516_);
lean_ctor_set(v_reuseFailAlloc_528_, 2, v_v_517_);
lean_ctor_set(v_reuseFailAlloc_528_, 3, v___x_523_);
lean_ctor_set(v_reuseFailAlloc_528_, 4, v___x_525_);
v___x_527_ = v_reuseFailAlloc_528_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
return v___x_527_;
}
}
}
}
}
}
else
{
lean_object* v_r_538_; 
v_r_538_ = lean_ctor_get(v_impl_424_, 4);
lean_inc(v_r_538_);
if (lean_obj_tag(v_r_538_) == 0)
{
lean_object* v_k_539_; lean_object* v_v_540_; lean_object* v___x_542_; uint8_t v_isShared_543_; uint8_t v_isSharedCheck_551_; 
v_k_539_ = lean_ctor_get(v_impl_424_, 1);
v_v_540_ = lean_ctor_get(v_impl_424_, 2);
v_isSharedCheck_551_ = !lean_is_exclusive(v_impl_424_);
if (v_isSharedCheck_551_ == 0)
{
lean_object* v_unused_552_; lean_object* v_unused_553_; lean_object* v_unused_554_; 
v_unused_552_ = lean_ctor_get(v_impl_424_, 4);
lean_dec(v_unused_552_);
v_unused_553_ = lean_ctor_get(v_impl_424_, 3);
lean_dec(v_unused_553_);
v_unused_554_ = lean_ctor_get(v_impl_424_, 0);
lean_dec(v_unused_554_);
v___x_542_ = v_impl_424_;
v_isShared_543_ = v_isSharedCheck_551_;
goto v_resetjp_541_;
}
else
{
lean_inc(v_v_540_);
lean_inc(v_k_539_);
lean_dec(v_impl_424_);
v___x_542_ = lean_box(0);
v_isShared_543_ = v_isSharedCheck_551_;
goto v_resetjp_541_;
}
v_resetjp_541_:
{
lean_object* v___x_544_; lean_object* v___x_546_; 
v___x_544_ = lean_unsigned_to_nat(3u);
if (v_isShared_543_ == 0)
{
lean_ctor_set(v___x_542_, 4, v_l_509_);
lean_ctor_set(v___x_542_, 2, v_v_415_);
lean_ctor_set(v___x_542_, 1, v_k_414_);
lean_ctor_set(v___x_542_, 0, v___x_425_);
v___x_546_ = v___x_542_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v___x_425_);
lean_ctor_set(v_reuseFailAlloc_550_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_550_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_550_, 3, v_l_509_);
lean_ctor_set(v_reuseFailAlloc_550_, 4, v_l_509_);
v___x_546_ = v_reuseFailAlloc_550_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
lean_object* v___x_548_; 
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v_r_538_);
lean_ctor_set(v___x_419_, 3, v___x_546_);
lean_ctor_set(v___x_419_, 2, v_v_540_);
lean_ctor_set(v___x_419_, 1, v_k_539_);
lean_ctor_set(v___x_419_, 0, v___x_544_);
v___x_548_ = v___x_419_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v___x_544_);
lean_ctor_set(v_reuseFailAlloc_549_, 1, v_k_539_);
lean_ctor_set(v_reuseFailAlloc_549_, 2, v_v_540_);
lean_ctor_set(v_reuseFailAlloc_549_, 3, v___x_546_);
lean_ctor_set(v_reuseFailAlloc_549_, 4, v_r_538_);
v___x_548_ = v_reuseFailAlloc_549_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
return v___x_548_;
}
}
}
}
else
{
lean_object* v___x_555_; lean_object* v___x_557_; 
v___x_555_ = lean_unsigned_to_nat(2u);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v_impl_424_);
lean_ctor_set(v___x_419_, 3, v_r_538_);
lean_ctor_set(v___x_419_, 0, v___x_555_);
v___x_557_ = v___x_419_;
goto v_reusejp_556_;
}
else
{
lean_object* v_reuseFailAlloc_558_; 
v_reuseFailAlloc_558_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_558_, 0, v___x_555_);
lean_ctor_set(v_reuseFailAlloc_558_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_558_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_558_, 3, v_r_538_);
lean_ctor_set(v_reuseFailAlloc_558_, 4, v_impl_424_);
v___x_557_ = v_reuseFailAlloc_558_;
goto v_reusejp_556_;
}
v_reusejp_556_:
{
return v___x_557_;
}
}
}
}
}
else
{
lean_object* v___x_560_; 
lean_dec(v_v_415_);
lean_dec(v_k_414_);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 2, v_v_411_);
lean_ctor_set(v___x_419_, 1, v_k_410_);
v___x_560_ = v___x_419_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v_size_413_);
lean_ctor_set(v_reuseFailAlloc_561_, 1, v_k_410_);
lean_ctor_set(v_reuseFailAlloc_561_, 2, v_v_411_);
lean_ctor_set(v_reuseFailAlloc_561_, 3, v_l_416_);
lean_ctor_set(v_reuseFailAlloc_561_, 4, v_r_417_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
}
else
{
lean_object* v_impl_562_; lean_object* v___x_563_; 
lean_dec(v_size_413_);
v_impl_562_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v_k_410_, v_v_411_, v_l_416_);
v___x_563_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_417_) == 0)
{
lean_object* v_size_564_; lean_object* v_size_565_; lean_object* v_k_566_; lean_object* v_v_567_; lean_object* v_l_568_; lean_object* v_r_569_; lean_object* v___x_570_; lean_object* v___x_571_; uint8_t v___x_572_; 
v_size_564_ = lean_ctor_get(v_r_417_, 0);
v_size_565_ = lean_ctor_get(v_impl_562_, 0);
lean_inc(v_size_565_);
v_k_566_ = lean_ctor_get(v_impl_562_, 1);
lean_inc(v_k_566_);
v_v_567_ = lean_ctor_get(v_impl_562_, 2);
lean_inc(v_v_567_);
v_l_568_ = lean_ctor_get(v_impl_562_, 3);
lean_inc(v_l_568_);
v_r_569_ = lean_ctor_get(v_impl_562_, 4);
lean_inc(v_r_569_);
v___x_570_ = lean_unsigned_to_nat(3u);
v___x_571_ = lean_nat_mul(v___x_570_, v_size_564_);
v___x_572_ = lean_nat_dec_lt(v___x_571_, v_size_565_);
lean_dec(v___x_571_);
if (v___x_572_ == 0)
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_576_; 
lean_dec(v_r_569_);
lean_dec(v_l_568_);
lean_dec(v_v_567_);
lean_dec(v_k_566_);
v___x_573_ = lean_nat_add(v___x_563_, v_size_565_);
lean_dec(v_size_565_);
v___x_574_ = lean_nat_add(v___x_573_, v_size_564_);
lean_dec(v___x_573_);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 3, v_impl_562_);
lean_ctor_set(v___x_419_, 0, v___x_574_);
v___x_576_ = v___x_419_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_577_; 
v_reuseFailAlloc_577_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_577_, 0, v___x_574_);
lean_ctor_set(v_reuseFailAlloc_577_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_577_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_577_, 3, v_impl_562_);
lean_ctor_set(v_reuseFailAlloc_577_, 4, v_r_417_);
v___x_576_ = v_reuseFailAlloc_577_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
return v___x_576_;
}
}
else
{
lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_643_; 
v_isSharedCheck_643_ = !lean_is_exclusive(v_impl_562_);
if (v_isSharedCheck_643_ == 0)
{
lean_object* v_unused_644_; lean_object* v_unused_645_; lean_object* v_unused_646_; lean_object* v_unused_647_; lean_object* v_unused_648_; 
v_unused_644_ = lean_ctor_get(v_impl_562_, 4);
lean_dec(v_unused_644_);
v_unused_645_ = lean_ctor_get(v_impl_562_, 3);
lean_dec(v_unused_645_);
v_unused_646_ = lean_ctor_get(v_impl_562_, 2);
lean_dec(v_unused_646_);
v_unused_647_ = lean_ctor_get(v_impl_562_, 1);
lean_dec(v_unused_647_);
v_unused_648_ = lean_ctor_get(v_impl_562_, 0);
lean_dec(v_unused_648_);
v___x_579_ = v_impl_562_;
v_isShared_580_ = v_isSharedCheck_643_;
goto v_resetjp_578_;
}
else
{
lean_dec(v_impl_562_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_643_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v_size_581_; lean_object* v_size_582_; lean_object* v_k_583_; lean_object* v_v_584_; lean_object* v_l_585_; lean_object* v_r_586_; lean_object* v___x_587_; lean_object* v___x_588_; uint8_t v___x_589_; 
v_size_581_ = lean_ctor_get(v_l_568_, 0);
v_size_582_ = lean_ctor_get(v_r_569_, 0);
v_k_583_ = lean_ctor_get(v_r_569_, 1);
v_v_584_ = lean_ctor_get(v_r_569_, 2);
v_l_585_ = lean_ctor_get(v_r_569_, 3);
v_r_586_ = lean_ctor_get(v_r_569_, 4);
v___x_587_ = lean_unsigned_to_nat(2u);
v___x_588_ = lean_nat_mul(v___x_587_, v_size_581_);
v___x_589_ = lean_nat_dec_lt(v_size_582_, v___x_588_);
lean_dec(v___x_588_);
if (v___x_589_ == 0)
{
lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_618_; 
lean_inc(v_r_586_);
lean_inc(v_l_585_);
lean_inc(v_v_584_);
lean_inc(v_k_583_);
v_isSharedCheck_618_ = !lean_is_exclusive(v_r_569_);
if (v_isSharedCheck_618_ == 0)
{
lean_object* v_unused_619_; lean_object* v_unused_620_; lean_object* v_unused_621_; lean_object* v_unused_622_; lean_object* v_unused_623_; 
v_unused_619_ = lean_ctor_get(v_r_569_, 4);
lean_dec(v_unused_619_);
v_unused_620_ = lean_ctor_get(v_r_569_, 3);
lean_dec(v_unused_620_);
v_unused_621_ = lean_ctor_get(v_r_569_, 2);
lean_dec(v_unused_621_);
v_unused_622_ = lean_ctor_get(v_r_569_, 1);
lean_dec(v_unused_622_);
v_unused_623_ = lean_ctor_get(v_r_569_, 0);
lean_dec(v_unused_623_);
v___x_591_ = v_r_569_;
v_isShared_592_ = v_isSharedCheck_618_;
goto v_resetjp_590_;
}
else
{
lean_dec(v_r_569_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_618_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___y_596_; lean_object* v___y_597_; lean_object* v___y_598_; lean_object* v___x_606_; lean_object* v___y_608_; 
v___x_593_ = lean_nat_add(v___x_563_, v_size_565_);
lean_dec(v_size_565_);
v___x_594_ = lean_nat_add(v___x_593_, v_size_564_);
lean_dec(v___x_593_);
v___x_606_ = lean_nat_add(v___x_563_, v_size_581_);
if (lean_obj_tag(v_l_585_) == 0)
{
lean_object* v_size_616_; 
v_size_616_ = lean_ctor_get(v_l_585_, 0);
lean_inc(v_size_616_);
v___y_608_ = v_size_616_;
goto v___jp_607_;
}
else
{
lean_object* v___x_617_; 
v___x_617_ = lean_unsigned_to_nat(0u);
v___y_608_ = v___x_617_;
goto v___jp_607_;
}
v___jp_595_:
{
lean_object* v___x_599_; lean_object* v___x_601_; 
v___x_599_ = lean_nat_add(v___y_597_, v___y_598_);
lean_dec(v___y_598_);
lean_dec(v___y_597_);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 4, v_r_417_);
lean_ctor_set(v___x_591_, 3, v_r_586_);
lean_ctor_set(v___x_591_, 2, v_v_415_);
lean_ctor_set(v___x_591_, 1, v_k_414_);
lean_ctor_set(v___x_591_, 0, v___x_599_);
v___x_601_ = v___x_591_;
goto v_reusejp_600_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v___x_599_);
lean_ctor_set(v_reuseFailAlloc_605_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_605_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_605_, 3, v_r_586_);
lean_ctor_set(v_reuseFailAlloc_605_, 4, v_r_417_);
v___x_601_ = v_reuseFailAlloc_605_;
goto v_reusejp_600_;
}
v_reusejp_600_:
{
lean_object* v___x_603_; 
if (v_isShared_580_ == 0)
{
lean_ctor_set(v___x_579_, 4, v___x_601_);
lean_ctor_set(v___x_579_, 3, v___y_596_);
lean_ctor_set(v___x_579_, 2, v_v_584_);
lean_ctor_set(v___x_579_, 1, v_k_583_);
lean_ctor_set(v___x_579_, 0, v___x_594_);
v___x_603_ = v___x_579_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v___x_594_);
lean_ctor_set(v_reuseFailAlloc_604_, 1, v_k_583_);
lean_ctor_set(v_reuseFailAlloc_604_, 2, v_v_584_);
lean_ctor_set(v_reuseFailAlloc_604_, 3, v___y_596_);
lean_ctor_set(v_reuseFailAlloc_604_, 4, v___x_601_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
v___jp_607_:
{
lean_object* v___x_609_; lean_object* v___x_611_; 
v___x_609_ = lean_nat_add(v___x_606_, v___y_608_);
lean_dec(v___y_608_);
lean_dec(v___x_606_);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v_l_585_);
lean_ctor_set(v___x_419_, 3, v_l_568_);
lean_ctor_set(v___x_419_, 2, v_v_567_);
lean_ctor_set(v___x_419_, 1, v_k_566_);
lean_ctor_set(v___x_419_, 0, v___x_609_);
v___x_611_ = v___x_419_;
goto v_reusejp_610_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v___x_609_);
lean_ctor_set(v_reuseFailAlloc_615_, 1, v_k_566_);
lean_ctor_set(v_reuseFailAlloc_615_, 2, v_v_567_);
lean_ctor_set(v_reuseFailAlloc_615_, 3, v_l_568_);
lean_ctor_set(v_reuseFailAlloc_615_, 4, v_l_585_);
v___x_611_ = v_reuseFailAlloc_615_;
goto v_reusejp_610_;
}
v_reusejp_610_:
{
lean_object* v___x_612_; 
v___x_612_ = lean_nat_add(v___x_563_, v_size_564_);
if (lean_obj_tag(v_r_586_) == 0)
{
lean_object* v_size_613_; 
v_size_613_ = lean_ctor_get(v_r_586_, 0);
lean_inc(v_size_613_);
v___y_596_ = v___x_611_;
v___y_597_ = v___x_612_;
v___y_598_ = v_size_613_;
goto v___jp_595_;
}
else
{
lean_object* v___x_614_; 
v___x_614_ = lean_unsigned_to_nat(0u);
v___y_596_ = v___x_611_;
v___y_597_ = v___x_612_;
v___y_598_ = v___x_614_;
goto v___jp_595_;
}
}
}
}
}
else
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_629_; 
lean_del_object(v___x_419_);
v___x_624_ = lean_nat_add(v___x_563_, v_size_565_);
lean_dec(v_size_565_);
v___x_625_ = lean_nat_add(v___x_624_, v_size_564_);
lean_dec(v___x_624_);
v___x_626_ = lean_nat_add(v___x_563_, v_size_564_);
v___x_627_ = lean_nat_add(v___x_626_, v_size_582_);
lean_dec(v___x_626_);
lean_inc_ref(v_r_417_);
if (v_isShared_580_ == 0)
{
lean_ctor_set(v___x_579_, 4, v_r_417_);
lean_ctor_set(v___x_579_, 3, v_r_569_);
lean_ctor_set(v___x_579_, 2, v_v_415_);
lean_ctor_set(v___x_579_, 1, v_k_414_);
lean_ctor_set(v___x_579_, 0, v___x_627_);
v___x_629_ = v___x_579_;
goto v_reusejp_628_;
}
else
{
lean_object* v_reuseFailAlloc_642_; 
v_reuseFailAlloc_642_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_642_, 0, v___x_627_);
lean_ctor_set(v_reuseFailAlloc_642_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_642_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_642_, 3, v_r_569_);
lean_ctor_set(v_reuseFailAlloc_642_, 4, v_r_417_);
v___x_629_ = v_reuseFailAlloc_642_;
goto v_reusejp_628_;
}
v_reusejp_628_:
{
lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
v_isSharedCheck_636_ = !lean_is_exclusive(v_r_417_);
if (v_isSharedCheck_636_ == 0)
{
lean_object* v_unused_637_; lean_object* v_unused_638_; lean_object* v_unused_639_; lean_object* v_unused_640_; lean_object* v_unused_641_; 
v_unused_637_ = lean_ctor_get(v_r_417_, 4);
lean_dec(v_unused_637_);
v_unused_638_ = lean_ctor_get(v_r_417_, 3);
lean_dec(v_unused_638_);
v_unused_639_ = lean_ctor_get(v_r_417_, 2);
lean_dec(v_unused_639_);
v_unused_640_ = lean_ctor_get(v_r_417_, 1);
lean_dec(v_unused_640_);
v_unused_641_ = lean_ctor_get(v_r_417_, 0);
lean_dec(v_unused_641_);
v___x_631_ = v_r_417_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_dec(v_r_417_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
lean_ctor_set(v___x_631_, 4, v___x_629_);
lean_ctor_set(v___x_631_, 3, v_l_568_);
lean_ctor_set(v___x_631_, 2, v_v_567_);
lean_ctor_set(v___x_631_, 1, v_k_566_);
lean_ctor_set(v___x_631_, 0, v___x_625_);
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v___x_625_);
lean_ctor_set(v_reuseFailAlloc_635_, 1, v_k_566_);
lean_ctor_set(v_reuseFailAlloc_635_, 2, v_v_567_);
lean_ctor_set(v_reuseFailAlloc_635_, 3, v_l_568_);
lean_ctor_set(v_reuseFailAlloc_635_, 4, v___x_629_);
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
}
}
}
else
{
lean_object* v_l_649_; 
v_l_649_ = lean_ctor_get(v_impl_562_, 3);
lean_inc(v_l_649_);
if (lean_obj_tag(v_l_649_) == 0)
{
lean_object* v_r_650_; lean_object* v_k_651_; lean_object* v_v_652_; lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_663_; 
v_r_650_ = lean_ctor_get(v_impl_562_, 4);
v_k_651_ = lean_ctor_get(v_impl_562_, 1);
v_v_652_ = lean_ctor_get(v_impl_562_, 2);
v_isSharedCheck_663_ = !lean_is_exclusive(v_impl_562_);
if (v_isSharedCheck_663_ == 0)
{
lean_object* v_unused_664_; lean_object* v_unused_665_; 
v_unused_664_ = lean_ctor_get(v_impl_562_, 3);
lean_dec(v_unused_664_);
v_unused_665_ = lean_ctor_get(v_impl_562_, 0);
lean_dec(v_unused_665_);
v___x_654_ = v_impl_562_;
v_isShared_655_ = v_isSharedCheck_663_;
goto v_resetjp_653_;
}
else
{
lean_inc(v_r_650_);
lean_inc(v_v_652_);
lean_inc(v_k_651_);
lean_dec(v_impl_562_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_663_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
lean_object* v___x_656_; lean_object* v___x_658_; 
v___x_656_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_650_);
if (v_isShared_655_ == 0)
{
lean_ctor_set(v___x_654_, 3, v_r_650_);
lean_ctor_set(v___x_654_, 2, v_v_415_);
lean_ctor_set(v___x_654_, 1, v_k_414_);
lean_ctor_set(v___x_654_, 0, v___x_563_);
v___x_658_ = v___x_654_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v___x_563_);
lean_ctor_set(v_reuseFailAlloc_662_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_662_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_662_, 3, v_r_650_);
lean_ctor_set(v_reuseFailAlloc_662_, 4, v_r_650_);
v___x_658_ = v_reuseFailAlloc_662_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
lean_object* v___x_660_; 
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v___x_658_);
lean_ctor_set(v___x_419_, 3, v_l_649_);
lean_ctor_set(v___x_419_, 2, v_v_652_);
lean_ctor_set(v___x_419_, 1, v_k_651_);
lean_ctor_set(v___x_419_, 0, v___x_656_);
v___x_660_ = v___x_419_;
goto v_reusejp_659_;
}
else
{
lean_object* v_reuseFailAlloc_661_; 
v_reuseFailAlloc_661_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_661_, 0, v___x_656_);
lean_ctor_set(v_reuseFailAlloc_661_, 1, v_k_651_);
lean_ctor_set(v_reuseFailAlloc_661_, 2, v_v_652_);
lean_ctor_set(v_reuseFailAlloc_661_, 3, v_l_649_);
lean_ctor_set(v_reuseFailAlloc_661_, 4, v___x_658_);
v___x_660_ = v_reuseFailAlloc_661_;
goto v_reusejp_659_;
}
v_reusejp_659_:
{
return v___x_660_;
}
}
}
}
else
{
lean_object* v_r_666_; 
v_r_666_ = lean_ctor_get(v_impl_562_, 4);
lean_inc(v_r_666_);
if (lean_obj_tag(v_r_666_) == 0)
{
lean_object* v_k_667_; lean_object* v_v_668_; lean_object* v___x_670_; uint8_t v_isShared_671_; uint8_t v_isSharedCheck_691_; 
v_k_667_ = lean_ctor_get(v_impl_562_, 1);
v_v_668_ = lean_ctor_get(v_impl_562_, 2);
v_isSharedCheck_691_ = !lean_is_exclusive(v_impl_562_);
if (v_isSharedCheck_691_ == 0)
{
lean_object* v_unused_692_; lean_object* v_unused_693_; lean_object* v_unused_694_; 
v_unused_692_ = lean_ctor_get(v_impl_562_, 4);
lean_dec(v_unused_692_);
v_unused_693_ = lean_ctor_get(v_impl_562_, 3);
lean_dec(v_unused_693_);
v_unused_694_ = lean_ctor_get(v_impl_562_, 0);
lean_dec(v_unused_694_);
v___x_670_ = v_impl_562_;
v_isShared_671_ = v_isSharedCheck_691_;
goto v_resetjp_669_;
}
else
{
lean_inc(v_v_668_);
lean_inc(v_k_667_);
lean_dec(v_impl_562_);
v___x_670_ = lean_box(0);
v_isShared_671_ = v_isSharedCheck_691_;
goto v_resetjp_669_;
}
v_resetjp_669_:
{
lean_object* v_k_672_; lean_object* v_v_673_; lean_object* v___x_675_; uint8_t v_isShared_676_; uint8_t v_isSharedCheck_687_; 
v_k_672_ = lean_ctor_get(v_r_666_, 1);
v_v_673_ = lean_ctor_get(v_r_666_, 2);
v_isSharedCheck_687_ = !lean_is_exclusive(v_r_666_);
if (v_isSharedCheck_687_ == 0)
{
lean_object* v_unused_688_; lean_object* v_unused_689_; lean_object* v_unused_690_; 
v_unused_688_ = lean_ctor_get(v_r_666_, 4);
lean_dec(v_unused_688_);
v_unused_689_ = lean_ctor_get(v_r_666_, 3);
lean_dec(v_unused_689_);
v_unused_690_ = lean_ctor_get(v_r_666_, 0);
lean_dec(v_unused_690_);
v___x_675_ = v_r_666_;
v_isShared_676_ = v_isSharedCheck_687_;
goto v_resetjp_674_;
}
else
{
lean_inc(v_v_673_);
lean_inc(v_k_672_);
lean_dec(v_r_666_);
v___x_675_ = lean_box(0);
v_isShared_676_ = v_isSharedCheck_687_;
goto v_resetjp_674_;
}
v_resetjp_674_:
{
lean_object* v___x_677_; lean_object* v___x_679_; 
v___x_677_ = lean_unsigned_to_nat(3u);
if (v_isShared_676_ == 0)
{
lean_ctor_set(v___x_675_, 4, v_l_649_);
lean_ctor_set(v___x_675_, 3, v_l_649_);
lean_ctor_set(v___x_675_, 2, v_v_668_);
lean_ctor_set(v___x_675_, 1, v_k_667_);
lean_ctor_set(v___x_675_, 0, v___x_563_);
v___x_679_ = v___x_675_;
goto v_reusejp_678_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v___x_563_);
lean_ctor_set(v_reuseFailAlloc_686_, 1, v_k_667_);
lean_ctor_set(v_reuseFailAlloc_686_, 2, v_v_668_);
lean_ctor_set(v_reuseFailAlloc_686_, 3, v_l_649_);
lean_ctor_set(v_reuseFailAlloc_686_, 4, v_l_649_);
v___x_679_ = v_reuseFailAlloc_686_;
goto v_reusejp_678_;
}
v_reusejp_678_:
{
lean_object* v___x_681_; 
if (v_isShared_671_ == 0)
{
lean_ctor_set(v___x_670_, 4, v_l_649_);
lean_ctor_set(v___x_670_, 2, v_v_415_);
lean_ctor_set(v___x_670_, 1, v_k_414_);
lean_ctor_set(v___x_670_, 0, v___x_563_);
v___x_681_ = v___x_670_;
goto v_reusejp_680_;
}
else
{
lean_object* v_reuseFailAlloc_685_; 
v_reuseFailAlloc_685_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_685_, 0, v___x_563_);
lean_ctor_set(v_reuseFailAlloc_685_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_685_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_685_, 3, v_l_649_);
lean_ctor_set(v_reuseFailAlloc_685_, 4, v_l_649_);
v___x_681_ = v_reuseFailAlloc_685_;
goto v_reusejp_680_;
}
v_reusejp_680_:
{
lean_object* v___x_683_; 
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v___x_681_);
lean_ctor_set(v___x_419_, 3, v___x_679_);
lean_ctor_set(v___x_419_, 2, v_v_673_);
lean_ctor_set(v___x_419_, 1, v_k_672_);
lean_ctor_set(v___x_419_, 0, v___x_677_);
v___x_683_ = v___x_419_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_684_; 
v_reuseFailAlloc_684_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_684_, 0, v___x_677_);
lean_ctor_set(v_reuseFailAlloc_684_, 1, v_k_672_);
lean_ctor_set(v_reuseFailAlloc_684_, 2, v_v_673_);
lean_ctor_set(v_reuseFailAlloc_684_, 3, v___x_679_);
lean_ctor_set(v_reuseFailAlloc_684_, 4, v___x_681_);
v___x_683_ = v_reuseFailAlloc_684_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
return v___x_683_;
}
}
}
}
}
}
else
{
lean_object* v___x_695_; lean_object* v___x_697_; 
v___x_695_ = lean_unsigned_to_nat(2u);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 4, v_r_666_);
lean_ctor_set(v___x_419_, 3, v_impl_562_);
lean_ctor_set(v___x_419_, 0, v___x_695_);
v___x_697_ = v___x_419_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_698_; 
v_reuseFailAlloc_698_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_698_, 0, v___x_695_);
lean_ctor_set(v_reuseFailAlloc_698_, 1, v_k_414_);
lean_ctor_set(v_reuseFailAlloc_698_, 2, v_v_415_);
lean_ctor_set(v_reuseFailAlloc_698_, 3, v_impl_562_);
lean_ctor_set(v_reuseFailAlloc_698_, 4, v_r_666_);
v___x_697_ = v_reuseFailAlloc_698_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
return v___x_697_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_700_; lean_object* v___x_701_; 
v___x_700_ = lean_unsigned_to_nat(1u);
v___x_701_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_701_, 0, v___x_700_);
lean_ctor_set(v___x_701_, 1, v_k_410_);
lean_ctor_set(v___x_701_, 2, v_v_411_);
lean_ctor_set(v___x_701_, 3, v_t_412_);
lean_ctor_set(v___x_701_, 4, v_t_412_);
return v___x_701_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0(void){
_start:
{
lean_object* v___x_702_; lean_object* v___x_703_; 
v___x_702_ = lean_unsigned_to_nat(1u);
v___x_703_ = lean_nat_to_int(v___x_702_);
return v___x_703_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__1(void){
_start:
{
lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; 
v___x_704_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0);
v___x_705_ = lean_box(1);
v___x_706_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v___x_705_, v___x_704_, v___x_705_);
return v___x_706_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one(void){
_start:
{
lean_object* v___x_707_; 
v___x_707_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__1, &lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__1);
return v___x_707_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0___redArg(lean_object* v_cmp_708_, lean_object* v_t_u2081_709_, lean_object* v_t_u2082_710_){
_start:
{
uint8_t v___x_711_; 
v___x_711_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v_cmp_708_, v_t_u2081_709_, v_t_u2082_710_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0___redArg___boxed(lean_object* v_cmp_712_, lean_object* v_t_u2081_713_, lean_object* v_t_u2082_714_){
_start:
{
uint8_t v_res_715_; lean_object* v_r_716_; 
v_res_715_ = lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0___redArg(v_cmp_712_, v_t_u2081_713_, v_t_u2082_714_);
v_r_716_ = lean_box(v_res_715_);
return v_r_716_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0(lean_object* v_00_u03b1_717_, lean_object* v_cmp_718_, lean_object* v_t_u2081_719_, lean_object* v_t_u2082_720_){
_start:
{
uint8_t v___x_721_; 
v___x_721_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v_cmp_718_, v_t_u2081_719_, v_t_u2082_720_);
return v___x_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0___boxed(lean_object* v_00_u03b1_722_, lean_object* v_cmp_723_, lean_object* v_t_u2081_724_, lean_object* v_t_u2082_725_){
_start:
{
uint8_t v_res_726_; lean_object* v_r_727_; 
v_res_726_ = lp_mathlib_Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0(v_00_u03b1_722_, v_cmp_723_, v_t_u2081_724_, v_t_u2082_725_);
v_r_727_ = lean_box(v_res_726_);
return v_r_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1(lean_object* v_00_u03b2_728_, lean_object* v_k_729_, lean_object* v_v_730_, lean_object* v_t_731_, lean_object* v_hl_732_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v_k_729_, v_v_730_, v_t_731_);
return v___x_733_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0___redArg(lean_object* v_cmp_734_, lean_object* v_t_u2081_735_, lean_object* v_t_u2082_736_){
_start:
{
uint8_t v___x_737_; 
v___x_737_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v_cmp_734_, v_t_u2081_735_, v_t_u2082_736_);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0___redArg___boxed(lean_object* v_cmp_738_, lean_object* v_t_u2081_739_, lean_object* v_t_u2082_740_){
_start:
{
uint8_t v_res_741_; lean_object* v_r_742_; 
v_res_741_ = lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0___redArg(v_cmp_738_, v_t_u2081_739_, v_t_u2082_740_);
v_r_742_ = lean_box(v_res_741_);
return v_r_742_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0(lean_object* v_00_u03b1_743_, lean_object* v_cmp_744_, lean_object* v_t_u2081_745_, lean_object* v_t_u2082_746_){
_start:
{
uint8_t v___x_747_; 
v___x_747_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v_cmp_744_, v_t_u2081_745_, v_t_u2082_746_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0___boxed(lean_object* v_00_u03b1_748_, lean_object* v_cmp_749_, lean_object* v_t_u2081_750_, lean_object* v_t_u2082_751_){
_start:
{
uint8_t v_res_752_; lean_object* v_r_753_; 
v_res_752_ = lp_mathlib_Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0(v_00_u03b1_748_, v_cmp_749_, v_t_u2081_750_, v_t_u2082_751_);
v_r_753_ = lean_box(v_res_752_);
return v_r_753_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_754_, lean_object* v_cmp_755_, lean_object* v_t_u2081_756_, lean_object* v_t_u2082_757_){
_start:
{
uint8_t v___x_758_; 
v___x_758_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v_cmp_755_, v_t_u2081_756_, v_t_u2082_757_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_759_, lean_object* v_cmp_760_, lean_object* v_t_u2081_761_, lean_object* v_t_u2082_762_){
_start:
{
uint8_t v_res_763_; lean_object* v_r_764_; 
v_res_763_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1(v_00_u03b1_759_, v_cmp_760_, v_t_u2081_761_, v_t_u2082_762_);
v_r_764_ = lean_box(v_res_763_);
return v_r_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b1_765_, lean_object* v_cmp_766_, lean_object* v_00_u03b4_767_, lean_object* v_t_768_, lean_object* v_k_769_){
_start:
{
lean_object* v___x_770_; 
v___x_770_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__3___redArg(v_cmp_766_, v_t_768_, v_k_769_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5(lean_object* v_00_u03b1_771_, lean_object* v_cmp_772_, lean_object* v_t_u2082_773_, lean_object* v_init_774_, lean_object* v_x_775_){
_start:
{
lean_object* v___x_776_; 
v___x_776_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1_spec__5___redArg(v_cmp_772_, v_t_u2082_773_, v_init_774_, v_x_775_);
return v___x_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg___lam__0(lean_object* v_b_u2082_777_, lean_object* v_x_778_){
_start:
{
if (lean_obj_tag(v_x_778_) == 0)
{
lean_object* v___x_779_; 
v___x_779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_779_, 0, v_b_u2082_777_);
return v___x_779_;
}
else
{
lean_object* v_val_780_; lean_object* v___x_782_; uint8_t v_isShared_783_; uint8_t v_isSharedCheck_788_; 
v_val_780_ = lean_ctor_get(v_x_778_, 0);
v_isSharedCheck_788_ = !lean_is_exclusive(v_x_778_);
if (v_isSharedCheck_788_ == 0)
{
v___x_782_ = v_x_778_;
v_isShared_783_ = v_isSharedCheck_788_;
goto v_resetjp_781_;
}
else
{
lean_inc(v_val_780_);
lean_dec(v_x_778_);
v___x_782_ = lean_box(0);
v_isShared_783_ = v_isSharedCheck_788_;
goto v_resetjp_781_;
}
v_resetjp_781_:
{
lean_object* v___x_784_; lean_object* v___x_786_; 
v___x_784_ = lean_nat_add(v_val_780_, v_b_u2082_777_);
lean_dec(v_b_u2082_777_);
lean_dec(v_val_780_);
if (v_isShared_783_ == 0)
{
lean_ctor_set(v___x_782_, 0, v___x_784_);
v___x_786_ = v___x_782_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_787_; 
v_reuseFailAlloc_787_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_787_, 0, v___x_784_);
v___x_786_ = v_reuseFailAlloc_787_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
return v___x_786_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg(lean_object* v_b_u2082_789_, lean_object* v_k_790_, lean_object* v_t_791_){
_start:
{
if (lean_obj_tag(v_t_791_) == 0)
{
lean_object* v_size_792_; lean_object* v_k_793_; lean_object* v_v_794_; lean_object* v_l_795_; lean_object* v_r_796_; lean_object* v___x_798_; uint8_t v_isShared_799_; uint8_t v_isSharedCheck_812_; 
v_size_792_ = lean_ctor_get(v_t_791_, 0);
v_k_793_ = lean_ctor_get(v_t_791_, 1);
v_v_794_ = lean_ctor_get(v_t_791_, 2);
v_l_795_ = lean_ctor_get(v_t_791_, 3);
v_r_796_ = lean_ctor_get(v_t_791_, 4);
v_isSharedCheck_812_ = !lean_is_exclusive(v_t_791_);
if (v_isSharedCheck_812_ == 0)
{
v___x_798_ = v_t_791_;
v_isShared_799_ = v_isSharedCheck_812_;
goto v_resetjp_797_;
}
else
{
lean_inc(v_r_796_);
lean_inc(v_l_795_);
lean_inc(v_v_794_);
lean_inc(v_k_793_);
lean_inc(v_size_792_);
lean_dec(v_t_791_);
v___x_798_ = lean_box(0);
v_isShared_799_ = v_isSharedCheck_812_;
goto v_resetjp_797_;
}
v_resetjp_797_:
{
uint8_t v___x_800_; 
v___x_800_ = lean_nat_dec_lt(v_k_790_, v_k_793_);
if (v___x_800_ == 0)
{
uint8_t v___x_801_; 
v___x_801_ = lean_nat_dec_eq(v_k_790_, v_k_793_);
if (v___x_801_ == 0)
{
lean_object* v_impl_802_; lean_object* v___x_803_; 
lean_del_object(v___x_798_);
lean_dec(v_size_792_);
v_impl_802_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg(v_b_u2082_789_, v_k_790_, v_r_796_);
v___x_803_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_793_, v_v_794_, v_l_795_, v_impl_802_);
return v___x_803_;
}
else
{
lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v_val_806_; lean_object* v___x_808_; 
lean_dec(v_k_793_);
v___x_804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_804_, 0, v_v_794_);
v___x_805_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg___lam__0(v_b_u2082_789_, v___x_804_);
v_val_806_ = lean_ctor_get(v___x_805_, 0);
lean_inc(v_val_806_);
lean_dec(v___x_805_);
if (v_isShared_799_ == 0)
{
lean_ctor_set(v___x_798_, 2, v_val_806_);
lean_ctor_set(v___x_798_, 1, v_k_790_);
v___x_808_ = v___x_798_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_size_792_);
lean_ctor_set(v_reuseFailAlloc_809_, 1, v_k_790_);
lean_ctor_set(v_reuseFailAlloc_809_, 2, v_val_806_);
lean_ctor_set(v_reuseFailAlloc_809_, 3, v_l_795_);
lean_ctor_set(v_reuseFailAlloc_809_, 4, v_r_796_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
return v___x_808_;
}
}
}
else
{
lean_object* v_impl_810_; lean_object* v___x_811_; 
lean_del_object(v___x_798_);
lean_dec(v_size_792_);
v_impl_810_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg(v_b_u2082_789_, v_k_790_, v_l_795_);
v___x_811_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_793_, v_v_794_, v_impl_810_, v_r_796_);
return v___x_811_;
}
}
}
else
{
lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v_val_815_; lean_object* v___x_816_; lean_object* v___x_817_; 
v___x_813_ = lean_box(0);
v___x_814_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg___lam__0(v_b_u2082_789_, v___x_813_);
v_val_815_ = lean_ctor_get(v___x_814_, 0);
lean_inc(v_val_815_);
lean_dec(v___x_814_);
v___x_816_ = lean_unsigned_to_nat(1u);
v___x_817_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_817_, 0, v___x_816_);
lean_ctor_set(v___x_817_, 1, v_k_790_);
lean_ctor_set(v___x_817_, 2, v_val_815_);
lean_ctor_set(v___x_817_, 3, v_t_791_);
lean_ctor_set(v___x_817_, 4, v_t_791_);
return v___x_817_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__1_spec__1(lean_object* v_init_818_, lean_object* v_x_819_){
_start:
{
if (lean_obj_tag(v_x_819_) == 0)
{
lean_object* v_k_820_; lean_object* v_v_821_; lean_object* v_l_822_; lean_object* v_r_823_; lean_object* v___x_824_; lean_object* v___x_825_; 
v_k_820_ = lean_ctor_get(v_x_819_, 1);
lean_inc(v_k_820_);
v_v_821_ = lean_ctor_get(v_x_819_, 2);
lean_inc(v_v_821_);
v_l_822_ = lean_ctor_get(v_x_819_, 3);
lean_inc(v_l_822_);
v_r_823_ = lean_ctor_get(v_x_819_, 4);
lean_inc(v_r_823_);
lean_dec_ref_known(v_x_819_, 5);
v___x_824_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__1_spec__1(v_init_818_, v_l_822_);
v___x_825_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg(v_v_821_, v_k_820_, v___x_824_);
v_init_818_ = v___x_825_;
v_x_819_ = v_r_823_;
goto _start;
}
else
{
return v_init_818_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(lean_object* v_t_827_){
_start:
{
if (lean_obj_tag(v_t_827_) == 0)
{
lean_object* v_k_828_; lean_object* v_v_829_; lean_object* v_l_830_; lean_object* v_r_831_; lean_object* v___x_832_; uint8_t v___x_833_; 
v_k_828_ = lean_ctor_get(v_t_827_, 1);
lean_inc(v_k_828_);
v_v_829_ = lean_ctor_get(v_t_827_, 2);
lean_inc(v_v_829_);
v_l_830_ = lean_ctor_get(v_t_827_, 3);
lean_inc(v_l_830_);
v_r_831_ = lean_ctor_get(v_t_827_, 4);
lean_inc(v_r_831_);
lean_dec_ref_known(v_t_827_, 5);
v___x_832_ = lean_unsigned_to_nat(0u);
v___x_833_ = lean_nat_dec_eq(v_v_829_, v___x_832_);
if (v___x_833_ == 0)
{
lean_object* v_impl_834_; lean_object* v_impl_835_; lean_object* v___x_836_; 
v_impl_834_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(v_l_830_);
v_impl_835_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(v_r_831_);
v___x_836_ = l_Std_DTreeMap_Internal_Impl_link___redArg(v_k_828_, v_v_829_, v_impl_834_, v_impl_835_);
return v___x_836_;
}
else
{
lean_object* v_impl_837_; lean_object* v_impl_838_; lean_object* v___x_839_; 
lean_dec(v_v_829_);
lean_dec(v_k_828_);
v_impl_837_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(v_l_830_);
v_impl_838_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(v_r_831_);
v___x_839_ = l_Std_DTreeMap_Internal_Impl_link2___redArg(v_impl_837_, v_impl_838_);
return v___x_839_;
}
}
else
{
return v_t_827_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__3(lean_object* v_m_840_, lean_object* v_init_841_, lean_object* v_x_842_){
_start:
{
if (lean_obj_tag(v_x_842_) == 0)
{
lean_object* v_k_843_; lean_object* v_v_844_; lean_object* v_l_845_; lean_object* v_r_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; 
v_k_843_ = lean_ctor_get(v_x_842_, 1);
lean_inc(v_k_843_);
v_v_844_ = lean_ctor_get(v_x_842_, 2);
lean_inc(v_v_844_);
v_l_845_ = lean_ctor_get(v_x_842_, 3);
lean_inc(v_l_845_);
v_r_846_ = lean_ctor_get(v_x_842_, 4);
lean_inc(v_r_846_);
lean_dec_ref_known(v_x_842_, 5);
lean_inc_n(v_m_840_, 2);
v___x_847_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__3(v_m_840_, v_init_841_, v_r_846_);
v___x_848_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__1_spec__1(v_m_840_, v_k_843_);
v___x_849_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(v___x_848_);
v___x_850_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v___x_849_, v_v_844_, v___x_847_);
v_init_841_ = v___x_850_;
v_x_842_ = v_l_845_;
goto _start;
}
else
{
lean_dec(v_m_840_);
return v_init_841_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_scaleByMonom(lean_object* v_s_852_, lean_object* v_m_853_){
_start:
{
lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_854_ = lean_box(1);
v___x_855_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__3(v_m_853_, v___x_854_, v_s_852_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0(lean_object* v_b_u2082_856_, lean_object* v_k_857_, lean_object* v_t_858_, lean_object* v_hl_859_){
_start:
{
lean_object* v___x_860_; 
v___x_860_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__0___redArg(v_b_u2082_856_, v_k_857_, v_t_858_);
return v___x_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__1(lean_object* v_init_861_, lean_object* v_t_862_){
_start:
{
lean_object* v___x_863_; 
v___x_863_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__1_spec__1(v_init_861_, v_t_862_);
return v___x_863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2(lean_object* v_t_864_, lean_object* v_hl_865_){
_start:
{
lean_object* v___x_866_; 
v___x_866_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_scaleByMonom_spec__2___redArg(v_t_864_);
return v___x_866_;
}
}
static lean_object* _init_lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_867_; lean_object* v___x_868_; 
v___x_867_ = lean_unsigned_to_nat(0u);
v___x_868_ = lean_nat_to_int(v___x_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(lean_object* v_t_869_){
_start:
{
if (lean_obj_tag(v_t_869_) == 0)
{
lean_object* v_k_870_; lean_object* v_v_871_; lean_object* v_l_872_; lean_object* v_r_873_; lean_object* v___x_874_; uint8_t v___x_875_; 
v_k_870_ = lean_ctor_get(v_t_869_, 1);
lean_inc(v_k_870_);
v_v_871_ = lean_ctor_get(v_t_869_, 2);
lean_inc(v_v_871_);
v_l_872_ = lean_ctor_get(v_t_869_, 3);
lean_inc(v_l_872_);
v_r_873_ = lean_ctor_get(v_t_869_, 4);
lean_inc(v_r_873_);
lean_dec_ref_known(v_t_869_, 5);
v___x_874_ = lean_obj_once(&lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg___closed__0, &lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg___closed__0_once, _init_lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg___closed__0);
v___x_875_ = lean_int_dec_eq(v_v_871_, v___x_874_);
if (v___x_875_ == 0)
{
lean_object* v_impl_876_; lean_object* v_impl_877_; lean_object* v___x_878_; 
v_impl_876_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v_l_872_);
v_impl_877_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v_r_873_);
v___x_878_ = l_Std_DTreeMap_Internal_Impl_link___redArg(v_k_870_, v_v_871_, v_impl_876_, v_impl_877_);
return v___x_878_;
}
else
{
lean_object* v_impl_879_; lean_object* v_impl_880_; lean_object* v___x_881_; 
lean_dec(v_v_871_);
lean_dec(v_k_870_);
v_impl_879_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v_l_872_);
v_impl_880_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v_r_873_);
v___x_881_ = l_Std_DTreeMap_Internal_Impl_link2___redArg(v_impl_879_, v_impl_880_);
return v___x_881_;
}
}
else
{
return v_t_869_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg___lam__0(lean_object* v_b_u2082_882_, lean_object* v_x_883_){
_start:
{
if (lean_obj_tag(v_x_883_) == 0)
{
lean_object* v___x_884_; 
v___x_884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_884_, 0, v_b_u2082_882_);
return v___x_884_;
}
else
{
lean_object* v_val_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_893_; 
v_val_885_ = lean_ctor_get(v_x_883_, 0);
v_isSharedCheck_893_ = !lean_is_exclusive(v_x_883_);
if (v_isSharedCheck_893_ == 0)
{
v___x_887_ = v_x_883_;
v_isShared_888_ = v_isSharedCheck_893_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_val_885_);
lean_dec(v_x_883_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_893_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_889_; lean_object* v___x_891_; 
v___x_889_ = lean_int_add(v_val_885_, v_b_u2082_882_);
lean_dec(v_b_u2082_882_);
lean_dec(v_val_885_);
if (v_isShared_888_ == 0)
{
lean_ctor_set(v___x_887_, 0, v___x_889_);
v___x_891_ = v___x_887_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v___x_889_);
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
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg(lean_object* v_b_u2082_894_, lean_object* v_k_895_, lean_object* v_t_896_){
_start:
{
if (lean_obj_tag(v_t_896_) == 0)
{
lean_object* v_size_897_; lean_object* v_k_898_; lean_object* v_v_899_; lean_object* v_l_900_; lean_object* v_r_901_; lean_object* v___x_903_; uint8_t v_isShared_904_; uint8_t v_isSharedCheck_918_; 
v_size_897_ = lean_ctor_get(v_t_896_, 0);
v_k_898_ = lean_ctor_get(v_t_896_, 1);
v_v_899_ = lean_ctor_get(v_t_896_, 2);
v_l_900_ = lean_ctor_get(v_t_896_, 3);
v_r_901_ = lean_ctor_get(v_t_896_, 4);
v_isSharedCheck_918_ = !lean_is_exclusive(v_t_896_);
if (v_isSharedCheck_918_ == 0)
{
v___x_903_ = v_t_896_;
v_isShared_904_ = v_isSharedCheck_918_;
goto v_resetjp_902_;
}
else
{
lean_inc(v_r_901_);
lean_inc(v_l_900_);
lean_inc(v_v_899_);
lean_inc(v_k_898_);
lean_inc(v_size_897_);
lean_dec(v_t_896_);
v___x_903_ = lean_box(0);
v_isShared_904_ = v_isSharedCheck_918_;
goto v_resetjp_902_;
}
v_resetjp_902_:
{
uint8_t v___x_905_; 
v___x_905_ = lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt(v_k_895_, v_k_898_);
if (v___x_905_ == 0)
{
lean_object* v___f_906_; uint8_t v___x_907_; 
v___f_906_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__0));
lean_inc(v_k_898_);
lean_inc(v_k_895_);
v___x_907_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v___f_906_, v_k_895_, v_k_898_);
if (v___x_907_ == 0)
{
lean_object* v_impl_908_; lean_object* v___x_909_; 
lean_del_object(v___x_903_);
lean_dec(v_size_897_);
v_impl_908_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg(v_b_u2082_894_, v_k_895_, v_r_901_);
v___x_909_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_898_, v_v_899_, v_l_900_, v_impl_908_);
return v___x_909_;
}
else
{
lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v_val_912_; lean_object* v___x_914_; 
lean_dec(v_k_898_);
v___x_910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_910_, 0, v_v_899_);
v___x_911_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg___lam__0(v_b_u2082_894_, v___x_910_);
v_val_912_ = lean_ctor_get(v___x_911_, 0);
lean_inc(v_val_912_);
lean_dec(v___x_911_);
if (v_isShared_904_ == 0)
{
lean_ctor_set(v___x_903_, 2, v_val_912_);
lean_ctor_set(v___x_903_, 1, v_k_895_);
v___x_914_ = v___x_903_;
goto v_reusejp_913_;
}
else
{
lean_object* v_reuseFailAlloc_915_; 
v_reuseFailAlloc_915_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_915_, 0, v_size_897_);
lean_ctor_set(v_reuseFailAlloc_915_, 1, v_k_895_);
lean_ctor_set(v_reuseFailAlloc_915_, 2, v_val_912_);
lean_ctor_set(v_reuseFailAlloc_915_, 3, v_l_900_);
lean_ctor_set(v_reuseFailAlloc_915_, 4, v_r_901_);
v___x_914_ = v_reuseFailAlloc_915_;
goto v_reusejp_913_;
}
v_reusejp_913_:
{
return v___x_914_;
}
}
}
else
{
lean_object* v_impl_916_; lean_object* v___x_917_; 
lean_del_object(v___x_903_);
lean_dec(v_size_897_);
v_impl_916_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg(v_b_u2082_894_, v_k_895_, v_l_900_);
v___x_917_ = l_Std_DTreeMap_Internal_Impl_balance___redArg(v_k_898_, v_v_899_, v_impl_916_, v_r_901_);
return v___x_917_;
}
}
}
else
{
lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v_val_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
v___x_919_ = lean_box(0);
v___x_920_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg___lam__0(v_b_u2082_894_, v___x_919_);
v_val_921_ = lean_ctor_get(v___x_920_, 0);
lean_inc(v_val_921_);
lean_dec(v___x_920_);
v___x_922_ = lean_unsigned_to_nat(1u);
v___x_923_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_923_, 0, v___x_922_);
lean_ctor_set(v___x_923_, 1, v_k_895_);
lean_ctor_set(v___x_923_, 2, v_val_921_);
lean_ctor_set(v___x_923_, 3, v_t_896_);
lean_ctor_set(v___x_923_, 4, v_t_896_);
return v___x_923_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2_spec__2(lean_object* v_init_924_, lean_object* v_x_925_){
_start:
{
if (lean_obj_tag(v_x_925_) == 0)
{
lean_object* v_k_926_; lean_object* v_v_927_; lean_object* v_l_928_; lean_object* v_r_929_; lean_object* v___x_930_; lean_object* v___x_931_; 
v_k_926_ = lean_ctor_get(v_x_925_, 1);
lean_inc(v_k_926_);
v_v_927_ = lean_ctor_get(v_x_925_, 2);
lean_inc(v_v_927_);
v_l_928_ = lean_ctor_get(v_x_925_, 3);
lean_inc(v_l_928_);
v_r_929_ = lean_ctor_get(v_x_925_, 4);
lean_inc(v_r_929_);
lean_dec_ref_known(v_x_925_, 5);
v___x_930_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2_spec__2(v_init_924_, v_l_928_);
v___x_931_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg(v_v_927_, v_k_926_, v___x_930_);
v_init_924_ = v___x_931_;
v_x_925_ = v_r_929_;
goto _start;
}
else
{
return v_init_924_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0(lean_object* v_x2_933_, lean_object* v_t_934_){
_start:
{
if (lean_obj_tag(v_t_934_) == 0)
{
lean_object* v_size_935_; lean_object* v_k_936_; lean_object* v_v_937_; lean_object* v_l_938_; lean_object* v_r_939_; lean_object* v___x_941_; uint8_t v_isShared_942_; uint8_t v_isSharedCheck_949_; 
v_size_935_ = lean_ctor_get(v_t_934_, 0);
v_k_936_ = lean_ctor_get(v_t_934_, 1);
v_v_937_ = lean_ctor_get(v_t_934_, 2);
v_l_938_ = lean_ctor_get(v_t_934_, 3);
v_r_939_ = lean_ctor_get(v_t_934_, 4);
v_isSharedCheck_949_ = !lean_is_exclusive(v_t_934_);
if (v_isSharedCheck_949_ == 0)
{
v___x_941_ = v_t_934_;
v_isShared_942_ = v_isSharedCheck_949_;
goto v_resetjp_940_;
}
else
{
lean_inc(v_r_939_);
lean_inc(v_l_938_);
lean_inc(v_v_937_);
lean_inc(v_k_936_);
lean_inc(v_size_935_);
lean_dec(v_t_934_);
v___x_941_ = lean_box(0);
v_isShared_942_ = v_isSharedCheck_949_;
goto v_resetjp_940_;
}
v_resetjp_940_:
{
lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_947_; 
v___x_943_ = lean_int_mul(v_v_937_, v_x2_933_);
lean_dec(v_v_937_);
v___x_944_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0(v_x2_933_, v_l_938_);
v___x_945_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0(v_x2_933_, v_r_939_);
if (v_isShared_942_ == 0)
{
lean_ctor_set(v___x_941_, 4, v___x_945_);
lean_ctor_set(v___x_941_, 3, v___x_944_);
lean_ctor_set(v___x_941_, 2, v___x_943_);
v___x_947_ = v___x_941_;
goto v_reusejp_946_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v_size_935_);
lean_ctor_set(v_reuseFailAlloc_948_, 1, v_k_936_);
lean_ctor_set(v_reuseFailAlloc_948_, 2, v___x_943_);
lean_ctor_set(v_reuseFailAlloc_948_, 3, v___x_944_);
lean_ctor_set(v_reuseFailAlloc_948_, 4, v___x_945_);
v___x_947_ = v_reuseFailAlloc_948_;
goto v_reusejp_946_;
}
v_reusejp_946_:
{
return v___x_947_;
}
}
}
else
{
return v_t_934_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0___boxed(lean_object* v_x2_950_, lean_object* v_t_951_){
_start:
{
lean_object* v_res_952_; 
v_res_952_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0(v_x2_950_, v_t_951_);
lean_dec(v_x2_950_);
return v_res_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__4(lean_object* v_s2_953_, lean_object* v_init_954_, lean_object* v_x_955_){
_start:
{
if (lean_obj_tag(v_x_955_) == 0)
{
lean_object* v_k_956_; lean_object* v_v_957_; lean_object* v_l_958_; lean_object* v_r_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; 
v_k_956_ = lean_ctor_get(v_x_955_, 1);
lean_inc(v_k_956_);
v_v_957_ = lean_ctor_get(v_x_955_, 2);
lean_inc(v_v_957_);
v_l_958_ = lean_ctor_get(v_x_955_, 3);
lean_inc(v_l_958_);
v_r_959_ = lean_ctor_get(v_x_955_, 4);
lean_inc(v_r_959_);
lean_dec_ref_known(v_x_955_, 5);
lean_inc_n(v_s2_953_, 2);
v___x_960_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__4(v_s2_953_, v_init_954_, v_r_959_);
v___x_961_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_scaleByMonom(v_s2_953_, v_k_956_);
v___x_962_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__0(v_v_957_, v___x_961_);
lean_dec(v_v_957_);
v___x_963_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2_spec__2(v___x_960_, v___x_962_);
v___x_964_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v___x_963_);
v_init_954_ = v___x_964_;
v_x_955_ = v_l_958_;
goto _start;
}
else
{
lean_dec(v_s2_953_);
return v_init_954_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_mul(lean_object* v_s1_966_, lean_object* v_s2_967_){
_start:
{
lean_object* v___x_968_; lean_object* v___x_969_; 
v___x_968_ = lean_box(1);
v___x_969_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__4(v_s2_967_, v___x_968_, v_s1_966_);
return v___x_969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1(lean_object* v_b_u2082_970_, lean_object* v_k_971_, lean_object* v_t_972_, lean_object* v_hl_973_){
_start:
{
lean_object* v___x_974_; 
v___x_974_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_alter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__1___redArg(v_b_u2082_970_, v_k_971_, v_t_972_);
return v___x_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2(lean_object* v_init_975_, lean_object* v_t_976_){
_start:
{
lean_object* v___x_977_; 
v___x_977_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2_spec__2(v_init_975_, v_t_976_);
return v___x_977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3(lean_object* v_t_978_, lean_object* v_hl_979_){
_start:
{
lean_object* v___x_980_; 
v___x_980_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v_t_978_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_pow(lean_object* v_s_981_, lean_object* v_x_982_){
_start:
{
lean_object* v___x_983_; uint8_t v___x_984_; 
v___x_983_ = lean_unsigned_to_nat(0u);
v___x_984_ = lean_nat_dec_eq(v_x_982_, v___x_983_);
if (v___x_984_ == 0)
{
lean_object* v___x_985_; uint8_t v___x_986_; 
v___x_985_ = lean_unsigned_to_nat(1u);
v___x_986_ = lean_nat_dec_eq(v_x_982_, v___x_985_);
if (v___x_986_ == 0)
{
lean_object* v_m_987_; lean_object* v_a_988_; lean_object* v___x_989_; uint8_t v___x_990_; 
v_m_987_ = lean_nat_shiftr(v_x_982_, v___x_985_);
lean_inc(v_s_981_);
v_a_988_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_pow(v_s_981_, v_m_987_);
lean_dec(v_m_987_);
v___x_989_ = lean_nat_land(v_x_982_, v___x_985_);
v___x_990_ = lean_nat_dec_eq(v___x_989_, v___x_983_);
lean_dec(v___x_989_);
if (v___x_990_ == 0)
{
lean_object* v___x_991_; lean_object* v___x_992_; 
lean_inc(v_a_988_);
v___x_991_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_mul(v_a_988_, v_a_988_);
v___x_992_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_mul(v___x_991_, v_s_981_);
return v___x_992_;
}
else
{
lean_object* v___x_993_; 
lean_dec(v_s_981_);
lean_inc(v_a_988_);
v___x_993_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_mul(v_a_988_, v_a_988_);
return v___x_993_;
}
}
else
{
return v_s_981_;
}
}
else
{
lean_object* v___x_994_; 
lean_dec(v_s_981_);
v___x_994_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_one;
return v___x_994_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_Sum_pow___boxed(lean_object* v_s_995_, lean_object* v_x_996_){
_start:
{
lean_object* v_res_997_; 
v_res_997_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_pow(v_s_995_, v_x_996_);
lean_dec(v_x_996_);
return v_res_997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_SumOfMonom(lean_object* v_m_998_){
_start:
{
lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; 
v___x_999_ = lean_box(1);
v___x_1000_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0);
v___x_1001_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v_m_998_, v___x_1000_, v___x_999_);
return v___x_1001_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_one(void){
_start:
{
lean_object* v___x_1002_; 
v___x_1002_ = lean_box(1);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_scalar(lean_object* v_z_1003_){
_start:
{
lean_object* v___x_1004_; lean_object* v___x_1005_; 
v___x_1004_ = lean_box(1);
v___x_1005_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v___x_1004_, v_z_1003_, v___x_1004_);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(lean_object* v_k_1006_, lean_object* v_v_1007_, lean_object* v_t_1008_){
_start:
{
if (lean_obj_tag(v_t_1008_) == 0)
{
lean_object* v_size_1009_; lean_object* v_k_1010_; lean_object* v_v_1011_; lean_object* v_l_1012_; lean_object* v_r_1013_; lean_object* v___x_1015_; uint8_t v_isShared_1016_; uint8_t v_isSharedCheck_1294_; 
v_size_1009_ = lean_ctor_get(v_t_1008_, 0);
v_k_1010_ = lean_ctor_get(v_t_1008_, 1);
v_v_1011_ = lean_ctor_get(v_t_1008_, 2);
v_l_1012_ = lean_ctor_get(v_t_1008_, 3);
v_r_1013_ = lean_ctor_get(v_t_1008_, 4);
v_isSharedCheck_1294_ = !lean_is_exclusive(v_t_1008_);
if (v_isSharedCheck_1294_ == 0)
{
v___x_1015_ = v_t_1008_;
v_isShared_1016_ = v_isSharedCheck_1294_;
goto v_resetjp_1014_;
}
else
{
lean_inc(v_r_1013_);
lean_inc(v_l_1012_);
lean_inc(v_v_1011_);
lean_inc(v_k_1010_);
lean_inc(v_size_1009_);
lean_dec(v_t_1008_);
v___x_1015_ = lean_box(0);
v_isShared_1016_ = v_isSharedCheck_1294_;
goto v_resetjp_1014_;
}
v_resetjp_1014_:
{
uint8_t v___x_1017_; 
v___x_1017_ = lean_nat_dec_lt(v_k_1006_, v_k_1010_);
if (v___x_1017_ == 0)
{
uint8_t v___x_1018_; 
v___x_1018_ = lean_nat_dec_eq(v_k_1006_, v_k_1010_);
if (v___x_1018_ == 0)
{
lean_object* v_impl_1019_; lean_object* v___x_1020_; 
lean_dec(v_size_1009_);
v_impl_1019_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(v_k_1006_, v_v_1007_, v_r_1013_);
v___x_1020_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_1012_) == 0)
{
lean_object* v_size_1021_; lean_object* v_size_1022_; lean_object* v_k_1023_; lean_object* v_v_1024_; lean_object* v_l_1025_; lean_object* v_r_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; uint8_t v___x_1029_; 
v_size_1021_ = lean_ctor_get(v_l_1012_, 0);
v_size_1022_ = lean_ctor_get(v_impl_1019_, 0);
lean_inc(v_size_1022_);
v_k_1023_ = lean_ctor_get(v_impl_1019_, 1);
lean_inc(v_k_1023_);
v_v_1024_ = lean_ctor_get(v_impl_1019_, 2);
lean_inc(v_v_1024_);
v_l_1025_ = lean_ctor_get(v_impl_1019_, 3);
lean_inc(v_l_1025_);
v_r_1026_ = lean_ctor_get(v_impl_1019_, 4);
lean_inc(v_r_1026_);
v___x_1027_ = lean_unsigned_to_nat(3u);
v___x_1028_ = lean_nat_mul(v___x_1027_, v_size_1021_);
v___x_1029_ = lean_nat_dec_lt(v___x_1028_, v_size_1022_);
lean_dec(v___x_1028_);
if (v___x_1029_ == 0)
{
lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1033_; 
lean_dec(v_r_1026_);
lean_dec(v_l_1025_);
lean_dec(v_v_1024_);
lean_dec(v_k_1023_);
v___x_1030_ = lean_nat_add(v___x_1020_, v_size_1021_);
v___x_1031_ = lean_nat_add(v___x_1030_, v_size_1022_);
lean_dec(v_size_1022_);
lean_dec(v___x_1030_);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v_impl_1019_);
lean_ctor_set(v___x_1015_, 0, v___x_1031_);
v___x_1033_ = v___x_1015_;
goto v_reusejp_1032_;
}
else
{
lean_object* v_reuseFailAlloc_1034_; 
v_reuseFailAlloc_1034_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1034_, 0, v___x_1031_);
lean_ctor_set(v_reuseFailAlloc_1034_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1034_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1034_, 3, v_l_1012_);
lean_ctor_set(v_reuseFailAlloc_1034_, 4, v_impl_1019_);
v___x_1033_ = v_reuseFailAlloc_1034_;
goto v_reusejp_1032_;
}
v_reusejp_1032_:
{
return v___x_1033_;
}
}
else
{
lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1098_; 
v_isSharedCheck_1098_ = !lean_is_exclusive(v_impl_1019_);
if (v_isSharedCheck_1098_ == 0)
{
lean_object* v_unused_1099_; lean_object* v_unused_1100_; lean_object* v_unused_1101_; lean_object* v_unused_1102_; lean_object* v_unused_1103_; 
v_unused_1099_ = lean_ctor_get(v_impl_1019_, 4);
lean_dec(v_unused_1099_);
v_unused_1100_ = lean_ctor_get(v_impl_1019_, 3);
lean_dec(v_unused_1100_);
v_unused_1101_ = lean_ctor_get(v_impl_1019_, 2);
lean_dec(v_unused_1101_);
v_unused_1102_ = lean_ctor_get(v_impl_1019_, 1);
lean_dec(v_unused_1102_);
v_unused_1103_ = lean_ctor_get(v_impl_1019_, 0);
lean_dec(v_unused_1103_);
v___x_1036_ = v_impl_1019_;
v_isShared_1037_ = v_isSharedCheck_1098_;
goto v_resetjp_1035_;
}
else
{
lean_dec(v_impl_1019_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1098_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
lean_object* v_size_1038_; lean_object* v_k_1039_; lean_object* v_v_1040_; lean_object* v_l_1041_; lean_object* v_r_1042_; lean_object* v_size_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; uint8_t v___x_1046_; 
v_size_1038_ = lean_ctor_get(v_l_1025_, 0);
v_k_1039_ = lean_ctor_get(v_l_1025_, 1);
v_v_1040_ = lean_ctor_get(v_l_1025_, 2);
v_l_1041_ = lean_ctor_get(v_l_1025_, 3);
v_r_1042_ = lean_ctor_get(v_l_1025_, 4);
v_size_1043_ = lean_ctor_get(v_r_1026_, 0);
v___x_1044_ = lean_unsigned_to_nat(2u);
v___x_1045_ = lean_nat_mul(v___x_1044_, v_size_1043_);
v___x_1046_ = lean_nat_dec_lt(v_size_1038_, v___x_1045_);
lean_dec(v___x_1045_);
if (v___x_1046_ == 0)
{
lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1074_; 
lean_inc(v_r_1042_);
lean_inc(v_l_1041_);
lean_inc(v_v_1040_);
lean_inc(v_k_1039_);
v_isSharedCheck_1074_ = !lean_is_exclusive(v_l_1025_);
if (v_isSharedCheck_1074_ == 0)
{
lean_object* v_unused_1075_; lean_object* v_unused_1076_; lean_object* v_unused_1077_; lean_object* v_unused_1078_; lean_object* v_unused_1079_; 
v_unused_1075_ = lean_ctor_get(v_l_1025_, 4);
lean_dec(v_unused_1075_);
v_unused_1076_ = lean_ctor_get(v_l_1025_, 3);
lean_dec(v_unused_1076_);
v_unused_1077_ = lean_ctor_get(v_l_1025_, 2);
lean_dec(v_unused_1077_);
v_unused_1078_ = lean_ctor_get(v_l_1025_, 1);
lean_dec(v_unused_1078_);
v_unused_1079_ = lean_ctor_get(v_l_1025_, 0);
lean_dec(v_unused_1079_);
v___x_1048_ = v_l_1025_;
v_isShared_1049_ = v_isSharedCheck_1074_;
goto v_resetjp_1047_;
}
else
{
lean_dec(v_l_1025_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1074_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___y_1053_; lean_object* v___y_1054_; lean_object* v___y_1055_; lean_object* v___y_1064_; 
v___x_1050_ = lean_nat_add(v___x_1020_, v_size_1021_);
v___x_1051_ = lean_nat_add(v___x_1050_, v_size_1022_);
lean_dec(v_size_1022_);
if (lean_obj_tag(v_l_1041_) == 0)
{
lean_object* v_size_1072_; 
v_size_1072_ = lean_ctor_get(v_l_1041_, 0);
lean_inc(v_size_1072_);
v___y_1064_ = v_size_1072_;
goto v___jp_1063_;
}
else
{
lean_object* v___x_1073_; 
v___x_1073_ = lean_unsigned_to_nat(0u);
v___y_1064_ = v___x_1073_;
goto v___jp_1063_;
}
v___jp_1052_:
{
lean_object* v___x_1056_; lean_object* v___x_1058_; 
v___x_1056_ = lean_nat_add(v___y_1053_, v___y_1055_);
lean_dec(v___y_1055_);
lean_dec(v___y_1053_);
if (v_isShared_1049_ == 0)
{
lean_ctor_set(v___x_1048_, 4, v_r_1026_);
lean_ctor_set(v___x_1048_, 3, v_r_1042_);
lean_ctor_set(v___x_1048_, 2, v_v_1024_);
lean_ctor_set(v___x_1048_, 1, v_k_1023_);
lean_ctor_set(v___x_1048_, 0, v___x_1056_);
v___x_1058_ = v___x_1048_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1062_; 
v_reuseFailAlloc_1062_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1062_, 0, v___x_1056_);
lean_ctor_set(v_reuseFailAlloc_1062_, 1, v_k_1023_);
lean_ctor_set(v_reuseFailAlloc_1062_, 2, v_v_1024_);
lean_ctor_set(v_reuseFailAlloc_1062_, 3, v_r_1042_);
lean_ctor_set(v_reuseFailAlloc_1062_, 4, v_r_1026_);
v___x_1058_ = v_reuseFailAlloc_1062_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
lean_object* v___x_1060_; 
if (v_isShared_1037_ == 0)
{
lean_ctor_set(v___x_1036_, 4, v___x_1058_);
lean_ctor_set(v___x_1036_, 3, v___y_1054_);
lean_ctor_set(v___x_1036_, 2, v_v_1040_);
lean_ctor_set(v___x_1036_, 1, v_k_1039_);
lean_ctor_set(v___x_1036_, 0, v___x_1051_);
v___x_1060_ = v___x_1036_;
goto v_reusejp_1059_;
}
else
{
lean_object* v_reuseFailAlloc_1061_; 
v_reuseFailAlloc_1061_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1061_, 0, v___x_1051_);
lean_ctor_set(v_reuseFailAlloc_1061_, 1, v_k_1039_);
lean_ctor_set(v_reuseFailAlloc_1061_, 2, v_v_1040_);
lean_ctor_set(v_reuseFailAlloc_1061_, 3, v___y_1054_);
lean_ctor_set(v_reuseFailAlloc_1061_, 4, v___x_1058_);
v___x_1060_ = v_reuseFailAlloc_1061_;
goto v_reusejp_1059_;
}
v_reusejp_1059_:
{
return v___x_1060_;
}
}
}
v___jp_1063_:
{
lean_object* v___x_1065_; lean_object* v___x_1067_; 
v___x_1065_ = lean_nat_add(v___x_1050_, v___y_1064_);
lean_dec(v___y_1064_);
lean_dec(v___x_1050_);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v_l_1041_);
lean_ctor_set(v___x_1015_, 0, v___x_1065_);
v___x_1067_ = v___x_1015_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1071_; 
v_reuseFailAlloc_1071_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1071_, 0, v___x_1065_);
lean_ctor_set(v_reuseFailAlloc_1071_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1071_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1071_, 3, v_l_1012_);
lean_ctor_set(v_reuseFailAlloc_1071_, 4, v_l_1041_);
v___x_1067_ = v_reuseFailAlloc_1071_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
lean_object* v___x_1068_; 
v___x_1068_ = lean_nat_add(v___x_1020_, v_size_1043_);
if (lean_obj_tag(v_r_1042_) == 0)
{
lean_object* v_size_1069_; 
v_size_1069_ = lean_ctor_get(v_r_1042_, 0);
lean_inc(v_size_1069_);
v___y_1053_ = v___x_1068_;
v___y_1054_ = v___x_1067_;
v___y_1055_ = v_size_1069_;
goto v___jp_1052_;
}
else
{
lean_object* v___x_1070_; 
v___x_1070_ = lean_unsigned_to_nat(0u);
v___y_1053_ = v___x_1068_;
v___y_1054_ = v___x_1067_;
v___y_1055_ = v___x_1070_;
goto v___jp_1052_;
}
}
}
}
}
else
{
lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1084_; 
lean_del_object(v___x_1015_);
v___x_1080_ = lean_nat_add(v___x_1020_, v_size_1021_);
v___x_1081_ = lean_nat_add(v___x_1080_, v_size_1022_);
lean_dec(v_size_1022_);
v___x_1082_ = lean_nat_add(v___x_1080_, v_size_1038_);
lean_dec(v___x_1080_);
lean_inc_ref(v_l_1012_);
if (v_isShared_1037_ == 0)
{
lean_ctor_set(v___x_1036_, 4, v_l_1025_);
lean_ctor_set(v___x_1036_, 3, v_l_1012_);
lean_ctor_set(v___x_1036_, 2, v_v_1011_);
lean_ctor_set(v___x_1036_, 1, v_k_1010_);
lean_ctor_set(v___x_1036_, 0, v___x_1082_);
v___x_1084_ = v___x_1036_;
goto v_reusejp_1083_;
}
else
{
lean_object* v_reuseFailAlloc_1097_; 
v_reuseFailAlloc_1097_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1097_, 0, v___x_1082_);
lean_ctor_set(v_reuseFailAlloc_1097_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1097_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1097_, 3, v_l_1012_);
lean_ctor_set(v_reuseFailAlloc_1097_, 4, v_l_1025_);
v___x_1084_ = v_reuseFailAlloc_1097_;
goto v_reusejp_1083_;
}
v_reusejp_1083_:
{
lean_object* v___x_1086_; uint8_t v_isShared_1087_; uint8_t v_isSharedCheck_1091_; 
v_isSharedCheck_1091_ = !lean_is_exclusive(v_l_1012_);
if (v_isSharedCheck_1091_ == 0)
{
lean_object* v_unused_1092_; lean_object* v_unused_1093_; lean_object* v_unused_1094_; lean_object* v_unused_1095_; lean_object* v_unused_1096_; 
v_unused_1092_ = lean_ctor_get(v_l_1012_, 4);
lean_dec(v_unused_1092_);
v_unused_1093_ = lean_ctor_get(v_l_1012_, 3);
lean_dec(v_unused_1093_);
v_unused_1094_ = lean_ctor_get(v_l_1012_, 2);
lean_dec(v_unused_1094_);
v_unused_1095_ = lean_ctor_get(v_l_1012_, 1);
lean_dec(v_unused_1095_);
v_unused_1096_ = lean_ctor_get(v_l_1012_, 0);
lean_dec(v_unused_1096_);
v___x_1086_ = v_l_1012_;
v_isShared_1087_ = v_isSharedCheck_1091_;
goto v_resetjp_1085_;
}
else
{
lean_dec(v_l_1012_);
v___x_1086_ = lean_box(0);
v_isShared_1087_ = v_isSharedCheck_1091_;
goto v_resetjp_1085_;
}
v_resetjp_1085_:
{
lean_object* v___x_1089_; 
if (v_isShared_1087_ == 0)
{
lean_ctor_set(v___x_1086_, 4, v_r_1026_);
lean_ctor_set(v___x_1086_, 3, v___x_1084_);
lean_ctor_set(v___x_1086_, 2, v_v_1024_);
lean_ctor_set(v___x_1086_, 1, v_k_1023_);
lean_ctor_set(v___x_1086_, 0, v___x_1081_);
v___x_1089_ = v___x_1086_;
goto v_reusejp_1088_;
}
else
{
lean_object* v_reuseFailAlloc_1090_; 
v_reuseFailAlloc_1090_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1090_, 0, v___x_1081_);
lean_ctor_set(v_reuseFailAlloc_1090_, 1, v_k_1023_);
lean_ctor_set(v_reuseFailAlloc_1090_, 2, v_v_1024_);
lean_ctor_set(v_reuseFailAlloc_1090_, 3, v___x_1084_);
lean_ctor_set(v_reuseFailAlloc_1090_, 4, v_r_1026_);
v___x_1089_ = v_reuseFailAlloc_1090_;
goto v_reusejp_1088_;
}
v_reusejp_1088_:
{
return v___x_1089_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_1104_; 
v_l_1104_ = lean_ctor_get(v_impl_1019_, 3);
lean_inc(v_l_1104_);
if (lean_obj_tag(v_l_1104_) == 0)
{
lean_object* v_r_1105_; lean_object* v_k_1106_; lean_object* v_v_1107_; lean_object* v___x_1109_; uint8_t v_isShared_1110_; uint8_t v_isSharedCheck_1130_; 
v_r_1105_ = lean_ctor_get(v_impl_1019_, 4);
v_k_1106_ = lean_ctor_get(v_impl_1019_, 1);
v_v_1107_ = lean_ctor_get(v_impl_1019_, 2);
v_isSharedCheck_1130_ = !lean_is_exclusive(v_impl_1019_);
if (v_isSharedCheck_1130_ == 0)
{
lean_object* v_unused_1131_; lean_object* v_unused_1132_; 
v_unused_1131_ = lean_ctor_get(v_impl_1019_, 3);
lean_dec(v_unused_1131_);
v_unused_1132_ = lean_ctor_get(v_impl_1019_, 0);
lean_dec(v_unused_1132_);
v___x_1109_ = v_impl_1019_;
v_isShared_1110_ = v_isSharedCheck_1130_;
goto v_resetjp_1108_;
}
else
{
lean_inc(v_r_1105_);
lean_inc(v_v_1107_);
lean_inc(v_k_1106_);
lean_dec(v_impl_1019_);
v___x_1109_ = lean_box(0);
v_isShared_1110_ = v_isSharedCheck_1130_;
goto v_resetjp_1108_;
}
v_resetjp_1108_:
{
lean_object* v_k_1111_; lean_object* v_v_1112_; lean_object* v___x_1114_; uint8_t v_isShared_1115_; uint8_t v_isSharedCheck_1126_; 
v_k_1111_ = lean_ctor_get(v_l_1104_, 1);
v_v_1112_ = lean_ctor_get(v_l_1104_, 2);
v_isSharedCheck_1126_ = !lean_is_exclusive(v_l_1104_);
if (v_isSharedCheck_1126_ == 0)
{
lean_object* v_unused_1127_; lean_object* v_unused_1128_; lean_object* v_unused_1129_; 
v_unused_1127_ = lean_ctor_get(v_l_1104_, 4);
lean_dec(v_unused_1127_);
v_unused_1128_ = lean_ctor_get(v_l_1104_, 3);
lean_dec(v_unused_1128_);
v_unused_1129_ = lean_ctor_get(v_l_1104_, 0);
lean_dec(v_unused_1129_);
v___x_1114_ = v_l_1104_;
v_isShared_1115_ = v_isSharedCheck_1126_;
goto v_resetjp_1113_;
}
else
{
lean_inc(v_v_1112_);
lean_inc(v_k_1111_);
lean_dec(v_l_1104_);
v___x_1114_ = lean_box(0);
v_isShared_1115_ = v_isSharedCheck_1126_;
goto v_resetjp_1113_;
}
v_resetjp_1113_:
{
lean_object* v___x_1116_; lean_object* v___x_1118_; 
v___x_1116_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_1105_, 2);
if (v_isShared_1115_ == 0)
{
lean_ctor_set(v___x_1114_, 4, v_r_1105_);
lean_ctor_set(v___x_1114_, 3, v_r_1105_);
lean_ctor_set(v___x_1114_, 2, v_v_1011_);
lean_ctor_set(v___x_1114_, 1, v_k_1010_);
lean_ctor_set(v___x_1114_, 0, v___x_1020_);
v___x_1118_ = v___x_1114_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v___x_1020_);
lean_ctor_set(v_reuseFailAlloc_1125_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1125_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1125_, 3, v_r_1105_);
lean_ctor_set(v_reuseFailAlloc_1125_, 4, v_r_1105_);
v___x_1118_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
lean_object* v___x_1120_; 
lean_inc(v_r_1105_);
if (v_isShared_1110_ == 0)
{
lean_ctor_set(v___x_1109_, 3, v_r_1105_);
lean_ctor_set(v___x_1109_, 0, v___x_1020_);
v___x_1120_ = v___x_1109_;
goto v_reusejp_1119_;
}
else
{
lean_object* v_reuseFailAlloc_1124_; 
v_reuseFailAlloc_1124_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1124_, 0, v___x_1020_);
lean_ctor_set(v_reuseFailAlloc_1124_, 1, v_k_1106_);
lean_ctor_set(v_reuseFailAlloc_1124_, 2, v_v_1107_);
lean_ctor_set(v_reuseFailAlloc_1124_, 3, v_r_1105_);
lean_ctor_set(v_reuseFailAlloc_1124_, 4, v_r_1105_);
v___x_1120_ = v_reuseFailAlloc_1124_;
goto v_reusejp_1119_;
}
v_reusejp_1119_:
{
lean_object* v___x_1122_; 
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v___x_1120_);
lean_ctor_set(v___x_1015_, 3, v___x_1118_);
lean_ctor_set(v___x_1015_, 2, v_v_1112_);
lean_ctor_set(v___x_1015_, 1, v_k_1111_);
lean_ctor_set(v___x_1015_, 0, v___x_1116_);
v___x_1122_ = v___x_1015_;
goto v_reusejp_1121_;
}
else
{
lean_object* v_reuseFailAlloc_1123_; 
v_reuseFailAlloc_1123_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1123_, 0, v___x_1116_);
lean_ctor_set(v_reuseFailAlloc_1123_, 1, v_k_1111_);
lean_ctor_set(v_reuseFailAlloc_1123_, 2, v_v_1112_);
lean_ctor_set(v_reuseFailAlloc_1123_, 3, v___x_1118_);
lean_ctor_set(v_reuseFailAlloc_1123_, 4, v___x_1120_);
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
else
{
lean_object* v_r_1133_; 
v_r_1133_ = lean_ctor_get(v_impl_1019_, 4);
lean_inc(v_r_1133_);
if (lean_obj_tag(v_r_1133_) == 0)
{
lean_object* v_k_1134_; lean_object* v_v_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1146_; 
v_k_1134_ = lean_ctor_get(v_impl_1019_, 1);
v_v_1135_ = lean_ctor_get(v_impl_1019_, 2);
v_isSharedCheck_1146_ = !lean_is_exclusive(v_impl_1019_);
if (v_isSharedCheck_1146_ == 0)
{
lean_object* v_unused_1147_; lean_object* v_unused_1148_; lean_object* v_unused_1149_; 
v_unused_1147_ = lean_ctor_get(v_impl_1019_, 4);
lean_dec(v_unused_1147_);
v_unused_1148_ = lean_ctor_get(v_impl_1019_, 3);
lean_dec(v_unused_1148_);
v_unused_1149_ = lean_ctor_get(v_impl_1019_, 0);
lean_dec(v_unused_1149_);
v___x_1137_ = v_impl_1019_;
v_isShared_1138_ = v_isSharedCheck_1146_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_v_1135_);
lean_inc(v_k_1134_);
lean_dec(v_impl_1019_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1146_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1139_; lean_object* v___x_1141_; 
v___x_1139_ = lean_unsigned_to_nat(3u);
if (v_isShared_1138_ == 0)
{
lean_ctor_set(v___x_1137_, 4, v_l_1104_);
lean_ctor_set(v___x_1137_, 2, v_v_1011_);
lean_ctor_set(v___x_1137_, 1, v_k_1010_);
lean_ctor_set(v___x_1137_, 0, v___x_1020_);
v___x_1141_ = v___x_1137_;
goto v_reusejp_1140_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v___x_1020_);
lean_ctor_set(v_reuseFailAlloc_1145_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1145_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1145_, 3, v_l_1104_);
lean_ctor_set(v_reuseFailAlloc_1145_, 4, v_l_1104_);
v___x_1141_ = v_reuseFailAlloc_1145_;
goto v_reusejp_1140_;
}
v_reusejp_1140_:
{
lean_object* v___x_1143_; 
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v_r_1133_);
lean_ctor_set(v___x_1015_, 3, v___x_1141_);
lean_ctor_set(v___x_1015_, 2, v_v_1135_);
lean_ctor_set(v___x_1015_, 1, v_k_1134_);
lean_ctor_set(v___x_1015_, 0, v___x_1139_);
v___x_1143_ = v___x_1015_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1144_; 
v_reuseFailAlloc_1144_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1144_, 0, v___x_1139_);
lean_ctor_set(v_reuseFailAlloc_1144_, 1, v_k_1134_);
lean_ctor_set(v_reuseFailAlloc_1144_, 2, v_v_1135_);
lean_ctor_set(v_reuseFailAlloc_1144_, 3, v___x_1141_);
lean_ctor_set(v_reuseFailAlloc_1144_, 4, v_r_1133_);
v___x_1143_ = v_reuseFailAlloc_1144_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
return v___x_1143_;
}
}
}
}
else
{
lean_object* v___x_1150_; lean_object* v___x_1152_; 
v___x_1150_ = lean_unsigned_to_nat(2u);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v_impl_1019_);
lean_ctor_set(v___x_1015_, 3, v_r_1133_);
lean_ctor_set(v___x_1015_, 0, v___x_1150_);
v___x_1152_ = v___x_1015_;
goto v_reusejp_1151_;
}
else
{
lean_object* v_reuseFailAlloc_1153_; 
v_reuseFailAlloc_1153_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1153_, 0, v___x_1150_);
lean_ctor_set(v_reuseFailAlloc_1153_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1153_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1153_, 3, v_r_1133_);
lean_ctor_set(v_reuseFailAlloc_1153_, 4, v_impl_1019_);
v___x_1152_ = v_reuseFailAlloc_1153_;
goto v_reusejp_1151_;
}
v_reusejp_1151_:
{
return v___x_1152_;
}
}
}
}
}
else
{
lean_object* v___x_1155_; 
lean_dec(v_v_1011_);
lean_dec(v_k_1010_);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 2, v_v_1007_);
lean_ctor_set(v___x_1015_, 1, v_k_1006_);
v___x_1155_ = v___x_1015_;
goto v_reusejp_1154_;
}
else
{
lean_object* v_reuseFailAlloc_1156_; 
v_reuseFailAlloc_1156_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1156_, 0, v_size_1009_);
lean_ctor_set(v_reuseFailAlloc_1156_, 1, v_k_1006_);
lean_ctor_set(v_reuseFailAlloc_1156_, 2, v_v_1007_);
lean_ctor_set(v_reuseFailAlloc_1156_, 3, v_l_1012_);
lean_ctor_set(v_reuseFailAlloc_1156_, 4, v_r_1013_);
v___x_1155_ = v_reuseFailAlloc_1156_;
goto v_reusejp_1154_;
}
v_reusejp_1154_:
{
return v___x_1155_;
}
}
}
else
{
lean_object* v_impl_1157_; lean_object* v___x_1158_; 
lean_dec(v_size_1009_);
v_impl_1157_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(v_k_1006_, v_v_1007_, v_l_1012_);
v___x_1158_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_1013_) == 0)
{
lean_object* v_size_1159_; lean_object* v_size_1160_; lean_object* v_k_1161_; lean_object* v_v_1162_; lean_object* v_l_1163_; lean_object* v_r_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; uint8_t v___x_1167_; 
v_size_1159_ = lean_ctor_get(v_r_1013_, 0);
v_size_1160_ = lean_ctor_get(v_impl_1157_, 0);
lean_inc(v_size_1160_);
v_k_1161_ = lean_ctor_get(v_impl_1157_, 1);
lean_inc(v_k_1161_);
v_v_1162_ = lean_ctor_get(v_impl_1157_, 2);
lean_inc(v_v_1162_);
v_l_1163_ = lean_ctor_get(v_impl_1157_, 3);
lean_inc(v_l_1163_);
v_r_1164_ = lean_ctor_get(v_impl_1157_, 4);
lean_inc(v_r_1164_);
v___x_1165_ = lean_unsigned_to_nat(3u);
v___x_1166_ = lean_nat_mul(v___x_1165_, v_size_1159_);
v___x_1167_ = lean_nat_dec_lt(v___x_1166_, v_size_1160_);
lean_dec(v___x_1166_);
if (v___x_1167_ == 0)
{
lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1171_; 
lean_dec(v_r_1164_);
lean_dec(v_l_1163_);
lean_dec(v_v_1162_);
lean_dec(v_k_1161_);
v___x_1168_ = lean_nat_add(v___x_1158_, v_size_1160_);
lean_dec(v_size_1160_);
v___x_1169_ = lean_nat_add(v___x_1168_, v_size_1159_);
lean_dec(v___x_1168_);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 3, v_impl_1157_);
lean_ctor_set(v___x_1015_, 0, v___x_1169_);
v___x_1171_ = v___x_1015_;
goto v_reusejp_1170_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v___x_1169_);
lean_ctor_set(v_reuseFailAlloc_1172_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1172_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1172_, 3, v_impl_1157_);
lean_ctor_set(v_reuseFailAlloc_1172_, 4, v_r_1013_);
v___x_1171_ = v_reuseFailAlloc_1172_;
goto v_reusejp_1170_;
}
v_reusejp_1170_:
{
return v___x_1171_;
}
}
else
{
lean_object* v___x_1174_; uint8_t v_isShared_1175_; uint8_t v_isSharedCheck_1238_; 
v_isSharedCheck_1238_ = !lean_is_exclusive(v_impl_1157_);
if (v_isSharedCheck_1238_ == 0)
{
lean_object* v_unused_1239_; lean_object* v_unused_1240_; lean_object* v_unused_1241_; lean_object* v_unused_1242_; lean_object* v_unused_1243_; 
v_unused_1239_ = lean_ctor_get(v_impl_1157_, 4);
lean_dec(v_unused_1239_);
v_unused_1240_ = lean_ctor_get(v_impl_1157_, 3);
lean_dec(v_unused_1240_);
v_unused_1241_ = lean_ctor_get(v_impl_1157_, 2);
lean_dec(v_unused_1241_);
v_unused_1242_ = lean_ctor_get(v_impl_1157_, 1);
lean_dec(v_unused_1242_);
v_unused_1243_ = lean_ctor_get(v_impl_1157_, 0);
lean_dec(v_unused_1243_);
v___x_1174_ = v_impl_1157_;
v_isShared_1175_ = v_isSharedCheck_1238_;
goto v_resetjp_1173_;
}
else
{
lean_dec(v_impl_1157_);
v___x_1174_ = lean_box(0);
v_isShared_1175_ = v_isSharedCheck_1238_;
goto v_resetjp_1173_;
}
v_resetjp_1173_:
{
lean_object* v_size_1176_; lean_object* v_size_1177_; lean_object* v_k_1178_; lean_object* v_v_1179_; lean_object* v_l_1180_; lean_object* v_r_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; uint8_t v___x_1184_; 
v_size_1176_ = lean_ctor_get(v_l_1163_, 0);
v_size_1177_ = lean_ctor_get(v_r_1164_, 0);
v_k_1178_ = lean_ctor_get(v_r_1164_, 1);
v_v_1179_ = lean_ctor_get(v_r_1164_, 2);
v_l_1180_ = lean_ctor_get(v_r_1164_, 3);
v_r_1181_ = lean_ctor_get(v_r_1164_, 4);
v___x_1182_ = lean_unsigned_to_nat(2u);
v___x_1183_ = lean_nat_mul(v___x_1182_, v_size_1176_);
v___x_1184_ = lean_nat_dec_lt(v_size_1177_, v___x_1183_);
lean_dec(v___x_1183_);
if (v___x_1184_ == 0)
{
lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1213_; 
lean_inc(v_r_1181_);
lean_inc(v_l_1180_);
lean_inc(v_v_1179_);
lean_inc(v_k_1178_);
v_isSharedCheck_1213_ = !lean_is_exclusive(v_r_1164_);
if (v_isSharedCheck_1213_ == 0)
{
lean_object* v_unused_1214_; lean_object* v_unused_1215_; lean_object* v_unused_1216_; lean_object* v_unused_1217_; lean_object* v_unused_1218_; 
v_unused_1214_ = lean_ctor_get(v_r_1164_, 4);
lean_dec(v_unused_1214_);
v_unused_1215_ = lean_ctor_get(v_r_1164_, 3);
lean_dec(v_unused_1215_);
v_unused_1216_ = lean_ctor_get(v_r_1164_, 2);
lean_dec(v_unused_1216_);
v_unused_1217_ = lean_ctor_get(v_r_1164_, 1);
lean_dec(v_unused_1217_);
v_unused_1218_ = lean_ctor_get(v_r_1164_, 0);
lean_dec(v_unused_1218_);
v___x_1186_ = v_r_1164_;
v_isShared_1187_ = v_isSharedCheck_1213_;
goto v_resetjp_1185_;
}
else
{
lean_dec(v_r_1164_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1213_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___y_1191_; lean_object* v___y_1192_; lean_object* v___y_1193_; lean_object* v___x_1201_; lean_object* v___y_1203_; 
v___x_1188_ = lean_nat_add(v___x_1158_, v_size_1160_);
lean_dec(v_size_1160_);
v___x_1189_ = lean_nat_add(v___x_1188_, v_size_1159_);
lean_dec(v___x_1188_);
v___x_1201_ = lean_nat_add(v___x_1158_, v_size_1176_);
if (lean_obj_tag(v_l_1180_) == 0)
{
lean_object* v_size_1211_; 
v_size_1211_ = lean_ctor_get(v_l_1180_, 0);
lean_inc(v_size_1211_);
v___y_1203_ = v_size_1211_;
goto v___jp_1202_;
}
else
{
lean_object* v___x_1212_; 
v___x_1212_ = lean_unsigned_to_nat(0u);
v___y_1203_ = v___x_1212_;
goto v___jp_1202_;
}
v___jp_1190_:
{
lean_object* v___x_1194_; lean_object* v___x_1196_; 
v___x_1194_ = lean_nat_add(v___y_1192_, v___y_1193_);
lean_dec(v___y_1193_);
lean_dec(v___y_1192_);
if (v_isShared_1187_ == 0)
{
lean_ctor_set(v___x_1186_, 4, v_r_1013_);
lean_ctor_set(v___x_1186_, 3, v_r_1181_);
lean_ctor_set(v___x_1186_, 2, v_v_1011_);
lean_ctor_set(v___x_1186_, 1, v_k_1010_);
lean_ctor_set(v___x_1186_, 0, v___x_1194_);
v___x_1196_ = v___x_1186_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1200_; 
v_reuseFailAlloc_1200_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1200_, 0, v___x_1194_);
lean_ctor_set(v_reuseFailAlloc_1200_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1200_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1200_, 3, v_r_1181_);
lean_ctor_set(v_reuseFailAlloc_1200_, 4, v_r_1013_);
v___x_1196_ = v_reuseFailAlloc_1200_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
lean_object* v___x_1198_; 
if (v_isShared_1175_ == 0)
{
lean_ctor_set(v___x_1174_, 4, v___x_1196_);
lean_ctor_set(v___x_1174_, 3, v___y_1191_);
lean_ctor_set(v___x_1174_, 2, v_v_1179_);
lean_ctor_set(v___x_1174_, 1, v_k_1178_);
lean_ctor_set(v___x_1174_, 0, v___x_1189_);
v___x_1198_ = v___x_1174_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v___x_1189_);
lean_ctor_set(v_reuseFailAlloc_1199_, 1, v_k_1178_);
lean_ctor_set(v_reuseFailAlloc_1199_, 2, v_v_1179_);
lean_ctor_set(v_reuseFailAlloc_1199_, 3, v___y_1191_);
lean_ctor_set(v_reuseFailAlloc_1199_, 4, v___x_1196_);
v___x_1198_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
return v___x_1198_;
}
}
}
v___jp_1202_:
{
lean_object* v___x_1204_; lean_object* v___x_1206_; 
v___x_1204_ = lean_nat_add(v___x_1201_, v___y_1203_);
lean_dec(v___y_1203_);
lean_dec(v___x_1201_);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v_l_1180_);
lean_ctor_set(v___x_1015_, 3, v_l_1163_);
lean_ctor_set(v___x_1015_, 2, v_v_1162_);
lean_ctor_set(v___x_1015_, 1, v_k_1161_);
lean_ctor_set(v___x_1015_, 0, v___x_1204_);
v___x_1206_ = v___x_1015_;
goto v_reusejp_1205_;
}
else
{
lean_object* v_reuseFailAlloc_1210_; 
v_reuseFailAlloc_1210_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1210_, 0, v___x_1204_);
lean_ctor_set(v_reuseFailAlloc_1210_, 1, v_k_1161_);
lean_ctor_set(v_reuseFailAlloc_1210_, 2, v_v_1162_);
lean_ctor_set(v_reuseFailAlloc_1210_, 3, v_l_1163_);
lean_ctor_set(v_reuseFailAlloc_1210_, 4, v_l_1180_);
v___x_1206_ = v_reuseFailAlloc_1210_;
goto v_reusejp_1205_;
}
v_reusejp_1205_:
{
lean_object* v___x_1207_; 
v___x_1207_ = lean_nat_add(v___x_1158_, v_size_1159_);
if (lean_obj_tag(v_r_1181_) == 0)
{
lean_object* v_size_1208_; 
v_size_1208_ = lean_ctor_get(v_r_1181_, 0);
lean_inc(v_size_1208_);
v___y_1191_ = v___x_1206_;
v___y_1192_ = v___x_1207_;
v___y_1193_ = v_size_1208_;
goto v___jp_1190_;
}
else
{
lean_object* v___x_1209_; 
v___x_1209_ = lean_unsigned_to_nat(0u);
v___y_1191_ = v___x_1206_;
v___y_1192_ = v___x_1207_;
v___y_1193_ = v___x_1209_;
goto v___jp_1190_;
}
}
}
}
}
else
{
lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1224_; 
lean_del_object(v___x_1015_);
v___x_1219_ = lean_nat_add(v___x_1158_, v_size_1160_);
lean_dec(v_size_1160_);
v___x_1220_ = lean_nat_add(v___x_1219_, v_size_1159_);
lean_dec(v___x_1219_);
v___x_1221_ = lean_nat_add(v___x_1158_, v_size_1159_);
v___x_1222_ = lean_nat_add(v___x_1221_, v_size_1177_);
lean_dec(v___x_1221_);
lean_inc_ref(v_r_1013_);
if (v_isShared_1175_ == 0)
{
lean_ctor_set(v___x_1174_, 4, v_r_1013_);
lean_ctor_set(v___x_1174_, 3, v_r_1164_);
lean_ctor_set(v___x_1174_, 2, v_v_1011_);
lean_ctor_set(v___x_1174_, 1, v_k_1010_);
lean_ctor_set(v___x_1174_, 0, v___x_1222_);
v___x_1224_ = v___x_1174_;
goto v_reusejp_1223_;
}
else
{
lean_object* v_reuseFailAlloc_1237_; 
v_reuseFailAlloc_1237_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1237_, 0, v___x_1222_);
lean_ctor_set(v_reuseFailAlloc_1237_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1237_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1237_, 3, v_r_1164_);
lean_ctor_set(v_reuseFailAlloc_1237_, 4, v_r_1013_);
v___x_1224_ = v_reuseFailAlloc_1237_;
goto v_reusejp_1223_;
}
v_reusejp_1223_:
{
lean_object* v___x_1226_; uint8_t v_isShared_1227_; uint8_t v_isSharedCheck_1231_; 
v_isSharedCheck_1231_ = !lean_is_exclusive(v_r_1013_);
if (v_isSharedCheck_1231_ == 0)
{
lean_object* v_unused_1232_; lean_object* v_unused_1233_; lean_object* v_unused_1234_; lean_object* v_unused_1235_; lean_object* v_unused_1236_; 
v_unused_1232_ = lean_ctor_get(v_r_1013_, 4);
lean_dec(v_unused_1232_);
v_unused_1233_ = lean_ctor_get(v_r_1013_, 3);
lean_dec(v_unused_1233_);
v_unused_1234_ = lean_ctor_get(v_r_1013_, 2);
lean_dec(v_unused_1234_);
v_unused_1235_ = lean_ctor_get(v_r_1013_, 1);
lean_dec(v_unused_1235_);
v_unused_1236_ = lean_ctor_get(v_r_1013_, 0);
lean_dec(v_unused_1236_);
v___x_1226_ = v_r_1013_;
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
else
{
lean_dec(v_r_1013_);
v___x_1226_ = lean_box(0);
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
v_resetjp_1225_:
{
lean_object* v___x_1229_; 
if (v_isShared_1227_ == 0)
{
lean_ctor_set(v___x_1226_, 4, v___x_1224_);
lean_ctor_set(v___x_1226_, 3, v_l_1163_);
lean_ctor_set(v___x_1226_, 2, v_v_1162_);
lean_ctor_set(v___x_1226_, 1, v_k_1161_);
lean_ctor_set(v___x_1226_, 0, v___x_1220_);
v___x_1229_ = v___x_1226_;
goto v_reusejp_1228_;
}
else
{
lean_object* v_reuseFailAlloc_1230_; 
v_reuseFailAlloc_1230_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1230_, 0, v___x_1220_);
lean_ctor_set(v_reuseFailAlloc_1230_, 1, v_k_1161_);
lean_ctor_set(v_reuseFailAlloc_1230_, 2, v_v_1162_);
lean_ctor_set(v_reuseFailAlloc_1230_, 3, v_l_1163_);
lean_ctor_set(v_reuseFailAlloc_1230_, 4, v___x_1224_);
v___x_1229_ = v_reuseFailAlloc_1230_;
goto v_reusejp_1228_;
}
v_reusejp_1228_:
{
return v___x_1229_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_1244_; 
v_l_1244_ = lean_ctor_get(v_impl_1157_, 3);
lean_inc(v_l_1244_);
if (lean_obj_tag(v_l_1244_) == 0)
{
lean_object* v_r_1245_; lean_object* v_k_1246_; lean_object* v_v_1247_; lean_object* v___x_1249_; uint8_t v_isShared_1250_; uint8_t v_isSharedCheck_1258_; 
v_r_1245_ = lean_ctor_get(v_impl_1157_, 4);
v_k_1246_ = lean_ctor_get(v_impl_1157_, 1);
v_v_1247_ = lean_ctor_get(v_impl_1157_, 2);
v_isSharedCheck_1258_ = !lean_is_exclusive(v_impl_1157_);
if (v_isSharedCheck_1258_ == 0)
{
lean_object* v_unused_1259_; lean_object* v_unused_1260_; 
v_unused_1259_ = lean_ctor_get(v_impl_1157_, 3);
lean_dec(v_unused_1259_);
v_unused_1260_ = lean_ctor_get(v_impl_1157_, 0);
lean_dec(v_unused_1260_);
v___x_1249_ = v_impl_1157_;
v_isShared_1250_ = v_isSharedCheck_1258_;
goto v_resetjp_1248_;
}
else
{
lean_inc(v_r_1245_);
lean_inc(v_v_1247_);
lean_inc(v_k_1246_);
lean_dec(v_impl_1157_);
v___x_1249_ = lean_box(0);
v_isShared_1250_ = v_isSharedCheck_1258_;
goto v_resetjp_1248_;
}
v_resetjp_1248_:
{
lean_object* v___x_1251_; lean_object* v___x_1253_; 
v___x_1251_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_1245_);
if (v_isShared_1250_ == 0)
{
lean_ctor_set(v___x_1249_, 3, v_r_1245_);
lean_ctor_set(v___x_1249_, 2, v_v_1011_);
lean_ctor_set(v___x_1249_, 1, v_k_1010_);
lean_ctor_set(v___x_1249_, 0, v___x_1158_);
v___x_1253_ = v___x_1249_;
goto v_reusejp_1252_;
}
else
{
lean_object* v_reuseFailAlloc_1257_; 
v_reuseFailAlloc_1257_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1257_, 0, v___x_1158_);
lean_ctor_set(v_reuseFailAlloc_1257_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1257_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1257_, 3, v_r_1245_);
lean_ctor_set(v_reuseFailAlloc_1257_, 4, v_r_1245_);
v___x_1253_ = v_reuseFailAlloc_1257_;
goto v_reusejp_1252_;
}
v_reusejp_1252_:
{
lean_object* v___x_1255_; 
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v___x_1253_);
lean_ctor_set(v___x_1015_, 3, v_l_1244_);
lean_ctor_set(v___x_1015_, 2, v_v_1247_);
lean_ctor_set(v___x_1015_, 1, v_k_1246_);
lean_ctor_set(v___x_1015_, 0, v___x_1251_);
v___x_1255_ = v___x_1015_;
goto v_reusejp_1254_;
}
else
{
lean_object* v_reuseFailAlloc_1256_; 
v_reuseFailAlloc_1256_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1256_, 0, v___x_1251_);
lean_ctor_set(v_reuseFailAlloc_1256_, 1, v_k_1246_);
lean_ctor_set(v_reuseFailAlloc_1256_, 2, v_v_1247_);
lean_ctor_set(v_reuseFailAlloc_1256_, 3, v_l_1244_);
lean_ctor_set(v_reuseFailAlloc_1256_, 4, v___x_1253_);
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
else
{
lean_object* v_r_1261_; 
v_r_1261_ = lean_ctor_get(v_impl_1157_, 4);
lean_inc(v_r_1261_);
if (lean_obj_tag(v_r_1261_) == 0)
{
lean_object* v_k_1262_; lean_object* v_v_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1286_; 
v_k_1262_ = lean_ctor_get(v_impl_1157_, 1);
v_v_1263_ = lean_ctor_get(v_impl_1157_, 2);
v_isSharedCheck_1286_ = !lean_is_exclusive(v_impl_1157_);
if (v_isSharedCheck_1286_ == 0)
{
lean_object* v_unused_1287_; lean_object* v_unused_1288_; lean_object* v_unused_1289_; 
v_unused_1287_ = lean_ctor_get(v_impl_1157_, 4);
lean_dec(v_unused_1287_);
v_unused_1288_ = lean_ctor_get(v_impl_1157_, 3);
lean_dec(v_unused_1288_);
v_unused_1289_ = lean_ctor_get(v_impl_1157_, 0);
lean_dec(v_unused_1289_);
v___x_1265_ = v_impl_1157_;
v_isShared_1266_ = v_isSharedCheck_1286_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_v_1263_);
lean_inc(v_k_1262_);
lean_dec(v_impl_1157_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1286_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v_k_1267_; lean_object* v_v_1268_; lean_object* v___x_1270_; uint8_t v_isShared_1271_; uint8_t v_isSharedCheck_1282_; 
v_k_1267_ = lean_ctor_get(v_r_1261_, 1);
v_v_1268_ = lean_ctor_get(v_r_1261_, 2);
v_isSharedCheck_1282_ = !lean_is_exclusive(v_r_1261_);
if (v_isSharedCheck_1282_ == 0)
{
lean_object* v_unused_1283_; lean_object* v_unused_1284_; lean_object* v_unused_1285_; 
v_unused_1283_ = lean_ctor_get(v_r_1261_, 4);
lean_dec(v_unused_1283_);
v_unused_1284_ = lean_ctor_get(v_r_1261_, 3);
lean_dec(v_unused_1284_);
v_unused_1285_ = lean_ctor_get(v_r_1261_, 0);
lean_dec(v_unused_1285_);
v___x_1270_ = v_r_1261_;
v_isShared_1271_ = v_isSharedCheck_1282_;
goto v_resetjp_1269_;
}
else
{
lean_inc(v_v_1268_);
lean_inc(v_k_1267_);
lean_dec(v_r_1261_);
v___x_1270_ = lean_box(0);
v_isShared_1271_ = v_isSharedCheck_1282_;
goto v_resetjp_1269_;
}
v_resetjp_1269_:
{
lean_object* v___x_1272_; lean_object* v___x_1274_; 
v___x_1272_ = lean_unsigned_to_nat(3u);
if (v_isShared_1271_ == 0)
{
lean_ctor_set(v___x_1270_, 4, v_l_1244_);
lean_ctor_set(v___x_1270_, 3, v_l_1244_);
lean_ctor_set(v___x_1270_, 2, v_v_1263_);
lean_ctor_set(v___x_1270_, 1, v_k_1262_);
lean_ctor_set(v___x_1270_, 0, v___x_1158_);
v___x_1274_ = v___x_1270_;
goto v_reusejp_1273_;
}
else
{
lean_object* v_reuseFailAlloc_1281_; 
v_reuseFailAlloc_1281_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1281_, 0, v___x_1158_);
lean_ctor_set(v_reuseFailAlloc_1281_, 1, v_k_1262_);
lean_ctor_set(v_reuseFailAlloc_1281_, 2, v_v_1263_);
lean_ctor_set(v_reuseFailAlloc_1281_, 3, v_l_1244_);
lean_ctor_set(v_reuseFailAlloc_1281_, 4, v_l_1244_);
v___x_1274_ = v_reuseFailAlloc_1281_;
goto v_reusejp_1273_;
}
v_reusejp_1273_:
{
lean_object* v___x_1276_; 
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 4, v_l_1244_);
lean_ctor_set(v___x_1265_, 2, v_v_1011_);
lean_ctor_set(v___x_1265_, 1, v_k_1010_);
lean_ctor_set(v___x_1265_, 0, v___x_1158_);
v___x_1276_ = v___x_1265_;
goto v_reusejp_1275_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v___x_1158_);
lean_ctor_set(v_reuseFailAlloc_1280_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1280_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1280_, 3, v_l_1244_);
lean_ctor_set(v_reuseFailAlloc_1280_, 4, v_l_1244_);
v___x_1276_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1275_;
}
v_reusejp_1275_:
{
lean_object* v___x_1278_; 
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v___x_1276_);
lean_ctor_set(v___x_1015_, 3, v___x_1274_);
lean_ctor_set(v___x_1015_, 2, v_v_1268_);
lean_ctor_set(v___x_1015_, 1, v_k_1267_);
lean_ctor_set(v___x_1015_, 0, v___x_1272_);
v___x_1278_ = v___x_1015_;
goto v_reusejp_1277_;
}
else
{
lean_object* v_reuseFailAlloc_1279_; 
v_reuseFailAlloc_1279_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1279_, 0, v___x_1272_);
lean_ctor_set(v_reuseFailAlloc_1279_, 1, v_k_1267_);
lean_ctor_set(v_reuseFailAlloc_1279_, 2, v_v_1268_);
lean_ctor_set(v_reuseFailAlloc_1279_, 3, v___x_1274_);
lean_ctor_set(v_reuseFailAlloc_1279_, 4, v___x_1276_);
v___x_1278_ = v_reuseFailAlloc_1279_;
goto v_reusejp_1277_;
}
v_reusejp_1277_:
{
return v___x_1278_;
}
}
}
}
}
}
else
{
lean_object* v___x_1290_; lean_object* v___x_1292_; 
v___x_1290_ = lean_unsigned_to_nat(2u);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 4, v_r_1261_);
lean_ctor_set(v___x_1015_, 3, v_impl_1157_);
lean_ctor_set(v___x_1015_, 0, v___x_1290_);
v___x_1292_ = v___x_1015_;
goto v_reusejp_1291_;
}
else
{
lean_object* v_reuseFailAlloc_1293_; 
v_reuseFailAlloc_1293_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1293_, 0, v___x_1290_);
lean_ctor_set(v_reuseFailAlloc_1293_, 1, v_k_1010_);
lean_ctor_set(v_reuseFailAlloc_1293_, 2, v_v_1011_);
lean_ctor_set(v_reuseFailAlloc_1293_, 3, v_impl_1157_);
lean_ctor_set(v_reuseFailAlloc_1293_, 4, v_r_1261_);
v___x_1292_ = v_reuseFailAlloc_1293_;
goto v_reusejp_1291_;
}
v_reusejp_1291_:
{
return v___x_1292_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1295_; lean_object* v___x_1296_; 
v___x_1295_ = lean_unsigned_to_nat(1u);
v___x_1296_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1296_, 0, v___x_1295_);
lean_ctor_set(v___x_1296_, 1, v_k_1006_);
lean_ctor_set(v___x_1296_, 2, v_v_1007_);
lean_ctor_set(v___x_1296_, 3, v_t_1008_);
lean_ctor_set(v___x_1296_, 4, v_t_1008_);
return v___x_1296_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_var(lean_object* v_n_1297_){
_start:
{
lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; 
v___x_1298_ = lean_box(1);
v___x_1299_ = lean_unsigned_to_nat(1u);
v___x_1300_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(v_n_1297_, v___x_1299_, v___x_1298_);
v___x_1301_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0);
v___x_1302_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v___x_1300_, v___x_1301_, v___x_1298_);
return v___x_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0(lean_object* v_00_u03b2_1303_, lean_object* v_k_1304_, lean_object* v_v_1305_, lean_object* v_t_1306_, lean_object* v_hl_1307_){
_start:
{
lean_object* v___x_1308_; 
v___x_1308_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(v_k_1304_, v_v_1305_, v_t_1306_);
return v___x_1308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(uint8_t v_red_1309_, lean_object* v_m_1310_, lean_object* v_e_1311_, lean_object* v_a_1312_, lean_object* v_a_1313_, lean_object* v_a_1314_, lean_object* v_a_1315_){
_start:
{
lean_object* v___x_1317_; 
lean_inc_ref(v_e_1311_);
lean_inc(v_m_1310_);
v___x_1317_ = lp_mathlib_List_findDefeq___redArg(v_red_1309_, v_m_1310_, v_e_1311_, v_a_1312_, v_a_1313_, v_a_1314_, v_a_1315_);
if (lean_obj_tag(v___x_1317_) == 0)
{
lean_object* v_a_1318_; lean_object* v___x_1320_; uint8_t v_isShared_1321_; uint8_t v_isSharedCheck_1327_; 
lean_dec_ref(v_e_1311_);
v_a_1318_ = lean_ctor_get(v___x_1317_, 0);
v_isSharedCheck_1327_ = !lean_is_exclusive(v___x_1317_);
if (v_isSharedCheck_1327_ == 0)
{
v___x_1320_ = v___x_1317_;
v_isShared_1321_ = v_isSharedCheck_1327_;
goto v_resetjp_1319_;
}
else
{
lean_inc(v_a_1318_);
lean_dec(v___x_1317_);
v___x_1320_ = lean_box(0);
v_isShared_1321_ = v_isSharedCheck_1327_;
goto v_resetjp_1319_;
}
v_resetjp_1319_:
{
lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1325_; 
v___x_1322_ = lp_mathlib_Mathlib_Tactic_Linarith_var(v_a_1318_);
v___x_1323_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1323_, 0, v_m_1310_);
lean_ctor_set(v___x_1323_, 1, v___x_1322_);
if (v_isShared_1321_ == 0)
{
lean_ctor_set(v___x_1320_, 0, v___x_1323_);
v___x_1325_ = v___x_1320_;
goto v_reusejp_1324_;
}
else
{
lean_object* v_reuseFailAlloc_1326_; 
v_reuseFailAlloc_1326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1326_, 0, v___x_1323_);
v___x_1325_ = v_reuseFailAlloc_1326_;
goto v_reusejp_1324_;
}
v_reusejp_1324_:
{
return v___x_1325_;
}
}
}
else
{
lean_object* v_a_1328_; lean_object* v___x_1330_; uint8_t v_isShared_1331_; uint8_t v_isSharedCheck_1349_; 
v_a_1328_ = lean_ctor_get(v___x_1317_, 0);
v_isSharedCheck_1349_ = !lean_is_exclusive(v___x_1317_);
if (v_isSharedCheck_1349_ == 0)
{
v___x_1330_ = v___x_1317_;
v_isShared_1331_ = v_isSharedCheck_1349_;
goto v_resetjp_1329_;
}
else
{
lean_inc(v_a_1328_);
lean_dec(v___x_1317_);
v___x_1330_ = lean_box(0);
v_isShared_1331_ = v_isSharedCheck_1349_;
goto v_resetjp_1329_;
}
v_resetjp_1329_:
{
uint8_t v___y_1333_; uint8_t v___x_1347_; 
v___x_1347_ = l_Lean_Exception_isInterrupt(v_a_1328_);
if (v___x_1347_ == 0)
{
uint8_t v___x_1348_; 
lean_inc(v_a_1328_);
v___x_1348_ = l_Lean_Exception_isRuntime(v_a_1328_);
v___y_1333_ = v___x_1348_;
goto v___jp_1332_;
}
else
{
v___y_1333_ = v___x_1347_;
goto v___jp_1332_;
}
v___jp_1332_:
{
if (v___y_1333_ == 0)
{
lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1342_; 
lean_dec(v_a_1328_);
v___x_1334_ = l_List_lengthTR___redArg(v_m_1310_);
v___x_1335_ = lean_unsigned_to_nat(1u);
v___x_1336_ = lean_nat_add(v___x_1334_, v___x_1335_);
lean_dec(v___x_1334_);
lean_inc(v___x_1336_);
v___x_1337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1337_, 0, v_e_1311_);
lean_ctor_set(v___x_1337_, 1, v___x_1336_);
v___x_1338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1338_, 0, v___x_1337_);
lean_ctor_set(v___x_1338_, 1, v_m_1310_);
v___x_1339_ = lp_mathlib_Mathlib_Tactic_Linarith_var(v___x_1336_);
v___x_1340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1340_, 0, v___x_1338_);
lean_ctor_set(v___x_1340_, 1, v___x_1339_);
if (v_isShared_1331_ == 0)
{
lean_ctor_set_tag(v___x_1330_, 0);
lean_ctor_set(v___x_1330_, 0, v___x_1340_);
v___x_1342_ = v___x_1330_;
goto v_reusejp_1341_;
}
else
{
lean_object* v_reuseFailAlloc_1343_; 
v_reuseFailAlloc_1343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1343_, 0, v___x_1340_);
v___x_1342_ = v_reuseFailAlloc_1343_;
goto v_reusejp_1341_;
}
v_reusejp_1341_:
{
return v___x_1342_;
}
}
else
{
lean_object* v___x_1345_; 
lean_dec_ref(v_e_1311_);
lean_dec(v_m_1310_);
if (v_isShared_1331_ == 0)
{
v___x_1345_ = v___x_1330_;
goto v_reusejp_1344_;
}
else
{
lean_object* v_reuseFailAlloc_1346_; 
v_reuseFailAlloc_1346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1346_, 0, v_a_1328_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom___boxed(lean_object* v_red_1350_, lean_object* v_m_1351_, lean_object* v_e_1352_, lean_object* v_a_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_, lean_object* v_a_1357_){
_start:
{
uint8_t v_red_boxed_1358_; lean_object* v_res_1359_; 
v_red_boxed_1358_ = lean_unbox(v_red_1350_);
v_res_1359_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_boxed_1358_, v_m_1351_, v_e_1352_, v_a_1353_, v_a_1354_, v_a_1355_, v_a_1356_);
lean_dec(v_a_1356_);
lean_dec_ref(v_a_1355_);
lean_dec(v_a_1354_);
lean_dec_ref(v_a_1353_);
return v_res_1359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__1(lean_object* v_a_1360_){
_start:
{
lean_object* v___x_1361_; 
v___x_1361_ = lean_nat_to_int(v_a_1360_);
return v___x_1361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__0(lean_object* v_t_1362_){
_start:
{
if (lean_obj_tag(v_t_1362_) == 0)
{
lean_object* v_size_1363_; lean_object* v_k_1364_; lean_object* v_v_1365_; lean_object* v_l_1366_; lean_object* v_r_1367_; lean_object* v___x_1369_; uint8_t v_isShared_1370_; uint8_t v_isSharedCheck_1377_; 
v_size_1363_ = lean_ctor_get(v_t_1362_, 0);
v_k_1364_ = lean_ctor_get(v_t_1362_, 1);
v_v_1365_ = lean_ctor_get(v_t_1362_, 2);
v_l_1366_ = lean_ctor_get(v_t_1362_, 3);
v_r_1367_ = lean_ctor_get(v_t_1362_, 4);
v_isSharedCheck_1377_ = !lean_is_exclusive(v_t_1362_);
if (v_isSharedCheck_1377_ == 0)
{
v___x_1369_ = v_t_1362_;
v_isShared_1370_ = v_isSharedCheck_1377_;
goto v_resetjp_1368_;
}
else
{
lean_inc(v_r_1367_);
lean_inc(v_l_1366_);
lean_inc(v_v_1365_);
lean_inc(v_k_1364_);
lean_inc(v_size_1363_);
lean_dec(v_t_1362_);
v___x_1369_ = lean_box(0);
v_isShared_1370_ = v_isSharedCheck_1377_;
goto v_resetjp_1368_;
}
v_resetjp_1368_:
{
lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1375_; 
v___x_1371_ = lean_int_neg(v_v_1365_);
lean_dec(v_v_1365_);
v___x_1372_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__0(v_l_1366_);
v___x_1373_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__0(v_r_1367_);
if (v_isShared_1370_ == 0)
{
lean_ctor_set(v___x_1369_, 4, v___x_1373_);
lean_ctor_set(v___x_1369_, 3, v___x_1372_);
lean_ctor_set(v___x_1369_, 2, v___x_1371_);
v___x_1375_ = v___x_1369_;
goto v_reusejp_1374_;
}
else
{
lean_object* v_reuseFailAlloc_1376_; 
v_reuseFailAlloc_1376_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1376_, 0, v_size_1363_);
lean_ctor_set(v_reuseFailAlloc_1376_, 1, v_k_1364_);
lean_ctor_set(v_reuseFailAlloc_1376_, 2, v___x_1371_);
lean_ctor_set(v_reuseFailAlloc_1376_, 3, v___x_1372_);
lean_ctor_set(v_reuseFailAlloc_1376_, 4, v___x_1373_);
v___x_1375_ = v_reuseFailAlloc_1376_;
goto v_reusejp_1374_;
}
v_reusejp_1374_:
{
return v___x_1375_;
}
}
}
else
{
return v_t_1362_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(uint8_t v_red_1388_, lean_object* v_m_1389_, lean_object* v_e_1390_, lean_object* v_a_1391_, lean_object* v_a_1392_, lean_object* v_a_1393_, lean_object* v_a_1394_){
_start:
{
lean_object* v___x_1396_; 
v___x_1396_ = l_Lean_Meta_whnfR(v_e_1390_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1396_) == 0)
{
lean_object* v_a_1397_; lean_object* v___x_1399_; uint8_t v_isShared_1400_; uint8_t v_isSharedCheck_1607_; 
v_a_1397_ = lean_ctor_get(v___x_1396_, 0);
v_isSharedCheck_1607_ = !lean_is_exclusive(v___x_1396_);
if (v_isSharedCheck_1607_ == 0)
{
v___x_1399_ = v___x_1396_;
v_isShared_1400_ = v_isSharedCheck_1607_;
goto v_resetjp_1398_;
}
else
{
lean_inc(v_a_1397_);
lean_dec(v___x_1396_);
v___x_1399_ = lean_box(0);
v_isShared_1400_ = v_isSharedCheck_1607_;
goto v_resetjp_1398_;
}
v_resetjp_1398_:
{
lean_object* v___x_1401_; 
lean_inc(v_a_1397_);
v___x_1401_ = lp_mathlib_Lean_Expr_numeral_x3f(v_a_1397_);
if (lean_obj_tag(v___x_1401_) == 0)
{
lean_object* v___x_1402_; lean_object* v_fst_1403_; 
lean_del_object(v___x_1399_);
lean_inc(v_a_1397_);
v___x_1402_ = l_Lean_Expr_getAppFnArgs(v_a_1397_);
v_fst_1403_ = lean_ctor_get(v___x_1402_, 0);
lean_inc(v_fst_1403_);
if (lean_obj_tag(v_fst_1403_) == 1)
{
lean_object* v_pre_1404_; 
v_pre_1404_ = lean_ctor_get(v_fst_1403_, 0);
lean_inc(v_pre_1404_);
if (lean_obj_tag(v_pre_1404_) == 1)
{
lean_object* v_pre_1405_; 
v_pre_1405_ = lean_ctor_get(v_pre_1404_, 0);
if (lean_obj_tag(v_pre_1405_) == 0)
{
lean_object* v_snd_1406_; lean_object* v_str_1407_; lean_object* v_str_1408_; lean_object* v___x_1409_; uint8_t v___x_1410_; 
v_snd_1406_ = lean_ctor_get(v___x_1402_, 1);
lean_inc(v_snd_1406_);
lean_dec_ref(v___x_1402_);
v_str_1407_ = lean_ctor_get(v_fst_1403_, 1);
lean_inc_ref(v_str_1407_);
lean_dec_ref_known(v_fst_1403_, 2);
v_str_1408_ = lean_ctor_get(v_pre_1404_, 1);
lean_inc_ref(v_str_1408_);
lean_dec_ref_known(v_pre_1404_, 2);
v___x_1409_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__0));
v___x_1410_ = lean_string_dec_eq(v_str_1408_, v___x_1409_);
if (v___x_1410_ == 0)
{
lean_object* v___x_1411_; uint8_t v___x_1412_; 
v___x_1411_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__1));
v___x_1412_ = lean_string_dec_eq(v_str_1408_, v___x_1411_);
if (v___x_1412_ == 0)
{
lean_object* v___x_1413_; uint8_t v___x_1414_; 
v___x_1413_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__2));
v___x_1414_ = lean_string_dec_eq(v_str_1408_, v___x_1413_);
if (v___x_1414_ == 0)
{
lean_object* v___x_1415_; uint8_t v___x_1416_; 
v___x_1415_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__3));
v___x_1416_ = lean_string_dec_eq(v_str_1408_, v___x_1415_);
if (v___x_1416_ == 0)
{
lean_object* v___x_1417_; uint8_t v___x_1418_; 
v___x_1417_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__4));
v___x_1418_ = lean_string_dec_eq(v_str_1408_, v___x_1417_);
lean_dec_ref(v_str_1408_);
if (v___x_1418_ == 0)
{
lean_object* v___x_1419_; 
lean_dec_ref(v_str_1407_);
lean_dec(v_snd_1406_);
v___x_1419_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1419_;
}
else
{
lean_object* v___x_1420_; uint8_t v___x_1421_; 
v___x_1420_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__5));
v___x_1421_ = lean_string_dec_eq(v_str_1407_, v___x_1420_);
lean_dec_ref(v_str_1407_);
if (v___x_1421_ == 0)
{
lean_object* v___x_1422_; 
lean_dec(v_snd_1406_);
v___x_1422_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1422_;
}
else
{
lean_object* v___x_1423_; lean_object* v___x_1424_; uint8_t v___x_1425_; 
v___x_1423_ = lean_array_get_size(v_snd_1406_);
v___x_1424_ = lean_unsigned_to_nat(6u);
v___x_1425_ = lean_nat_dec_eq(v___x_1423_, v___x_1424_);
if (v___x_1425_ == 0)
{
lean_object* v___x_1426_; 
lean_dec(v_snd_1406_);
v___x_1426_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1426_;
}
else
{
lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; 
v___x_1427_ = lean_unsigned_to_nat(5u);
v___x_1428_ = lean_array_fget_borrowed(v_snd_1406_, v___x_1427_);
lean_inc(v___x_1428_);
v___x_1429_ = lp_mathlib_Lean_Expr_numeral_x3f(v___x_1428_);
if (lean_obj_tag(v___x_1429_) == 0)
{
lean_object* v___x_1430_; 
lean_dec(v_snd_1406_);
v___x_1430_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1430_;
}
else
{
lean_object* v_val_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; 
lean_dec(v_a_1397_);
v_val_1431_ = lean_ctor_get(v___x_1429_, 0);
lean_inc(v_val_1431_);
lean_dec_ref_known(v___x_1429_, 1);
v___x_1432_ = lean_unsigned_to_nat(4u);
v___x_1433_ = lean_array_fget(v_snd_1406_, v___x_1432_);
lean_dec(v_snd_1406_);
v___x_1434_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_m_1389_, v___x_1433_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1434_) == 0)
{
lean_object* v_a_1435_; lean_object* v___x_1437_; uint8_t v_isShared_1438_; uint8_t v_isSharedCheck_1452_; 
v_a_1435_ = lean_ctor_get(v___x_1434_, 0);
v_isSharedCheck_1452_ = !lean_is_exclusive(v___x_1434_);
if (v_isSharedCheck_1452_ == 0)
{
v___x_1437_ = v___x_1434_;
v_isShared_1438_ = v_isSharedCheck_1452_;
goto v_resetjp_1436_;
}
else
{
lean_inc(v_a_1435_);
lean_dec(v___x_1434_);
v___x_1437_ = lean_box(0);
v_isShared_1438_ = v_isSharedCheck_1452_;
goto v_resetjp_1436_;
}
v_resetjp_1436_:
{
lean_object* v_fst_1439_; lean_object* v_snd_1440_; lean_object* v___x_1442_; uint8_t v_isShared_1443_; uint8_t v_isSharedCheck_1451_; 
v_fst_1439_ = lean_ctor_get(v_a_1435_, 0);
v_snd_1440_ = lean_ctor_get(v_a_1435_, 1);
v_isSharedCheck_1451_ = !lean_is_exclusive(v_a_1435_);
if (v_isSharedCheck_1451_ == 0)
{
v___x_1442_ = v_a_1435_;
v_isShared_1443_ = v_isSharedCheck_1451_;
goto v_resetjp_1441_;
}
else
{
lean_inc(v_snd_1440_);
lean_inc(v_fst_1439_);
lean_dec(v_a_1435_);
v___x_1442_ = lean_box(0);
v_isShared_1443_ = v_isSharedCheck_1451_;
goto v_resetjp_1441_;
}
v_resetjp_1441_:
{
lean_object* v___x_1444_; lean_object* v___x_1446_; 
v___x_1444_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_pow(v_snd_1440_, v_val_1431_);
lean_dec(v_val_1431_);
if (v_isShared_1443_ == 0)
{
lean_ctor_set(v___x_1442_, 1, v___x_1444_);
v___x_1446_ = v___x_1442_;
goto v_reusejp_1445_;
}
else
{
lean_object* v_reuseFailAlloc_1450_; 
v_reuseFailAlloc_1450_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1450_, 0, v_fst_1439_);
lean_ctor_set(v_reuseFailAlloc_1450_, 1, v___x_1444_);
v___x_1446_ = v_reuseFailAlloc_1450_;
goto v_reusejp_1445_;
}
v_reusejp_1445_:
{
lean_object* v___x_1448_; 
if (v_isShared_1438_ == 0)
{
lean_ctor_set(v___x_1437_, 0, v___x_1446_);
v___x_1448_ = v___x_1437_;
goto v_reusejp_1447_;
}
else
{
lean_object* v_reuseFailAlloc_1449_; 
v_reuseFailAlloc_1449_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1449_, 0, v___x_1446_);
v___x_1448_ = v_reuseFailAlloc_1449_;
goto v_reusejp_1447_;
}
v_reusejp_1447_:
{
return v___x_1448_;
}
}
}
}
}
else
{
lean_dec(v_val_1431_);
return v___x_1434_;
}
}
}
}
}
}
else
{
lean_object* v___x_1453_; uint8_t v___x_1454_; 
lean_dec_ref(v_str_1408_);
v___x_1453_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__6));
v___x_1454_ = lean_string_dec_eq(v_str_1407_, v___x_1453_);
lean_dec_ref(v_str_1407_);
if (v___x_1454_ == 0)
{
lean_object* v___x_1455_; 
lean_dec(v_snd_1406_);
v___x_1455_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1455_;
}
else
{
lean_object* v___x_1456_; lean_object* v___x_1457_; uint8_t v___x_1458_; 
v___x_1456_ = lean_array_get_size(v_snd_1406_);
v___x_1457_ = lean_unsigned_to_nat(3u);
v___x_1458_ = lean_nat_dec_eq(v___x_1456_, v___x_1457_);
if (v___x_1458_ == 0)
{
lean_object* v___x_1459_; 
lean_dec(v_snd_1406_);
v___x_1459_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1459_;
}
else
{
lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; 
lean_dec(v_a_1397_);
v___x_1460_ = lean_unsigned_to_nat(2u);
v___x_1461_ = lean_array_fget(v_snd_1406_, v___x_1460_);
lean_dec(v_snd_1406_);
v___x_1462_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_m_1389_, v___x_1461_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1462_) == 0)
{
lean_object* v_a_1463_; lean_object* v___x_1465_; uint8_t v_isShared_1466_; uint8_t v_isSharedCheck_1480_; 
v_a_1463_ = lean_ctor_get(v___x_1462_, 0);
v_isSharedCheck_1480_ = !lean_is_exclusive(v___x_1462_);
if (v_isSharedCheck_1480_ == 0)
{
v___x_1465_ = v___x_1462_;
v_isShared_1466_ = v_isSharedCheck_1480_;
goto v_resetjp_1464_;
}
else
{
lean_inc(v_a_1463_);
lean_dec(v___x_1462_);
v___x_1465_ = lean_box(0);
v_isShared_1466_ = v_isSharedCheck_1480_;
goto v_resetjp_1464_;
}
v_resetjp_1464_:
{
lean_object* v_fst_1467_; lean_object* v_snd_1468_; lean_object* v___x_1470_; uint8_t v_isShared_1471_; uint8_t v_isSharedCheck_1479_; 
v_fst_1467_ = lean_ctor_get(v_a_1463_, 0);
v_snd_1468_ = lean_ctor_get(v_a_1463_, 1);
v_isSharedCheck_1479_ = !lean_is_exclusive(v_a_1463_);
if (v_isSharedCheck_1479_ == 0)
{
v___x_1470_ = v_a_1463_;
v_isShared_1471_ = v_isSharedCheck_1479_;
goto v_resetjp_1469_;
}
else
{
lean_inc(v_snd_1468_);
lean_inc(v_fst_1467_);
lean_dec(v_a_1463_);
v___x_1470_ = lean_box(0);
v_isShared_1471_ = v_isSharedCheck_1479_;
goto v_resetjp_1469_;
}
v_resetjp_1469_:
{
lean_object* v___x_1472_; lean_object* v___x_1474_; 
v___x_1472_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__0(v_snd_1468_);
if (v_isShared_1471_ == 0)
{
lean_ctor_set(v___x_1470_, 1, v___x_1472_);
v___x_1474_ = v___x_1470_;
goto v_reusejp_1473_;
}
else
{
lean_object* v_reuseFailAlloc_1478_; 
v_reuseFailAlloc_1478_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1478_, 0, v_fst_1467_);
lean_ctor_set(v_reuseFailAlloc_1478_, 1, v___x_1472_);
v___x_1474_ = v_reuseFailAlloc_1478_;
goto v_reusejp_1473_;
}
v_reusejp_1473_:
{
lean_object* v___x_1476_; 
if (v_isShared_1466_ == 0)
{
lean_ctor_set(v___x_1465_, 0, v___x_1474_);
v___x_1476_ = v___x_1465_;
goto v_reusejp_1475_;
}
else
{
lean_object* v_reuseFailAlloc_1477_; 
v_reuseFailAlloc_1477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1477_, 0, v___x_1474_);
v___x_1476_ = v_reuseFailAlloc_1477_;
goto v_reusejp_1475_;
}
v_reusejp_1475_:
{
return v___x_1476_;
}
}
}
}
}
else
{
return v___x_1462_;
}
}
}
}
}
else
{
lean_object* v___x_1481_; uint8_t v___x_1482_; 
lean_dec_ref(v_str_1408_);
v___x_1481_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__7));
v___x_1482_ = lean_string_dec_eq(v_str_1407_, v___x_1481_);
lean_dec_ref(v_str_1407_);
if (v___x_1482_ == 0)
{
lean_object* v___x_1483_; 
lean_dec(v_snd_1406_);
v___x_1483_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1483_;
}
else
{
lean_object* v___x_1484_; lean_object* v___x_1485_; uint8_t v___x_1486_; 
v___x_1484_ = lean_array_get_size(v_snd_1406_);
v___x_1485_ = lean_unsigned_to_nat(6u);
v___x_1486_ = lean_nat_dec_eq(v___x_1484_, v___x_1485_);
if (v___x_1486_ == 0)
{
lean_object* v___x_1487_; 
lean_dec(v_snd_1406_);
v___x_1487_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1487_;
}
else
{
lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; 
lean_dec(v_a_1397_);
v___x_1488_ = lean_unsigned_to_nat(4u);
v___x_1489_ = lean_array_fget_borrowed(v_snd_1406_, v___x_1488_);
lean_inc(v___x_1489_);
v___x_1490_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_m_1389_, v___x_1489_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1490_) == 0)
{
lean_object* v_a_1491_; lean_object* v_fst_1492_; lean_object* v_snd_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; 
v_a_1491_ = lean_ctor_get(v___x_1490_, 0);
lean_inc(v_a_1491_);
lean_dec_ref_known(v___x_1490_, 1);
v_fst_1492_ = lean_ctor_get(v_a_1491_, 0);
lean_inc(v_fst_1492_);
v_snd_1493_ = lean_ctor_get(v_a_1491_, 1);
lean_inc(v_snd_1493_);
lean_dec(v_a_1491_);
v___x_1494_ = lean_unsigned_to_nat(5u);
v___x_1495_ = lean_array_fget(v_snd_1406_, v___x_1494_);
lean_dec(v_snd_1406_);
v___x_1496_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_fst_1492_, v___x_1495_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1496_) == 0)
{
lean_object* v_a_1497_; lean_object* v___x_1499_; uint8_t v_isShared_1500_; uint8_t v_isSharedCheck_1516_; 
v_a_1497_ = lean_ctor_get(v___x_1496_, 0);
v_isSharedCheck_1516_ = !lean_is_exclusive(v___x_1496_);
if (v_isSharedCheck_1516_ == 0)
{
v___x_1499_ = v___x_1496_;
v_isShared_1500_ = v_isSharedCheck_1516_;
goto v_resetjp_1498_;
}
else
{
lean_inc(v_a_1497_);
lean_dec(v___x_1496_);
v___x_1499_ = lean_box(0);
v_isShared_1500_ = v_isSharedCheck_1516_;
goto v_resetjp_1498_;
}
v_resetjp_1498_:
{
lean_object* v_fst_1501_; lean_object* v_snd_1502_; lean_object* v___x_1504_; uint8_t v_isShared_1505_; uint8_t v_isSharedCheck_1515_; 
v_fst_1501_ = lean_ctor_get(v_a_1497_, 0);
v_snd_1502_ = lean_ctor_get(v_a_1497_, 1);
v_isSharedCheck_1515_ = !lean_is_exclusive(v_a_1497_);
if (v_isSharedCheck_1515_ == 0)
{
v___x_1504_ = v_a_1497_;
v_isShared_1505_ = v_isSharedCheck_1515_;
goto v_resetjp_1503_;
}
else
{
lean_inc(v_snd_1502_);
lean_inc(v_fst_1501_);
lean_dec(v_a_1497_);
v___x_1504_ = lean_box(0);
v_isShared_1505_ = v_isSharedCheck_1515_;
goto v_resetjp_1503_;
}
v_resetjp_1503_:
{
lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1510_; 
v___x_1506_ = lp_mathlib_Std_DTreeMap_Internal_Impl_map___at___00Mathlib_Tactic_Linarith_linearFormOfExpr_spec__0(v_snd_1502_);
v___x_1507_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2_spec__2(v_snd_1493_, v___x_1506_);
v___x_1508_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v___x_1507_);
if (v_isShared_1505_ == 0)
{
lean_ctor_set(v___x_1504_, 1, v___x_1508_);
v___x_1510_ = v___x_1504_;
goto v_reusejp_1509_;
}
else
{
lean_object* v_reuseFailAlloc_1514_; 
v_reuseFailAlloc_1514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1514_, 0, v_fst_1501_);
lean_ctor_set(v_reuseFailAlloc_1514_, 1, v___x_1508_);
v___x_1510_ = v_reuseFailAlloc_1514_;
goto v_reusejp_1509_;
}
v_reusejp_1509_:
{
lean_object* v___x_1512_; 
if (v_isShared_1500_ == 0)
{
lean_ctor_set(v___x_1499_, 0, v___x_1510_);
v___x_1512_ = v___x_1499_;
goto v_reusejp_1511_;
}
else
{
lean_object* v_reuseFailAlloc_1513_; 
v_reuseFailAlloc_1513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1513_, 0, v___x_1510_);
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
}
else
{
lean_dec(v_snd_1493_);
return v___x_1496_;
}
}
else
{
lean_dec(v_snd_1406_);
return v___x_1490_;
}
}
}
}
}
else
{
lean_object* v___x_1517_; uint8_t v___x_1518_; 
lean_dec_ref(v_str_1408_);
v___x_1517_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__8));
v___x_1518_ = lean_string_dec_eq(v_str_1407_, v___x_1517_);
lean_dec_ref(v_str_1407_);
if (v___x_1518_ == 0)
{
lean_object* v___x_1519_; 
lean_dec(v_snd_1406_);
v___x_1519_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1519_;
}
else
{
lean_object* v___x_1520_; lean_object* v___x_1521_; uint8_t v___x_1522_; 
v___x_1520_ = lean_array_get_size(v_snd_1406_);
v___x_1521_ = lean_unsigned_to_nat(6u);
v___x_1522_ = lean_nat_dec_eq(v___x_1520_, v___x_1521_);
if (v___x_1522_ == 0)
{
lean_object* v___x_1523_; 
lean_dec(v_snd_1406_);
v___x_1523_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1523_;
}
else
{
lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; 
lean_dec(v_a_1397_);
v___x_1524_ = lean_unsigned_to_nat(4u);
v___x_1525_ = lean_array_fget_borrowed(v_snd_1406_, v___x_1524_);
lean_inc(v___x_1525_);
v___x_1526_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_m_1389_, v___x_1525_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1526_) == 0)
{
lean_object* v_a_1527_; lean_object* v_fst_1528_; lean_object* v_snd_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; 
v_a_1527_ = lean_ctor_get(v___x_1526_, 0);
lean_inc(v_a_1527_);
lean_dec_ref_known(v___x_1526_, 1);
v_fst_1528_ = lean_ctor_get(v_a_1527_, 0);
lean_inc(v_fst_1528_);
v_snd_1529_ = lean_ctor_get(v_a_1527_, 1);
lean_inc(v_snd_1529_);
lean_dec(v_a_1527_);
v___x_1530_ = lean_unsigned_to_nat(5u);
v___x_1531_ = lean_array_fget(v_snd_1406_, v___x_1530_);
lean_dec(v_snd_1406_);
v___x_1532_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_fst_1528_, v___x_1531_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1532_) == 0)
{
lean_object* v_a_1533_; lean_object* v___x_1535_; uint8_t v_isShared_1536_; uint8_t v_isSharedCheck_1551_; 
v_a_1533_ = lean_ctor_get(v___x_1532_, 0);
v_isSharedCheck_1551_ = !lean_is_exclusive(v___x_1532_);
if (v_isSharedCheck_1551_ == 0)
{
v___x_1535_ = v___x_1532_;
v_isShared_1536_ = v_isSharedCheck_1551_;
goto v_resetjp_1534_;
}
else
{
lean_inc(v_a_1533_);
lean_dec(v___x_1532_);
v___x_1535_ = lean_box(0);
v_isShared_1536_ = v_isSharedCheck_1551_;
goto v_resetjp_1534_;
}
v_resetjp_1534_:
{
lean_object* v_fst_1537_; lean_object* v_snd_1538_; lean_object* v___x_1540_; uint8_t v_isShared_1541_; uint8_t v_isSharedCheck_1550_; 
v_fst_1537_ = lean_ctor_get(v_a_1533_, 0);
v_snd_1538_ = lean_ctor_get(v_a_1533_, 1);
v_isSharedCheck_1550_ = !lean_is_exclusive(v_a_1533_);
if (v_isSharedCheck_1550_ == 0)
{
v___x_1540_ = v_a_1533_;
v_isShared_1541_ = v_isSharedCheck_1550_;
goto v_resetjp_1539_;
}
else
{
lean_inc(v_snd_1538_);
lean_inc(v_fst_1537_);
lean_dec(v_a_1533_);
v___x_1540_ = lean_box(0);
v_isShared_1541_ = v_isSharedCheck_1550_;
goto v_resetjp_1539_;
}
v_resetjp_1539_:
{
lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1545_; 
v___x_1542_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__2_spec__2(v_snd_1529_, v_snd_1538_);
v___x_1543_ = lp_mathlib_Std_DTreeMap_Internal_Impl_filter___at___00Mathlib_Tactic_Linarith_Sum_mul_spec__3___redArg(v___x_1542_);
if (v_isShared_1541_ == 0)
{
lean_ctor_set(v___x_1540_, 1, v___x_1543_);
v___x_1545_ = v___x_1540_;
goto v_reusejp_1544_;
}
else
{
lean_object* v_reuseFailAlloc_1549_; 
v_reuseFailAlloc_1549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1549_, 0, v_fst_1537_);
lean_ctor_set(v_reuseFailAlloc_1549_, 1, v___x_1543_);
v___x_1545_ = v_reuseFailAlloc_1549_;
goto v_reusejp_1544_;
}
v_reusejp_1544_:
{
lean_object* v___x_1547_; 
if (v_isShared_1536_ == 0)
{
lean_ctor_set(v___x_1535_, 0, v___x_1545_);
v___x_1547_ = v___x_1535_;
goto v_reusejp_1546_;
}
else
{
lean_object* v_reuseFailAlloc_1548_; 
v_reuseFailAlloc_1548_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1548_, 0, v___x_1545_);
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
else
{
lean_dec(v_snd_1529_);
return v___x_1532_;
}
}
else
{
lean_dec(v_snd_1406_);
return v___x_1526_;
}
}
}
}
}
else
{
lean_object* v___x_1552_; uint8_t v___x_1553_; 
lean_dec_ref(v_str_1408_);
v___x_1552_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___closed__9));
v___x_1553_ = lean_string_dec_eq(v_str_1407_, v___x_1552_);
lean_dec_ref(v_str_1407_);
if (v___x_1553_ == 0)
{
lean_object* v___x_1554_; 
lean_dec(v_snd_1406_);
v___x_1554_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1554_;
}
else
{
lean_object* v___x_1555_; lean_object* v___x_1556_; uint8_t v___x_1557_; 
v___x_1555_ = lean_array_get_size(v_snd_1406_);
v___x_1556_ = lean_unsigned_to_nat(6u);
v___x_1557_ = lean_nat_dec_eq(v___x_1555_, v___x_1556_);
if (v___x_1557_ == 0)
{
lean_object* v___x_1558_; 
lean_dec(v_snd_1406_);
v___x_1558_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1558_;
}
else
{
lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; 
lean_dec(v_a_1397_);
v___x_1559_ = lean_unsigned_to_nat(4u);
v___x_1560_ = lean_array_fget_borrowed(v_snd_1406_, v___x_1559_);
lean_inc(v___x_1560_);
v___x_1561_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_m_1389_, v___x_1560_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1561_) == 0)
{
lean_object* v_a_1562_; lean_object* v_fst_1563_; lean_object* v_snd_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; 
v_a_1562_ = lean_ctor_get(v___x_1561_, 0);
lean_inc(v_a_1562_);
lean_dec_ref_known(v___x_1561_, 1);
v_fst_1563_ = lean_ctor_get(v_a_1562_, 0);
lean_inc(v_fst_1563_);
v_snd_1564_ = lean_ctor_get(v_a_1562_, 1);
lean_inc(v_snd_1564_);
lean_dec(v_a_1562_);
v___x_1565_ = lean_unsigned_to_nat(5u);
v___x_1566_ = lean_array_fget(v_snd_1406_, v___x_1565_);
lean_dec(v_snd_1406_);
v___x_1567_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1388_, v_fst_1563_, v___x_1566_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
if (lean_obj_tag(v___x_1567_) == 0)
{
lean_object* v_a_1568_; lean_object* v___x_1570_; uint8_t v_isShared_1571_; uint8_t v_isSharedCheck_1585_; 
v_a_1568_ = lean_ctor_get(v___x_1567_, 0);
v_isSharedCheck_1585_ = !lean_is_exclusive(v___x_1567_);
if (v_isSharedCheck_1585_ == 0)
{
v___x_1570_ = v___x_1567_;
v_isShared_1571_ = v_isSharedCheck_1585_;
goto v_resetjp_1569_;
}
else
{
lean_inc(v_a_1568_);
lean_dec(v___x_1567_);
v___x_1570_ = lean_box(0);
v_isShared_1571_ = v_isSharedCheck_1585_;
goto v_resetjp_1569_;
}
v_resetjp_1569_:
{
lean_object* v_fst_1572_; lean_object* v_snd_1573_; lean_object* v___x_1575_; uint8_t v_isShared_1576_; uint8_t v_isSharedCheck_1584_; 
v_fst_1572_ = lean_ctor_get(v_a_1568_, 0);
v_snd_1573_ = lean_ctor_get(v_a_1568_, 1);
v_isSharedCheck_1584_ = !lean_is_exclusive(v_a_1568_);
if (v_isSharedCheck_1584_ == 0)
{
v___x_1575_ = v_a_1568_;
v_isShared_1576_ = v_isSharedCheck_1584_;
goto v_resetjp_1574_;
}
else
{
lean_inc(v_snd_1573_);
lean_inc(v_fst_1572_);
lean_dec(v_a_1568_);
v___x_1575_ = lean_box(0);
v_isShared_1576_ = v_isSharedCheck_1584_;
goto v_resetjp_1574_;
}
v_resetjp_1574_:
{
lean_object* v___x_1577_; lean_object* v___x_1579_; 
v___x_1577_ = lp_mathlib_Mathlib_Tactic_Linarith_Sum_mul(v_snd_1564_, v_snd_1573_);
if (v_isShared_1576_ == 0)
{
lean_ctor_set(v___x_1575_, 1, v___x_1577_);
v___x_1579_ = v___x_1575_;
goto v_reusejp_1578_;
}
else
{
lean_object* v_reuseFailAlloc_1583_; 
v_reuseFailAlloc_1583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1583_, 0, v_fst_1572_);
lean_ctor_set(v_reuseFailAlloc_1583_, 1, v___x_1577_);
v___x_1579_ = v_reuseFailAlloc_1583_;
goto v_reusejp_1578_;
}
v_reusejp_1578_:
{
lean_object* v___x_1581_; 
if (v_isShared_1571_ == 0)
{
lean_ctor_set(v___x_1570_, 0, v___x_1579_);
v___x_1581_ = v___x_1570_;
goto v_reusejp_1580_;
}
else
{
lean_object* v_reuseFailAlloc_1582_; 
v_reuseFailAlloc_1582_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1582_, 0, v___x_1579_);
v___x_1581_ = v_reuseFailAlloc_1582_;
goto v_reusejp_1580_;
}
v_reusejp_1580_:
{
return v___x_1581_;
}
}
}
}
}
else
{
lean_dec(v_snd_1564_);
return v___x_1567_;
}
}
else
{
lean_dec(v_snd_1406_);
return v___x_1561_;
}
}
}
}
}
else
{
lean_object* v___x_1586_; 
lean_dec_ref_known(v_pre_1404_, 2);
lean_dec_ref_known(v_fst_1403_, 2);
lean_dec_ref(v___x_1402_);
v___x_1586_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1586_;
}
}
else
{
lean_object* v___x_1587_; 
lean_dec(v_pre_1404_);
lean_dec_ref_known(v_fst_1403_, 2);
lean_dec_ref(v___x_1402_);
v___x_1587_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1587_;
}
}
else
{
lean_object* v___x_1588_; 
lean_dec(v_fst_1403_);
lean_dec_ref(v___x_1402_);
v___x_1588_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfAtom(v_red_1388_, v_m_1389_, v_a_1397_, v_a_1391_, v_a_1392_, v_a_1393_, v_a_1394_);
return v___x_1588_;
}
}
else
{
lean_object* v_val_1589_; lean_object* v_zero_1590_; uint8_t v_isZero_1591_; 
lean_dec(v_a_1397_);
v_val_1589_ = lean_ctor_get(v___x_1401_, 0);
lean_inc(v_val_1589_);
lean_dec_ref_known(v___x_1401_, 1);
v_zero_1590_ = lean_unsigned_to_nat(0u);
v_isZero_1591_ = lean_nat_dec_eq(v_val_1589_, v_zero_1590_);
if (v_isZero_1591_ == 1)
{
lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1595_; 
lean_dec(v_val_1589_);
v___x_1592_ = lean_box(1);
v___x_1593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1593_, 0, v_m_1389_);
lean_ctor_set(v___x_1593_, 1, v___x_1592_);
if (v_isShared_1400_ == 0)
{
lean_ctor_set(v___x_1399_, 0, v___x_1593_);
v___x_1595_ = v___x_1399_;
goto v_reusejp_1594_;
}
else
{
lean_object* v_reuseFailAlloc_1596_; 
v_reuseFailAlloc_1596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1596_, 0, v___x_1593_);
v___x_1595_ = v_reuseFailAlloc_1596_;
goto v_reusejp_1594_;
}
v_reusejp_1594_:
{
return v___x_1595_;
}
}
else
{
lean_object* v_one_1597_; lean_object* v_n_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1605_; 
v_one_1597_ = lean_unsigned_to_nat(1u);
v_n_1598_ = lean_nat_sub(v_val_1589_, v_one_1597_);
lean_dec(v_val_1589_);
v___x_1599_ = lean_nat_to_int(v_n_1598_);
v___x_1600_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0, &lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one___closed__0);
v___x_1601_ = lean_int_add(v___x_1599_, v___x_1600_);
lean_dec(v___x_1599_);
v___x_1602_ = lp_mathlib_Mathlib_Tactic_Linarith_scalar(v___x_1601_);
v___x_1603_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1603_, 0, v_m_1389_);
lean_ctor_set(v___x_1603_, 1, v___x_1602_);
if (v_isShared_1400_ == 0)
{
lean_ctor_set(v___x_1399_, 0, v___x_1603_);
v___x_1605_ = v___x_1399_;
goto v_reusejp_1604_;
}
else
{
lean_object* v_reuseFailAlloc_1606_; 
v_reuseFailAlloc_1606_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1606_, 0, v___x_1603_);
v___x_1605_ = v_reuseFailAlloc_1606_;
goto v_reusejp_1604_;
}
v_reusejp_1604_:
{
return v___x_1605_;
}
}
}
}
}
else
{
lean_object* v_a_1608_; lean_object* v___x_1610_; uint8_t v_isShared_1611_; uint8_t v_isSharedCheck_1615_; 
lean_dec(v_m_1389_);
v_a_1608_ = lean_ctor_get(v___x_1396_, 0);
v_isSharedCheck_1615_ = !lean_is_exclusive(v___x_1396_);
if (v_isSharedCheck_1615_ == 0)
{
v___x_1610_ = v___x_1396_;
v_isShared_1611_ = v_isSharedCheck_1615_;
goto v_resetjp_1609_;
}
else
{
lean_inc(v_a_1608_);
lean_dec(v___x_1396_);
v___x_1610_ = lean_box(0);
v_isShared_1611_ = v_isSharedCheck_1615_;
goto v_resetjp_1609_;
}
v_resetjp_1609_:
{
lean_object* v___x_1613_; 
if (v_isShared_1611_ == 0)
{
v___x_1613_ = v___x_1610_;
goto v_reusejp_1612_;
}
else
{
lean_object* v_reuseFailAlloc_1614_; 
v_reuseFailAlloc_1614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1614_, 0, v_a_1608_);
v___x_1613_ = v_reuseFailAlloc_1614_;
goto v_reusejp_1612_;
}
v_reusejp_1612_:
{
return v___x_1613_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr___boxed(lean_object* v_red_1616_, lean_object* v_m_1617_, lean_object* v_e_1618_, lean_object* v_a_1619_, lean_object* v_a_1620_, lean_object* v_a_1621_, lean_object* v_a_1622_, lean_object* v_a_1623_){
_start:
{
uint8_t v_red_boxed_1624_; lean_object* v_res_1625_; 
v_red_boxed_1624_ = lean_unbox(v_red_1616_);
v_res_1625_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_boxed_1624_, v_m_1617_, v_e_1618_, v_a_1619_, v_a_1620_, v_a_1621_, v_a_1622_);
lean_dec(v_a_1622_);
lean_dec_ref(v_a_1621_);
lean_dec(v_a_1620_);
lean_dec_ref(v_a_1619_);
return v_res_1625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_Linarith_elimMonom_spec__0___redArg(lean_object* v_t_1626_, lean_object* v_k_1627_){
_start:
{
if (lean_obj_tag(v_t_1626_) == 0)
{
lean_object* v_k_1628_; lean_object* v_v_1629_; lean_object* v_l_1630_; lean_object* v_r_1631_; uint8_t v___x_1632_; 
v_k_1628_ = lean_ctor_get(v_t_1626_, 1);
lean_inc(v_k_1628_);
v_v_1629_ = lean_ctor_get(v_t_1626_, 2);
lean_inc(v_v_1629_);
v_l_1630_ = lean_ctor_get(v_t_1626_, 3);
lean_inc(v_l_1630_);
v_r_1631_ = lean_ctor_get(v_t_1626_, 4);
lean_inc(v_r_1631_);
lean_dec_ref_known(v_t_1626_, 5);
v___x_1632_ = lp_mathlib_Mathlib_Tactic_Linarith_Monom_lt(v_k_1627_, v_k_1628_);
if (v___x_1632_ == 0)
{
lean_object* v___f_1633_; uint8_t v___x_1634_; 
lean_dec(v_l_1630_);
v___f_1633_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_instOrdMonom___closed__0));
lean_inc(v_k_1627_);
v___x_1634_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_beq___at___00Std_DTreeMap_Const_beq___at___00Std_TreeMap_beq___at___00Mathlib_Tactic_Linarith_Sum_one_spec__0_spec__0_spec__1___redArg(v___f_1633_, v_k_1627_, v_k_1628_);
if (v___x_1634_ == 0)
{
lean_dec(v_v_1629_);
v_t_1626_ = v_r_1631_;
goto _start;
}
else
{
lean_object* v___x_1636_; 
lean_dec(v_r_1631_);
lean_dec(v_k_1627_);
v___x_1636_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1636_, 0, v_v_1629_);
return v___x_1636_;
}
}
else
{
lean_dec(v_r_1631_);
lean_dec(v_v_1629_);
lean_dec(v_k_1628_);
v_t_1626_ = v_l_1630_;
goto _start;
}
}
else
{
lean_object* v___x_1638_; 
lean_dec(v_k_1627_);
v___x_1638_ = lean_box(0);
return v___x_1638_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_elimMonom_spec__1(lean_object* v_init_1639_, lean_object* v_x_1640_){
_start:
{
if (lean_obj_tag(v_x_1640_) == 0)
{
lean_object* v_k_1641_; lean_object* v_v_1642_; lean_object* v_l_1643_; lean_object* v_r_1644_; lean_object* v___x_1645_; lean_object* v_fst_1646_; lean_object* v_snd_1647_; lean_object* v___x_1649_; uint8_t v_isShared_1650_; uint8_t v_isSharedCheck_1666_; 
v_k_1641_ = lean_ctor_get(v_x_1640_, 1);
lean_inc(v_k_1641_);
v_v_1642_ = lean_ctor_get(v_x_1640_, 2);
lean_inc(v_v_1642_);
v_l_1643_ = lean_ctor_get(v_x_1640_, 3);
lean_inc(v_l_1643_);
v_r_1644_ = lean_ctor_get(v_x_1640_, 4);
lean_inc(v_r_1644_);
lean_dec_ref_known(v_x_1640_, 5);
v___x_1645_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_elimMonom_spec__1(v_init_1639_, v_r_1644_);
v_fst_1646_ = lean_ctor_get(v___x_1645_, 0);
v_snd_1647_ = lean_ctor_get(v___x_1645_, 1);
v_isSharedCheck_1666_ = !lean_is_exclusive(v___x_1645_);
if (v_isSharedCheck_1666_ == 0)
{
v___x_1649_ = v___x_1645_;
v_isShared_1650_ = v_isSharedCheck_1666_;
goto v_resetjp_1648_;
}
else
{
lean_inc(v_snd_1647_);
lean_inc(v_fst_1646_);
lean_dec(v___x_1645_);
v___x_1649_ = lean_box(0);
v_isShared_1650_ = v_isSharedCheck_1666_;
goto v_resetjp_1648_;
}
v_resetjp_1648_:
{
lean_object* v___y_1652_; lean_object* v___x_1659_; 
lean_inc(v_k_1641_);
lean_inc(v_fst_1646_);
v___x_1659_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_Linarith_elimMonom_spec__0___redArg(v_fst_1646_, v_k_1641_);
if (lean_obj_tag(v___x_1659_) == 0)
{
if (lean_obj_tag(v_fst_1646_) == 0)
{
lean_object* v_size_1660_; 
v_size_1660_ = lean_ctor_get(v_fst_1646_, 0);
lean_inc(v_size_1660_);
v___y_1652_ = v_size_1660_;
goto v___jp_1651_;
}
else
{
lean_object* v___x_1661_; 
v___x_1661_ = lean_unsigned_to_nat(0u);
v___y_1652_ = v___x_1661_;
goto v___jp_1651_;
}
}
else
{
lean_object* v_val_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; 
lean_del_object(v___x_1649_);
lean_dec(v_k_1641_);
v_val_1662_ = lean_ctor_get(v___x_1659_, 0);
lean_inc(v_val_1662_);
lean_dec_ref_known(v___x_1659_, 1);
v___x_1663_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(v_val_1662_, v_v_1642_, v_snd_1647_);
v___x_1664_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1664_, 0, v_fst_1646_);
lean_ctor_set(v___x_1664_, 1, v___x_1663_);
v_init_1639_ = v___x_1664_;
v_x_1640_ = v_l_1643_;
goto _start;
}
v___jp_1651_:
{
lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1656_; 
lean_inc(v___y_1652_);
v___x_1653_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_Sum_one_spec__1___redArg(v_k_1641_, v___y_1652_, v_fst_1646_);
v___x_1654_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_Linarith_var_spec__0___redArg(v___y_1652_, v_v_1642_, v_snd_1647_);
if (v_isShared_1650_ == 0)
{
lean_ctor_set(v___x_1649_, 1, v___x_1654_);
lean_ctor_set(v___x_1649_, 0, v___x_1653_);
v___x_1656_ = v___x_1649_;
goto v_reusejp_1655_;
}
else
{
lean_object* v_reuseFailAlloc_1658_; 
v_reuseFailAlloc_1658_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1658_, 0, v___x_1653_);
lean_ctor_set(v_reuseFailAlloc_1658_, 1, v___x_1654_);
v___x_1656_ = v_reuseFailAlloc_1658_;
goto v_reusejp_1655_;
}
v_reusejp_1655_:
{
v_init_1639_ = v___x_1656_;
v_x_1640_ = v_l_1643_;
goto _start;
}
}
}
}
else
{
return v_init_1639_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_elimMonom(lean_object* v_s_1667_, lean_object* v_m_1668_){
_start:
{
lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; 
v___x_1669_ = lean_box(1);
v___x_1670_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1670_, 0, v_m_1668_);
lean_ctor_set(v___x_1670_, 1, v___x_1669_);
v___x_1671_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_elimMonom_spec__1(v___x_1670_, v_s_1667_);
return v___x_1671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_Linarith_elimMonom_spec__0(lean_object* v_00_u03b4_1672_, lean_object* v_t_1673_, lean_object* v_k_1674_){
_start:
{
lean_object* v___x_1675_; 
v___x_1675_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_Linarith_elimMonom_spec__0___redArg(v_t_1673_, v_k_1674_);
return v___x_1675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_toComp_spec__0(lean_object* v_init_1676_, lean_object* v_x_1677_){
_start:
{
if (lean_obj_tag(v_x_1677_) == 0)
{
lean_object* v_k_1678_; lean_object* v_v_1679_; lean_object* v_l_1680_; lean_object* v_r_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; 
v_k_1678_ = lean_ctor_get(v_x_1677_, 1);
v_v_1679_ = lean_ctor_get(v_x_1677_, 2);
v_l_1680_ = lean_ctor_get(v_x_1677_, 3);
v_r_1681_ = lean_ctor_get(v_x_1677_, 4);
v___x_1682_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_toComp_spec__0(v_init_1676_, v_r_1681_);
lean_inc(v_v_1679_);
lean_inc(v_k_1678_);
v___x_1683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1683_, 0, v_k_1678_);
lean_ctor_set(v___x_1683_, 1, v_v_1679_);
v___x_1684_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1684_, 0, v___x_1683_);
lean_ctor_set(v___x_1684_, 1, v___x_1682_);
v_init_1676_ = v___x_1684_;
v_x_1677_ = v_l_1680_;
goto _start;
}
else
{
return v_init_1676_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_toComp_spec__0___boxed(lean_object* v_init_1686_, lean_object* v_x_1687_){
_start:
{
lean_object* v_res_1688_; 
v_res_1688_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_toComp_spec__0(v_init_1686_, v_x_1687_);
lean_dec(v_x_1687_);
return v_res_1688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toComp(uint8_t v_red_1689_, lean_object* v_e_1690_, lean_object* v_e__map_1691_, lean_object* v_monom__map_1692_, lean_object* v_a_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_){
_start:
{
lean_object* v___x_1698_; 
v___x_1698_ = lp_mathlib_Mathlib_Tactic_Linarith_parseCompAndExpr(v_e_1690_, v_a_1693_, v_a_1694_, v_a_1695_, v_a_1696_);
if (lean_obj_tag(v___x_1698_) == 0)
{
lean_object* v_a_1699_; lean_object* v_fst_1700_; lean_object* v_snd_1701_; lean_object* v___x_1702_; 
v_a_1699_ = lean_ctor_get(v___x_1698_, 0);
lean_inc(v_a_1699_);
lean_dec_ref_known(v___x_1698_, 1);
v_fst_1700_ = lean_ctor_get(v_a_1699_, 0);
lean_inc(v_fst_1700_);
v_snd_1701_ = lean_ctor_get(v_a_1699_, 1);
lean_inc(v_snd_1701_);
lean_dec(v_a_1699_);
v___x_1702_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormOfExpr(v_red_1689_, v_e__map_1691_, v_snd_1701_, v_a_1693_, v_a_1694_, v_a_1695_, v_a_1696_);
if (lean_obj_tag(v___x_1702_) == 0)
{
lean_object* v_a_1703_; lean_object* v___x_1705_; uint8_t v_isShared_1706_; uint8_t v_isSharedCheck_1734_; 
v_a_1703_ = lean_ctor_get(v___x_1702_, 0);
v_isSharedCheck_1734_ = !lean_is_exclusive(v___x_1702_);
if (v_isSharedCheck_1734_ == 0)
{
v___x_1705_ = v___x_1702_;
v_isShared_1706_ = v_isSharedCheck_1734_;
goto v_resetjp_1704_;
}
else
{
lean_inc(v_a_1703_);
lean_dec(v___x_1702_);
v___x_1705_ = lean_box(0);
v_isShared_1706_ = v_isSharedCheck_1734_;
goto v_resetjp_1704_;
}
v_resetjp_1704_:
{
lean_object* v_fst_1707_; lean_object* v_snd_1708_; lean_object* v___x_1710_; uint8_t v_isShared_1711_; uint8_t v_isSharedCheck_1733_; 
v_fst_1707_ = lean_ctor_get(v_a_1703_, 0);
v_snd_1708_ = lean_ctor_get(v_a_1703_, 1);
v_isSharedCheck_1733_ = !lean_is_exclusive(v_a_1703_);
if (v_isSharedCheck_1733_ == 0)
{
v___x_1710_ = v_a_1703_;
v_isShared_1711_ = v_isSharedCheck_1733_;
goto v_resetjp_1709_;
}
else
{
lean_inc(v_snd_1708_);
lean_inc(v_fst_1707_);
lean_dec(v_a_1703_);
v___x_1710_ = lean_box(0);
v_isShared_1711_ = v_isSharedCheck_1733_;
goto v_resetjp_1709_;
}
v_resetjp_1709_:
{
lean_object* v___x_1712_; lean_object* v_fst_1713_; lean_object* v_snd_1714_; lean_object* v___x_1716_; uint8_t v_isShared_1717_; uint8_t v_isSharedCheck_1732_; 
v___x_1712_ = lp_mathlib_Mathlib_Tactic_Linarith_elimMonom(v_snd_1708_, v_monom__map_1692_);
v_fst_1713_ = lean_ctor_get(v___x_1712_, 0);
v_snd_1714_ = lean_ctor_get(v___x_1712_, 1);
v_isSharedCheck_1732_ = !lean_is_exclusive(v___x_1712_);
if (v_isSharedCheck_1732_ == 0)
{
v___x_1716_ = v___x_1712_;
v_isShared_1717_ = v_isSharedCheck_1732_;
goto v_resetjp_1715_;
}
else
{
lean_inc(v_snd_1714_);
lean_inc(v_fst_1713_);
lean_dec(v___x_1712_);
v___x_1716_ = lean_box(0);
v_isShared_1717_ = v_isSharedCheck_1732_;
goto v_resetjp_1715_;
}
v_resetjp_1715_:
{
lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; uint8_t v___x_1722_; lean_object* v___x_1724_; 
v___x_1718_ = lean_box(0);
v___x_1719_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_toComp_spec__0(v___x_1718_, v_snd_1714_);
lean_dec(v_snd_1714_);
v___x_1720_ = l_List_reverse___redArg(v___x_1719_);
v___x_1721_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1721_, 0, v___x_1720_);
v___x_1722_ = lean_unbox(v_fst_1700_);
lean_dec(v_fst_1700_);
lean_ctor_set_uint8(v___x_1721_, sizeof(void*)*1, v___x_1722_);
if (v_isShared_1717_ == 0)
{
lean_ctor_set(v___x_1716_, 1, v_fst_1713_);
lean_ctor_set(v___x_1716_, 0, v_fst_1707_);
v___x_1724_ = v___x_1716_;
goto v_reusejp_1723_;
}
else
{
lean_object* v_reuseFailAlloc_1731_; 
v_reuseFailAlloc_1731_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1731_, 0, v_fst_1707_);
lean_ctor_set(v_reuseFailAlloc_1731_, 1, v_fst_1713_);
v___x_1724_ = v_reuseFailAlloc_1731_;
goto v_reusejp_1723_;
}
v_reusejp_1723_:
{
lean_object* v___x_1726_; 
if (v_isShared_1711_ == 0)
{
lean_ctor_set(v___x_1710_, 1, v___x_1724_);
lean_ctor_set(v___x_1710_, 0, v___x_1721_);
v___x_1726_ = v___x_1710_;
goto v_reusejp_1725_;
}
else
{
lean_object* v_reuseFailAlloc_1730_; 
v_reuseFailAlloc_1730_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1730_, 0, v___x_1721_);
lean_ctor_set(v_reuseFailAlloc_1730_, 1, v___x_1724_);
v___x_1726_ = v_reuseFailAlloc_1730_;
goto v_reusejp_1725_;
}
v_reusejp_1725_:
{
lean_object* v___x_1728_; 
if (v_isShared_1706_ == 0)
{
lean_ctor_set(v___x_1705_, 0, v___x_1726_);
v___x_1728_ = v___x_1705_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v___x_1726_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1735_; lean_object* v___x_1737_; uint8_t v_isShared_1738_; uint8_t v_isSharedCheck_1742_; 
lean_dec(v_fst_1700_);
lean_dec(v_monom__map_1692_);
v_a_1735_ = lean_ctor_get(v___x_1702_, 0);
v_isSharedCheck_1742_ = !lean_is_exclusive(v___x_1702_);
if (v_isSharedCheck_1742_ == 0)
{
v___x_1737_ = v___x_1702_;
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
else
{
lean_inc(v_a_1735_);
lean_dec(v___x_1702_);
v___x_1737_ = lean_box(0);
v_isShared_1738_ = v_isSharedCheck_1742_;
goto v_resetjp_1736_;
}
v_resetjp_1736_:
{
lean_object* v___x_1740_; 
if (v_isShared_1738_ == 0)
{
v___x_1740_ = v___x_1737_;
goto v_reusejp_1739_;
}
else
{
lean_object* v_reuseFailAlloc_1741_; 
v_reuseFailAlloc_1741_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1741_, 0, v_a_1735_);
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
lean_object* v_a_1743_; lean_object* v___x_1745_; uint8_t v_isShared_1746_; uint8_t v_isSharedCheck_1750_; 
lean_dec(v_monom__map_1692_);
lean_dec(v_e__map_1691_);
v_a_1743_ = lean_ctor_get(v___x_1698_, 0);
v_isSharedCheck_1750_ = !lean_is_exclusive(v___x_1698_);
if (v_isSharedCheck_1750_ == 0)
{
v___x_1745_ = v___x_1698_;
v_isShared_1746_ = v_isSharedCheck_1750_;
goto v_resetjp_1744_;
}
else
{
lean_inc(v_a_1743_);
lean_dec(v___x_1698_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toComp___boxed(lean_object* v_red_1751_, lean_object* v_e_1752_, lean_object* v_e__map_1753_, lean_object* v_monom__map_1754_, lean_object* v_a_1755_, lean_object* v_a_1756_, lean_object* v_a_1757_, lean_object* v_a_1758_, lean_object* v_a_1759_){
_start:
{
uint8_t v_red_boxed_1760_; lean_object* v_res_1761_; 
v_red_boxed_1760_ = lean_unbox(v_red_1751_);
v_res_1761_ = lp_mathlib_Mathlib_Tactic_Linarith_toComp(v_red_boxed_1760_, v_e_1752_, v_e__map_1753_, v_monom__map_1754_, v_a_1755_, v_a_1756_, v_a_1757_, v_a_1758_);
lean_dec(v_a_1758_);
lean_dec_ref(v_a_1757_);
lean_dec(v_a_1756_);
lean_dec_ref(v_a_1755_);
return v_res_1761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toCompFold(uint8_t v_red_1762_, lean_object* v_x_1763_, lean_object* v_x_1764_, lean_object* v_x_1765_, lean_object* v_a_1766_, lean_object* v_a_1767_, lean_object* v_a_1768_, lean_object* v_a_1769_){
_start:
{
if (lean_obj_tag(v_x_1764_) == 0)
{
lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; 
v___x_1771_ = lean_box(0);
v___x_1772_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1772_, 0, v_x_1763_);
lean_ctor_set(v___x_1772_, 1, v_x_1765_);
v___x_1773_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1773_, 0, v___x_1771_);
lean_ctor_set(v___x_1773_, 1, v___x_1772_);
v___x_1774_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1774_, 0, v___x_1773_);
return v___x_1774_;
}
else
{
lean_object* v_head_1775_; lean_object* v_tail_1776_; lean_object* v___x_1778_; uint8_t v_isShared_1779_; uint8_t v_isSharedCheck_1815_; 
v_head_1775_ = lean_ctor_get(v_x_1764_, 0);
v_tail_1776_ = lean_ctor_get(v_x_1764_, 1);
v_isSharedCheck_1815_ = !lean_is_exclusive(v_x_1764_);
if (v_isSharedCheck_1815_ == 0)
{
v___x_1778_ = v_x_1764_;
v_isShared_1779_ = v_isSharedCheck_1815_;
goto v_resetjp_1777_;
}
else
{
lean_inc(v_tail_1776_);
lean_inc(v_head_1775_);
lean_dec(v_x_1764_);
v___x_1778_ = lean_box(0);
v_isShared_1779_ = v_isSharedCheck_1815_;
goto v_resetjp_1777_;
}
v_resetjp_1777_:
{
lean_object* v___x_1780_; 
v___x_1780_ = lp_mathlib_Mathlib_Tactic_Linarith_toComp(v_red_1762_, v_head_1775_, v_x_1763_, v_x_1765_, v_a_1766_, v_a_1767_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_1780_) == 0)
{
lean_object* v_a_1781_; lean_object* v_snd_1782_; lean_object* v_fst_1783_; lean_object* v_fst_1784_; lean_object* v_snd_1785_; lean_object* v___x_1786_; 
v_a_1781_ = lean_ctor_get(v___x_1780_, 0);
lean_inc(v_a_1781_);
lean_dec_ref_known(v___x_1780_, 1);
v_snd_1782_ = lean_ctor_get(v_a_1781_, 1);
lean_inc(v_snd_1782_);
v_fst_1783_ = lean_ctor_get(v_a_1781_, 0);
lean_inc(v_fst_1783_);
lean_dec(v_a_1781_);
v_fst_1784_ = lean_ctor_get(v_snd_1782_, 0);
lean_inc(v_fst_1784_);
v_snd_1785_ = lean_ctor_get(v_snd_1782_, 1);
lean_inc(v_snd_1785_);
lean_dec(v_snd_1782_);
v___x_1786_ = lp_mathlib_Mathlib_Tactic_Linarith_toCompFold(v_red_1762_, v_fst_1784_, v_tail_1776_, v_snd_1785_, v_a_1766_, v_a_1767_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_1786_) == 0)
{
lean_object* v_a_1787_; lean_object* v___x_1789_; uint8_t v_isShared_1790_; uint8_t v_isSharedCheck_1806_; 
v_a_1787_ = lean_ctor_get(v___x_1786_, 0);
v_isSharedCheck_1806_ = !lean_is_exclusive(v___x_1786_);
if (v_isSharedCheck_1806_ == 0)
{
v___x_1789_ = v___x_1786_;
v_isShared_1790_ = v_isSharedCheck_1806_;
goto v_resetjp_1788_;
}
else
{
lean_inc(v_a_1787_);
lean_dec(v___x_1786_);
v___x_1789_ = lean_box(0);
v_isShared_1790_ = v_isSharedCheck_1806_;
goto v_resetjp_1788_;
}
v_resetjp_1788_:
{
lean_object* v_fst_1791_; lean_object* v_snd_1792_; lean_object* v___x_1794_; uint8_t v_isShared_1795_; uint8_t v_isSharedCheck_1805_; 
v_fst_1791_ = lean_ctor_get(v_a_1787_, 0);
v_snd_1792_ = lean_ctor_get(v_a_1787_, 1);
v_isSharedCheck_1805_ = !lean_is_exclusive(v_a_1787_);
if (v_isSharedCheck_1805_ == 0)
{
v___x_1794_ = v_a_1787_;
v_isShared_1795_ = v_isSharedCheck_1805_;
goto v_resetjp_1793_;
}
else
{
lean_inc(v_snd_1792_);
lean_inc(v_fst_1791_);
lean_dec(v_a_1787_);
v___x_1794_ = lean_box(0);
v_isShared_1795_ = v_isSharedCheck_1805_;
goto v_resetjp_1793_;
}
v_resetjp_1793_:
{
lean_object* v___x_1797_; 
if (v_isShared_1779_ == 0)
{
lean_ctor_set(v___x_1778_, 1, v_fst_1791_);
lean_ctor_set(v___x_1778_, 0, v_fst_1783_);
v___x_1797_ = v___x_1778_;
goto v_reusejp_1796_;
}
else
{
lean_object* v_reuseFailAlloc_1804_; 
v_reuseFailAlloc_1804_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1804_, 0, v_fst_1783_);
lean_ctor_set(v_reuseFailAlloc_1804_, 1, v_fst_1791_);
v___x_1797_ = v_reuseFailAlloc_1804_;
goto v_reusejp_1796_;
}
v_reusejp_1796_:
{
lean_object* v___x_1799_; 
if (v_isShared_1795_ == 0)
{
lean_ctor_set(v___x_1794_, 0, v___x_1797_);
v___x_1799_ = v___x_1794_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1803_; 
v_reuseFailAlloc_1803_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1803_, 0, v___x_1797_);
lean_ctor_set(v_reuseFailAlloc_1803_, 1, v_snd_1792_);
v___x_1799_ = v_reuseFailAlloc_1803_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
lean_object* v___x_1801_; 
if (v_isShared_1790_ == 0)
{
lean_ctor_set(v___x_1789_, 0, v___x_1799_);
v___x_1801_ = v___x_1789_;
goto v_reusejp_1800_;
}
else
{
lean_object* v_reuseFailAlloc_1802_; 
v_reuseFailAlloc_1802_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1802_, 0, v___x_1799_);
v___x_1801_ = v_reuseFailAlloc_1802_;
goto v_reusejp_1800_;
}
v_reusejp_1800_:
{
return v___x_1801_;
}
}
}
}
}
}
else
{
lean_dec(v_fst_1783_);
lean_del_object(v___x_1778_);
return v___x_1786_;
}
}
else
{
lean_object* v_a_1807_; lean_object* v___x_1809_; uint8_t v_isShared_1810_; uint8_t v_isSharedCheck_1814_; 
lean_del_object(v___x_1778_);
lean_dec(v_tail_1776_);
v_a_1807_ = lean_ctor_get(v___x_1780_, 0);
v_isSharedCheck_1814_ = !lean_is_exclusive(v___x_1780_);
if (v_isSharedCheck_1814_ == 0)
{
v___x_1809_ = v___x_1780_;
v_isShared_1810_ = v_isSharedCheck_1814_;
goto v_resetjp_1808_;
}
else
{
lean_inc(v_a_1807_);
lean_dec(v___x_1780_);
v___x_1809_ = lean_box(0);
v_isShared_1810_ = v_isSharedCheck_1814_;
goto v_resetjp_1808_;
}
v_resetjp_1808_:
{
lean_object* v___x_1812_; 
if (v_isShared_1810_ == 0)
{
v___x_1812_ = v___x_1809_;
goto v_reusejp_1811_;
}
else
{
lean_object* v_reuseFailAlloc_1813_; 
v_reuseFailAlloc_1813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1813_, 0, v_a_1807_);
v___x_1812_ = v_reuseFailAlloc_1813_;
goto v_reusejp_1811_;
}
v_reusejp_1811_:
{
return v___x_1812_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_toCompFold___boxed(lean_object* v_red_1816_, lean_object* v_x_1817_, lean_object* v_x_1818_, lean_object* v_x_1819_, lean_object* v_a_1820_, lean_object* v_a_1821_, lean_object* v_a_1822_, lean_object* v_a_1823_, lean_object* v_a_1824_){
_start:
{
uint8_t v_red_boxed_1825_; lean_object* v_res_1826_; 
v_red_boxed_1825_ = lean_unbox(v_red_1816_);
v_res_1826_ = lp_mathlib_Mathlib_Tactic_Linarith_toCompFold(v_red_boxed_1825_, v_x_1817_, v_x_1818_, v_x_1819_, v_a_1820_, v_a_1821_, v_a_1822_, v_a_1823_);
lean_dec(v_a_1823_);
lean_dec_ref(v_a_1822_);
lean_dec(v_a_1821_);
lean_dec_ref(v_a_1820_);
return v_res_1826_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__0(void){
_start:
{
lean_object* v___x_1827_; double v___x_1828_; 
v___x_1827_ = lean_unsigned_to_nat(0u);
v___x_1828_ = lean_float_of_nat(v___x_1827_);
return v___x_1828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6(lean_object* v_cls_1832_, lean_object* v_msg_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_, lean_object* v___y_1837_){
_start:
{
lean_object* v_ref_1839_; lean_object* v___x_1840_; lean_object* v_a_1841_; lean_object* v___x_1843_; uint8_t v_isShared_1844_; uint8_t v_isSharedCheck_1885_; 
v_ref_1839_ = lean_ctor_get(v___y_1836_, 5);
v___x_1840_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00List_findDefeq_spec__1_spec__1(v_msg_1833_, v___y_1834_, v___y_1835_, v___y_1836_, v___y_1837_);
v_a_1841_ = lean_ctor_get(v___x_1840_, 0);
v_isSharedCheck_1885_ = !lean_is_exclusive(v___x_1840_);
if (v_isSharedCheck_1885_ == 0)
{
v___x_1843_ = v___x_1840_;
v_isShared_1844_ = v_isSharedCheck_1885_;
goto v_resetjp_1842_;
}
else
{
lean_inc(v_a_1841_);
lean_dec(v___x_1840_);
v___x_1843_ = lean_box(0);
v_isShared_1844_ = v_isSharedCheck_1885_;
goto v_resetjp_1842_;
}
v_resetjp_1842_:
{
lean_object* v___x_1845_; lean_object* v_traceState_1846_; lean_object* v_env_1847_; lean_object* v_nextMacroScope_1848_; lean_object* v_ngen_1849_; lean_object* v_auxDeclNGen_1850_; lean_object* v_cache_1851_; lean_object* v_messages_1852_; lean_object* v_infoState_1853_; lean_object* v_snapshotTasks_1854_; lean_object* v___x_1856_; uint8_t v_isShared_1857_; uint8_t v_isSharedCheck_1884_; 
v___x_1845_ = lean_st_ref_take(v___y_1837_);
v_traceState_1846_ = lean_ctor_get(v___x_1845_, 4);
v_env_1847_ = lean_ctor_get(v___x_1845_, 0);
v_nextMacroScope_1848_ = lean_ctor_get(v___x_1845_, 1);
v_ngen_1849_ = lean_ctor_get(v___x_1845_, 2);
v_auxDeclNGen_1850_ = lean_ctor_get(v___x_1845_, 3);
v_cache_1851_ = lean_ctor_get(v___x_1845_, 5);
v_messages_1852_ = lean_ctor_get(v___x_1845_, 6);
v_infoState_1853_ = lean_ctor_get(v___x_1845_, 7);
v_snapshotTasks_1854_ = lean_ctor_get(v___x_1845_, 8);
v_isSharedCheck_1884_ = !lean_is_exclusive(v___x_1845_);
if (v_isSharedCheck_1884_ == 0)
{
v___x_1856_ = v___x_1845_;
v_isShared_1857_ = v_isSharedCheck_1884_;
goto v_resetjp_1855_;
}
else
{
lean_inc(v_snapshotTasks_1854_);
lean_inc(v_infoState_1853_);
lean_inc(v_messages_1852_);
lean_inc(v_cache_1851_);
lean_inc(v_traceState_1846_);
lean_inc(v_auxDeclNGen_1850_);
lean_inc(v_ngen_1849_);
lean_inc(v_nextMacroScope_1848_);
lean_inc(v_env_1847_);
lean_dec(v___x_1845_);
v___x_1856_ = lean_box(0);
v_isShared_1857_ = v_isSharedCheck_1884_;
goto v_resetjp_1855_;
}
v_resetjp_1855_:
{
uint64_t v_tid_1858_; lean_object* v_traces_1859_; lean_object* v___x_1861_; uint8_t v_isShared_1862_; uint8_t v_isSharedCheck_1883_; 
v_tid_1858_ = lean_ctor_get_uint64(v_traceState_1846_, sizeof(void*)*1);
v_traces_1859_ = lean_ctor_get(v_traceState_1846_, 0);
v_isSharedCheck_1883_ = !lean_is_exclusive(v_traceState_1846_);
if (v_isSharedCheck_1883_ == 0)
{
v___x_1861_ = v_traceState_1846_;
v_isShared_1862_ = v_isSharedCheck_1883_;
goto v_resetjp_1860_;
}
else
{
lean_inc(v_traces_1859_);
lean_dec(v_traceState_1846_);
v___x_1861_ = lean_box(0);
v_isShared_1862_ = v_isSharedCheck_1883_;
goto v_resetjp_1860_;
}
v_resetjp_1860_:
{
lean_object* v___x_1863_; double v___x_1864_; uint8_t v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1873_; 
v___x_1863_ = lean_box(0);
v___x_1864_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__0);
v___x_1865_ = 0;
v___x_1866_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__1));
v___x_1867_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1867_, 0, v_cls_1832_);
lean_ctor_set(v___x_1867_, 1, v___x_1863_);
lean_ctor_set(v___x_1867_, 2, v___x_1866_);
lean_ctor_set_float(v___x_1867_, sizeof(void*)*3, v___x_1864_);
lean_ctor_set_float(v___x_1867_, sizeof(void*)*3 + 8, v___x_1864_);
lean_ctor_set_uint8(v___x_1867_, sizeof(void*)*3 + 16, v___x_1865_);
v___x_1868_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___closed__2));
v___x_1869_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1869_, 0, v___x_1867_);
lean_ctor_set(v___x_1869_, 1, v_a_1841_);
lean_ctor_set(v___x_1869_, 2, v___x_1868_);
lean_inc(v_ref_1839_);
v___x_1870_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1870_, 0, v_ref_1839_);
lean_ctor_set(v___x_1870_, 1, v___x_1869_);
v___x_1871_ = l_Lean_PersistentArray_push___redArg(v_traces_1859_, v___x_1870_);
if (v_isShared_1862_ == 0)
{
lean_ctor_set(v___x_1861_, 0, v___x_1871_);
v___x_1873_ = v___x_1861_;
goto v_reusejp_1872_;
}
else
{
lean_object* v_reuseFailAlloc_1882_; 
v_reuseFailAlloc_1882_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1882_, 0, v___x_1871_);
lean_ctor_set_uint64(v_reuseFailAlloc_1882_, sizeof(void*)*1, v_tid_1858_);
v___x_1873_ = v_reuseFailAlloc_1882_;
goto v_reusejp_1872_;
}
v_reusejp_1872_:
{
lean_object* v___x_1875_; 
if (v_isShared_1857_ == 0)
{
lean_ctor_set(v___x_1856_, 4, v___x_1873_);
v___x_1875_ = v___x_1856_;
goto v_reusejp_1874_;
}
else
{
lean_object* v_reuseFailAlloc_1881_; 
v_reuseFailAlloc_1881_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1881_, 0, v_env_1847_);
lean_ctor_set(v_reuseFailAlloc_1881_, 1, v_nextMacroScope_1848_);
lean_ctor_set(v_reuseFailAlloc_1881_, 2, v_ngen_1849_);
lean_ctor_set(v_reuseFailAlloc_1881_, 3, v_auxDeclNGen_1850_);
lean_ctor_set(v_reuseFailAlloc_1881_, 4, v___x_1873_);
lean_ctor_set(v_reuseFailAlloc_1881_, 5, v_cache_1851_);
lean_ctor_set(v_reuseFailAlloc_1881_, 6, v_messages_1852_);
lean_ctor_set(v_reuseFailAlloc_1881_, 7, v_infoState_1853_);
lean_ctor_set(v_reuseFailAlloc_1881_, 8, v_snapshotTasks_1854_);
v___x_1875_ = v_reuseFailAlloc_1881_;
goto v_reusejp_1874_;
}
v_reusejp_1874_:
{
lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1879_; 
v___x_1876_ = lean_st_ref_set(v___y_1837_, v___x_1875_);
v___x_1877_ = lean_box(0);
if (v_isShared_1844_ == 0)
{
lean_ctor_set(v___x_1843_, 0, v___x_1877_);
v___x_1879_ = v___x_1843_;
goto v_reusejp_1878_;
}
else
{
lean_object* v_reuseFailAlloc_1880_; 
v_reuseFailAlloc_1880_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1880_, 0, v___x_1877_);
v___x_1879_ = v_reuseFailAlloc_1880_;
goto v_reusejp_1878_;
}
v_reusejp_1878_:
{
return v___x_1879_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6___boxed(lean_object* v_cls_1886_, lean_object* v_msg_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_){
_start:
{
lean_object* v_res_1893_; 
v_res_1893_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6(v_cls_1886_, v_msg_1887_, v___y_1888_, v___y_1889_, v___y_1890_, v___y_1891_);
lean_dec(v___y_1891_);
lean_dec_ref(v___y_1890_);
lean_dec(v___y_1889_);
lean_dec_ref(v___y_1888_);
return v_res_1893_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2(void){
_start:
{
lean_object* v___x_1897_; lean_object* v___x_1898_; 
v___x_1897_ = ((lean_object*)(lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__1));
v___x_1898_ = l_Lean_MessageData_ofFormat(v___x_1897_);
return v___x_1898_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1899_; lean_object* v___x_1900_; 
v___x_1899_ = lean_box(1);
v___x_1900_ = l_Lean_MessageData_ofFormat(v___x_1899_);
return v___x_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1(lean_object* v_a_1901_, lean_object* v_a_1902_){
_start:
{
if (lean_obj_tag(v_a_1901_) == 0)
{
lean_object* v___x_1903_; 
v___x_1903_ = l_List_reverse___redArg(v_a_1902_);
return v___x_1903_;
}
else
{
lean_object* v_head_1904_; lean_object* v_tail_1905_; lean_object* v___x_1907_; uint8_t v_isShared_1908_; uint8_t v_isSharedCheck_1933_; 
v_head_1904_ = lean_ctor_get(v_a_1901_, 0);
v_tail_1905_ = lean_ctor_get(v_a_1901_, 1);
v_isSharedCheck_1933_ = !lean_is_exclusive(v_a_1901_);
if (v_isSharedCheck_1933_ == 0)
{
v___x_1907_ = v_a_1901_;
v_isShared_1908_ = v_isSharedCheck_1933_;
goto v_resetjp_1906_;
}
else
{
lean_inc(v_tail_1905_);
lean_inc(v_head_1904_);
lean_dec(v_a_1901_);
v___x_1907_ = lean_box(0);
v_isShared_1908_ = v_isSharedCheck_1933_;
goto v_resetjp_1906_;
}
v_resetjp_1906_:
{
lean_object* v_fst_1909_; lean_object* v_snd_1910_; lean_object* v___x_1912_; uint8_t v_isShared_1913_; uint8_t v_isSharedCheck_1932_; 
v_fst_1909_ = lean_ctor_get(v_head_1904_, 0);
v_snd_1910_ = lean_ctor_get(v_head_1904_, 1);
v_isSharedCheck_1932_ = !lean_is_exclusive(v_head_1904_);
if (v_isSharedCheck_1932_ == 0)
{
v___x_1912_ = v_head_1904_;
v_isShared_1913_ = v_isSharedCheck_1932_;
goto v_resetjp_1911_;
}
else
{
lean_inc(v_snd_1910_);
lean_inc(v_fst_1909_);
lean_dec(v_head_1904_);
v___x_1912_ = lean_box(0);
v_isShared_1913_ = v_isSharedCheck_1932_;
goto v_resetjp_1911_;
}
v_resetjp_1911_:
{
lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1919_; 
v___x_1914_ = l_Nat_reprFast(v_fst_1909_);
v___x_1915_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1915_, 0, v___x_1914_);
v___x_1916_ = l_Lean_MessageData_ofFormat(v___x_1915_);
v___x_1917_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2, &lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2);
if (v_isShared_1913_ == 0)
{
lean_ctor_set_tag(v___x_1912_, 7);
lean_ctor_set(v___x_1912_, 1, v___x_1917_);
lean_ctor_set(v___x_1912_, 0, v___x_1916_);
v___x_1919_ = v___x_1912_;
goto v_reusejp_1918_;
}
else
{
lean_object* v_reuseFailAlloc_1931_; 
v_reuseFailAlloc_1931_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1931_, 0, v___x_1916_);
lean_ctor_set(v_reuseFailAlloc_1931_, 1, v___x_1917_);
v___x_1919_ = v_reuseFailAlloc_1931_;
goto v_reusejp_1918_;
}
v_reusejp_1918_:
{
lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1928_; 
v___x_1920_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3, &lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3);
v___x_1921_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1921_, 0, v___x_1919_);
lean_ctor_set(v___x_1921_, 1, v___x_1920_);
v___x_1922_ = l_Nat_reprFast(v_snd_1910_);
v___x_1923_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1923_, 0, v___x_1922_);
v___x_1924_ = l_Lean_MessageData_ofFormat(v___x_1923_);
v___x_1925_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1925_, 0, v___x_1921_);
lean_ctor_set(v___x_1925_, 1, v___x_1924_);
v___x_1926_ = l_Lean_MessageData_paren(v___x_1925_);
if (v_isShared_1908_ == 0)
{
lean_ctor_set(v___x_1907_, 1, v_a_1902_);
lean_ctor_set(v___x_1907_, 0, v___x_1926_);
v___x_1928_ = v___x_1907_;
goto v_reusejp_1927_;
}
else
{
lean_object* v_reuseFailAlloc_1930_; 
v_reuseFailAlloc_1930_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1930_, 0, v___x_1926_);
lean_ctor_set(v_reuseFailAlloc_1930_, 1, v_a_1902_);
v___x_1928_ = v_reuseFailAlloc_1930_;
goto v_reusejp_1927_;
}
v_reusejp_1927_:
{
v_a_1901_ = v_tail_1905_;
v_a_1902_ = v___x_1928_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__5(lean_object* v_a_1934_, lean_object* v_a_1935_){
_start:
{
if (lean_obj_tag(v_a_1934_) == 0)
{
lean_object* v___x_1936_; 
v___x_1936_ = l_List_reverse___redArg(v_a_1935_);
return v___x_1936_;
}
else
{
lean_object* v_head_1937_; lean_object* v_tail_1938_; lean_object* v___x_1940_; uint8_t v_isShared_1941_; uint8_t v_isSharedCheck_1966_; 
v_head_1937_ = lean_ctor_get(v_a_1934_, 0);
v_tail_1938_ = lean_ctor_get(v_a_1934_, 1);
v_isSharedCheck_1966_ = !lean_is_exclusive(v_a_1934_);
if (v_isSharedCheck_1966_ == 0)
{
v___x_1940_ = v_a_1934_;
v_isShared_1941_ = v_isSharedCheck_1966_;
goto v_resetjp_1939_;
}
else
{
lean_inc(v_tail_1938_);
lean_inc(v_head_1937_);
lean_dec(v_a_1934_);
v___x_1940_ = lean_box(0);
v_isShared_1941_ = v_isSharedCheck_1966_;
goto v_resetjp_1939_;
}
v_resetjp_1939_:
{
lean_object* v_fst_1942_; lean_object* v_snd_1943_; lean_object* v___x_1945_; uint8_t v_isShared_1946_; uint8_t v_isSharedCheck_1965_; 
v_fst_1942_ = lean_ctor_get(v_head_1937_, 0);
v_snd_1943_ = lean_ctor_get(v_head_1937_, 1);
v_isSharedCheck_1965_ = !lean_is_exclusive(v_head_1937_);
if (v_isSharedCheck_1965_ == 0)
{
v___x_1945_ = v_head_1937_;
v_isShared_1946_ = v_isSharedCheck_1965_;
goto v_resetjp_1944_;
}
else
{
lean_inc(v_snd_1943_);
lean_inc(v_fst_1942_);
lean_dec(v_head_1937_);
v___x_1945_ = lean_box(0);
v_isShared_1946_ = v_isSharedCheck_1965_;
goto v_resetjp_1944_;
}
v_resetjp_1944_:
{
lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1952_; 
v___x_1947_ = lean_box(0);
v___x_1948_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1(v_fst_1942_, v___x_1947_);
v___x_1949_ = l_Lean_MessageData_ofList(v___x_1948_);
v___x_1950_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2, &lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__2);
if (v_isShared_1946_ == 0)
{
lean_ctor_set_tag(v___x_1945_, 7);
lean_ctor_set(v___x_1945_, 1, v___x_1950_);
lean_ctor_set(v___x_1945_, 0, v___x_1949_);
v___x_1952_ = v___x_1945_;
goto v_reusejp_1951_;
}
else
{
lean_object* v_reuseFailAlloc_1964_; 
v_reuseFailAlloc_1964_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1964_, 0, v___x_1949_);
lean_ctor_set(v_reuseFailAlloc_1964_, 1, v___x_1950_);
v___x_1952_ = v_reuseFailAlloc_1964_;
goto v_reusejp_1951_;
}
v_reusejp_1951_:
{
lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1961_; 
v___x_1953_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3, &lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3_once, _init_lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__1___closed__3);
v___x_1954_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1954_, 0, v___x_1952_);
lean_ctor_set(v___x_1954_, 1, v___x_1953_);
v___x_1955_ = l_Nat_reprFast(v_snd_1943_);
v___x_1956_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1956_, 0, v___x_1955_);
v___x_1957_ = l_Lean_MessageData_ofFormat(v___x_1956_);
v___x_1958_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1958_, 0, v___x_1954_);
lean_ctor_set(v___x_1958_, 1, v___x_1957_);
v___x_1959_ = l_Lean_MessageData_paren(v___x_1958_);
if (v_isShared_1941_ == 0)
{
lean_ctor_set(v___x_1940_, 1, v_a_1935_);
lean_ctor_set(v___x_1940_, 0, v___x_1959_);
v___x_1961_ = v___x_1940_;
goto v_reusejp_1960_;
}
else
{
lean_object* v_reuseFailAlloc_1963_; 
v_reuseFailAlloc_1963_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1963_, 0, v___x_1959_);
lean_ctor_set(v_reuseFailAlloc_1963_, 1, v_a_1935_);
v___x_1961_ = v_reuseFailAlloc_1963_;
goto v_reusejp_1960_;
}
v_reusejp_1960_:
{
v_a_1934_ = v_tail_1938_;
v_a_1935_ = v___x_1961_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__0(lean_object* v_init_1967_, lean_object* v_x_1968_){
_start:
{
if (lean_obj_tag(v_x_1968_) == 0)
{
lean_object* v_k_1969_; lean_object* v_v_1970_; lean_object* v_l_1971_; lean_object* v_r_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; lean_object* v___x_1975_; 
v_k_1969_ = lean_ctor_get(v_x_1968_, 1);
v_v_1970_ = lean_ctor_get(v_x_1968_, 2);
v_l_1971_ = lean_ctor_get(v_x_1968_, 3);
v_r_1972_ = lean_ctor_get(v_x_1968_, 4);
v___x_1973_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__0(v_init_1967_, v_r_1972_);
lean_inc(v_v_1970_);
lean_inc(v_k_1969_);
v___x_1974_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1974_, 0, v_k_1969_);
lean_ctor_set(v___x_1974_, 1, v_v_1970_);
v___x_1975_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1975_, 0, v___x_1974_);
lean_ctor_set(v___x_1975_, 1, v___x_1973_);
v_init_1967_ = v___x_1975_;
v_x_1968_ = v_l_1971_;
goto _start;
}
else
{
return v_init_1967_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__0___boxed(lean_object* v_init_1977_, lean_object* v_x_1978_){
_start:
{
lean_object* v_res_1979_; 
v_res_1979_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__0(v_init_1977_, v_x_1978_);
lean_dec(v_x_1978_);
return v_res_1979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__4(lean_object* v_a_1980_, lean_object* v_a_1981_){
_start:
{
if (lean_obj_tag(v_a_1980_) == 0)
{
lean_object* v___x_1982_; 
v___x_1982_ = l_List_reverse___redArg(v_a_1981_);
return v___x_1982_;
}
else
{
lean_object* v_head_1983_; lean_object* v_tail_1984_; lean_object* v___x_1986_; uint8_t v_isShared_1987_; uint8_t v_isSharedCheck_2003_; 
v_head_1983_ = lean_ctor_get(v_a_1980_, 0);
v_tail_1984_ = lean_ctor_get(v_a_1980_, 1);
v_isSharedCheck_2003_ = !lean_is_exclusive(v_a_1980_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1986_ = v_a_1980_;
v_isShared_1987_ = v_isSharedCheck_2003_;
goto v_resetjp_1985_;
}
else
{
lean_inc(v_tail_1984_);
lean_inc(v_head_1983_);
lean_dec(v_a_1980_);
v___x_1986_ = lean_box(0);
v_isShared_1987_ = v_isSharedCheck_2003_;
goto v_resetjp_1985_;
}
v_resetjp_1985_:
{
lean_object* v_fst_1988_; lean_object* v_snd_1989_; lean_object* v___x_1991_; uint8_t v_isShared_1992_; uint8_t v_isSharedCheck_2002_; 
v_fst_1988_ = lean_ctor_get(v_head_1983_, 0);
v_snd_1989_ = lean_ctor_get(v_head_1983_, 1);
v_isSharedCheck_2002_ = !lean_is_exclusive(v_head_1983_);
if (v_isSharedCheck_2002_ == 0)
{
v___x_1991_ = v_head_1983_;
v_isShared_1992_ = v_isSharedCheck_2002_;
goto v_resetjp_1990_;
}
else
{
lean_inc(v_snd_1989_);
lean_inc(v_fst_1988_);
lean_dec(v_head_1983_);
v___x_1991_ = lean_box(0);
v_isShared_1992_ = v_isSharedCheck_2002_;
goto v_resetjp_1990_;
}
v_resetjp_1990_:
{
lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1996_; 
v___x_1993_ = lean_box(0);
v___x_1994_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__0(v___x_1993_, v_fst_1988_);
lean_dec(v_fst_1988_);
if (v_isShared_1992_ == 0)
{
lean_ctor_set(v___x_1991_, 0, v___x_1994_);
v___x_1996_ = v___x_1991_;
goto v_reusejp_1995_;
}
else
{
lean_object* v_reuseFailAlloc_2001_; 
v_reuseFailAlloc_2001_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2001_, 0, v___x_1994_);
lean_ctor_set(v_reuseFailAlloc_2001_, 1, v_snd_1989_);
v___x_1996_ = v_reuseFailAlloc_2001_;
goto v_reusejp_1995_;
}
v_reusejp_1995_:
{
lean_object* v___x_1998_; 
if (v_isShared_1987_ == 0)
{
lean_ctor_set(v___x_1986_, 1, v_a_1981_);
lean_ctor_set(v___x_1986_, 0, v___x_1996_);
v___x_1998_ = v___x_1986_;
goto v_reusejp_1997_;
}
else
{
lean_object* v_reuseFailAlloc_2000_; 
v_reuseFailAlloc_2000_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2000_, 0, v___x_1996_);
lean_ctor_set(v_reuseFailAlloc_2000_, 1, v_a_1981_);
v___x_1998_ = v_reuseFailAlloc_2000_;
goto v_reusejp_1997_;
}
v_reusejp_1997_:
{
v_a_1980_ = v_tail_1984_;
v_a_1981_ = v___x_1998_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__3(lean_object* v_init_2004_, lean_object* v_x_2005_){
_start:
{
if (lean_obj_tag(v_x_2005_) == 0)
{
lean_object* v_k_2006_; lean_object* v_v_2007_; lean_object* v_l_2008_; lean_object* v_r_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; lean_object* v___x_2012_; 
v_k_2006_ = lean_ctor_get(v_x_2005_, 1);
v_v_2007_ = lean_ctor_get(v_x_2005_, 2);
v_l_2008_ = lean_ctor_get(v_x_2005_, 3);
v_r_2009_ = lean_ctor_get(v_x_2005_, 4);
v___x_2010_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__3(v_init_2004_, v_r_2009_);
lean_inc(v_v_2007_);
lean_inc(v_k_2006_);
v___x_2011_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2011_, 0, v_k_2006_);
lean_ctor_set(v___x_2011_, 1, v_v_2007_);
v___x_2012_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2012_, 0, v___x_2011_);
lean_ctor_set(v___x_2012_, 1, v___x_2010_);
v_init_2004_ = v___x_2012_;
v_x_2005_ = v_l_2008_;
goto _start;
}
else
{
return v_init_2004_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__3___boxed(lean_object* v_init_2014_, lean_object* v_x_2015_){
_start:
{
lean_object* v_res_2016_; 
v_res_2016_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__3(v_init_2014_, v_x_2015_);
lean_dec(v_x_2015_);
return v_res_2016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__2(lean_object* v_x_2017_, lean_object* v_x_2018_, lean_object* v___y_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_){
_start:
{
if (lean_obj_tag(v_x_2017_) == 0)
{
lean_object* v___x_2024_; lean_object* v___x_2025_; 
v___x_2024_ = l_List_reverse___redArg(v_x_2018_);
v___x_2025_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2025_, 0, v___x_2024_);
return v___x_2025_;
}
else
{
lean_object* v_head_2026_; lean_object* v_tail_2027_; lean_object* v___x_2029_; uint8_t v_isShared_2030_; uint8_t v_isSharedCheck_2045_; 
v_head_2026_ = lean_ctor_get(v_x_2017_, 0);
v_tail_2027_ = lean_ctor_get(v_x_2017_, 1);
v_isSharedCheck_2045_ = !lean_is_exclusive(v_x_2017_);
if (v_isSharedCheck_2045_ == 0)
{
v___x_2029_ = v_x_2017_;
v_isShared_2030_ = v_isSharedCheck_2045_;
goto v_resetjp_2028_;
}
else
{
lean_inc(v_tail_2027_);
lean_inc(v_head_2026_);
lean_dec(v_x_2017_);
v___x_2029_ = lean_box(0);
v_isShared_2030_ = v_isSharedCheck_2045_;
goto v_resetjp_2028_;
}
v_resetjp_2028_:
{
lean_object* v___x_2031_; 
lean_inc(v___y_2022_);
lean_inc_ref(v___y_2021_);
lean_inc(v___y_2020_);
lean_inc_ref(v___y_2019_);
v___x_2031_ = lean_infer_type(v_head_2026_, v___y_2019_, v___y_2020_, v___y_2021_, v___y_2022_);
if (lean_obj_tag(v___x_2031_) == 0)
{
lean_object* v_a_2032_; lean_object* v___x_2034_; 
v_a_2032_ = lean_ctor_get(v___x_2031_, 0);
lean_inc(v_a_2032_);
lean_dec_ref_known(v___x_2031_, 1);
if (v_isShared_2030_ == 0)
{
lean_ctor_set(v___x_2029_, 1, v_x_2018_);
lean_ctor_set(v___x_2029_, 0, v_a_2032_);
v___x_2034_ = v___x_2029_;
goto v_reusejp_2033_;
}
else
{
lean_object* v_reuseFailAlloc_2036_; 
v_reuseFailAlloc_2036_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2036_, 0, v_a_2032_);
lean_ctor_set(v_reuseFailAlloc_2036_, 1, v_x_2018_);
v___x_2034_ = v_reuseFailAlloc_2036_;
goto v_reusejp_2033_;
}
v_reusejp_2033_:
{
v_x_2017_ = v_tail_2027_;
v_x_2018_ = v___x_2034_;
goto _start;
}
}
else
{
lean_object* v_a_2037_; lean_object* v___x_2039_; uint8_t v_isShared_2040_; uint8_t v_isSharedCheck_2044_; 
lean_del_object(v___x_2029_);
lean_dec(v_tail_2027_);
lean_dec(v_x_2018_);
v_a_2037_ = lean_ctor_get(v___x_2031_, 0);
v_isSharedCheck_2044_ = !lean_is_exclusive(v___x_2031_);
if (v_isSharedCheck_2044_ == 0)
{
v___x_2039_ = v___x_2031_;
v_isShared_2040_ = v_isSharedCheck_2044_;
goto v_resetjp_2038_;
}
else
{
lean_inc(v_a_2037_);
lean_dec(v___x_2031_);
v___x_2039_ = lean_box(0);
v_isShared_2040_ = v_isSharedCheck_2044_;
goto v_resetjp_2038_;
}
v_resetjp_2038_:
{
lean_object* v___x_2042_; 
if (v_isShared_2040_ == 0)
{
v___x_2042_ = v___x_2039_;
goto v_reusejp_2041_;
}
else
{
lean_object* v_reuseFailAlloc_2043_; 
v_reuseFailAlloc_2043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2043_, 0, v_a_2037_);
v___x_2042_ = v_reuseFailAlloc_2043_;
goto v_reusejp_2041_;
}
v_reusejp_2041_:
{
return v___x_2042_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__2___boxed(lean_object* v_x_2046_, lean_object* v_x_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_){
_start:
{
lean_object* v_res_2053_; 
v_res_2053_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__2(v_x_2046_, v_x_2047_, v___y_2048_, v___y_2049_, v___y_2050_, v___y_2051_);
lean_dec(v___y_2051_);
lean_dec_ref(v___y_2050_);
lean_dec(v___y_2049_);
lean_dec_ref(v___y_2048_);
return v_res_2053_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__5(void){
_start:
{
lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; 
v___x_2062_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__2));
v___x_2063_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__4));
v___x_2064_ = l_Lean_Name_append(v___x_2063_, v___x_2062_);
return v___x_2064_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__7(void){
_start:
{
lean_object* v___x_2066_; lean_object* v___x_2067_; 
v___x_2066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__6));
v___x_2067_ = l_Lean_stringToMessageData(v___x_2066_);
return v___x_2067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar(uint8_t v_red_2068_, lean_object* v_pfs_2069_, lean_object* v_a_2070_, lean_object* v_a_2071_, lean_object* v_a_2072_, lean_object* v_a_2073_){
_start:
{
lean_object* v___x_2075_; lean_object* v___x_2076_; 
v___x_2075_ = lean_box(0);
v___x_2076_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__2(v_pfs_2069_, v___x_2075_, v_a_2070_, v_a_2071_, v_a_2072_, v_a_2073_);
if (lean_obj_tag(v___x_2076_) == 0)
{
lean_object* v_a_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; 
v_a_2077_ = lean_ctor_get(v___x_2076_, 0);
lean_inc(v_a_2077_);
lean_dec_ref_known(v___x_2076_, 1);
v___x_2078_ = lean_box(1);
v___x_2079_ = lp_mathlib_Mathlib_Tactic_Linarith_toCompFold(v_red_2068_, v___x_2075_, v_a_2077_, v___x_2078_, v_a_2070_, v_a_2071_, v_a_2072_, v_a_2073_);
if (lean_obj_tag(v___x_2079_) == 0)
{
lean_object* v_a_2080_; lean_object* v___x_2082_; uint8_t v_isShared_2083_; uint8_t v_isSharedCheck_2132_; 
v_a_2080_ = lean_ctor_get(v___x_2079_, 0);
v_isSharedCheck_2132_ = !lean_is_exclusive(v___x_2079_);
if (v_isSharedCheck_2132_ == 0)
{
v___x_2082_ = v___x_2079_;
v_isShared_2083_ = v_isSharedCheck_2132_;
goto v_resetjp_2081_;
}
else
{
lean_inc(v_a_2080_);
lean_dec(v___x_2079_);
v___x_2082_ = lean_box(0);
v_isShared_2083_ = v_isSharedCheck_2132_;
goto v_resetjp_2081_;
}
v_resetjp_2081_:
{
lean_object* v_fst_2084_; lean_object* v_snd_2085_; lean_object* v___x_2087_; uint8_t v_isShared_2088_; uint8_t v_isSharedCheck_2131_; 
v_fst_2084_ = lean_ctor_get(v_a_2080_, 0);
v_snd_2085_ = lean_ctor_get(v_a_2080_, 1);
v_isSharedCheck_2131_ = !lean_is_exclusive(v_a_2080_);
if (v_isSharedCheck_2131_ == 0)
{
v___x_2087_ = v_a_2080_;
v_isShared_2088_ = v_isSharedCheck_2131_;
goto v_resetjp_2086_;
}
else
{
lean_inc(v_snd_2085_);
lean_inc(v_fst_2084_);
lean_dec(v_a_2080_);
v___x_2087_ = lean_box(0);
v_isShared_2088_ = v_isSharedCheck_2131_;
goto v_resetjp_2086_;
}
v_resetjp_2086_:
{
lean_object* v___y_2090_; lean_object* v_snd_2099_; lean_object* v___x_2101_; uint8_t v_isShared_2102_; uint8_t v_isSharedCheck_2129_; 
v_snd_2099_ = lean_ctor_get(v_snd_2085_, 1);
v_isSharedCheck_2129_ = !lean_is_exclusive(v_snd_2085_);
if (v_isSharedCheck_2129_ == 0)
{
lean_object* v_unused_2130_; 
v_unused_2130_ = lean_ctor_get(v_snd_2085_, 0);
lean_dec(v_unused_2130_);
v___x_2101_ = v_snd_2085_;
v_isShared_2102_ = v_isSharedCheck_2129_;
goto v_resetjp_2100_;
}
else
{
lean_inc(v_snd_2099_);
lean_dec(v_snd_2085_);
v___x_2101_ = lean_box(0);
v_isShared_2102_ = v_isSharedCheck_2129_;
goto v_resetjp_2100_;
}
v___jp_2089_:
{
lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2094_; 
v___x_2091_ = lean_unsigned_to_nat(1u);
v___x_2092_ = lean_nat_sub(v___y_2090_, v___x_2091_);
lean_dec(v___y_2090_);
if (v_isShared_2088_ == 0)
{
lean_ctor_set(v___x_2087_, 1, v___x_2092_);
v___x_2094_ = v___x_2087_;
goto v_reusejp_2093_;
}
else
{
lean_object* v_reuseFailAlloc_2098_; 
v_reuseFailAlloc_2098_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2098_, 0, v_fst_2084_);
lean_ctor_set(v_reuseFailAlloc_2098_, 1, v___x_2092_);
v___x_2094_ = v_reuseFailAlloc_2098_;
goto v_reusejp_2093_;
}
v_reusejp_2093_:
{
lean_object* v___x_2096_; 
if (v_isShared_2083_ == 0)
{
lean_ctor_set(v___x_2082_, 0, v___x_2094_);
v___x_2096_ = v___x_2082_;
goto v_reusejp_2095_;
}
else
{
lean_object* v_reuseFailAlloc_2097_; 
v_reuseFailAlloc_2097_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2097_, 0, v___x_2094_);
v___x_2096_ = v_reuseFailAlloc_2097_;
goto v_reusejp_2095_;
}
v_reusejp_2095_:
{
return v___x_2096_;
}
}
}
v_resetjp_2100_:
{
lean_object* v_options_2106_; uint8_t v_hasTrace_2107_; 
v_options_2106_ = lean_ctor_get(v_a_2072_, 2);
v_hasTrace_2107_ = lean_ctor_get_uint8(v_options_2106_, sizeof(void*)*1);
if (v_hasTrace_2107_ == 0)
{
lean_del_object(v___x_2101_);
goto v___jp_2103_;
}
else
{
lean_object* v_inheritedTraceOptions_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; uint8_t v___x_2111_; 
v_inheritedTraceOptions_2108_ = lean_ctor_get(v_a_2072_, 13);
v___x_2109_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__2));
v___x_2110_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__5, &lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__5);
v___x_2111_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2108_, v_options_2106_, v___x_2110_);
if (v___x_2111_ == 0)
{
lean_del_object(v___x_2101_);
goto v___jp_2103_;
}
else
{
lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; lean_object* v___x_2118_; 
v___x_2112_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__7, &lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___closed__7);
v___x_2113_ = lp_mathlib_Std_DTreeMap_Internal_Impl_foldrM___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__3(v___x_2075_, v_snd_2099_);
v___x_2114_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__4(v___x_2113_, v___x_2075_);
v___x_2115_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__5(v___x_2114_, v___x_2075_);
v___x_2116_ = l_Lean_MessageData_ofList(v___x_2115_);
if (v_isShared_2102_ == 0)
{
lean_ctor_set_tag(v___x_2101_, 7);
lean_ctor_set(v___x_2101_, 1, v___x_2116_);
lean_ctor_set(v___x_2101_, 0, v___x_2112_);
v___x_2118_ = v___x_2101_;
goto v_reusejp_2117_;
}
else
{
lean_object* v_reuseFailAlloc_2128_; 
v_reuseFailAlloc_2128_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2128_, 0, v___x_2112_);
lean_ctor_set(v_reuseFailAlloc_2128_, 1, v___x_2116_);
v___x_2118_ = v_reuseFailAlloc_2128_;
goto v_reusejp_2117_;
}
v_reusejp_2117_:
{
lean_object* v___x_2119_; 
v___x_2119_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Linarith_linearFormsAndMaxVar_spec__6(v___x_2109_, v___x_2118_, v_a_2070_, v_a_2071_, v_a_2072_, v_a_2073_);
if (lean_obj_tag(v___x_2119_) == 0)
{
lean_dec_ref_known(v___x_2119_, 1);
goto v___jp_2103_;
}
else
{
lean_object* v_a_2120_; lean_object* v___x_2122_; uint8_t v_isShared_2123_; uint8_t v_isSharedCheck_2127_; 
lean_dec(v_snd_2099_);
lean_del_object(v___x_2087_);
lean_dec(v_fst_2084_);
lean_del_object(v___x_2082_);
v_a_2120_ = lean_ctor_get(v___x_2119_, 0);
v_isSharedCheck_2127_ = !lean_is_exclusive(v___x_2119_);
if (v_isSharedCheck_2127_ == 0)
{
v___x_2122_ = v___x_2119_;
v_isShared_2123_ = v_isSharedCheck_2127_;
goto v_resetjp_2121_;
}
else
{
lean_inc(v_a_2120_);
lean_dec(v___x_2119_);
v___x_2122_ = lean_box(0);
v_isShared_2123_ = v_isSharedCheck_2127_;
goto v_resetjp_2121_;
}
v_resetjp_2121_:
{
lean_object* v___x_2125_; 
if (v_isShared_2123_ == 0)
{
v___x_2125_ = v___x_2122_;
goto v_reusejp_2124_;
}
else
{
lean_object* v_reuseFailAlloc_2126_; 
v_reuseFailAlloc_2126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2126_, 0, v_a_2120_);
v___x_2125_ = v_reuseFailAlloc_2126_;
goto v_reusejp_2124_;
}
v_reusejp_2124_:
{
return v___x_2125_;
}
}
}
}
}
}
v___jp_2103_:
{
if (lean_obj_tag(v_snd_2099_) == 0)
{
lean_object* v_size_2104_; 
v_size_2104_ = lean_ctor_get(v_snd_2099_, 0);
lean_inc(v_size_2104_);
lean_dec_ref_known(v_snd_2099_, 5);
v___y_2090_ = v_size_2104_;
goto v___jp_2089_;
}
else
{
lean_object* v___x_2105_; 
v___x_2105_ = lean_unsigned_to_nat(0u);
v___y_2090_ = v___x_2105_;
goto v___jp_2089_;
}
}
}
}
}
}
else
{
lean_object* v_a_2133_; lean_object* v___x_2135_; uint8_t v_isShared_2136_; uint8_t v_isSharedCheck_2140_; 
v_a_2133_ = lean_ctor_get(v___x_2079_, 0);
v_isSharedCheck_2140_ = !lean_is_exclusive(v___x_2079_);
if (v_isSharedCheck_2140_ == 0)
{
v___x_2135_ = v___x_2079_;
v_isShared_2136_ = v_isSharedCheck_2140_;
goto v_resetjp_2134_;
}
else
{
lean_inc(v_a_2133_);
lean_dec(v___x_2079_);
v___x_2135_ = lean_box(0);
v_isShared_2136_ = v_isSharedCheck_2140_;
goto v_resetjp_2134_;
}
v_resetjp_2134_:
{
lean_object* v___x_2138_; 
if (v_isShared_2136_ == 0)
{
v___x_2138_ = v___x_2135_;
goto v_reusejp_2137_;
}
else
{
lean_object* v_reuseFailAlloc_2139_; 
v_reuseFailAlloc_2139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2139_, 0, v_a_2133_);
v___x_2138_ = v_reuseFailAlloc_2139_;
goto v_reusejp_2137_;
}
v_reusejp_2137_:
{
return v___x_2138_;
}
}
}
}
else
{
lean_object* v_a_2141_; lean_object* v___x_2143_; uint8_t v_isShared_2144_; uint8_t v_isSharedCheck_2148_; 
v_a_2141_ = lean_ctor_get(v___x_2076_, 0);
v_isSharedCheck_2148_ = !lean_is_exclusive(v___x_2076_);
if (v_isSharedCheck_2148_ == 0)
{
v___x_2143_ = v___x_2076_;
v_isShared_2144_ = v_isSharedCheck_2148_;
goto v_resetjp_2142_;
}
else
{
lean_inc(v_a_2141_);
lean_dec(v___x_2076_);
v___x_2143_ = lean_box(0);
v_isShared_2144_ = v_isSharedCheck_2148_;
goto v_resetjp_2142_;
}
v_resetjp_2142_:
{
lean_object* v___x_2146_; 
if (v_isShared_2144_ == 0)
{
v___x_2146_ = v___x_2143_;
goto v_reusejp_2145_;
}
else
{
lean_object* v_reuseFailAlloc_2147_; 
v_reuseFailAlloc_2147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2147_, 0, v_a_2141_);
v___x_2146_ = v_reuseFailAlloc_2147_;
goto v_reusejp_2145_;
}
v_reusejp_2145_:
{
return v___x_2146_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar___boxed(lean_object* v_red_2149_, lean_object* v_pfs_2150_, lean_object* v_a_2151_, lean_object* v_a_2152_, lean_object* v_a_2153_, lean_object* v_a_2154_, lean_object* v_a_2155_){
_start:
{
uint8_t v_red_boxed_2156_; lean_object* v_res_2157_; 
v_red_boxed_2156_ = lean_unbox(v_red_2149_);
v_res_2157_ = lp_mathlib_Mathlib_Tactic_Linarith_linearFormsAndMaxVar(v_red_boxed_2156_, v_pfs_2150_, v_a_2151_, v_a_2152_, v_a_2153_, v_a_2154_);
lean_dec(v_a_2154_);
lean_dec_ref(v_a_2153_);
lean_dec(v_a_2152_);
lean_dec_ref(v_a_2151_);
return v_res_2157_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Linarith_Monom_one = _init_lp_mathlib_Mathlib_Tactic_Linarith_Monom_one();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Linarith_Monom_one);
lp_mathlib_Mathlib_Tactic_Linarith_Sum_one = _init_lp_mathlib_Mathlib_Tactic_Linarith_Sum_one();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Linarith_Sum_one);
lp_mathlib_Mathlib_Tactic_Linarith_one = _init_lp_mathlib_Mathlib_Tactic_Linarith_one();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Linarith_one);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linarith_Datatypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linarith_Parsing(builtin);
}
#ifdef __cplusplus
}
#endif
