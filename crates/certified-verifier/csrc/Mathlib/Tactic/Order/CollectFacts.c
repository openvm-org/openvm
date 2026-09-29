// Lean compiler output
// Module: Mathlib.Tactic.Order.CollectFacts
// Imports: public import Init public meta import Init public import Mathlib.Order.BoundedOrder.Basic public import Mathlib.Order.Lattice public meta import Mathlib.Tactic.ToDual public import Mathlib.Util.AtomM
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
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_Qq_Qq_inferTypeQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
size_t lean_array_size(lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_eq_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_eq_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ne_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ne_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_le_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_le_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nle_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nle_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_lt_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_lt_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nlt_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nlt_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isTop_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isTop_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isBot_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isBot_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isInf_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isInf_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isSup_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isSup_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " = #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≠ #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≤ #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 3, .m_data = "¬ #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " < #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " := ⊤"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " := ⊥"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " := #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ⊓ #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ⊔ #"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Order_addType_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Order_addType_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Order_addType_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Order_addType_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_Order_addType___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addType___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addType___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_Const_modify___at___00Std_DHashMap_Internal_Raw_u2080_Const_modify___at___00Mathlib_Tactic_Order_addFact_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_modify___at___00Mathlib_Tactic_Order_addFact_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "OrderTop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 115, 10, 82, 184, 170, 227, 111)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Top"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "top"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(17, 209, 230, 57, 51, 197, 162, 233)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(101, 62, 44, 17, 165, 201, 212, 212)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toTop"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 115, 10, 82, 184, 170, 227, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(208, 68, 117, 122, 159, 191, 199, 143)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "OrderBot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(138, 76, 152, 81, 44, 99, 224, 67)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Bot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "bot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(192, 138, 190, 95, 247, 78, 16, 101)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(98, 132, 46, 181, 27, 87, 250, 96)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toBot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(138, 76, 152, 81, 44, 99, 224, 67)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(183, 116, 107, 42, 167, 233, 160, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Max"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "max"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(169, 95, 226, 81, 206, 208, 89, 76)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(247, 27, 157, 195, 66, 157, 90, 150)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMax"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Min"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "min"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(132, 99, 105, 121, 176, 241, 22, 117)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(0, 174, 129, 224, 94, 80, 42, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMin"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SemilatticeInf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 131, 181, 193, 54, 206, 77, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "SemilatticeSup"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__2_value),LEAN_SCALAR_PTR_LITERAL(208, 106, 75, 248, 165, 223, 103, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__4_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__4_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(71, 235, 154, 184, 62, 135, 30, 248)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(54, 235, 251, 9, 4, 74, 57, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Ne"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 247, 70, 70, 118, 145, 235, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__1_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__7(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__8(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "choose_spec"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__1_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(164, 197, 253, 65, 182, 223, 203, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "left"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(12, 252, 227, 83, 88, 185, 40, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__5;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "right"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__7_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__6_value),LEAN_SCALAR_PTR_LITERAL(18, 204, 165, 192, 253, 41, 237, 145)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Preorder"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__9_value),LEAN_SCALAR_PTR_LITERAL(171, 85, 2, 192, 23, 244, 204, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_collectFactsImp_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_collectFactsImp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFactsImp(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFactsImp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFacts(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFacts___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorIdx(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
case 3:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
case 4:
{
lean_object* v___x_6_; 
v___x_6_ = lean_unsigned_to_nat(4u);
return v___x_6_;
}
case 5:
{
lean_object* v___x_7_; 
v___x_7_ = lean_unsigned_to_nat(5u);
return v___x_7_;
}
case 6:
{
lean_object* v___x_8_; 
v___x_8_ = lean_unsigned_to_nat(6u);
return v___x_8_;
}
case 7:
{
lean_object* v___x_9_; 
v___x_9_ = lean_unsigned_to_nat(7u);
return v___x_9_;
}
case 8:
{
lean_object* v___x_10_; 
v___x_10_ = lean_unsigned_to_nat(8u);
return v___x_10_;
}
default: 
{
lean_object* v___x_11_; 
v___x_11_ = lean_unsigned_to_nat(9u);
return v___x_11_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorIdx___boxed(lean_object* v_x_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorIdx(v_x_12_);
lean_dec_ref(v_x_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(lean_object* v_t_14_, lean_object* v_k_15_){
_start:
{
switch(lean_obj_tag(v_t_14_))
{
case 6:
{
lean_object* v_idx_16_; lean_object* v___x_17_; 
v_idx_16_ = lean_ctor_get(v_t_14_, 0);
lean_inc(v_idx_16_);
lean_dec_ref_known(v_t_14_, 1);
v___x_17_ = lean_apply_1(v_k_15_, v_idx_16_);
return v___x_17_;
}
case 7:
{
lean_object* v_idx_18_; lean_object* v___x_19_; 
v_idx_18_ = lean_ctor_get(v_t_14_, 0);
lean_inc(v_idx_18_);
lean_dec_ref_known(v_t_14_, 1);
v___x_19_ = lean_apply_1(v_k_15_, v_idx_18_);
return v___x_19_;
}
case 8:
{
lean_object* v_lhs_20_; lean_object* v_rhs_21_; lean_object* v_res_22_; lean_object* v___x_23_; 
v_lhs_20_ = lean_ctor_get(v_t_14_, 0);
lean_inc(v_lhs_20_);
v_rhs_21_ = lean_ctor_get(v_t_14_, 1);
lean_inc(v_rhs_21_);
v_res_22_ = lean_ctor_get(v_t_14_, 2);
lean_inc(v_res_22_);
lean_dec_ref_known(v_t_14_, 3);
v___x_23_ = lean_apply_3(v_k_15_, v_lhs_20_, v_rhs_21_, v_res_22_);
return v___x_23_;
}
case 9:
{
lean_object* v_lhs_24_; lean_object* v_rhs_25_; lean_object* v_res_26_; lean_object* v___x_27_; 
v_lhs_24_ = lean_ctor_get(v_t_14_, 0);
lean_inc(v_lhs_24_);
v_rhs_25_ = lean_ctor_get(v_t_14_, 1);
lean_inc(v_rhs_25_);
v_res_26_ = lean_ctor_get(v_t_14_, 2);
lean_inc(v_res_26_);
lean_dec_ref_known(v_t_14_, 3);
v___x_27_ = lean_apply_3(v_k_15_, v_lhs_24_, v_rhs_25_, v_res_26_);
return v___x_27_;
}
default: 
{
lean_object* v_lhs_28_; lean_object* v_rhs_29_; lean_object* v_proof_30_; lean_object* v___x_31_; 
v_lhs_28_ = lean_ctor_get(v_t_14_, 0);
lean_inc(v_lhs_28_);
v_rhs_29_ = lean_ctor_get(v_t_14_, 1);
lean_inc(v_rhs_29_);
v_proof_30_ = lean_ctor_get(v_t_14_, 2);
lean_inc_ref(v_proof_30_);
lean_dec_ref(v_t_14_);
v___x_31_ = lean_apply_3(v_k_15_, v_lhs_28_, v_rhs_29_, v_proof_30_);
return v___x_31_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim(lean_object* v_motive_32_, lean_object* v_ctorIdx_33_, lean_object* v_t_34_, lean_object* v_h_35_, lean_object* v_k_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_34_, v_k_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___boxed(lean_object* v_motive_38_, lean_object* v_ctorIdx_39_, lean_object* v_t_40_, lean_object* v_h_41_, lean_object* v_k_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim(v_motive_38_, v_ctorIdx_39_, v_t_40_, v_h_41_, v_k_42_);
lean_dec(v_ctorIdx_39_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_eq_elim___redArg(lean_object* v_t_44_, lean_object* v_eq_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_44_, v_eq_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_eq_elim(lean_object* v_motive_47_, lean_object* v_t_48_, lean_object* v_h_49_, lean_object* v_eq_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_48_, v_eq_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ne_elim___redArg(lean_object* v_t_52_, lean_object* v_ne_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_52_, v_ne_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ne_elim(lean_object* v_motive_55_, lean_object* v_t_56_, lean_object* v_h_57_, lean_object* v_ne_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_56_, v_ne_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_le_elim___redArg(lean_object* v_t_60_, lean_object* v_le_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_60_, v_le_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_le_elim(lean_object* v_motive_63_, lean_object* v_t_64_, lean_object* v_h_65_, lean_object* v_le_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_64_, v_le_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nle_elim___redArg(lean_object* v_t_68_, lean_object* v_nle_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_68_, v_nle_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nle_elim(lean_object* v_motive_71_, lean_object* v_t_72_, lean_object* v_h_73_, lean_object* v_nle_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_72_, v_nle_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_lt_elim___redArg(lean_object* v_t_76_, lean_object* v_lt_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_76_, v_lt_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_lt_elim(lean_object* v_motive_79_, lean_object* v_t_80_, lean_object* v_h_81_, lean_object* v_lt_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_80_, v_lt_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nlt_elim___redArg(lean_object* v_t_84_, lean_object* v_nlt_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_84_, v_nlt_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_nlt_elim(lean_object* v_motive_87_, lean_object* v_t_88_, lean_object* v_h_89_, lean_object* v_nlt_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_88_, v_nlt_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isTop_elim___redArg(lean_object* v_t_92_, lean_object* v_isTop_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_92_, v_isTop_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isTop_elim(lean_object* v_motive_95_, lean_object* v_t_96_, lean_object* v_h_97_, lean_object* v_isTop_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_96_, v_isTop_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isBot_elim___redArg(lean_object* v_t_100_, lean_object* v_isBot_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_100_, v_isBot_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isBot_elim(lean_object* v_motive_103_, lean_object* v_t_104_, lean_object* v_h_105_, lean_object* v_isBot_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_104_, v_isBot_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isInf_elim___redArg(lean_object* v_t_108_, lean_object* v_isInf_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_108_, v_isInf_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isInf_elim(lean_object* v_motive_111_, lean_object* v_t_112_, lean_object* v_h_113_, lean_object* v_isInf_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_112_, v_isInf_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isSup_elim___redArg(lean_object* v_t_116_, lean_object* v_isSup_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_116_, v_isSup_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_AtomicFact_isSup_elim(lean_object* v_motive_119_, lean_object* v_t_120_, lean_object* v_h_121_, lean_object* v_isSup_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorElim___redArg(v_t_120_, v_isSup_122_);
return v___x_123_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__2(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_127_ = lean_box(0);
v___x_128_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__1));
v___x_129_ = l_Lean_Expr_const___override(v___x_128_, v___x_127_);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__3(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_130_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__2, &lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__2);
v___x_131_ = lean_unsigned_to_nat(0u);
v___x_132_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
lean_ctor_set(v___x_132_, 2, v___x_130_);
return v___x_132_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default(void){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__3, &lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default___closed__3);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact(void){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default;
return v___x_134_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact_beq(lean_object* v_x_135_, lean_object* v_x_136_){
_start:
{
lean_object* v_lhs_138_; lean_object* v_rhs_139_; lean_object* v_proof_140_; lean_object* v_lhs_x27_141_; lean_object* v_rhs_x27_142_; lean_object* v_proof_x27_143_; lean_object* v_lhs_148_; lean_object* v_rhs_149_; lean_object* v_res_150_; lean_object* v_lhs_x27_151_; lean_object* v_rhs_x27_152_; lean_object* v_res_x27_153_; lean_object* v___x_157_; lean_object* v___x_158_; uint8_t v___x_159_; 
v___x_157_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorIdx(v_x_135_);
v___x_158_ = lp_mathlib_Mathlib_Tactic_Order_AtomicFact_ctorIdx(v_x_136_);
v___x_159_ = lean_nat_dec_eq(v___x_157_, v___x_158_);
lean_dec(v___x_158_);
lean_dec(v___x_157_);
if (v___x_159_ == 0)
{
return v___x_159_;
}
else
{
switch(lean_obj_tag(v_x_135_))
{
case 6:
{
lean_object* v_idx_160_; lean_object* v_idx_161_; uint8_t v___x_162_; 
v_idx_160_ = lean_ctor_get(v_x_135_, 0);
v_idx_161_ = lean_ctor_get(v_x_136_, 0);
v___x_162_ = lean_nat_dec_eq(v_idx_160_, v_idx_161_);
return v___x_162_;
}
case 7:
{
lean_object* v_idx_163_; lean_object* v_idx_164_; uint8_t v___x_165_; 
v_idx_163_ = lean_ctor_get(v_x_135_, 0);
v_idx_164_ = lean_ctor_get(v_x_136_, 0);
v___x_165_ = lean_nat_dec_eq(v_idx_163_, v_idx_164_);
return v___x_165_;
}
case 8:
{
lean_object* v_lhs_166_; lean_object* v_rhs_167_; lean_object* v_res_168_; lean_object* v_lhs_169_; lean_object* v_rhs_170_; lean_object* v_res_171_; 
v_lhs_166_ = lean_ctor_get(v_x_135_, 0);
v_rhs_167_ = lean_ctor_get(v_x_135_, 1);
v_res_168_ = lean_ctor_get(v_x_135_, 2);
v_lhs_169_ = lean_ctor_get(v_x_136_, 0);
v_rhs_170_ = lean_ctor_get(v_x_136_, 1);
v_res_171_ = lean_ctor_get(v_x_136_, 2);
v_lhs_148_ = v_lhs_166_;
v_rhs_149_ = v_rhs_167_;
v_res_150_ = v_res_168_;
v_lhs_x27_151_ = v_lhs_169_;
v_rhs_x27_152_ = v_rhs_170_;
v_res_x27_153_ = v_res_171_;
goto v___jp_147_;
}
case 9:
{
lean_object* v_lhs_172_; lean_object* v_rhs_173_; lean_object* v_res_174_; lean_object* v_lhs_175_; lean_object* v_rhs_176_; lean_object* v_res_177_; 
v_lhs_172_ = lean_ctor_get(v_x_135_, 0);
v_rhs_173_ = lean_ctor_get(v_x_135_, 1);
v_res_174_ = lean_ctor_get(v_x_135_, 2);
v_lhs_175_ = lean_ctor_get(v_x_136_, 0);
v_rhs_176_ = lean_ctor_get(v_x_136_, 1);
v_res_177_ = lean_ctor_get(v_x_136_, 2);
v_lhs_148_ = v_lhs_172_;
v_rhs_149_ = v_rhs_173_;
v_res_150_ = v_res_174_;
v_lhs_x27_151_ = v_lhs_175_;
v_rhs_x27_152_ = v_rhs_176_;
v_res_x27_153_ = v_res_177_;
goto v___jp_147_;
}
default: 
{
lean_object* v_lhs_178_; lean_object* v_rhs_179_; lean_object* v_proof_180_; lean_object* v_lhs_181_; lean_object* v_rhs_182_; lean_object* v_proof_183_; 
v_lhs_178_ = lean_ctor_get(v_x_135_, 0);
v_rhs_179_ = lean_ctor_get(v_x_135_, 1);
v_proof_180_ = lean_ctor_get(v_x_135_, 2);
v_lhs_181_ = lean_ctor_get(v_x_136_, 0);
v_rhs_182_ = lean_ctor_get(v_x_136_, 1);
v_proof_183_ = lean_ctor_get(v_x_136_, 2);
v_lhs_138_ = v_lhs_178_;
v_rhs_139_ = v_rhs_179_;
v_proof_140_ = v_proof_180_;
v_lhs_x27_141_ = v_lhs_181_;
v_rhs_x27_142_ = v_rhs_182_;
v_proof_x27_143_ = v_proof_183_;
goto v___jp_137_;
}
}
}
v___jp_137_:
{
uint8_t v___x_144_; 
v___x_144_ = lean_nat_dec_eq(v_lhs_138_, v_lhs_x27_141_);
if (v___x_144_ == 0)
{
return v___x_144_;
}
else
{
uint8_t v___x_145_; 
v___x_145_ = lean_nat_dec_eq(v_rhs_139_, v_rhs_x27_142_);
if (v___x_145_ == 0)
{
return v___x_145_;
}
else
{
uint8_t v___x_146_; 
v___x_146_ = lean_expr_eqv(v_proof_140_, v_proof_x27_143_);
return v___x_146_;
}
}
}
v___jp_147_:
{
uint8_t v___x_154_; 
v___x_154_ = lean_nat_dec_eq(v_lhs_148_, v_lhs_x27_151_);
if (v___x_154_ == 0)
{
return v___x_154_;
}
else
{
uint8_t v___x_155_; 
v___x_155_ = lean_nat_dec_eq(v_rhs_149_, v_rhs_x27_152_);
if (v___x_155_ == 0)
{
return v___x_155_;
}
else
{
uint8_t v___x_156_; 
v___x_156_ = lean_nat_dec_eq(v_res_150_, v_res_x27_153_);
return v___x_156_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact_beq___boxed(lean_object* v_x_184_, lean_object* v_x_185_){
_start:
{
uint8_t v_res_186_; lean_object* v_r_187_; 
v_res_186_ = lp_mathlib_Mathlib_Tactic_Order_instBEqAtomicFact_beq(v_x_184_, v_x_185_);
lean_dec_ref(v_x_185_);
lean_dec_ref(v_x_184_);
v_r_187_ = lean_box(v_res_186_);
return v_r_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0(lean_object* v_fa_201_){
_start:
{
switch(lean_obj_tag(v_fa_201_))
{
case 0:
{
lean_object* v_lhs_202_; lean_object* v_rhs_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; 
v_lhs_202_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_202_);
v_rhs_203_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_203_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_204_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_205_ = l_Nat_reprFast(v_lhs_202_);
v___x_206_ = lean_string_append(v___x_204_, v___x_205_);
lean_dec_ref(v___x_205_);
v___x_207_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__1));
v___x_208_ = lean_string_append(v___x_206_, v___x_207_);
v___x_209_ = l_Nat_reprFast(v_rhs_203_);
v___x_210_ = lean_string_append(v___x_208_, v___x_209_);
lean_dec_ref(v___x_209_);
return v___x_210_;
}
case 1:
{
lean_object* v_lhs_211_; lean_object* v_rhs_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_lhs_211_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_211_);
v_rhs_212_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_212_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_213_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_214_ = l_Nat_reprFast(v_lhs_211_);
v___x_215_ = lean_string_append(v___x_213_, v___x_214_);
lean_dec_ref(v___x_214_);
v___x_216_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__2));
v___x_217_ = lean_string_append(v___x_215_, v___x_216_);
v___x_218_ = l_Nat_reprFast(v_rhs_212_);
v___x_219_ = lean_string_append(v___x_217_, v___x_218_);
lean_dec_ref(v___x_218_);
return v___x_219_;
}
case 2:
{
lean_object* v_lhs_220_; lean_object* v_rhs_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v_lhs_220_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_220_);
v_rhs_221_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_221_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_222_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_223_ = l_Nat_reprFast(v_lhs_220_);
v___x_224_ = lean_string_append(v___x_222_, v___x_223_);
lean_dec_ref(v___x_223_);
v___x_225_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__3));
v___x_226_ = lean_string_append(v___x_224_, v___x_225_);
v___x_227_ = l_Nat_reprFast(v_rhs_221_);
v___x_228_ = lean_string_append(v___x_226_, v___x_227_);
lean_dec_ref(v___x_227_);
return v___x_228_;
}
case 3:
{
lean_object* v_lhs_229_; lean_object* v_rhs_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v_lhs_229_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_229_);
v_rhs_230_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_230_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_231_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__4));
v___x_232_ = l_Nat_reprFast(v_lhs_229_);
v___x_233_ = lean_string_append(v___x_231_, v___x_232_);
lean_dec_ref(v___x_232_);
v___x_234_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__3));
v___x_235_ = lean_string_append(v___x_233_, v___x_234_);
v___x_236_ = l_Nat_reprFast(v_rhs_230_);
v___x_237_ = lean_string_append(v___x_235_, v___x_236_);
lean_dec_ref(v___x_236_);
return v___x_237_;
}
case 4:
{
lean_object* v_lhs_238_; lean_object* v_rhs_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v_lhs_238_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_238_);
v_rhs_239_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_239_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_240_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_241_ = l_Nat_reprFast(v_lhs_238_);
v___x_242_ = lean_string_append(v___x_240_, v___x_241_);
lean_dec_ref(v___x_241_);
v___x_243_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__5));
v___x_244_ = lean_string_append(v___x_242_, v___x_243_);
v___x_245_ = l_Nat_reprFast(v_rhs_239_);
v___x_246_ = lean_string_append(v___x_244_, v___x_245_);
lean_dec_ref(v___x_245_);
return v___x_246_;
}
case 5:
{
lean_object* v_lhs_247_; lean_object* v_rhs_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v_lhs_247_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_247_);
v_rhs_248_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_248_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_249_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__4));
v___x_250_ = l_Nat_reprFast(v_lhs_247_);
v___x_251_ = lean_string_append(v___x_249_, v___x_250_);
lean_dec_ref(v___x_250_);
v___x_252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__5));
v___x_253_ = lean_string_append(v___x_251_, v___x_252_);
v___x_254_ = l_Nat_reprFast(v_rhs_248_);
v___x_255_ = lean_string_append(v___x_253_, v___x_254_);
lean_dec_ref(v___x_254_);
return v___x_255_;
}
case 6:
{
lean_object* v_idx_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v_idx_256_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_idx_256_);
lean_dec_ref_known(v_fa_201_, 1);
v___x_257_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_258_ = l_Nat_reprFast(v_idx_256_);
v___x_259_ = lean_string_append(v___x_257_, v___x_258_);
lean_dec_ref(v___x_258_);
v___x_260_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__6));
v___x_261_ = lean_string_append(v___x_259_, v___x_260_);
return v___x_261_;
}
case 7:
{
lean_object* v_idx_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v_idx_262_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_idx_262_);
lean_dec_ref_known(v_fa_201_, 1);
v___x_263_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_264_ = l_Nat_reprFast(v_idx_262_);
v___x_265_ = lean_string_append(v___x_263_, v___x_264_);
lean_dec_ref(v___x_264_);
v___x_266_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__7));
v___x_267_ = lean_string_append(v___x_265_, v___x_266_);
return v___x_267_;
}
case 8:
{
lean_object* v_lhs_268_; lean_object* v_rhs_269_; lean_object* v_res_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v_lhs_268_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_268_);
v_rhs_269_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_269_);
v_res_270_ = lean_ctor_get(v_fa_201_, 2);
lean_inc(v_res_270_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_271_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_272_ = l_Nat_reprFast(v_res_270_);
v___x_273_ = lean_string_append(v___x_271_, v___x_272_);
lean_dec_ref(v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__8));
v___x_275_ = lean_string_append(v___x_273_, v___x_274_);
v___x_276_ = l_Nat_reprFast(v_lhs_268_);
v___x_277_ = lean_string_append(v___x_275_, v___x_276_);
lean_dec_ref(v___x_276_);
v___x_278_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__9));
v___x_279_ = lean_string_append(v___x_277_, v___x_278_);
v___x_280_ = l_Nat_reprFast(v_rhs_269_);
v___x_281_ = lean_string_append(v___x_279_, v___x_280_);
lean_dec_ref(v___x_280_);
return v___x_281_;
}
default: 
{
lean_object* v_lhs_282_; lean_object* v_rhs_283_; lean_object* v_res_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v_lhs_282_ = lean_ctor_get(v_fa_201_, 0);
lean_inc(v_lhs_282_);
v_rhs_283_ = lean_ctor_get(v_fa_201_, 1);
lean_inc(v_rhs_283_);
v_res_284_ = lean_ctor_get(v_fa_201_, 2);
lean_inc(v_res_284_);
lean_dec_ref_known(v_fa_201_, 3);
v___x_285_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__0));
v___x_286_ = l_Nat_reprFast(v_res_284_);
v___x_287_ = lean_string_append(v___x_285_, v___x_286_);
lean_dec_ref(v___x_286_);
v___x_288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__8));
v___x_289_ = lean_string_append(v___x_287_, v___x_288_);
v___x_290_ = l_Nat_reprFast(v_lhs_282_);
v___x_291_ = lean_string_append(v___x_289_, v___x_290_);
lean_dec_ref(v___x_290_);
v___x_292_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringAtomicFact___lam__0___closed__10));
v___x_293_ = lean_string_append(v___x_291_, v___x_292_);
v___x_294_ = l_Nat_reprFast(v_rhs_283_);
v___x_295_ = lean_string_append(v___x_293_, v___x_294_);
lean_dec_ref(v___x_294_);
return v___x_295_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3_spec__6___redArg(lean_object* v_x_298_, lean_object* v_x_299_){
_start:
{
if (lean_obj_tag(v_x_299_) == 0)
{
return v_x_298_;
}
else
{
lean_object* v_key_300_; lean_object* v_value_301_; lean_object* v_tail_302_; lean_object* v___x_304_; uint8_t v_isShared_305_; uint8_t v_isSharedCheck_325_; 
v_key_300_ = lean_ctor_get(v_x_299_, 0);
v_value_301_ = lean_ctor_get(v_x_299_, 1);
v_tail_302_ = lean_ctor_get(v_x_299_, 2);
v_isSharedCheck_325_ = !lean_is_exclusive(v_x_299_);
if (v_isSharedCheck_325_ == 0)
{
v___x_304_ = v_x_299_;
v_isShared_305_ = v_isSharedCheck_325_;
goto v_resetjp_303_;
}
else
{
lean_inc(v_tail_302_);
lean_inc(v_value_301_);
lean_inc(v_key_300_);
lean_dec(v_x_299_);
v___x_304_ = lean_box(0);
v_isShared_305_ = v_isSharedCheck_325_;
goto v_resetjp_303_;
}
v_resetjp_303_:
{
lean_object* v___x_306_; uint64_t v___x_307_; uint64_t v___x_308_; uint64_t v___x_309_; uint64_t v_fold_310_; uint64_t v___x_311_; uint64_t v___x_312_; uint64_t v___x_313_; size_t v___x_314_; size_t v___x_315_; size_t v___x_316_; size_t v___x_317_; size_t v___x_318_; lean_object* v___x_319_; lean_object* v___x_321_; 
v___x_306_ = lean_array_get_size(v_x_298_);
v___x_307_ = l_Lean_Expr_hash(v_key_300_);
v___x_308_ = 32ULL;
v___x_309_ = lean_uint64_shift_right(v___x_307_, v___x_308_);
v_fold_310_ = lean_uint64_xor(v___x_307_, v___x_309_);
v___x_311_ = 16ULL;
v___x_312_ = lean_uint64_shift_right(v_fold_310_, v___x_311_);
v___x_313_ = lean_uint64_xor(v_fold_310_, v___x_312_);
v___x_314_ = lean_uint64_to_usize(v___x_313_);
v___x_315_ = lean_usize_of_nat(v___x_306_);
v___x_316_ = ((size_t)1ULL);
v___x_317_ = lean_usize_sub(v___x_315_, v___x_316_);
v___x_318_ = lean_usize_land(v___x_314_, v___x_317_);
v___x_319_ = lean_array_uget_borrowed(v_x_298_, v___x_318_);
lean_inc(v___x_319_);
if (v_isShared_305_ == 0)
{
lean_ctor_set(v___x_304_, 2, v___x_319_);
v___x_321_ = v___x_304_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_324_; 
v_reuseFailAlloc_324_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_324_, 0, v_key_300_);
lean_ctor_set(v_reuseFailAlloc_324_, 1, v_value_301_);
lean_ctor_set(v_reuseFailAlloc_324_, 2, v___x_319_);
v___x_321_ = v_reuseFailAlloc_324_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
lean_object* v___x_322_; 
v___x_322_ = lean_array_uset(v_x_298_, v___x_318_, v___x_321_);
v_x_298_ = v___x_322_;
v_x_299_ = v_tail_302_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3___redArg(lean_object* v_i_326_, lean_object* v_source_327_, lean_object* v_target_328_){
_start:
{
lean_object* v___x_329_; uint8_t v___x_330_; 
v___x_329_ = lean_array_get_size(v_source_327_);
v___x_330_ = lean_nat_dec_lt(v_i_326_, v___x_329_);
if (v___x_330_ == 0)
{
lean_dec_ref(v_source_327_);
lean_dec(v_i_326_);
return v_target_328_;
}
else
{
lean_object* v_es_331_; lean_object* v___x_332_; lean_object* v_source_333_; lean_object* v_target_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v_es_331_ = lean_array_fget(v_source_327_, v_i_326_);
v___x_332_ = lean_box(0);
v_source_333_ = lean_array_fset(v_source_327_, v_i_326_, v___x_332_);
v_target_334_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3_spec__6___redArg(v_target_328_, v_es_331_);
v___x_335_ = lean_unsigned_to_nat(1u);
v___x_336_ = lean_nat_add(v_i_326_, v___x_335_);
lean_dec(v_i_326_);
v_i_326_ = v___x_336_;
v_source_327_ = v_source_333_;
v_target_328_ = v_target_334_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2___redArg(lean_object* v_data_338_){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v_nbuckets_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_339_ = lean_array_get_size(v_data_338_);
v___x_340_ = lean_unsigned_to_nat(2u);
v_nbuckets_341_ = lean_nat_mul(v___x_339_, v___x_340_);
v___x_342_ = lean_unsigned_to_nat(0u);
v___x_343_ = lean_box(0);
v___x_344_ = lean_mk_array(v_nbuckets_341_, v___x_343_);
v___x_345_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3___redArg(v___x_342_, v_data_338_, v___x_344_);
return v___x_345_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg(lean_object* v_a_346_, lean_object* v_x_347_){
_start:
{
if (lean_obj_tag(v_x_347_) == 0)
{
uint8_t v___x_348_; 
v___x_348_ = 0;
return v___x_348_;
}
else
{
lean_object* v_key_349_; lean_object* v_tail_350_; uint8_t v___x_351_; 
v_key_349_ = lean_ctor_get(v_x_347_, 0);
v_tail_350_ = lean_ctor_get(v_x_347_, 2);
v___x_351_ = lean_expr_eqv(v_key_349_, v_a_346_);
if (v___x_351_ == 0)
{
v_x_347_ = v_tail_350_;
goto _start;
}
else
{
return v___x_351_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg___boxed(lean_object* v_a_353_, lean_object* v_x_354_){
_start:
{
uint8_t v_res_355_; lean_object* v_r_356_; 
v_res_355_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg(v_a_353_, v_x_354_);
lean_dec(v_x_354_);
lean_dec_ref(v_a_353_);
v_r_356_ = lean_box(v_res_355_);
return v_r_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__3___redArg(lean_object* v_a_357_, lean_object* v_b_358_, lean_object* v_x_359_){
_start:
{
if (lean_obj_tag(v_x_359_) == 0)
{
lean_dec(v_b_358_);
lean_dec_ref(v_a_357_);
return v_x_359_;
}
else
{
lean_object* v_key_360_; lean_object* v_value_361_; lean_object* v_tail_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_374_; 
v_key_360_ = lean_ctor_get(v_x_359_, 0);
v_value_361_ = lean_ctor_get(v_x_359_, 1);
v_tail_362_ = lean_ctor_get(v_x_359_, 2);
v_isSharedCheck_374_ = !lean_is_exclusive(v_x_359_);
if (v_isSharedCheck_374_ == 0)
{
v___x_364_ = v_x_359_;
v_isShared_365_ = v_isSharedCheck_374_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_tail_362_);
lean_inc(v_value_361_);
lean_inc(v_key_360_);
lean_dec(v_x_359_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_374_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
uint8_t v___x_366_; 
v___x_366_ = lean_expr_eqv(v_key_360_, v_a_357_);
if (v___x_366_ == 0)
{
lean_object* v___x_367_; lean_object* v___x_369_; 
v___x_367_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__3___redArg(v_a_357_, v_b_358_, v_tail_362_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 2, v___x_367_);
v___x_369_ = v___x_364_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v_key_360_);
lean_ctor_set(v_reuseFailAlloc_370_, 1, v_value_361_);
lean_ctor_set(v_reuseFailAlloc_370_, 2, v___x_367_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
else
{
lean_object* v___x_372_; 
lean_dec(v_value_361_);
lean_dec(v_key_360_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 1, v_b_358_);
lean_ctor_set(v___x_364_, 0, v_a_357_);
v___x_372_ = v___x_364_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v_a_357_);
lean_ctor_set(v_reuseFailAlloc_373_, 1, v_b_358_);
lean_ctor_set(v_reuseFailAlloc_373_, 2, v_tail_362_);
v___x_372_ = v_reuseFailAlloc_373_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
return v___x_372_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1___redArg(lean_object* v_m_375_, lean_object* v_a_376_, lean_object* v_b_377_){
_start:
{
lean_object* v_size_378_; lean_object* v_buckets_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_422_; 
v_size_378_ = lean_ctor_get(v_m_375_, 0);
v_buckets_379_ = lean_ctor_get(v_m_375_, 1);
v_isSharedCheck_422_ = !lean_is_exclusive(v_m_375_);
if (v_isSharedCheck_422_ == 0)
{
v___x_381_ = v_m_375_;
v_isShared_382_ = v_isSharedCheck_422_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_buckets_379_);
lean_inc(v_size_378_);
lean_dec(v_m_375_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_422_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_383_; uint64_t v___x_384_; uint64_t v___x_385_; uint64_t v___x_386_; uint64_t v_fold_387_; uint64_t v___x_388_; uint64_t v___x_389_; uint64_t v___x_390_; size_t v___x_391_; size_t v___x_392_; size_t v___x_393_; size_t v___x_394_; size_t v___x_395_; lean_object* v_bkt_396_; uint8_t v___x_397_; 
v___x_383_ = lean_array_get_size(v_buckets_379_);
v___x_384_ = l_Lean_Expr_hash(v_a_376_);
v___x_385_ = 32ULL;
v___x_386_ = lean_uint64_shift_right(v___x_384_, v___x_385_);
v_fold_387_ = lean_uint64_xor(v___x_384_, v___x_386_);
v___x_388_ = 16ULL;
v___x_389_ = lean_uint64_shift_right(v_fold_387_, v___x_388_);
v___x_390_ = lean_uint64_xor(v_fold_387_, v___x_389_);
v___x_391_ = lean_uint64_to_usize(v___x_390_);
v___x_392_ = lean_usize_of_nat(v___x_383_);
v___x_393_ = ((size_t)1ULL);
v___x_394_ = lean_usize_sub(v___x_392_, v___x_393_);
v___x_395_ = lean_usize_land(v___x_391_, v___x_394_);
v_bkt_396_ = lean_array_uget_borrowed(v_buckets_379_, v___x_395_);
v___x_397_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg(v_a_376_, v_bkt_396_);
if (v___x_397_ == 0)
{
lean_object* v___x_398_; lean_object* v_size_x27_399_; lean_object* v___x_400_; lean_object* v_buckets_x27_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; uint8_t v___x_407_; 
v___x_398_ = lean_unsigned_to_nat(1u);
v_size_x27_399_ = lean_nat_add(v_size_378_, v___x_398_);
lean_dec(v_size_378_);
lean_inc(v_bkt_396_);
v___x_400_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_400_, 0, v_a_376_);
lean_ctor_set(v___x_400_, 1, v_b_377_);
lean_ctor_set(v___x_400_, 2, v_bkt_396_);
v_buckets_x27_401_ = lean_array_uset(v_buckets_379_, v___x_395_, v___x_400_);
v___x_402_ = lean_unsigned_to_nat(4u);
v___x_403_ = lean_nat_mul(v_size_x27_399_, v___x_402_);
v___x_404_ = lean_unsigned_to_nat(3u);
v___x_405_ = lean_nat_div(v___x_403_, v___x_404_);
lean_dec(v___x_403_);
v___x_406_ = lean_array_get_size(v_buckets_x27_401_);
v___x_407_ = lean_nat_dec_le(v___x_405_, v___x_406_);
lean_dec(v___x_405_);
if (v___x_407_ == 0)
{
lean_object* v_val_408_; lean_object* v___x_410_; 
v_val_408_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2___redArg(v_buckets_x27_401_);
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 1, v_val_408_);
lean_ctor_set(v___x_381_, 0, v_size_x27_399_);
v___x_410_ = v___x_381_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_size_x27_399_);
lean_ctor_set(v_reuseFailAlloc_411_, 1, v_val_408_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
else
{
lean_object* v___x_413_; 
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 1, v_buckets_x27_401_);
lean_ctor_set(v___x_381_, 0, v_size_x27_399_);
v___x_413_ = v___x_381_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_size_x27_399_);
lean_ctor_set(v_reuseFailAlloc_414_, 1, v_buckets_x27_401_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
return v___x_413_;
}
}
}
else
{
lean_object* v___x_415_; lean_object* v_buckets_x27_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_420_; 
lean_inc(v_bkt_396_);
v___x_415_ = lean_box(0);
v_buckets_x27_416_ = lean_array_uset(v_buckets_379_, v___x_395_, v___x_415_);
v___x_417_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__3___redArg(v_a_376_, v_b_377_, v_bkt_396_);
v___x_418_ = lean_array_uset(v_buckets_x27_416_, v___x_395_, v___x_417_);
if (v_isShared_382_ == 0)
{
lean_ctor_set(v___x_381_, 1, v___x_418_);
v___x_420_ = v___x_381_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v_size_378_);
lean_ctor_set(v_reuseFailAlloc_421_, 1, v___x_418_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
return v___x_420_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Order_addType_spec__2(lean_object* v_x_423_, lean_object* v_x_424_){
_start:
{
if (lean_obj_tag(v_x_424_) == 0)
{
lean_inc(v_x_423_);
return v_x_423_;
}
else
{
lean_object* v_key_425_; lean_object* v_tail_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v_key_425_ = lean_ctor_get(v_x_424_, 0);
v_tail_426_ = lean_ctor_get(v_x_424_, 2);
v___x_427_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Order_addType_spec__2(v_x_423_, v_tail_426_);
lean_inc(v_key_425_);
v___x_428_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_428_, 0, v_key_425_);
lean_ctor_set(v___x_428_, 1, v___x_427_);
return v___x_428_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Order_addType_spec__2___boxed(lean_object* v_x_429_, lean_object* v_x_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Order_addType_spec__2(v_x_429_, v_x_430_);
lean_dec(v_x_430_);
lean_dec(v_x_429_);
return v_res_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Order_addType_spec__3(lean_object* v_as_432_, size_t v_i_433_, size_t v_stop_434_, lean_object* v_b_435_){
_start:
{
uint8_t v___x_436_; 
v___x_436_ = lean_usize_dec_eq(v_i_433_, v_stop_434_);
if (v___x_436_ == 0)
{
size_t v___x_437_; size_t v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
v___x_437_ = ((size_t)1ULL);
v___x_438_ = lean_usize_sub(v_i_433_, v___x_437_);
v___x_439_ = lean_array_uget_borrowed(v_as_432_, v___x_438_);
v___x_440_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldrM___at___00Mathlib_Tactic_Order_addType_spec__2(v_b_435_, v___x_439_);
lean_dec(v_b_435_);
v_i_433_ = v___x_438_;
v_b_435_ = v___x_440_;
goto _start;
}
else
{
return v_b_435_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Order_addType_spec__3___boxed(lean_object* v_as_442_, lean_object* v_i_443_, lean_object* v_stop_444_, lean_object* v_b_445_){
_start:
{
size_t v_i_boxed_446_; size_t v_stop_boxed_447_; lean_object* v_res_448_; 
v_i_boxed_446_ = lean_unbox_usize(v_i_443_);
lean_dec(v_i_443_);
v_stop_boxed_447_ = lean_unbox_usize(v_stop_444_);
lean_dec(v_stop_444_);
v_res_448_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Order_addType_spec__3(v_as_442_, v_i_boxed_446_, v_stop_boxed_447_, v_b_445_);
lean_dec_ref(v_as_442_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___redArg(lean_object* v_type_449_, lean_object* v_x_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_){
_start:
{
if (lean_obj_tag(v_x_450_) == 0)
{
lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; 
lean_dec_ref(v_type_449_);
v___x_457_ = lean_box(0);
v___x_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_457_);
lean_ctor_set(v___x_458_, 1, v___y_451_);
v___x_459_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_459_, 0, v___x_458_);
return v___x_459_;
}
else
{
lean_object* v_head_460_; lean_object* v_tail_461_; lean_object* v___x_463_; uint8_t v_isShared_464_; uint8_t v_isSharedCheck_502_; 
v_head_460_ = lean_ctor_get(v_x_450_, 0);
v_tail_461_ = lean_ctor_get(v_x_450_, 1);
v_isSharedCheck_502_ = !lean_is_exclusive(v_x_450_);
if (v_isSharedCheck_502_ == 0)
{
v___x_463_ = v_x_450_;
v_isShared_464_ = v_isSharedCheck_502_;
goto v_resetjp_462_;
}
else
{
lean_inc(v_tail_461_);
lean_inc(v_head_460_);
lean_dec(v_x_450_);
v___x_463_ = lean_box(0);
v_isShared_464_ = v_isSharedCheck_502_;
goto v_resetjp_462_;
}
v_resetjp_462_:
{
lean_object* v_keyedConfig_465_; uint8_t v_trackZetaDelta_466_; lean_object* v_zetaDeltaSet_467_; lean_object* v_lctx_468_; lean_object* v_localInstances_469_; lean_object* v_defEqCtx_x3f_470_; lean_object* v_synthPendingDepth_471_; lean_object* v_customCanUnfoldPredicate_x3f_472_; uint8_t v_univApprox_473_; uint8_t v_inTypeClassResolution_474_; uint8_t v_cacheInferType_475_; uint8_t v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; 
v_keyedConfig_465_ = lean_ctor_get(v___y_452_, 0);
v_trackZetaDelta_466_ = lean_ctor_get_uint8(v___y_452_, sizeof(void*)*7);
v_zetaDeltaSet_467_ = lean_ctor_get(v___y_452_, 1);
v_lctx_468_ = lean_ctor_get(v___y_452_, 2);
v_localInstances_469_ = lean_ctor_get(v___y_452_, 3);
v_defEqCtx_x3f_470_ = lean_ctor_get(v___y_452_, 4);
v_synthPendingDepth_471_ = lean_ctor_get(v___y_452_, 5);
v_customCanUnfoldPredicate_x3f_472_ = lean_ctor_get(v___y_452_, 6);
v_univApprox_473_ = lean_ctor_get_uint8(v___y_452_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_474_ = lean_ctor_get_uint8(v___y_452_, sizeof(void*)*7 + 2);
v_cacheInferType_475_ = lean_ctor_get_uint8(v___y_452_, sizeof(void*)*7 + 3);
v___x_476_ = 3;
lean_inc_ref(v_keyedConfig_465_);
v___x_477_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_476_, v_keyedConfig_465_);
lean_inc(v_customCanUnfoldPredicate_x3f_472_);
lean_inc(v_synthPendingDepth_471_);
lean_inc(v_defEqCtx_x3f_470_);
lean_inc_ref(v_localInstances_469_);
lean_inc_ref(v_lctx_468_);
lean_inc(v_zetaDeltaSet_467_);
v___x_478_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_478_, 0, v___x_477_);
lean_ctor_set(v___x_478_, 1, v_zetaDeltaSet_467_);
lean_ctor_set(v___x_478_, 2, v_lctx_468_);
lean_ctor_set(v___x_478_, 3, v_localInstances_469_);
lean_ctor_set(v___x_478_, 4, v_defEqCtx_x3f_470_);
lean_ctor_set(v___x_478_, 5, v_synthPendingDepth_471_);
lean_ctor_set(v___x_478_, 6, v_customCanUnfoldPredicate_x3f_472_);
lean_ctor_set_uint8(v___x_478_, sizeof(void*)*7, v_trackZetaDelta_466_);
lean_ctor_set_uint8(v___x_478_, sizeof(void*)*7 + 1, v_univApprox_473_);
lean_ctor_set_uint8(v___x_478_, sizeof(void*)*7 + 2, v_inTypeClassResolution_474_);
lean_ctor_set_uint8(v___x_478_, sizeof(void*)*7 + 3, v_cacheInferType_475_);
lean_inc(v_head_460_);
lean_inc_ref(v_type_449_);
v___x_479_ = l_Lean_Meta_isExprDefEq(v_type_449_, v_head_460_, v___x_478_, v___y_453_, v___y_454_, v___y_455_);
lean_dec_ref_known(v___x_478_, 7);
if (lean_obj_tag(v___x_479_) == 0)
{
lean_object* v_a_480_; lean_object* v___x_482_; uint8_t v_isShared_483_; uint8_t v_isSharedCheck_493_; 
v_a_480_ = lean_ctor_get(v___x_479_, 0);
v_isSharedCheck_493_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_493_ == 0)
{
v___x_482_ = v___x_479_;
v_isShared_483_ = v_isSharedCheck_493_;
goto v_resetjp_481_;
}
else
{
lean_inc(v_a_480_);
lean_dec(v___x_479_);
v___x_482_ = lean_box(0);
v_isShared_483_ = v_isSharedCheck_493_;
goto v_resetjp_481_;
}
v_resetjp_481_:
{
uint8_t v___x_484_; 
v___x_484_ = lean_unbox(v_a_480_);
lean_dec(v_a_480_);
if (v___x_484_ == 0)
{
lean_del_object(v___x_482_);
lean_del_object(v___x_463_);
lean_dec(v_head_460_);
v_x_450_ = v_tail_461_;
goto _start;
}
else
{
lean_object* v___x_486_; lean_object* v___x_488_; 
lean_dec(v_tail_461_);
lean_dec_ref(v_type_449_);
v___x_486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_486_, 0, v_head_460_);
if (v_isShared_464_ == 0)
{
lean_ctor_set_tag(v___x_463_, 0);
lean_ctor_set(v___x_463_, 1, v___y_451_);
lean_ctor_set(v___x_463_, 0, v___x_486_);
v___x_488_ = v___x_463_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_492_; 
v_reuseFailAlloc_492_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_492_, 0, v___x_486_);
lean_ctor_set(v_reuseFailAlloc_492_, 1, v___y_451_);
v___x_488_ = v_reuseFailAlloc_492_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
lean_object* v___x_490_; 
if (v_isShared_483_ == 0)
{
lean_ctor_set(v___x_482_, 0, v___x_488_);
v___x_490_ = v___x_482_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_491_; 
v_reuseFailAlloc_491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_491_, 0, v___x_488_);
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
}
else
{
lean_object* v_a_494_; lean_object* v___x_496_; uint8_t v_isShared_497_; uint8_t v_isSharedCheck_501_; 
lean_del_object(v___x_463_);
lean_dec(v_tail_461_);
lean_dec(v_head_460_);
lean_dec_ref(v___y_451_);
lean_dec_ref(v_type_449_);
v_a_494_ = lean_ctor_get(v___x_479_, 0);
v_isSharedCheck_501_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_501_ == 0)
{
v___x_496_ = v___x_479_;
v_isShared_497_ = v_isSharedCheck_501_;
goto v_resetjp_495_;
}
else
{
lean_inc(v_a_494_);
lean_dec(v___x_479_);
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
v_reuseFailAlloc_500_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___redArg___boxed(lean_object* v_type_503_, lean_object* v_x_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___redArg(v_type_503_, v_x_504_, v___y_505_, v___y_506_, v___y_507_, v___y_508_, v___y_509_);
lean_dec(v___y_509_);
lean_dec_ref(v___y_508_);
lean_dec(v___y_507_);
lean_dec_ref(v___y_506_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType___redArg(lean_object* v_type_514_, lean_object* v_a_515_, lean_object* v_a_516_, lean_object* v_a_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_, lean_object* v_a_521_){
_start:
{
lean_object* v___y_524_; lean_object* v_buckets_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; uint8_t v___x_571_; 
v_buckets_567_ = lean_ctor_get(v_a_515_, 1);
v___x_568_ = lean_box(0);
v___x_569_ = lean_array_get_size(v_buckets_567_);
v___x_570_ = lean_unsigned_to_nat(0u);
v___x_571_ = lean_nat_dec_lt(v___x_570_, v___x_569_);
if (v___x_571_ == 0)
{
v___y_524_ = v___x_568_;
goto v___jp_523_;
}
else
{
size_t v___x_572_; size_t v___x_573_; lean_object* v___x_574_; 
v___x_572_ = lean_usize_of_nat(v___x_569_);
v___x_573_ = ((size_t)0ULL);
v___x_574_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00Mathlib_Tactic_Order_addType_spec__3(v_buckets_567_, v___x_572_, v___x_573_, v___x_568_);
v___y_524_ = v___x_574_;
goto v___jp_523_;
}
v___jp_523_:
{
lean_object* v___x_525_; 
lean_inc_ref(v_type_514_);
v___x_525_ = lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___redArg(v_type_514_, v___y_524_, v_a_515_, v_a_518_, v_a_519_, v_a_520_, v_a_521_);
if (lean_obj_tag(v___x_525_) == 0)
{
lean_object* v_a_526_; lean_object* v___x_528_; uint8_t v_isShared_529_; uint8_t v_isSharedCheck_558_; 
v_a_526_ = lean_ctor_get(v___x_525_, 0);
v_isSharedCheck_558_ = !lean_is_exclusive(v___x_525_);
if (v_isSharedCheck_558_ == 0)
{
v___x_528_ = v___x_525_;
v_isShared_529_ = v_isSharedCheck_558_;
goto v_resetjp_527_;
}
else
{
lean_inc(v_a_526_);
lean_dec(v___x_525_);
v___x_528_ = lean_box(0);
v_isShared_529_ = v_isSharedCheck_558_;
goto v_resetjp_527_;
}
v_resetjp_527_:
{
lean_object* v_fst_530_; 
v_fst_530_ = lean_ctor_get(v_a_526_, 0);
if (lean_obj_tag(v_fst_530_) == 0)
{
lean_object* v_snd_531_; lean_object* v___x_533_; uint8_t v_isShared_534_; uint8_t v_isSharedCheck_543_; 
v_snd_531_ = lean_ctor_get(v_a_526_, 1);
v_isSharedCheck_543_ = !lean_is_exclusive(v_a_526_);
if (v_isSharedCheck_543_ == 0)
{
lean_object* v_unused_544_; 
v_unused_544_ = lean_ctor_get(v_a_526_, 0);
lean_dec(v_unused_544_);
v___x_533_ = v_a_526_;
v_isShared_534_ = v_isSharedCheck_543_;
goto v_resetjp_532_;
}
else
{
lean_inc(v_snd_531_);
lean_dec(v_a_526_);
v___x_533_ = lean_box(0);
v_isShared_534_ = v_isSharedCheck_543_;
goto v_resetjp_532_;
}
v_resetjp_532_:
{
lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_538_; 
v___x_535_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addType___redArg___closed__0));
lean_inc_ref(v_type_514_);
v___x_536_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1___redArg(v_snd_531_, v_type_514_, v___x_535_);
if (v_isShared_534_ == 0)
{
lean_ctor_set(v___x_533_, 1, v___x_536_);
lean_ctor_set(v___x_533_, 0, v_type_514_);
v___x_538_ = v___x_533_;
goto v_reusejp_537_;
}
else
{
lean_object* v_reuseFailAlloc_542_; 
v_reuseFailAlloc_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_542_, 0, v_type_514_);
lean_ctor_set(v_reuseFailAlloc_542_, 1, v___x_536_);
v___x_538_ = v_reuseFailAlloc_542_;
goto v_reusejp_537_;
}
v_reusejp_537_:
{
lean_object* v___x_540_; 
if (v_isShared_529_ == 0)
{
lean_ctor_set(v___x_528_, 0, v___x_538_);
v___x_540_ = v___x_528_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v___x_538_);
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
lean_object* v_snd_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_556_; 
lean_inc_ref(v_fst_530_);
lean_dec_ref(v_type_514_);
v_snd_545_ = lean_ctor_get(v_a_526_, 1);
v_isSharedCheck_556_ = !lean_is_exclusive(v_a_526_);
if (v_isSharedCheck_556_ == 0)
{
lean_object* v_unused_557_; 
v_unused_557_ = lean_ctor_get(v_a_526_, 0);
lean_dec(v_unused_557_);
v___x_547_ = v_a_526_;
v_isShared_548_ = v_isSharedCheck_556_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_snd_545_);
lean_dec(v_a_526_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_556_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v_val_549_; lean_object* v___x_551_; 
v_val_549_ = lean_ctor_get(v_fst_530_, 0);
lean_inc(v_val_549_);
lean_dec_ref_known(v_fst_530_, 1);
if (v_isShared_548_ == 0)
{
lean_ctor_set(v___x_547_, 0, v_val_549_);
v___x_551_ = v___x_547_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_val_549_);
lean_ctor_set(v_reuseFailAlloc_555_, 1, v_snd_545_);
v___x_551_ = v_reuseFailAlloc_555_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
lean_object* v___x_553_; 
if (v_isShared_529_ == 0)
{
lean_ctor_set(v___x_528_, 0, v___x_551_);
v___x_553_ = v___x_528_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v___x_551_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
}
}
else
{
lean_object* v_a_559_; lean_object* v___x_561_; uint8_t v_isShared_562_; uint8_t v_isSharedCheck_566_; 
lean_dec_ref(v_type_514_);
v_a_559_ = lean_ctor_get(v___x_525_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_525_);
if (v_isSharedCheck_566_ == 0)
{
v___x_561_ = v___x_525_;
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
else
{
lean_inc(v_a_559_);
lean_dec(v___x_525_);
v___x_561_ = lean_box(0);
v_isShared_562_ = v_isSharedCheck_566_;
goto v_resetjp_560_;
}
v_resetjp_560_:
{
lean_object* v___x_564_; 
if (v_isShared_562_ == 0)
{
v___x_564_ = v___x_561_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v_a_559_);
v___x_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
return v___x_564_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType___redArg___boxed(lean_object* v_type_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_, lean_object* v_a_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_type_575_, v_a_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_);
lean_dec(v_a_582_);
lean_dec_ref(v_a_581_);
lean_dec(v_a_580_);
lean_dec_ref(v_a_579_);
lean_dec(v_a_578_);
lean_dec_ref(v_a_577_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType(lean_object* v_u_585_, lean_object* v_type_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_, lean_object* v_a_592_, lean_object* v_a_593_){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_type_586_, v_a_587_, v_a_588_, v_a_589_, v_a_590_, v_a_591_, v_a_592_, v_a_593_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addType___boxed(lean_object* v_u_596_, lean_object* v_type_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_Mathlib_Tactic_Order_addType(v_u_596_, v_type_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_, v_a_602_, v_a_603_, v_a_604_);
lean_dec(v_a_604_);
lean_dec_ref(v_a_603_);
lean_dec(v_a_602_);
lean_dec_ref(v_a_601_);
lean_dec(v_a_600_);
lean_dec_ref(v_a_599_);
lean_dec(v_u_596_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0(lean_object* v_type_607_, lean_object* v_x_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___redArg(v_type_607_, v_x_608_, v___y_609_, v___y_612_, v___y_613_, v___y_614_, v___y_615_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0___boxed(lean_object* v_type_618_, lean_object* v_x_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_){
_start:
{
lean_object* v_res_628_; 
v_res_628_ = lp_mathlib_List_findM_x3f___at___00Mathlib_Tactic_Order_addType_spec__0(v_type_618_, v_x_619_, v___y_620_, v___y_621_, v___y_622_, v___y_623_, v___y_624_, v___y_625_, v___y_626_);
lean_dec(v___y_626_);
lean_dec_ref(v___y_625_);
lean_dec(v___y_624_);
lean_dec_ref(v___y_623_);
lean_dec(v___y_622_);
lean_dec_ref(v___y_621_);
return v_res_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1(lean_object* v_00_u03b2_629_, lean_object* v_m_630_, lean_object* v_a_631_, lean_object* v_b_632_){
_start:
{
lean_object* v___x_633_; 
v___x_633_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1___redArg(v_m_630_, v_a_631_, v_b_632_);
return v___x_633_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1(lean_object* v_00_u03b2_634_, lean_object* v_a_635_, lean_object* v_x_636_){
_start:
{
uint8_t v___x_637_; 
v___x_637_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg(v_a_635_, v_x_636_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___boxed(lean_object* v_00_u03b2_638_, lean_object* v_a_639_, lean_object* v_x_640_){
_start:
{
uint8_t v_res_641_; lean_object* v_r_642_; 
v_res_641_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1(v_00_u03b2_638_, v_a_639_, v_x_640_);
lean_dec(v_x_640_);
lean_dec_ref(v_a_639_);
v_r_642_ = lean_box(v_res_641_);
return v_r_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2(lean_object* v_00_u03b2_643_, lean_object* v_data_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2___redArg(v_data_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__3(lean_object* v_00_u03b2_646_, lean_object* v_a_647_, lean_object* v_b_648_, lean_object* v_x_649_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__3___redArg(v_a_647_, v_b_648_, v_x_649_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_651_, lean_object* v_i_652_, lean_object* v_source_653_, lean_object* v_target_654_){
_start:
{
lean_object* v___x_655_; 
v___x_655_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3___redArg(v_i_652_, v_source_653_, v_target_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3_spec__6(lean_object* v_00_u03b2_656_, lean_object* v_x_657_, lean_object* v_x_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__2_spec__3_spec__6___redArg(v_x_657_, v_x_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_Const_modify___at___00Std_DHashMap_Internal_Raw_u2080_Const_modify___at___00Mathlib_Tactic_Order_addFact_spec__0_spec__0(lean_object* v_fact_660_, lean_object* v_a_661_, lean_object* v_x_662_){
_start:
{
if (lean_obj_tag(v_x_662_) == 0)
{
lean_dec_ref(v_a_661_);
lean_dec_ref(v_fact_660_);
return v_x_662_;
}
else
{
lean_object* v_key_663_; lean_object* v_value_664_; lean_object* v_tail_665_; lean_object* v___x_667_; uint8_t v_isShared_668_; uint8_t v_isSharedCheck_678_; 
v_key_663_ = lean_ctor_get(v_x_662_, 0);
v_value_664_ = lean_ctor_get(v_x_662_, 1);
v_tail_665_ = lean_ctor_get(v_x_662_, 2);
v_isSharedCheck_678_ = !lean_is_exclusive(v_x_662_);
if (v_isSharedCheck_678_ == 0)
{
v___x_667_ = v_x_662_;
v_isShared_668_ = v_isSharedCheck_678_;
goto v_resetjp_666_;
}
else
{
lean_inc(v_tail_665_);
lean_inc(v_value_664_);
lean_inc(v_key_663_);
lean_dec(v_x_662_);
v___x_667_ = lean_box(0);
v_isShared_668_ = v_isSharedCheck_678_;
goto v_resetjp_666_;
}
v_resetjp_666_:
{
uint8_t v___x_669_; 
v___x_669_ = lean_expr_eqv(v_key_663_, v_a_661_);
if (v___x_669_ == 0)
{
lean_object* v___x_670_; lean_object* v___x_672_; 
v___x_670_ = lp_mathlib_Std_DHashMap_Internal_AssocList_Const_modify___at___00Std_DHashMap_Internal_Raw_u2080_Const_modify___at___00Mathlib_Tactic_Order_addFact_spec__0_spec__0(v_fact_660_, v_a_661_, v_tail_665_);
if (v_isShared_668_ == 0)
{
lean_ctor_set(v___x_667_, 2, v___x_670_);
v___x_672_ = v___x_667_;
goto v_reusejp_671_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v_key_663_);
lean_ctor_set(v_reuseFailAlloc_673_, 1, v_value_664_);
lean_ctor_set(v_reuseFailAlloc_673_, 2, v___x_670_);
v___x_672_ = v_reuseFailAlloc_673_;
goto v_reusejp_671_;
}
v_reusejp_671_:
{
return v___x_672_;
}
}
else
{
lean_object* v___x_674_; lean_object* v___x_676_; 
lean_dec(v_key_663_);
v___x_674_ = lean_array_push(v_value_664_, v_fact_660_);
if (v_isShared_668_ == 0)
{
lean_ctor_set(v___x_667_, 1, v___x_674_);
lean_ctor_set(v___x_667_, 0, v_a_661_);
v___x_676_ = v___x_667_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_a_661_);
lean_ctor_set(v_reuseFailAlloc_677_, 1, v___x_674_);
lean_ctor_set(v_reuseFailAlloc_677_, 2, v_tail_665_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_modify___at___00Mathlib_Tactic_Order_addFact_spec__0(lean_object* v_fact_679_, lean_object* v_m_680_, lean_object* v_a_681_){
_start:
{
lean_object* v_size_682_; lean_object* v_buckets_683_; lean_object* v___x_684_; uint64_t v___x_685_; uint64_t v___x_686_; uint64_t v___x_687_; uint64_t v_fold_688_; uint64_t v___x_689_; uint64_t v___x_690_; uint64_t v___x_691_; size_t v___x_692_; size_t v___x_693_; size_t v___x_694_; size_t v___x_695_; size_t v___x_696_; lean_object* v_bucket_697_; uint8_t v___x_698_; 
v_size_682_ = lean_ctor_get(v_m_680_, 0);
v_buckets_683_ = lean_ctor_get(v_m_680_, 1);
v___x_684_ = lean_array_get_size(v_buckets_683_);
v___x_685_ = l_Lean_Expr_hash(v_a_681_);
v___x_686_ = 32ULL;
v___x_687_ = lean_uint64_shift_right(v___x_685_, v___x_686_);
v_fold_688_ = lean_uint64_xor(v___x_685_, v___x_687_);
v___x_689_ = 16ULL;
v___x_690_ = lean_uint64_shift_right(v_fold_688_, v___x_689_);
v___x_691_ = lean_uint64_xor(v_fold_688_, v___x_690_);
v___x_692_ = lean_uint64_to_usize(v___x_691_);
v___x_693_ = lean_usize_of_nat(v___x_684_);
v___x_694_ = ((size_t)1ULL);
v___x_695_ = lean_usize_sub(v___x_693_, v___x_694_);
v___x_696_ = lean_usize_land(v___x_692_, v___x_695_);
v_bucket_697_ = lean_array_uget_borrowed(v_buckets_683_, v___x_696_);
v___x_698_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_addType_spec__1_spec__1___redArg(v_a_681_, v_bucket_697_);
if (v___x_698_ == 0)
{
lean_dec_ref(v_a_681_);
lean_dec_ref(v_fact_679_);
return v_m_680_;
}
else
{
lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_709_; 
lean_inc(v_bucket_697_);
lean_inc_ref(v_buckets_683_);
lean_inc(v_size_682_);
v_isSharedCheck_709_ = !lean_is_exclusive(v_m_680_);
if (v_isSharedCheck_709_ == 0)
{
lean_object* v_unused_710_; lean_object* v_unused_711_; 
v_unused_710_ = lean_ctor_get(v_m_680_, 1);
lean_dec(v_unused_710_);
v_unused_711_ = lean_ctor_get(v_m_680_, 0);
lean_dec(v_unused_711_);
v___x_700_ = v_m_680_;
v_isShared_701_ = v_isSharedCheck_709_;
goto v_resetjp_699_;
}
else
{
lean_dec(v_m_680_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_709_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v___x_702_; lean_object* v_buckets_703_; lean_object* v_bucket_704_; lean_object* v___x_705_; lean_object* v___x_707_; 
v___x_702_ = lean_box(0);
v_buckets_703_ = lean_array_uset(v_buckets_683_, v___x_696_, v___x_702_);
v_bucket_704_ = lp_mathlib_Std_DHashMap_Internal_AssocList_Const_modify___at___00Std_DHashMap_Internal_Raw_u2080_Const_modify___at___00Mathlib_Tactic_Order_addFact_spec__0_spec__0(v_fact_679_, v_a_681_, v_bucket_697_);
v___x_705_ = lean_array_uset(v_buckets_703_, v___x_696_, v_bucket_704_);
if (v_isShared_701_ == 0)
{
lean_ctor_set(v___x_700_, 1, v___x_705_);
v___x_707_ = v___x_700_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_708_; 
v_reuseFailAlloc_708_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_708_, 0, v_size_682_);
lean_ctor_set(v_reuseFailAlloc_708_, 1, v___x_705_);
v___x_707_ = v_reuseFailAlloc_708_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
return v___x_707_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(lean_object* v_type_712_, lean_object* v_fact_713_, lean_object* v_a_714_){
_start:
{
lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_716_ = lean_box(0);
v___x_717_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_modify___at___00Mathlib_Tactic_Order_addFact_spec__0(v_fact_713_, v_a_714_, v_type_712_);
v___x_718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_718_, 0, v___x_716_);
lean_ctor_set(v___x_718_, 1, v___x_717_);
v___x_719_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact___redArg___boxed(lean_object* v_type_720_, lean_object* v_fact_721_, lean_object* v_a_722_, lean_object* v_a_723_){
_start:
{
lean_object* v_res_724_; 
v_res_724_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_type_720_, v_fact_721_, v_a_722_);
return v_res_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact(lean_object* v_type_725_, lean_object* v_fact_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_, lean_object* v_a_732_, lean_object* v_a_733_){
_start:
{
lean_object* v___x_735_; 
v___x_735_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_type_725_, v_fact_726_, v_a_727_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addFact___boxed(lean_object* v_type_736_, lean_object* v_fact_737_, lean_object* v_a_738_, lean_object* v_a_739_, lean_object* v_a_740_, lean_object* v_a_741_, lean_object* v_a_742_, lean_object* v_a_743_, lean_object* v_a_744_, lean_object* v_a_745_){
_start:
{
lean_object* v_res_746_; 
v_res_746_ = lp_mathlib_Mathlib_Tactic_Order_addFact(v_type_736_, v_fact_737_, v_a_738_, v_a_739_, v_a_740_, v_a_741_, v_a_742_, v_a_743_, v_a_744_);
lean_dec(v_a_744_);
lean_dec_ref(v_a_743_);
lean_dec(v_a_742_);
lean_dec_ref(v_a_741_);
lean_dec(v_a_740_);
lean_dec_ref(v_a_739_);
return v_res_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(lean_object* v_e_747_, lean_object* v___y_748_){
_start:
{
uint8_t v___x_750_; 
v___x_750_ = l_Lean_Expr_hasMVar(v_e_747_);
if (v___x_750_ == 0)
{
lean_object* v___x_751_; 
v___x_751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_751_, 0, v_e_747_);
return v___x_751_;
}
else
{
lean_object* v___x_752_; lean_object* v_mctx_753_; lean_object* v___x_754_; lean_object* v_fst_755_; lean_object* v_snd_756_; lean_object* v___x_757_; lean_object* v_cache_758_; lean_object* v_zetaDeltaFVarIds_759_; lean_object* v_postponed_760_; lean_object* v_diag_761_; lean_object* v___x_763_; uint8_t v_isShared_764_; uint8_t v_isSharedCheck_770_; 
v___x_752_ = lean_st_ref_get(v___y_748_);
v_mctx_753_ = lean_ctor_get(v___x_752_, 0);
lean_inc_ref(v_mctx_753_);
lean_dec(v___x_752_);
v___x_754_ = l_Lean_instantiateMVarsCore(v_mctx_753_, v_e_747_);
v_fst_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_fst_755_);
v_snd_756_ = lean_ctor_get(v___x_754_, 1);
lean_inc(v_snd_756_);
lean_dec_ref(v___x_754_);
v___x_757_ = lean_st_ref_take(v___y_748_);
v_cache_758_ = lean_ctor_get(v___x_757_, 1);
v_zetaDeltaFVarIds_759_ = lean_ctor_get(v___x_757_, 2);
v_postponed_760_ = lean_ctor_get(v___x_757_, 3);
v_diag_761_ = lean_ctor_get(v___x_757_, 4);
v_isSharedCheck_770_ = !lean_is_exclusive(v___x_757_);
if (v_isSharedCheck_770_ == 0)
{
lean_object* v_unused_771_; 
v_unused_771_ = lean_ctor_get(v___x_757_, 0);
lean_dec(v_unused_771_);
v___x_763_ = v___x_757_;
v_isShared_764_ = v_isSharedCheck_770_;
goto v_resetjp_762_;
}
else
{
lean_inc(v_diag_761_);
lean_inc(v_postponed_760_);
lean_inc(v_zetaDeltaFVarIds_759_);
lean_inc(v_cache_758_);
lean_dec(v___x_757_);
v___x_763_ = lean_box(0);
v_isShared_764_ = v_isSharedCheck_770_;
goto v_resetjp_762_;
}
v_resetjp_762_:
{
lean_object* v___x_766_; 
if (v_isShared_764_ == 0)
{
lean_ctor_set(v___x_763_, 0, v_snd_756_);
v___x_766_ = v___x_763_;
goto v_reusejp_765_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v_snd_756_);
lean_ctor_set(v_reuseFailAlloc_769_, 1, v_cache_758_);
lean_ctor_set(v_reuseFailAlloc_769_, 2, v_zetaDeltaFVarIds_759_);
lean_ctor_set(v_reuseFailAlloc_769_, 3, v_postponed_760_);
lean_ctor_set(v_reuseFailAlloc_769_, 4, v_diag_761_);
v___x_766_ = v_reuseFailAlloc_769_;
goto v_reusejp_765_;
}
v_reusejp_765_:
{
lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_767_ = lean_st_ref_set(v___y_748_, v___x_766_);
v___x_768_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_768_, 0, v_fst_755_);
return v___x_768_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg___boxed(lean_object* v_e_772_, lean_object* v___y_773_, lean_object* v___y_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_e_772_, v___y_773_);
lean_dec(v___y_773_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0(lean_object* v_e_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
lean_object* v___x_782_; 
v___x_782_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_e_776_, v___y_778_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___boxed(lean_object* v_e_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_){
_start:
{
lean_object* v_res_789_; 
v_res_789_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0(v_e_783_, v___y_784_, v___y_785_, v___y_786_, v___y_787_);
lean_dec(v___y_787_);
lean_dec_ref(v___y_786_);
lean_dec(v___y_785_);
lean_dec_ref(v___y_784_);
return v_res_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(lean_object* v_k_790_, uint8_t v_allowLevelAssignments_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_791_, v_k_790_, v___y_792_, v___y_793_, v___y_794_, v___y_795_);
if (lean_obj_tag(v___x_797_) == 0)
{
lean_object* v_a_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_805_; 
v_a_798_ = lean_ctor_get(v___x_797_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_805_ == 0)
{
v___x_800_ = v___x_797_;
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_a_798_);
lean_dec(v___x_797_);
v___x_800_ = lean_box(0);
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
v_resetjp_799_:
{
lean_object* v___x_803_; 
if (v_isShared_801_ == 0)
{
v___x_803_ = v___x_800_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v_a_798_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
else
{
lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_813_; 
v_a_806_ = lean_ctor_get(v___x_797_, 0);
v_isSharedCheck_813_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_813_ == 0)
{
v___x_808_ = v___x_797_;
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_797_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v___x_811_; 
if (v_isShared_809_ == 0)
{
v___x_811_ = v___x_808_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v_a_806_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
return v___x_811_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg___boxed(lean_object* v_k_814_, lean_object* v_allowLevelAssignments_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_821_; lean_object* v_res_822_; 
v_allowLevelAssignments_boxed_821_ = lean_unbox(v_allowLevelAssignments_815_);
v_res_822_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v_k_814_, v_allowLevelAssignments_boxed_821_, v___y_816_, v___y_817_, v___y_818_, v___y_819_);
lean_dec(v___y_819_);
lean_dec_ref(v___y_818_);
lean_dec(v___y_817_);
lean_dec_ref(v___y_816_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1(lean_object* v_00_u03b1_823_, lean_object* v_k_824_, uint8_t v_allowLevelAssignments_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_){
_start:
{
lean_object* v___x_831_; 
v___x_831_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v_k_824_, v_allowLevelAssignments_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___boxed(lean_object* v_00_u03b1_832_, lean_object* v_k_833_, lean_object* v_allowLevelAssignments_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_840_; lean_object* v_res_841_; 
v_allowLevelAssignments_boxed_840_ = lean_unbox(v_allowLevelAssignments_834_);
v_res_841_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1(v_00_u03b1_832_, v_k_833_, v_allowLevelAssignments_boxed_840_, v___y_835_, v___y_836_, v___y_837_, v___y_838_);
lean_dec(v___y_838_);
lean_dec_ref(v___y_837_);
lean_dec(v___y_836_);
lean_dec_ref(v___y_835_);
return v_res_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0(lean_object* v___x_854_, uint8_t v___x_855_, lean_object* v___x_856_, lean_object* v___x_857_, lean_object* v_type_858_, lean_object* v_snd_859_, uint8_t v_fst_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_){
_start:
{
lean_object* v___x_866_; 
lean_inc(v___x_856_);
v___x_866_ = l_Lean_Meta_mkFreshExprMVar(v___x_854_, v___x_855_, v___x_856_, v___y_861_, v___y_862_, v___y_863_, v___y_864_);
if (lean_obj_tag(v___x_866_) == 0)
{
lean_object* v_a_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; 
v_a_867_ = lean_ctor_get(v___x_866_, 0);
lean_inc_n(v_a_867_, 2);
lean_dec_ref_known(v___x_866_, 1);
v___x_868_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__1));
lean_inc(v___x_857_);
v___x_869_ = l_Lean_Expr_const___override(v___x_868_, v___x_857_);
lean_inc_ref(v_type_858_);
v___x_870_ = l_Lean_Expr_app___override(v___x_869_, v_type_858_);
v___x_871_ = l_Lean_Expr_app___override(v___x_870_, v_a_867_);
v___x_872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_872_, 0, v___x_871_);
v___x_873_ = l_Lean_Meta_mkFreshExprMVar(v___x_872_, v___x_855_, v___x_856_, v___y_861_, v___y_862_, v___y_863_, v___y_864_);
if (lean_obj_tag(v___x_873_) == 0)
{
lean_object* v_a_874_; lean_object* v_keyedConfig_875_; uint8_t v_trackZetaDelta_876_; lean_object* v_zetaDeltaSet_877_; lean_object* v_lctx_878_; lean_object* v_localInstances_879_; lean_object* v_defEqCtx_x3f_880_; lean_object* v_synthPendingDepth_881_; lean_object* v_customCanUnfoldPredicate_x3f_882_; uint8_t v_univApprox_883_; uint8_t v_inTypeClassResolution_884_; uint8_t v_cacheInferType_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_953_; 
v_a_874_ = lean_ctor_get(v___x_873_, 0);
lean_inc(v_a_874_);
lean_dec_ref_known(v___x_873_, 1);
v_keyedConfig_875_ = lean_ctor_get(v___y_861_, 0);
v_trackZetaDelta_876_ = lean_ctor_get_uint8(v___y_861_, sizeof(void*)*7);
v_zetaDeltaSet_877_ = lean_ctor_get(v___y_861_, 1);
v_lctx_878_ = lean_ctor_get(v___y_861_, 2);
v_localInstances_879_ = lean_ctor_get(v___y_861_, 3);
v_defEqCtx_x3f_880_ = lean_ctor_get(v___y_861_, 4);
v_synthPendingDepth_881_ = lean_ctor_get(v___y_861_, 5);
v_customCanUnfoldPredicate_x3f_882_ = lean_ctor_get(v___y_861_, 6);
v_univApprox_883_ = lean_ctor_get_uint8(v___y_861_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_884_ = lean_ctor_get_uint8(v___y_861_, sizeof(void*)*7 + 2);
v_cacheInferType_885_ = lean_ctor_get_uint8(v___y_861_, sizeof(void*)*7 + 3);
v_isSharedCheck_953_ = !lean_is_exclusive(v___y_861_);
if (v_isSharedCheck_953_ == 0)
{
v___x_887_ = v___y_861_;
v_isShared_888_ = v_isSharedCheck_953_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_882_);
lean_inc(v_synthPendingDepth_881_);
lean_inc(v_defEqCtx_x3f_880_);
lean_inc(v_localInstances_879_);
lean_inc(v_lctx_878_);
lean_inc(v_zetaDeltaSet_877_);
lean_inc(v_keyedConfig_875_);
lean_dec(v___y_861_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_953_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; uint8_t v___x_898_; lean_object* v___x_899_; lean_object* v___x_901_; 
v___x_889_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__4));
lean_inc(v___x_857_);
v___x_890_ = l_Lean_Expr_const___override(v___x_889_, v___x_857_);
lean_inc_ref(v_type_858_);
v___x_891_ = l_Lean_Expr_app___override(v___x_890_, v_type_858_);
v___x_892_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___closed__6));
v___x_893_ = l_Lean_Expr_const___override(v___x_892_, v___x_857_);
v___x_894_ = l_Lean_Expr_app___override(v___x_893_, v_type_858_);
lean_inc(v_a_867_);
v___x_895_ = l_Lean_Expr_app___override(v___x_894_, v_a_867_);
lean_inc(v_a_874_);
v___x_896_ = l_Lean_Expr_app___override(v___x_895_, v_a_874_);
v___x_897_ = l_Lean_Expr_app___override(v___x_891_, v___x_896_);
v___x_898_ = 2;
v___x_899_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_898_, v_keyedConfig_875_);
if (v_isShared_888_ == 0)
{
lean_ctor_set(v___x_887_, 0, v___x_899_);
v___x_901_ = v___x_887_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_952_; 
v_reuseFailAlloc_952_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_952_, 0, v___x_899_);
lean_ctor_set(v_reuseFailAlloc_952_, 1, v_zetaDeltaSet_877_);
lean_ctor_set(v_reuseFailAlloc_952_, 2, v_lctx_878_);
lean_ctor_set(v_reuseFailAlloc_952_, 3, v_localInstances_879_);
lean_ctor_set(v_reuseFailAlloc_952_, 4, v_defEqCtx_x3f_880_);
lean_ctor_set(v_reuseFailAlloc_952_, 5, v_synthPendingDepth_881_);
lean_ctor_set(v_reuseFailAlloc_952_, 6, v_customCanUnfoldPredicate_x3f_882_);
lean_ctor_set_uint8(v_reuseFailAlloc_952_, sizeof(void*)*7, v_trackZetaDelta_876_);
lean_ctor_set_uint8(v_reuseFailAlloc_952_, sizeof(void*)*7 + 1, v_univApprox_883_);
lean_ctor_set_uint8(v_reuseFailAlloc_952_, sizeof(void*)*7 + 2, v_inTypeClassResolution_884_);
lean_ctor_set_uint8(v_reuseFailAlloc_952_, sizeof(void*)*7 + 3, v_cacheInferType_885_);
v___x_901_ = v_reuseFailAlloc_952_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
lean_object* v___x_902_; 
v___x_902_ = l_Lean_Meta_isExprDefEq(v___x_897_, v_snd_859_, v___x_901_, v___y_862_, v___y_863_, v___y_864_);
lean_dec_ref(v___x_901_);
if (lean_obj_tag(v___x_902_) == 0)
{
lean_object* v_a_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_943_; 
v_a_903_ = lean_ctor_get(v___x_902_, 0);
v_isSharedCheck_943_ = !lean_is_exclusive(v___x_902_);
if (v_isSharedCheck_943_ == 0)
{
v___x_905_ = v___x_902_;
v_isShared_906_ = v_isSharedCheck_943_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_a_903_);
lean_dec(v___x_902_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_943_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
uint8_t v___x_907_; 
v___x_907_ = lean_unbox(v_a_903_);
if (v___x_907_ == 0)
{
lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_912_; 
lean_dec(v_a_903_);
v___x_908_ = lean_box(v_fst_860_);
v___x_909_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_909_, 0, v_a_874_);
lean_ctor_set(v___x_909_, 1, v___x_908_);
v___x_910_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_910_, 0, v_a_867_);
lean_ctor_set(v___x_910_, 1, v___x_909_);
if (v_isShared_906_ == 0)
{
lean_ctor_set(v___x_905_, 0, v___x_910_);
v___x_912_ = v___x_905_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v___x_910_);
v___x_912_ = v_reuseFailAlloc_913_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
return v___x_912_;
}
}
else
{
lean_object* v___x_914_; 
lean_del_object(v___x_905_);
v___x_914_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_867_, v___y_862_);
if (lean_obj_tag(v___x_914_) == 0)
{
lean_object* v_a_915_; lean_object* v___x_916_; 
v_a_915_ = lean_ctor_get(v___x_914_, 0);
lean_inc(v_a_915_);
lean_dec_ref_known(v___x_914_, 1);
v___x_916_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_874_, v___y_862_);
if (lean_obj_tag(v___x_916_) == 0)
{
lean_object* v_a_917_; lean_object* v___x_919_; uint8_t v_isShared_920_; uint8_t v_isSharedCheck_926_; 
v_a_917_ = lean_ctor_get(v___x_916_, 0);
v_isSharedCheck_926_ = !lean_is_exclusive(v___x_916_);
if (v_isSharedCheck_926_ == 0)
{
v___x_919_ = v___x_916_;
v_isShared_920_ = v_isSharedCheck_926_;
goto v_resetjp_918_;
}
else
{
lean_inc(v_a_917_);
lean_dec(v___x_916_);
v___x_919_ = lean_box(0);
v_isShared_920_ = v_isSharedCheck_926_;
goto v_resetjp_918_;
}
v_resetjp_918_:
{
lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_924_; 
v___x_921_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_921_, 0, v_a_917_);
lean_ctor_set(v___x_921_, 1, v_a_903_);
v___x_922_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_922_, 0, v_a_915_);
lean_ctor_set(v___x_922_, 1, v___x_921_);
if (v_isShared_920_ == 0)
{
lean_ctor_set(v___x_919_, 0, v___x_922_);
v___x_924_ = v___x_919_;
goto v_reusejp_923_;
}
else
{
lean_object* v_reuseFailAlloc_925_; 
v_reuseFailAlloc_925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_925_, 0, v___x_922_);
v___x_924_ = v_reuseFailAlloc_925_;
goto v_reusejp_923_;
}
v_reusejp_923_:
{
return v___x_924_;
}
}
}
else
{
lean_object* v_a_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_934_; 
lean_dec(v_a_915_);
lean_dec(v_a_903_);
v_a_927_ = lean_ctor_get(v___x_916_, 0);
v_isSharedCheck_934_ = !lean_is_exclusive(v___x_916_);
if (v_isSharedCheck_934_ == 0)
{
v___x_929_ = v___x_916_;
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_a_927_);
lean_dec(v___x_916_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_932_; 
if (v_isShared_930_ == 0)
{
v___x_932_ = v___x_929_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_a_927_);
v___x_932_ = v_reuseFailAlloc_933_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
return v___x_932_;
}
}
}
}
else
{
lean_object* v_a_935_; lean_object* v___x_937_; uint8_t v_isShared_938_; uint8_t v_isSharedCheck_942_; 
lean_dec(v_a_903_);
lean_dec(v_a_874_);
v_a_935_ = lean_ctor_get(v___x_914_, 0);
v_isSharedCheck_942_ = !lean_is_exclusive(v___x_914_);
if (v_isSharedCheck_942_ == 0)
{
v___x_937_ = v___x_914_;
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
else
{
lean_inc(v_a_935_);
lean_dec(v___x_914_);
v___x_937_ = lean_box(0);
v_isShared_938_ = v_isSharedCheck_942_;
goto v_resetjp_936_;
}
v_resetjp_936_:
{
lean_object* v___x_940_; 
if (v_isShared_938_ == 0)
{
v___x_940_ = v___x_937_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v_a_935_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
return v___x_940_;
}
}
}
}
}
}
else
{
lean_object* v_a_944_; lean_object* v___x_946_; uint8_t v_isShared_947_; uint8_t v_isSharedCheck_951_; 
lean_dec(v_a_874_);
lean_dec(v_a_867_);
v_a_944_ = lean_ctor_get(v___x_902_, 0);
v_isSharedCheck_951_ = !lean_is_exclusive(v___x_902_);
if (v_isSharedCheck_951_ == 0)
{
v___x_946_ = v___x_902_;
v_isShared_947_ = v_isSharedCheck_951_;
goto v_resetjp_945_;
}
else
{
lean_inc(v_a_944_);
lean_dec(v___x_902_);
v___x_946_ = lean_box(0);
v_isShared_947_ = v_isSharedCheck_951_;
goto v_resetjp_945_;
}
v_resetjp_945_:
{
lean_object* v___x_949_; 
if (v_isShared_947_ == 0)
{
v___x_949_ = v___x_946_;
goto v_reusejp_948_;
}
else
{
lean_object* v_reuseFailAlloc_950_; 
v_reuseFailAlloc_950_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_950_, 0, v_a_944_);
v___x_949_ = v_reuseFailAlloc_950_;
goto v_reusejp_948_;
}
v_reusejp_948_:
{
return v___x_949_;
}
}
}
}
}
}
else
{
lean_object* v_a_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_961_; 
lean_dec(v_a_867_);
lean_dec_ref(v___y_861_);
lean_dec_ref(v_snd_859_);
lean_dec_ref(v_type_858_);
lean_dec(v___x_857_);
v_a_954_ = lean_ctor_get(v___x_873_, 0);
v_isSharedCheck_961_ = !lean_is_exclusive(v___x_873_);
if (v_isSharedCheck_961_ == 0)
{
v___x_956_ = v___x_873_;
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_a_954_);
lean_dec(v___x_873_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v___x_959_; 
if (v_isShared_957_ == 0)
{
v___x_959_ = v___x_956_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v_a_954_);
v___x_959_ = v_reuseFailAlloc_960_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
return v___x_959_;
}
}
}
}
else
{
lean_object* v_a_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_969_; 
lean_dec_ref(v___y_861_);
lean_dec_ref(v_snd_859_);
lean_dec_ref(v_type_858_);
lean_dec(v___x_857_);
lean_dec(v___x_856_);
v_a_962_ = lean_ctor_get(v___x_866_, 0);
v_isSharedCheck_969_ = !lean_is_exclusive(v___x_866_);
if (v_isSharedCheck_969_ == 0)
{
v___x_964_ = v___x_866_;
v_isShared_965_ = v_isSharedCheck_969_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_a_962_);
lean_dec(v___x_866_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_969_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v___x_967_; 
if (v_isShared_965_ == 0)
{
v___x_967_ = v___x_964_;
goto v_reusejp_966_;
}
else
{
lean_object* v_reuseFailAlloc_968_; 
v_reuseFailAlloc_968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_968_, 0, v_a_962_);
v___x_967_ = v_reuseFailAlloc_968_;
goto v_reusejp_966_;
}
v_reusejp_966_:
{
return v___x_967_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___boxed(lean_object* v___x_970_, lean_object* v___x_971_, lean_object* v___x_972_, lean_object* v___x_973_, lean_object* v_type_974_, lean_object* v_snd_975_, lean_object* v_fst_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_){
_start:
{
uint8_t v___x_27891__boxed_982_; uint8_t v_fst_27895__boxed_983_; lean_object* v_res_984_; 
v___x_27891__boxed_982_ = lean_unbox(v___x_971_);
v_fst_27895__boxed_983_ = lean_unbox(v_fst_976_);
v_res_984_ = lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0(v___x_970_, v___x_27891__boxed_982_, v___x_972_, v___x_973_, v_type_974_, v_snd_975_, v_fst_27895__boxed_983_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
lean_dec(v___y_978_);
return v_res_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1(lean_object* v___x_997_, uint8_t v___x_998_, lean_object* v___x_999_, lean_object* v___x_1000_, lean_object* v_type_1001_, lean_object* v_snd_1002_, uint8_t v_fst_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_){
_start:
{
lean_object* v___x_1009_; 
lean_inc(v___x_999_);
v___x_1009_ = l_Lean_Meta_mkFreshExprMVar(v___x_997_, v___x_998_, v___x_999_, v___y_1004_, v___y_1005_, v___y_1006_, v___y_1007_);
if (lean_obj_tag(v___x_1009_) == 0)
{
lean_object* v_a_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; 
v_a_1010_ = lean_ctor_get(v___x_1009_, 0);
lean_inc_n(v_a_1010_, 2);
lean_dec_ref_known(v___x_1009_, 1);
v___x_1011_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__1));
lean_inc(v___x_1000_);
v___x_1012_ = l_Lean_Expr_const___override(v___x_1011_, v___x_1000_);
lean_inc_ref(v_type_1001_);
v___x_1013_ = l_Lean_Expr_app___override(v___x_1012_, v_type_1001_);
v___x_1014_ = l_Lean_Expr_app___override(v___x_1013_, v_a_1010_);
v___x_1015_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1014_);
v___x_1016_ = l_Lean_Meta_mkFreshExprMVar(v___x_1015_, v___x_998_, v___x_999_, v___y_1004_, v___y_1005_, v___y_1006_, v___y_1007_);
if (lean_obj_tag(v___x_1016_) == 0)
{
lean_object* v_a_1017_; lean_object* v_keyedConfig_1018_; uint8_t v_trackZetaDelta_1019_; lean_object* v_zetaDeltaSet_1020_; lean_object* v_lctx_1021_; lean_object* v_localInstances_1022_; lean_object* v_defEqCtx_x3f_1023_; lean_object* v_synthPendingDepth_1024_; lean_object* v_customCanUnfoldPredicate_x3f_1025_; uint8_t v_univApprox_1026_; uint8_t v_inTypeClassResolution_1027_; uint8_t v_cacheInferType_1028_; lean_object* v___x_1030_; uint8_t v_isShared_1031_; uint8_t v_isSharedCheck_1096_; 
v_a_1017_ = lean_ctor_get(v___x_1016_, 0);
lean_inc(v_a_1017_);
lean_dec_ref_known(v___x_1016_, 1);
v_keyedConfig_1018_ = lean_ctor_get(v___y_1004_, 0);
v_trackZetaDelta_1019_ = lean_ctor_get_uint8(v___y_1004_, sizeof(void*)*7);
v_zetaDeltaSet_1020_ = lean_ctor_get(v___y_1004_, 1);
v_lctx_1021_ = lean_ctor_get(v___y_1004_, 2);
v_localInstances_1022_ = lean_ctor_get(v___y_1004_, 3);
v_defEqCtx_x3f_1023_ = lean_ctor_get(v___y_1004_, 4);
v_synthPendingDepth_1024_ = lean_ctor_get(v___y_1004_, 5);
v_customCanUnfoldPredicate_x3f_1025_ = lean_ctor_get(v___y_1004_, 6);
v_univApprox_1026_ = lean_ctor_get_uint8(v___y_1004_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1027_ = lean_ctor_get_uint8(v___y_1004_, sizeof(void*)*7 + 2);
v_cacheInferType_1028_ = lean_ctor_get_uint8(v___y_1004_, sizeof(void*)*7 + 3);
v_isSharedCheck_1096_ = !lean_is_exclusive(v___y_1004_);
if (v_isSharedCheck_1096_ == 0)
{
v___x_1030_ = v___y_1004_;
v_isShared_1031_ = v_isSharedCheck_1096_;
goto v_resetjp_1029_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1025_);
lean_inc(v_synthPendingDepth_1024_);
lean_inc(v_defEqCtx_x3f_1023_);
lean_inc(v_localInstances_1022_);
lean_inc(v_lctx_1021_);
lean_inc(v_zetaDeltaSet_1020_);
lean_inc(v_keyedConfig_1018_);
lean_dec(v___y_1004_);
v___x_1030_ = lean_box(0);
v_isShared_1031_ = v_isSharedCheck_1096_;
goto v_resetjp_1029_;
}
v_resetjp_1029_:
{
lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; uint8_t v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1044_; 
v___x_1032_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__4));
lean_inc(v___x_1000_);
v___x_1033_ = l_Lean_Expr_const___override(v___x_1032_, v___x_1000_);
lean_inc_ref(v_type_1001_);
v___x_1034_ = l_Lean_Expr_app___override(v___x_1033_, v_type_1001_);
v___x_1035_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___closed__6));
v___x_1036_ = l_Lean_Expr_const___override(v___x_1035_, v___x_1000_);
v___x_1037_ = l_Lean_Expr_app___override(v___x_1036_, v_type_1001_);
lean_inc(v_a_1010_);
v___x_1038_ = l_Lean_Expr_app___override(v___x_1037_, v_a_1010_);
lean_inc(v_a_1017_);
v___x_1039_ = l_Lean_Expr_app___override(v___x_1038_, v_a_1017_);
v___x_1040_ = l_Lean_Expr_app___override(v___x_1034_, v___x_1039_);
v___x_1041_ = 2;
v___x_1042_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1041_, v_keyedConfig_1018_);
if (v_isShared_1031_ == 0)
{
lean_ctor_set(v___x_1030_, 0, v___x_1042_);
v___x_1044_ = v___x_1030_;
goto v_reusejp_1043_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v___x_1042_);
lean_ctor_set(v_reuseFailAlloc_1095_, 1, v_zetaDeltaSet_1020_);
lean_ctor_set(v_reuseFailAlloc_1095_, 2, v_lctx_1021_);
lean_ctor_set(v_reuseFailAlloc_1095_, 3, v_localInstances_1022_);
lean_ctor_set(v_reuseFailAlloc_1095_, 4, v_defEqCtx_x3f_1023_);
lean_ctor_set(v_reuseFailAlloc_1095_, 5, v_synthPendingDepth_1024_);
lean_ctor_set(v_reuseFailAlloc_1095_, 6, v_customCanUnfoldPredicate_x3f_1025_);
lean_ctor_set_uint8(v_reuseFailAlloc_1095_, sizeof(void*)*7, v_trackZetaDelta_1019_);
lean_ctor_set_uint8(v_reuseFailAlloc_1095_, sizeof(void*)*7 + 1, v_univApprox_1026_);
lean_ctor_set_uint8(v_reuseFailAlloc_1095_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1027_);
lean_ctor_set_uint8(v_reuseFailAlloc_1095_, sizeof(void*)*7 + 3, v_cacheInferType_1028_);
v___x_1044_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1043_;
}
v_reusejp_1043_:
{
lean_object* v___x_1045_; 
v___x_1045_ = l_Lean_Meta_isExprDefEq(v___x_1040_, v_snd_1002_, v___x_1044_, v___y_1005_, v___y_1006_, v___y_1007_);
lean_dec_ref(v___x_1044_);
if (lean_obj_tag(v___x_1045_) == 0)
{
lean_object* v_a_1046_; lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1086_; 
v_a_1046_ = lean_ctor_get(v___x_1045_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v___x_1045_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1048_ = v___x_1045_;
v_isShared_1049_ = v_isSharedCheck_1086_;
goto v_resetjp_1047_;
}
else
{
lean_inc(v_a_1046_);
lean_dec(v___x_1045_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1086_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
uint8_t v___x_1050_; 
v___x_1050_ = lean_unbox(v_a_1046_);
if (v___x_1050_ == 0)
{
lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; lean_object* v___x_1055_; 
lean_dec(v_a_1046_);
v___x_1051_ = lean_box(v_fst_1003_);
v___x_1052_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1052_, 0, v_a_1017_);
lean_ctor_set(v___x_1052_, 1, v___x_1051_);
v___x_1053_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1053_, 0, v_a_1010_);
lean_ctor_set(v___x_1053_, 1, v___x_1052_);
if (v_isShared_1049_ == 0)
{
lean_ctor_set(v___x_1048_, 0, v___x_1053_);
v___x_1055_ = v___x_1048_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1056_; 
v_reuseFailAlloc_1056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1056_, 0, v___x_1053_);
v___x_1055_ = v_reuseFailAlloc_1056_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
return v___x_1055_;
}
}
else
{
lean_object* v___x_1057_; 
lean_del_object(v___x_1048_);
v___x_1057_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1010_, v___y_1005_);
if (lean_obj_tag(v___x_1057_) == 0)
{
lean_object* v_a_1058_; lean_object* v___x_1059_; 
v_a_1058_ = lean_ctor_get(v___x_1057_, 0);
lean_inc(v_a_1058_);
lean_dec_ref_known(v___x_1057_, 1);
v___x_1059_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1017_, v___y_1005_);
if (lean_obj_tag(v___x_1059_) == 0)
{
lean_object* v_a_1060_; lean_object* v___x_1062_; uint8_t v_isShared_1063_; uint8_t v_isSharedCheck_1069_; 
v_a_1060_ = lean_ctor_get(v___x_1059_, 0);
v_isSharedCheck_1069_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1069_ == 0)
{
v___x_1062_ = v___x_1059_;
v_isShared_1063_ = v_isSharedCheck_1069_;
goto v_resetjp_1061_;
}
else
{
lean_inc(v_a_1060_);
lean_dec(v___x_1059_);
v___x_1062_ = lean_box(0);
v_isShared_1063_ = v_isSharedCheck_1069_;
goto v_resetjp_1061_;
}
v_resetjp_1061_:
{
lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1067_; 
v___x_1064_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1064_, 0, v_a_1060_);
lean_ctor_set(v___x_1064_, 1, v_a_1046_);
v___x_1065_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1065_, 0, v_a_1058_);
lean_ctor_set(v___x_1065_, 1, v___x_1064_);
if (v_isShared_1063_ == 0)
{
lean_ctor_set(v___x_1062_, 0, v___x_1065_);
v___x_1067_ = v___x_1062_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v___x_1065_);
v___x_1067_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
return v___x_1067_;
}
}
}
else
{
lean_object* v_a_1070_; lean_object* v___x_1072_; uint8_t v_isShared_1073_; uint8_t v_isSharedCheck_1077_; 
lean_dec(v_a_1058_);
lean_dec(v_a_1046_);
v_a_1070_ = lean_ctor_get(v___x_1059_, 0);
v_isSharedCheck_1077_ = !lean_is_exclusive(v___x_1059_);
if (v_isSharedCheck_1077_ == 0)
{
v___x_1072_ = v___x_1059_;
v_isShared_1073_ = v_isSharedCheck_1077_;
goto v_resetjp_1071_;
}
else
{
lean_inc(v_a_1070_);
lean_dec(v___x_1059_);
v___x_1072_ = lean_box(0);
v_isShared_1073_ = v_isSharedCheck_1077_;
goto v_resetjp_1071_;
}
v_resetjp_1071_:
{
lean_object* v___x_1075_; 
if (v_isShared_1073_ == 0)
{
v___x_1075_ = v___x_1072_;
goto v_reusejp_1074_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v_a_1070_);
v___x_1075_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1074_;
}
v_reusejp_1074_:
{
return v___x_1075_;
}
}
}
}
else
{
lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1085_; 
lean_dec(v_a_1046_);
lean_dec(v_a_1017_);
v_a_1078_ = lean_ctor_get(v___x_1057_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_1057_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1080_ = v___x_1057_;
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1057_);
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
}
}
else
{
lean_object* v_a_1087_; lean_object* v___x_1089_; uint8_t v_isShared_1090_; uint8_t v_isSharedCheck_1094_; 
lean_dec(v_a_1017_);
lean_dec(v_a_1010_);
v_a_1087_ = lean_ctor_get(v___x_1045_, 0);
v_isSharedCheck_1094_ = !lean_is_exclusive(v___x_1045_);
if (v_isSharedCheck_1094_ == 0)
{
v___x_1089_ = v___x_1045_;
v_isShared_1090_ = v_isSharedCheck_1094_;
goto v_resetjp_1088_;
}
else
{
lean_inc(v_a_1087_);
lean_dec(v___x_1045_);
v___x_1089_ = lean_box(0);
v_isShared_1090_ = v_isSharedCheck_1094_;
goto v_resetjp_1088_;
}
v_resetjp_1088_:
{
lean_object* v___x_1092_; 
if (v_isShared_1090_ == 0)
{
v___x_1092_ = v___x_1089_;
goto v_reusejp_1091_;
}
else
{
lean_object* v_reuseFailAlloc_1093_; 
v_reuseFailAlloc_1093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1093_, 0, v_a_1087_);
v___x_1092_ = v_reuseFailAlloc_1093_;
goto v_reusejp_1091_;
}
v_reusejp_1091_:
{
return v___x_1092_;
}
}
}
}
}
}
else
{
lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1104_; 
lean_dec(v_a_1010_);
lean_dec_ref(v___y_1004_);
lean_dec_ref(v_snd_1002_);
lean_dec_ref(v_type_1001_);
lean_dec(v___x_1000_);
v_a_1097_ = lean_ctor_get(v___x_1016_, 0);
v_isSharedCheck_1104_ = !lean_is_exclusive(v___x_1016_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1099_ = v___x_1016_;
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1016_);
v___x_1099_ = lean_box(0);
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
v_resetjp_1098_:
{
lean_object* v___x_1102_; 
if (v_isShared_1100_ == 0)
{
v___x_1102_ = v___x_1099_;
goto v_reusejp_1101_;
}
else
{
lean_object* v_reuseFailAlloc_1103_; 
v_reuseFailAlloc_1103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1103_, 0, v_a_1097_);
v___x_1102_ = v_reuseFailAlloc_1103_;
goto v_reusejp_1101_;
}
v_reusejp_1101_:
{
return v___x_1102_;
}
}
}
}
else
{
lean_object* v_a_1105_; lean_object* v___x_1107_; uint8_t v_isShared_1108_; uint8_t v_isSharedCheck_1112_; 
lean_dec_ref(v___y_1004_);
lean_dec_ref(v_snd_1002_);
lean_dec_ref(v_type_1001_);
lean_dec(v___x_1000_);
lean_dec(v___x_999_);
v_a_1105_ = lean_ctor_get(v___x_1009_, 0);
v_isSharedCheck_1112_ = !lean_is_exclusive(v___x_1009_);
if (v_isSharedCheck_1112_ == 0)
{
v___x_1107_ = v___x_1009_;
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
else
{
lean_inc(v_a_1105_);
lean_dec(v___x_1009_);
v___x_1107_ = lean_box(0);
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
v_resetjp_1106_:
{
lean_object* v___x_1110_; 
if (v_isShared_1108_ == 0)
{
v___x_1110_ = v___x_1107_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1111_; 
v_reuseFailAlloc_1111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1111_, 0, v_a_1105_);
v___x_1110_ = v_reuseFailAlloc_1111_;
goto v_reusejp_1109_;
}
v_reusejp_1109_:
{
return v___x_1110_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___boxed(lean_object* v___x_1113_, lean_object* v___x_1114_, lean_object* v___x_1115_, lean_object* v___x_1116_, lean_object* v_type_1117_, lean_object* v_snd_1118_, lean_object* v_fst_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_){
_start:
{
uint8_t v___x_28146__boxed_1125_; uint8_t v_fst_28150__boxed_1126_; lean_object* v_res_1127_; 
v___x_28146__boxed_1125_ = lean_unbox(v___x_1114_);
v_fst_28150__boxed_1126_ = lean_unbox(v_fst_1119_);
v_res_1127_ = lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1(v___x_1113_, v___x_28146__boxed_1125_, v___x_1115_, v___x_1116_, v_type_1117_, v_snd_1118_, v_fst_28150__boxed_1126_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_);
lean_dec(v___y_1123_);
lean_dec_ref(v___y_1122_);
lean_dec(v___y_1121_);
return v_res_1127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2(lean_object* v___x_1134_, uint8_t v___x_1135_, lean_object* v___x_1136_, lean_object* v_type_1137_, lean_object* v___x_1138_, lean_object* v___x_1139_, lean_object* v_snd_1140_, uint8_t v_fst_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_){
_start:
{
lean_object* v___x_1147_; 
lean_inc(v___x_1136_);
v___x_1147_ = l_Lean_Meta_mkFreshExprMVar(v___x_1134_, v___x_1135_, v___x_1136_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_);
if (lean_obj_tag(v___x_1147_) == 0)
{
lean_object* v_a_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; 
v_a_1148_ = lean_ctor_get(v___x_1147_, 0);
lean_inc(v_a_1148_);
lean_dec_ref_known(v___x_1147_, 1);
lean_inc_ref(v_type_1137_);
v___x_1149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1149_, 0, v_type_1137_);
lean_inc(v___x_1136_);
lean_inc_ref(v___x_1149_);
v___x_1150_ = l_Lean_Meta_mkFreshExprMVar(v___x_1149_, v___x_1135_, v___x_1136_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_);
if (lean_obj_tag(v___x_1150_) == 0)
{
lean_object* v_a_1151_; lean_object* v___x_1152_; 
v_a_1151_ = lean_ctor_get(v___x_1150_, 0);
lean_inc(v_a_1151_);
lean_dec_ref_known(v___x_1150_, 1);
v___x_1152_ = l_Lean_Meta_mkFreshExprMVar(v___x_1149_, v___x_1135_, v___x_1136_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_);
if (lean_obj_tag(v___x_1152_) == 0)
{
lean_object* v_a_1153_; lean_object* v_keyedConfig_1154_; uint8_t v_trackZetaDelta_1155_; lean_object* v_zetaDeltaSet_1156_; lean_object* v_lctx_1157_; lean_object* v_localInstances_1158_; lean_object* v_defEqCtx_x3f_1159_; lean_object* v_synthPendingDepth_1160_; lean_object* v_customCanUnfoldPredicate_x3f_1161_; uint8_t v_univApprox_1162_; uint8_t v_inTypeClassResolution_1163_; uint8_t v_cacheInferType_1164_; lean_object* v___x_1166_; uint8_t v_isShared_1167_; uint8_t v_isSharedCheck_1246_; 
v_a_1153_ = lean_ctor_get(v___x_1152_, 0);
lean_inc(v_a_1153_);
lean_dec_ref_known(v___x_1152_, 1);
v_keyedConfig_1154_ = lean_ctor_get(v___y_1142_, 0);
v_trackZetaDelta_1155_ = lean_ctor_get_uint8(v___y_1142_, sizeof(void*)*7);
v_zetaDeltaSet_1156_ = lean_ctor_get(v___y_1142_, 1);
v_lctx_1157_ = lean_ctor_get(v___y_1142_, 2);
v_localInstances_1158_ = lean_ctor_get(v___y_1142_, 3);
v_defEqCtx_x3f_1159_ = lean_ctor_get(v___y_1142_, 4);
v_synthPendingDepth_1160_ = lean_ctor_get(v___y_1142_, 5);
v_customCanUnfoldPredicate_x3f_1161_ = lean_ctor_get(v___y_1142_, 6);
v_univApprox_1162_ = lean_ctor_get_uint8(v___y_1142_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1163_ = lean_ctor_get_uint8(v___y_1142_, sizeof(void*)*7 + 2);
v_cacheInferType_1164_ = lean_ctor_get_uint8(v___y_1142_, sizeof(void*)*7 + 3);
v_isSharedCheck_1246_ = !lean_is_exclusive(v___y_1142_);
if (v_isSharedCheck_1246_ == 0)
{
v___x_1166_ = v___y_1142_;
v_isShared_1167_ = v_isSharedCheck_1246_;
goto v_resetjp_1165_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1161_);
lean_inc(v_synthPendingDepth_1160_);
lean_inc(v_defEqCtx_x3f_1159_);
lean_inc(v_localInstances_1158_);
lean_inc(v_lctx_1157_);
lean_inc(v_zetaDeltaSet_1156_);
lean_inc(v_keyedConfig_1154_);
lean_dec(v___y_1142_);
v___x_1166_ = lean_box(0);
v_isShared_1167_ = v_isSharedCheck_1246_;
goto v_resetjp_1165_;
}
v_resetjp_1165_:
{
lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; uint8_t v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1182_; 
v___x_1168_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__2));
lean_inc(v___x_1138_);
v___x_1169_ = l_Lean_Expr_const___override(v___x_1168_, v___x_1138_);
lean_inc_ref(v_type_1137_);
v___x_1170_ = l_Lean_Expr_app___override(v___x_1169_, v_type_1137_);
v___x_1171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___closed__3));
v___x_1172_ = l_Lean_Name_mkStr2(v___x_1139_, v___x_1171_);
v___x_1173_ = l_Lean_Expr_const___override(v___x_1172_, v___x_1138_);
v___x_1174_ = l_Lean_Expr_app___override(v___x_1173_, v_type_1137_);
lean_inc(v_a_1148_);
v___x_1175_ = l_Lean_Expr_app___override(v___x_1174_, v_a_1148_);
v___x_1176_ = l_Lean_Expr_app___override(v___x_1170_, v___x_1175_);
lean_inc(v_a_1151_);
v___x_1177_ = l_Lean_Expr_app___override(v___x_1176_, v_a_1151_);
lean_inc(v_a_1153_);
v___x_1178_ = l_Lean_Expr_app___override(v___x_1177_, v_a_1153_);
v___x_1179_ = 2;
v___x_1180_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1179_, v_keyedConfig_1154_);
if (v_isShared_1167_ == 0)
{
lean_ctor_set(v___x_1166_, 0, v___x_1180_);
v___x_1182_ = v___x_1166_;
goto v_reusejp_1181_;
}
else
{
lean_object* v_reuseFailAlloc_1245_; 
v_reuseFailAlloc_1245_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1245_, 0, v___x_1180_);
lean_ctor_set(v_reuseFailAlloc_1245_, 1, v_zetaDeltaSet_1156_);
lean_ctor_set(v_reuseFailAlloc_1245_, 2, v_lctx_1157_);
lean_ctor_set(v_reuseFailAlloc_1245_, 3, v_localInstances_1158_);
lean_ctor_set(v_reuseFailAlloc_1245_, 4, v_defEqCtx_x3f_1159_);
lean_ctor_set(v_reuseFailAlloc_1245_, 5, v_synthPendingDepth_1160_);
lean_ctor_set(v_reuseFailAlloc_1245_, 6, v_customCanUnfoldPredicate_x3f_1161_);
lean_ctor_set_uint8(v_reuseFailAlloc_1245_, sizeof(void*)*7, v_trackZetaDelta_1155_);
lean_ctor_set_uint8(v_reuseFailAlloc_1245_, sizeof(void*)*7 + 1, v_univApprox_1162_);
lean_ctor_set_uint8(v_reuseFailAlloc_1245_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1163_);
lean_ctor_set_uint8(v_reuseFailAlloc_1245_, sizeof(void*)*7 + 3, v_cacheInferType_1164_);
v___x_1182_ = v_reuseFailAlloc_1245_;
goto v_reusejp_1181_;
}
v_reusejp_1181_:
{
lean_object* v___x_1183_; 
v___x_1183_ = l_Lean_Meta_isExprDefEq(v___x_1178_, v_snd_1140_, v___x_1182_, v___y_1143_, v___y_1144_, v___y_1145_);
lean_dec_ref(v___x_1182_);
if (lean_obj_tag(v___x_1183_) == 0)
{
lean_object* v_a_1184_; lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1236_; 
v_a_1184_ = lean_ctor_get(v___x_1183_, 0);
v_isSharedCheck_1236_ = !lean_is_exclusive(v___x_1183_);
if (v_isSharedCheck_1236_ == 0)
{
v___x_1186_ = v___x_1183_;
v_isShared_1187_ = v_isSharedCheck_1236_;
goto v_resetjp_1185_;
}
else
{
lean_inc(v_a_1184_);
lean_dec(v___x_1183_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1236_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
uint8_t v___x_1188_; 
v___x_1188_ = lean_unbox(v_a_1184_);
if (v___x_1188_ == 0)
{
lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1194_; 
lean_dec(v_a_1184_);
v___x_1189_ = lean_box(v_fst_1141_);
v___x_1190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1190_, 0, v_a_1153_);
lean_ctor_set(v___x_1190_, 1, v___x_1189_);
v___x_1191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1191_, 0, v_a_1151_);
lean_ctor_set(v___x_1191_, 1, v___x_1190_);
v___x_1192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1192_, 0, v_a_1148_);
lean_ctor_set(v___x_1192_, 1, v___x_1191_);
if (v_isShared_1187_ == 0)
{
lean_ctor_set(v___x_1186_, 0, v___x_1192_);
v___x_1194_ = v___x_1186_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v___x_1192_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
else
{
lean_object* v___x_1196_; 
lean_del_object(v___x_1186_);
v___x_1196_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1148_, v___y_1143_);
if (lean_obj_tag(v___x_1196_) == 0)
{
lean_object* v_a_1197_; lean_object* v___x_1198_; 
v_a_1197_ = lean_ctor_get(v___x_1196_, 0);
lean_inc(v_a_1197_);
lean_dec_ref_known(v___x_1196_, 1);
v___x_1198_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1151_, v___y_1143_);
if (lean_obj_tag(v___x_1198_) == 0)
{
lean_object* v_a_1199_; lean_object* v___x_1200_; 
v_a_1199_ = lean_ctor_get(v___x_1198_, 0);
lean_inc(v_a_1199_);
lean_dec_ref_known(v___x_1198_, 1);
v___x_1200_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1153_, v___y_1143_);
if (lean_obj_tag(v___x_1200_) == 0)
{
lean_object* v_a_1201_; lean_object* v___x_1203_; uint8_t v_isShared_1204_; uint8_t v_isSharedCheck_1211_; 
v_a_1201_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1211_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1211_ == 0)
{
v___x_1203_ = v___x_1200_;
v_isShared_1204_ = v_isSharedCheck_1211_;
goto v_resetjp_1202_;
}
else
{
lean_inc(v_a_1201_);
lean_dec(v___x_1200_);
v___x_1203_ = lean_box(0);
v_isShared_1204_ = v_isSharedCheck_1211_;
goto v_resetjp_1202_;
}
v_resetjp_1202_:
{
lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1209_; 
v___x_1205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1205_, 0, v_a_1201_);
lean_ctor_set(v___x_1205_, 1, v_a_1184_);
v___x_1206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1206_, 0, v_a_1199_);
lean_ctor_set(v___x_1206_, 1, v___x_1205_);
v___x_1207_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1207_, 0, v_a_1197_);
lean_ctor_set(v___x_1207_, 1, v___x_1206_);
if (v_isShared_1204_ == 0)
{
lean_ctor_set(v___x_1203_, 0, v___x_1207_);
v___x_1209_ = v___x_1203_;
goto v_reusejp_1208_;
}
else
{
lean_object* v_reuseFailAlloc_1210_; 
v_reuseFailAlloc_1210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1210_, 0, v___x_1207_);
v___x_1209_ = v_reuseFailAlloc_1210_;
goto v_reusejp_1208_;
}
v_reusejp_1208_:
{
return v___x_1209_;
}
}
}
else
{
lean_object* v_a_1212_; lean_object* v___x_1214_; uint8_t v_isShared_1215_; uint8_t v_isSharedCheck_1219_; 
lean_dec(v_a_1199_);
lean_dec(v_a_1197_);
lean_dec(v_a_1184_);
v_a_1212_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1219_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1219_ == 0)
{
v___x_1214_ = v___x_1200_;
v_isShared_1215_ = v_isSharedCheck_1219_;
goto v_resetjp_1213_;
}
else
{
lean_inc(v_a_1212_);
lean_dec(v___x_1200_);
v___x_1214_ = lean_box(0);
v_isShared_1215_ = v_isSharedCheck_1219_;
goto v_resetjp_1213_;
}
v_resetjp_1213_:
{
lean_object* v___x_1217_; 
if (v_isShared_1215_ == 0)
{
v___x_1217_ = v___x_1214_;
goto v_reusejp_1216_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v_a_1212_);
v___x_1217_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1216_;
}
v_reusejp_1216_:
{
return v___x_1217_;
}
}
}
}
else
{
lean_object* v_a_1220_; lean_object* v___x_1222_; uint8_t v_isShared_1223_; uint8_t v_isSharedCheck_1227_; 
lean_dec(v_a_1197_);
lean_dec(v_a_1184_);
lean_dec(v_a_1153_);
v_a_1220_ = lean_ctor_get(v___x_1198_, 0);
v_isSharedCheck_1227_ = !lean_is_exclusive(v___x_1198_);
if (v_isSharedCheck_1227_ == 0)
{
v___x_1222_ = v___x_1198_;
v_isShared_1223_ = v_isSharedCheck_1227_;
goto v_resetjp_1221_;
}
else
{
lean_inc(v_a_1220_);
lean_dec(v___x_1198_);
v___x_1222_ = lean_box(0);
v_isShared_1223_ = v_isSharedCheck_1227_;
goto v_resetjp_1221_;
}
v_resetjp_1221_:
{
lean_object* v___x_1225_; 
if (v_isShared_1223_ == 0)
{
v___x_1225_ = v___x_1222_;
goto v_reusejp_1224_;
}
else
{
lean_object* v_reuseFailAlloc_1226_; 
v_reuseFailAlloc_1226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1226_, 0, v_a_1220_);
v___x_1225_ = v_reuseFailAlloc_1226_;
goto v_reusejp_1224_;
}
v_reusejp_1224_:
{
return v___x_1225_;
}
}
}
}
else
{
lean_object* v_a_1228_; lean_object* v___x_1230_; uint8_t v_isShared_1231_; uint8_t v_isSharedCheck_1235_; 
lean_dec(v_a_1184_);
lean_dec(v_a_1153_);
lean_dec(v_a_1151_);
v_a_1228_ = lean_ctor_get(v___x_1196_, 0);
v_isSharedCheck_1235_ = !lean_is_exclusive(v___x_1196_);
if (v_isSharedCheck_1235_ == 0)
{
v___x_1230_ = v___x_1196_;
v_isShared_1231_ = v_isSharedCheck_1235_;
goto v_resetjp_1229_;
}
else
{
lean_inc(v_a_1228_);
lean_dec(v___x_1196_);
v___x_1230_ = lean_box(0);
v_isShared_1231_ = v_isSharedCheck_1235_;
goto v_resetjp_1229_;
}
v_resetjp_1229_:
{
lean_object* v___x_1233_; 
if (v_isShared_1231_ == 0)
{
v___x_1233_ = v___x_1230_;
goto v_reusejp_1232_;
}
else
{
lean_object* v_reuseFailAlloc_1234_; 
v_reuseFailAlloc_1234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1234_, 0, v_a_1228_);
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
}
}
else
{
lean_object* v_a_1237_; lean_object* v___x_1239_; uint8_t v_isShared_1240_; uint8_t v_isSharedCheck_1244_; 
lean_dec(v_a_1153_);
lean_dec(v_a_1151_);
lean_dec(v_a_1148_);
v_a_1237_ = lean_ctor_get(v___x_1183_, 0);
v_isSharedCheck_1244_ = !lean_is_exclusive(v___x_1183_);
if (v_isSharedCheck_1244_ == 0)
{
v___x_1239_ = v___x_1183_;
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
else
{
lean_inc(v_a_1237_);
lean_dec(v___x_1183_);
v___x_1239_ = lean_box(0);
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
v_resetjp_1238_:
{
lean_object* v___x_1242_; 
if (v_isShared_1240_ == 0)
{
v___x_1242_ = v___x_1239_;
goto v_reusejp_1241_;
}
else
{
lean_object* v_reuseFailAlloc_1243_; 
v_reuseFailAlloc_1243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1243_, 0, v_a_1237_);
v___x_1242_ = v_reuseFailAlloc_1243_;
goto v_reusejp_1241_;
}
v_reusejp_1241_:
{
return v___x_1242_;
}
}
}
}
}
}
else
{
lean_object* v_a_1247_; lean_object* v___x_1249_; uint8_t v_isShared_1250_; uint8_t v_isSharedCheck_1254_; 
lean_dec(v_a_1151_);
lean_dec(v_a_1148_);
lean_dec_ref(v___y_1142_);
lean_dec_ref(v_snd_1140_);
lean_dec_ref(v___x_1139_);
lean_dec(v___x_1138_);
lean_dec_ref(v_type_1137_);
v_a_1247_ = lean_ctor_get(v___x_1152_, 0);
v_isSharedCheck_1254_ = !lean_is_exclusive(v___x_1152_);
if (v_isSharedCheck_1254_ == 0)
{
v___x_1249_ = v___x_1152_;
v_isShared_1250_ = v_isSharedCheck_1254_;
goto v_resetjp_1248_;
}
else
{
lean_inc(v_a_1247_);
lean_dec(v___x_1152_);
v___x_1249_ = lean_box(0);
v_isShared_1250_ = v_isSharedCheck_1254_;
goto v_resetjp_1248_;
}
v_resetjp_1248_:
{
lean_object* v___x_1252_; 
if (v_isShared_1250_ == 0)
{
v___x_1252_ = v___x_1249_;
goto v_reusejp_1251_;
}
else
{
lean_object* v_reuseFailAlloc_1253_; 
v_reuseFailAlloc_1253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1253_, 0, v_a_1247_);
v___x_1252_ = v_reuseFailAlloc_1253_;
goto v_reusejp_1251_;
}
v_reusejp_1251_:
{
return v___x_1252_;
}
}
}
}
else
{
lean_object* v_a_1255_; lean_object* v___x_1257_; uint8_t v_isShared_1258_; uint8_t v_isSharedCheck_1262_; 
lean_dec_ref_known(v___x_1149_, 1);
lean_dec(v_a_1148_);
lean_dec_ref(v___y_1142_);
lean_dec_ref(v_snd_1140_);
lean_dec_ref(v___x_1139_);
lean_dec(v___x_1138_);
lean_dec_ref(v_type_1137_);
lean_dec(v___x_1136_);
v_a_1255_ = lean_ctor_get(v___x_1150_, 0);
v_isSharedCheck_1262_ = !lean_is_exclusive(v___x_1150_);
if (v_isSharedCheck_1262_ == 0)
{
v___x_1257_ = v___x_1150_;
v_isShared_1258_ = v_isSharedCheck_1262_;
goto v_resetjp_1256_;
}
else
{
lean_inc(v_a_1255_);
lean_dec(v___x_1150_);
v___x_1257_ = lean_box(0);
v_isShared_1258_ = v_isSharedCheck_1262_;
goto v_resetjp_1256_;
}
v_resetjp_1256_:
{
lean_object* v___x_1260_; 
if (v_isShared_1258_ == 0)
{
v___x_1260_ = v___x_1257_;
goto v_reusejp_1259_;
}
else
{
lean_object* v_reuseFailAlloc_1261_; 
v_reuseFailAlloc_1261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1261_, 0, v_a_1255_);
v___x_1260_ = v_reuseFailAlloc_1261_;
goto v_reusejp_1259_;
}
v_reusejp_1259_:
{
return v___x_1260_;
}
}
}
}
else
{
lean_object* v_a_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1270_; 
lean_dec_ref(v___y_1142_);
lean_dec_ref(v_snd_1140_);
lean_dec_ref(v___x_1139_);
lean_dec(v___x_1138_);
lean_dec_ref(v_type_1137_);
lean_dec(v___x_1136_);
v_a_1263_ = lean_ctor_get(v___x_1147_, 0);
v_isSharedCheck_1270_ = !lean_is_exclusive(v___x_1147_);
if (v_isSharedCheck_1270_ == 0)
{
v___x_1265_ = v___x_1147_;
v_isShared_1266_ = v_isSharedCheck_1270_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_a_1263_);
lean_dec(v___x_1147_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1270_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v___x_1268_; 
if (v_isShared_1266_ == 0)
{
v___x_1268_ = v___x_1265_;
goto v_reusejp_1267_;
}
else
{
lean_object* v_reuseFailAlloc_1269_; 
v_reuseFailAlloc_1269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1269_, 0, v_a_1263_);
v___x_1268_ = v_reuseFailAlloc_1269_;
goto v_reusejp_1267_;
}
v_reusejp_1267_:
{
return v___x_1268_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___boxed(lean_object* v___x_1271_, lean_object* v___x_1272_, lean_object* v___x_1273_, lean_object* v_type_1274_, lean_object* v___x_1275_, lean_object* v___x_1276_, lean_object* v_snd_1277_, lean_object* v_fst_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_){
_start:
{
uint8_t v___x_28389__boxed_1284_; uint8_t v_fst_28394__boxed_1285_; lean_object* v_res_1286_; 
v___x_28389__boxed_1284_ = lean_unbox(v___x_1272_);
v_fst_28394__boxed_1285_ = lean_unbox(v_fst_1278_);
v_res_1286_ = lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2(v___x_1271_, v___x_28389__boxed_1284_, v___x_1273_, v_type_1274_, v___x_1275_, v___x_1276_, v_snd_1277_, v_fst_28394__boxed_1285_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_);
lean_dec(v___y_1282_);
lean_dec_ref(v___y_1281_);
lean_dec(v___y_1280_);
return v_res_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3(lean_object* v___x_1293_, uint8_t v___x_1294_, lean_object* v___x_1295_, lean_object* v_type_1296_, lean_object* v___x_1297_, lean_object* v___x_1298_, lean_object* v_snd_1299_, uint8_t v_fst_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_){
_start:
{
lean_object* v___x_1306_; 
lean_inc(v___x_1295_);
v___x_1306_ = l_Lean_Meta_mkFreshExprMVar(v___x_1293_, v___x_1294_, v___x_1295_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
if (lean_obj_tag(v___x_1306_) == 0)
{
lean_object* v_a_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; 
v_a_1307_ = lean_ctor_get(v___x_1306_, 0);
lean_inc(v_a_1307_);
lean_dec_ref_known(v___x_1306_, 1);
lean_inc_ref(v_type_1296_);
v___x_1308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1308_, 0, v_type_1296_);
lean_inc(v___x_1295_);
lean_inc_ref(v___x_1308_);
v___x_1309_ = l_Lean_Meta_mkFreshExprMVar(v___x_1308_, v___x_1294_, v___x_1295_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
if (lean_obj_tag(v___x_1309_) == 0)
{
lean_object* v_a_1310_; lean_object* v___x_1311_; 
v_a_1310_ = lean_ctor_get(v___x_1309_, 0);
lean_inc(v_a_1310_);
lean_dec_ref_known(v___x_1309_, 1);
v___x_1311_ = l_Lean_Meta_mkFreshExprMVar(v___x_1308_, v___x_1294_, v___x_1295_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
if (lean_obj_tag(v___x_1311_) == 0)
{
lean_object* v_a_1312_; lean_object* v_keyedConfig_1313_; uint8_t v_trackZetaDelta_1314_; lean_object* v_zetaDeltaSet_1315_; lean_object* v_lctx_1316_; lean_object* v_localInstances_1317_; lean_object* v_defEqCtx_x3f_1318_; lean_object* v_synthPendingDepth_1319_; lean_object* v_customCanUnfoldPredicate_x3f_1320_; uint8_t v_univApprox_1321_; uint8_t v_inTypeClassResolution_1322_; uint8_t v_cacheInferType_1323_; lean_object* v___x_1325_; uint8_t v_isShared_1326_; uint8_t v_isSharedCheck_1405_; 
v_a_1312_ = lean_ctor_get(v___x_1311_, 0);
lean_inc(v_a_1312_);
lean_dec_ref_known(v___x_1311_, 1);
v_keyedConfig_1313_ = lean_ctor_get(v___y_1301_, 0);
v_trackZetaDelta_1314_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7);
v_zetaDeltaSet_1315_ = lean_ctor_get(v___y_1301_, 1);
v_lctx_1316_ = lean_ctor_get(v___y_1301_, 2);
v_localInstances_1317_ = lean_ctor_get(v___y_1301_, 3);
v_defEqCtx_x3f_1318_ = lean_ctor_get(v___y_1301_, 4);
v_synthPendingDepth_1319_ = lean_ctor_get(v___y_1301_, 5);
v_customCanUnfoldPredicate_x3f_1320_ = lean_ctor_get(v___y_1301_, 6);
v_univApprox_1321_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1322_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7 + 2);
v_cacheInferType_1323_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7 + 3);
v_isSharedCheck_1405_ = !lean_is_exclusive(v___y_1301_);
if (v_isSharedCheck_1405_ == 0)
{
v___x_1325_ = v___y_1301_;
v_isShared_1326_ = v_isSharedCheck_1405_;
goto v_resetjp_1324_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1320_);
lean_inc(v_synthPendingDepth_1319_);
lean_inc(v_defEqCtx_x3f_1318_);
lean_inc(v_localInstances_1317_);
lean_inc(v_lctx_1316_);
lean_inc(v_zetaDeltaSet_1315_);
lean_inc(v_keyedConfig_1313_);
lean_dec(v___y_1301_);
v___x_1325_ = lean_box(0);
v_isShared_1326_ = v_isSharedCheck_1405_;
goto v_resetjp_1324_;
}
v_resetjp_1324_:
{
lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; uint8_t v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1341_; 
v___x_1327_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__2));
lean_inc(v___x_1297_);
v___x_1328_ = l_Lean_Expr_const___override(v___x_1327_, v___x_1297_);
lean_inc_ref(v_type_1296_);
v___x_1329_ = l_Lean_Expr_app___override(v___x_1328_, v_type_1296_);
v___x_1330_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___closed__3));
v___x_1331_ = l_Lean_Name_mkStr2(v___x_1298_, v___x_1330_);
v___x_1332_ = l_Lean_Expr_const___override(v___x_1331_, v___x_1297_);
v___x_1333_ = l_Lean_Expr_app___override(v___x_1332_, v_type_1296_);
lean_inc(v_a_1307_);
v___x_1334_ = l_Lean_Expr_app___override(v___x_1333_, v_a_1307_);
v___x_1335_ = l_Lean_Expr_app___override(v___x_1329_, v___x_1334_);
lean_inc(v_a_1310_);
v___x_1336_ = l_Lean_Expr_app___override(v___x_1335_, v_a_1310_);
lean_inc(v_a_1312_);
v___x_1337_ = l_Lean_Expr_app___override(v___x_1336_, v_a_1312_);
v___x_1338_ = 2;
v___x_1339_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1338_, v_keyedConfig_1313_);
if (v_isShared_1326_ == 0)
{
lean_ctor_set(v___x_1325_, 0, v___x_1339_);
v___x_1341_ = v___x_1325_;
goto v_reusejp_1340_;
}
else
{
lean_object* v_reuseFailAlloc_1404_; 
v_reuseFailAlloc_1404_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1404_, 0, v___x_1339_);
lean_ctor_set(v_reuseFailAlloc_1404_, 1, v_zetaDeltaSet_1315_);
lean_ctor_set(v_reuseFailAlloc_1404_, 2, v_lctx_1316_);
lean_ctor_set(v_reuseFailAlloc_1404_, 3, v_localInstances_1317_);
lean_ctor_set(v_reuseFailAlloc_1404_, 4, v_defEqCtx_x3f_1318_);
lean_ctor_set(v_reuseFailAlloc_1404_, 5, v_synthPendingDepth_1319_);
lean_ctor_set(v_reuseFailAlloc_1404_, 6, v_customCanUnfoldPredicate_x3f_1320_);
lean_ctor_set_uint8(v_reuseFailAlloc_1404_, sizeof(void*)*7, v_trackZetaDelta_1314_);
lean_ctor_set_uint8(v_reuseFailAlloc_1404_, sizeof(void*)*7 + 1, v_univApprox_1321_);
lean_ctor_set_uint8(v_reuseFailAlloc_1404_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1322_);
lean_ctor_set_uint8(v_reuseFailAlloc_1404_, sizeof(void*)*7 + 3, v_cacheInferType_1323_);
v___x_1341_ = v_reuseFailAlloc_1404_;
goto v_reusejp_1340_;
}
v_reusejp_1340_:
{
lean_object* v___x_1342_; 
v___x_1342_ = l_Lean_Meta_isExprDefEq(v___x_1337_, v_snd_1299_, v___x_1341_, v___y_1302_, v___y_1303_, v___y_1304_);
lean_dec_ref(v___x_1341_);
if (lean_obj_tag(v___x_1342_) == 0)
{
lean_object* v_a_1343_; lean_object* v___x_1345_; uint8_t v_isShared_1346_; uint8_t v_isSharedCheck_1395_; 
v_a_1343_ = lean_ctor_get(v___x_1342_, 0);
v_isSharedCheck_1395_ = !lean_is_exclusive(v___x_1342_);
if (v_isSharedCheck_1395_ == 0)
{
v___x_1345_ = v___x_1342_;
v_isShared_1346_ = v_isSharedCheck_1395_;
goto v_resetjp_1344_;
}
else
{
lean_inc(v_a_1343_);
lean_dec(v___x_1342_);
v___x_1345_ = lean_box(0);
v_isShared_1346_ = v_isSharedCheck_1395_;
goto v_resetjp_1344_;
}
v_resetjp_1344_:
{
uint8_t v___x_1347_; 
v___x_1347_ = lean_unbox(v_a_1343_);
if (v___x_1347_ == 0)
{
lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1353_; 
lean_dec(v_a_1343_);
v___x_1348_ = lean_box(v_fst_1300_);
v___x_1349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1349_, 0, v_a_1312_);
lean_ctor_set(v___x_1349_, 1, v___x_1348_);
v___x_1350_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1350_, 0, v_a_1310_);
lean_ctor_set(v___x_1350_, 1, v___x_1349_);
v___x_1351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1351_, 0, v_a_1307_);
lean_ctor_set(v___x_1351_, 1, v___x_1350_);
if (v_isShared_1346_ == 0)
{
lean_ctor_set(v___x_1345_, 0, v___x_1351_);
v___x_1353_ = v___x_1345_;
goto v_reusejp_1352_;
}
else
{
lean_object* v_reuseFailAlloc_1354_; 
v_reuseFailAlloc_1354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1354_, 0, v___x_1351_);
v___x_1353_ = v_reuseFailAlloc_1354_;
goto v_reusejp_1352_;
}
v_reusejp_1352_:
{
return v___x_1353_;
}
}
else
{
lean_object* v___x_1355_; 
lean_del_object(v___x_1345_);
v___x_1355_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1307_, v___y_1302_);
if (lean_obj_tag(v___x_1355_) == 0)
{
lean_object* v_a_1356_; lean_object* v___x_1357_; 
v_a_1356_ = lean_ctor_get(v___x_1355_, 0);
lean_inc(v_a_1356_);
lean_dec_ref_known(v___x_1355_, 1);
v___x_1357_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1310_, v___y_1302_);
if (lean_obj_tag(v___x_1357_) == 0)
{
lean_object* v_a_1358_; lean_object* v___x_1359_; 
v_a_1358_ = lean_ctor_get(v___x_1357_, 0);
lean_inc(v_a_1358_);
lean_dec_ref_known(v___x_1357_, 1);
v___x_1359_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1312_, v___y_1302_);
if (lean_obj_tag(v___x_1359_) == 0)
{
lean_object* v_a_1360_; lean_object* v___x_1362_; uint8_t v_isShared_1363_; uint8_t v_isSharedCheck_1370_; 
v_a_1360_ = lean_ctor_get(v___x_1359_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1359_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1362_ = v___x_1359_;
v_isShared_1363_ = v_isSharedCheck_1370_;
goto v_resetjp_1361_;
}
else
{
lean_inc(v_a_1360_);
lean_dec(v___x_1359_);
v___x_1362_ = lean_box(0);
v_isShared_1363_ = v_isSharedCheck_1370_;
goto v_resetjp_1361_;
}
v_resetjp_1361_:
{
lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1368_; 
v___x_1364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1364_, 0, v_a_1360_);
lean_ctor_set(v___x_1364_, 1, v_a_1343_);
v___x_1365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1365_, 0, v_a_1358_);
lean_ctor_set(v___x_1365_, 1, v___x_1364_);
v___x_1366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1366_, 0, v_a_1356_);
lean_ctor_set(v___x_1366_, 1, v___x_1365_);
if (v_isShared_1363_ == 0)
{
lean_ctor_set(v___x_1362_, 0, v___x_1366_);
v___x_1368_ = v___x_1362_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v___x_1366_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
else
{
lean_object* v_a_1371_; lean_object* v___x_1373_; uint8_t v_isShared_1374_; uint8_t v_isSharedCheck_1378_; 
lean_dec(v_a_1358_);
lean_dec(v_a_1356_);
lean_dec(v_a_1343_);
v_a_1371_ = lean_ctor_get(v___x_1359_, 0);
v_isSharedCheck_1378_ = !lean_is_exclusive(v___x_1359_);
if (v_isSharedCheck_1378_ == 0)
{
v___x_1373_ = v___x_1359_;
v_isShared_1374_ = v_isSharedCheck_1378_;
goto v_resetjp_1372_;
}
else
{
lean_inc(v_a_1371_);
lean_dec(v___x_1359_);
v___x_1373_ = lean_box(0);
v_isShared_1374_ = v_isSharedCheck_1378_;
goto v_resetjp_1372_;
}
v_resetjp_1372_:
{
lean_object* v___x_1376_; 
if (v_isShared_1374_ == 0)
{
v___x_1376_ = v___x_1373_;
goto v_reusejp_1375_;
}
else
{
lean_object* v_reuseFailAlloc_1377_; 
v_reuseFailAlloc_1377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1377_, 0, v_a_1371_);
v___x_1376_ = v_reuseFailAlloc_1377_;
goto v_reusejp_1375_;
}
v_reusejp_1375_:
{
return v___x_1376_;
}
}
}
}
else
{
lean_object* v_a_1379_; lean_object* v___x_1381_; uint8_t v_isShared_1382_; uint8_t v_isSharedCheck_1386_; 
lean_dec(v_a_1356_);
lean_dec(v_a_1343_);
lean_dec(v_a_1312_);
v_a_1379_ = lean_ctor_get(v___x_1357_, 0);
v_isSharedCheck_1386_ = !lean_is_exclusive(v___x_1357_);
if (v_isSharedCheck_1386_ == 0)
{
v___x_1381_ = v___x_1357_;
v_isShared_1382_ = v_isSharedCheck_1386_;
goto v_resetjp_1380_;
}
else
{
lean_inc(v_a_1379_);
lean_dec(v___x_1357_);
v___x_1381_ = lean_box(0);
v_isShared_1382_ = v_isSharedCheck_1386_;
goto v_resetjp_1380_;
}
v_resetjp_1380_:
{
lean_object* v___x_1384_; 
if (v_isShared_1382_ == 0)
{
v___x_1384_ = v___x_1381_;
goto v_reusejp_1383_;
}
else
{
lean_object* v_reuseFailAlloc_1385_; 
v_reuseFailAlloc_1385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1385_, 0, v_a_1379_);
v___x_1384_ = v_reuseFailAlloc_1385_;
goto v_reusejp_1383_;
}
v_reusejp_1383_:
{
return v___x_1384_;
}
}
}
}
else
{
lean_object* v_a_1387_; lean_object* v___x_1389_; uint8_t v_isShared_1390_; uint8_t v_isSharedCheck_1394_; 
lean_dec(v_a_1343_);
lean_dec(v_a_1312_);
lean_dec(v_a_1310_);
v_a_1387_ = lean_ctor_get(v___x_1355_, 0);
v_isSharedCheck_1394_ = !lean_is_exclusive(v___x_1355_);
if (v_isSharedCheck_1394_ == 0)
{
v___x_1389_ = v___x_1355_;
v_isShared_1390_ = v_isSharedCheck_1394_;
goto v_resetjp_1388_;
}
else
{
lean_inc(v_a_1387_);
lean_dec(v___x_1355_);
v___x_1389_ = lean_box(0);
v_isShared_1390_ = v_isSharedCheck_1394_;
goto v_resetjp_1388_;
}
v_resetjp_1388_:
{
lean_object* v___x_1392_; 
if (v_isShared_1390_ == 0)
{
v___x_1392_ = v___x_1389_;
goto v_reusejp_1391_;
}
else
{
lean_object* v_reuseFailAlloc_1393_; 
v_reuseFailAlloc_1393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1393_, 0, v_a_1387_);
v___x_1392_ = v_reuseFailAlloc_1393_;
goto v_reusejp_1391_;
}
v_reusejp_1391_:
{
return v___x_1392_;
}
}
}
}
}
}
else
{
lean_object* v_a_1396_; lean_object* v___x_1398_; uint8_t v_isShared_1399_; uint8_t v_isSharedCheck_1403_; 
lean_dec(v_a_1312_);
lean_dec(v_a_1310_);
lean_dec(v_a_1307_);
v_a_1396_ = lean_ctor_get(v___x_1342_, 0);
v_isSharedCheck_1403_ = !lean_is_exclusive(v___x_1342_);
if (v_isSharedCheck_1403_ == 0)
{
v___x_1398_ = v___x_1342_;
v_isShared_1399_ = v_isSharedCheck_1403_;
goto v_resetjp_1397_;
}
else
{
lean_inc(v_a_1396_);
lean_dec(v___x_1342_);
v___x_1398_ = lean_box(0);
v_isShared_1399_ = v_isSharedCheck_1403_;
goto v_resetjp_1397_;
}
v_resetjp_1397_:
{
lean_object* v___x_1401_; 
if (v_isShared_1399_ == 0)
{
v___x_1401_ = v___x_1398_;
goto v_reusejp_1400_;
}
else
{
lean_object* v_reuseFailAlloc_1402_; 
v_reuseFailAlloc_1402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1402_, 0, v_a_1396_);
v___x_1401_ = v_reuseFailAlloc_1402_;
goto v_reusejp_1400_;
}
v_reusejp_1400_:
{
return v___x_1401_;
}
}
}
}
}
}
else
{
lean_object* v_a_1406_; lean_object* v___x_1408_; uint8_t v_isShared_1409_; uint8_t v_isSharedCheck_1413_; 
lean_dec(v_a_1310_);
lean_dec(v_a_1307_);
lean_dec_ref(v___y_1301_);
lean_dec_ref(v_snd_1299_);
lean_dec_ref(v___x_1298_);
lean_dec(v___x_1297_);
lean_dec_ref(v_type_1296_);
v_a_1406_ = lean_ctor_get(v___x_1311_, 0);
v_isSharedCheck_1413_ = !lean_is_exclusive(v___x_1311_);
if (v_isSharedCheck_1413_ == 0)
{
v___x_1408_ = v___x_1311_;
v_isShared_1409_ = v_isSharedCheck_1413_;
goto v_resetjp_1407_;
}
else
{
lean_inc(v_a_1406_);
lean_dec(v___x_1311_);
v___x_1408_ = lean_box(0);
v_isShared_1409_ = v_isSharedCheck_1413_;
goto v_resetjp_1407_;
}
v_resetjp_1407_:
{
lean_object* v___x_1411_; 
if (v_isShared_1409_ == 0)
{
v___x_1411_ = v___x_1408_;
goto v_reusejp_1410_;
}
else
{
lean_object* v_reuseFailAlloc_1412_; 
v_reuseFailAlloc_1412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1412_, 0, v_a_1406_);
v___x_1411_ = v_reuseFailAlloc_1412_;
goto v_reusejp_1410_;
}
v_reusejp_1410_:
{
return v___x_1411_;
}
}
}
}
else
{
lean_object* v_a_1414_; lean_object* v___x_1416_; uint8_t v_isShared_1417_; uint8_t v_isSharedCheck_1421_; 
lean_dec_ref_known(v___x_1308_, 1);
lean_dec(v_a_1307_);
lean_dec_ref(v___y_1301_);
lean_dec_ref(v_snd_1299_);
lean_dec_ref(v___x_1298_);
lean_dec(v___x_1297_);
lean_dec_ref(v_type_1296_);
lean_dec(v___x_1295_);
v_a_1414_ = lean_ctor_get(v___x_1309_, 0);
v_isSharedCheck_1421_ = !lean_is_exclusive(v___x_1309_);
if (v_isSharedCheck_1421_ == 0)
{
v___x_1416_ = v___x_1309_;
v_isShared_1417_ = v_isSharedCheck_1421_;
goto v_resetjp_1415_;
}
else
{
lean_inc(v_a_1414_);
lean_dec(v___x_1309_);
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
lean_dec_ref(v___y_1301_);
lean_dec_ref(v_snd_1299_);
lean_dec_ref(v___x_1298_);
lean_dec(v___x_1297_);
lean_dec_ref(v_type_1296_);
lean_dec(v___x_1295_);
v_a_1422_ = lean_ctor_get(v___x_1306_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___x_1306_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1424_ = v___x_1306_;
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
else
{
lean_inc(v_a_1422_);
lean_dec(v___x_1306_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___boxed(lean_object* v___x_1430_, lean_object* v___x_1431_, lean_object* v___x_1432_, lean_object* v_type_1433_, lean_object* v___x_1434_, lean_object* v___x_1435_, lean_object* v_snd_1436_, lean_object* v_fst_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_){
_start:
{
uint8_t v___x_28669__boxed_1443_; uint8_t v_fst_28674__boxed_1444_; lean_object* v_res_1445_; 
v___x_28669__boxed_1443_ = lean_unbox(v___x_1431_);
v_fst_28674__boxed_1444_ = lean_unbox(v_fst_1437_);
v_res_1445_ = lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3(v___x_1430_, v___x_28669__boxed_1443_, v___x_1432_, v_type_1433_, v___x_1434_, v___x_1435_, v_snd_1436_, v_fst_28674__boxed_1444_, v___y_1438_, v___y_1439_, v___y_1440_, v___y_1441_);
lean_dec(v___y_1441_);
lean_dec_ref(v___y_1440_);
lean_dec(v___y_1439_);
return v_res_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom(lean_object* v_u_1455_, lean_object* v_type_1456_, lean_object* v_x_1457_, lean_object* v_a_1458_, lean_object* v_a_1459_, lean_object* v_a_1460_, lean_object* v_a_1461_, lean_object* v_a_1462_, lean_object* v_a_1463_, lean_object* v_a_1464_){
_start:
{
lean_object* v___x_1466_; 
v___x_1466_ = lp_mathlib_Mathlib_Tactic_AtomM_containsThenAddQ___redArg(v_x_1457_, v_a_1459_, v_a_1460_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1466_) == 0)
{
lean_object* v_a_1467_; lean_object* v___x_1469_; uint8_t v_isShared_1470_; uint8_t v_isSharedCheck_1632_; 
v_a_1467_ = lean_ctor_get(v___x_1466_, 0);
v_isSharedCheck_1632_ = !lean_is_exclusive(v___x_1466_);
if (v_isSharedCheck_1632_ == 0)
{
v___x_1469_ = v___x_1466_;
v_isShared_1470_ = v_isSharedCheck_1632_;
goto v_resetjp_1468_;
}
else
{
lean_inc(v_a_1467_);
lean_dec(v___x_1466_);
v___x_1469_ = lean_box(0);
v_isShared_1470_ = v_isSharedCheck_1632_;
goto v_resetjp_1468_;
}
v_resetjp_1468_:
{
lean_object* v_fst_1471_; uint8_t v___x_1472_; 
v_fst_1471_ = lean_ctor_get(v_a_1467_, 0);
v___x_1472_ = lean_unbox(v_fst_1471_);
if (v___x_1472_ == 0)
{
lean_object* v_snd_1473_; lean_object* v___x_1475_; uint8_t v_isShared_1476_; uint8_t v_isSharedCheck_1617_; 
lean_inc(v_fst_1471_);
v_snd_1473_ = lean_ctor_get(v_a_1467_, 1);
v_isSharedCheck_1617_ = !lean_is_exclusive(v_a_1467_);
if (v_isSharedCheck_1617_ == 0)
{
lean_object* v_unused_1618_; 
v_unused_1618_ = lean_ctor_get(v_a_1467_, 0);
lean_dec(v_unused_1618_);
v___x_1475_ = v_a_1467_;
v_isShared_1476_ = v_isSharedCheck_1617_;
goto v_resetjp_1474_;
}
else
{
lean_inc(v_snd_1473_);
lean_dec(v_a_1467_);
v___x_1475_ = lean_box(0);
v_isShared_1476_ = v_isSharedCheck_1617_;
goto v_resetjp_1474_;
}
v_resetjp_1474_:
{
lean_object* v_fst_1477_; lean_object* v_snd_1478_; lean_object* v___x_1480_; uint8_t v_isShared_1481_; uint8_t v_isSharedCheck_1616_; 
v_fst_1477_ = lean_ctor_get(v_snd_1473_, 0);
v_snd_1478_ = lean_ctor_get(v_snd_1473_, 1);
v_isSharedCheck_1616_ = !lean_is_exclusive(v_snd_1473_);
if (v_isSharedCheck_1616_ == 0)
{
v___x_1480_ = v_snd_1473_;
v_isShared_1481_ = v_isSharedCheck_1616_;
goto v_resetjp_1479_;
}
else
{
lean_inc(v_snd_1478_);
lean_inc(v_fst_1477_);
lean_dec(v_snd_1473_);
v___x_1480_ = lean_box(0);
v_isShared_1481_ = v_isSharedCheck_1616_;
goto v_resetjp_1479_;
}
v_resetjp_1479_:
{
lean_object* v_snd_1483_; lean_object* v___y_1491_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1506_; 
v___x_1502_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__0));
v___x_1503_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__1));
v___x_1504_ = lean_box(0);
lean_inc(v_u_1455_);
if (v_isShared_1476_ == 0)
{
lean_ctor_set_tag(v___x_1475_, 1);
lean_ctor_set(v___x_1475_, 1, v___x_1504_);
lean_ctor_set(v___x_1475_, 0, v_u_1455_);
v___x_1506_ = v___x_1475_;
goto v_reusejp_1505_;
}
else
{
lean_object* v_reuseFailAlloc_1615_; 
v_reuseFailAlloc_1615_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1615_, 0, v_u_1455_);
lean_ctor_set(v_reuseFailAlloc_1615_, 1, v___x_1504_);
v___x_1506_ = v_reuseFailAlloc_1615_;
goto v_reusejp_1505_;
}
v___jp_1482_:
{
lean_object* v___x_1485_; 
if (v_isShared_1481_ == 0)
{
lean_ctor_set(v___x_1480_, 1, v_snd_1483_);
v___x_1485_ = v___x_1480_;
goto v_reusejp_1484_;
}
else
{
lean_object* v_reuseFailAlloc_1489_; 
v_reuseFailAlloc_1489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1489_, 0, v_fst_1477_);
lean_ctor_set(v_reuseFailAlloc_1489_, 1, v_snd_1483_);
v___x_1485_ = v_reuseFailAlloc_1489_;
goto v_reusejp_1484_;
}
v_reusejp_1484_:
{
lean_object* v___x_1487_; 
if (v_isShared_1470_ == 0)
{
lean_ctor_set(v___x_1469_, 0, v___x_1485_);
v___x_1487_ = v___x_1469_;
goto v_reusejp_1486_;
}
else
{
lean_object* v_reuseFailAlloc_1488_; 
v_reuseFailAlloc_1488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1488_, 0, v___x_1485_);
v___x_1487_ = v_reuseFailAlloc_1488_;
goto v_reusejp_1486_;
}
v_reusejp_1486_:
{
return v___x_1487_;
}
}
}
v___jp_1490_:
{
if (lean_obj_tag(v___y_1491_) == 0)
{
lean_object* v_a_1492_; lean_object* v_snd_1493_; 
v_a_1492_ = lean_ctor_get(v___y_1491_, 0);
lean_inc(v_a_1492_);
lean_dec_ref_known(v___y_1491_, 1);
v_snd_1493_ = lean_ctor_get(v_a_1492_, 1);
lean_inc(v_snd_1493_);
lean_dec(v_a_1492_);
v_snd_1483_ = v_snd_1493_;
goto v___jp_1482_;
}
else
{
lean_object* v_a_1494_; lean_object* v___x_1496_; uint8_t v_isShared_1497_; uint8_t v_isSharedCheck_1501_; 
lean_del_object(v___x_1480_);
lean_dec(v_fst_1477_);
lean_del_object(v___x_1469_);
v_a_1494_ = lean_ctor_get(v___y_1491_, 0);
v_isSharedCheck_1501_ = !lean_is_exclusive(v___y_1491_);
if (v_isSharedCheck_1501_ == 0)
{
v___x_1496_ = v___y_1491_;
v_isShared_1497_ = v_isSharedCheck_1501_;
goto v_resetjp_1495_;
}
else
{
lean_inc(v_a_1494_);
lean_dec(v___y_1491_);
v___x_1496_ = lean_box(0);
v_isShared_1497_ = v_isSharedCheck_1501_;
goto v_resetjp_1495_;
}
v_resetjp_1495_:
{
lean_object* v___x_1499_; 
if (v_isShared_1497_ == 0)
{
v___x_1499_ = v___x_1496_;
goto v_reusejp_1498_;
}
else
{
lean_object* v_reuseFailAlloc_1500_; 
v_reuseFailAlloc_1500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1500_, 0, v_a_1494_);
v___x_1499_ = v_reuseFailAlloc_1500_;
goto v_reusejp_1498_;
}
v_reusejp_1498_:
{
return v___x_1499_;
}
}
}
}
v_reusejp_1505_:
{
lean_object* v___x_1507_; lean_object* v___x_1508_; uint8_t v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; lean_object* v___f_1520_; uint8_t v___x_1521_; lean_object* v___x_1522_; 
lean_inc_ref_n(v___x_1506_, 4);
v___x_1507_ = l_Lean_Expr_const___override(v___x_1503_, v___x_1506_);
lean_inc_ref_n(v_type_1456_, 4);
v___x_1508_ = l_Lean_Expr_app___override(v___x_1507_, v_type_1456_);
v___x_1509_ = 0;
v___x_1510_ = lean_box(0);
v___x_1511_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__2));
v___x_1512_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__3));
v___x_1513_ = l_Lean_Expr_const___override(v___x_1512_, v___x_1506_);
v___x_1514_ = l_Lean_Expr_app___override(v___x_1513_, v_type_1456_);
v___x_1515_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__5));
v___x_1516_ = l_Lean_Expr_const___override(v___x_1515_, v___x_1506_);
v___x_1517_ = l_Lean_Expr_app___override(v___x_1516_, v_type_1456_);
v___x_1518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1518_, 0, v___x_1517_);
v___x_1519_ = lean_box(v___x_1509_);
lean_inc(v_fst_1471_);
lean_inc(v_snd_1478_);
lean_inc_ref(v___x_1518_);
v___f_1520_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__0___boxed), 12, 7);
lean_closure_set(v___f_1520_, 0, v___x_1518_);
lean_closure_set(v___f_1520_, 1, v___x_1519_);
lean_closure_set(v___f_1520_, 2, v___x_1510_);
lean_closure_set(v___f_1520_, 3, v___x_1506_);
lean_closure_set(v___f_1520_, 4, v_type_1456_);
lean_closure_set(v___f_1520_, 5, v_snd_1478_);
lean_closure_set(v___f_1520_, 6, v_fst_1471_);
v___x_1521_ = lean_unbox(v_fst_1471_);
v___x_1522_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_1520_, v___x_1521_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1522_) == 0)
{
lean_object* v_a_1523_; lean_object* v_snd_1524_; lean_object* v_snd_1525_; uint8_t v___x_1526_; 
v_a_1523_ = lean_ctor_get(v___x_1522_, 0);
lean_inc(v_a_1523_);
lean_dec_ref_known(v___x_1522_, 1);
v_snd_1524_ = lean_ctor_get(v_a_1523_, 1);
lean_inc(v_snd_1524_);
lean_dec(v_a_1523_);
v_snd_1525_ = lean_ctor_get(v_snd_1524_, 1);
lean_inc(v_snd_1525_);
lean_dec(v_snd_1524_);
v___x_1526_ = lean_unbox(v_snd_1525_);
lean_dec(v_snd_1525_);
if (v___x_1526_ == 0)
{
lean_object* v___x_1527_; lean_object* v___f_1528_; uint8_t v___x_1529_; lean_object* v___x_1530_; 
v___x_1527_ = lean_box(v___x_1509_);
lean_inc(v_fst_1471_);
lean_inc(v_snd_1478_);
lean_inc_ref(v_type_1456_);
lean_inc_ref(v___x_1506_);
v___f_1528_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__1___boxed), 12, 7);
lean_closure_set(v___f_1528_, 0, v___x_1518_);
lean_closure_set(v___f_1528_, 1, v___x_1527_);
lean_closure_set(v___f_1528_, 2, v___x_1510_);
lean_closure_set(v___f_1528_, 3, v___x_1506_);
lean_closure_set(v___f_1528_, 4, v_type_1456_);
lean_closure_set(v___f_1528_, 5, v_snd_1478_);
lean_closure_set(v___f_1528_, 6, v_fst_1471_);
v___x_1529_ = lean_unbox(v_fst_1471_);
v___x_1530_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_1528_, v___x_1529_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1530_) == 0)
{
lean_object* v_a_1531_; lean_object* v_snd_1532_; lean_object* v_snd_1533_; uint8_t v___x_1534_; 
v_a_1531_ = lean_ctor_get(v___x_1530_, 0);
lean_inc(v_a_1531_);
lean_dec_ref_known(v___x_1530_, 1);
v_snd_1532_ = lean_ctor_get(v_a_1531_, 1);
lean_inc(v_snd_1532_);
lean_dec(v_a_1531_);
v_snd_1533_ = lean_ctor_get(v_snd_1532_, 1);
lean_inc(v_snd_1533_);
lean_dec(v_snd_1532_);
v___x_1534_ = lean_unbox(v_snd_1533_);
lean_dec(v_snd_1533_);
if (v___x_1534_ == 0)
{
lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___f_1537_; uint8_t v___x_1538_; lean_object* v___x_1539_; 
v___x_1535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1535_, 0, v___x_1514_);
v___x_1536_ = lean_box(v___x_1509_);
lean_inc(v_fst_1471_);
lean_inc(v_snd_1478_);
lean_inc_ref(v___x_1506_);
lean_inc_ref(v_type_1456_);
v___f_1537_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__2___boxed), 13, 8);
lean_closure_set(v___f_1537_, 0, v___x_1535_);
lean_closure_set(v___f_1537_, 1, v___x_1536_);
lean_closure_set(v___f_1537_, 2, v___x_1510_);
lean_closure_set(v___f_1537_, 3, v_type_1456_);
lean_closure_set(v___f_1537_, 4, v___x_1506_);
lean_closure_set(v___f_1537_, 5, v___x_1511_);
lean_closure_set(v___f_1537_, 6, v_snd_1478_);
lean_closure_set(v___f_1537_, 7, v_fst_1471_);
v___x_1538_ = lean_unbox(v_fst_1471_);
v___x_1539_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_1537_, v___x_1538_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1539_) == 0)
{
lean_object* v_a_1540_; lean_object* v_snd_1541_; lean_object* v_snd_1542_; lean_object* v_snd_1543_; uint8_t v___x_1544_; 
v_a_1540_ = lean_ctor_get(v___x_1539_, 0);
lean_inc(v_a_1540_);
lean_dec_ref_known(v___x_1539_, 1);
v_snd_1541_ = lean_ctor_get(v_a_1540_, 1);
lean_inc(v_snd_1541_);
lean_dec(v_a_1540_);
v_snd_1542_ = lean_ctor_get(v_snd_1541_, 1);
lean_inc(v_snd_1542_);
v_snd_1543_ = lean_ctor_get(v_snd_1542_, 1);
v___x_1544_ = lean_unbox(v_snd_1543_);
if (v___x_1544_ == 0)
{
lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___f_1547_; uint8_t v___x_1548_; lean_object* v___x_1549_; 
lean_dec(v_snd_1542_);
lean_dec(v_snd_1541_);
v___x_1545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1545_, 0, v___x_1508_);
v___x_1546_ = lean_box(v___x_1509_);
lean_inc(v_fst_1471_);
lean_inc_ref(v_type_1456_);
v___f_1547_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___lam__3___boxed), 13, 8);
lean_closure_set(v___f_1547_, 0, v___x_1545_);
lean_closure_set(v___f_1547_, 1, v___x_1546_);
lean_closure_set(v___f_1547_, 2, v___x_1510_);
lean_closure_set(v___f_1547_, 3, v_type_1456_);
lean_closure_set(v___f_1547_, 4, v___x_1506_);
lean_closure_set(v___f_1547_, 5, v___x_1502_);
lean_closure_set(v___f_1547_, 6, v_snd_1478_);
lean_closure_set(v___f_1547_, 7, v_fst_1471_);
v___x_1548_ = lean_unbox(v_fst_1471_);
lean_dec(v_fst_1471_);
v___x_1549_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_1547_, v___x_1548_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1549_) == 0)
{
lean_object* v_a_1550_; lean_object* v_snd_1551_; lean_object* v_snd_1552_; lean_object* v_snd_1553_; uint8_t v___x_1554_; 
v_a_1550_ = lean_ctor_get(v___x_1549_, 0);
lean_inc(v_a_1550_);
lean_dec_ref_known(v___x_1549_, 1);
v_snd_1551_ = lean_ctor_get(v_a_1550_, 1);
lean_inc(v_snd_1551_);
lean_dec(v_a_1550_);
v_snd_1552_ = lean_ctor_get(v_snd_1551_, 1);
lean_inc(v_snd_1552_);
v_snd_1553_ = lean_ctor_get(v_snd_1552_, 1);
v___x_1554_ = lean_unbox(v_snd_1553_);
if (v___x_1554_ == 0)
{
lean_dec(v_snd_1552_);
lean_dec(v_snd_1551_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
v_snd_1483_ = v_a_1458_;
goto v___jp_1482_;
}
else
{
lean_object* v_fst_1555_; lean_object* v_fst_1556_; lean_object* v___x_1557_; 
v_fst_1555_ = lean_ctor_get(v_snd_1551_, 0);
lean_inc(v_fst_1555_);
lean_dec(v_snd_1551_);
v_fst_1556_ = lean_ctor_get(v_snd_1552_, 0);
lean_inc(v_fst_1556_);
lean_dec(v_snd_1552_);
lean_inc_ref(v_type_1456_);
lean_inc(v_u_1455_);
v___x_1557_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_u_1455_, v_type_1456_, v_fst_1555_, v_a_1458_, v_a_1459_, v_a_1460_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1557_) == 0)
{
lean_object* v_a_1558_; lean_object* v_fst_1559_; lean_object* v_snd_1560_; lean_object* v___x_1561_; 
v_a_1558_ = lean_ctor_get(v___x_1557_, 0);
lean_inc(v_a_1558_);
lean_dec_ref_known(v___x_1557_, 1);
v_fst_1559_ = lean_ctor_get(v_a_1558_, 0);
lean_inc(v_fst_1559_);
v_snd_1560_ = lean_ctor_get(v_a_1558_, 1);
lean_inc(v_snd_1560_);
lean_dec(v_a_1558_);
lean_inc_ref(v_type_1456_);
v___x_1561_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_u_1455_, v_type_1456_, v_fst_1556_, v_snd_1560_, v_a_1459_, v_a_1460_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1561_) == 0)
{
lean_object* v_a_1562_; lean_object* v_fst_1563_; lean_object* v_snd_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; 
v_a_1562_ = lean_ctor_get(v___x_1561_, 0);
lean_inc(v_a_1562_);
lean_dec_ref_known(v___x_1561_, 1);
v_fst_1563_ = lean_ctor_get(v_a_1562_, 0);
lean_inc(v_fst_1563_);
v_snd_1564_ = lean_ctor_get(v_a_1562_, 1);
lean_inc(v_snd_1564_);
lean_dec(v_a_1562_);
lean_inc(v_fst_1477_);
v___x_1565_ = lean_alloc_ctor(8, 3, 0);
lean_ctor_set(v___x_1565_, 0, v_fst_1559_);
lean_ctor_set(v___x_1565_, 1, v_fst_1563_);
lean_ctor_set(v___x_1565_, 2, v_fst_1477_);
v___x_1566_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_type_1456_, v___x_1565_, v_snd_1564_);
v___y_1491_ = v___x_1566_;
goto v___jp_1490_;
}
else
{
lean_dec(v_fst_1559_);
lean_del_object(v___x_1480_);
lean_dec(v_fst_1477_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_type_1456_);
return v___x_1561_;
}
}
else
{
lean_dec(v_fst_1556_);
lean_del_object(v___x_1480_);
lean_dec(v_fst_1477_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
return v___x_1557_;
}
}
}
else
{
lean_object* v_a_1567_; lean_object* v___x_1569_; uint8_t v_isShared_1570_; uint8_t v_isSharedCheck_1574_; 
lean_del_object(v___x_1480_);
lean_dec(v_fst_1477_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_a_1458_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
v_a_1567_ = lean_ctor_get(v___x_1549_, 0);
v_isSharedCheck_1574_ = !lean_is_exclusive(v___x_1549_);
if (v_isSharedCheck_1574_ == 0)
{
v___x_1569_ = v___x_1549_;
v_isShared_1570_ = v_isSharedCheck_1574_;
goto v_resetjp_1568_;
}
else
{
lean_inc(v_a_1567_);
lean_dec(v___x_1549_);
v___x_1569_ = lean_box(0);
v_isShared_1570_ = v_isSharedCheck_1574_;
goto v_resetjp_1568_;
}
v_resetjp_1568_:
{
lean_object* v___x_1572_; 
if (v_isShared_1570_ == 0)
{
v___x_1572_ = v___x_1569_;
goto v_reusejp_1571_;
}
else
{
lean_object* v_reuseFailAlloc_1573_; 
v_reuseFailAlloc_1573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1573_, 0, v_a_1567_);
v___x_1572_ = v_reuseFailAlloc_1573_;
goto v_reusejp_1571_;
}
v_reusejp_1571_:
{
return v___x_1572_;
}
}
}
}
else
{
lean_object* v_fst_1575_; lean_object* v_fst_1576_; lean_object* v___x_1577_; 
lean_dec_ref(v___x_1508_);
lean_dec_ref(v___x_1506_);
lean_dec(v_snd_1478_);
lean_dec(v_fst_1471_);
v_fst_1575_ = lean_ctor_get(v_snd_1541_, 0);
lean_inc(v_fst_1575_);
lean_dec(v_snd_1541_);
v_fst_1576_ = lean_ctor_get(v_snd_1542_, 0);
lean_inc(v_fst_1576_);
lean_dec(v_snd_1542_);
lean_inc_ref(v_type_1456_);
lean_inc(v_u_1455_);
v___x_1577_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_u_1455_, v_type_1456_, v_fst_1575_, v_a_1458_, v_a_1459_, v_a_1460_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1577_) == 0)
{
lean_object* v_a_1578_; lean_object* v_fst_1579_; lean_object* v_snd_1580_; lean_object* v___x_1581_; 
v_a_1578_ = lean_ctor_get(v___x_1577_, 0);
lean_inc(v_a_1578_);
lean_dec_ref_known(v___x_1577_, 1);
v_fst_1579_ = lean_ctor_get(v_a_1578_, 0);
lean_inc(v_fst_1579_);
v_snd_1580_ = lean_ctor_get(v_a_1578_, 1);
lean_inc(v_snd_1580_);
lean_dec(v_a_1578_);
lean_inc_ref(v_type_1456_);
v___x_1581_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_u_1455_, v_type_1456_, v_fst_1576_, v_snd_1580_, v_a_1459_, v_a_1460_, v_a_1461_, v_a_1462_, v_a_1463_, v_a_1464_);
if (lean_obj_tag(v___x_1581_) == 0)
{
lean_object* v_a_1582_; lean_object* v_fst_1583_; lean_object* v_snd_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; 
v_a_1582_ = lean_ctor_get(v___x_1581_, 0);
lean_inc(v_a_1582_);
lean_dec_ref_known(v___x_1581_, 1);
v_fst_1583_ = lean_ctor_get(v_a_1582_, 0);
lean_inc(v_fst_1583_);
v_snd_1584_ = lean_ctor_get(v_a_1582_, 1);
lean_inc(v_snd_1584_);
lean_dec(v_a_1582_);
lean_inc(v_fst_1477_);
v___x_1585_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1585_, 0, v_fst_1579_);
lean_ctor_set(v___x_1585_, 1, v_fst_1583_);
lean_ctor_set(v___x_1585_, 2, v_fst_1477_);
v___x_1586_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_type_1456_, v___x_1585_, v_snd_1584_);
v___y_1491_ = v___x_1586_;
goto v___jp_1490_;
}
else
{
lean_dec(v_fst_1579_);
lean_del_object(v___x_1480_);
lean_dec(v_fst_1477_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_type_1456_);
return v___x_1581_;
}
}
else
{
lean_dec(v_fst_1576_);
lean_del_object(v___x_1480_);
lean_dec(v_fst_1477_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
return v___x_1577_;
}
}
}
else
{
lean_object* v_a_1587_; lean_object* v___x_1589_; uint8_t v_isShared_1590_; uint8_t v_isSharedCheck_1594_; 
lean_dec_ref(v___x_1508_);
lean_dec_ref(v___x_1506_);
lean_del_object(v___x_1480_);
lean_dec(v_snd_1478_);
lean_dec(v_fst_1477_);
lean_dec(v_fst_1471_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_a_1458_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
v_a_1587_ = lean_ctor_get(v___x_1539_, 0);
v_isSharedCheck_1594_ = !lean_is_exclusive(v___x_1539_);
if (v_isSharedCheck_1594_ == 0)
{
v___x_1589_ = v___x_1539_;
v_isShared_1590_ = v_isSharedCheck_1594_;
goto v_resetjp_1588_;
}
else
{
lean_inc(v_a_1587_);
lean_dec(v___x_1539_);
v___x_1589_ = lean_box(0);
v_isShared_1590_ = v_isSharedCheck_1594_;
goto v_resetjp_1588_;
}
v_resetjp_1588_:
{
lean_object* v___x_1592_; 
if (v_isShared_1590_ == 0)
{
v___x_1592_ = v___x_1589_;
goto v_reusejp_1591_;
}
else
{
lean_object* v_reuseFailAlloc_1593_; 
v_reuseFailAlloc_1593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1593_, 0, v_a_1587_);
v___x_1592_ = v_reuseFailAlloc_1593_;
goto v_reusejp_1591_;
}
v_reusejp_1591_:
{
return v___x_1592_;
}
}
}
}
else
{
lean_object* v___x_1595_; lean_object* v___x_1596_; 
lean_dec_ref(v___x_1514_);
lean_dec_ref(v___x_1508_);
lean_dec_ref(v___x_1506_);
lean_dec(v_snd_1478_);
lean_dec(v_fst_1471_);
lean_dec(v_u_1455_);
lean_inc(v_fst_1477_);
v___x_1595_ = lean_alloc_ctor(7, 1, 0);
lean_ctor_set(v___x_1595_, 0, v_fst_1477_);
v___x_1596_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_type_1456_, v___x_1595_, v_a_1458_);
v___y_1491_ = v___x_1596_;
goto v___jp_1490_;
}
}
else
{
lean_object* v_a_1597_; lean_object* v___x_1599_; uint8_t v_isShared_1600_; uint8_t v_isSharedCheck_1604_; 
lean_dec_ref(v___x_1514_);
lean_dec_ref(v___x_1508_);
lean_dec_ref(v___x_1506_);
lean_del_object(v___x_1480_);
lean_dec(v_snd_1478_);
lean_dec(v_fst_1477_);
lean_dec(v_fst_1471_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_a_1458_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
v_a_1597_ = lean_ctor_get(v___x_1530_, 0);
v_isSharedCheck_1604_ = !lean_is_exclusive(v___x_1530_);
if (v_isSharedCheck_1604_ == 0)
{
v___x_1599_ = v___x_1530_;
v_isShared_1600_ = v_isSharedCheck_1604_;
goto v_resetjp_1598_;
}
else
{
lean_inc(v_a_1597_);
lean_dec(v___x_1530_);
v___x_1599_ = lean_box(0);
v_isShared_1600_ = v_isSharedCheck_1604_;
goto v_resetjp_1598_;
}
v_resetjp_1598_:
{
lean_object* v___x_1602_; 
if (v_isShared_1600_ == 0)
{
v___x_1602_ = v___x_1599_;
goto v_reusejp_1601_;
}
else
{
lean_object* v_reuseFailAlloc_1603_; 
v_reuseFailAlloc_1603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1603_, 0, v_a_1597_);
v___x_1602_ = v_reuseFailAlloc_1603_;
goto v_reusejp_1601_;
}
v_reusejp_1601_:
{
return v___x_1602_;
}
}
}
}
else
{
lean_object* v___x_1605_; lean_object* v___x_1606_; 
lean_dec_ref_known(v___x_1518_, 1);
lean_dec_ref(v___x_1514_);
lean_dec_ref(v___x_1508_);
lean_dec_ref(v___x_1506_);
lean_dec(v_snd_1478_);
lean_dec(v_fst_1471_);
lean_dec(v_u_1455_);
lean_inc(v_fst_1477_);
v___x_1605_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v___x_1605_, 0, v_fst_1477_);
v___x_1606_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_type_1456_, v___x_1605_, v_a_1458_);
v___y_1491_ = v___x_1606_;
goto v___jp_1490_;
}
}
else
{
lean_object* v_a_1607_; lean_object* v___x_1609_; uint8_t v_isShared_1610_; uint8_t v_isSharedCheck_1614_; 
lean_dec_ref_known(v___x_1518_, 1);
lean_dec_ref(v___x_1514_);
lean_dec_ref(v___x_1508_);
lean_dec_ref(v___x_1506_);
lean_del_object(v___x_1480_);
lean_dec(v_snd_1478_);
lean_dec(v_fst_1477_);
lean_dec(v_fst_1471_);
lean_del_object(v___x_1469_);
lean_dec_ref(v_a_1458_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
v_a_1607_ = lean_ctor_get(v___x_1522_, 0);
v_isSharedCheck_1614_ = !lean_is_exclusive(v___x_1522_);
if (v_isSharedCheck_1614_ == 0)
{
v___x_1609_ = v___x_1522_;
v_isShared_1610_ = v_isSharedCheck_1614_;
goto v_resetjp_1608_;
}
else
{
lean_inc(v_a_1607_);
lean_dec(v___x_1522_);
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
}
}
}
else
{
lean_object* v_snd_1619_; lean_object* v_fst_1620_; lean_object* v___x_1622_; uint8_t v_isShared_1623_; uint8_t v_isSharedCheck_1630_; 
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
v_snd_1619_ = lean_ctor_get(v_a_1467_, 1);
lean_inc(v_snd_1619_);
lean_dec(v_a_1467_);
v_fst_1620_ = lean_ctor_get(v_snd_1619_, 0);
v_isSharedCheck_1630_ = !lean_is_exclusive(v_snd_1619_);
if (v_isSharedCheck_1630_ == 0)
{
lean_object* v_unused_1631_; 
v_unused_1631_ = lean_ctor_get(v_snd_1619_, 1);
lean_dec(v_unused_1631_);
v___x_1622_ = v_snd_1619_;
v_isShared_1623_ = v_isSharedCheck_1630_;
goto v_resetjp_1621_;
}
else
{
lean_inc(v_fst_1620_);
lean_dec(v_snd_1619_);
v___x_1622_ = lean_box(0);
v_isShared_1623_ = v_isSharedCheck_1630_;
goto v_resetjp_1621_;
}
v_resetjp_1621_:
{
lean_object* v___x_1625_; 
if (v_isShared_1623_ == 0)
{
lean_ctor_set(v___x_1622_, 1, v_a_1458_);
v___x_1625_ = v___x_1622_;
goto v_reusejp_1624_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v_fst_1620_);
lean_ctor_set(v_reuseFailAlloc_1629_, 1, v_a_1458_);
v___x_1625_ = v_reuseFailAlloc_1629_;
goto v_reusejp_1624_;
}
v_reusejp_1624_:
{
lean_object* v___x_1627_; 
if (v_isShared_1470_ == 0)
{
lean_ctor_set(v___x_1469_, 0, v___x_1625_);
v___x_1627_ = v___x_1469_;
goto v_reusejp_1626_;
}
else
{
lean_object* v_reuseFailAlloc_1628_; 
v_reuseFailAlloc_1628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1628_, 0, v___x_1625_);
v___x_1627_ = v_reuseFailAlloc_1628_;
goto v_reusejp_1626_;
}
v_reusejp_1626_:
{
return v___x_1627_;
}
}
}
}
}
}
else
{
lean_object* v_a_1633_; lean_object* v___x_1635_; uint8_t v_isShared_1636_; uint8_t v_isSharedCheck_1640_; 
lean_dec_ref(v_a_1458_);
lean_dec_ref(v_type_1456_);
lean_dec(v_u_1455_);
v_a_1633_ = lean_ctor_get(v___x_1466_, 0);
v_isSharedCheck_1640_ = !lean_is_exclusive(v___x_1466_);
if (v_isSharedCheck_1640_ == 0)
{
v___x_1635_ = v___x_1466_;
v_isShared_1636_ = v_isSharedCheck_1640_;
goto v_resetjp_1634_;
}
else
{
lean_inc(v_a_1633_);
lean_dec(v___x_1466_);
v___x_1635_ = lean_box(0);
v_isShared_1636_ = v_isSharedCheck_1640_;
goto v_resetjp_1634_;
}
v_resetjp_1634_:
{
lean_object* v___x_1638_; 
if (v_isShared_1636_ == 0)
{
v___x_1638_ = v___x_1635_;
goto v_reusejp_1637_;
}
else
{
lean_object* v_reuseFailAlloc_1639_; 
v_reuseFailAlloc_1639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1639_, 0, v_a_1633_);
v___x_1638_ = v_reuseFailAlloc_1639_;
goto v_reusejp_1637_;
}
v_reusejp_1637_:
{
return v___x_1638_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_addAtom___boxed(lean_object* v_u_1641_, lean_object* v_type_1642_, lean_object* v_x_1643_, lean_object* v_a_1644_, lean_object* v_a_1645_, lean_object* v_a_1646_, lean_object* v_a_1647_, lean_object* v_a_1648_, lean_object* v_a_1649_, lean_object* v_a_1650_, lean_object* v_a_1651_){
_start:
{
lean_object* v_res_1652_; 
v_res_1652_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_u_1641_, v_type_1642_, v_x_1643_, v_a_1644_, v_a_1645_, v_a_1646_, v_a_1647_, v_a_1648_, v_a_1649_, v_a_1650_);
lean_dec(v_a_1650_);
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1648_);
lean_dec_ref(v_a_1647_);
lean_dec(v_a_1646_);
lean_dec_ref(v_a_1645_);
return v_res_1652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(lean_object* v_l_1653_, lean_object* v___y_1654_){
_start:
{
lean_object* v___x_1656_; lean_object* v_mctx_1657_; lean_object* v___x_1658_; lean_object* v_fst_1659_; lean_object* v_snd_1660_; lean_object* v___x_1661_; lean_object* v_cache_1662_; lean_object* v_zetaDeltaFVarIds_1663_; lean_object* v_postponed_1664_; lean_object* v_diag_1665_; lean_object* v___x_1667_; uint8_t v_isShared_1668_; uint8_t v_isSharedCheck_1674_; 
v___x_1656_ = lean_st_ref_get(v___y_1654_);
v_mctx_1657_ = lean_ctor_get(v___x_1656_, 0);
lean_inc_ref(v_mctx_1657_);
lean_dec(v___x_1656_);
v___x_1658_ = lean_instantiate_level_mvars(v_mctx_1657_, v_l_1653_);
v_fst_1659_ = lean_ctor_get(v___x_1658_, 0);
lean_inc(v_fst_1659_);
v_snd_1660_ = lean_ctor_get(v___x_1658_, 1);
lean_inc(v_snd_1660_);
lean_dec_ref(v___x_1658_);
v___x_1661_ = lean_st_ref_take(v___y_1654_);
v_cache_1662_ = lean_ctor_get(v___x_1661_, 1);
v_zetaDeltaFVarIds_1663_ = lean_ctor_get(v___x_1661_, 2);
v_postponed_1664_ = lean_ctor_get(v___x_1661_, 3);
v_diag_1665_ = lean_ctor_get(v___x_1661_, 4);
v_isSharedCheck_1674_ = !lean_is_exclusive(v___x_1661_);
if (v_isSharedCheck_1674_ == 0)
{
lean_object* v_unused_1675_; 
v_unused_1675_ = lean_ctor_get(v___x_1661_, 0);
lean_dec(v_unused_1675_);
v___x_1667_ = v___x_1661_;
v_isShared_1668_ = v_isSharedCheck_1674_;
goto v_resetjp_1666_;
}
else
{
lean_inc(v_diag_1665_);
lean_inc(v_postponed_1664_);
lean_inc(v_zetaDeltaFVarIds_1663_);
lean_inc(v_cache_1662_);
lean_dec(v___x_1661_);
v___x_1667_ = lean_box(0);
v_isShared_1668_ = v_isSharedCheck_1674_;
goto v_resetjp_1666_;
}
v_resetjp_1666_:
{
lean_object* v___x_1670_; 
if (v_isShared_1668_ == 0)
{
lean_ctor_set(v___x_1667_, 0, v_fst_1659_);
v___x_1670_ = v___x_1667_;
goto v_reusejp_1669_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v_fst_1659_);
lean_ctor_set(v_reuseFailAlloc_1673_, 1, v_cache_1662_);
lean_ctor_set(v_reuseFailAlloc_1673_, 2, v_zetaDeltaFVarIds_1663_);
lean_ctor_set(v_reuseFailAlloc_1673_, 3, v_postponed_1664_);
lean_ctor_set(v_reuseFailAlloc_1673_, 4, v_diag_1665_);
v___x_1670_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1669_;
}
v_reusejp_1669_:
{
lean_object* v___x_1671_; lean_object* v___x_1672_; 
v___x_1671_ = lean_st_ref_set(v___y_1654_, v___x_1670_);
v___x_1672_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1672_, 0, v_snd_1660_);
return v___x_1672_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg___boxed(lean_object* v_l_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_){
_start:
{
lean_object* v_res_1679_; 
v_res_1679_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_l_1676_, v___y_1677_);
lean_dec(v___y_1677_);
return v_res_1679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0(lean_object* v_l_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_){
_start:
{
lean_object* v___x_1686_; 
v___x_1686_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_l_1680_, v___y_1682_);
return v___x_1686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___boxed(lean_object* v_l_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_){
_start:
{
lean_object* v_res_1693_; 
v_res_1693_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0(v_l_1687_, v___y_1688_, v___y_1689_, v___y_1690_, v___y_1691_);
lean_dec(v___y_1691_);
lean_dec_ref(v___y_1690_);
lean_dec(v___y_1689_);
lean_dec_ref(v___y_1688_);
return v_res_1693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0(uint8_t v___x_1697_, lean_object* v___x_1698_, lean_object* v_fst_1699_, uint8_t v___x_1700_, uint8_t v_a_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_){
_start:
{
lean_object* v___x_1707_; 
v___x_1707_ = l_Lean_Meta_mkFreshLevelMVar(v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_);
if (lean_obj_tag(v___x_1707_) == 0)
{
lean_object* v_a_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; 
v_a_1708_ = lean_ctor_get(v___x_1707_, 0);
lean_inc_n(v_a_1708_, 2);
lean_dec_ref_known(v___x_1707_, 1);
v___x_1709_ = l_Lean_Level_succ___override(v_a_1708_);
lean_inc(v___x_1709_);
v___x_1710_ = l_Lean_Expr_sort___override(v___x_1709_);
v___x_1711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1711_, 0, v___x_1710_);
lean_inc(v___x_1698_);
v___x_1712_ = l_Lean_Meta_mkFreshExprMVar(v___x_1711_, v___x_1697_, v___x_1698_, v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_);
if (lean_obj_tag(v___x_1712_) == 0)
{
lean_object* v_a_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; 
v_a_1713_ = lean_ctor_get(v___x_1712_, 0);
lean_inc_n(v_a_1713_, 2);
lean_dec_ref_known(v___x_1712_, 1);
v___x_1714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1714_, 0, v_a_1713_);
lean_inc(v___x_1698_);
lean_inc_ref(v___x_1714_);
v___x_1715_ = l_Lean_Meta_mkFreshExprMVar(v___x_1714_, v___x_1697_, v___x_1698_, v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_);
if (lean_obj_tag(v___x_1715_) == 0)
{
lean_object* v_a_1716_; lean_object* v___x_1717_; 
v_a_1716_ = lean_ctor_get(v___x_1715_, 0);
lean_inc(v_a_1716_);
lean_dec_ref_known(v___x_1715_, 1);
v___x_1717_ = l_Lean_Meta_mkFreshExprMVar(v___x_1714_, v___x_1697_, v___x_1698_, v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_);
if (lean_obj_tag(v___x_1717_) == 0)
{
lean_object* v_a_1718_; lean_object* v_keyedConfig_1719_; uint8_t v_trackZetaDelta_1720_; lean_object* v_zetaDeltaSet_1721_; lean_object* v_lctx_1722_; lean_object* v_localInstances_1723_; lean_object* v_defEqCtx_x3f_1724_; lean_object* v_synthPendingDepth_1725_; lean_object* v_customCanUnfoldPredicate_x3f_1726_; uint8_t v_univApprox_1727_; uint8_t v_inTypeClassResolution_1728_; uint8_t v_cacheInferType_1729_; lean_object* v___x_1731_; uint8_t v_isShared_1732_; uint8_t v_isSharedCheck_1820_; 
v_a_1718_ = lean_ctor_get(v___x_1717_, 0);
lean_inc(v_a_1718_);
lean_dec_ref_known(v___x_1717_, 1);
v_keyedConfig_1719_ = lean_ctor_get(v___y_1702_, 0);
v_trackZetaDelta_1720_ = lean_ctor_get_uint8(v___y_1702_, sizeof(void*)*7);
v_zetaDeltaSet_1721_ = lean_ctor_get(v___y_1702_, 1);
v_lctx_1722_ = lean_ctor_get(v___y_1702_, 2);
v_localInstances_1723_ = lean_ctor_get(v___y_1702_, 3);
v_defEqCtx_x3f_1724_ = lean_ctor_get(v___y_1702_, 4);
v_synthPendingDepth_1725_ = lean_ctor_get(v___y_1702_, 5);
v_customCanUnfoldPredicate_x3f_1726_ = lean_ctor_get(v___y_1702_, 6);
v_univApprox_1727_ = lean_ctor_get_uint8(v___y_1702_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1728_ = lean_ctor_get_uint8(v___y_1702_, sizeof(void*)*7 + 2);
v_cacheInferType_1729_ = lean_ctor_get_uint8(v___y_1702_, sizeof(void*)*7 + 3);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___y_1702_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1731_ = v___y_1702_;
v_isShared_1732_ = v_isSharedCheck_1820_;
goto v_resetjp_1730_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1726_);
lean_inc(v_synthPendingDepth_1725_);
lean_inc(v_defEqCtx_x3f_1724_);
lean_inc(v_localInstances_1723_);
lean_inc(v_lctx_1722_);
lean_inc(v_zetaDeltaSet_1721_);
lean_inc(v_keyedConfig_1719_);
lean_dec(v___y_1702_);
v___x_1731_ = lean_box(0);
v_isShared_1732_ = v_isSharedCheck_1820_;
goto v_resetjp_1730_;
}
v_resetjp_1730_:
{
lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; uint8_t v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1743_; 
v___x_1733_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___closed__1));
v___x_1734_ = lean_box(0);
v___x_1735_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1735_, 0, v___x_1709_);
lean_ctor_set(v___x_1735_, 1, v___x_1734_);
v___x_1736_ = l_Lean_Expr_const___override(v___x_1733_, v___x_1735_);
lean_inc(v_a_1713_);
v___x_1737_ = l_Lean_Expr_app___override(v___x_1736_, v_a_1713_);
lean_inc(v_a_1716_);
v___x_1738_ = l_Lean_Expr_app___override(v___x_1737_, v_a_1716_);
lean_inc(v_a_1718_);
v___x_1739_ = l_Lean_Expr_app___override(v___x_1738_, v_a_1718_);
v___x_1740_ = 2;
v___x_1741_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1740_, v_keyedConfig_1719_);
if (v_isShared_1732_ == 0)
{
lean_ctor_set(v___x_1731_, 0, v___x_1741_);
v___x_1743_ = v___x_1731_;
goto v_reusejp_1742_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v___x_1741_);
lean_ctor_set(v_reuseFailAlloc_1819_, 1, v_zetaDeltaSet_1721_);
lean_ctor_set(v_reuseFailAlloc_1819_, 2, v_lctx_1722_);
lean_ctor_set(v_reuseFailAlloc_1819_, 3, v_localInstances_1723_);
lean_ctor_set(v_reuseFailAlloc_1819_, 4, v_defEqCtx_x3f_1724_);
lean_ctor_set(v_reuseFailAlloc_1819_, 5, v_synthPendingDepth_1725_);
lean_ctor_set(v_reuseFailAlloc_1819_, 6, v_customCanUnfoldPredicate_x3f_1726_);
lean_ctor_set_uint8(v_reuseFailAlloc_1819_, sizeof(void*)*7, v_trackZetaDelta_1720_);
lean_ctor_set_uint8(v_reuseFailAlloc_1819_, sizeof(void*)*7 + 1, v_univApprox_1727_);
lean_ctor_set_uint8(v_reuseFailAlloc_1819_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1728_);
lean_ctor_set_uint8(v_reuseFailAlloc_1819_, sizeof(void*)*7 + 3, v_cacheInferType_1729_);
v___x_1743_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1742_;
}
v_reusejp_1742_:
{
lean_object* v___x_1744_; 
v___x_1744_ = l_Lean_Meta_isExprDefEq(v___x_1739_, v_fst_1699_, v___x_1743_, v___y_1703_, v___y_1704_, v___y_1705_);
lean_dec_ref(v___x_1743_);
if (lean_obj_tag(v___x_1744_) == 0)
{
lean_object* v_a_1745_; lean_object* v___x_1747_; uint8_t v_isShared_1748_; uint8_t v_isSharedCheck_1810_; 
v_a_1745_ = lean_ctor_get(v___x_1744_, 0);
v_isSharedCheck_1810_ = !lean_is_exclusive(v___x_1744_);
if (v_isSharedCheck_1810_ == 0)
{
v___x_1747_ = v___x_1744_;
v_isShared_1748_ = v_isSharedCheck_1810_;
goto v_resetjp_1746_;
}
else
{
lean_inc(v_a_1745_);
lean_dec(v___x_1744_);
v___x_1747_ = lean_box(0);
v_isShared_1748_ = v_isSharedCheck_1810_;
goto v_resetjp_1746_;
}
v_resetjp_1746_:
{
uint8_t v___x_1749_; 
v___x_1749_ = lean_unbox(v_a_1745_);
lean_dec(v_a_1745_);
if (v___x_1749_ == 0)
{
lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1756_; 
v___x_1750_ = lean_box(v___x_1700_);
v___x_1751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1751_, 0, v_a_1718_);
lean_ctor_set(v___x_1751_, 1, v___x_1750_);
v___x_1752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1752_, 0, v_a_1716_);
lean_ctor_set(v___x_1752_, 1, v___x_1751_);
v___x_1753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1753_, 0, v_a_1713_);
lean_ctor_set(v___x_1753_, 1, v___x_1752_);
v___x_1754_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1754_, 0, v_a_1708_);
lean_ctor_set(v___x_1754_, 1, v___x_1753_);
if (v_isShared_1748_ == 0)
{
lean_ctor_set(v___x_1747_, 0, v___x_1754_);
v___x_1756_ = v___x_1747_;
goto v_reusejp_1755_;
}
else
{
lean_object* v_reuseFailAlloc_1757_; 
v_reuseFailAlloc_1757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1757_, 0, v___x_1754_);
v___x_1756_ = v_reuseFailAlloc_1757_;
goto v_reusejp_1755_;
}
v_reusejp_1755_:
{
return v___x_1756_;
}
}
else
{
lean_object* v___x_1758_; 
lean_del_object(v___x_1747_);
v___x_1758_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_a_1708_, v___y_1703_);
if (lean_obj_tag(v___x_1758_) == 0)
{
lean_object* v_a_1759_; lean_object* v___x_1760_; 
v_a_1759_ = lean_ctor_get(v___x_1758_, 0);
lean_inc(v_a_1759_);
lean_dec_ref_known(v___x_1758_, 1);
v___x_1760_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1713_, v___y_1703_);
if (lean_obj_tag(v___x_1760_) == 0)
{
lean_object* v_a_1761_; lean_object* v___x_1762_; 
v_a_1761_ = lean_ctor_get(v___x_1760_, 0);
lean_inc(v_a_1761_);
lean_dec_ref_known(v___x_1760_, 1);
v___x_1762_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1716_, v___y_1703_);
if (lean_obj_tag(v___x_1762_) == 0)
{
lean_object* v_a_1763_; lean_object* v___x_1764_; 
v_a_1763_ = lean_ctor_get(v___x_1762_, 0);
lean_inc(v_a_1763_);
lean_dec_ref_known(v___x_1762_, 1);
v___x_1764_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1718_, v___y_1703_);
if (lean_obj_tag(v___x_1764_) == 0)
{
lean_object* v_a_1765_; lean_object* v___x_1767_; uint8_t v_isShared_1768_; uint8_t v_isSharedCheck_1777_; 
v_a_1765_ = lean_ctor_get(v___x_1764_, 0);
v_isSharedCheck_1777_ = !lean_is_exclusive(v___x_1764_);
if (v_isSharedCheck_1777_ == 0)
{
v___x_1767_ = v___x_1764_;
v_isShared_1768_ = v_isSharedCheck_1777_;
goto v_resetjp_1766_;
}
else
{
lean_inc(v_a_1765_);
lean_dec(v___x_1764_);
v___x_1767_ = lean_box(0);
v_isShared_1768_ = v_isSharedCheck_1777_;
goto v_resetjp_1766_;
}
v_resetjp_1766_:
{
lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1775_; 
v___x_1769_ = lean_box(v_a_1701_);
v___x_1770_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1770_, 0, v_a_1765_);
lean_ctor_set(v___x_1770_, 1, v___x_1769_);
v___x_1771_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1771_, 0, v_a_1763_);
lean_ctor_set(v___x_1771_, 1, v___x_1770_);
v___x_1772_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1772_, 0, v_a_1761_);
lean_ctor_set(v___x_1772_, 1, v___x_1771_);
v___x_1773_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1773_, 0, v_a_1759_);
lean_ctor_set(v___x_1773_, 1, v___x_1772_);
if (v_isShared_1768_ == 0)
{
lean_ctor_set(v___x_1767_, 0, v___x_1773_);
v___x_1775_ = v___x_1767_;
goto v_reusejp_1774_;
}
else
{
lean_object* v_reuseFailAlloc_1776_; 
v_reuseFailAlloc_1776_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1776_, 0, v___x_1773_);
v___x_1775_ = v_reuseFailAlloc_1776_;
goto v_reusejp_1774_;
}
v_reusejp_1774_:
{
return v___x_1775_;
}
}
}
else
{
lean_object* v_a_1778_; lean_object* v___x_1780_; uint8_t v_isShared_1781_; uint8_t v_isSharedCheck_1785_; 
lean_dec(v_a_1763_);
lean_dec(v_a_1761_);
lean_dec(v_a_1759_);
v_a_1778_ = lean_ctor_get(v___x_1764_, 0);
v_isSharedCheck_1785_ = !lean_is_exclusive(v___x_1764_);
if (v_isSharedCheck_1785_ == 0)
{
v___x_1780_ = v___x_1764_;
v_isShared_1781_ = v_isSharedCheck_1785_;
goto v_resetjp_1779_;
}
else
{
lean_inc(v_a_1778_);
lean_dec(v___x_1764_);
v___x_1780_ = lean_box(0);
v_isShared_1781_ = v_isSharedCheck_1785_;
goto v_resetjp_1779_;
}
v_resetjp_1779_:
{
lean_object* v___x_1783_; 
if (v_isShared_1781_ == 0)
{
v___x_1783_ = v___x_1780_;
goto v_reusejp_1782_;
}
else
{
lean_object* v_reuseFailAlloc_1784_; 
v_reuseFailAlloc_1784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1784_, 0, v_a_1778_);
v___x_1783_ = v_reuseFailAlloc_1784_;
goto v_reusejp_1782_;
}
v_reusejp_1782_:
{
return v___x_1783_;
}
}
}
}
else
{
lean_object* v_a_1786_; lean_object* v___x_1788_; uint8_t v_isShared_1789_; uint8_t v_isSharedCheck_1793_; 
lean_dec(v_a_1761_);
lean_dec(v_a_1759_);
lean_dec(v_a_1718_);
v_a_1786_ = lean_ctor_get(v___x_1762_, 0);
v_isSharedCheck_1793_ = !lean_is_exclusive(v___x_1762_);
if (v_isSharedCheck_1793_ == 0)
{
v___x_1788_ = v___x_1762_;
v_isShared_1789_ = v_isSharedCheck_1793_;
goto v_resetjp_1787_;
}
else
{
lean_inc(v_a_1786_);
lean_dec(v___x_1762_);
v___x_1788_ = lean_box(0);
v_isShared_1789_ = v_isSharedCheck_1793_;
goto v_resetjp_1787_;
}
v_resetjp_1787_:
{
lean_object* v___x_1791_; 
if (v_isShared_1789_ == 0)
{
v___x_1791_ = v___x_1788_;
goto v_reusejp_1790_;
}
else
{
lean_object* v_reuseFailAlloc_1792_; 
v_reuseFailAlloc_1792_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1792_, 0, v_a_1786_);
v___x_1791_ = v_reuseFailAlloc_1792_;
goto v_reusejp_1790_;
}
v_reusejp_1790_:
{
return v___x_1791_;
}
}
}
}
else
{
lean_object* v_a_1794_; lean_object* v___x_1796_; uint8_t v_isShared_1797_; uint8_t v_isSharedCheck_1801_; 
lean_dec(v_a_1759_);
lean_dec(v_a_1718_);
lean_dec(v_a_1716_);
v_a_1794_ = lean_ctor_get(v___x_1760_, 0);
v_isSharedCheck_1801_ = !lean_is_exclusive(v___x_1760_);
if (v_isSharedCheck_1801_ == 0)
{
v___x_1796_ = v___x_1760_;
v_isShared_1797_ = v_isSharedCheck_1801_;
goto v_resetjp_1795_;
}
else
{
lean_inc(v_a_1794_);
lean_dec(v___x_1760_);
v___x_1796_ = lean_box(0);
v_isShared_1797_ = v_isSharedCheck_1801_;
goto v_resetjp_1795_;
}
v_resetjp_1795_:
{
lean_object* v___x_1799_; 
if (v_isShared_1797_ == 0)
{
v___x_1799_ = v___x_1796_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1800_; 
v_reuseFailAlloc_1800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1800_, 0, v_a_1794_);
v___x_1799_ = v_reuseFailAlloc_1800_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
return v___x_1799_;
}
}
}
}
else
{
lean_object* v_a_1802_; lean_object* v___x_1804_; uint8_t v_isShared_1805_; uint8_t v_isSharedCheck_1809_; 
lean_dec(v_a_1718_);
lean_dec(v_a_1716_);
lean_dec(v_a_1713_);
v_a_1802_ = lean_ctor_get(v___x_1758_, 0);
v_isSharedCheck_1809_ = !lean_is_exclusive(v___x_1758_);
if (v_isSharedCheck_1809_ == 0)
{
v___x_1804_ = v___x_1758_;
v_isShared_1805_ = v_isSharedCheck_1809_;
goto v_resetjp_1803_;
}
else
{
lean_inc(v_a_1802_);
lean_dec(v___x_1758_);
v___x_1804_ = lean_box(0);
v_isShared_1805_ = v_isSharedCheck_1809_;
goto v_resetjp_1803_;
}
v_resetjp_1803_:
{
lean_object* v___x_1807_; 
if (v_isShared_1805_ == 0)
{
v___x_1807_ = v___x_1804_;
goto v_reusejp_1806_;
}
else
{
lean_object* v_reuseFailAlloc_1808_; 
v_reuseFailAlloc_1808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1808_, 0, v_a_1802_);
v___x_1807_ = v_reuseFailAlloc_1808_;
goto v_reusejp_1806_;
}
v_reusejp_1806_:
{
return v___x_1807_;
}
}
}
}
}
}
else
{
lean_object* v_a_1811_; lean_object* v___x_1813_; uint8_t v_isShared_1814_; uint8_t v_isSharedCheck_1818_; 
lean_dec(v_a_1718_);
lean_dec(v_a_1716_);
lean_dec(v_a_1713_);
lean_dec(v_a_1708_);
v_a_1811_ = lean_ctor_get(v___x_1744_, 0);
v_isSharedCheck_1818_ = !lean_is_exclusive(v___x_1744_);
if (v_isSharedCheck_1818_ == 0)
{
v___x_1813_ = v___x_1744_;
v_isShared_1814_ = v_isSharedCheck_1818_;
goto v_resetjp_1812_;
}
else
{
lean_inc(v_a_1811_);
lean_dec(v___x_1744_);
v___x_1813_ = lean_box(0);
v_isShared_1814_ = v_isSharedCheck_1818_;
goto v_resetjp_1812_;
}
v_resetjp_1812_:
{
lean_object* v___x_1816_; 
if (v_isShared_1814_ == 0)
{
v___x_1816_ = v___x_1813_;
goto v_reusejp_1815_;
}
else
{
lean_object* v_reuseFailAlloc_1817_; 
v_reuseFailAlloc_1817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1817_, 0, v_a_1811_);
v___x_1816_ = v_reuseFailAlloc_1817_;
goto v_reusejp_1815_;
}
v_reusejp_1815_:
{
return v___x_1816_;
}
}
}
}
}
}
else
{
lean_object* v_a_1821_; lean_object* v___x_1823_; uint8_t v_isShared_1824_; uint8_t v_isSharedCheck_1828_; 
lean_dec(v_a_1716_);
lean_dec(v_a_1713_);
lean_dec(v___x_1709_);
lean_dec(v_a_1708_);
lean_dec_ref(v___y_1702_);
lean_dec_ref(v_fst_1699_);
v_a_1821_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1828_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1828_ == 0)
{
v___x_1823_ = v___x_1717_;
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
else
{
lean_inc(v_a_1821_);
lean_dec(v___x_1717_);
v___x_1823_ = lean_box(0);
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
v_resetjp_1822_:
{
lean_object* v___x_1826_; 
if (v_isShared_1824_ == 0)
{
v___x_1826_ = v___x_1823_;
goto v_reusejp_1825_;
}
else
{
lean_object* v_reuseFailAlloc_1827_; 
v_reuseFailAlloc_1827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1827_, 0, v_a_1821_);
v___x_1826_ = v_reuseFailAlloc_1827_;
goto v_reusejp_1825_;
}
v_reusejp_1825_:
{
return v___x_1826_;
}
}
}
}
else
{
lean_object* v_a_1829_; lean_object* v___x_1831_; uint8_t v_isShared_1832_; uint8_t v_isSharedCheck_1836_; 
lean_dec_ref_known(v___x_1714_, 1);
lean_dec(v_a_1713_);
lean_dec(v___x_1709_);
lean_dec(v_a_1708_);
lean_dec_ref(v___y_1702_);
lean_dec_ref(v_fst_1699_);
lean_dec(v___x_1698_);
v_a_1829_ = lean_ctor_get(v___x_1715_, 0);
v_isSharedCheck_1836_ = !lean_is_exclusive(v___x_1715_);
if (v_isSharedCheck_1836_ == 0)
{
v___x_1831_ = v___x_1715_;
v_isShared_1832_ = v_isSharedCheck_1836_;
goto v_resetjp_1830_;
}
else
{
lean_inc(v_a_1829_);
lean_dec(v___x_1715_);
v___x_1831_ = lean_box(0);
v_isShared_1832_ = v_isSharedCheck_1836_;
goto v_resetjp_1830_;
}
v_resetjp_1830_:
{
lean_object* v___x_1834_; 
if (v_isShared_1832_ == 0)
{
v___x_1834_ = v___x_1831_;
goto v_reusejp_1833_;
}
else
{
lean_object* v_reuseFailAlloc_1835_; 
v_reuseFailAlloc_1835_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1835_, 0, v_a_1829_);
v___x_1834_ = v_reuseFailAlloc_1835_;
goto v_reusejp_1833_;
}
v_reusejp_1833_:
{
return v___x_1834_;
}
}
}
}
else
{
lean_object* v_a_1837_; lean_object* v___x_1839_; uint8_t v_isShared_1840_; uint8_t v_isSharedCheck_1844_; 
lean_dec(v___x_1709_);
lean_dec(v_a_1708_);
lean_dec_ref(v___y_1702_);
lean_dec_ref(v_fst_1699_);
lean_dec(v___x_1698_);
v_a_1837_ = lean_ctor_get(v___x_1712_, 0);
v_isSharedCheck_1844_ = !lean_is_exclusive(v___x_1712_);
if (v_isSharedCheck_1844_ == 0)
{
v___x_1839_ = v___x_1712_;
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
else
{
lean_inc(v_a_1837_);
lean_dec(v___x_1712_);
v___x_1839_ = lean_box(0);
v_isShared_1840_ = v_isSharedCheck_1844_;
goto v_resetjp_1838_;
}
v_resetjp_1838_:
{
lean_object* v___x_1842_; 
if (v_isShared_1840_ == 0)
{
v___x_1842_ = v___x_1839_;
goto v_reusejp_1841_;
}
else
{
lean_object* v_reuseFailAlloc_1843_; 
v_reuseFailAlloc_1843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1843_, 0, v_a_1837_);
v___x_1842_ = v_reuseFailAlloc_1843_;
goto v_reusejp_1841_;
}
v_reusejp_1841_:
{
return v___x_1842_;
}
}
}
}
else
{
lean_object* v_a_1845_; lean_object* v___x_1847_; uint8_t v_isShared_1848_; uint8_t v_isSharedCheck_1852_; 
lean_dec_ref(v___y_1702_);
lean_dec_ref(v_fst_1699_);
lean_dec(v___x_1698_);
v_a_1845_ = lean_ctor_get(v___x_1707_, 0);
v_isSharedCheck_1852_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1852_ == 0)
{
v___x_1847_ = v___x_1707_;
v_isShared_1848_ = v_isSharedCheck_1852_;
goto v_resetjp_1846_;
}
else
{
lean_inc(v_a_1845_);
lean_dec(v___x_1707_);
v___x_1847_ = lean_box(0);
v_isShared_1848_ = v_isSharedCheck_1852_;
goto v_resetjp_1846_;
}
v_resetjp_1846_:
{
lean_object* v___x_1850_; 
if (v_isShared_1848_ == 0)
{
v___x_1850_ = v___x_1847_;
goto v_reusejp_1849_;
}
else
{
lean_object* v_reuseFailAlloc_1851_; 
v_reuseFailAlloc_1851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1851_, 0, v_a_1845_);
v___x_1850_ = v_reuseFailAlloc_1851_;
goto v_reusejp_1849_;
}
v_reusejp_1849_:
{
return v___x_1850_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___boxed(lean_object* v___x_1853_, lean_object* v___x_1854_, lean_object* v_fst_1855_, lean_object* v___x_1856_, lean_object* v_a_1857_, lean_object* v___y_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_){
_start:
{
uint8_t v___x_190197__boxed_1863_; uint8_t v___x_190200__boxed_1864_; uint8_t v_a_190201__boxed_1865_; lean_object* v_res_1866_; 
v___x_190197__boxed_1863_ = lean_unbox(v___x_1853_);
v___x_190200__boxed_1864_ = lean_unbox(v___x_1856_);
v_a_190201__boxed_1865_ = lean_unbox(v_a_1857_);
v_res_1866_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0(v___x_190197__boxed_1863_, v___x_1854_, v_fst_1855_, v___x_190200__boxed_1864_, v_a_190201__boxed_1865_, v___y_1858_, v___y_1859_, v___y_1860_, v___y_1861_);
lean_dec(v___y_1861_);
lean_dec_ref(v___y_1860_);
lean_dec(v___y_1859_);
return v_res_1866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1(uint8_t v___x_1871_, lean_object* v___x_1872_, lean_object* v_fst_1873_, uint8_t v___x_1874_, uint8_t v_a_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_){
_start:
{
lean_object* v___x_1881_; 
v___x_1881_ = l_Lean_Meta_mkFreshLevelMVar(v___y_1876_, v___y_1877_, v___y_1878_, v___y_1879_);
if (lean_obj_tag(v___x_1881_) == 0)
{
lean_object* v_a_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; 
v_a_1882_ = lean_ctor_get(v___x_1881_, 0);
lean_inc_n(v_a_1882_, 2);
lean_dec_ref_known(v___x_1881_, 1);
v___x_1883_ = l_Lean_Level_succ___override(v_a_1882_);
v___x_1884_ = l_Lean_Expr_sort___override(v___x_1883_);
v___x_1885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1885_, 0, v___x_1884_);
lean_inc(v___x_1872_);
v___x_1886_ = l_Lean_Meta_mkFreshExprMVar(v___x_1885_, v___x_1871_, v___x_1872_, v___y_1876_, v___y_1877_, v___y_1878_, v___y_1879_);
if (lean_obj_tag(v___x_1886_) == 0)
{
lean_object* v_a_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; 
v_a_1887_ = lean_ctor_get(v___x_1886_, 0);
lean_inc_n(v_a_1887_, 2);
lean_dec_ref_known(v___x_1886_, 1);
v___x_1888_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__5));
v___x_1889_ = lean_box(0);
lean_inc(v_a_1882_);
v___x_1890_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1890_, 0, v_a_1882_);
lean_ctor_set(v___x_1890_, 1, v___x_1889_);
lean_inc_ref(v___x_1890_);
v___x_1891_ = l_Lean_Expr_const___override(v___x_1888_, v___x_1890_);
v___x_1892_ = l_Lean_Expr_app___override(v___x_1891_, v_a_1887_);
v___x_1893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1893_, 0, v___x_1892_);
lean_inc(v___x_1872_);
v___x_1894_ = l_Lean_Meta_mkFreshExprMVar(v___x_1893_, v___x_1871_, v___x_1872_, v___y_1876_, v___y_1877_, v___y_1878_, v___y_1879_);
if (lean_obj_tag(v___x_1894_) == 0)
{
lean_object* v_a_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; 
v_a_1895_ = lean_ctor_get(v___x_1894_, 0);
lean_inc(v_a_1895_);
lean_dec_ref_known(v___x_1894_, 1);
lean_inc(v_a_1887_);
v___x_1896_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1896_, 0, v_a_1887_);
lean_inc(v___x_1872_);
lean_inc_ref(v___x_1896_);
v___x_1897_ = l_Lean_Meta_mkFreshExprMVar(v___x_1896_, v___x_1871_, v___x_1872_, v___y_1876_, v___y_1877_, v___y_1878_, v___y_1879_);
if (lean_obj_tag(v___x_1897_) == 0)
{
lean_object* v_a_1898_; lean_object* v___x_1899_; 
v_a_1898_ = lean_ctor_get(v___x_1897_, 0);
lean_inc(v_a_1898_);
lean_dec_ref_known(v___x_1897_, 1);
v___x_1899_ = l_Lean_Meta_mkFreshExprMVar(v___x_1896_, v___x_1871_, v___x_1872_, v___y_1876_, v___y_1877_, v___y_1878_, v___y_1879_);
if (lean_obj_tag(v___x_1899_) == 0)
{
lean_object* v_a_1900_; lean_object* v_keyedConfig_1901_; uint8_t v_trackZetaDelta_1902_; lean_object* v_zetaDeltaSet_1903_; lean_object* v_lctx_1904_; lean_object* v_localInstances_1905_; lean_object* v_defEqCtx_x3f_1906_; lean_object* v_synthPendingDepth_1907_; lean_object* v_customCanUnfoldPredicate_x3f_1908_; uint8_t v_univApprox_1909_; uint8_t v_inTypeClassResolution_1910_; uint8_t v_cacheInferType_1911_; lean_object* v___x_1913_; uint8_t v_isShared_1914_; uint8_t v_isSharedCheck_2013_; 
v_a_1900_ = lean_ctor_get(v___x_1899_, 0);
lean_inc(v_a_1900_);
lean_dec_ref_known(v___x_1899_, 1);
v_keyedConfig_1901_ = lean_ctor_get(v___y_1876_, 0);
v_trackZetaDelta_1902_ = lean_ctor_get_uint8(v___y_1876_, sizeof(void*)*7);
v_zetaDeltaSet_1903_ = lean_ctor_get(v___y_1876_, 1);
v_lctx_1904_ = lean_ctor_get(v___y_1876_, 2);
v_localInstances_1905_ = lean_ctor_get(v___y_1876_, 3);
v_defEqCtx_x3f_1906_ = lean_ctor_get(v___y_1876_, 4);
v_synthPendingDepth_1907_ = lean_ctor_get(v___y_1876_, 5);
v_customCanUnfoldPredicate_x3f_1908_ = lean_ctor_get(v___y_1876_, 6);
v_univApprox_1909_ = lean_ctor_get_uint8(v___y_1876_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1910_ = lean_ctor_get_uint8(v___y_1876_, sizeof(void*)*7 + 2);
v_cacheInferType_1911_ = lean_ctor_get_uint8(v___y_1876_, sizeof(void*)*7 + 3);
v_isSharedCheck_2013_ = !lean_is_exclusive(v___y_1876_);
if (v_isSharedCheck_2013_ == 0)
{
v___x_1913_ = v___y_1876_;
v_isShared_1914_ = v_isSharedCheck_2013_;
goto v_resetjp_1912_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1908_);
lean_inc(v_synthPendingDepth_1907_);
lean_inc(v_defEqCtx_x3f_1906_);
lean_inc(v_localInstances_1905_);
lean_inc(v_lctx_1904_);
lean_inc(v_zetaDeltaSet_1903_);
lean_inc(v_keyedConfig_1901_);
lean_dec(v___y_1876_);
v___x_1913_ = lean_box(0);
v_isShared_1914_ = v_isSharedCheck_2013_;
goto v_resetjp_1912_;
}
v_resetjp_1912_:
{
lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; uint8_t v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1924_; 
v___x_1915_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__1));
v___x_1916_ = l_Lean_Expr_const___override(v___x_1915_, v___x_1890_);
lean_inc(v_a_1887_);
v___x_1917_ = l_Lean_Expr_app___override(v___x_1916_, v_a_1887_);
lean_inc(v_a_1895_);
v___x_1918_ = l_Lean_Expr_app___override(v___x_1917_, v_a_1895_);
lean_inc(v_a_1898_);
v___x_1919_ = l_Lean_Expr_app___override(v___x_1918_, v_a_1898_);
lean_inc(v_a_1900_);
v___x_1920_ = l_Lean_Expr_app___override(v___x_1919_, v_a_1900_);
v___x_1921_ = 2;
v___x_1922_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1921_, v_keyedConfig_1901_);
if (v_isShared_1914_ == 0)
{
lean_ctor_set(v___x_1913_, 0, v___x_1922_);
v___x_1924_ = v___x_1913_;
goto v_reusejp_1923_;
}
else
{
lean_object* v_reuseFailAlloc_2012_; 
v_reuseFailAlloc_2012_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2012_, 0, v___x_1922_);
lean_ctor_set(v_reuseFailAlloc_2012_, 1, v_zetaDeltaSet_1903_);
lean_ctor_set(v_reuseFailAlloc_2012_, 2, v_lctx_1904_);
lean_ctor_set(v_reuseFailAlloc_2012_, 3, v_localInstances_1905_);
lean_ctor_set(v_reuseFailAlloc_2012_, 4, v_defEqCtx_x3f_1906_);
lean_ctor_set(v_reuseFailAlloc_2012_, 5, v_synthPendingDepth_1907_);
lean_ctor_set(v_reuseFailAlloc_2012_, 6, v_customCanUnfoldPredicate_x3f_1908_);
lean_ctor_set_uint8(v_reuseFailAlloc_2012_, sizeof(void*)*7, v_trackZetaDelta_1902_);
lean_ctor_set_uint8(v_reuseFailAlloc_2012_, sizeof(void*)*7 + 1, v_univApprox_1909_);
lean_ctor_set_uint8(v_reuseFailAlloc_2012_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1910_);
lean_ctor_set_uint8(v_reuseFailAlloc_2012_, sizeof(void*)*7 + 3, v_cacheInferType_1911_);
v___x_1924_ = v_reuseFailAlloc_2012_;
goto v_reusejp_1923_;
}
v_reusejp_1923_:
{
lean_object* v___x_1925_; 
v___x_1925_ = l_Lean_Meta_isExprDefEq(v___x_1920_, v_fst_1873_, v___x_1924_, v___y_1877_, v___y_1878_, v___y_1879_);
lean_dec_ref(v___x_1924_);
if (lean_obj_tag(v___x_1925_) == 0)
{
lean_object* v_a_1926_; lean_object* v___x_1928_; uint8_t v_isShared_1929_; uint8_t v_isSharedCheck_2003_; 
v_a_1926_ = lean_ctor_get(v___x_1925_, 0);
v_isSharedCheck_2003_ = !lean_is_exclusive(v___x_1925_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1928_ = v___x_1925_;
v_isShared_1929_ = v_isSharedCheck_2003_;
goto v_resetjp_1927_;
}
else
{
lean_inc(v_a_1926_);
lean_dec(v___x_1925_);
v___x_1928_ = lean_box(0);
v_isShared_1929_ = v_isSharedCheck_2003_;
goto v_resetjp_1927_;
}
v_resetjp_1927_:
{
uint8_t v___x_1930_; 
v___x_1930_ = lean_unbox(v_a_1926_);
lean_dec(v_a_1926_);
if (v___x_1930_ == 0)
{
lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1938_; 
v___x_1931_ = lean_box(v___x_1874_);
v___x_1932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1932_, 0, v_a_1900_);
lean_ctor_set(v___x_1932_, 1, v___x_1931_);
v___x_1933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1933_, 0, v_a_1898_);
lean_ctor_set(v___x_1933_, 1, v___x_1932_);
v___x_1934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1934_, 0, v_a_1895_);
lean_ctor_set(v___x_1934_, 1, v___x_1933_);
v___x_1935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1935_, 0, v_a_1887_);
lean_ctor_set(v___x_1935_, 1, v___x_1934_);
v___x_1936_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1936_, 0, v_a_1882_);
lean_ctor_set(v___x_1936_, 1, v___x_1935_);
if (v_isShared_1929_ == 0)
{
lean_ctor_set(v___x_1928_, 0, v___x_1936_);
v___x_1938_ = v___x_1928_;
goto v_reusejp_1937_;
}
else
{
lean_object* v_reuseFailAlloc_1939_; 
v_reuseFailAlloc_1939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1939_, 0, v___x_1936_);
v___x_1938_ = v_reuseFailAlloc_1939_;
goto v_reusejp_1937_;
}
v_reusejp_1937_:
{
return v___x_1938_;
}
}
else
{
lean_object* v___x_1940_; 
lean_del_object(v___x_1928_);
v___x_1940_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_a_1882_, v___y_1877_);
if (lean_obj_tag(v___x_1940_) == 0)
{
lean_object* v_a_1941_; lean_object* v___x_1942_; 
v_a_1941_ = lean_ctor_get(v___x_1940_, 0);
lean_inc(v_a_1941_);
lean_dec_ref_known(v___x_1940_, 1);
v___x_1942_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1887_, v___y_1877_);
if (lean_obj_tag(v___x_1942_) == 0)
{
lean_object* v_a_1943_; lean_object* v___x_1944_; 
v_a_1943_ = lean_ctor_get(v___x_1942_, 0);
lean_inc(v_a_1943_);
lean_dec_ref_known(v___x_1942_, 1);
v___x_1944_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1895_, v___y_1877_);
if (lean_obj_tag(v___x_1944_) == 0)
{
lean_object* v_a_1945_; lean_object* v___x_1946_; 
v_a_1945_ = lean_ctor_get(v___x_1944_, 0);
lean_inc(v_a_1945_);
lean_dec_ref_known(v___x_1944_, 1);
v___x_1946_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1898_, v___y_1877_);
if (lean_obj_tag(v___x_1946_) == 0)
{
lean_object* v_a_1947_; lean_object* v___x_1948_; 
v_a_1947_ = lean_ctor_get(v___x_1946_, 0);
lean_inc(v_a_1947_);
lean_dec_ref_known(v___x_1946_, 1);
v___x_1948_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_1900_, v___y_1877_);
if (lean_obj_tag(v___x_1948_) == 0)
{
lean_object* v_a_1949_; lean_object* v___x_1951_; uint8_t v_isShared_1952_; uint8_t v_isSharedCheck_1962_; 
v_a_1949_ = lean_ctor_get(v___x_1948_, 0);
v_isSharedCheck_1962_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1962_ == 0)
{
v___x_1951_ = v___x_1948_;
v_isShared_1952_ = v_isSharedCheck_1962_;
goto v_resetjp_1950_;
}
else
{
lean_inc(v_a_1949_);
lean_dec(v___x_1948_);
v___x_1951_ = lean_box(0);
v_isShared_1952_ = v_isSharedCheck_1962_;
goto v_resetjp_1950_;
}
v_resetjp_1950_:
{
lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; lean_object* v___x_1960_; 
v___x_1953_ = lean_box(v_a_1875_);
v___x_1954_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1954_, 0, v_a_1949_);
lean_ctor_set(v___x_1954_, 1, v___x_1953_);
v___x_1955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1955_, 0, v_a_1947_);
lean_ctor_set(v___x_1955_, 1, v___x_1954_);
v___x_1956_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1956_, 0, v_a_1945_);
lean_ctor_set(v___x_1956_, 1, v___x_1955_);
v___x_1957_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1957_, 0, v_a_1943_);
lean_ctor_set(v___x_1957_, 1, v___x_1956_);
v___x_1958_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1958_, 0, v_a_1941_);
lean_ctor_set(v___x_1958_, 1, v___x_1957_);
if (v_isShared_1952_ == 0)
{
lean_ctor_set(v___x_1951_, 0, v___x_1958_);
v___x_1960_ = v___x_1951_;
goto v_reusejp_1959_;
}
else
{
lean_object* v_reuseFailAlloc_1961_; 
v_reuseFailAlloc_1961_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1961_, 0, v___x_1958_);
v___x_1960_ = v_reuseFailAlloc_1961_;
goto v_reusejp_1959_;
}
v_reusejp_1959_:
{
return v___x_1960_;
}
}
}
else
{
lean_object* v_a_1963_; lean_object* v___x_1965_; uint8_t v_isShared_1966_; uint8_t v_isSharedCheck_1970_; 
lean_dec(v_a_1947_);
lean_dec(v_a_1945_);
lean_dec(v_a_1943_);
lean_dec(v_a_1941_);
v_a_1963_ = lean_ctor_get(v___x_1948_, 0);
v_isSharedCheck_1970_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1970_ == 0)
{
v___x_1965_ = v___x_1948_;
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
else
{
lean_inc(v_a_1963_);
lean_dec(v___x_1948_);
v___x_1965_ = lean_box(0);
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
v_resetjp_1964_:
{
lean_object* v___x_1968_; 
if (v_isShared_1966_ == 0)
{
v___x_1968_ = v___x_1965_;
goto v_reusejp_1967_;
}
else
{
lean_object* v_reuseFailAlloc_1969_; 
v_reuseFailAlloc_1969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1969_, 0, v_a_1963_);
v___x_1968_ = v_reuseFailAlloc_1969_;
goto v_reusejp_1967_;
}
v_reusejp_1967_:
{
return v___x_1968_;
}
}
}
}
else
{
lean_object* v_a_1971_; lean_object* v___x_1973_; uint8_t v_isShared_1974_; uint8_t v_isSharedCheck_1978_; 
lean_dec(v_a_1945_);
lean_dec(v_a_1943_);
lean_dec(v_a_1941_);
lean_dec(v_a_1900_);
v_a_1971_ = lean_ctor_get(v___x_1946_, 0);
v_isSharedCheck_1978_ = !lean_is_exclusive(v___x_1946_);
if (v_isSharedCheck_1978_ == 0)
{
v___x_1973_ = v___x_1946_;
v_isShared_1974_ = v_isSharedCheck_1978_;
goto v_resetjp_1972_;
}
else
{
lean_inc(v_a_1971_);
lean_dec(v___x_1946_);
v___x_1973_ = lean_box(0);
v_isShared_1974_ = v_isSharedCheck_1978_;
goto v_resetjp_1972_;
}
v_resetjp_1972_:
{
lean_object* v___x_1976_; 
if (v_isShared_1974_ == 0)
{
v___x_1976_ = v___x_1973_;
goto v_reusejp_1975_;
}
else
{
lean_object* v_reuseFailAlloc_1977_; 
v_reuseFailAlloc_1977_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1977_, 0, v_a_1971_);
v___x_1976_ = v_reuseFailAlloc_1977_;
goto v_reusejp_1975_;
}
v_reusejp_1975_:
{
return v___x_1976_;
}
}
}
}
else
{
lean_object* v_a_1979_; lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_1986_; 
lean_dec(v_a_1943_);
lean_dec(v_a_1941_);
lean_dec(v_a_1900_);
lean_dec(v_a_1898_);
v_a_1979_ = lean_ctor_get(v___x_1944_, 0);
v_isSharedCheck_1986_ = !lean_is_exclusive(v___x_1944_);
if (v_isSharedCheck_1986_ == 0)
{
v___x_1981_ = v___x_1944_;
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
else
{
lean_inc(v_a_1979_);
lean_dec(v___x_1944_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
lean_object* v___x_1984_; 
if (v_isShared_1982_ == 0)
{
v___x_1984_ = v___x_1981_;
goto v_reusejp_1983_;
}
else
{
lean_object* v_reuseFailAlloc_1985_; 
v_reuseFailAlloc_1985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1985_, 0, v_a_1979_);
v___x_1984_ = v_reuseFailAlloc_1985_;
goto v_reusejp_1983_;
}
v_reusejp_1983_:
{
return v___x_1984_;
}
}
}
}
else
{
lean_object* v_a_1987_; lean_object* v___x_1989_; uint8_t v_isShared_1990_; uint8_t v_isSharedCheck_1994_; 
lean_dec(v_a_1941_);
lean_dec(v_a_1900_);
lean_dec(v_a_1898_);
lean_dec(v_a_1895_);
v_a_1987_ = lean_ctor_get(v___x_1942_, 0);
v_isSharedCheck_1994_ = !lean_is_exclusive(v___x_1942_);
if (v_isSharedCheck_1994_ == 0)
{
v___x_1989_ = v___x_1942_;
v_isShared_1990_ = v_isSharedCheck_1994_;
goto v_resetjp_1988_;
}
else
{
lean_inc(v_a_1987_);
lean_dec(v___x_1942_);
v___x_1989_ = lean_box(0);
v_isShared_1990_ = v_isSharedCheck_1994_;
goto v_resetjp_1988_;
}
v_resetjp_1988_:
{
lean_object* v___x_1992_; 
if (v_isShared_1990_ == 0)
{
v___x_1992_ = v___x_1989_;
goto v_reusejp_1991_;
}
else
{
lean_object* v_reuseFailAlloc_1993_; 
v_reuseFailAlloc_1993_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1993_, 0, v_a_1987_);
v___x_1992_ = v_reuseFailAlloc_1993_;
goto v_reusejp_1991_;
}
v_reusejp_1991_:
{
return v___x_1992_;
}
}
}
}
else
{
lean_object* v_a_1995_; lean_object* v___x_1997_; uint8_t v_isShared_1998_; uint8_t v_isSharedCheck_2002_; 
lean_dec(v_a_1900_);
lean_dec(v_a_1898_);
lean_dec(v_a_1895_);
lean_dec(v_a_1887_);
v_a_1995_ = lean_ctor_get(v___x_1940_, 0);
v_isSharedCheck_2002_ = !lean_is_exclusive(v___x_1940_);
if (v_isSharedCheck_2002_ == 0)
{
v___x_1997_ = v___x_1940_;
v_isShared_1998_ = v_isSharedCheck_2002_;
goto v_resetjp_1996_;
}
else
{
lean_inc(v_a_1995_);
lean_dec(v___x_1940_);
v___x_1997_ = lean_box(0);
v_isShared_1998_ = v_isSharedCheck_2002_;
goto v_resetjp_1996_;
}
v_resetjp_1996_:
{
lean_object* v___x_2000_; 
if (v_isShared_1998_ == 0)
{
v___x_2000_ = v___x_1997_;
goto v_reusejp_1999_;
}
else
{
lean_object* v_reuseFailAlloc_2001_; 
v_reuseFailAlloc_2001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2001_, 0, v_a_1995_);
v___x_2000_ = v_reuseFailAlloc_2001_;
goto v_reusejp_1999_;
}
v_reusejp_1999_:
{
return v___x_2000_;
}
}
}
}
}
}
else
{
lean_object* v_a_2004_; lean_object* v___x_2006_; uint8_t v_isShared_2007_; uint8_t v_isSharedCheck_2011_; 
lean_dec(v_a_1900_);
lean_dec(v_a_1898_);
lean_dec(v_a_1895_);
lean_dec(v_a_1887_);
lean_dec(v_a_1882_);
v_a_2004_ = lean_ctor_get(v___x_1925_, 0);
v_isSharedCheck_2011_ = !lean_is_exclusive(v___x_1925_);
if (v_isSharedCheck_2011_ == 0)
{
v___x_2006_ = v___x_1925_;
v_isShared_2007_ = v_isSharedCheck_2011_;
goto v_resetjp_2005_;
}
else
{
lean_inc(v_a_2004_);
lean_dec(v___x_1925_);
v___x_2006_ = lean_box(0);
v_isShared_2007_ = v_isSharedCheck_2011_;
goto v_resetjp_2005_;
}
v_resetjp_2005_:
{
lean_object* v___x_2009_; 
if (v_isShared_2007_ == 0)
{
v___x_2009_ = v___x_2006_;
goto v_reusejp_2008_;
}
else
{
lean_object* v_reuseFailAlloc_2010_; 
v_reuseFailAlloc_2010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2010_, 0, v_a_2004_);
v___x_2009_ = v_reuseFailAlloc_2010_;
goto v_reusejp_2008_;
}
v_reusejp_2008_:
{
return v___x_2009_;
}
}
}
}
}
}
else
{
lean_object* v_a_2014_; lean_object* v___x_2016_; uint8_t v_isShared_2017_; uint8_t v_isSharedCheck_2021_; 
lean_dec(v_a_1898_);
lean_dec(v_a_1895_);
lean_dec_ref_known(v___x_1890_, 2);
lean_dec(v_a_1887_);
lean_dec(v_a_1882_);
lean_dec_ref(v___y_1876_);
lean_dec_ref(v_fst_1873_);
v_a_2014_ = lean_ctor_get(v___x_1899_, 0);
v_isSharedCheck_2021_ = !lean_is_exclusive(v___x_1899_);
if (v_isSharedCheck_2021_ == 0)
{
v___x_2016_ = v___x_1899_;
v_isShared_2017_ = v_isSharedCheck_2021_;
goto v_resetjp_2015_;
}
else
{
lean_inc(v_a_2014_);
lean_dec(v___x_1899_);
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
lean_dec_ref_known(v___x_1896_, 1);
lean_dec(v_a_1895_);
lean_dec_ref_known(v___x_1890_, 2);
lean_dec(v_a_1887_);
lean_dec(v_a_1882_);
lean_dec_ref(v___y_1876_);
lean_dec_ref(v_fst_1873_);
lean_dec(v___x_1872_);
v_a_2022_ = lean_ctor_get(v___x_1897_, 0);
v_isSharedCheck_2029_ = !lean_is_exclusive(v___x_1897_);
if (v_isSharedCheck_2029_ == 0)
{
v___x_2024_ = v___x_1897_;
v_isShared_2025_ = v_isSharedCheck_2029_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_a_2022_);
lean_dec(v___x_1897_);
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
else
{
lean_object* v_a_2030_; lean_object* v___x_2032_; uint8_t v_isShared_2033_; uint8_t v_isSharedCheck_2037_; 
lean_dec_ref_known(v___x_1890_, 2);
lean_dec(v_a_1887_);
lean_dec(v_a_1882_);
lean_dec_ref(v___y_1876_);
lean_dec_ref(v_fst_1873_);
lean_dec(v___x_1872_);
v_a_2030_ = lean_ctor_get(v___x_1894_, 0);
v_isSharedCheck_2037_ = !lean_is_exclusive(v___x_1894_);
if (v_isSharedCheck_2037_ == 0)
{
v___x_2032_ = v___x_1894_;
v_isShared_2033_ = v_isSharedCheck_2037_;
goto v_resetjp_2031_;
}
else
{
lean_inc(v_a_2030_);
lean_dec(v___x_1894_);
v___x_2032_ = lean_box(0);
v_isShared_2033_ = v_isSharedCheck_2037_;
goto v_resetjp_2031_;
}
v_resetjp_2031_:
{
lean_object* v___x_2035_; 
if (v_isShared_2033_ == 0)
{
v___x_2035_ = v___x_2032_;
goto v_reusejp_2034_;
}
else
{
lean_object* v_reuseFailAlloc_2036_; 
v_reuseFailAlloc_2036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2036_, 0, v_a_2030_);
v___x_2035_ = v_reuseFailAlloc_2036_;
goto v_reusejp_2034_;
}
v_reusejp_2034_:
{
return v___x_2035_;
}
}
}
}
else
{
lean_object* v_a_2038_; lean_object* v___x_2040_; uint8_t v_isShared_2041_; uint8_t v_isSharedCheck_2045_; 
lean_dec(v_a_1882_);
lean_dec_ref(v___y_1876_);
lean_dec_ref(v_fst_1873_);
lean_dec(v___x_1872_);
v_a_2038_ = lean_ctor_get(v___x_1886_, 0);
v_isSharedCheck_2045_ = !lean_is_exclusive(v___x_1886_);
if (v_isSharedCheck_2045_ == 0)
{
v___x_2040_ = v___x_1886_;
v_isShared_2041_ = v_isSharedCheck_2045_;
goto v_resetjp_2039_;
}
else
{
lean_inc(v_a_2038_);
lean_dec(v___x_1886_);
v___x_2040_ = lean_box(0);
v_isShared_2041_ = v_isSharedCheck_2045_;
goto v_resetjp_2039_;
}
v_resetjp_2039_:
{
lean_object* v___x_2043_; 
if (v_isShared_2041_ == 0)
{
v___x_2043_ = v___x_2040_;
goto v_reusejp_2042_;
}
else
{
lean_object* v_reuseFailAlloc_2044_; 
v_reuseFailAlloc_2044_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2044_, 0, v_a_2038_);
v___x_2043_ = v_reuseFailAlloc_2044_;
goto v_reusejp_2042_;
}
v_reusejp_2042_:
{
return v___x_2043_;
}
}
}
}
else
{
lean_object* v_a_2046_; lean_object* v___x_2048_; uint8_t v_isShared_2049_; uint8_t v_isSharedCheck_2053_; 
lean_dec_ref(v___y_1876_);
lean_dec_ref(v_fst_1873_);
lean_dec(v___x_1872_);
v_a_2046_ = lean_ctor_get(v___x_1881_, 0);
v_isSharedCheck_2053_ = !lean_is_exclusive(v___x_1881_);
if (v_isSharedCheck_2053_ == 0)
{
v___x_2048_ = v___x_1881_;
v_isShared_2049_ = v_isSharedCheck_2053_;
goto v_resetjp_2047_;
}
else
{
lean_inc(v_a_2046_);
lean_dec(v___x_1881_);
v___x_2048_ = lean_box(0);
v_isShared_2049_ = v_isSharedCheck_2053_;
goto v_resetjp_2047_;
}
v_resetjp_2047_:
{
lean_object* v___x_2051_; 
if (v_isShared_2049_ == 0)
{
v___x_2051_ = v___x_2048_;
goto v_reusejp_2050_;
}
else
{
lean_object* v_reuseFailAlloc_2052_; 
v_reuseFailAlloc_2052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2052_, 0, v_a_2046_);
v___x_2051_ = v_reuseFailAlloc_2052_;
goto v_reusejp_2050_;
}
v_reusejp_2050_:
{
return v___x_2051_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___boxed(lean_object* v___x_2054_, lean_object* v___x_2055_, lean_object* v_fst_2056_, lean_object* v___x_2057_, lean_object* v_a_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_, lean_object* v___y_2063_){
_start:
{
uint8_t v___x_190511__boxed_2064_; uint8_t v___x_190514__boxed_2065_; uint8_t v_a_190515__boxed_2066_; lean_object* v_res_2067_; 
v___x_190511__boxed_2064_ = lean_unbox(v___x_2054_);
v___x_190514__boxed_2065_ = lean_unbox(v___x_2057_);
v_a_190515__boxed_2066_ = lean_unbox(v_a_2058_);
v_res_2067_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1(v___x_190511__boxed_2064_, v___x_2055_, v_fst_2056_, v___x_190514__boxed_2065_, v_a_190515__boxed_2066_, v___y_2059_, v___y_2060_, v___y_2061_, v___y_2062_);
lean_dec(v___y_2062_);
lean_dec_ref(v___y_2061_);
lean_dec(v___y_2060_);
return v_res_2067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2(uint8_t v___x_2075_, lean_object* v___x_2076_, lean_object* v_fst_2077_, uint8_t v___x_2078_, uint8_t v_a_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_){
_start:
{
lean_object* v___x_2085_; 
v___x_2085_ = l_Lean_Meta_mkFreshLevelMVar(v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_);
if (lean_obj_tag(v___x_2085_) == 0)
{
lean_object* v_a_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; 
v_a_2086_ = lean_ctor_get(v___x_2085_, 0);
lean_inc_n(v_a_2086_, 2);
lean_dec_ref_known(v___x_2085_, 1);
v___x_2087_ = l_Lean_Level_succ___override(v_a_2086_);
v___x_2088_ = l_Lean_Expr_sort___override(v___x_2087_);
v___x_2089_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2089_, 0, v___x_2088_);
lean_inc(v___x_2076_);
v___x_2090_ = l_Lean_Meta_mkFreshExprMVar(v___x_2089_, v___x_2075_, v___x_2076_, v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_);
if (lean_obj_tag(v___x_2090_) == 0)
{
lean_object* v_a_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; 
v_a_2091_ = lean_ctor_get(v___x_2090_, 0);
lean_inc_n(v_a_2091_, 2);
lean_dec_ref_known(v___x_2090_, 1);
v___x_2092_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__1));
v___x_2093_ = lean_box(0);
lean_inc(v_a_2086_);
v___x_2094_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2094_, 0, v_a_2086_);
lean_ctor_set(v___x_2094_, 1, v___x_2093_);
lean_inc_ref(v___x_2094_);
v___x_2095_ = l_Lean_Expr_const___override(v___x_2092_, v___x_2094_);
v___x_2096_ = l_Lean_Expr_app___override(v___x_2095_, v_a_2091_);
v___x_2097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2097_, 0, v___x_2096_);
lean_inc(v___x_2076_);
v___x_2098_ = l_Lean_Meta_mkFreshExprMVar(v___x_2097_, v___x_2075_, v___x_2076_, v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_);
if (lean_obj_tag(v___x_2098_) == 0)
{
lean_object* v_a_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; 
v_a_2099_ = lean_ctor_get(v___x_2098_, 0);
lean_inc(v_a_2099_);
lean_dec_ref_known(v___x_2098_, 1);
lean_inc(v_a_2091_);
v___x_2100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2100_, 0, v_a_2091_);
lean_inc(v___x_2076_);
lean_inc_ref(v___x_2100_);
v___x_2101_ = l_Lean_Meta_mkFreshExprMVar(v___x_2100_, v___x_2075_, v___x_2076_, v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_);
if (lean_obj_tag(v___x_2101_) == 0)
{
lean_object* v_a_2102_; lean_object* v___x_2103_; 
v_a_2102_ = lean_ctor_get(v___x_2101_, 0);
lean_inc(v_a_2102_);
lean_dec_ref_known(v___x_2101_, 1);
v___x_2103_ = l_Lean_Meta_mkFreshExprMVar(v___x_2100_, v___x_2075_, v___x_2076_, v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_);
if (lean_obj_tag(v___x_2103_) == 0)
{
lean_object* v_a_2104_; lean_object* v_keyedConfig_2105_; uint8_t v_trackZetaDelta_2106_; lean_object* v_zetaDeltaSet_2107_; lean_object* v_lctx_2108_; lean_object* v_localInstances_2109_; lean_object* v_defEqCtx_x3f_2110_; lean_object* v_synthPendingDepth_2111_; lean_object* v_customCanUnfoldPredicate_x3f_2112_; uint8_t v_univApprox_2113_; uint8_t v_inTypeClassResolution_2114_; uint8_t v_cacheInferType_2115_; lean_object* v___x_2117_; uint8_t v_isShared_2118_; uint8_t v_isSharedCheck_2217_; 
v_a_2104_ = lean_ctor_get(v___x_2103_, 0);
lean_inc(v_a_2104_);
lean_dec_ref_known(v___x_2103_, 1);
v_keyedConfig_2105_ = lean_ctor_get(v___y_2080_, 0);
v_trackZetaDelta_2106_ = lean_ctor_get_uint8(v___y_2080_, sizeof(void*)*7);
v_zetaDeltaSet_2107_ = lean_ctor_get(v___y_2080_, 1);
v_lctx_2108_ = lean_ctor_get(v___y_2080_, 2);
v_localInstances_2109_ = lean_ctor_get(v___y_2080_, 3);
v_defEqCtx_x3f_2110_ = lean_ctor_get(v___y_2080_, 4);
v_synthPendingDepth_2111_ = lean_ctor_get(v___y_2080_, 5);
v_customCanUnfoldPredicate_x3f_2112_ = lean_ctor_get(v___y_2080_, 6);
v_univApprox_2113_ = lean_ctor_get_uint8(v___y_2080_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2114_ = lean_ctor_get_uint8(v___y_2080_, sizeof(void*)*7 + 2);
v_cacheInferType_2115_ = lean_ctor_get_uint8(v___y_2080_, sizeof(void*)*7 + 3);
v_isSharedCheck_2217_ = !lean_is_exclusive(v___y_2080_);
if (v_isSharedCheck_2217_ == 0)
{
v___x_2117_ = v___y_2080_;
v_isShared_2118_ = v_isSharedCheck_2217_;
goto v_resetjp_2116_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2112_);
lean_inc(v_synthPendingDepth_2111_);
lean_inc(v_defEqCtx_x3f_2110_);
lean_inc(v_localInstances_2109_);
lean_inc(v_lctx_2108_);
lean_inc(v_zetaDeltaSet_2107_);
lean_inc(v_keyedConfig_2105_);
lean_dec(v___y_2080_);
v___x_2117_ = lean_box(0);
v_isShared_2118_ = v_isSharedCheck_2217_;
goto v_resetjp_2116_;
}
v_resetjp_2116_:
{
lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; uint8_t v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2128_; 
v___x_2119_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__3));
v___x_2120_ = l_Lean_Expr_const___override(v___x_2119_, v___x_2094_);
lean_inc(v_a_2091_);
v___x_2121_ = l_Lean_Expr_app___override(v___x_2120_, v_a_2091_);
lean_inc(v_a_2099_);
v___x_2122_ = l_Lean_Expr_app___override(v___x_2121_, v_a_2099_);
lean_inc(v_a_2102_);
v___x_2123_ = l_Lean_Expr_app___override(v___x_2122_, v_a_2102_);
lean_inc(v_a_2104_);
v___x_2124_ = l_Lean_Expr_app___override(v___x_2123_, v_a_2104_);
v___x_2125_ = 2;
v___x_2126_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2125_, v_keyedConfig_2105_);
if (v_isShared_2118_ == 0)
{
lean_ctor_set(v___x_2117_, 0, v___x_2126_);
v___x_2128_ = v___x_2117_;
goto v_reusejp_2127_;
}
else
{
lean_object* v_reuseFailAlloc_2216_; 
v_reuseFailAlloc_2216_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2216_, 0, v___x_2126_);
lean_ctor_set(v_reuseFailAlloc_2216_, 1, v_zetaDeltaSet_2107_);
lean_ctor_set(v_reuseFailAlloc_2216_, 2, v_lctx_2108_);
lean_ctor_set(v_reuseFailAlloc_2216_, 3, v_localInstances_2109_);
lean_ctor_set(v_reuseFailAlloc_2216_, 4, v_defEqCtx_x3f_2110_);
lean_ctor_set(v_reuseFailAlloc_2216_, 5, v_synthPendingDepth_2111_);
lean_ctor_set(v_reuseFailAlloc_2216_, 6, v_customCanUnfoldPredicate_x3f_2112_);
lean_ctor_set_uint8(v_reuseFailAlloc_2216_, sizeof(void*)*7, v_trackZetaDelta_2106_);
lean_ctor_set_uint8(v_reuseFailAlloc_2216_, sizeof(void*)*7 + 1, v_univApprox_2113_);
lean_ctor_set_uint8(v_reuseFailAlloc_2216_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2114_);
lean_ctor_set_uint8(v_reuseFailAlloc_2216_, sizeof(void*)*7 + 3, v_cacheInferType_2115_);
v___x_2128_ = v_reuseFailAlloc_2216_;
goto v_reusejp_2127_;
}
v_reusejp_2127_:
{
lean_object* v___x_2129_; 
v___x_2129_ = l_Lean_Meta_isExprDefEq(v___x_2124_, v_fst_2077_, v___x_2128_, v___y_2081_, v___y_2082_, v___y_2083_);
lean_dec_ref(v___x_2128_);
if (lean_obj_tag(v___x_2129_) == 0)
{
lean_object* v_a_2130_; lean_object* v___x_2132_; uint8_t v_isShared_2133_; uint8_t v_isSharedCheck_2207_; 
v_a_2130_ = lean_ctor_get(v___x_2129_, 0);
v_isSharedCheck_2207_ = !lean_is_exclusive(v___x_2129_);
if (v_isSharedCheck_2207_ == 0)
{
v___x_2132_ = v___x_2129_;
v_isShared_2133_ = v_isSharedCheck_2207_;
goto v_resetjp_2131_;
}
else
{
lean_inc(v_a_2130_);
lean_dec(v___x_2129_);
v___x_2132_ = lean_box(0);
v_isShared_2133_ = v_isSharedCheck_2207_;
goto v_resetjp_2131_;
}
v_resetjp_2131_:
{
uint8_t v___x_2134_; 
v___x_2134_ = lean_unbox(v_a_2130_);
lean_dec(v_a_2130_);
if (v___x_2134_ == 0)
{
lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2142_; 
v___x_2135_ = lean_box(v___x_2078_);
v___x_2136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2136_, 0, v_a_2104_);
lean_ctor_set(v___x_2136_, 1, v___x_2135_);
v___x_2137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2137_, 0, v_a_2102_);
lean_ctor_set(v___x_2137_, 1, v___x_2136_);
v___x_2138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2138_, 0, v_a_2099_);
lean_ctor_set(v___x_2138_, 1, v___x_2137_);
v___x_2139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2139_, 0, v_a_2091_);
lean_ctor_set(v___x_2139_, 1, v___x_2138_);
v___x_2140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2140_, 0, v_a_2086_);
lean_ctor_set(v___x_2140_, 1, v___x_2139_);
if (v_isShared_2133_ == 0)
{
lean_ctor_set(v___x_2132_, 0, v___x_2140_);
v___x_2142_ = v___x_2132_;
goto v_reusejp_2141_;
}
else
{
lean_object* v_reuseFailAlloc_2143_; 
v_reuseFailAlloc_2143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2143_, 0, v___x_2140_);
v___x_2142_ = v_reuseFailAlloc_2143_;
goto v_reusejp_2141_;
}
v_reusejp_2141_:
{
return v___x_2142_;
}
}
else
{
lean_object* v___x_2144_; 
lean_del_object(v___x_2132_);
v___x_2144_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_a_2086_, v___y_2081_);
if (lean_obj_tag(v___x_2144_) == 0)
{
lean_object* v_a_2145_; lean_object* v___x_2146_; 
v_a_2145_ = lean_ctor_get(v___x_2144_, 0);
lean_inc(v_a_2145_);
lean_dec_ref_known(v___x_2144_, 1);
v___x_2146_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2091_, v___y_2081_);
if (lean_obj_tag(v___x_2146_) == 0)
{
lean_object* v_a_2147_; lean_object* v___x_2148_; 
v_a_2147_ = lean_ctor_get(v___x_2146_, 0);
lean_inc(v_a_2147_);
lean_dec_ref_known(v___x_2146_, 1);
v___x_2148_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2099_, v___y_2081_);
if (lean_obj_tag(v___x_2148_) == 0)
{
lean_object* v_a_2149_; lean_object* v___x_2150_; 
v_a_2149_ = lean_ctor_get(v___x_2148_, 0);
lean_inc(v_a_2149_);
lean_dec_ref_known(v___x_2148_, 1);
v___x_2150_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2102_, v___y_2081_);
if (lean_obj_tag(v___x_2150_) == 0)
{
lean_object* v_a_2151_; lean_object* v___x_2152_; 
v_a_2151_ = lean_ctor_get(v___x_2150_, 0);
lean_inc(v_a_2151_);
lean_dec_ref_known(v___x_2150_, 1);
v___x_2152_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2104_, v___y_2081_);
if (lean_obj_tag(v___x_2152_) == 0)
{
lean_object* v_a_2153_; lean_object* v___x_2155_; uint8_t v_isShared_2156_; uint8_t v_isSharedCheck_2166_; 
v_a_2153_ = lean_ctor_get(v___x_2152_, 0);
v_isSharedCheck_2166_ = !lean_is_exclusive(v___x_2152_);
if (v_isSharedCheck_2166_ == 0)
{
v___x_2155_ = v___x_2152_;
v_isShared_2156_ = v_isSharedCheck_2166_;
goto v_resetjp_2154_;
}
else
{
lean_inc(v_a_2153_);
lean_dec(v___x_2152_);
v___x_2155_ = lean_box(0);
v_isShared_2156_ = v_isSharedCheck_2166_;
goto v_resetjp_2154_;
}
v_resetjp_2154_:
{
lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2164_; 
v___x_2157_ = lean_box(v_a_2079_);
v___x_2158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2158_, 0, v_a_2153_);
lean_ctor_set(v___x_2158_, 1, v___x_2157_);
v___x_2159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2159_, 0, v_a_2151_);
lean_ctor_set(v___x_2159_, 1, v___x_2158_);
v___x_2160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2160_, 0, v_a_2149_);
lean_ctor_set(v___x_2160_, 1, v___x_2159_);
v___x_2161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2161_, 0, v_a_2147_);
lean_ctor_set(v___x_2161_, 1, v___x_2160_);
v___x_2162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2162_, 0, v_a_2145_);
lean_ctor_set(v___x_2162_, 1, v___x_2161_);
if (v_isShared_2156_ == 0)
{
lean_ctor_set(v___x_2155_, 0, v___x_2162_);
v___x_2164_ = v___x_2155_;
goto v_reusejp_2163_;
}
else
{
lean_object* v_reuseFailAlloc_2165_; 
v_reuseFailAlloc_2165_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2165_, 0, v___x_2162_);
v___x_2164_ = v_reuseFailAlloc_2165_;
goto v_reusejp_2163_;
}
v_reusejp_2163_:
{
return v___x_2164_;
}
}
}
else
{
lean_object* v_a_2167_; lean_object* v___x_2169_; uint8_t v_isShared_2170_; uint8_t v_isSharedCheck_2174_; 
lean_dec(v_a_2151_);
lean_dec(v_a_2149_);
lean_dec(v_a_2147_);
lean_dec(v_a_2145_);
v_a_2167_ = lean_ctor_get(v___x_2152_, 0);
v_isSharedCheck_2174_ = !lean_is_exclusive(v___x_2152_);
if (v_isSharedCheck_2174_ == 0)
{
v___x_2169_ = v___x_2152_;
v_isShared_2170_ = v_isSharedCheck_2174_;
goto v_resetjp_2168_;
}
else
{
lean_inc(v_a_2167_);
lean_dec(v___x_2152_);
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
else
{
lean_object* v_a_2175_; lean_object* v___x_2177_; uint8_t v_isShared_2178_; uint8_t v_isSharedCheck_2182_; 
lean_dec(v_a_2149_);
lean_dec(v_a_2147_);
lean_dec(v_a_2145_);
lean_dec(v_a_2104_);
v_a_2175_ = lean_ctor_get(v___x_2150_, 0);
v_isSharedCheck_2182_ = !lean_is_exclusive(v___x_2150_);
if (v_isSharedCheck_2182_ == 0)
{
v___x_2177_ = v___x_2150_;
v_isShared_2178_ = v_isSharedCheck_2182_;
goto v_resetjp_2176_;
}
else
{
lean_inc(v_a_2175_);
lean_dec(v___x_2150_);
v___x_2177_ = lean_box(0);
v_isShared_2178_ = v_isSharedCheck_2182_;
goto v_resetjp_2176_;
}
v_resetjp_2176_:
{
lean_object* v___x_2180_; 
if (v_isShared_2178_ == 0)
{
v___x_2180_ = v___x_2177_;
goto v_reusejp_2179_;
}
else
{
lean_object* v_reuseFailAlloc_2181_; 
v_reuseFailAlloc_2181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2181_, 0, v_a_2175_);
v___x_2180_ = v_reuseFailAlloc_2181_;
goto v_reusejp_2179_;
}
v_reusejp_2179_:
{
return v___x_2180_;
}
}
}
}
else
{
lean_object* v_a_2183_; lean_object* v___x_2185_; uint8_t v_isShared_2186_; uint8_t v_isSharedCheck_2190_; 
lean_dec(v_a_2147_);
lean_dec(v_a_2145_);
lean_dec(v_a_2104_);
lean_dec(v_a_2102_);
v_a_2183_ = lean_ctor_get(v___x_2148_, 0);
v_isSharedCheck_2190_ = !lean_is_exclusive(v___x_2148_);
if (v_isSharedCheck_2190_ == 0)
{
v___x_2185_ = v___x_2148_;
v_isShared_2186_ = v_isSharedCheck_2190_;
goto v_resetjp_2184_;
}
else
{
lean_inc(v_a_2183_);
lean_dec(v___x_2148_);
v___x_2185_ = lean_box(0);
v_isShared_2186_ = v_isSharedCheck_2190_;
goto v_resetjp_2184_;
}
v_resetjp_2184_:
{
lean_object* v___x_2188_; 
if (v_isShared_2186_ == 0)
{
v___x_2188_ = v___x_2185_;
goto v_reusejp_2187_;
}
else
{
lean_object* v_reuseFailAlloc_2189_; 
v_reuseFailAlloc_2189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2189_, 0, v_a_2183_);
v___x_2188_ = v_reuseFailAlloc_2189_;
goto v_reusejp_2187_;
}
v_reusejp_2187_:
{
return v___x_2188_;
}
}
}
}
else
{
lean_object* v_a_2191_; lean_object* v___x_2193_; uint8_t v_isShared_2194_; uint8_t v_isSharedCheck_2198_; 
lean_dec(v_a_2145_);
lean_dec(v_a_2104_);
lean_dec(v_a_2102_);
lean_dec(v_a_2099_);
v_a_2191_ = lean_ctor_get(v___x_2146_, 0);
v_isSharedCheck_2198_ = !lean_is_exclusive(v___x_2146_);
if (v_isSharedCheck_2198_ == 0)
{
v___x_2193_ = v___x_2146_;
v_isShared_2194_ = v_isSharedCheck_2198_;
goto v_resetjp_2192_;
}
else
{
lean_inc(v_a_2191_);
lean_dec(v___x_2146_);
v___x_2193_ = lean_box(0);
v_isShared_2194_ = v_isSharedCheck_2198_;
goto v_resetjp_2192_;
}
v_resetjp_2192_:
{
lean_object* v___x_2196_; 
if (v_isShared_2194_ == 0)
{
v___x_2196_ = v___x_2193_;
goto v_reusejp_2195_;
}
else
{
lean_object* v_reuseFailAlloc_2197_; 
v_reuseFailAlloc_2197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2197_, 0, v_a_2191_);
v___x_2196_ = v_reuseFailAlloc_2197_;
goto v_reusejp_2195_;
}
v_reusejp_2195_:
{
return v___x_2196_;
}
}
}
}
else
{
lean_object* v_a_2199_; lean_object* v___x_2201_; uint8_t v_isShared_2202_; uint8_t v_isSharedCheck_2206_; 
lean_dec(v_a_2104_);
lean_dec(v_a_2102_);
lean_dec(v_a_2099_);
lean_dec(v_a_2091_);
v_a_2199_ = lean_ctor_get(v___x_2144_, 0);
v_isSharedCheck_2206_ = !lean_is_exclusive(v___x_2144_);
if (v_isSharedCheck_2206_ == 0)
{
v___x_2201_ = v___x_2144_;
v_isShared_2202_ = v_isSharedCheck_2206_;
goto v_resetjp_2200_;
}
else
{
lean_inc(v_a_2199_);
lean_dec(v___x_2144_);
v___x_2201_ = lean_box(0);
v_isShared_2202_ = v_isSharedCheck_2206_;
goto v_resetjp_2200_;
}
v_resetjp_2200_:
{
lean_object* v___x_2204_; 
if (v_isShared_2202_ == 0)
{
v___x_2204_ = v___x_2201_;
goto v_reusejp_2203_;
}
else
{
lean_object* v_reuseFailAlloc_2205_; 
v_reuseFailAlloc_2205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2205_, 0, v_a_2199_);
v___x_2204_ = v_reuseFailAlloc_2205_;
goto v_reusejp_2203_;
}
v_reusejp_2203_:
{
return v___x_2204_;
}
}
}
}
}
}
else
{
lean_object* v_a_2208_; lean_object* v___x_2210_; uint8_t v_isShared_2211_; uint8_t v_isSharedCheck_2215_; 
lean_dec(v_a_2104_);
lean_dec(v_a_2102_);
lean_dec(v_a_2099_);
lean_dec(v_a_2091_);
lean_dec(v_a_2086_);
v_a_2208_ = lean_ctor_get(v___x_2129_, 0);
v_isSharedCheck_2215_ = !lean_is_exclusive(v___x_2129_);
if (v_isSharedCheck_2215_ == 0)
{
v___x_2210_ = v___x_2129_;
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
else
{
lean_inc(v_a_2208_);
lean_dec(v___x_2129_);
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
v_reuseFailAlloc_2214_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
else
{
lean_object* v_a_2218_; lean_object* v___x_2220_; uint8_t v_isShared_2221_; uint8_t v_isSharedCheck_2225_; 
lean_dec(v_a_2102_);
lean_dec(v_a_2099_);
lean_dec_ref_known(v___x_2094_, 2);
lean_dec(v_a_2091_);
lean_dec(v_a_2086_);
lean_dec_ref(v___y_2080_);
lean_dec_ref(v_fst_2077_);
v_a_2218_ = lean_ctor_get(v___x_2103_, 0);
v_isSharedCheck_2225_ = !lean_is_exclusive(v___x_2103_);
if (v_isSharedCheck_2225_ == 0)
{
v___x_2220_ = v___x_2103_;
v_isShared_2221_ = v_isSharedCheck_2225_;
goto v_resetjp_2219_;
}
else
{
lean_inc(v_a_2218_);
lean_dec(v___x_2103_);
v___x_2220_ = lean_box(0);
v_isShared_2221_ = v_isSharedCheck_2225_;
goto v_resetjp_2219_;
}
v_resetjp_2219_:
{
lean_object* v___x_2223_; 
if (v_isShared_2221_ == 0)
{
v___x_2223_ = v___x_2220_;
goto v_reusejp_2222_;
}
else
{
lean_object* v_reuseFailAlloc_2224_; 
v_reuseFailAlloc_2224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2224_, 0, v_a_2218_);
v___x_2223_ = v_reuseFailAlloc_2224_;
goto v_reusejp_2222_;
}
v_reusejp_2222_:
{
return v___x_2223_;
}
}
}
}
else
{
lean_object* v_a_2226_; lean_object* v___x_2228_; uint8_t v_isShared_2229_; uint8_t v_isSharedCheck_2233_; 
lean_dec_ref_known(v___x_2100_, 1);
lean_dec(v_a_2099_);
lean_dec_ref_known(v___x_2094_, 2);
lean_dec(v_a_2091_);
lean_dec(v_a_2086_);
lean_dec_ref(v___y_2080_);
lean_dec_ref(v_fst_2077_);
lean_dec(v___x_2076_);
v_a_2226_ = lean_ctor_get(v___x_2101_, 0);
v_isSharedCheck_2233_ = !lean_is_exclusive(v___x_2101_);
if (v_isSharedCheck_2233_ == 0)
{
v___x_2228_ = v___x_2101_;
v_isShared_2229_ = v_isSharedCheck_2233_;
goto v_resetjp_2227_;
}
else
{
lean_inc(v_a_2226_);
lean_dec(v___x_2101_);
v___x_2228_ = lean_box(0);
v_isShared_2229_ = v_isSharedCheck_2233_;
goto v_resetjp_2227_;
}
v_resetjp_2227_:
{
lean_object* v___x_2231_; 
if (v_isShared_2229_ == 0)
{
v___x_2231_ = v___x_2228_;
goto v_reusejp_2230_;
}
else
{
lean_object* v_reuseFailAlloc_2232_; 
v_reuseFailAlloc_2232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2232_, 0, v_a_2226_);
v___x_2231_ = v_reuseFailAlloc_2232_;
goto v_reusejp_2230_;
}
v_reusejp_2230_:
{
return v___x_2231_;
}
}
}
}
else
{
lean_object* v_a_2234_; lean_object* v___x_2236_; uint8_t v_isShared_2237_; uint8_t v_isSharedCheck_2241_; 
lean_dec_ref_known(v___x_2094_, 2);
lean_dec(v_a_2091_);
lean_dec(v_a_2086_);
lean_dec_ref(v___y_2080_);
lean_dec_ref(v_fst_2077_);
lean_dec(v___x_2076_);
v_a_2234_ = lean_ctor_get(v___x_2098_, 0);
v_isSharedCheck_2241_ = !lean_is_exclusive(v___x_2098_);
if (v_isSharedCheck_2241_ == 0)
{
v___x_2236_ = v___x_2098_;
v_isShared_2237_ = v_isSharedCheck_2241_;
goto v_resetjp_2235_;
}
else
{
lean_inc(v_a_2234_);
lean_dec(v___x_2098_);
v___x_2236_ = lean_box(0);
v_isShared_2237_ = v_isSharedCheck_2241_;
goto v_resetjp_2235_;
}
v_resetjp_2235_:
{
lean_object* v___x_2239_; 
if (v_isShared_2237_ == 0)
{
v___x_2239_ = v___x_2236_;
goto v_reusejp_2238_;
}
else
{
lean_object* v_reuseFailAlloc_2240_; 
v_reuseFailAlloc_2240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2240_, 0, v_a_2234_);
v___x_2239_ = v_reuseFailAlloc_2240_;
goto v_reusejp_2238_;
}
v_reusejp_2238_:
{
return v___x_2239_;
}
}
}
}
else
{
lean_object* v_a_2242_; lean_object* v___x_2244_; uint8_t v_isShared_2245_; uint8_t v_isSharedCheck_2249_; 
lean_dec(v_a_2086_);
lean_dec_ref(v___y_2080_);
lean_dec_ref(v_fst_2077_);
lean_dec(v___x_2076_);
v_a_2242_ = lean_ctor_get(v___x_2090_, 0);
v_isSharedCheck_2249_ = !lean_is_exclusive(v___x_2090_);
if (v_isSharedCheck_2249_ == 0)
{
v___x_2244_ = v___x_2090_;
v_isShared_2245_ = v_isSharedCheck_2249_;
goto v_resetjp_2243_;
}
else
{
lean_inc(v_a_2242_);
lean_dec(v___x_2090_);
v___x_2244_ = lean_box(0);
v_isShared_2245_ = v_isSharedCheck_2249_;
goto v_resetjp_2243_;
}
v_resetjp_2243_:
{
lean_object* v___x_2247_; 
if (v_isShared_2245_ == 0)
{
v___x_2247_ = v___x_2244_;
goto v_reusejp_2246_;
}
else
{
lean_object* v_reuseFailAlloc_2248_; 
v_reuseFailAlloc_2248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2248_, 0, v_a_2242_);
v___x_2247_ = v_reuseFailAlloc_2248_;
goto v_reusejp_2246_;
}
v_reusejp_2246_:
{
return v___x_2247_;
}
}
}
}
else
{
lean_object* v_a_2250_; lean_object* v___x_2252_; uint8_t v_isShared_2253_; uint8_t v_isSharedCheck_2257_; 
lean_dec_ref(v___y_2080_);
lean_dec_ref(v_fst_2077_);
lean_dec(v___x_2076_);
v_a_2250_ = lean_ctor_get(v___x_2085_, 0);
v_isSharedCheck_2257_ = !lean_is_exclusive(v___x_2085_);
if (v_isSharedCheck_2257_ == 0)
{
v___x_2252_ = v___x_2085_;
v_isShared_2253_ = v_isSharedCheck_2257_;
goto v_resetjp_2251_;
}
else
{
lean_inc(v_a_2250_);
lean_dec(v___x_2085_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___boxed(lean_object* v___x_2258_, lean_object* v___x_2259_, lean_object* v_fst_2260_, lean_object* v___x_2261_, lean_object* v_a_2262_, lean_object* v___y_2263_, lean_object* v___y_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_, lean_object* v___y_2267_){
_start:
{
uint8_t v___x_190883__boxed_2268_; uint8_t v___x_190886__boxed_2269_; uint8_t v_a_190887__boxed_2270_; lean_object* v_res_2271_; 
v___x_190883__boxed_2268_ = lean_unbox(v___x_2258_);
v___x_190886__boxed_2269_ = lean_unbox(v___x_2261_);
v_a_190887__boxed_2270_ = lean_unbox(v_a_2262_);
v_res_2271_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2(v___x_190883__boxed_2268_, v___x_2259_, v_fst_2260_, v___x_190886__boxed_2269_, v_a_190887__boxed_2270_, v___y_2263_, v___y_2264_, v___y_2265_, v___y_2266_);
lean_dec(v___y_2266_);
lean_dec_ref(v___y_2265_);
lean_dec(v___y_2264_);
return v_res_2271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3(uint8_t v___x_2275_, lean_object* v___x_2276_, lean_object* v_fst_2277_, uint8_t v___x_2278_, uint8_t v_a_2279_, lean_object* v___y_2280_, lean_object* v___y_2281_, lean_object* v___y_2282_, lean_object* v___y_2283_){
_start:
{
lean_object* v___x_2285_; 
v___x_2285_ = l_Lean_Meta_mkFreshLevelMVar(v___y_2280_, v___y_2281_, v___y_2282_, v___y_2283_);
if (lean_obj_tag(v___x_2285_) == 0)
{
lean_object* v_a_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; lean_object* v___x_2289_; lean_object* v___x_2290_; 
v_a_2286_ = lean_ctor_get(v___x_2285_, 0);
lean_inc_n(v_a_2286_, 2);
lean_dec_ref_known(v___x_2285_, 1);
v___x_2287_ = l_Lean_Level_succ___override(v_a_2286_);
lean_inc(v___x_2287_);
v___x_2288_ = l_Lean_Expr_sort___override(v___x_2287_);
v___x_2289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2289_, 0, v___x_2288_);
lean_inc(v___x_2276_);
v___x_2290_ = l_Lean_Meta_mkFreshExprMVar(v___x_2289_, v___x_2275_, v___x_2276_, v___y_2280_, v___y_2281_, v___y_2282_, v___y_2283_);
if (lean_obj_tag(v___x_2290_) == 0)
{
lean_object* v_a_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; 
v_a_2291_ = lean_ctor_get(v___x_2290_, 0);
lean_inc_n(v_a_2291_, 2);
lean_dec_ref_known(v___x_2290_, 1);
v___x_2292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2292_, 0, v_a_2291_);
lean_inc(v___x_2276_);
lean_inc_ref(v___x_2292_);
v___x_2293_ = l_Lean_Meta_mkFreshExprMVar(v___x_2292_, v___x_2275_, v___x_2276_, v___y_2280_, v___y_2281_, v___y_2282_, v___y_2283_);
if (lean_obj_tag(v___x_2293_) == 0)
{
lean_object* v_a_2294_; lean_object* v___x_2295_; 
v_a_2294_ = lean_ctor_get(v___x_2293_, 0);
lean_inc(v_a_2294_);
lean_dec_ref_known(v___x_2293_, 1);
v___x_2295_ = l_Lean_Meta_mkFreshExprMVar(v___x_2292_, v___x_2275_, v___x_2276_, v___y_2280_, v___y_2281_, v___y_2282_, v___y_2283_);
if (lean_obj_tag(v___x_2295_) == 0)
{
lean_object* v_a_2296_; lean_object* v_keyedConfig_2297_; uint8_t v_trackZetaDelta_2298_; lean_object* v_zetaDeltaSet_2299_; lean_object* v_lctx_2300_; lean_object* v_localInstances_2301_; lean_object* v_defEqCtx_x3f_2302_; lean_object* v_synthPendingDepth_2303_; lean_object* v_customCanUnfoldPredicate_x3f_2304_; uint8_t v_univApprox_2305_; uint8_t v_inTypeClassResolution_2306_; uint8_t v_cacheInferType_2307_; lean_object* v___x_2309_; uint8_t v_isShared_2310_; uint8_t v_isSharedCheck_2398_; 
v_a_2296_ = lean_ctor_get(v___x_2295_, 0);
lean_inc(v_a_2296_);
lean_dec_ref_known(v___x_2295_, 1);
v_keyedConfig_2297_ = lean_ctor_get(v___y_2280_, 0);
v_trackZetaDelta_2298_ = lean_ctor_get_uint8(v___y_2280_, sizeof(void*)*7);
v_zetaDeltaSet_2299_ = lean_ctor_get(v___y_2280_, 1);
v_lctx_2300_ = lean_ctor_get(v___y_2280_, 2);
v_localInstances_2301_ = lean_ctor_get(v___y_2280_, 3);
v_defEqCtx_x3f_2302_ = lean_ctor_get(v___y_2280_, 4);
v_synthPendingDepth_2303_ = lean_ctor_get(v___y_2280_, 5);
v_customCanUnfoldPredicate_x3f_2304_ = lean_ctor_get(v___y_2280_, 6);
v_univApprox_2305_ = lean_ctor_get_uint8(v___y_2280_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2306_ = lean_ctor_get_uint8(v___y_2280_, sizeof(void*)*7 + 2);
v_cacheInferType_2307_ = lean_ctor_get_uint8(v___y_2280_, sizeof(void*)*7 + 3);
v_isSharedCheck_2398_ = !lean_is_exclusive(v___y_2280_);
if (v_isSharedCheck_2398_ == 0)
{
v___x_2309_ = v___y_2280_;
v_isShared_2310_ = v_isSharedCheck_2398_;
goto v_resetjp_2308_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2304_);
lean_inc(v_synthPendingDepth_2303_);
lean_inc(v_defEqCtx_x3f_2302_);
lean_inc(v_localInstances_2301_);
lean_inc(v_lctx_2300_);
lean_inc(v_zetaDeltaSet_2299_);
lean_inc(v_keyedConfig_2297_);
lean_dec(v___y_2280_);
v___x_2309_ = lean_box(0);
v_isShared_2310_ = v_isSharedCheck_2398_;
goto v_resetjp_2308_;
}
v_resetjp_2308_:
{
lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; uint8_t v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2321_; 
v___x_2311_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___closed__1));
v___x_2312_ = lean_box(0);
v___x_2313_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2313_, 0, v___x_2287_);
lean_ctor_set(v___x_2313_, 1, v___x_2312_);
v___x_2314_ = l_Lean_Expr_const___override(v___x_2311_, v___x_2313_);
lean_inc(v_a_2291_);
v___x_2315_ = l_Lean_Expr_app___override(v___x_2314_, v_a_2291_);
lean_inc(v_a_2294_);
v___x_2316_ = l_Lean_Expr_app___override(v___x_2315_, v_a_2294_);
lean_inc(v_a_2296_);
v___x_2317_ = l_Lean_Expr_app___override(v___x_2316_, v_a_2296_);
v___x_2318_ = 2;
v___x_2319_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2318_, v_keyedConfig_2297_);
if (v_isShared_2310_ == 0)
{
lean_ctor_set(v___x_2309_, 0, v___x_2319_);
v___x_2321_ = v___x_2309_;
goto v_reusejp_2320_;
}
else
{
lean_object* v_reuseFailAlloc_2397_; 
v_reuseFailAlloc_2397_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2397_, 0, v___x_2319_);
lean_ctor_set(v_reuseFailAlloc_2397_, 1, v_zetaDeltaSet_2299_);
lean_ctor_set(v_reuseFailAlloc_2397_, 2, v_lctx_2300_);
lean_ctor_set(v_reuseFailAlloc_2397_, 3, v_localInstances_2301_);
lean_ctor_set(v_reuseFailAlloc_2397_, 4, v_defEqCtx_x3f_2302_);
lean_ctor_set(v_reuseFailAlloc_2397_, 5, v_synthPendingDepth_2303_);
lean_ctor_set(v_reuseFailAlloc_2397_, 6, v_customCanUnfoldPredicate_x3f_2304_);
lean_ctor_set_uint8(v_reuseFailAlloc_2397_, sizeof(void*)*7, v_trackZetaDelta_2298_);
lean_ctor_set_uint8(v_reuseFailAlloc_2397_, sizeof(void*)*7 + 1, v_univApprox_2305_);
lean_ctor_set_uint8(v_reuseFailAlloc_2397_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2306_);
lean_ctor_set_uint8(v_reuseFailAlloc_2397_, sizeof(void*)*7 + 3, v_cacheInferType_2307_);
v___x_2321_ = v_reuseFailAlloc_2397_;
goto v_reusejp_2320_;
}
v_reusejp_2320_:
{
lean_object* v___x_2322_; 
v___x_2322_ = l_Lean_Meta_isExprDefEq(v___x_2317_, v_fst_2277_, v___x_2321_, v___y_2281_, v___y_2282_, v___y_2283_);
lean_dec_ref(v___x_2321_);
if (lean_obj_tag(v___x_2322_) == 0)
{
lean_object* v_a_2323_; lean_object* v___x_2325_; uint8_t v_isShared_2326_; uint8_t v_isSharedCheck_2388_; 
v_a_2323_ = lean_ctor_get(v___x_2322_, 0);
v_isSharedCheck_2388_ = !lean_is_exclusive(v___x_2322_);
if (v_isSharedCheck_2388_ == 0)
{
v___x_2325_ = v___x_2322_;
v_isShared_2326_ = v_isSharedCheck_2388_;
goto v_resetjp_2324_;
}
else
{
lean_inc(v_a_2323_);
lean_dec(v___x_2322_);
v___x_2325_ = lean_box(0);
v_isShared_2326_ = v_isSharedCheck_2388_;
goto v_resetjp_2324_;
}
v_resetjp_2324_:
{
uint8_t v___x_2327_; 
v___x_2327_ = lean_unbox(v_a_2323_);
lean_dec(v_a_2323_);
if (v___x_2327_ == 0)
{
lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2334_; 
v___x_2328_ = lean_box(v___x_2278_);
v___x_2329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2329_, 0, v_a_2296_);
lean_ctor_set(v___x_2329_, 1, v___x_2328_);
v___x_2330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2330_, 0, v_a_2294_);
lean_ctor_set(v___x_2330_, 1, v___x_2329_);
v___x_2331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2331_, 0, v_a_2291_);
lean_ctor_set(v___x_2331_, 1, v___x_2330_);
v___x_2332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2332_, 0, v_a_2286_);
lean_ctor_set(v___x_2332_, 1, v___x_2331_);
if (v_isShared_2326_ == 0)
{
lean_ctor_set(v___x_2325_, 0, v___x_2332_);
v___x_2334_ = v___x_2325_;
goto v_reusejp_2333_;
}
else
{
lean_object* v_reuseFailAlloc_2335_; 
v_reuseFailAlloc_2335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2335_, 0, v___x_2332_);
v___x_2334_ = v_reuseFailAlloc_2335_;
goto v_reusejp_2333_;
}
v_reusejp_2333_:
{
return v___x_2334_;
}
}
else
{
lean_object* v___x_2336_; 
lean_del_object(v___x_2325_);
v___x_2336_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_a_2286_, v___y_2281_);
if (lean_obj_tag(v___x_2336_) == 0)
{
lean_object* v_a_2337_; lean_object* v___x_2338_; 
v_a_2337_ = lean_ctor_get(v___x_2336_, 0);
lean_inc(v_a_2337_);
lean_dec_ref_known(v___x_2336_, 1);
v___x_2338_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2291_, v___y_2281_);
if (lean_obj_tag(v___x_2338_) == 0)
{
lean_object* v_a_2339_; lean_object* v___x_2340_; 
v_a_2339_ = lean_ctor_get(v___x_2338_, 0);
lean_inc(v_a_2339_);
lean_dec_ref_known(v___x_2338_, 1);
v___x_2340_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2294_, v___y_2281_);
if (lean_obj_tag(v___x_2340_) == 0)
{
lean_object* v_a_2341_; lean_object* v___x_2342_; 
v_a_2341_ = lean_ctor_get(v___x_2340_, 0);
lean_inc(v_a_2341_);
lean_dec_ref_known(v___x_2340_, 1);
v___x_2342_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2296_, v___y_2281_);
if (lean_obj_tag(v___x_2342_) == 0)
{
lean_object* v_a_2343_; lean_object* v___x_2345_; uint8_t v_isShared_2346_; uint8_t v_isSharedCheck_2355_; 
v_a_2343_ = lean_ctor_get(v___x_2342_, 0);
v_isSharedCheck_2355_ = !lean_is_exclusive(v___x_2342_);
if (v_isSharedCheck_2355_ == 0)
{
v___x_2345_ = v___x_2342_;
v_isShared_2346_ = v_isSharedCheck_2355_;
goto v_resetjp_2344_;
}
else
{
lean_inc(v_a_2343_);
lean_dec(v___x_2342_);
v___x_2345_ = lean_box(0);
v_isShared_2346_ = v_isSharedCheck_2355_;
goto v_resetjp_2344_;
}
v_resetjp_2344_:
{
lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2353_; 
v___x_2347_ = lean_box(v_a_2279_);
v___x_2348_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2348_, 0, v_a_2343_);
lean_ctor_set(v___x_2348_, 1, v___x_2347_);
v___x_2349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2349_, 0, v_a_2341_);
lean_ctor_set(v___x_2349_, 1, v___x_2348_);
v___x_2350_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2350_, 0, v_a_2339_);
lean_ctor_set(v___x_2350_, 1, v___x_2349_);
v___x_2351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2351_, 0, v_a_2337_);
lean_ctor_set(v___x_2351_, 1, v___x_2350_);
if (v_isShared_2346_ == 0)
{
lean_ctor_set(v___x_2345_, 0, v___x_2351_);
v___x_2353_ = v___x_2345_;
goto v_reusejp_2352_;
}
else
{
lean_object* v_reuseFailAlloc_2354_; 
v_reuseFailAlloc_2354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2354_, 0, v___x_2351_);
v___x_2353_ = v_reuseFailAlloc_2354_;
goto v_reusejp_2352_;
}
v_reusejp_2352_:
{
return v___x_2353_;
}
}
}
else
{
lean_object* v_a_2356_; lean_object* v___x_2358_; uint8_t v_isShared_2359_; uint8_t v_isSharedCheck_2363_; 
lean_dec(v_a_2341_);
lean_dec(v_a_2339_);
lean_dec(v_a_2337_);
v_a_2356_ = lean_ctor_get(v___x_2342_, 0);
v_isSharedCheck_2363_ = !lean_is_exclusive(v___x_2342_);
if (v_isSharedCheck_2363_ == 0)
{
v___x_2358_ = v___x_2342_;
v_isShared_2359_ = v_isSharedCheck_2363_;
goto v_resetjp_2357_;
}
else
{
lean_inc(v_a_2356_);
lean_dec(v___x_2342_);
v___x_2358_ = lean_box(0);
v_isShared_2359_ = v_isSharedCheck_2363_;
goto v_resetjp_2357_;
}
v_resetjp_2357_:
{
lean_object* v___x_2361_; 
if (v_isShared_2359_ == 0)
{
v___x_2361_ = v___x_2358_;
goto v_reusejp_2360_;
}
else
{
lean_object* v_reuseFailAlloc_2362_; 
v_reuseFailAlloc_2362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2362_, 0, v_a_2356_);
v___x_2361_ = v_reuseFailAlloc_2362_;
goto v_reusejp_2360_;
}
v_reusejp_2360_:
{
return v___x_2361_;
}
}
}
}
else
{
lean_object* v_a_2364_; lean_object* v___x_2366_; uint8_t v_isShared_2367_; uint8_t v_isSharedCheck_2371_; 
lean_dec(v_a_2339_);
lean_dec(v_a_2337_);
lean_dec(v_a_2296_);
v_a_2364_ = lean_ctor_get(v___x_2340_, 0);
v_isSharedCheck_2371_ = !lean_is_exclusive(v___x_2340_);
if (v_isSharedCheck_2371_ == 0)
{
v___x_2366_ = v___x_2340_;
v_isShared_2367_ = v_isSharedCheck_2371_;
goto v_resetjp_2365_;
}
else
{
lean_inc(v_a_2364_);
lean_dec(v___x_2340_);
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
lean_dec(v_a_2337_);
lean_dec(v_a_2296_);
lean_dec(v_a_2294_);
v_a_2372_ = lean_ctor_get(v___x_2338_, 0);
v_isSharedCheck_2379_ = !lean_is_exclusive(v___x_2338_);
if (v_isSharedCheck_2379_ == 0)
{
v___x_2374_ = v___x_2338_;
v_isShared_2375_ = v_isSharedCheck_2379_;
goto v_resetjp_2373_;
}
else
{
lean_inc(v_a_2372_);
lean_dec(v___x_2338_);
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
else
{
lean_object* v_a_2380_; lean_object* v___x_2382_; uint8_t v_isShared_2383_; uint8_t v_isSharedCheck_2387_; 
lean_dec(v_a_2296_);
lean_dec(v_a_2294_);
lean_dec(v_a_2291_);
v_a_2380_ = lean_ctor_get(v___x_2336_, 0);
v_isSharedCheck_2387_ = !lean_is_exclusive(v___x_2336_);
if (v_isSharedCheck_2387_ == 0)
{
v___x_2382_ = v___x_2336_;
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
else
{
lean_inc(v_a_2380_);
lean_dec(v___x_2336_);
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
}
}
else
{
lean_object* v_a_2389_; lean_object* v___x_2391_; uint8_t v_isShared_2392_; uint8_t v_isSharedCheck_2396_; 
lean_dec(v_a_2296_);
lean_dec(v_a_2294_);
lean_dec(v_a_2291_);
lean_dec(v_a_2286_);
v_a_2389_ = lean_ctor_get(v___x_2322_, 0);
v_isSharedCheck_2396_ = !lean_is_exclusive(v___x_2322_);
if (v_isSharedCheck_2396_ == 0)
{
v___x_2391_ = v___x_2322_;
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
else
{
lean_inc(v_a_2389_);
lean_dec(v___x_2322_);
v___x_2391_ = lean_box(0);
v_isShared_2392_ = v_isSharedCheck_2396_;
goto v_resetjp_2390_;
}
v_resetjp_2390_:
{
lean_object* v___x_2394_; 
if (v_isShared_2392_ == 0)
{
v___x_2394_ = v___x_2391_;
goto v_reusejp_2393_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v_a_2389_);
v___x_2394_ = v_reuseFailAlloc_2395_;
goto v_reusejp_2393_;
}
v_reusejp_2393_:
{
return v___x_2394_;
}
}
}
}
}
}
else
{
lean_object* v_a_2399_; lean_object* v___x_2401_; uint8_t v_isShared_2402_; uint8_t v_isSharedCheck_2406_; 
lean_dec(v_a_2294_);
lean_dec(v_a_2291_);
lean_dec(v___x_2287_);
lean_dec(v_a_2286_);
lean_dec_ref(v___y_2280_);
lean_dec_ref(v_fst_2277_);
v_a_2399_ = lean_ctor_get(v___x_2295_, 0);
v_isSharedCheck_2406_ = !lean_is_exclusive(v___x_2295_);
if (v_isSharedCheck_2406_ == 0)
{
v___x_2401_ = v___x_2295_;
v_isShared_2402_ = v_isSharedCheck_2406_;
goto v_resetjp_2400_;
}
else
{
lean_inc(v_a_2399_);
lean_dec(v___x_2295_);
v___x_2401_ = lean_box(0);
v_isShared_2402_ = v_isSharedCheck_2406_;
goto v_resetjp_2400_;
}
v_resetjp_2400_:
{
lean_object* v___x_2404_; 
if (v_isShared_2402_ == 0)
{
v___x_2404_ = v___x_2401_;
goto v_reusejp_2403_;
}
else
{
lean_object* v_reuseFailAlloc_2405_; 
v_reuseFailAlloc_2405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2405_, 0, v_a_2399_);
v___x_2404_ = v_reuseFailAlloc_2405_;
goto v_reusejp_2403_;
}
v_reusejp_2403_:
{
return v___x_2404_;
}
}
}
}
else
{
lean_object* v_a_2407_; lean_object* v___x_2409_; uint8_t v_isShared_2410_; uint8_t v_isSharedCheck_2414_; 
lean_dec_ref_known(v___x_2292_, 1);
lean_dec(v_a_2291_);
lean_dec(v___x_2287_);
lean_dec(v_a_2286_);
lean_dec_ref(v___y_2280_);
lean_dec_ref(v_fst_2277_);
lean_dec(v___x_2276_);
v_a_2407_ = lean_ctor_get(v___x_2293_, 0);
v_isSharedCheck_2414_ = !lean_is_exclusive(v___x_2293_);
if (v_isSharedCheck_2414_ == 0)
{
v___x_2409_ = v___x_2293_;
v_isShared_2410_ = v_isSharedCheck_2414_;
goto v_resetjp_2408_;
}
else
{
lean_inc(v_a_2407_);
lean_dec(v___x_2293_);
v___x_2409_ = lean_box(0);
v_isShared_2410_ = v_isSharedCheck_2414_;
goto v_resetjp_2408_;
}
v_resetjp_2408_:
{
lean_object* v___x_2412_; 
if (v_isShared_2410_ == 0)
{
v___x_2412_ = v___x_2409_;
goto v_reusejp_2411_;
}
else
{
lean_object* v_reuseFailAlloc_2413_; 
v_reuseFailAlloc_2413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2413_, 0, v_a_2407_);
v___x_2412_ = v_reuseFailAlloc_2413_;
goto v_reusejp_2411_;
}
v_reusejp_2411_:
{
return v___x_2412_;
}
}
}
}
else
{
lean_object* v_a_2415_; lean_object* v___x_2417_; uint8_t v_isShared_2418_; uint8_t v_isSharedCheck_2422_; 
lean_dec(v___x_2287_);
lean_dec(v_a_2286_);
lean_dec_ref(v___y_2280_);
lean_dec_ref(v_fst_2277_);
lean_dec(v___x_2276_);
v_a_2415_ = lean_ctor_get(v___x_2290_, 0);
v_isSharedCheck_2422_ = !lean_is_exclusive(v___x_2290_);
if (v_isSharedCheck_2422_ == 0)
{
v___x_2417_ = v___x_2290_;
v_isShared_2418_ = v_isSharedCheck_2422_;
goto v_resetjp_2416_;
}
else
{
lean_inc(v_a_2415_);
lean_dec(v___x_2290_);
v___x_2417_ = lean_box(0);
v_isShared_2418_ = v_isSharedCheck_2422_;
goto v_resetjp_2416_;
}
v_resetjp_2416_:
{
lean_object* v___x_2420_; 
if (v_isShared_2418_ == 0)
{
v___x_2420_ = v___x_2417_;
goto v_reusejp_2419_;
}
else
{
lean_object* v_reuseFailAlloc_2421_; 
v_reuseFailAlloc_2421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2421_, 0, v_a_2415_);
v___x_2420_ = v_reuseFailAlloc_2421_;
goto v_reusejp_2419_;
}
v_reusejp_2419_:
{
return v___x_2420_;
}
}
}
}
else
{
lean_object* v_a_2423_; lean_object* v___x_2425_; uint8_t v_isShared_2426_; uint8_t v_isSharedCheck_2430_; 
lean_dec_ref(v___y_2280_);
lean_dec_ref(v_fst_2277_);
lean_dec(v___x_2276_);
v_a_2423_ = lean_ctor_get(v___x_2285_, 0);
v_isSharedCheck_2430_ = !lean_is_exclusive(v___x_2285_);
if (v_isSharedCheck_2430_ == 0)
{
v___x_2425_ = v___x_2285_;
v_isShared_2426_ = v_isSharedCheck_2430_;
goto v_resetjp_2424_;
}
else
{
lean_inc(v_a_2423_);
lean_dec(v___x_2285_);
v___x_2425_ = lean_box(0);
v_isShared_2426_ = v_isSharedCheck_2430_;
goto v_resetjp_2424_;
}
v_resetjp_2424_:
{
lean_object* v___x_2428_; 
if (v_isShared_2426_ == 0)
{
v___x_2428_ = v___x_2425_;
goto v_reusejp_2427_;
}
else
{
lean_object* v_reuseFailAlloc_2429_; 
v_reuseFailAlloc_2429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2429_, 0, v_a_2423_);
v___x_2428_ = v_reuseFailAlloc_2429_;
goto v_reusejp_2427_;
}
v_reusejp_2427_:
{
return v___x_2428_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___boxed(lean_object* v___x_2431_, lean_object* v___x_2432_, lean_object* v_fst_2433_, lean_object* v___x_2434_, lean_object* v_a_2435_, lean_object* v___y_2436_, lean_object* v___y_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_){
_start:
{
uint8_t v___x_191250__boxed_2441_; uint8_t v___x_191253__boxed_2442_; uint8_t v_a_191254__boxed_2443_; lean_object* v_res_2444_; 
v___x_191250__boxed_2441_ = lean_unbox(v___x_2431_);
v___x_191253__boxed_2442_ = lean_unbox(v___x_2434_);
v_a_191254__boxed_2443_ = lean_unbox(v_a_2435_);
v_res_2444_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3(v___x_191250__boxed_2441_, v___x_2432_, v_fst_2433_, v___x_191253__boxed_2442_, v_a_191254__boxed_2443_, v___y_2436_, v___y_2437_, v___y_2438_, v___y_2439_);
lean_dec(v___y_2439_);
lean_dec_ref(v___y_2438_);
lean_dec(v___y_2437_);
return v_res_2444_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__2(void){
_start:
{
lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; 
v___x_2448_ = lean_box(0);
v___x_2449_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__1));
v___x_2450_ = l_Lean_Expr_const___override(v___x_2449_, v___x_2448_);
return v___x_2450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4(lean_object* v___x_2451_, uint8_t v___x_2452_, lean_object* v___x_2453_, lean_object* v_fst_2454_, uint8_t v___x_2455_, uint8_t v_a_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_){
_start:
{
lean_object* v___x_2462_; 
v___x_2462_ = l_Lean_Meta_mkFreshExprMVar(v___x_2451_, v___x_2452_, v___x_2453_, v___y_2457_, v___y_2458_, v___y_2459_, v___y_2460_);
if (lean_obj_tag(v___x_2462_) == 0)
{
lean_object* v_a_2463_; lean_object* v_keyedConfig_2464_; uint8_t v_trackZetaDelta_2465_; lean_object* v_zetaDeltaSet_2466_; lean_object* v_lctx_2467_; lean_object* v_localInstances_2468_; lean_object* v_defEqCtx_x3f_2469_; lean_object* v_synthPendingDepth_2470_; lean_object* v_customCanUnfoldPredicate_x3f_2471_; uint8_t v_univApprox_2472_; uint8_t v_inTypeClassResolution_2473_; uint8_t v_cacheInferType_2474_; lean_object* v___x_2476_; uint8_t v_isShared_2477_; uint8_t v_isSharedCheck_2524_; 
v_a_2463_ = lean_ctor_get(v___x_2462_, 0);
lean_inc(v_a_2463_);
lean_dec_ref_known(v___x_2462_, 1);
v_keyedConfig_2464_ = lean_ctor_get(v___y_2457_, 0);
v_trackZetaDelta_2465_ = lean_ctor_get_uint8(v___y_2457_, sizeof(void*)*7);
v_zetaDeltaSet_2466_ = lean_ctor_get(v___y_2457_, 1);
v_lctx_2467_ = lean_ctor_get(v___y_2457_, 2);
v_localInstances_2468_ = lean_ctor_get(v___y_2457_, 3);
v_defEqCtx_x3f_2469_ = lean_ctor_get(v___y_2457_, 4);
v_synthPendingDepth_2470_ = lean_ctor_get(v___y_2457_, 5);
v_customCanUnfoldPredicate_x3f_2471_ = lean_ctor_get(v___y_2457_, 6);
v_univApprox_2472_ = lean_ctor_get_uint8(v___y_2457_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2473_ = lean_ctor_get_uint8(v___y_2457_, sizeof(void*)*7 + 2);
v_cacheInferType_2474_ = lean_ctor_get_uint8(v___y_2457_, sizeof(void*)*7 + 3);
v_isSharedCheck_2524_ = !lean_is_exclusive(v___y_2457_);
if (v_isSharedCheck_2524_ == 0)
{
v___x_2476_ = v___y_2457_;
v_isShared_2477_ = v_isSharedCheck_2524_;
goto v_resetjp_2475_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2471_);
lean_inc(v_synthPendingDepth_2470_);
lean_inc(v_defEqCtx_x3f_2469_);
lean_inc(v_localInstances_2468_);
lean_inc(v_lctx_2467_);
lean_inc(v_zetaDeltaSet_2466_);
lean_inc(v_keyedConfig_2464_);
lean_dec(v___y_2457_);
v___x_2476_ = lean_box(0);
v_isShared_2477_ = v_isSharedCheck_2524_;
goto v_resetjp_2475_;
}
v_resetjp_2475_:
{
lean_object* v___x_2478_; lean_object* v___x_2479_; uint8_t v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2483_; 
v___x_2478_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__2, &lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___closed__2);
lean_inc(v_a_2463_);
v___x_2479_ = l_Lean_Expr_app___override(v___x_2478_, v_a_2463_);
v___x_2480_ = 2;
v___x_2481_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2480_, v_keyedConfig_2464_);
if (v_isShared_2477_ == 0)
{
lean_ctor_set(v___x_2476_, 0, v___x_2481_);
v___x_2483_ = v___x_2476_;
goto v_reusejp_2482_;
}
else
{
lean_object* v_reuseFailAlloc_2523_; 
v_reuseFailAlloc_2523_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2523_, 0, v___x_2481_);
lean_ctor_set(v_reuseFailAlloc_2523_, 1, v_zetaDeltaSet_2466_);
lean_ctor_set(v_reuseFailAlloc_2523_, 2, v_lctx_2467_);
lean_ctor_set(v_reuseFailAlloc_2523_, 3, v_localInstances_2468_);
lean_ctor_set(v_reuseFailAlloc_2523_, 4, v_defEqCtx_x3f_2469_);
lean_ctor_set(v_reuseFailAlloc_2523_, 5, v_synthPendingDepth_2470_);
lean_ctor_set(v_reuseFailAlloc_2523_, 6, v_customCanUnfoldPredicate_x3f_2471_);
lean_ctor_set_uint8(v_reuseFailAlloc_2523_, sizeof(void*)*7, v_trackZetaDelta_2465_);
lean_ctor_set_uint8(v_reuseFailAlloc_2523_, sizeof(void*)*7 + 1, v_univApprox_2472_);
lean_ctor_set_uint8(v_reuseFailAlloc_2523_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2473_);
lean_ctor_set_uint8(v_reuseFailAlloc_2523_, sizeof(void*)*7 + 3, v_cacheInferType_2474_);
v___x_2483_ = v_reuseFailAlloc_2523_;
goto v_reusejp_2482_;
}
v_reusejp_2482_:
{
lean_object* v___x_2484_; 
v___x_2484_ = l_Lean_Meta_isExprDefEq(v___x_2479_, v_fst_2454_, v___x_2483_, v___y_2458_, v___y_2459_, v___y_2460_);
lean_dec_ref(v___x_2483_);
if (lean_obj_tag(v___x_2484_) == 0)
{
lean_object* v_a_2485_; lean_object* v___x_2487_; uint8_t v_isShared_2488_; uint8_t v_isSharedCheck_2514_; 
v_a_2485_ = lean_ctor_get(v___x_2484_, 0);
v_isSharedCheck_2514_ = !lean_is_exclusive(v___x_2484_);
if (v_isSharedCheck_2514_ == 0)
{
v___x_2487_ = v___x_2484_;
v_isShared_2488_ = v_isSharedCheck_2514_;
goto v_resetjp_2486_;
}
else
{
lean_inc(v_a_2485_);
lean_dec(v___x_2484_);
v___x_2487_ = lean_box(0);
v_isShared_2488_ = v_isSharedCheck_2514_;
goto v_resetjp_2486_;
}
v_resetjp_2486_:
{
uint8_t v___x_2489_; 
v___x_2489_ = lean_unbox(v_a_2485_);
lean_dec(v_a_2485_);
if (v___x_2489_ == 0)
{
lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2493_; 
v___x_2490_ = lean_box(v___x_2455_);
v___x_2491_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2491_, 0, v_a_2463_);
lean_ctor_set(v___x_2491_, 1, v___x_2490_);
if (v_isShared_2488_ == 0)
{
lean_ctor_set(v___x_2487_, 0, v___x_2491_);
v___x_2493_ = v___x_2487_;
goto v_reusejp_2492_;
}
else
{
lean_object* v_reuseFailAlloc_2494_; 
v_reuseFailAlloc_2494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2494_, 0, v___x_2491_);
v___x_2493_ = v_reuseFailAlloc_2494_;
goto v_reusejp_2492_;
}
v_reusejp_2492_:
{
return v___x_2493_;
}
}
else
{
lean_object* v___x_2495_; 
lean_del_object(v___x_2487_);
v___x_2495_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2463_, v___y_2458_);
if (lean_obj_tag(v___x_2495_) == 0)
{
lean_object* v_a_2496_; lean_object* v___x_2498_; uint8_t v_isShared_2499_; uint8_t v_isSharedCheck_2505_; 
v_a_2496_ = lean_ctor_get(v___x_2495_, 0);
v_isSharedCheck_2505_ = !lean_is_exclusive(v___x_2495_);
if (v_isSharedCheck_2505_ == 0)
{
v___x_2498_ = v___x_2495_;
v_isShared_2499_ = v_isSharedCheck_2505_;
goto v_resetjp_2497_;
}
else
{
lean_inc(v_a_2496_);
lean_dec(v___x_2495_);
v___x_2498_ = lean_box(0);
v_isShared_2499_ = v_isSharedCheck_2505_;
goto v_resetjp_2497_;
}
v_resetjp_2497_:
{
lean_object* v___x_2500_; lean_object* v___x_2501_; lean_object* v___x_2503_; 
v___x_2500_ = lean_box(v_a_2456_);
v___x_2501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2501_, 0, v_a_2496_);
lean_ctor_set(v___x_2501_, 1, v___x_2500_);
if (v_isShared_2499_ == 0)
{
lean_ctor_set(v___x_2498_, 0, v___x_2501_);
v___x_2503_ = v___x_2498_;
goto v_reusejp_2502_;
}
else
{
lean_object* v_reuseFailAlloc_2504_; 
v_reuseFailAlloc_2504_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2504_, 0, v___x_2501_);
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
lean_object* v_a_2506_; lean_object* v___x_2508_; uint8_t v_isShared_2509_; uint8_t v_isSharedCheck_2513_; 
v_a_2506_ = lean_ctor_get(v___x_2495_, 0);
v_isSharedCheck_2513_ = !lean_is_exclusive(v___x_2495_);
if (v_isSharedCheck_2513_ == 0)
{
v___x_2508_ = v___x_2495_;
v_isShared_2509_ = v_isSharedCheck_2513_;
goto v_resetjp_2507_;
}
else
{
lean_inc(v_a_2506_);
lean_dec(v___x_2495_);
v___x_2508_ = lean_box(0);
v_isShared_2509_ = v_isSharedCheck_2513_;
goto v_resetjp_2507_;
}
v_resetjp_2507_:
{
lean_object* v___x_2511_; 
if (v_isShared_2509_ == 0)
{
v___x_2511_ = v___x_2508_;
goto v_reusejp_2510_;
}
else
{
lean_object* v_reuseFailAlloc_2512_; 
v_reuseFailAlloc_2512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2512_, 0, v_a_2506_);
v___x_2511_ = v_reuseFailAlloc_2512_;
goto v_reusejp_2510_;
}
v_reusejp_2510_:
{
return v___x_2511_;
}
}
}
}
}
}
else
{
lean_object* v_a_2515_; lean_object* v___x_2517_; uint8_t v_isShared_2518_; uint8_t v_isSharedCheck_2522_; 
lean_dec(v_a_2463_);
v_a_2515_ = lean_ctor_get(v___x_2484_, 0);
v_isSharedCheck_2522_ = !lean_is_exclusive(v___x_2484_);
if (v_isSharedCheck_2522_ == 0)
{
v___x_2517_ = v___x_2484_;
v_isShared_2518_ = v_isSharedCheck_2522_;
goto v_resetjp_2516_;
}
else
{
lean_inc(v_a_2515_);
lean_dec(v___x_2484_);
v___x_2517_ = lean_box(0);
v_isShared_2518_ = v_isSharedCheck_2522_;
goto v_resetjp_2516_;
}
v_resetjp_2516_:
{
lean_object* v___x_2520_; 
if (v_isShared_2518_ == 0)
{
v___x_2520_ = v___x_2517_;
goto v_reusejp_2519_;
}
else
{
lean_object* v_reuseFailAlloc_2521_; 
v_reuseFailAlloc_2521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2521_, 0, v_a_2515_);
v___x_2520_ = v_reuseFailAlloc_2521_;
goto v_reusejp_2519_;
}
v_reusejp_2519_:
{
return v___x_2520_;
}
}
}
}
}
}
else
{
lean_object* v_a_2525_; lean_object* v___x_2527_; uint8_t v_isShared_2528_; uint8_t v_isSharedCheck_2532_; 
lean_dec_ref(v___y_2457_);
lean_dec_ref(v_fst_2454_);
v_a_2525_ = lean_ctor_get(v___x_2462_, 0);
v_isSharedCheck_2532_ = !lean_is_exclusive(v___x_2462_);
if (v_isSharedCheck_2532_ == 0)
{
v___x_2527_ = v___x_2462_;
v_isShared_2528_ = v_isSharedCheck_2532_;
goto v_resetjp_2526_;
}
else
{
lean_inc(v_a_2525_);
lean_dec(v___x_2462_);
v___x_2527_ = lean_box(0);
v_isShared_2528_ = v_isSharedCheck_2532_;
goto v_resetjp_2526_;
}
v_resetjp_2526_:
{
lean_object* v___x_2530_; 
if (v_isShared_2528_ == 0)
{
v___x_2530_ = v___x_2527_;
goto v_reusejp_2529_;
}
else
{
lean_object* v_reuseFailAlloc_2531_; 
v_reuseFailAlloc_2531_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2531_, 0, v_a_2525_);
v___x_2530_ = v_reuseFailAlloc_2531_;
goto v_reusejp_2529_;
}
v_reusejp_2529_:
{
return v___x_2530_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___boxed(lean_object* v___x_2533_, lean_object* v___x_2534_, lean_object* v___x_2535_, lean_object* v_fst_2536_, lean_object* v___x_2537_, lean_object* v_a_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_){
_start:
{
uint8_t v___x_191566__boxed_2544_; uint8_t v___x_191569__boxed_2545_; uint8_t v_a_191570__boxed_2546_; lean_object* v_res_2547_; 
v___x_191566__boxed_2544_ = lean_unbox(v___x_2534_);
v___x_191569__boxed_2545_ = lean_unbox(v___x_2537_);
v_a_191570__boxed_2546_ = lean_unbox(v_a_2538_);
v_res_2547_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4(v___x_2533_, v___x_191566__boxed_2544_, v___x_2535_, v_fst_2536_, v___x_191569__boxed_2545_, v_a_191570__boxed_2546_, v___y_2539_, v___y_2540_, v___y_2541_, v___y_2542_);
lean_dec(v___y_2542_);
lean_dec_ref(v___y_2541_);
lean_dec(v___y_2540_);
return v_res_2547_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__2(void){
_start:
{
lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; 
v___x_2551_ = lean_box(0);
v___x_2552_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__1));
v___x_2553_ = l_Lean_Expr_const___override(v___x_2552_, v___x_2551_);
return v___x_2553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5(lean_object* v___x_2554_, uint8_t v___x_2555_, lean_object* v___x_2556_, lean_object* v_fst_2557_, uint8_t v___x_2558_, uint8_t v_a_2559_, lean_object* v___y_2560_, lean_object* v___y_2561_, lean_object* v___y_2562_, lean_object* v___y_2563_){
_start:
{
lean_object* v___x_2565_; 
lean_inc(v___x_2556_);
lean_inc(v___x_2554_);
v___x_2565_ = l_Lean_Meta_mkFreshExprMVar(v___x_2554_, v___x_2555_, v___x_2556_, v___y_2560_, v___y_2561_, v___y_2562_, v___y_2563_);
if (lean_obj_tag(v___x_2565_) == 0)
{
lean_object* v_a_2566_; lean_object* v___x_2567_; 
v_a_2566_ = lean_ctor_get(v___x_2565_, 0);
lean_inc(v_a_2566_);
lean_dec_ref_known(v___x_2565_, 1);
v___x_2567_ = l_Lean_Meta_mkFreshExprMVar(v___x_2554_, v___x_2555_, v___x_2556_, v___y_2560_, v___y_2561_, v___y_2562_, v___y_2563_);
if (lean_obj_tag(v___x_2567_) == 0)
{
lean_object* v_a_2568_; lean_object* v_keyedConfig_2569_; uint8_t v_trackZetaDelta_2570_; lean_object* v_zetaDeltaSet_2571_; lean_object* v_lctx_2572_; lean_object* v_localInstances_2573_; lean_object* v_defEqCtx_x3f_2574_; lean_object* v_synthPendingDepth_2575_; lean_object* v_customCanUnfoldPredicate_x3f_2576_; uint8_t v_univApprox_2577_; uint8_t v_inTypeClassResolution_2578_; uint8_t v_cacheInferType_2579_; lean_object* v___x_2581_; uint8_t v_isShared_2582_; uint8_t v_isSharedCheck_2642_; 
v_a_2568_ = lean_ctor_get(v___x_2567_, 0);
lean_inc(v_a_2568_);
lean_dec_ref_known(v___x_2567_, 1);
v_keyedConfig_2569_ = lean_ctor_get(v___y_2560_, 0);
v_trackZetaDelta_2570_ = lean_ctor_get_uint8(v___y_2560_, sizeof(void*)*7);
v_zetaDeltaSet_2571_ = lean_ctor_get(v___y_2560_, 1);
v_lctx_2572_ = lean_ctor_get(v___y_2560_, 2);
v_localInstances_2573_ = lean_ctor_get(v___y_2560_, 3);
v_defEqCtx_x3f_2574_ = lean_ctor_get(v___y_2560_, 4);
v_synthPendingDepth_2575_ = lean_ctor_get(v___y_2560_, 5);
v_customCanUnfoldPredicate_x3f_2576_ = lean_ctor_get(v___y_2560_, 6);
v_univApprox_2577_ = lean_ctor_get_uint8(v___y_2560_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2578_ = lean_ctor_get_uint8(v___y_2560_, sizeof(void*)*7 + 2);
v_cacheInferType_2579_ = lean_ctor_get_uint8(v___y_2560_, sizeof(void*)*7 + 3);
v_isSharedCheck_2642_ = !lean_is_exclusive(v___y_2560_);
if (v_isSharedCheck_2642_ == 0)
{
v___x_2581_ = v___y_2560_;
v_isShared_2582_ = v_isSharedCheck_2642_;
goto v_resetjp_2580_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2576_);
lean_inc(v_synthPendingDepth_2575_);
lean_inc(v_defEqCtx_x3f_2574_);
lean_inc(v_localInstances_2573_);
lean_inc(v_lctx_2572_);
lean_inc(v_zetaDeltaSet_2571_);
lean_inc(v_keyedConfig_2569_);
lean_dec(v___y_2560_);
v___x_2581_ = lean_box(0);
v_isShared_2582_ = v_isSharedCheck_2642_;
goto v_resetjp_2580_;
}
v_resetjp_2580_:
{
lean_object* v___x_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; uint8_t v___x_2586_; lean_object* v___x_2587_; lean_object* v___x_2589_; 
v___x_2583_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__2, &lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___closed__2);
lean_inc(v_a_2566_);
v___x_2584_ = l_Lean_Expr_app___override(v___x_2583_, v_a_2566_);
lean_inc(v_a_2568_);
v___x_2585_ = l_Lean_Expr_app___override(v___x_2584_, v_a_2568_);
v___x_2586_ = 2;
v___x_2587_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2586_, v_keyedConfig_2569_);
if (v_isShared_2582_ == 0)
{
lean_ctor_set(v___x_2581_, 0, v___x_2587_);
v___x_2589_ = v___x_2581_;
goto v_reusejp_2588_;
}
else
{
lean_object* v_reuseFailAlloc_2641_; 
v_reuseFailAlloc_2641_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2641_, 0, v___x_2587_);
lean_ctor_set(v_reuseFailAlloc_2641_, 1, v_zetaDeltaSet_2571_);
lean_ctor_set(v_reuseFailAlloc_2641_, 2, v_lctx_2572_);
lean_ctor_set(v_reuseFailAlloc_2641_, 3, v_localInstances_2573_);
lean_ctor_set(v_reuseFailAlloc_2641_, 4, v_defEqCtx_x3f_2574_);
lean_ctor_set(v_reuseFailAlloc_2641_, 5, v_synthPendingDepth_2575_);
lean_ctor_set(v_reuseFailAlloc_2641_, 6, v_customCanUnfoldPredicate_x3f_2576_);
lean_ctor_set_uint8(v_reuseFailAlloc_2641_, sizeof(void*)*7, v_trackZetaDelta_2570_);
lean_ctor_set_uint8(v_reuseFailAlloc_2641_, sizeof(void*)*7 + 1, v_univApprox_2577_);
lean_ctor_set_uint8(v_reuseFailAlloc_2641_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2578_);
lean_ctor_set_uint8(v_reuseFailAlloc_2641_, sizeof(void*)*7 + 3, v_cacheInferType_2579_);
v___x_2589_ = v_reuseFailAlloc_2641_;
goto v_reusejp_2588_;
}
v_reusejp_2588_:
{
lean_object* v___x_2590_; 
v___x_2590_ = l_Lean_Meta_isExprDefEq(v___x_2585_, v_fst_2557_, v___x_2589_, v___y_2561_, v___y_2562_, v___y_2563_);
lean_dec_ref(v___x_2589_);
if (lean_obj_tag(v___x_2590_) == 0)
{
lean_object* v_a_2591_; lean_object* v___x_2593_; uint8_t v_isShared_2594_; uint8_t v_isSharedCheck_2632_; 
v_a_2591_ = lean_ctor_get(v___x_2590_, 0);
v_isSharedCheck_2632_ = !lean_is_exclusive(v___x_2590_);
if (v_isSharedCheck_2632_ == 0)
{
v___x_2593_ = v___x_2590_;
v_isShared_2594_ = v_isSharedCheck_2632_;
goto v_resetjp_2592_;
}
else
{
lean_inc(v_a_2591_);
lean_dec(v___x_2590_);
v___x_2593_ = lean_box(0);
v_isShared_2594_ = v_isSharedCheck_2632_;
goto v_resetjp_2592_;
}
v_resetjp_2592_:
{
uint8_t v___x_2595_; 
v___x_2595_ = lean_unbox(v_a_2591_);
lean_dec(v_a_2591_);
if (v___x_2595_ == 0)
{
lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2600_; 
v___x_2596_ = lean_box(v___x_2558_);
v___x_2597_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2597_, 0, v_a_2568_);
lean_ctor_set(v___x_2597_, 1, v___x_2596_);
v___x_2598_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2598_, 0, v_a_2566_);
lean_ctor_set(v___x_2598_, 1, v___x_2597_);
if (v_isShared_2594_ == 0)
{
lean_ctor_set(v___x_2593_, 0, v___x_2598_);
v___x_2600_ = v___x_2593_;
goto v_reusejp_2599_;
}
else
{
lean_object* v_reuseFailAlloc_2601_; 
v_reuseFailAlloc_2601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2601_, 0, v___x_2598_);
v___x_2600_ = v_reuseFailAlloc_2601_;
goto v_reusejp_2599_;
}
v_reusejp_2599_:
{
return v___x_2600_;
}
}
else
{
lean_object* v___x_2602_; 
lean_del_object(v___x_2593_);
v___x_2602_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2566_, v___y_2561_);
if (lean_obj_tag(v___x_2602_) == 0)
{
lean_object* v_a_2603_; lean_object* v___x_2604_; 
v_a_2603_ = lean_ctor_get(v___x_2602_, 0);
lean_inc(v_a_2603_);
lean_dec_ref_known(v___x_2602_, 1);
v___x_2604_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2568_, v___y_2561_);
if (lean_obj_tag(v___x_2604_) == 0)
{
lean_object* v_a_2605_; lean_object* v___x_2607_; uint8_t v_isShared_2608_; uint8_t v_isSharedCheck_2615_; 
v_a_2605_ = lean_ctor_get(v___x_2604_, 0);
v_isSharedCheck_2615_ = !lean_is_exclusive(v___x_2604_);
if (v_isSharedCheck_2615_ == 0)
{
v___x_2607_ = v___x_2604_;
v_isShared_2608_ = v_isSharedCheck_2615_;
goto v_resetjp_2606_;
}
else
{
lean_inc(v_a_2605_);
lean_dec(v___x_2604_);
v___x_2607_ = lean_box(0);
v_isShared_2608_ = v_isSharedCheck_2615_;
goto v_resetjp_2606_;
}
v_resetjp_2606_:
{
lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2613_; 
v___x_2609_ = lean_box(v_a_2559_);
v___x_2610_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2610_, 0, v_a_2605_);
lean_ctor_set(v___x_2610_, 1, v___x_2609_);
v___x_2611_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2611_, 0, v_a_2603_);
lean_ctor_set(v___x_2611_, 1, v___x_2610_);
if (v_isShared_2608_ == 0)
{
lean_ctor_set(v___x_2607_, 0, v___x_2611_);
v___x_2613_ = v___x_2607_;
goto v_reusejp_2612_;
}
else
{
lean_object* v_reuseFailAlloc_2614_; 
v_reuseFailAlloc_2614_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2614_, 0, v___x_2611_);
v___x_2613_ = v_reuseFailAlloc_2614_;
goto v_reusejp_2612_;
}
v_reusejp_2612_:
{
return v___x_2613_;
}
}
}
else
{
lean_object* v_a_2616_; lean_object* v___x_2618_; uint8_t v_isShared_2619_; uint8_t v_isSharedCheck_2623_; 
lean_dec(v_a_2603_);
v_a_2616_ = lean_ctor_get(v___x_2604_, 0);
v_isSharedCheck_2623_ = !lean_is_exclusive(v___x_2604_);
if (v_isSharedCheck_2623_ == 0)
{
v___x_2618_ = v___x_2604_;
v_isShared_2619_ = v_isSharedCheck_2623_;
goto v_resetjp_2617_;
}
else
{
lean_inc(v_a_2616_);
lean_dec(v___x_2604_);
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
else
{
lean_object* v_a_2624_; lean_object* v___x_2626_; uint8_t v_isShared_2627_; uint8_t v_isSharedCheck_2631_; 
lean_dec(v_a_2568_);
v_a_2624_ = lean_ctor_get(v___x_2602_, 0);
v_isSharedCheck_2631_ = !lean_is_exclusive(v___x_2602_);
if (v_isSharedCheck_2631_ == 0)
{
v___x_2626_ = v___x_2602_;
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
else
{
lean_inc(v_a_2624_);
lean_dec(v___x_2602_);
v___x_2626_ = lean_box(0);
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
v_resetjp_2625_:
{
lean_object* v___x_2629_; 
if (v_isShared_2627_ == 0)
{
v___x_2629_ = v___x_2626_;
goto v_reusejp_2628_;
}
else
{
lean_object* v_reuseFailAlloc_2630_; 
v_reuseFailAlloc_2630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2630_, 0, v_a_2624_);
v___x_2629_ = v_reuseFailAlloc_2630_;
goto v_reusejp_2628_;
}
v_reusejp_2628_:
{
return v___x_2629_;
}
}
}
}
}
}
else
{
lean_object* v_a_2633_; lean_object* v___x_2635_; uint8_t v_isShared_2636_; uint8_t v_isSharedCheck_2640_; 
lean_dec(v_a_2568_);
lean_dec(v_a_2566_);
v_a_2633_ = lean_ctor_get(v___x_2590_, 0);
v_isSharedCheck_2640_ = !lean_is_exclusive(v___x_2590_);
if (v_isSharedCheck_2640_ == 0)
{
v___x_2635_ = v___x_2590_;
v_isShared_2636_ = v_isSharedCheck_2640_;
goto v_resetjp_2634_;
}
else
{
lean_inc(v_a_2633_);
lean_dec(v___x_2590_);
v___x_2635_ = lean_box(0);
v_isShared_2636_ = v_isSharedCheck_2640_;
goto v_resetjp_2634_;
}
v_resetjp_2634_:
{
lean_object* v___x_2638_; 
if (v_isShared_2636_ == 0)
{
v___x_2638_ = v___x_2635_;
goto v_reusejp_2637_;
}
else
{
lean_object* v_reuseFailAlloc_2639_; 
v_reuseFailAlloc_2639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2639_, 0, v_a_2633_);
v___x_2638_ = v_reuseFailAlloc_2639_;
goto v_reusejp_2637_;
}
v_reusejp_2637_:
{
return v___x_2638_;
}
}
}
}
}
}
else
{
lean_object* v_a_2643_; lean_object* v___x_2645_; uint8_t v_isShared_2646_; uint8_t v_isSharedCheck_2650_; 
lean_dec(v_a_2566_);
lean_dec_ref(v___y_2560_);
lean_dec_ref(v_fst_2557_);
v_a_2643_ = lean_ctor_get(v___x_2567_, 0);
v_isSharedCheck_2650_ = !lean_is_exclusive(v___x_2567_);
if (v_isSharedCheck_2650_ == 0)
{
v___x_2645_ = v___x_2567_;
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
else
{
lean_inc(v_a_2643_);
lean_dec(v___x_2567_);
v___x_2645_ = lean_box(0);
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
v_resetjp_2644_:
{
lean_object* v___x_2648_; 
if (v_isShared_2646_ == 0)
{
v___x_2648_ = v___x_2645_;
goto v_reusejp_2647_;
}
else
{
lean_object* v_reuseFailAlloc_2649_; 
v_reuseFailAlloc_2649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2649_, 0, v_a_2643_);
v___x_2648_ = v_reuseFailAlloc_2649_;
goto v_reusejp_2647_;
}
v_reusejp_2647_:
{
return v___x_2648_;
}
}
}
}
else
{
lean_object* v_a_2651_; lean_object* v___x_2653_; uint8_t v_isShared_2654_; uint8_t v_isSharedCheck_2658_; 
lean_dec_ref(v___y_2560_);
lean_dec_ref(v_fst_2557_);
lean_dec(v___x_2556_);
lean_dec(v___x_2554_);
v_a_2651_ = lean_ctor_get(v___x_2565_, 0);
v_isSharedCheck_2658_ = !lean_is_exclusive(v___x_2565_);
if (v_isSharedCheck_2658_ == 0)
{
v___x_2653_ = v___x_2565_;
v_isShared_2654_ = v_isSharedCheck_2658_;
goto v_resetjp_2652_;
}
else
{
lean_inc(v_a_2651_);
lean_dec(v___x_2565_);
v___x_2653_ = lean_box(0);
v_isShared_2654_ = v_isSharedCheck_2658_;
goto v_resetjp_2652_;
}
v_resetjp_2652_:
{
lean_object* v___x_2656_; 
if (v_isShared_2654_ == 0)
{
v___x_2656_ = v___x_2653_;
goto v_reusejp_2655_;
}
else
{
lean_object* v_reuseFailAlloc_2657_; 
v_reuseFailAlloc_2657_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2657_, 0, v_a_2651_);
v___x_2656_ = v_reuseFailAlloc_2657_;
goto v_reusejp_2655_;
}
v_reusejp_2655_:
{
return v___x_2656_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___boxed(lean_object* v___x_2659_, lean_object* v___x_2660_, lean_object* v___x_2661_, lean_object* v_fst_2662_, lean_object* v___x_2663_, lean_object* v_a_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_, lean_object* v___y_2669_){
_start:
{
uint8_t v___x_191737__boxed_2670_; uint8_t v___x_191740__boxed_2671_; uint8_t v_a_191741__boxed_2672_; lean_object* v_res_2673_; 
v___x_191737__boxed_2670_ = lean_unbox(v___x_2660_);
v___x_191740__boxed_2671_ = lean_unbox(v___x_2663_);
v_a_191741__boxed_2672_ = lean_unbox(v_a_2664_);
v_res_2673_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5(v___x_2659_, v___x_191737__boxed_2670_, v___x_2661_, v_fst_2662_, v___x_191740__boxed_2671_, v_a_191741__boxed_2672_, v___y_2665_, v___y_2666_, v___y_2667_, v___y_2668_);
lean_dec(v___y_2668_);
lean_dec_ref(v___y_2667_);
lean_dec(v___y_2666_);
return v_res_2673_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0(void){
_start:
{
lean_object* v___x_2674_; lean_object* v___x_2675_; 
v___x_2674_ = lean_box(0);
v___x_2675_ = l_Lean_Expr_sort___override(v___x_2674_);
return v___x_2675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6(lean_object* v_fst_2679_, uint8_t v_a_2680_, lean_object* v___y_2681_, lean_object* v___y_2682_, lean_object* v___y_2683_, lean_object* v___y_2684_){
_start:
{
lean_object* v___x_2686_; 
v___x_2686_ = l_Lean_Meta_mkFreshLevelMVar(v___y_2681_, v___y_2682_, v___y_2683_, v___y_2684_);
if (lean_obj_tag(v___x_2686_) == 0)
{
lean_object* v_a_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; uint8_t v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; 
v_a_2687_ = lean_ctor_get(v___x_2686_, 0);
lean_inc_n(v_a_2687_, 2);
lean_dec_ref_known(v___x_2686_, 1);
v___x_2688_ = l_Lean_Expr_sort___override(v_a_2687_);
v___x_2689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2689_, 0, v___x_2688_);
v___x_2690_ = 0;
v___x_2691_ = lean_box(0);
v___x_2692_ = l_Lean_Meta_mkFreshExprMVar(v___x_2689_, v___x_2690_, v___x_2691_, v___y_2681_, v___y_2682_, v___y_2683_, v___y_2684_);
if (lean_obj_tag(v___x_2692_) == 0)
{
lean_object* v_a_2693_; lean_object* v___x_2694_; uint8_t v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; 
v_a_2693_ = lean_ctor_get(v___x_2692_, 0);
lean_inc_n(v_a_2693_, 2);
lean_dec_ref_known(v___x_2692_, 1);
v___x_2694_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0, &lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0);
v___x_2695_ = 0;
v___x_2696_ = l_Lean_Expr_forallE___override(v___x_2691_, v_a_2693_, v___x_2694_, v___x_2695_);
v___x_2697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2697_, 0, v___x_2696_);
v___x_2698_ = l_Lean_Meta_mkFreshExprMVar(v___x_2697_, v___x_2690_, v___x_2691_, v___y_2681_, v___y_2682_, v___y_2683_, v___y_2684_);
if (lean_obj_tag(v___x_2698_) == 0)
{
lean_object* v_a_2699_; lean_object* v_keyedConfig_2700_; uint8_t v_trackZetaDelta_2701_; lean_object* v_zetaDeltaSet_2702_; lean_object* v_lctx_2703_; lean_object* v_localInstances_2704_; lean_object* v_defEqCtx_x3f_2705_; lean_object* v_synthPendingDepth_2706_; lean_object* v_customCanUnfoldPredicate_x3f_2707_; uint8_t v_univApprox_2708_; uint8_t v_inTypeClassResolution_2709_; uint8_t v_cacheInferType_2710_; lean_object* v___x_2712_; uint8_t v_isShared_2713_; uint8_t v_isSharedCheck_2787_; 
v_a_2699_ = lean_ctor_get(v___x_2698_, 0);
lean_inc(v_a_2699_);
lean_dec_ref_known(v___x_2698_, 1);
v_keyedConfig_2700_ = lean_ctor_get(v___y_2681_, 0);
v_trackZetaDelta_2701_ = lean_ctor_get_uint8(v___y_2681_, sizeof(void*)*7);
v_zetaDeltaSet_2702_ = lean_ctor_get(v___y_2681_, 1);
v_lctx_2703_ = lean_ctor_get(v___y_2681_, 2);
v_localInstances_2704_ = lean_ctor_get(v___y_2681_, 3);
v_defEqCtx_x3f_2705_ = lean_ctor_get(v___y_2681_, 4);
v_synthPendingDepth_2706_ = lean_ctor_get(v___y_2681_, 5);
v_customCanUnfoldPredicate_x3f_2707_ = lean_ctor_get(v___y_2681_, 6);
v_univApprox_2708_ = lean_ctor_get_uint8(v___y_2681_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2709_ = lean_ctor_get_uint8(v___y_2681_, sizeof(void*)*7 + 2);
v_cacheInferType_2710_ = lean_ctor_get_uint8(v___y_2681_, sizeof(void*)*7 + 3);
v_isSharedCheck_2787_ = !lean_is_exclusive(v___y_2681_);
if (v_isSharedCheck_2787_ == 0)
{
v___x_2712_ = v___y_2681_;
v_isShared_2713_ = v_isSharedCheck_2787_;
goto v_resetjp_2711_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2707_);
lean_inc(v_synthPendingDepth_2706_);
lean_inc(v_defEqCtx_x3f_2705_);
lean_inc(v_localInstances_2704_);
lean_inc(v_lctx_2703_);
lean_inc(v_zetaDeltaSet_2702_);
lean_inc(v_keyedConfig_2700_);
lean_dec(v___y_2681_);
v___x_2712_ = lean_box(0);
v_isShared_2713_ = v_isSharedCheck_2787_;
goto v_resetjp_2711_;
}
v_resetjp_2711_:
{
lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; uint8_t v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2723_; 
v___x_2714_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__2));
v___x_2715_ = lean_box(0);
lean_inc(v_a_2687_);
v___x_2716_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2716_, 0, v_a_2687_);
lean_ctor_set(v___x_2716_, 1, v___x_2715_);
v___x_2717_ = l_Lean_Expr_const___override(v___x_2714_, v___x_2716_);
lean_inc(v_a_2693_);
v___x_2718_ = l_Lean_Expr_app___override(v___x_2717_, v_a_2693_);
lean_inc(v_a_2699_);
v___x_2719_ = l_Lean_Expr_app___override(v___x_2718_, v_a_2699_);
v___x_2720_ = 2;
v___x_2721_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2720_, v_keyedConfig_2700_);
if (v_isShared_2713_ == 0)
{
lean_ctor_set(v___x_2712_, 0, v___x_2721_);
v___x_2723_ = v___x_2712_;
goto v_reusejp_2722_;
}
else
{
lean_object* v_reuseFailAlloc_2786_; 
v_reuseFailAlloc_2786_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2786_, 0, v___x_2721_);
lean_ctor_set(v_reuseFailAlloc_2786_, 1, v_zetaDeltaSet_2702_);
lean_ctor_set(v_reuseFailAlloc_2786_, 2, v_lctx_2703_);
lean_ctor_set(v_reuseFailAlloc_2786_, 3, v_localInstances_2704_);
lean_ctor_set(v_reuseFailAlloc_2786_, 4, v_defEqCtx_x3f_2705_);
lean_ctor_set(v_reuseFailAlloc_2786_, 5, v_synthPendingDepth_2706_);
lean_ctor_set(v_reuseFailAlloc_2786_, 6, v_customCanUnfoldPredicate_x3f_2707_);
lean_ctor_set_uint8(v_reuseFailAlloc_2786_, sizeof(void*)*7, v_trackZetaDelta_2701_);
lean_ctor_set_uint8(v_reuseFailAlloc_2786_, sizeof(void*)*7 + 1, v_univApprox_2708_);
lean_ctor_set_uint8(v_reuseFailAlloc_2786_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2709_);
lean_ctor_set_uint8(v_reuseFailAlloc_2786_, sizeof(void*)*7 + 3, v_cacheInferType_2710_);
v___x_2723_ = v_reuseFailAlloc_2786_;
goto v_reusejp_2722_;
}
v_reusejp_2722_:
{
lean_object* v___x_2724_; 
v___x_2724_ = l_Lean_Meta_isExprDefEq(v___x_2719_, v_fst_2679_, v___x_2723_, v___y_2682_, v___y_2683_, v___y_2684_);
lean_dec_ref(v___x_2723_);
if (lean_obj_tag(v___x_2724_) == 0)
{
lean_object* v_a_2725_; lean_object* v___x_2727_; uint8_t v_isShared_2728_; uint8_t v_isSharedCheck_2777_; 
v_a_2725_ = lean_ctor_get(v___x_2724_, 0);
v_isSharedCheck_2777_ = !lean_is_exclusive(v___x_2724_);
if (v_isSharedCheck_2777_ == 0)
{
v___x_2727_ = v___x_2724_;
v_isShared_2728_ = v_isSharedCheck_2777_;
goto v_resetjp_2726_;
}
else
{
lean_inc(v_a_2725_);
lean_dec(v___x_2724_);
v___x_2727_ = lean_box(0);
v_isShared_2728_ = v_isSharedCheck_2777_;
goto v_resetjp_2726_;
}
v_resetjp_2726_:
{
uint8_t v___x_2729_; 
v___x_2729_ = lean_unbox(v_a_2725_);
if (v___x_2729_ == 0)
{
lean_object* v___x_2730_; lean_object* v___x_2731_; lean_object* v___x_2732_; lean_object* v___x_2734_; 
v___x_2730_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2730_, 0, v_a_2699_);
lean_ctor_set(v___x_2730_, 1, v_a_2725_);
v___x_2731_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2731_, 0, v_a_2693_);
lean_ctor_set(v___x_2731_, 1, v___x_2730_);
v___x_2732_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2732_, 0, v_a_2687_);
lean_ctor_set(v___x_2732_, 1, v___x_2731_);
if (v_isShared_2728_ == 0)
{
lean_ctor_set(v___x_2727_, 0, v___x_2732_);
v___x_2734_ = v___x_2727_;
goto v_reusejp_2733_;
}
else
{
lean_object* v_reuseFailAlloc_2735_; 
v_reuseFailAlloc_2735_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2735_, 0, v___x_2732_);
v___x_2734_ = v_reuseFailAlloc_2735_;
goto v_reusejp_2733_;
}
v_reusejp_2733_:
{
return v___x_2734_;
}
}
else
{
lean_object* v___x_2736_; 
lean_del_object(v___x_2727_);
lean_dec(v_a_2725_);
v___x_2736_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_a_2687_, v___y_2682_);
if (lean_obj_tag(v___x_2736_) == 0)
{
lean_object* v_a_2737_; lean_object* v___x_2738_; 
v_a_2737_ = lean_ctor_get(v___x_2736_, 0);
lean_inc(v_a_2737_);
lean_dec_ref_known(v___x_2736_, 1);
v___x_2738_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2693_, v___y_2682_);
if (lean_obj_tag(v___x_2738_) == 0)
{
lean_object* v_a_2739_; lean_object* v___x_2740_; 
v_a_2739_ = lean_ctor_get(v___x_2738_, 0);
lean_inc(v_a_2739_);
lean_dec_ref_known(v___x_2738_, 1);
v___x_2740_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2699_, v___y_2682_);
if (lean_obj_tag(v___x_2740_) == 0)
{
lean_object* v_a_2741_; lean_object* v___x_2743_; uint8_t v_isShared_2744_; uint8_t v_isSharedCheck_2752_; 
v_a_2741_ = lean_ctor_get(v___x_2740_, 0);
v_isSharedCheck_2752_ = !lean_is_exclusive(v___x_2740_);
if (v_isSharedCheck_2752_ == 0)
{
v___x_2743_ = v___x_2740_;
v_isShared_2744_ = v_isSharedCheck_2752_;
goto v_resetjp_2742_;
}
else
{
lean_inc(v_a_2741_);
lean_dec(v___x_2740_);
v___x_2743_ = lean_box(0);
v_isShared_2744_ = v_isSharedCheck_2752_;
goto v_resetjp_2742_;
}
v_resetjp_2742_:
{
lean_object* v___x_2745_; lean_object* v___x_2746_; lean_object* v___x_2747_; lean_object* v___x_2748_; lean_object* v___x_2750_; 
v___x_2745_ = lean_box(v_a_2680_);
v___x_2746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2746_, 0, v_a_2741_);
lean_ctor_set(v___x_2746_, 1, v___x_2745_);
v___x_2747_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2747_, 0, v_a_2739_);
lean_ctor_set(v___x_2747_, 1, v___x_2746_);
v___x_2748_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2748_, 0, v_a_2737_);
lean_ctor_set(v___x_2748_, 1, v___x_2747_);
if (v_isShared_2744_ == 0)
{
lean_ctor_set(v___x_2743_, 0, v___x_2748_);
v___x_2750_ = v___x_2743_;
goto v_reusejp_2749_;
}
else
{
lean_object* v_reuseFailAlloc_2751_; 
v_reuseFailAlloc_2751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2751_, 0, v___x_2748_);
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
lean_object* v_a_2753_; lean_object* v___x_2755_; uint8_t v_isShared_2756_; uint8_t v_isSharedCheck_2760_; 
lean_dec(v_a_2739_);
lean_dec(v_a_2737_);
v_a_2753_ = lean_ctor_get(v___x_2740_, 0);
v_isSharedCheck_2760_ = !lean_is_exclusive(v___x_2740_);
if (v_isSharedCheck_2760_ == 0)
{
v___x_2755_ = v___x_2740_;
v_isShared_2756_ = v_isSharedCheck_2760_;
goto v_resetjp_2754_;
}
else
{
lean_inc(v_a_2753_);
lean_dec(v___x_2740_);
v___x_2755_ = lean_box(0);
v_isShared_2756_ = v_isSharedCheck_2760_;
goto v_resetjp_2754_;
}
v_resetjp_2754_:
{
lean_object* v___x_2758_; 
if (v_isShared_2756_ == 0)
{
v___x_2758_ = v___x_2755_;
goto v_reusejp_2757_;
}
else
{
lean_object* v_reuseFailAlloc_2759_; 
v_reuseFailAlloc_2759_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2759_, 0, v_a_2753_);
v___x_2758_ = v_reuseFailAlloc_2759_;
goto v_reusejp_2757_;
}
v_reusejp_2757_:
{
return v___x_2758_;
}
}
}
}
else
{
lean_object* v_a_2761_; lean_object* v___x_2763_; uint8_t v_isShared_2764_; uint8_t v_isSharedCheck_2768_; 
lean_dec(v_a_2737_);
lean_dec(v_a_2699_);
v_a_2761_ = lean_ctor_get(v___x_2738_, 0);
v_isSharedCheck_2768_ = !lean_is_exclusive(v___x_2738_);
if (v_isSharedCheck_2768_ == 0)
{
v___x_2763_ = v___x_2738_;
v_isShared_2764_ = v_isSharedCheck_2768_;
goto v_resetjp_2762_;
}
else
{
lean_inc(v_a_2761_);
lean_dec(v___x_2738_);
v___x_2763_ = lean_box(0);
v_isShared_2764_ = v_isSharedCheck_2768_;
goto v_resetjp_2762_;
}
v_resetjp_2762_:
{
lean_object* v___x_2766_; 
if (v_isShared_2764_ == 0)
{
v___x_2766_ = v___x_2763_;
goto v_reusejp_2765_;
}
else
{
lean_object* v_reuseFailAlloc_2767_; 
v_reuseFailAlloc_2767_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2767_, 0, v_a_2761_);
v___x_2766_ = v_reuseFailAlloc_2767_;
goto v_reusejp_2765_;
}
v_reusejp_2765_:
{
return v___x_2766_;
}
}
}
}
else
{
lean_object* v_a_2769_; lean_object* v___x_2771_; uint8_t v_isShared_2772_; uint8_t v_isSharedCheck_2776_; 
lean_dec(v_a_2699_);
lean_dec(v_a_2693_);
v_a_2769_ = lean_ctor_get(v___x_2736_, 0);
v_isSharedCheck_2776_ = !lean_is_exclusive(v___x_2736_);
if (v_isSharedCheck_2776_ == 0)
{
v___x_2771_ = v___x_2736_;
v_isShared_2772_ = v_isSharedCheck_2776_;
goto v_resetjp_2770_;
}
else
{
lean_inc(v_a_2769_);
lean_dec(v___x_2736_);
v___x_2771_ = lean_box(0);
v_isShared_2772_ = v_isSharedCheck_2776_;
goto v_resetjp_2770_;
}
v_resetjp_2770_:
{
lean_object* v___x_2774_; 
if (v_isShared_2772_ == 0)
{
v___x_2774_ = v___x_2771_;
goto v_reusejp_2773_;
}
else
{
lean_object* v_reuseFailAlloc_2775_; 
v_reuseFailAlloc_2775_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2775_, 0, v_a_2769_);
v___x_2774_ = v_reuseFailAlloc_2775_;
goto v_reusejp_2773_;
}
v_reusejp_2773_:
{
return v___x_2774_;
}
}
}
}
}
}
else
{
lean_object* v_a_2778_; lean_object* v___x_2780_; uint8_t v_isShared_2781_; uint8_t v_isSharedCheck_2785_; 
lean_dec(v_a_2699_);
lean_dec(v_a_2693_);
lean_dec(v_a_2687_);
v_a_2778_ = lean_ctor_get(v___x_2724_, 0);
v_isSharedCheck_2785_ = !lean_is_exclusive(v___x_2724_);
if (v_isSharedCheck_2785_ == 0)
{
v___x_2780_ = v___x_2724_;
v_isShared_2781_ = v_isSharedCheck_2785_;
goto v_resetjp_2779_;
}
else
{
lean_inc(v_a_2778_);
lean_dec(v___x_2724_);
v___x_2780_ = lean_box(0);
v_isShared_2781_ = v_isSharedCheck_2785_;
goto v_resetjp_2779_;
}
v_resetjp_2779_:
{
lean_object* v___x_2783_; 
if (v_isShared_2781_ == 0)
{
v___x_2783_ = v___x_2780_;
goto v_reusejp_2782_;
}
else
{
lean_object* v_reuseFailAlloc_2784_; 
v_reuseFailAlloc_2784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2784_, 0, v_a_2778_);
v___x_2783_ = v_reuseFailAlloc_2784_;
goto v_reusejp_2782_;
}
v_reusejp_2782_:
{
return v___x_2783_;
}
}
}
}
}
}
else
{
lean_object* v_a_2788_; lean_object* v___x_2790_; uint8_t v_isShared_2791_; uint8_t v_isSharedCheck_2795_; 
lean_dec(v_a_2693_);
lean_dec(v_a_2687_);
lean_dec_ref(v___y_2681_);
lean_dec_ref(v_fst_2679_);
v_a_2788_ = lean_ctor_get(v___x_2698_, 0);
v_isSharedCheck_2795_ = !lean_is_exclusive(v___x_2698_);
if (v_isSharedCheck_2795_ == 0)
{
v___x_2790_ = v___x_2698_;
v_isShared_2791_ = v_isSharedCheck_2795_;
goto v_resetjp_2789_;
}
else
{
lean_inc(v_a_2788_);
lean_dec(v___x_2698_);
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
else
{
lean_object* v_a_2796_; lean_object* v___x_2798_; uint8_t v_isShared_2799_; uint8_t v_isSharedCheck_2803_; 
lean_dec(v_a_2687_);
lean_dec_ref(v___y_2681_);
lean_dec_ref(v_fst_2679_);
v_a_2796_ = lean_ctor_get(v___x_2692_, 0);
v_isSharedCheck_2803_ = !lean_is_exclusive(v___x_2692_);
if (v_isSharedCheck_2803_ == 0)
{
v___x_2798_ = v___x_2692_;
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
else
{
lean_inc(v_a_2796_);
lean_dec(v___x_2692_);
v___x_2798_ = lean_box(0);
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
v_resetjp_2797_:
{
lean_object* v___x_2801_; 
if (v_isShared_2799_ == 0)
{
v___x_2801_ = v___x_2798_;
goto v_reusejp_2800_;
}
else
{
lean_object* v_reuseFailAlloc_2802_; 
v_reuseFailAlloc_2802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2802_, 0, v_a_2796_);
v___x_2801_ = v_reuseFailAlloc_2802_;
goto v_reusejp_2800_;
}
v_reusejp_2800_:
{
return v___x_2801_;
}
}
}
}
else
{
lean_object* v_a_2804_; lean_object* v___x_2806_; uint8_t v_isShared_2807_; uint8_t v_isSharedCheck_2811_; 
lean_dec_ref(v___y_2681_);
lean_dec_ref(v_fst_2679_);
v_a_2804_ = lean_ctor_get(v___x_2686_, 0);
v_isSharedCheck_2811_ = !lean_is_exclusive(v___x_2686_);
if (v_isSharedCheck_2811_ == 0)
{
v___x_2806_ = v___x_2686_;
v_isShared_2807_ = v_isSharedCheck_2811_;
goto v_resetjp_2805_;
}
else
{
lean_inc(v_a_2804_);
lean_dec(v___x_2686_);
v___x_2806_ = lean_box(0);
v_isShared_2807_ = v_isSharedCheck_2811_;
goto v_resetjp_2805_;
}
v_resetjp_2805_:
{
lean_object* v___x_2809_; 
if (v_isShared_2807_ == 0)
{
v___x_2809_ = v___x_2806_;
goto v_reusejp_2808_;
}
else
{
lean_object* v_reuseFailAlloc_2810_; 
v_reuseFailAlloc_2810_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2810_, 0, v_a_2804_);
v___x_2809_ = v_reuseFailAlloc_2810_;
goto v_reusejp_2808_;
}
v_reusejp_2808_:
{
return v___x_2809_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___boxed(lean_object* v_fst_2812_, lean_object* v_a_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_){
_start:
{
uint8_t v_a_191952__boxed_2819_; lean_object* v_res_2820_; 
v_a_191952__boxed_2819_ = lean_unbox(v_a_2813_);
v_res_2820_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6(v_fst_2812_, v_a_191952__boxed_2819_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_);
lean_dec(v___y_2817_);
lean_dec_ref(v___y_2816_);
lean_dec(v___y_2815_);
return v_res_2820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__7(uint8_t v___x_2821_, lean_object* v___x_2822_, lean_object* v_fst_2823_, uint8_t v___x_2824_, uint8_t v_a_2825_, lean_object* v___y_2826_, lean_object* v___y_2827_, lean_object* v___y_2828_, lean_object* v___y_2829_){
_start:
{
lean_object* v___x_2831_; 
v___x_2831_ = l_Lean_Meta_mkFreshLevelMVar(v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_);
if (lean_obj_tag(v___x_2831_) == 0)
{
lean_object* v_a_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; 
v_a_2832_ = lean_ctor_get(v___x_2831_, 0);
lean_inc_n(v_a_2832_, 2);
lean_dec_ref_known(v___x_2831_, 1);
v___x_2833_ = l_Lean_Level_succ___override(v_a_2832_);
v___x_2834_ = l_Lean_Expr_sort___override(v___x_2833_);
v___x_2835_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2835_, 0, v___x_2834_);
lean_inc(v___x_2822_);
v___x_2836_ = l_Lean_Meta_mkFreshExprMVar(v___x_2835_, v___x_2821_, v___x_2822_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_);
if (lean_obj_tag(v___x_2836_) == 0)
{
lean_object* v_a_2837_; lean_object* v___x_2838_; lean_object* v___x_2839_; lean_object* v___x_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; 
v_a_2837_ = lean_ctor_get(v___x_2836_, 0);
lean_inc_n(v_a_2837_, 2);
lean_dec_ref_known(v___x_2836_, 1);
v___x_2838_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_addAtom___closed__5));
v___x_2839_ = lean_box(0);
lean_inc(v_a_2832_);
v___x_2840_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2840_, 0, v_a_2832_);
lean_ctor_set(v___x_2840_, 1, v___x_2839_);
lean_inc_ref(v___x_2840_);
v___x_2841_ = l_Lean_Expr_const___override(v___x_2838_, v___x_2840_);
v___x_2842_ = l_Lean_Expr_app___override(v___x_2841_, v_a_2837_);
v___x_2843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2843_, 0, v___x_2842_);
lean_inc(v___x_2822_);
v___x_2844_ = l_Lean_Meta_mkFreshExprMVar(v___x_2843_, v___x_2821_, v___x_2822_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_);
if (lean_obj_tag(v___x_2844_) == 0)
{
lean_object* v_a_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; 
v_a_2845_ = lean_ctor_get(v___x_2844_, 0);
lean_inc(v_a_2845_);
lean_dec_ref_known(v___x_2844_, 1);
lean_inc(v_a_2837_);
v___x_2846_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2846_, 0, v_a_2837_);
lean_inc(v___x_2822_);
lean_inc_ref(v___x_2846_);
v___x_2847_ = l_Lean_Meta_mkFreshExprMVar(v___x_2846_, v___x_2821_, v___x_2822_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_);
if (lean_obj_tag(v___x_2847_) == 0)
{
lean_object* v_a_2848_; lean_object* v___x_2849_; 
v_a_2848_ = lean_ctor_get(v___x_2847_, 0);
lean_inc(v_a_2848_);
lean_dec_ref_known(v___x_2847_, 1);
v___x_2849_ = l_Lean_Meta_mkFreshExprMVar(v___x_2846_, v___x_2821_, v___x_2822_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_);
if (lean_obj_tag(v___x_2849_) == 0)
{
lean_object* v_a_2850_; lean_object* v_keyedConfig_2851_; uint8_t v_trackZetaDelta_2852_; lean_object* v_zetaDeltaSet_2853_; lean_object* v_lctx_2854_; lean_object* v_localInstances_2855_; lean_object* v_defEqCtx_x3f_2856_; lean_object* v_synthPendingDepth_2857_; lean_object* v_customCanUnfoldPredicate_x3f_2858_; uint8_t v_univApprox_2859_; uint8_t v_inTypeClassResolution_2860_; uint8_t v_cacheInferType_2861_; lean_object* v___x_2863_; uint8_t v_isShared_2864_; uint8_t v_isSharedCheck_2963_; 
v_a_2850_ = lean_ctor_get(v___x_2849_, 0);
lean_inc(v_a_2850_);
lean_dec_ref_known(v___x_2849_, 1);
v_keyedConfig_2851_ = lean_ctor_get(v___y_2826_, 0);
v_trackZetaDelta_2852_ = lean_ctor_get_uint8(v___y_2826_, sizeof(void*)*7);
v_zetaDeltaSet_2853_ = lean_ctor_get(v___y_2826_, 1);
v_lctx_2854_ = lean_ctor_get(v___y_2826_, 2);
v_localInstances_2855_ = lean_ctor_get(v___y_2826_, 3);
v_defEqCtx_x3f_2856_ = lean_ctor_get(v___y_2826_, 4);
v_synthPendingDepth_2857_ = lean_ctor_get(v___y_2826_, 5);
v_customCanUnfoldPredicate_x3f_2858_ = lean_ctor_get(v___y_2826_, 6);
v_univApprox_2859_ = lean_ctor_get_uint8(v___y_2826_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2860_ = lean_ctor_get_uint8(v___y_2826_, sizeof(void*)*7 + 2);
v_cacheInferType_2861_ = lean_ctor_get_uint8(v___y_2826_, sizeof(void*)*7 + 3);
v_isSharedCheck_2963_ = !lean_is_exclusive(v___y_2826_);
if (v_isSharedCheck_2963_ == 0)
{
v___x_2863_ = v___y_2826_;
v_isShared_2864_ = v_isSharedCheck_2963_;
goto v_resetjp_2862_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2858_);
lean_inc(v_synthPendingDepth_2857_);
lean_inc(v_defEqCtx_x3f_2856_);
lean_inc(v_localInstances_2855_);
lean_inc(v_lctx_2854_);
lean_inc(v_zetaDeltaSet_2853_);
lean_inc(v_keyedConfig_2851_);
lean_dec(v___y_2826_);
v___x_2863_ = lean_box(0);
v_isShared_2864_ = v_isSharedCheck_2963_;
goto v_resetjp_2862_;
}
v_resetjp_2862_:
{
lean_object* v___x_2865_; lean_object* v___x_2866_; lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; lean_object* v___x_2870_; uint8_t v___x_2871_; lean_object* v___x_2872_; lean_object* v___x_2874_; 
v___x_2865_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___closed__1));
v___x_2866_ = l_Lean_Expr_const___override(v___x_2865_, v___x_2840_);
lean_inc(v_a_2837_);
v___x_2867_ = l_Lean_Expr_app___override(v___x_2866_, v_a_2837_);
lean_inc(v_a_2845_);
v___x_2868_ = l_Lean_Expr_app___override(v___x_2867_, v_a_2845_);
lean_inc(v_a_2848_);
v___x_2869_ = l_Lean_Expr_app___override(v___x_2868_, v_a_2848_);
lean_inc(v_a_2850_);
v___x_2870_ = l_Lean_Expr_app___override(v___x_2869_, v_a_2850_);
v___x_2871_ = 2;
v___x_2872_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2871_, v_keyedConfig_2851_);
if (v_isShared_2864_ == 0)
{
lean_ctor_set(v___x_2863_, 0, v___x_2872_);
v___x_2874_ = v___x_2863_;
goto v_reusejp_2873_;
}
else
{
lean_object* v_reuseFailAlloc_2962_; 
v_reuseFailAlloc_2962_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2962_, 0, v___x_2872_);
lean_ctor_set(v_reuseFailAlloc_2962_, 1, v_zetaDeltaSet_2853_);
lean_ctor_set(v_reuseFailAlloc_2962_, 2, v_lctx_2854_);
lean_ctor_set(v_reuseFailAlloc_2962_, 3, v_localInstances_2855_);
lean_ctor_set(v_reuseFailAlloc_2962_, 4, v_defEqCtx_x3f_2856_);
lean_ctor_set(v_reuseFailAlloc_2962_, 5, v_synthPendingDepth_2857_);
lean_ctor_set(v_reuseFailAlloc_2962_, 6, v_customCanUnfoldPredicate_x3f_2858_);
lean_ctor_set_uint8(v_reuseFailAlloc_2962_, sizeof(void*)*7, v_trackZetaDelta_2852_);
lean_ctor_set_uint8(v_reuseFailAlloc_2962_, sizeof(void*)*7 + 1, v_univApprox_2859_);
lean_ctor_set_uint8(v_reuseFailAlloc_2962_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2860_);
lean_ctor_set_uint8(v_reuseFailAlloc_2962_, sizeof(void*)*7 + 3, v_cacheInferType_2861_);
v___x_2874_ = v_reuseFailAlloc_2962_;
goto v_reusejp_2873_;
}
v_reusejp_2873_:
{
lean_object* v___x_2875_; 
v___x_2875_ = l_Lean_Meta_isExprDefEq(v___x_2870_, v_fst_2823_, v___x_2874_, v___y_2827_, v___y_2828_, v___y_2829_);
lean_dec_ref(v___x_2874_);
if (lean_obj_tag(v___x_2875_) == 0)
{
lean_object* v_a_2876_; lean_object* v___x_2878_; uint8_t v_isShared_2879_; uint8_t v_isSharedCheck_2953_; 
v_a_2876_ = lean_ctor_get(v___x_2875_, 0);
v_isSharedCheck_2953_ = !lean_is_exclusive(v___x_2875_);
if (v_isSharedCheck_2953_ == 0)
{
v___x_2878_ = v___x_2875_;
v_isShared_2879_ = v_isSharedCheck_2953_;
goto v_resetjp_2877_;
}
else
{
lean_inc(v_a_2876_);
lean_dec(v___x_2875_);
v___x_2878_ = lean_box(0);
v_isShared_2879_ = v_isSharedCheck_2953_;
goto v_resetjp_2877_;
}
v_resetjp_2877_:
{
uint8_t v___x_2880_; 
v___x_2880_ = lean_unbox(v_a_2876_);
lean_dec(v_a_2876_);
if (v___x_2880_ == 0)
{
lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; lean_object* v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2888_; 
v___x_2881_ = lean_box(v___x_2824_);
v___x_2882_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2882_, 0, v_a_2850_);
lean_ctor_set(v___x_2882_, 1, v___x_2881_);
v___x_2883_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2883_, 0, v_a_2848_);
lean_ctor_set(v___x_2883_, 1, v___x_2882_);
v___x_2884_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2884_, 0, v_a_2845_);
lean_ctor_set(v___x_2884_, 1, v___x_2883_);
v___x_2885_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2885_, 0, v_a_2837_);
lean_ctor_set(v___x_2885_, 1, v___x_2884_);
v___x_2886_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2886_, 0, v_a_2832_);
lean_ctor_set(v___x_2886_, 1, v___x_2885_);
if (v_isShared_2879_ == 0)
{
lean_ctor_set(v___x_2878_, 0, v___x_2886_);
v___x_2888_ = v___x_2878_;
goto v_reusejp_2887_;
}
else
{
lean_object* v_reuseFailAlloc_2889_; 
v_reuseFailAlloc_2889_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2889_, 0, v___x_2886_);
v___x_2888_ = v_reuseFailAlloc_2889_;
goto v_reusejp_2887_;
}
v_reusejp_2887_:
{
return v___x_2888_;
}
}
else
{
lean_object* v___x_2890_; 
lean_del_object(v___x_2878_);
v___x_2890_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_a_2832_, v___y_2827_);
if (lean_obj_tag(v___x_2890_) == 0)
{
lean_object* v_a_2891_; lean_object* v___x_2892_; 
v_a_2891_ = lean_ctor_get(v___x_2890_, 0);
lean_inc(v_a_2891_);
lean_dec_ref_known(v___x_2890_, 1);
v___x_2892_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2837_, v___y_2827_);
if (lean_obj_tag(v___x_2892_) == 0)
{
lean_object* v_a_2893_; lean_object* v___x_2894_; 
v_a_2893_ = lean_ctor_get(v___x_2892_, 0);
lean_inc(v_a_2893_);
lean_dec_ref_known(v___x_2892_, 1);
v___x_2894_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2845_, v___y_2827_);
if (lean_obj_tag(v___x_2894_) == 0)
{
lean_object* v_a_2895_; lean_object* v___x_2896_; 
v_a_2895_ = lean_ctor_get(v___x_2894_, 0);
lean_inc(v_a_2895_);
lean_dec_ref_known(v___x_2894_, 1);
v___x_2896_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2848_, v___y_2827_);
if (lean_obj_tag(v___x_2896_) == 0)
{
lean_object* v_a_2897_; lean_object* v___x_2898_; 
v_a_2897_ = lean_ctor_get(v___x_2896_, 0);
lean_inc(v_a_2897_);
lean_dec_ref_known(v___x_2896_, 1);
v___x_2898_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_2850_, v___y_2827_);
if (lean_obj_tag(v___x_2898_) == 0)
{
lean_object* v_a_2899_; lean_object* v___x_2901_; uint8_t v_isShared_2902_; uint8_t v_isSharedCheck_2912_; 
v_a_2899_ = lean_ctor_get(v___x_2898_, 0);
v_isSharedCheck_2912_ = !lean_is_exclusive(v___x_2898_);
if (v_isSharedCheck_2912_ == 0)
{
v___x_2901_ = v___x_2898_;
v_isShared_2902_ = v_isSharedCheck_2912_;
goto v_resetjp_2900_;
}
else
{
lean_inc(v_a_2899_);
lean_dec(v___x_2898_);
v___x_2901_ = lean_box(0);
v_isShared_2902_ = v_isSharedCheck_2912_;
goto v_resetjp_2900_;
}
v_resetjp_2900_:
{
lean_object* v___x_2903_; lean_object* v___x_2904_; lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; lean_object* v___x_2908_; lean_object* v___x_2910_; 
v___x_2903_ = lean_box(v_a_2825_);
v___x_2904_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2904_, 0, v_a_2899_);
lean_ctor_set(v___x_2904_, 1, v___x_2903_);
v___x_2905_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2905_, 0, v_a_2897_);
lean_ctor_set(v___x_2905_, 1, v___x_2904_);
v___x_2906_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2906_, 0, v_a_2895_);
lean_ctor_set(v___x_2906_, 1, v___x_2905_);
v___x_2907_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2907_, 0, v_a_2893_);
lean_ctor_set(v___x_2907_, 1, v___x_2906_);
v___x_2908_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2908_, 0, v_a_2891_);
lean_ctor_set(v___x_2908_, 1, v___x_2907_);
if (v_isShared_2902_ == 0)
{
lean_ctor_set(v___x_2901_, 0, v___x_2908_);
v___x_2910_ = v___x_2901_;
goto v_reusejp_2909_;
}
else
{
lean_object* v_reuseFailAlloc_2911_; 
v_reuseFailAlloc_2911_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2911_, 0, v___x_2908_);
v___x_2910_ = v_reuseFailAlloc_2911_;
goto v_reusejp_2909_;
}
v_reusejp_2909_:
{
return v___x_2910_;
}
}
}
else
{
lean_object* v_a_2913_; lean_object* v___x_2915_; uint8_t v_isShared_2916_; uint8_t v_isSharedCheck_2920_; 
lean_dec(v_a_2897_);
lean_dec(v_a_2895_);
lean_dec(v_a_2893_);
lean_dec(v_a_2891_);
v_a_2913_ = lean_ctor_get(v___x_2898_, 0);
v_isSharedCheck_2920_ = !lean_is_exclusive(v___x_2898_);
if (v_isSharedCheck_2920_ == 0)
{
v___x_2915_ = v___x_2898_;
v_isShared_2916_ = v_isSharedCheck_2920_;
goto v_resetjp_2914_;
}
else
{
lean_inc(v_a_2913_);
lean_dec(v___x_2898_);
v___x_2915_ = lean_box(0);
v_isShared_2916_ = v_isSharedCheck_2920_;
goto v_resetjp_2914_;
}
v_resetjp_2914_:
{
lean_object* v___x_2918_; 
if (v_isShared_2916_ == 0)
{
v___x_2918_ = v___x_2915_;
goto v_reusejp_2917_;
}
else
{
lean_object* v_reuseFailAlloc_2919_; 
v_reuseFailAlloc_2919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2919_, 0, v_a_2913_);
v___x_2918_ = v_reuseFailAlloc_2919_;
goto v_reusejp_2917_;
}
v_reusejp_2917_:
{
return v___x_2918_;
}
}
}
}
else
{
lean_object* v_a_2921_; lean_object* v___x_2923_; uint8_t v_isShared_2924_; uint8_t v_isSharedCheck_2928_; 
lean_dec(v_a_2895_);
lean_dec(v_a_2893_);
lean_dec(v_a_2891_);
lean_dec(v_a_2850_);
v_a_2921_ = lean_ctor_get(v___x_2896_, 0);
v_isSharedCheck_2928_ = !lean_is_exclusive(v___x_2896_);
if (v_isSharedCheck_2928_ == 0)
{
v___x_2923_ = v___x_2896_;
v_isShared_2924_ = v_isSharedCheck_2928_;
goto v_resetjp_2922_;
}
else
{
lean_inc(v_a_2921_);
lean_dec(v___x_2896_);
v___x_2923_ = lean_box(0);
v_isShared_2924_ = v_isSharedCheck_2928_;
goto v_resetjp_2922_;
}
v_resetjp_2922_:
{
lean_object* v___x_2926_; 
if (v_isShared_2924_ == 0)
{
v___x_2926_ = v___x_2923_;
goto v_reusejp_2925_;
}
else
{
lean_object* v_reuseFailAlloc_2927_; 
v_reuseFailAlloc_2927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2927_, 0, v_a_2921_);
v___x_2926_ = v_reuseFailAlloc_2927_;
goto v_reusejp_2925_;
}
v_reusejp_2925_:
{
return v___x_2926_;
}
}
}
}
else
{
lean_object* v_a_2929_; lean_object* v___x_2931_; uint8_t v_isShared_2932_; uint8_t v_isSharedCheck_2936_; 
lean_dec(v_a_2893_);
lean_dec(v_a_2891_);
lean_dec(v_a_2850_);
lean_dec(v_a_2848_);
v_a_2929_ = lean_ctor_get(v___x_2894_, 0);
v_isSharedCheck_2936_ = !lean_is_exclusive(v___x_2894_);
if (v_isSharedCheck_2936_ == 0)
{
v___x_2931_ = v___x_2894_;
v_isShared_2932_ = v_isSharedCheck_2936_;
goto v_resetjp_2930_;
}
else
{
lean_inc(v_a_2929_);
lean_dec(v___x_2894_);
v___x_2931_ = lean_box(0);
v_isShared_2932_ = v_isSharedCheck_2936_;
goto v_resetjp_2930_;
}
v_resetjp_2930_:
{
lean_object* v___x_2934_; 
if (v_isShared_2932_ == 0)
{
v___x_2934_ = v___x_2931_;
goto v_reusejp_2933_;
}
else
{
lean_object* v_reuseFailAlloc_2935_; 
v_reuseFailAlloc_2935_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2935_, 0, v_a_2929_);
v___x_2934_ = v_reuseFailAlloc_2935_;
goto v_reusejp_2933_;
}
v_reusejp_2933_:
{
return v___x_2934_;
}
}
}
}
else
{
lean_object* v_a_2937_; lean_object* v___x_2939_; uint8_t v_isShared_2940_; uint8_t v_isSharedCheck_2944_; 
lean_dec(v_a_2891_);
lean_dec(v_a_2850_);
lean_dec(v_a_2848_);
lean_dec(v_a_2845_);
v_a_2937_ = lean_ctor_get(v___x_2892_, 0);
v_isSharedCheck_2944_ = !lean_is_exclusive(v___x_2892_);
if (v_isSharedCheck_2944_ == 0)
{
v___x_2939_ = v___x_2892_;
v_isShared_2940_ = v_isSharedCheck_2944_;
goto v_resetjp_2938_;
}
else
{
lean_inc(v_a_2937_);
lean_dec(v___x_2892_);
v___x_2939_ = lean_box(0);
v_isShared_2940_ = v_isSharedCheck_2944_;
goto v_resetjp_2938_;
}
v_resetjp_2938_:
{
lean_object* v___x_2942_; 
if (v_isShared_2940_ == 0)
{
v___x_2942_ = v___x_2939_;
goto v_reusejp_2941_;
}
else
{
lean_object* v_reuseFailAlloc_2943_; 
v_reuseFailAlloc_2943_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2943_, 0, v_a_2937_);
v___x_2942_ = v_reuseFailAlloc_2943_;
goto v_reusejp_2941_;
}
v_reusejp_2941_:
{
return v___x_2942_;
}
}
}
}
else
{
lean_object* v_a_2945_; lean_object* v___x_2947_; uint8_t v_isShared_2948_; uint8_t v_isSharedCheck_2952_; 
lean_dec(v_a_2850_);
lean_dec(v_a_2848_);
lean_dec(v_a_2845_);
lean_dec(v_a_2837_);
v_a_2945_ = lean_ctor_get(v___x_2890_, 0);
v_isSharedCheck_2952_ = !lean_is_exclusive(v___x_2890_);
if (v_isSharedCheck_2952_ == 0)
{
v___x_2947_ = v___x_2890_;
v_isShared_2948_ = v_isSharedCheck_2952_;
goto v_resetjp_2946_;
}
else
{
lean_inc(v_a_2945_);
lean_dec(v___x_2890_);
v___x_2947_ = lean_box(0);
v_isShared_2948_ = v_isSharedCheck_2952_;
goto v_resetjp_2946_;
}
v_resetjp_2946_:
{
lean_object* v___x_2950_; 
if (v_isShared_2948_ == 0)
{
v___x_2950_ = v___x_2947_;
goto v_reusejp_2949_;
}
else
{
lean_object* v_reuseFailAlloc_2951_; 
v_reuseFailAlloc_2951_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2951_, 0, v_a_2945_);
v___x_2950_ = v_reuseFailAlloc_2951_;
goto v_reusejp_2949_;
}
v_reusejp_2949_:
{
return v___x_2950_;
}
}
}
}
}
}
else
{
lean_object* v_a_2954_; lean_object* v___x_2956_; uint8_t v_isShared_2957_; uint8_t v_isSharedCheck_2961_; 
lean_dec(v_a_2850_);
lean_dec(v_a_2848_);
lean_dec(v_a_2845_);
lean_dec(v_a_2837_);
lean_dec(v_a_2832_);
v_a_2954_ = lean_ctor_get(v___x_2875_, 0);
v_isSharedCheck_2961_ = !lean_is_exclusive(v___x_2875_);
if (v_isSharedCheck_2961_ == 0)
{
v___x_2956_ = v___x_2875_;
v_isShared_2957_ = v_isSharedCheck_2961_;
goto v_resetjp_2955_;
}
else
{
lean_inc(v_a_2954_);
lean_dec(v___x_2875_);
v___x_2956_ = lean_box(0);
v_isShared_2957_ = v_isSharedCheck_2961_;
goto v_resetjp_2955_;
}
v_resetjp_2955_:
{
lean_object* v___x_2959_; 
if (v_isShared_2957_ == 0)
{
v___x_2959_ = v___x_2956_;
goto v_reusejp_2958_;
}
else
{
lean_object* v_reuseFailAlloc_2960_; 
v_reuseFailAlloc_2960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2960_, 0, v_a_2954_);
v___x_2959_ = v_reuseFailAlloc_2960_;
goto v_reusejp_2958_;
}
v_reusejp_2958_:
{
return v___x_2959_;
}
}
}
}
}
}
else
{
lean_object* v_a_2964_; lean_object* v___x_2966_; uint8_t v_isShared_2967_; uint8_t v_isSharedCheck_2971_; 
lean_dec(v_a_2848_);
lean_dec(v_a_2845_);
lean_dec_ref_known(v___x_2840_, 2);
lean_dec(v_a_2837_);
lean_dec(v_a_2832_);
lean_dec_ref(v___y_2826_);
lean_dec(v_fst_2823_);
v_a_2964_ = lean_ctor_get(v___x_2849_, 0);
v_isSharedCheck_2971_ = !lean_is_exclusive(v___x_2849_);
if (v_isSharedCheck_2971_ == 0)
{
v___x_2966_ = v___x_2849_;
v_isShared_2967_ = v_isSharedCheck_2971_;
goto v_resetjp_2965_;
}
else
{
lean_inc(v_a_2964_);
lean_dec(v___x_2849_);
v___x_2966_ = lean_box(0);
v_isShared_2967_ = v_isSharedCheck_2971_;
goto v_resetjp_2965_;
}
v_resetjp_2965_:
{
lean_object* v___x_2969_; 
if (v_isShared_2967_ == 0)
{
v___x_2969_ = v___x_2966_;
goto v_reusejp_2968_;
}
else
{
lean_object* v_reuseFailAlloc_2970_; 
v_reuseFailAlloc_2970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2970_, 0, v_a_2964_);
v___x_2969_ = v_reuseFailAlloc_2970_;
goto v_reusejp_2968_;
}
v_reusejp_2968_:
{
return v___x_2969_;
}
}
}
}
else
{
lean_object* v_a_2972_; lean_object* v___x_2974_; uint8_t v_isShared_2975_; uint8_t v_isSharedCheck_2979_; 
lean_dec_ref_known(v___x_2846_, 1);
lean_dec(v_a_2845_);
lean_dec_ref_known(v___x_2840_, 2);
lean_dec(v_a_2837_);
lean_dec(v_a_2832_);
lean_dec_ref(v___y_2826_);
lean_dec(v_fst_2823_);
lean_dec(v___x_2822_);
v_a_2972_ = lean_ctor_get(v___x_2847_, 0);
v_isSharedCheck_2979_ = !lean_is_exclusive(v___x_2847_);
if (v_isSharedCheck_2979_ == 0)
{
v___x_2974_ = v___x_2847_;
v_isShared_2975_ = v_isSharedCheck_2979_;
goto v_resetjp_2973_;
}
else
{
lean_inc(v_a_2972_);
lean_dec(v___x_2847_);
v___x_2974_ = lean_box(0);
v_isShared_2975_ = v_isSharedCheck_2979_;
goto v_resetjp_2973_;
}
v_resetjp_2973_:
{
lean_object* v___x_2977_; 
if (v_isShared_2975_ == 0)
{
v___x_2977_ = v___x_2974_;
goto v_reusejp_2976_;
}
else
{
lean_object* v_reuseFailAlloc_2978_; 
v_reuseFailAlloc_2978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2978_, 0, v_a_2972_);
v___x_2977_ = v_reuseFailAlloc_2978_;
goto v_reusejp_2976_;
}
v_reusejp_2976_:
{
return v___x_2977_;
}
}
}
}
else
{
lean_object* v_a_2980_; lean_object* v___x_2982_; uint8_t v_isShared_2983_; uint8_t v_isSharedCheck_2987_; 
lean_dec_ref_known(v___x_2840_, 2);
lean_dec(v_a_2837_);
lean_dec(v_a_2832_);
lean_dec_ref(v___y_2826_);
lean_dec(v_fst_2823_);
lean_dec(v___x_2822_);
v_a_2980_ = lean_ctor_get(v___x_2844_, 0);
v_isSharedCheck_2987_ = !lean_is_exclusive(v___x_2844_);
if (v_isSharedCheck_2987_ == 0)
{
v___x_2982_ = v___x_2844_;
v_isShared_2983_ = v_isSharedCheck_2987_;
goto v_resetjp_2981_;
}
else
{
lean_inc(v_a_2980_);
lean_dec(v___x_2844_);
v___x_2982_ = lean_box(0);
v_isShared_2983_ = v_isSharedCheck_2987_;
goto v_resetjp_2981_;
}
v_resetjp_2981_:
{
lean_object* v___x_2985_; 
if (v_isShared_2983_ == 0)
{
v___x_2985_ = v___x_2982_;
goto v_reusejp_2984_;
}
else
{
lean_object* v_reuseFailAlloc_2986_; 
v_reuseFailAlloc_2986_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2986_, 0, v_a_2980_);
v___x_2985_ = v_reuseFailAlloc_2986_;
goto v_reusejp_2984_;
}
v_reusejp_2984_:
{
return v___x_2985_;
}
}
}
}
else
{
lean_object* v_a_2988_; lean_object* v___x_2990_; uint8_t v_isShared_2991_; uint8_t v_isSharedCheck_2995_; 
lean_dec(v_a_2832_);
lean_dec_ref(v___y_2826_);
lean_dec(v_fst_2823_);
lean_dec(v___x_2822_);
v_a_2988_ = lean_ctor_get(v___x_2836_, 0);
v_isSharedCheck_2995_ = !lean_is_exclusive(v___x_2836_);
if (v_isSharedCheck_2995_ == 0)
{
v___x_2990_ = v___x_2836_;
v_isShared_2991_ = v_isSharedCheck_2995_;
goto v_resetjp_2989_;
}
else
{
lean_inc(v_a_2988_);
lean_dec(v___x_2836_);
v___x_2990_ = lean_box(0);
v_isShared_2991_ = v_isSharedCheck_2995_;
goto v_resetjp_2989_;
}
v_resetjp_2989_:
{
lean_object* v___x_2993_; 
if (v_isShared_2991_ == 0)
{
v___x_2993_ = v___x_2990_;
goto v_reusejp_2992_;
}
else
{
lean_object* v_reuseFailAlloc_2994_; 
v_reuseFailAlloc_2994_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2994_, 0, v_a_2988_);
v___x_2993_ = v_reuseFailAlloc_2994_;
goto v_reusejp_2992_;
}
v_reusejp_2992_:
{
return v___x_2993_;
}
}
}
}
else
{
lean_object* v_a_2996_; lean_object* v___x_2998_; uint8_t v_isShared_2999_; uint8_t v_isSharedCheck_3003_; 
lean_dec_ref(v___y_2826_);
lean_dec(v_fst_2823_);
lean_dec(v___x_2822_);
v_a_2996_ = lean_ctor_get(v___x_2831_, 0);
v_isSharedCheck_3003_ = !lean_is_exclusive(v___x_2831_);
if (v_isSharedCheck_3003_ == 0)
{
v___x_2998_ = v___x_2831_;
v_isShared_2999_ = v_isSharedCheck_3003_;
goto v_resetjp_2997_;
}
else
{
lean_inc(v_a_2996_);
lean_dec(v___x_2831_);
v___x_2998_ = lean_box(0);
v_isShared_2999_ = v_isSharedCheck_3003_;
goto v_resetjp_2997_;
}
v_resetjp_2997_:
{
lean_object* v___x_3001_; 
if (v_isShared_2999_ == 0)
{
v___x_3001_ = v___x_2998_;
goto v_reusejp_3000_;
}
else
{
lean_object* v_reuseFailAlloc_3002_; 
v_reuseFailAlloc_3002_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3002_, 0, v_a_2996_);
v___x_3001_ = v_reuseFailAlloc_3002_;
goto v_reusejp_3000_;
}
v_reusejp_3000_:
{
return v___x_3001_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__7___boxed(lean_object* v___x_3004_, lean_object* v___x_3005_, lean_object* v_fst_3006_, lean_object* v___x_3007_, lean_object* v_a_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_){
_start:
{
uint8_t v___x_192214__boxed_3014_; uint8_t v___x_192216__boxed_3015_; uint8_t v_a_192217__boxed_3016_; lean_object* v_res_3017_; 
v___x_192214__boxed_3014_ = lean_unbox(v___x_3004_);
v___x_192216__boxed_3015_ = lean_unbox(v___x_3007_);
v_a_192217__boxed_3016_ = lean_unbox(v_a_3008_);
v_res_3017_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__7(v___x_192214__boxed_3014_, v___x_3005_, v_fst_3006_, v___x_192216__boxed_3015_, v_a_192217__boxed_3016_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_);
lean_dec(v___y_3012_);
lean_dec_ref(v___y_3011_);
lean_dec(v___y_3010_);
return v_res_3017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__8(uint8_t v___x_3018_, lean_object* v___x_3019_, lean_object* v_fst_3020_, uint8_t v___x_3021_, uint8_t v_a_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_){
_start:
{
lean_object* v___x_3028_; 
v___x_3028_ = l_Lean_Meta_mkFreshLevelMVar(v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_);
if (lean_obj_tag(v___x_3028_) == 0)
{
lean_object* v_a_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; lean_object* v___x_3033_; 
v_a_3029_ = lean_ctor_get(v___x_3028_, 0);
lean_inc_n(v_a_3029_, 2);
lean_dec_ref_known(v___x_3028_, 1);
v___x_3030_ = l_Lean_Level_succ___override(v_a_3029_);
v___x_3031_ = l_Lean_Expr_sort___override(v___x_3030_);
v___x_3032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3032_, 0, v___x_3031_);
lean_inc(v___x_3019_);
v___x_3033_ = l_Lean_Meta_mkFreshExprMVar(v___x_3032_, v___x_3018_, v___x_3019_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_);
if (lean_obj_tag(v___x_3033_) == 0)
{
lean_object* v_a_3034_; lean_object* v___x_3035_; lean_object* v___x_3036_; lean_object* v___x_3037_; lean_object* v___x_3038_; lean_object* v___x_3039_; lean_object* v___x_3040_; lean_object* v___x_3041_; 
v_a_3034_ = lean_ctor_get(v___x_3033_, 0);
lean_inc_n(v_a_3034_, 2);
lean_dec_ref_known(v___x_3033_, 1);
v___x_3035_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__1));
v___x_3036_ = lean_box(0);
lean_inc(v_a_3029_);
v___x_3037_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3037_, 0, v_a_3029_);
lean_ctor_set(v___x_3037_, 1, v___x_3036_);
lean_inc_ref(v___x_3037_);
v___x_3038_ = l_Lean_Expr_const___override(v___x_3035_, v___x_3037_);
v___x_3039_ = l_Lean_Expr_app___override(v___x_3038_, v_a_3034_);
v___x_3040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3040_, 0, v___x_3039_);
lean_inc(v___x_3019_);
v___x_3041_ = l_Lean_Meta_mkFreshExprMVar(v___x_3040_, v___x_3018_, v___x_3019_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_);
if (lean_obj_tag(v___x_3041_) == 0)
{
lean_object* v_a_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; 
v_a_3042_ = lean_ctor_get(v___x_3041_, 0);
lean_inc(v_a_3042_);
lean_dec_ref_known(v___x_3041_, 1);
lean_inc(v_a_3034_);
v___x_3043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3043_, 0, v_a_3034_);
lean_inc(v___x_3019_);
lean_inc_ref(v___x_3043_);
v___x_3044_ = l_Lean_Meta_mkFreshExprMVar(v___x_3043_, v___x_3018_, v___x_3019_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_);
if (lean_obj_tag(v___x_3044_) == 0)
{
lean_object* v_a_3045_; lean_object* v___x_3046_; 
v_a_3045_ = lean_ctor_get(v___x_3044_, 0);
lean_inc(v_a_3045_);
lean_dec_ref_known(v___x_3044_, 1);
v___x_3046_ = l_Lean_Meta_mkFreshExprMVar(v___x_3043_, v___x_3018_, v___x_3019_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_);
if (lean_obj_tag(v___x_3046_) == 0)
{
lean_object* v_a_3047_; lean_object* v_keyedConfig_3048_; uint8_t v_trackZetaDelta_3049_; lean_object* v_zetaDeltaSet_3050_; lean_object* v_lctx_3051_; lean_object* v_localInstances_3052_; lean_object* v_defEqCtx_x3f_3053_; lean_object* v_synthPendingDepth_3054_; lean_object* v_customCanUnfoldPredicate_x3f_3055_; uint8_t v_univApprox_3056_; uint8_t v_inTypeClassResolution_3057_; uint8_t v_cacheInferType_3058_; lean_object* v___x_3060_; uint8_t v_isShared_3061_; uint8_t v_isSharedCheck_3160_; 
v_a_3047_ = lean_ctor_get(v___x_3046_, 0);
lean_inc(v_a_3047_);
lean_dec_ref_known(v___x_3046_, 1);
v_keyedConfig_3048_ = lean_ctor_get(v___y_3023_, 0);
v_trackZetaDelta_3049_ = lean_ctor_get_uint8(v___y_3023_, sizeof(void*)*7);
v_zetaDeltaSet_3050_ = lean_ctor_get(v___y_3023_, 1);
v_lctx_3051_ = lean_ctor_get(v___y_3023_, 2);
v_localInstances_3052_ = lean_ctor_get(v___y_3023_, 3);
v_defEqCtx_x3f_3053_ = lean_ctor_get(v___y_3023_, 4);
v_synthPendingDepth_3054_ = lean_ctor_get(v___y_3023_, 5);
v_customCanUnfoldPredicate_x3f_3055_ = lean_ctor_get(v___y_3023_, 6);
v_univApprox_3056_ = lean_ctor_get_uint8(v___y_3023_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3057_ = lean_ctor_get_uint8(v___y_3023_, sizeof(void*)*7 + 2);
v_cacheInferType_3058_ = lean_ctor_get_uint8(v___y_3023_, sizeof(void*)*7 + 3);
v_isSharedCheck_3160_ = !lean_is_exclusive(v___y_3023_);
if (v_isSharedCheck_3160_ == 0)
{
v___x_3060_ = v___y_3023_;
v_isShared_3061_ = v_isSharedCheck_3160_;
goto v_resetjp_3059_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3055_);
lean_inc(v_synthPendingDepth_3054_);
lean_inc(v_defEqCtx_x3f_3053_);
lean_inc(v_localInstances_3052_);
lean_inc(v_lctx_3051_);
lean_inc(v_zetaDeltaSet_3050_);
lean_inc(v_keyedConfig_3048_);
lean_dec(v___y_3023_);
v___x_3060_ = lean_box(0);
v_isShared_3061_ = v_isSharedCheck_3160_;
goto v_resetjp_3059_;
}
v_resetjp_3059_:
{
lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; lean_object* v___x_3065_; lean_object* v___x_3066_; lean_object* v___x_3067_; uint8_t v___x_3068_; lean_object* v___x_3069_; lean_object* v___x_3071_; 
v___x_3062_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___closed__3));
v___x_3063_ = l_Lean_Expr_const___override(v___x_3062_, v___x_3037_);
lean_inc(v_a_3034_);
v___x_3064_ = l_Lean_Expr_app___override(v___x_3063_, v_a_3034_);
lean_inc(v_a_3042_);
v___x_3065_ = l_Lean_Expr_app___override(v___x_3064_, v_a_3042_);
lean_inc(v_a_3045_);
v___x_3066_ = l_Lean_Expr_app___override(v___x_3065_, v_a_3045_);
lean_inc(v_a_3047_);
v___x_3067_ = l_Lean_Expr_app___override(v___x_3066_, v_a_3047_);
v___x_3068_ = 2;
v___x_3069_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3068_, v_keyedConfig_3048_);
if (v_isShared_3061_ == 0)
{
lean_ctor_set(v___x_3060_, 0, v___x_3069_);
v___x_3071_ = v___x_3060_;
goto v_reusejp_3070_;
}
else
{
lean_object* v_reuseFailAlloc_3159_; 
v_reuseFailAlloc_3159_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3159_, 0, v___x_3069_);
lean_ctor_set(v_reuseFailAlloc_3159_, 1, v_zetaDeltaSet_3050_);
lean_ctor_set(v_reuseFailAlloc_3159_, 2, v_lctx_3051_);
lean_ctor_set(v_reuseFailAlloc_3159_, 3, v_localInstances_3052_);
lean_ctor_set(v_reuseFailAlloc_3159_, 4, v_defEqCtx_x3f_3053_);
lean_ctor_set(v_reuseFailAlloc_3159_, 5, v_synthPendingDepth_3054_);
lean_ctor_set(v_reuseFailAlloc_3159_, 6, v_customCanUnfoldPredicate_x3f_3055_);
lean_ctor_set_uint8(v_reuseFailAlloc_3159_, sizeof(void*)*7, v_trackZetaDelta_3049_);
lean_ctor_set_uint8(v_reuseFailAlloc_3159_, sizeof(void*)*7 + 1, v_univApprox_3056_);
lean_ctor_set_uint8(v_reuseFailAlloc_3159_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3057_);
lean_ctor_set_uint8(v_reuseFailAlloc_3159_, sizeof(void*)*7 + 3, v_cacheInferType_3058_);
v___x_3071_ = v_reuseFailAlloc_3159_;
goto v_reusejp_3070_;
}
v_reusejp_3070_:
{
lean_object* v___x_3072_; 
v___x_3072_ = l_Lean_Meta_isExprDefEq(v___x_3067_, v_fst_3020_, v___x_3071_, v___y_3024_, v___y_3025_, v___y_3026_);
lean_dec_ref(v___x_3071_);
if (lean_obj_tag(v___x_3072_) == 0)
{
lean_object* v_a_3073_; lean_object* v___x_3075_; uint8_t v_isShared_3076_; uint8_t v_isSharedCheck_3150_; 
v_a_3073_ = lean_ctor_get(v___x_3072_, 0);
v_isSharedCheck_3150_ = !lean_is_exclusive(v___x_3072_);
if (v_isSharedCheck_3150_ == 0)
{
v___x_3075_ = v___x_3072_;
v_isShared_3076_ = v_isSharedCheck_3150_;
goto v_resetjp_3074_;
}
else
{
lean_inc(v_a_3073_);
lean_dec(v___x_3072_);
v___x_3075_ = lean_box(0);
v_isShared_3076_ = v_isSharedCheck_3150_;
goto v_resetjp_3074_;
}
v_resetjp_3074_:
{
uint8_t v___x_3077_; 
v___x_3077_ = lean_unbox(v_a_3073_);
lean_dec(v_a_3073_);
if (v___x_3077_ == 0)
{
lean_object* v___x_3078_; lean_object* v___x_3079_; lean_object* v___x_3080_; lean_object* v___x_3081_; lean_object* v___x_3082_; lean_object* v___x_3083_; lean_object* v___x_3085_; 
v___x_3078_ = lean_box(v___x_3021_);
v___x_3079_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3079_, 0, v_a_3047_);
lean_ctor_set(v___x_3079_, 1, v___x_3078_);
v___x_3080_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3080_, 0, v_a_3045_);
lean_ctor_set(v___x_3080_, 1, v___x_3079_);
v___x_3081_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3081_, 0, v_a_3042_);
lean_ctor_set(v___x_3081_, 1, v___x_3080_);
v___x_3082_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3082_, 0, v_a_3034_);
lean_ctor_set(v___x_3082_, 1, v___x_3081_);
v___x_3083_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3083_, 0, v_a_3029_);
lean_ctor_set(v___x_3083_, 1, v___x_3082_);
if (v_isShared_3076_ == 0)
{
lean_ctor_set(v___x_3075_, 0, v___x_3083_);
v___x_3085_ = v___x_3075_;
goto v_reusejp_3084_;
}
else
{
lean_object* v_reuseFailAlloc_3086_; 
v_reuseFailAlloc_3086_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3086_, 0, v___x_3083_);
v___x_3085_ = v_reuseFailAlloc_3086_;
goto v_reusejp_3084_;
}
v_reusejp_3084_:
{
return v___x_3085_;
}
}
else
{
lean_object* v___x_3087_; 
lean_del_object(v___x_3075_);
v___x_3087_ = lp_mathlib_Lean_instantiateLevelMVars___at___00__private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr_spec__0___redArg(v_a_3029_, v___y_3024_);
if (lean_obj_tag(v___x_3087_) == 0)
{
lean_object* v_a_3088_; lean_object* v___x_3089_; 
v_a_3088_ = lean_ctor_get(v___x_3087_, 0);
lean_inc(v_a_3088_);
lean_dec_ref_known(v___x_3087_, 1);
v___x_3089_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_3034_, v___y_3024_);
if (lean_obj_tag(v___x_3089_) == 0)
{
lean_object* v_a_3090_; lean_object* v___x_3091_; 
v_a_3090_ = lean_ctor_get(v___x_3089_, 0);
lean_inc(v_a_3090_);
lean_dec_ref_known(v___x_3089_, 1);
v___x_3091_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_3042_, v___y_3024_);
if (lean_obj_tag(v___x_3091_) == 0)
{
lean_object* v_a_3092_; lean_object* v___x_3093_; 
v_a_3092_ = lean_ctor_get(v___x_3091_, 0);
lean_inc(v_a_3092_);
lean_dec_ref_known(v___x_3091_, 1);
v___x_3093_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_3045_, v___y_3024_);
if (lean_obj_tag(v___x_3093_) == 0)
{
lean_object* v_a_3094_; lean_object* v___x_3095_; 
v_a_3094_ = lean_ctor_get(v___x_3093_, 0);
lean_inc(v_a_3094_);
lean_dec_ref_known(v___x_3093_, 1);
v___x_3095_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Order_addAtom_spec__0___redArg(v_a_3047_, v___y_3024_);
if (lean_obj_tag(v___x_3095_) == 0)
{
lean_object* v_a_3096_; lean_object* v___x_3098_; uint8_t v_isShared_3099_; uint8_t v_isSharedCheck_3109_; 
v_a_3096_ = lean_ctor_get(v___x_3095_, 0);
v_isSharedCheck_3109_ = !lean_is_exclusive(v___x_3095_);
if (v_isSharedCheck_3109_ == 0)
{
v___x_3098_ = v___x_3095_;
v_isShared_3099_ = v_isSharedCheck_3109_;
goto v_resetjp_3097_;
}
else
{
lean_inc(v_a_3096_);
lean_dec(v___x_3095_);
v___x_3098_ = lean_box(0);
v_isShared_3099_ = v_isSharedCheck_3109_;
goto v_resetjp_3097_;
}
v_resetjp_3097_:
{
lean_object* v___x_3100_; lean_object* v___x_3101_; lean_object* v___x_3102_; lean_object* v___x_3103_; lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3107_; 
v___x_3100_ = lean_box(v_a_3022_);
v___x_3101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3101_, 0, v_a_3096_);
lean_ctor_set(v___x_3101_, 1, v___x_3100_);
v___x_3102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3102_, 0, v_a_3094_);
lean_ctor_set(v___x_3102_, 1, v___x_3101_);
v___x_3103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3103_, 0, v_a_3092_);
lean_ctor_set(v___x_3103_, 1, v___x_3102_);
v___x_3104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3104_, 0, v_a_3090_);
lean_ctor_set(v___x_3104_, 1, v___x_3103_);
v___x_3105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3105_, 0, v_a_3088_);
lean_ctor_set(v___x_3105_, 1, v___x_3104_);
if (v_isShared_3099_ == 0)
{
lean_ctor_set(v___x_3098_, 0, v___x_3105_);
v___x_3107_ = v___x_3098_;
goto v_reusejp_3106_;
}
else
{
lean_object* v_reuseFailAlloc_3108_; 
v_reuseFailAlloc_3108_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3108_, 0, v___x_3105_);
v___x_3107_ = v_reuseFailAlloc_3108_;
goto v_reusejp_3106_;
}
v_reusejp_3106_:
{
return v___x_3107_;
}
}
}
else
{
lean_object* v_a_3110_; lean_object* v___x_3112_; uint8_t v_isShared_3113_; uint8_t v_isSharedCheck_3117_; 
lean_dec(v_a_3094_);
lean_dec(v_a_3092_);
lean_dec(v_a_3090_);
lean_dec(v_a_3088_);
v_a_3110_ = lean_ctor_get(v___x_3095_, 0);
v_isSharedCheck_3117_ = !lean_is_exclusive(v___x_3095_);
if (v_isSharedCheck_3117_ == 0)
{
v___x_3112_ = v___x_3095_;
v_isShared_3113_ = v_isSharedCheck_3117_;
goto v_resetjp_3111_;
}
else
{
lean_inc(v_a_3110_);
lean_dec(v___x_3095_);
v___x_3112_ = lean_box(0);
v_isShared_3113_ = v_isSharedCheck_3117_;
goto v_resetjp_3111_;
}
v_resetjp_3111_:
{
lean_object* v___x_3115_; 
if (v_isShared_3113_ == 0)
{
v___x_3115_ = v___x_3112_;
goto v_reusejp_3114_;
}
else
{
lean_object* v_reuseFailAlloc_3116_; 
v_reuseFailAlloc_3116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3116_, 0, v_a_3110_);
v___x_3115_ = v_reuseFailAlloc_3116_;
goto v_reusejp_3114_;
}
v_reusejp_3114_:
{
return v___x_3115_;
}
}
}
}
else
{
lean_object* v_a_3118_; lean_object* v___x_3120_; uint8_t v_isShared_3121_; uint8_t v_isSharedCheck_3125_; 
lean_dec(v_a_3092_);
lean_dec(v_a_3090_);
lean_dec(v_a_3088_);
lean_dec(v_a_3047_);
v_a_3118_ = lean_ctor_get(v___x_3093_, 0);
v_isSharedCheck_3125_ = !lean_is_exclusive(v___x_3093_);
if (v_isSharedCheck_3125_ == 0)
{
v___x_3120_ = v___x_3093_;
v_isShared_3121_ = v_isSharedCheck_3125_;
goto v_resetjp_3119_;
}
else
{
lean_inc(v_a_3118_);
lean_dec(v___x_3093_);
v___x_3120_ = lean_box(0);
v_isShared_3121_ = v_isSharedCheck_3125_;
goto v_resetjp_3119_;
}
v_resetjp_3119_:
{
lean_object* v___x_3123_; 
if (v_isShared_3121_ == 0)
{
v___x_3123_ = v___x_3120_;
goto v_reusejp_3122_;
}
else
{
lean_object* v_reuseFailAlloc_3124_; 
v_reuseFailAlloc_3124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3124_, 0, v_a_3118_);
v___x_3123_ = v_reuseFailAlloc_3124_;
goto v_reusejp_3122_;
}
v_reusejp_3122_:
{
return v___x_3123_;
}
}
}
}
else
{
lean_object* v_a_3126_; lean_object* v___x_3128_; uint8_t v_isShared_3129_; uint8_t v_isSharedCheck_3133_; 
lean_dec(v_a_3090_);
lean_dec(v_a_3088_);
lean_dec(v_a_3047_);
lean_dec(v_a_3045_);
v_a_3126_ = lean_ctor_get(v___x_3091_, 0);
v_isSharedCheck_3133_ = !lean_is_exclusive(v___x_3091_);
if (v_isSharedCheck_3133_ == 0)
{
v___x_3128_ = v___x_3091_;
v_isShared_3129_ = v_isSharedCheck_3133_;
goto v_resetjp_3127_;
}
else
{
lean_inc(v_a_3126_);
lean_dec(v___x_3091_);
v___x_3128_ = lean_box(0);
v_isShared_3129_ = v_isSharedCheck_3133_;
goto v_resetjp_3127_;
}
v_resetjp_3127_:
{
lean_object* v___x_3131_; 
if (v_isShared_3129_ == 0)
{
v___x_3131_ = v___x_3128_;
goto v_reusejp_3130_;
}
else
{
lean_object* v_reuseFailAlloc_3132_; 
v_reuseFailAlloc_3132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3132_, 0, v_a_3126_);
v___x_3131_ = v_reuseFailAlloc_3132_;
goto v_reusejp_3130_;
}
v_reusejp_3130_:
{
return v___x_3131_;
}
}
}
}
else
{
lean_object* v_a_3134_; lean_object* v___x_3136_; uint8_t v_isShared_3137_; uint8_t v_isSharedCheck_3141_; 
lean_dec(v_a_3088_);
lean_dec(v_a_3047_);
lean_dec(v_a_3045_);
lean_dec(v_a_3042_);
v_a_3134_ = lean_ctor_get(v___x_3089_, 0);
v_isSharedCheck_3141_ = !lean_is_exclusive(v___x_3089_);
if (v_isSharedCheck_3141_ == 0)
{
v___x_3136_ = v___x_3089_;
v_isShared_3137_ = v_isSharedCheck_3141_;
goto v_resetjp_3135_;
}
else
{
lean_inc(v_a_3134_);
lean_dec(v___x_3089_);
v___x_3136_ = lean_box(0);
v_isShared_3137_ = v_isSharedCheck_3141_;
goto v_resetjp_3135_;
}
v_resetjp_3135_:
{
lean_object* v___x_3139_; 
if (v_isShared_3137_ == 0)
{
v___x_3139_ = v___x_3136_;
goto v_reusejp_3138_;
}
else
{
lean_object* v_reuseFailAlloc_3140_; 
v_reuseFailAlloc_3140_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3140_, 0, v_a_3134_);
v___x_3139_ = v_reuseFailAlloc_3140_;
goto v_reusejp_3138_;
}
v_reusejp_3138_:
{
return v___x_3139_;
}
}
}
}
else
{
lean_object* v_a_3142_; lean_object* v___x_3144_; uint8_t v_isShared_3145_; uint8_t v_isSharedCheck_3149_; 
lean_dec(v_a_3047_);
lean_dec(v_a_3045_);
lean_dec(v_a_3042_);
lean_dec(v_a_3034_);
v_a_3142_ = lean_ctor_get(v___x_3087_, 0);
v_isSharedCheck_3149_ = !lean_is_exclusive(v___x_3087_);
if (v_isSharedCheck_3149_ == 0)
{
v___x_3144_ = v___x_3087_;
v_isShared_3145_ = v_isSharedCheck_3149_;
goto v_resetjp_3143_;
}
else
{
lean_inc(v_a_3142_);
lean_dec(v___x_3087_);
v___x_3144_ = lean_box(0);
v_isShared_3145_ = v_isSharedCheck_3149_;
goto v_resetjp_3143_;
}
v_resetjp_3143_:
{
lean_object* v___x_3147_; 
if (v_isShared_3145_ == 0)
{
v___x_3147_ = v___x_3144_;
goto v_reusejp_3146_;
}
else
{
lean_object* v_reuseFailAlloc_3148_; 
v_reuseFailAlloc_3148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3148_, 0, v_a_3142_);
v___x_3147_ = v_reuseFailAlloc_3148_;
goto v_reusejp_3146_;
}
v_reusejp_3146_:
{
return v___x_3147_;
}
}
}
}
}
}
else
{
lean_object* v_a_3151_; lean_object* v___x_3153_; uint8_t v_isShared_3154_; uint8_t v_isSharedCheck_3158_; 
lean_dec(v_a_3047_);
lean_dec(v_a_3045_);
lean_dec(v_a_3042_);
lean_dec(v_a_3034_);
lean_dec(v_a_3029_);
v_a_3151_ = lean_ctor_get(v___x_3072_, 0);
v_isSharedCheck_3158_ = !lean_is_exclusive(v___x_3072_);
if (v_isSharedCheck_3158_ == 0)
{
v___x_3153_ = v___x_3072_;
v_isShared_3154_ = v_isSharedCheck_3158_;
goto v_resetjp_3152_;
}
else
{
lean_inc(v_a_3151_);
lean_dec(v___x_3072_);
v___x_3153_ = lean_box(0);
v_isShared_3154_ = v_isSharedCheck_3158_;
goto v_resetjp_3152_;
}
v_resetjp_3152_:
{
lean_object* v___x_3156_; 
if (v_isShared_3154_ == 0)
{
v___x_3156_ = v___x_3153_;
goto v_reusejp_3155_;
}
else
{
lean_object* v_reuseFailAlloc_3157_; 
v_reuseFailAlloc_3157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3157_, 0, v_a_3151_);
v___x_3156_ = v_reuseFailAlloc_3157_;
goto v_reusejp_3155_;
}
v_reusejp_3155_:
{
return v___x_3156_;
}
}
}
}
}
}
else
{
lean_object* v_a_3161_; lean_object* v___x_3163_; uint8_t v_isShared_3164_; uint8_t v_isSharedCheck_3168_; 
lean_dec(v_a_3045_);
lean_dec(v_a_3042_);
lean_dec_ref_known(v___x_3037_, 2);
lean_dec(v_a_3034_);
lean_dec(v_a_3029_);
lean_dec_ref(v___y_3023_);
lean_dec(v_fst_3020_);
v_a_3161_ = lean_ctor_get(v___x_3046_, 0);
v_isSharedCheck_3168_ = !lean_is_exclusive(v___x_3046_);
if (v_isSharedCheck_3168_ == 0)
{
v___x_3163_ = v___x_3046_;
v_isShared_3164_ = v_isSharedCheck_3168_;
goto v_resetjp_3162_;
}
else
{
lean_inc(v_a_3161_);
lean_dec(v___x_3046_);
v___x_3163_ = lean_box(0);
v_isShared_3164_ = v_isSharedCheck_3168_;
goto v_resetjp_3162_;
}
v_resetjp_3162_:
{
lean_object* v___x_3166_; 
if (v_isShared_3164_ == 0)
{
v___x_3166_ = v___x_3163_;
goto v_reusejp_3165_;
}
else
{
lean_object* v_reuseFailAlloc_3167_; 
v_reuseFailAlloc_3167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3167_, 0, v_a_3161_);
v___x_3166_ = v_reuseFailAlloc_3167_;
goto v_reusejp_3165_;
}
v_reusejp_3165_:
{
return v___x_3166_;
}
}
}
}
else
{
lean_object* v_a_3169_; lean_object* v___x_3171_; uint8_t v_isShared_3172_; uint8_t v_isSharedCheck_3176_; 
lean_dec_ref_known(v___x_3043_, 1);
lean_dec(v_a_3042_);
lean_dec_ref_known(v___x_3037_, 2);
lean_dec(v_a_3034_);
lean_dec(v_a_3029_);
lean_dec_ref(v___y_3023_);
lean_dec(v_fst_3020_);
lean_dec(v___x_3019_);
v_a_3169_ = lean_ctor_get(v___x_3044_, 0);
v_isSharedCheck_3176_ = !lean_is_exclusive(v___x_3044_);
if (v_isSharedCheck_3176_ == 0)
{
v___x_3171_ = v___x_3044_;
v_isShared_3172_ = v_isSharedCheck_3176_;
goto v_resetjp_3170_;
}
else
{
lean_inc(v_a_3169_);
lean_dec(v___x_3044_);
v___x_3171_ = lean_box(0);
v_isShared_3172_ = v_isSharedCheck_3176_;
goto v_resetjp_3170_;
}
v_resetjp_3170_:
{
lean_object* v___x_3174_; 
if (v_isShared_3172_ == 0)
{
v___x_3174_ = v___x_3171_;
goto v_reusejp_3173_;
}
else
{
lean_object* v_reuseFailAlloc_3175_; 
v_reuseFailAlloc_3175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3175_, 0, v_a_3169_);
v___x_3174_ = v_reuseFailAlloc_3175_;
goto v_reusejp_3173_;
}
v_reusejp_3173_:
{
return v___x_3174_;
}
}
}
}
else
{
lean_object* v_a_3177_; lean_object* v___x_3179_; uint8_t v_isShared_3180_; uint8_t v_isSharedCheck_3184_; 
lean_dec_ref_known(v___x_3037_, 2);
lean_dec(v_a_3034_);
lean_dec(v_a_3029_);
lean_dec_ref(v___y_3023_);
lean_dec(v_fst_3020_);
lean_dec(v___x_3019_);
v_a_3177_ = lean_ctor_get(v___x_3041_, 0);
v_isSharedCheck_3184_ = !lean_is_exclusive(v___x_3041_);
if (v_isSharedCheck_3184_ == 0)
{
v___x_3179_ = v___x_3041_;
v_isShared_3180_ = v_isSharedCheck_3184_;
goto v_resetjp_3178_;
}
else
{
lean_inc(v_a_3177_);
lean_dec(v___x_3041_);
v___x_3179_ = lean_box(0);
v_isShared_3180_ = v_isSharedCheck_3184_;
goto v_resetjp_3178_;
}
v_resetjp_3178_:
{
lean_object* v___x_3182_; 
if (v_isShared_3180_ == 0)
{
v___x_3182_ = v___x_3179_;
goto v_reusejp_3181_;
}
else
{
lean_object* v_reuseFailAlloc_3183_; 
v_reuseFailAlloc_3183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3183_, 0, v_a_3177_);
v___x_3182_ = v_reuseFailAlloc_3183_;
goto v_reusejp_3181_;
}
v_reusejp_3181_:
{
return v___x_3182_;
}
}
}
}
else
{
lean_object* v_a_3185_; lean_object* v___x_3187_; uint8_t v_isShared_3188_; uint8_t v_isSharedCheck_3192_; 
lean_dec(v_a_3029_);
lean_dec_ref(v___y_3023_);
lean_dec(v_fst_3020_);
lean_dec(v___x_3019_);
v_a_3185_ = lean_ctor_get(v___x_3033_, 0);
v_isSharedCheck_3192_ = !lean_is_exclusive(v___x_3033_);
if (v_isSharedCheck_3192_ == 0)
{
v___x_3187_ = v___x_3033_;
v_isShared_3188_ = v_isSharedCheck_3192_;
goto v_resetjp_3186_;
}
else
{
lean_inc(v_a_3185_);
lean_dec(v___x_3033_);
v___x_3187_ = lean_box(0);
v_isShared_3188_ = v_isSharedCheck_3192_;
goto v_resetjp_3186_;
}
v_resetjp_3186_:
{
lean_object* v___x_3190_; 
if (v_isShared_3188_ == 0)
{
v___x_3190_ = v___x_3187_;
goto v_reusejp_3189_;
}
else
{
lean_object* v_reuseFailAlloc_3191_; 
v_reuseFailAlloc_3191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3191_, 0, v_a_3185_);
v___x_3190_ = v_reuseFailAlloc_3191_;
goto v_reusejp_3189_;
}
v_reusejp_3189_:
{
return v___x_3190_;
}
}
}
}
else
{
lean_object* v_a_3193_; lean_object* v___x_3195_; uint8_t v_isShared_3196_; uint8_t v_isSharedCheck_3200_; 
lean_dec_ref(v___y_3023_);
lean_dec(v_fst_3020_);
lean_dec(v___x_3019_);
v_a_3193_ = lean_ctor_get(v___x_3028_, 0);
v_isSharedCheck_3200_ = !lean_is_exclusive(v___x_3028_);
if (v_isSharedCheck_3200_ == 0)
{
v___x_3195_ = v___x_3028_;
v_isShared_3196_ = v_isSharedCheck_3200_;
goto v_resetjp_3194_;
}
else
{
lean_inc(v_a_3193_);
lean_dec(v___x_3028_);
v___x_3195_ = lean_box(0);
v_isShared_3196_ = v_isSharedCheck_3200_;
goto v_resetjp_3194_;
}
v_resetjp_3194_:
{
lean_object* v___x_3198_; 
if (v_isShared_3196_ == 0)
{
v___x_3198_ = v___x_3195_;
goto v_reusejp_3197_;
}
else
{
lean_object* v_reuseFailAlloc_3199_; 
v_reuseFailAlloc_3199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3199_, 0, v_a_3193_);
v___x_3198_ = v_reuseFailAlloc_3199_;
goto v_reusejp_3197_;
}
v_reusejp_3197_:
{
return v___x_3198_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__8___boxed(lean_object* v___x_3201_, lean_object* v___x_3202_, lean_object* v_fst_3203_, lean_object* v___x_3204_, lean_object* v_a_3205_, lean_object* v___y_3206_, lean_object* v___y_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_, lean_object* v___y_3210_){
_start:
{
uint8_t v___x_192574__boxed_3211_; uint8_t v___x_192576__boxed_3212_; uint8_t v_a_192577__boxed_3213_; lean_object* v_res_3214_; 
v___x_192574__boxed_3211_ = lean_unbox(v___x_3201_);
v___x_192576__boxed_3212_ = lean_unbox(v___x_3204_);
v_a_192577__boxed_3213_ = lean_unbox(v_a_3205_);
v_res_3214_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__8(v___x_192574__boxed_3211_, v___x_3202_, v_fst_3203_, v___x_192576__boxed_3212_, v_a_192577__boxed_3213_, v___y_3206_, v___y_3207_, v___y_3208_, v___y_3209_);
lean_dec(v___y_3209_);
lean_dec_ref(v___y_3208_);
lean_dec(v___y_3207_);
return v_res_3214_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__0(void){
_start:
{
lean_object* v___x_3215_; lean_object* v___x_3216_; 
v___x_3215_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0, &lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___closed__0);
v___x_3216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3216_, 0, v___x_3215_);
return v___x_3216_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__5(void){
_start:
{
lean_object* v___x_3225_; lean_object* v___x_3226_; lean_object* v___x_3227_; 
v___x_3225_ = lean_box(0);
v___x_3226_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__4));
v___x_3227_ = l_Lean_Expr_const___override(v___x_3226_, v___x_3225_);
return v___x_3227_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__8(void){
_start:
{
lean_object* v___x_3232_; lean_object* v___x_3233_; lean_object* v___x_3234_; 
v___x_3232_ = lean_box(0);
v___x_3233_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__7));
v___x_3234_ = l_Lean_Expr_const___override(v___x_3233_, v___x_3232_);
return v___x_3234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(lean_object* v_expr_3238_, lean_object* v_a_3239_, lean_object* v_a_3240_, lean_object* v_a_3241_, lean_object* v_a_3242_, lean_object* v_a_3243_, lean_object* v_a_3244_, lean_object* v_a_3245_){
_start:
{
lean_object* v___x_3247_; 
lean_inc(v_a_3245_);
lean_inc_ref(v_a_3244_);
lean_inc(v_a_3243_);
lean_inc_ref(v_a_3242_);
lean_inc_ref(v_expr_3238_);
v___x_3247_ = lean_infer_type(v_expr_3238_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3247_) == 0)
{
lean_object* v_a_3248_; lean_object* v___x_3249_; 
v_a_3248_ = lean_ctor_get(v___x_3247_, 0);
lean_inc(v_a_3248_);
lean_dec_ref_known(v___x_3247_, 1);
v___x_3249_ = l_Lean_Meta_isProp(v_a_3248_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3249_) == 0)
{
lean_object* v_a_3250_; lean_object* v___x_3252_; uint8_t v_isShared_3253_; uint8_t v_isSharedCheck_3828_; 
v_a_3250_ = lean_ctor_get(v___x_3249_, 0);
v_isSharedCheck_3828_ = !lean_is_exclusive(v___x_3249_);
if (v_isSharedCheck_3828_ == 0)
{
v___x_3252_ = v___x_3249_;
v_isShared_3253_ = v_isSharedCheck_3828_;
goto v_resetjp_3251_;
}
else
{
lean_inc(v_a_3250_);
lean_dec(v___x_3249_);
v___x_3252_ = lean_box(0);
v_isShared_3253_ = v_isSharedCheck_3828_;
goto v_resetjp_3251_;
}
v_resetjp_3251_:
{
uint8_t v___x_3254_; 
v___x_3254_ = lean_unbox(v_a_3250_);
if (v___x_3254_ == 0)
{
lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3258_; 
lean_dec(v_a_3250_);
lean_dec_ref(v_expr_3238_);
v___x_3255_ = lean_box(0);
v___x_3256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3256_, 0, v___x_3255_);
lean_ctor_set(v___x_3256_, 1, v_a_3239_);
if (v_isShared_3253_ == 0)
{
lean_ctor_set(v___x_3252_, 0, v___x_3256_);
v___x_3258_ = v___x_3252_;
goto v_reusejp_3257_;
}
else
{
lean_object* v_reuseFailAlloc_3259_; 
v_reuseFailAlloc_3259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3259_, 0, v___x_3256_);
v___x_3258_ = v_reuseFailAlloc_3259_;
goto v_reusejp_3257_;
}
v_reusejp_3257_:
{
return v___x_3258_;
}
}
else
{
lean_object* v___x_3260_; 
lean_del_object(v___x_3252_);
v___x_3260_ = lp_Qq_Qq_inferTypeQ(v_expr_3238_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3260_) == 0)
{
lean_object* v_a_3261_; lean_object* v_snd_3262_; lean_object* v_fst_3263_; lean_object* v_snd_3264_; uint8_t v___x_3265_; uint8_t v___x_3266_; lean_object* v___x_3267_; lean_object* v___x_3268_; lean_object* v___x_3269_; lean_object* v___f_3270_; lean_object* v___x_3271_; 
v_a_3261_ = lean_ctor_get(v___x_3260_, 0);
lean_inc(v_a_3261_);
lean_dec_ref_known(v___x_3260_, 1);
v_snd_3262_ = lean_ctor_get(v_a_3261_, 1);
lean_inc(v_snd_3262_);
lean_dec(v_a_3261_);
v_fst_3263_ = lean_ctor_get(v_snd_3262_, 0);
lean_inc_n(v_fst_3263_, 2);
v_snd_3264_ = lean_ctor_get(v_snd_3262_, 1);
lean_inc(v_snd_3264_);
lean_dec(v_snd_3262_);
v___x_3265_ = 0;
v___x_3266_ = 0;
v___x_3267_ = lean_box(0);
v___x_3268_ = lean_box(v___x_3266_);
v___x_3269_ = lean_box(v___x_3265_);
lean_inc(v_a_3250_);
v___f_3270_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__0___boxed), 10, 5);
lean_closure_set(v___f_3270_, 0, v___x_3268_);
lean_closure_set(v___f_3270_, 1, v___x_3267_);
lean_closure_set(v___f_3270_, 2, v_fst_3263_);
lean_closure_set(v___f_3270_, 3, v___x_3269_);
lean_closure_set(v___f_3270_, 4, v_a_3250_);
v___x_3271_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3270_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3271_) == 0)
{
lean_object* v_a_3272_; lean_object* v_snd_3273_; lean_object* v_snd_3274_; lean_object* v_snd_3275_; lean_object* v_fst_3276_; lean_object* v_fst_3277_; lean_object* v_fst_3278_; lean_object* v___x_3280_; uint8_t v_isShared_3281_; uint8_t v_isSharedCheck_3810_; 
v_a_3272_ = lean_ctor_get(v___x_3271_, 0);
lean_inc(v_a_3272_);
lean_dec_ref_known(v___x_3271_, 1);
v_snd_3273_ = lean_ctor_get(v_a_3272_, 1);
lean_inc(v_snd_3273_);
v_snd_3274_ = lean_ctor_get(v_snd_3273_, 1);
lean_inc(v_snd_3274_);
v_snd_3275_ = lean_ctor_get(v_snd_3274_, 1);
lean_inc(v_snd_3275_);
v_fst_3276_ = lean_ctor_get(v_a_3272_, 0);
lean_inc(v_fst_3276_);
lean_dec(v_a_3272_);
v_fst_3277_ = lean_ctor_get(v_snd_3273_, 0);
lean_inc(v_fst_3277_);
lean_dec(v_snd_3273_);
v_fst_3278_ = lean_ctor_get(v_snd_3274_, 0);
v_isSharedCheck_3810_ = !lean_is_exclusive(v_snd_3274_);
if (v_isSharedCheck_3810_ == 0)
{
lean_object* v_unused_3811_; 
v_unused_3811_ = lean_ctor_get(v_snd_3274_, 1);
lean_dec(v_unused_3811_);
v___x_3280_ = v_snd_3274_;
v_isShared_3281_ = v_isSharedCheck_3810_;
goto v_resetjp_3279_;
}
else
{
lean_inc(v_fst_3278_);
lean_dec(v_snd_3274_);
v___x_3280_ = lean_box(0);
v_isShared_3281_ = v_isSharedCheck_3810_;
goto v_resetjp_3279_;
}
v_resetjp_3279_:
{
lean_object* v_fst_3282_; lean_object* v_snd_3283_; lean_object* v___x_3285_; uint8_t v_isShared_3286_; uint8_t v_isSharedCheck_3809_; 
v_fst_3282_ = lean_ctor_get(v_snd_3275_, 0);
v_snd_3283_ = lean_ctor_get(v_snd_3275_, 1);
v_isSharedCheck_3809_ = !lean_is_exclusive(v_snd_3275_);
if (v_isSharedCheck_3809_ == 0)
{
v___x_3285_ = v_snd_3275_;
v_isShared_3286_ = v_isSharedCheck_3809_;
goto v_resetjp_3284_;
}
else
{
lean_inc(v_snd_3283_);
lean_inc(v_fst_3282_);
lean_dec(v_snd_3275_);
v___x_3285_ = lean_box(0);
v_isShared_3286_ = v_isSharedCheck_3809_;
goto v_resetjp_3284_;
}
v_resetjp_3284_:
{
lean_object* v___x_3287_; uint8_t v___x_3288_; 
v___x_3287_ = lean_box(0);
v___x_3288_ = lean_unbox(v_snd_3283_);
lean_dec(v_snd_3283_);
if (v___x_3288_ == 0)
{
lean_object* v___x_3289_; lean_object* v___x_3290_; lean_object* v___f_3291_; lean_object* v___x_3292_; 
lean_del_object(v___x_3285_);
lean_dec(v_fst_3282_);
lean_del_object(v___x_3280_);
lean_dec(v_fst_3278_);
lean_dec(v_fst_3277_);
lean_dec(v_fst_3276_);
v___x_3289_ = lean_box(v___x_3266_);
v___x_3290_ = lean_box(v___x_3265_);
lean_inc(v_a_3250_);
lean_inc(v_fst_3263_);
v___f_3291_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__1___boxed), 10, 5);
lean_closure_set(v___f_3291_, 0, v___x_3289_);
lean_closure_set(v___f_3291_, 1, v___x_3267_);
lean_closure_set(v___f_3291_, 2, v_fst_3263_);
lean_closure_set(v___f_3291_, 3, v___x_3290_);
lean_closure_set(v___f_3291_, 4, v_a_3250_);
v___x_3292_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3291_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3292_) == 0)
{
lean_object* v_a_3293_; lean_object* v_snd_3294_; lean_object* v_snd_3295_; lean_object* v_snd_3296_; lean_object* v_snd_3297_; lean_object* v_snd_3298_; uint8_t v___x_3299_; 
v_a_3293_ = lean_ctor_get(v___x_3292_, 0);
lean_inc(v_a_3293_);
lean_dec_ref_known(v___x_3292_, 1);
v_snd_3294_ = lean_ctor_get(v_a_3293_, 1);
lean_inc(v_snd_3294_);
v_snd_3295_ = lean_ctor_get(v_snd_3294_, 1);
v_snd_3296_ = lean_ctor_get(v_snd_3295_, 1);
lean_inc(v_snd_3296_);
v_snd_3297_ = lean_ctor_get(v_snd_3296_, 1);
lean_inc(v_snd_3297_);
v_snd_3298_ = lean_ctor_get(v_snd_3297_, 1);
v___x_3299_ = lean_unbox(v_snd_3298_);
if (v___x_3299_ == 0)
{
lean_object* v___x_3300_; lean_object* v___x_3301_; lean_object* v___f_3302_; lean_object* v___x_3303_; 
lean_dec(v_snd_3297_);
lean_dec(v_snd_3296_);
lean_dec(v_snd_3294_);
lean_dec(v_a_3293_);
v___x_3300_ = lean_box(v___x_3266_);
v___x_3301_ = lean_box(v___x_3265_);
lean_inc(v_a_3250_);
lean_inc(v_fst_3263_);
v___f_3302_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__2___boxed), 10, 5);
lean_closure_set(v___f_3302_, 0, v___x_3300_);
lean_closure_set(v___f_3302_, 1, v___x_3267_);
lean_closure_set(v___f_3302_, 2, v_fst_3263_);
lean_closure_set(v___f_3302_, 3, v___x_3301_);
lean_closure_set(v___f_3302_, 4, v_a_3250_);
v___x_3303_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3302_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3303_) == 0)
{
lean_object* v_a_3304_; lean_object* v_snd_3305_; lean_object* v_snd_3306_; lean_object* v_snd_3307_; lean_object* v_snd_3308_; lean_object* v_snd_3309_; uint8_t v___x_3310_; 
v_a_3304_ = lean_ctor_get(v___x_3303_, 0);
lean_inc(v_a_3304_);
lean_dec_ref_known(v___x_3303_, 1);
v_snd_3305_ = lean_ctor_get(v_a_3304_, 1);
lean_inc(v_snd_3305_);
v_snd_3306_ = lean_ctor_get(v_snd_3305_, 1);
v_snd_3307_ = lean_ctor_get(v_snd_3306_, 1);
lean_inc(v_snd_3307_);
v_snd_3308_ = lean_ctor_get(v_snd_3307_, 1);
lean_inc(v_snd_3308_);
v_snd_3309_ = lean_ctor_get(v_snd_3308_, 1);
v___x_3310_ = lean_unbox(v_snd_3309_);
if (v___x_3310_ == 0)
{
lean_object* v___x_3311_; lean_object* v___x_3312_; lean_object* v___f_3313_; lean_object* v___x_3314_; 
lean_dec(v_snd_3308_);
lean_dec(v_snd_3307_);
lean_dec(v_snd_3305_);
lean_dec(v_a_3304_);
v___x_3311_ = lean_box(v___x_3266_);
v___x_3312_ = lean_box(v___x_3265_);
lean_inc(v_a_3250_);
lean_inc(v_fst_3263_);
v___f_3313_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__3___boxed), 10, 5);
lean_closure_set(v___f_3313_, 0, v___x_3311_);
lean_closure_set(v___f_3313_, 1, v___x_3267_);
lean_closure_set(v___f_3313_, 2, v_fst_3263_);
lean_closure_set(v___f_3313_, 3, v___x_3312_);
lean_closure_set(v___f_3313_, 4, v_a_3250_);
v___x_3314_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3313_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3314_) == 0)
{
lean_object* v_a_3315_; lean_object* v_snd_3316_; lean_object* v_snd_3317_; lean_object* v_snd_3318_; lean_object* v_snd_3319_; uint8_t v___x_3320_; 
v_a_3315_ = lean_ctor_get(v___x_3314_, 0);
lean_inc(v_a_3315_);
lean_dec_ref_known(v___x_3314_, 1);
v_snd_3316_ = lean_ctor_get(v_a_3315_, 1);
lean_inc(v_snd_3316_);
v_snd_3317_ = lean_ctor_get(v_snd_3316_, 1);
lean_inc(v_snd_3317_);
v_snd_3318_ = lean_ctor_get(v_snd_3317_, 1);
lean_inc(v_snd_3318_);
v_snd_3319_ = lean_ctor_get(v_snd_3318_, 1);
v___x_3320_ = lean_unbox(v_snd_3319_);
if (v___x_3320_ == 0)
{
lean_object* v___x_3321_; lean_object* v___x_3322_; lean_object* v___x_3323_; lean_object* v___f_3324_; lean_object* v___x_3325_; 
lean_dec(v_snd_3318_);
lean_dec(v_snd_3317_);
lean_dec(v_snd_3316_);
lean_dec(v_a_3315_);
v___x_3321_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__0, &lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__0);
v___x_3322_ = lean_box(v___x_3266_);
v___x_3323_ = lean_box(v___x_3265_);
lean_inc(v_a_3250_);
lean_inc(v_fst_3263_);
v___f_3324_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__4___boxed), 11, 6);
lean_closure_set(v___f_3324_, 0, v___x_3321_);
lean_closure_set(v___f_3324_, 1, v___x_3322_);
lean_closure_set(v___f_3324_, 2, v___x_3267_);
lean_closure_set(v___f_3324_, 3, v_fst_3263_);
lean_closure_set(v___f_3324_, 4, v___x_3323_);
lean_closure_set(v___f_3324_, 5, v_a_3250_);
v___x_3325_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3324_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3325_) == 0)
{
lean_object* v_a_3326_; lean_object* v_snd_3327_; uint8_t v___x_3328_; 
v_a_3326_ = lean_ctor_get(v___x_3325_, 0);
lean_inc(v_a_3326_);
lean_dec_ref_known(v___x_3325_, 1);
v_snd_3327_ = lean_ctor_get(v_a_3326_, 1);
v___x_3328_ = lean_unbox(v_snd_3327_);
if (v___x_3328_ == 0)
{
lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___f_3331_; lean_object* v___x_3332_; 
lean_dec(v_a_3326_);
v___x_3329_ = lean_box(v___x_3266_);
v___x_3330_ = lean_box(v___x_3265_);
lean_inc(v_a_3250_);
lean_inc(v_fst_3263_);
v___f_3331_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__5___boxed), 11, 6);
lean_closure_set(v___f_3331_, 0, v___x_3321_);
lean_closure_set(v___f_3331_, 1, v___x_3329_);
lean_closure_set(v___f_3331_, 2, v___x_3267_);
lean_closure_set(v___f_3331_, 3, v_fst_3263_);
lean_closure_set(v___f_3331_, 4, v___x_3330_);
lean_closure_set(v___f_3331_, 5, v_a_3250_);
v___x_3332_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3331_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3332_) == 0)
{
lean_object* v_a_3333_; lean_object* v_snd_3334_; lean_object* v_snd_3335_; uint8_t v___x_3336_; 
v_a_3333_ = lean_ctor_get(v___x_3332_, 0);
lean_inc(v_a_3333_);
lean_dec_ref_known(v___x_3332_, 1);
v_snd_3334_ = lean_ctor_get(v_a_3333_, 1);
lean_inc(v_snd_3334_);
v_snd_3335_ = lean_ctor_get(v_snd_3334_, 1);
v___x_3336_ = lean_unbox(v_snd_3335_);
if (v___x_3336_ == 0)
{
lean_object* v___f_3337_; lean_object* v___x_3338_; 
lean_dec(v_snd_3334_);
lean_dec(v_a_3333_);
v___f_3337_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__6___boxed), 7, 2);
lean_closure_set(v___f_3337_, 0, v_fst_3263_);
lean_closure_set(v___f_3337_, 1, v_a_3250_);
v___x_3338_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3337_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3338_) == 0)
{
lean_object* v_a_3339_; lean_object* v___x_3341_; uint8_t v_isShared_3342_; uint8_t v_isSharedCheck_3377_; 
v_a_3339_ = lean_ctor_get(v___x_3338_, 0);
v_isSharedCheck_3377_ = !lean_is_exclusive(v___x_3338_);
if (v_isSharedCheck_3377_ == 0)
{
v___x_3341_ = v___x_3338_;
v_isShared_3342_ = v_isSharedCheck_3377_;
goto v_resetjp_3340_;
}
else
{
lean_inc(v_a_3339_);
lean_dec(v___x_3338_);
v___x_3341_ = lean_box(0);
v_isShared_3342_ = v_isSharedCheck_3377_;
goto v_resetjp_3340_;
}
v_resetjp_3340_:
{
lean_object* v_snd_3343_; lean_object* v_snd_3344_; lean_object* v_snd_3345_; uint8_t v___x_3346_; 
v_snd_3343_ = lean_ctor_get(v_a_3339_, 1);
lean_inc(v_snd_3343_);
v_snd_3344_ = lean_ctor_get(v_snd_3343_, 1);
lean_inc(v_snd_3344_);
v_snd_3345_ = lean_ctor_get(v_snd_3344_, 1);
v___x_3346_ = lean_unbox(v_snd_3345_);
if (v___x_3346_ == 0)
{
lean_object* v___x_3348_; uint8_t v_isShared_3349_; uint8_t v_isSharedCheck_3356_; 
lean_dec(v_snd_3343_);
lean_dec(v_a_3339_);
lean_dec(v_snd_3264_);
v_isSharedCheck_3356_ = !lean_is_exclusive(v_snd_3344_);
if (v_isSharedCheck_3356_ == 0)
{
lean_object* v_unused_3357_; lean_object* v_unused_3358_; 
v_unused_3357_ = lean_ctor_get(v_snd_3344_, 1);
lean_dec(v_unused_3357_);
v_unused_3358_ = lean_ctor_get(v_snd_3344_, 0);
lean_dec(v_unused_3358_);
v___x_3348_ = v_snd_3344_;
v_isShared_3349_ = v_isSharedCheck_3356_;
goto v_resetjp_3347_;
}
else
{
lean_dec(v_snd_3344_);
v___x_3348_ = lean_box(0);
v_isShared_3349_ = v_isSharedCheck_3356_;
goto v_resetjp_3347_;
}
v_resetjp_3347_:
{
lean_object* v___x_3351_; 
if (v_isShared_3349_ == 0)
{
lean_ctor_set(v___x_3348_, 1, v_a_3239_);
lean_ctor_set(v___x_3348_, 0, v___x_3287_);
v___x_3351_ = v___x_3348_;
goto v_reusejp_3350_;
}
else
{
lean_object* v_reuseFailAlloc_3355_; 
v_reuseFailAlloc_3355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3355_, 0, v___x_3287_);
lean_ctor_set(v_reuseFailAlloc_3355_, 1, v_a_3239_);
v___x_3351_ = v_reuseFailAlloc_3355_;
goto v_reusejp_3350_;
}
v_reusejp_3350_:
{
lean_object* v___x_3353_; 
if (v_isShared_3342_ == 0)
{
lean_ctor_set(v___x_3341_, 0, v___x_3351_);
v___x_3353_ = v___x_3341_;
goto v_reusejp_3352_;
}
else
{
lean_object* v_reuseFailAlloc_3354_; 
v_reuseFailAlloc_3354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3354_, 0, v___x_3351_);
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
lean_object* v_fst_3359_; lean_object* v_fst_3360_; lean_object* v___x_3362_; uint8_t v_isShared_3363_; uint8_t v_isSharedCheck_3375_; 
lean_del_object(v___x_3341_);
v_fst_3359_ = lean_ctor_get(v_a_3339_, 0);
lean_inc(v_fst_3359_);
lean_dec(v_a_3339_);
v_fst_3360_ = lean_ctor_get(v_snd_3343_, 0);
v_isSharedCheck_3375_ = !lean_is_exclusive(v_snd_3343_);
if (v_isSharedCheck_3375_ == 0)
{
lean_object* v_unused_3376_; 
v_unused_3376_ = lean_ctor_get(v_snd_3343_, 1);
lean_dec(v_unused_3376_);
v___x_3362_ = v_snd_3343_;
v_isShared_3363_ = v_isSharedCheck_3375_;
goto v_resetjp_3361_;
}
else
{
lean_inc(v_fst_3360_);
lean_dec(v_snd_3343_);
v___x_3362_ = lean_box(0);
v_isShared_3363_ = v_isSharedCheck_3375_;
goto v_resetjp_3361_;
}
v_resetjp_3361_:
{
lean_object* v_fst_3364_; lean_object* v___x_3365_; lean_object* v___x_3367_; 
v_fst_3364_ = lean_ctor_get(v_snd_3344_, 0);
lean_inc(v_fst_3364_);
lean_dec(v_snd_3344_);
v___x_3365_ = lean_box(0);
if (v_isShared_3363_ == 0)
{
lean_ctor_set_tag(v___x_3362_, 1);
lean_ctor_set(v___x_3362_, 1, v___x_3365_);
lean_ctor_set(v___x_3362_, 0, v_fst_3359_);
v___x_3367_ = v___x_3362_;
goto v_reusejp_3366_;
}
else
{
lean_object* v_reuseFailAlloc_3374_; 
v_reuseFailAlloc_3374_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3374_, 0, v_fst_3359_);
lean_ctor_set(v_reuseFailAlloc_3374_, 1, v___x_3365_);
v___x_3367_ = v_reuseFailAlloc_3374_;
goto v_reusejp_3366_;
}
v_reusejp_3366_:
{
lean_object* v___x_3368_; lean_object* v___x_3369_; lean_object* v___x_3370_; lean_object* v___x_3371_; lean_object* v___x_3372_; 
v___x_3368_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__2));
v___x_3369_ = l_Lean_Expr_const___override(v___x_3368_, v___x_3367_);
v___x_3370_ = l_Lean_Expr_app___override(v___x_3369_, v_fst_3360_);
v___x_3371_ = l_Lean_Expr_app___override(v___x_3370_, v_fst_3364_);
v___x_3372_ = l_Lean_Expr_app___override(v___x_3371_, v_snd_3264_);
v_expr_3238_ = v___x_3372_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_3378_; lean_object* v___x_3380_; uint8_t v_isShared_3381_; uint8_t v_isSharedCheck_3385_; 
lean_dec(v_snd_3264_);
lean_dec_ref(v_a_3239_);
v_a_3378_ = lean_ctor_get(v___x_3338_, 0);
v_isSharedCheck_3385_ = !lean_is_exclusive(v___x_3338_);
if (v_isSharedCheck_3385_ == 0)
{
v___x_3380_ = v___x_3338_;
v_isShared_3381_ = v_isSharedCheck_3385_;
goto v_resetjp_3379_;
}
else
{
lean_inc(v_a_3378_);
lean_dec(v___x_3338_);
v___x_3380_ = lean_box(0);
v_isShared_3381_ = v_isSharedCheck_3385_;
goto v_resetjp_3379_;
}
v_resetjp_3379_:
{
lean_object* v___x_3383_; 
if (v_isShared_3381_ == 0)
{
v___x_3383_ = v___x_3380_;
goto v_reusejp_3382_;
}
else
{
lean_object* v_reuseFailAlloc_3384_; 
v_reuseFailAlloc_3384_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3384_, 0, v_a_3378_);
v___x_3383_ = v_reuseFailAlloc_3384_;
goto v_reusejp_3382_;
}
v_reusejp_3382_:
{
return v___x_3383_;
}
}
}
}
else
{
lean_object* v_fst_3386_; lean_object* v_fst_3387_; lean_object* v___x_3388_; lean_object* v___x_3389_; lean_object* v___x_3390_; lean_object* v___x_3391_; lean_object* v___x_3392_; 
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
v_fst_3386_ = lean_ctor_get(v_a_3333_, 0);
lean_inc_n(v_fst_3386_, 2);
lean_dec(v_a_3333_);
v_fst_3387_ = lean_ctor_get(v_snd_3334_, 0);
lean_inc_n(v_fst_3387_, 2);
lean_dec(v_snd_3334_);
v___x_3388_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__5, &lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__5);
v___x_3389_ = l_Lean_Expr_app___override(v___x_3388_, v_fst_3386_);
v___x_3390_ = l_Lean_Expr_app___override(v___x_3389_, v_fst_3387_);
lean_inc(v_snd_3264_);
v___x_3391_ = l_Lean_Expr_app___override(v___x_3390_, v_snd_3264_);
v___x_3392_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v___x_3391_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3392_) == 0)
{
lean_object* v_a_3393_; lean_object* v_snd_3394_; lean_object* v___x_3395_; lean_object* v___x_3396_; lean_object* v___x_3397_; lean_object* v___x_3398_; 
v_a_3393_ = lean_ctor_get(v___x_3392_, 0);
lean_inc(v_a_3393_);
lean_dec_ref_known(v___x_3392_, 1);
v_snd_3394_ = lean_ctor_get(v_a_3393_, 1);
lean_inc(v_snd_3394_);
lean_dec(v_a_3393_);
v___x_3395_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__8, &lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__8);
v___x_3396_ = l_Lean_Expr_app___override(v___x_3395_, v_fst_3386_);
v___x_3397_ = l_Lean_Expr_app___override(v___x_3396_, v_fst_3387_);
v___x_3398_ = l_Lean_Expr_app___override(v___x_3397_, v_snd_3264_);
v_expr_3238_ = v___x_3398_;
v_a_3239_ = v_snd_3394_;
goto _start;
}
else
{
lean_dec(v_fst_3387_);
lean_dec(v_fst_3386_);
lean_dec(v_snd_3264_);
return v___x_3392_;
}
}
}
else
{
lean_object* v_a_3400_; lean_object* v___x_3402_; uint8_t v_isShared_3403_; uint8_t v_isSharedCheck_3407_; 
lean_dec(v_snd_3264_);
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3400_ = lean_ctor_get(v___x_3332_, 0);
v_isSharedCheck_3407_ = !lean_is_exclusive(v___x_3332_);
if (v_isSharedCheck_3407_ == 0)
{
v___x_3402_ = v___x_3332_;
v_isShared_3403_ = v_isSharedCheck_3407_;
goto v_resetjp_3401_;
}
else
{
lean_inc(v_a_3400_);
lean_dec(v___x_3332_);
v___x_3402_ = lean_box(0);
v_isShared_3403_ = v_isSharedCheck_3407_;
goto v_resetjp_3401_;
}
v_resetjp_3401_:
{
lean_object* v___x_3405_; 
if (v_isShared_3403_ == 0)
{
v___x_3405_ = v___x_3402_;
goto v_reusejp_3404_;
}
else
{
lean_object* v_reuseFailAlloc_3406_; 
v_reuseFailAlloc_3406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3406_, 0, v_a_3400_);
v___x_3405_ = v_reuseFailAlloc_3406_;
goto v_reusejp_3404_;
}
v_reusejp_3404_:
{
return v___x_3405_;
}
}
}
}
else
{
lean_object* v_fst_3408_; lean_object* v___x_3409_; lean_object* v___x_3410_; lean_object* v___f_3411_; lean_object* v___x_3412_; 
lean_dec(v_fst_3263_);
v_fst_3408_ = lean_ctor_get(v_a_3326_, 0);
lean_inc_n(v_fst_3408_, 2);
lean_dec(v_a_3326_);
v___x_3409_ = lean_box(v___x_3266_);
v___x_3410_ = lean_box(v___x_3265_);
lean_inc(v_a_3250_);
v___f_3411_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__7___boxed), 10, 5);
lean_closure_set(v___f_3411_, 0, v___x_3409_);
lean_closure_set(v___f_3411_, 1, v___x_3267_);
lean_closure_set(v___f_3411_, 2, v_fst_3408_);
lean_closure_set(v___f_3411_, 3, v___x_3410_);
lean_closure_set(v___f_3411_, 4, v_a_3250_);
v___x_3412_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3411_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3412_) == 0)
{
lean_object* v_a_3413_; lean_object* v_snd_3414_; lean_object* v_snd_3415_; lean_object* v_snd_3416_; lean_object* v_snd_3417_; lean_object* v_snd_3418_; uint8_t v___x_3419_; 
v_a_3413_ = lean_ctor_get(v___x_3412_, 0);
lean_inc(v_a_3413_);
lean_dec_ref_known(v___x_3412_, 1);
v_snd_3414_ = lean_ctor_get(v_a_3413_, 1);
lean_inc(v_snd_3414_);
v_snd_3415_ = lean_ctor_get(v_snd_3414_, 1);
v_snd_3416_ = lean_ctor_get(v_snd_3415_, 1);
lean_inc(v_snd_3416_);
v_snd_3417_ = lean_ctor_get(v_snd_3416_, 1);
lean_inc(v_snd_3417_);
v_snd_3418_ = lean_ctor_get(v_snd_3417_, 1);
v___x_3419_ = lean_unbox(v_snd_3418_);
if (v___x_3419_ == 0)
{
lean_object* v___x_3420_; lean_object* v___x_3421_; lean_object* v___f_3422_; lean_object* v___x_3423_; 
lean_dec(v_snd_3417_);
lean_dec(v_snd_3416_);
lean_dec(v_snd_3414_);
lean_dec(v_a_3413_);
v___x_3420_ = lean_box(v___x_3266_);
v___x_3421_ = lean_box(v___x_3265_);
v___f_3422_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___lam__8___boxed), 10, 5);
lean_closure_set(v___f_3422_, 0, v___x_3420_);
lean_closure_set(v___f_3422_, 1, v___x_3267_);
lean_closure_set(v___f_3422_, 2, v_fst_3408_);
lean_closure_set(v___f_3422_, 3, v___x_3421_);
lean_closure_set(v___f_3422_, 4, v_a_3250_);
v___x_3423_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Order_addAtom_spec__1___redArg(v___f_3422_, v___x_3265_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3423_) == 0)
{
lean_object* v_a_3424_; lean_object* v___x_3426_; uint8_t v_isShared_3427_; uint8_t v_isSharedCheck_3488_; 
v_a_3424_ = lean_ctor_get(v___x_3423_, 0);
v_isSharedCheck_3488_ = !lean_is_exclusive(v___x_3423_);
if (v_isSharedCheck_3488_ == 0)
{
v___x_3426_ = v___x_3423_;
v_isShared_3427_ = v_isSharedCheck_3488_;
goto v_resetjp_3425_;
}
else
{
lean_inc(v_a_3424_);
lean_dec(v___x_3423_);
v___x_3426_ = lean_box(0);
v_isShared_3427_ = v_isSharedCheck_3488_;
goto v_resetjp_3425_;
}
v_resetjp_3425_:
{
lean_object* v_snd_3428_; lean_object* v_snd_3429_; lean_object* v_snd_3430_; lean_object* v_snd_3431_; lean_object* v_snd_3432_; uint8_t v___x_3433_; 
v_snd_3428_ = lean_ctor_get(v_a_3424_, 1);
lean_inc(v_snd_3428_);
v_snd_3429_ = lean_ctor_get(v_snd_3428_, 1);
v_snd_3430_ = lean_ctor_get(v_snd_3429_, 1);
lean_inc(v_snd_3430_);
v_snd_3431_ = lean_ctor_get(v_snd_3430_, 1);
lean_inc(v_snd_3431_);
v_snd_3432_ = lean_ctor_get(v_snd_3431_, 1);
v___x_3433_ = lean_unbox(v_snd_3432_);
if (v___x_3433_ == 0)
{
lean_object* v___x_3435_; uint8_t v_isShared_3436_; uint8_t v_isSharedCheck_3443_; 
lean_dec(v_snd_3430_);
lean_dec(v_snd_3428_);
lean_dec(v_a_3424_);
lean_dec(v_snd_3264_);
v_isSharedCheck_3443_ = !lean_is_exclusive(v_snd_3431_);
if (v_isSharedCheck_3443_ == 0)
{
lean_object* v_unused_3444_; lean_object* v_unused_3445_; 
v_unused_3444_ = lean_ctor_get(v_snd_3431_, 1);
lean_dec(v_unused_3444_);
v_unused_3445_ = lean_ctor_get(v_snd_3431_, 0);
lean_dec(v_unused_3445_);
v___x_3435_ = v_snd_3431_;
v_isShared_3436_ = v_isSharedCheck_3443_;
goto v_resetjp_3434_;
}
else
{
lean_dec(v_snd_3431_);
v___x_3435_ = lean_box(0);
v_isShared_3436_ = v_isSharedCheck_3443_;
goto v_resetjp_3434_;
}
v_resetjp_3434_:
{
lean_object* v___x_3438_; 
if (v_isShared_3436_ == 0)
{
lean_ctor_set(v___x_3435_, 1, v_a_3239_);
lean_ctor_set(v___x_3435_, 0, v___x_3287_);
v___x_3438_ = v___x_3435_;
goto v_reusejp_3437_;
}
else
{
lean_object* v_reuseFailAlloc_3442_; 
v_reuseFailAlloc_3442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3442_, 0, v___x_3287_);
lean_ctor_set(v_reuseFailAlloc_3442_, 1, v_a_3239_);
v___x_3438_ = v_reuseFailAlloc_3442_;
goto v_reusejp_3437_;
}
v_reusejp_3437_:
{
lean_object* v___x_3440_; 
if (v_isShared_3427_ == 0)
{
lean_ctor_set(v___x_3426_, 0, v___x_3438_);
v___x_3440_ = v___x_3426_;
goto v_reusejp_3439_;
}
else
{
lean_object* v_reuseFailAlloc_3441_; 
v_reuseFailAlloc_3441_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3441_, 0, v___x_3438_);
v___x_3440_ = v_reuseFailAlloc_3441_;
goto v_reusejp_3439_;
}
v_reusejp_3439_:
{
return v___x_3440_;
}
}
}
}
else
{
lean_object* v_fst_3446_; lean_object* v_fst_3447_; lean_object* v_fst_3448_; lean_object* v_fst_3449_; lean_object* v___x_3450_; 
lean_del_object(v___x_3426_);
v_fst_3446_ = lean_ctor_get(v_a_3424_, 0);
lean_inc(v_fst_3446_);
lean_dec(v_a_3424_);
v_fst_3447_ = lean_ctor_get(v_snd_3428_, 0);
lean_inc(v_fst_3447_);
lean_dec(v_snd_3428_);
v_fst_3448_ = lean_ctor_get(v_snd_3430_, 0);
lean_inc(v_fst_3448_);
lean_dec(v_snd_3430_);
v_fst_3449_ = lean_ctor_get(v_snd_3431_, 0);
lean_inc(v_fst_3449_);
lean_dec(v_snd_3431_);
v___x_3450_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_fst_3447_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3450_) == 0)
{
lean_object* v_a_3451_; lean_object* v_fst_3452_; lean_object* v_snd_3453_; lean_object* v___x_3454_; 
v_a_3451_ = lean_ctor_get(v___x_3450_, 0);
lean_inc(v_a_3451_);
lean_dec_ref_known(v___x_3450_, 1);
v_fst_3452_ = lean_ctor_get(v_a_3451_, 0);
lean_inc_n(v_fst_3452_, 2);
v_snd_3453_ = lean_ctor_get(v_a_3451_, 1);
lean_inc(v_snd_3453_);
lean_dec(v_a_3451_);
lean_inc(v_fst_3446_);
v___x_3454_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3446_, v_fst_3452_, v_fst_3448_, v_snd_3453_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3454_) == 0)
{
lean_object* v_a_3455_; lean_object* v_fst_3456_; lean_object* v_snd_3457_; lean_object* v___x_3458_; 
v_a_3455_ = lean_ctor_get(v___x_3454_, 0);
lean_inc(v_a_3455_);
lean_dec_ref_known(v___x_3454_, 1);
v_fst_3456_ = lean_ctor_get(v_a_3455_, 0);
lean_inc(v_fst_3456_);
v_snd_3457_ = lean_ctor_get(v_a_3455_, 1);
lean_inc(v_snd_3457_);
lean_dec(v_a_3455_);
lean_inc(v_fst_3452_);
v___x_3458_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3446_, v_fst_3452_, v_fst_3449_, v_snd_3457_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3458_) == 0)
{
lean_object* v_a_3459_; lean_object* v_fst_3460_; lean_object* v_snd_3461_; lean_object* v___x_3462_; lean_object* v___x_3463_; 
v_a_3459_ = lean_ctor_get(v___x_3458_, 0);
lean_inc(v_a_3459_);
lean_dec_ref_known(v___x_3458_, 1);
v_fst_3460_ = lean_ctor_get(v_a_3459_, 0);
lean_inc(v_fst_3460_);
v_snd_3461_ = lean_ctor_get(v_a_3459_, 1);
lean_inc(v_snd_3461_);
lean_dec(v_a_3459_);
v___x_3462_ = lean_alloc_ctor(5, 3, 0);
lean_ctor_set(v___x_3462_, 0, v_fst_3456_);
lean_ctor_set(v___x_3462_, 1, v_fst_3460_);
lean_ctor_set(v___x_3462_, 2, v_snd_3264_);
v___x_3463_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_fst_3452_, v___x_3462_, v_snd_3461_);
return v___x_3463_;
}
else
{
lean_object* v_a_3464_; lean_object* v___x_3466_; uint8_t v_isShared_3467_; uint8_t v_isSharedCheck_3471_; 
lean_dec(v_fst_3456_);
lean_dec(v_fst_3452_);
lean_dec(v_snd_3264_);
v_a_3464_ = lean_ctor_get(v___x_3458_, 0);
v_isSharedCheck_3471_ = !lean_is_exclusive(v___x_3458_);
if (v_isSharedCheck_3471_ == 0)
{
v___x_3466_ = v___x_3458_;
v_isShared_3467_ = v_isSharedCheck_3471_;
goto v_resetjp_3465_;
}
else
{
lean_inc(v_a_3464_);
lean_dec(v___x_3458_);
v___x_3466_ = lean_box(0);
v_isShared_3467_ = v_isSharedCheck_3471_;
goto v_resetjp_3465_;
}
v_resetjp_3465_:
{
lean_object* v___x_3469_; 
if (v_isShared_3467_ == 0)
{
v___x_3469_ = v___x_3466_;
goto v_reusejp_3468_;
}
else
{
lean_object* v_reuseFailAlloc_3470_; 
v_reuseFailAlloc_3470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3470_, 0, v_a_3464_);
v___x_3469_ = v_reuseFailAlloc_3470_;
goto v_reusejp_3468_;
}
v_reusejp_3468_:
{
return v___x_3469_;
}
}
}
}
else
{
lean_object* v_a_3472_; lean_object* v___x_3474_; uint8_t v_isShared_3475_; uint8_t v_isSharedCheck_3479_; 
lean_dec(v_fst_3452_);
lean_dec(v_fst_3449_);
lean_dec(v_fst_3446_);
lean_dec(v_snd_3264_);
v_a_3472_ = lean_ctor_get(v___x_3454_, 0);
v_isSharedCheck_3479_ = !lean_is_exclusive(v___x_3454_);
if (v_isSharedCheck_3479_ == 0)
{
v___x_3474_ = v___x_3454_;
v_isShared_3475_ = v_isSharedCheck_3479_;
goto v_resetjp_3473_;
}
else
{
lean_inc(v_a_3472_);
lean_dec(v___x_3454_);
v___x_3474_ = lean_box(0);
v_isShared_3475_ = v_isSharedCheck_3479_;
goto v_resetjp_3473_;
}
v_resetjp_3473_:
{
lean_object* v___x_3477_; 
if (v_isShared_3475_ == 0)
{
v___x_3477_ = v___x_3474_;
goto v_reusejp_3476_;
}
else
{
lean_object* v_reuseFailAlloc_3478_; 
v_reuseFailAlloc_3478_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3478_, 0, v_a_3472_);
v___x_3477_ = v_reuseFailAlloc_3478_;
goto v_reusejp_3476_;
}
v_reusejp_3476_:
{
return v___x_3477_;
}
}
}
}
else
{
lean_object* v_a_3480_; lean_object* v___x_3482_; uint8_t v_isShared_3483_; uint8_t v_isSharedCheck_3487_; 
lean_dec(v_fst_3449_);
lean_dec(v_fst_3448_);
lean_dec(v_fst_3446_);
lean_dec(v_snd_3264_);
v_a_3480_ = lean_ctor_get(v___x_3450_, 0);
v_isSharedCheck_3487_ = !lean_is_exclusive(v___x_3450_);
if (v_isSharedCheck_3487_ == 0)
{
v___x_3482_ = v___x_3450_;
v_isShared_3483_ = v_isSharedCheck_3487_;
goto v_resetjp_3481_;
}
else
{
lean_inc(v_a_3480_);
lean_dec(v___x_3450_);
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
v_reuseFailAlloc_3486_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
else
{
lean_object* v_a_3489_; lean_object* v___x_3491_; uint8_t v_isShared_3492_; uint8_t v_isSharedCheck_3496_; 
lean_dec(v_snd_3264_);
lean_dec_ref(v_a_3239_);
v_a_3489_ = lean_ctor_get(v___x_3423_, 0);
v_isSharedCheck_3496_ = !lean_is_exclusive(v___x_3423_);
if (v_isSharedCheck_3496_ == 0)
{
v___x_3491_ = v___x_3423_;
v_isShared_3492_ = v_isSharedCheck_3496_;
goto v_resetjp_3490_;
}
else
{
lean_inc(v_a_3489_);
lean_dec(v___x_3423_);
v___x_3491_ = lean_box(0);
v_isShared_3492_ = v_isSharedCheck_3496_;
goto v_resetjp_3490_;
}
v_resetjp_3490_:
{
lean_object* v___x_3494_; 
if (v_isShared_3492_ == 0)
{
v___x_3494_ = v___x_3491_;
goto v_reusejp_3493_;
}
else
{
lean_object* v_reuseFailAlloc_3495_; 
v_reuseFailAlloc_3495_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3495_, 0, v_a_3489_);
v___x_3494_ = v_reuseFailAlloc_3495_;
goto v_reusejp_3493_;
}
v_reusejp_3493_:
{
return v___x_3494_;
}
}
}
}
else
{
lean_object* v_fst_3497_; lean_object* v_fst_3498_; lean_object* v_fst_3499_; lean_object* v_fst_3500_; lean_object* v___x_3501_; 
lean_dec(v_fst_3408_);
lean_dec(v_a_3250_);
v_fst_3497_ = lean_ctor_get(v_a_3413_, 0);
lean_inc(v_fst_3497_);
lean_dec(v_a_3413_);
v_fst_3498_ = lean_ctor_get(v_snd_3414_, 0);
lean_inc(v_fst_3498_);
lean_dec(v_snd_3414_);
v_fst_3499_ = lean_ctor_get(v_snd_3416_, 0);
lean_inc(v_fst_3499_);
lean_dec(v_snd_3416_);
v_fst_3500_ = lean_ctor_get(v_snd_3417_, 0);
lean_inc(v_fst_3500_);
lean_dec(v_snd_3417_);
v___x_3501_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_fst_3498_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3501_) == 0)
{
lean_object* v_a_3502_; lean_object* v_fst_3503_; lean_object* v_snd_3504_; lean_object* v___x_3505_; 
v_a_3502_ = lean_ctor_get(v___x_3501_, 0);
lean_inc(v_a_3502_);
lean_dec_ref_known(v___x_3501_, 1);
v_fst_3503_ = lean_ctor_get(v_a_3502_, 0);
lean_inc_n(v_fst_3503_, 2);
v_snd_3504_ = lean_ctor_get(v_a_3502_, 1);
lean_inc(v_snd_3504_);
lean_dec(v_a_3502_);
lean_inc(v_fst_3497_);
v___x_3505_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3497_, v_fst_3503_, v_fst_3499_, v_snd_3504_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3505_) == 0)
{
lean_object* v_a_3506_; lean_object* v_fst_3507_; lean_object* v_snd_3508_; lean_object* v___x_3509_; 
v_a_3506_ = lean_ctor_get(v___x_3505_, 0);
lean_inc(v_a_3506_);
lean_dec_ref_known(v___x_3505_, 1);
v_fst_3507_ = lean_ctor_get(v_a_3506_, 0);
lean_inc(v_fst_3507_);
v_snd_3508_ = lean_ctor_get(v_a_3506_, 1);
lean_inc(v_snd_3508_);
lean_dec(v_a_3506_);
lean_inc(v_fst_3503_);
v___x_3509_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3497_, v_fst_3503_, v_fst_3500_, v_snd_3508_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3509_) == 0)
{
lean_object* v_a_3510_; lean_object* v_fst_3511_; lean_object* v_snd_3512_; lean_object* v___x_3513_; lean_object* v___x_3514_; 
v_a_3510_ = lean_ctor_get(v___x_3509_, 0);
lean_inc(v_a_3510_);
lean_dec_ref_known(v___x_3509_, 1);
v_fst_3511_ = lean_ctor_get(v_a_3510_, 0);
lean_inc(v_fst_3511_);
v_snd_3512_ = lean_ctor_get(v_a_3510_, 1);
lean_inc(v_snd_3512_);
lean_dec(v_a_3510_);
v___x_3513_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3513_, 0, v_fst_3507_);
lean_ctor_set(v___x_3513_, 1, v_fst_3511_);
lean_ctor_set(v___x_3513_, 2, v_snd_3264_);
v___x_3514_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_fst_3503_, v___x_3513_, v_snd_3512_);
return v___x_3514_;
}
else
{
lean_object* v_a_3515_; lean_object* v___x_3517_; uint8_t v_isShared_3518_; uint8_t v_isSharedCheck_3522_; 
lean_dec(v_fst_3507_);
lean_dec(v_fst_3503_);
lean_dec(v_snd_3264_);
v_a_3515_ = lean_ctor_get(v___x_3509_, 0);
v_isSharedCheck_3522_ = !lean_is_exclusive(v___x_3509_);
if (v_isSharedCheck_3522_ == 0)
{
v___x_3517_ = v___x_3509_;
v_isShared_3518_ = v_isSharedCheck_3522_;
goto v_resetjp_3516_;
}
else
{
lean_inc(v_a_3515_);
lean_dec(v___x_3509_);
v___x_3517_ = lean_box(0);
v_isShared_3518_ = v_isSharedCheck_3522_;
goto v_resetjp_3516_;
}
v_resetjp_3516_:
{
lean_object* v___x_3520_; 
if (v_isShared_3518_ == 0)
{
v___x_3520_ = v___x_3517_;
goto v_reusejp_3519_;
}
else
{
lean_object* v_reuseFailAlloc_3521_; 
v_reuseFailAlloc_3521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3521_, 0, v_a_3515_);
v___x_3520_ = v_reuseFailAlloc_3521_;
goto v_reusejp_3519_;
}
v_reusejp_3519_:
{
return v___x_3520_;
}
}
}
}
else
{
lean_object* v_a_3523_; lean_object* v___x_3525_; uint8_t v_isShared_3526_; uint8_t v_isSharedCheck_3530_; 
lean_dec(v_fst_3503_);
lean_dec(v_fst_3500_);
lean_dec(v_fst_3497_);
lean_dec(v_snd_3264_);
v_a_3523_ = lean_ctor_get(v___x_3505_, 0);
v_isSharedCheck_3530_ = !lean_is_exclusive(v___x_3505_);
if (v_isSharedCheck_3530_ == 0)
{
v___x_3525_ = v___x_3505_;
v_isShared_3526_ = v_isSharedCheck_3530_;
goto v_resetjp_3524_;
}
else
{
lean_inc(v_a_3523_);
lean_dec(v___x_3505_);
v___x_3525_ = lean_box(0);
v_isShared_3526_ = v_isSharedCheck_3530_;
goto v_resetjp_3524_;
}
v_resetjp_3524_:
{
lean_object* v___x_3528_; 
if (v_isShared_3526_ == 0)
{
v___x_3528_ = v___x_3525_;
goto v_reusejp_3527_;
}
else
{
lean_object* v_reuseFailAlloc_3529_; 
v_reuseFailAlloc_3529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3529_, 0, v_a_3523_);
v___x_3528_ = v_reuseFailAlloc_3529_;
goto v_reusejp_3527_;
}
v_reusejp_3527_:
{
return v___x_3528_;
}
}
}
}
else
{
lean_object* v_a_3531_; lean_object* v___x_3533_; uint8_t v_isShared_3534_; uint8_t v_isSharedCheck_3538_; 
lean_dec(v_fst_3500_);
lean_dec(v_fst_3499_);
lean_dec(v_fst_3497_);
lean_dec(v_snd_3264_);
v_a_3531_ = lean_ctor_get(v___x_3501_, 0);
v_isSharedCheck_3538_ = !lean_is_exclusive(v___x_3501_);
if (v_isSharedCheck_3538_ == 0)
{
v___x_3533_ = v___x_3501_;
v_isShared_3534_ = v_isSharedCheck_3538_;
goto v_resetjp_3532_;
}
else
{
lean_inc(v_a_3531_);
lean_dec(v___x_3501_);
v___x_3533_ = lean_box(0);
v_isShared_3534_ = v_isSharedCheck_3538_;
goto v_resetjp_3532_;
}
v_resetjp_3532_:
{
lean_object* v___x_3536_; 
if (v_isShared_3534_ == 0)
{
v___x_3536_ = v___x_3533_;
goto v_reusejp_3535_;
}
else
{
lean_object* v_reuseFailAlloc_3537_; 
v_reuseFailAlloc_3537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3537_, 0, v_a_3531_);
v___x_3536_ = v_reuseFailAlloc_3537_;
goto v_reusejp_3535_;
}
v_reusejp_3535_:
{
return v___x_3536_;
}
}
}
}
}
else
{
lean_object* v_a_3539_; lean_object* v___x_3541_; uint8_t v_isShared_3542_; uint8_t v_isSharedCheck_3546_; 
lean_dec(v_fst_3408_);
lean_dec(v_snd_3264_);
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3539_ = lean_ctor_get(v___x_3412_, 0);
v_isSharedCheck_3546_ = !lean_is_exclusive(v___x_3412_);
if (v_isSharedCheck_3546_ == 0)
{
v___x_3541_ = v___x_3412_;
v_isShared_3542_ = v_isSharedCheck_3546_;
goto v_resetjp_3540_;
}
else
{
lean_inc(v_a_3539_);
lean_dec(v___x_3412_);
v___x_3541_ = lean_box(0);
v_isShared_3542_ = v_isSharedCheck_3546_;
goto v_resetjp_3540_;
}
v_resetjp_3540_:
{
lean_object* v___x_3544_; 
if (v_isShared_3542_ == 0)
{
v___x_3544_ = v___x_3541_;
goto v_reusejp_3543_;
}
else
{
lean_object* v_reuseFailAlloc_3545_; 
v_reuseFailAlloc_3545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3545_, 0, v_a_3539_);
v___x_3544_ = v_reuseFailAlloc_3545_;
goto v_reusejp_3543_;
}
v_reusejp_3543_:
{
return v___x_3544_;
}
}
}
}
}
else
{
lean_object* v_a_3547_; lean_object* v___x_3549_; uint8_t v_isShared_3550_; uint8_t v_isSharedCheck_3554_; 
lean_dec(v_snd_3264_);
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3547_ = lean_ctor_get(v___x_3325_, 0);
v_isSharedCheck_3554_ = !lean_is_exclusive(v___x_3325_);
if (v_isSharedCheck_3554_ == 0)
{
v___x_3549_ = v___x_3325_;
v_isShared_3550_ = v_isSharedCheck_3554_;
goto v_resetjp_3548_;
}
else
{
lean_inc(v_a_3547_);
lean_dec(v___x_3325_);
v___x_3549_ = lean_box(0);
v_isShared_3550_ = v_isSharedCheck_3554_;
goto v_resetjp_3548_;
}
v_resetjp_3548_:
{
lean_object* v___x_3552_; 
if (v_isShared_3550_ == 0)
{
v___x_3552_ = v___x_3549_;
goto v_reusejp_3551_;
}
else
{
lean_object* v_reuseFailAlloc_3553_; 
v_reuseFailAlloc_3553_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3553_, 0, v_a_3547_);
v___x_3552_ = v_reuseFailAlloc_3553_;
goto v_reusejp_3551_;
}
v_reusejp_3551_:
{
return v___x_3552_;
}
}
}
}
else
{
lean_object* v_fst_3555_; lean_object* v_fst_3556_; lean_object* v_fst_3557_; lean_object* v___x_3559_; uint8_t v_isShared_3560_; uint8_t v_isSharedCheck_3633_; 
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
v_fst_3555_ = lean_ctor_get(v_a_3315_, 0);
lean_inc(v_fst_3555_);
lean_dec(v_a_3315_);
v_fst_3556_ = lean_ctor_get(v_snd_3316_, 0);
lean_inc(v_fst_3556_);
lean_dec(v_snd_3316_);
v_fst_3557_ = lean_ctor_get(v_snd_3317_, 0);
v_isSharedCheck_3633_ = !lean_is_exclusive(v_snd_3317_);
if (v_isSharedCheck_3633_ == 0)
{
lean_object* v_unused_3634_; 
v_unused_3634_ = lean_ctor_get(v_snd_3317_, 1);
lean_dec(v_unused_3634_);
v___x_3559_ = v_snd_3317_;
v_isShared_3560_ = v_isSharedCheck_3633_;
goto v_resetjp_3558_;
}
else
{
lean_inc(v_fst_3557_);
lean_dec(v_snd_3317_);
v___x_3559_ = lean_box(0);
v_isShared_3560_ = v_isSharedCheck_3633_;
goto v_resetjp_3558_;
}
v_resetjp_3558_:
{
lean_object* v_fst_3561_; lean_object* v___x_3563_; uint8_t v_isShared_3564_; uint8_t v_isSharedCheck_3631_; 
v_fst_3561_ = lean_ctor_get(v_snd_3318_, 0);
v_isSharedCheck_3631_ = !lean_is_exclusive(v_snd_3318_);
if (v_isSharedCheck_3631_ == 0)
{
lean_object* v_unused_3632_; 
v_unused_3632_ = lean_ctor_get(v_snd_3318_, 1);
lean_dec(v_unused_3632_);
v___x_3563_ = v_snd_3318_;
v_isShared_3564_ = v_isSharedCheck_3631_;
goto v_resetjp_3562_;
}
else
{
lean_inc(v_fst_3561_);
lean_dec(v_snd_3318_);
v___x_3563_ = lean_box(0);
v_isShared_3564_ = v_isSharedCheck_3631_;
goto v_resetjp_3562_;
}
v_resetjp_3562_:
{
lean_object* v___x_3565_; lean_object* v___x_3566_; lean_object* v___x_3568_; 
v___x_3565_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__10));
v___x_3566_ = lean_box(0);
lean_inc(v_fst_3555_);
if (v_isShared_3560_ == 0)
{
lean_ctor_set_tag(v___x_3559_, 1);
lean_ctor_set(v___x_3559_, 1, v___x_3566_);
lean_ctor_set(v___x_3559_, 0, v_fst_3555_);
v___x_3568_ = v___x_3559_;
goto v_reusejp_3567_;
}
else
{
lean_object* v_reuseFailAlloc_3630_; 
v_reuseFailAlloc_3630_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3630_, 0, v_fst_3555_);
lean_ctor_set(v_reuseFailAlloc_3630_, 1, v___x_3566_);
v___x_3568_ = v_reuseFailAlloc_3630_;
goto v_reusejp_3567_;
}
v_reusejp_3567_:
{
lean_object* v___x_3569_; lean_object* v___x_3570_; lean_object* v___x_3571_; lean_object* v___x_3572_; 
v___x_3569_ = l_Lean_Expr_const___override(v___x_3565_, v___x_3568_);
lean_inc(v_fst_3556_);
v___x_3570_ = l_Lean_Expr_app___override(v___x_3569_, v_fst_3556_);
v___x_3571_ = lean_box(0);
v___x_3572_ = l_Lean_Meta_synthInstance_x3f(v___x_3570_, v___x_3571_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3572_) == 0)
{
lean_object* v_a_3573_; lean_object* v___x_3575_; uint8_t v_isShared_3576_; uint8_t v_isSharedCheck_3621_; 
v_a_3573_ = lean_ctor_get(v___x_3572_, 0);
v_isSharedCheck_3621_ = !lean_is_exclusive(v___x_3572_);
if (v_isSharedCheck_3621_ == 0)
{
v___x_3575_ = v___x_3572_;
v_isShared_3576_ = v_isSharedCheck_3621_;
goto v_resetjp_3574_;
}
else
{
lean_inc(v_a_3573_);
lean_dec(v___x_3572_);
v___x_3575_ = lean_box(0);
v_isShared_3576_ = v_isSharedCheck_3621_;
goto v_resetjp_3574_;
}
v_resetjp_3574_:
{
if (lean_obj_tag(v_a_3573_) == 0)
{
lean_object* v___x_3578_; 
lean_dec(v_fst_3561_);
lean_dec(v_fst_3557_);
lean_dec(v_fst_3556_);
lean_dec(v_fst_3555_);
lean_dec(v_snd_3264_);
if (v_isShared_3564_ == 0)
{
lean_ctor_set(v___x_3563_, 1, v_a_3239_);
lean_ctor_set(v___x_3563_, 0, v___x_3287_);
v___x_3578_ = v___x_3563_;
goto v_reusejp_3577_;
}
else
{
lean_object* v_reuseFailAlloc_3582_; 
v_reuseFailAlloc_3582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3582_, 0, v___x_3287_);
lean_ctor_set(v_reuseFailAlloc_3582_, 1, v_a_3239_);
v___x_3578_ = v_reuseFailAlloc_3582_;
goto v_reusejp_3577_;
}
v_reusejp_3577_:
{
lean_object* v___x_3580_; 
if (v_isShared_3576_ == 0)
{
lean_ctor_set(v___x_3575_, 0, v___x_3578_);
v___x_3580_ = v___x_3575_;
goto v_reusejp_3579_;
}
else
{
lean_object* v_reuseFailAlloc_3581_; 
v_reuseFailAlloc_3581_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3581_, 0, v___x_3578_);
v___x_3580_ = v_reuseFailAlloc_3581_;
goto v_reusejp_3579_;
}
v_reusejp_3579_:
{
return v___x_3580_;
}
}
}
else
{
lean_object* v___x_3583_; 
lean_dec_ref_known(v_a_3573_, 1);
lean_del_object(v___x_3575_);
lean_del_object(v___x_3563_);
v___x_3583_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_fst_3556_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3583_) == 0)
{
lean_object* v_a_3584_; lean_object* v_fst_3585_; lean_object* v_snd_3586_; lean_object* v___x_3587_; 
v_a_3584_ = lean_ctor_get(v___x_3583_, 0);
lean_inc(v_a_3584_);
lean_dec_ref_known(v___x_3583_, 1);
v_fst_3585_ = lean_ctor_get(v_a_3584_, 0);
lean_inc_n(v_fst_3585_, 2);
v_snd_3586_ = lean_ctor_get(v_a_3584_, 1);
lean_inc(v_snd_3586_);
lean_dec(v_a_3584_);
lean_inc(v_fst_3555_);
v___x_3587_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3555_, v_fst_3585_, v_fst_3557_, v_snd_3586_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3587_) == 0)
{
lean_object* v_a_3588_; lean_object* v_fst_3589_; lean_object* v_snd_3590_; lean_object* v___x_3591_; 
v_a_3588_ = lean_ctor_get(v___x_3587_, 0);
lean_inc(v_a_3588_);
lean_dec_ref_known(v___x_3587_, 1);
v_fst_3589_ = lean_ctor_get(v_a_3588_, 0);
lean_inc(v_fst_3589_);
v_snd_3590_ = lean_ctor_get(v_a_3588_, 1);
lean_inc(v_snd_3590_);
lean_dec(v_a_3588_);
lean_inc(v_fst_3585_);
v___x_3591_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3555_, v_fst_3585_, v_fst_3561_, v_snd_3590_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3591_) == 0)
{
lean_object* v_a_3592_; lean_object* v_fst_3593_; lean_object* v_snd_3594_; lean_object* v___x_3595_; lean_object* v___x_3596_; 
v_a_3592_ = lean_ctor_get(v___x_3591_, 0);
lean_inc(v_a_3592_);
lean_dec_ref_known(v___x_3591_, 1);
v_fst_3593_ = lean_ctor_get(v_a_3592_, 0);
lean_inc(v_fst_3593_);
v_snd_3594_ = lean_ctor_get(v_a_3592_, 1);
lean_inc(v_snd_3594_);
lean_dec(v_a_3592_);
v___x_3595_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3595_, 0, v_fst_3589_);
lean_ctor_set(v___x_3595_, 1, v_fst_3593_);
lean_ctor_set(v___x_3595_, 2, v_snd_3264_);
v___x_3596_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_fst_3585_, v___x_3595_, v_snd_3594_);
return v___x_3596_;
}
else
{
lean_object* v_a_3597_; lean_object* v___x_3599_; uint8_t v_isShared_3600_; uint8_t v_isSharedCheck_3604_; 
lean_dec(v_fst_3589_);
lean_dec(v_fst_3585_);
lean_dec(v_snd_3264_);
v_a_3597_ = lean_ctor_get(v___x_3591_, 0);
v_isSharedCheck_3604_ = !lean_is_exclusive(v___x_3591_);
if (v_isSharedCheck_3604_ == 0)
{
v___x_3599_ = v___x_3591_;
v_isShared_3600_ = v_isSharedCheck_3604_;
goto v_resetjp_3598_;
}
else
{
lean_inc(v_a_3597_);
lean_dec(v___x_3591_);
v___x_3599_ = lean_box(0);
v_isShared_3600_ = v_isSharedCheck_3604_;
goto v_resetjp_3598_;
}
v_resetjp_3598_:
{
lean_object* v___x_3602_; 
if (v_isShared_3600_ == 0)
{
v___x_3602_ = v___x_3599_;
goto v_reusejp_3601_;
}
else
{
lean_object* v_reuseFailAlloc_3603_; 
v_reuseFailAlloc_3603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3603_, 0, v_a_3597_);
v___x_3602_ = v_reuseFailAlloc_3603_;
goto v_reusejp_3601_;
}
v_reusejp_3601_:
{
return v___x_3602_;
}
}
}
}
else
{
lean_object* v_a_3605_; lean_object* v___x_3607_; uint8_t v_isShared_3608_; uint8_t v_isSharedCheck_3612_; 
lean_dec(v_fst_3585_);
lean_dec(v_fst_3561_);
lean_dec(v_fst_3555_);
lean_dec(v_snd_3264_);
v_a_3605_ = lean_ctor_get(v___x_3587_, 0);
v_isSharedCheck_3612_ = !lean_is_exclusive(v___x_3587_);
if (v_isSharedCheck_3612_ == 0)
{
v___x_3607_ = v___x_3587_;
v_isShared_3608_ = v_isSharedCheck_3612_;
goto v_resetjp_3606_;
}
else
{
lean_inc(v_a_3605_);
lean_dec(v___x_3587_);
v___x_3607_ = lean_box(0);
v_isShared_3608_ = v_isSharedCheck_3612_;
goto v_resetjp_3606_;
}
v_resetjp_3606_:
{
lean_object* v___x_3610_; 
if (v_isShared_3608_ == 0)
{
v___x_3610_ = v___x_3607_;
goto v_reusejp_3609_;
}
else
{
lean_object* v_reuseFailAlloc_3611_; 
v_reuseFailAlloc_3611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3611_, 0, v_a_3605_);
v___x_3610_ = v_reuseFailAlloc_3611_;
goto v_reusejp_3609_;
}
v_reusejp_3609_:
{
return v___x_3610_;
}
}
}
}
else
{
lean_object* v_a_3613_; lean_object* v___x_3615_; uint8_t v_isShared_3616_; uint8_t v_isSharedCheck_3620_; 
lean_dec(v_fst_3561_);
lean_dec(v_fst_3557_);
lean_dec(v_fst_3555_);
lean_dec(v_snd_3264_);
v_a_3613_ = lean_ctor_get(v___x_3583_, 0);
v_isSharedCheck_3620_ = !lean_is_exclusive(v___x_3583_);
if (v_isSharedCheck_3620_ == 0)
{
v___x_3615_ = v___x_3583_;
v_isShared_3616_ = v_isSharedCheck_3620_;
goto v_resetjp_3614_;
}
else
{
lean_inc(v_a_3613_);
lean_dec(v___x_3583_);
v___x_3615_ = lean_box(0);
v_isShared_3616_ = v_isSharedCheck_3620_;
goto v_resetjp_3614_;
}
v_resetjp_3614_:
{
lean_object* v___x_3618_; 
if (v_isShared_3616_ == 0)
{
v___x_3618_ = v___x_3615_;
goto v_reusejp_3617_;
}
else
{
lean_object* v_reuseFailAlloc_3619_; 
v_reuseFailAlloc_3619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3619_, 0, v_a_3613_);
v___x_3618_ = v_reuseFailAlloc_3619_;
goto v_reusejp_3617_;
}
v_reusejp_3617_:
{
return v___x_3618_;
}
}
}
}
}
}
else
{
lean_object* v_a_3622_; lean_object* v___x_3624_; uint8_t v_isShared_3625_; uint8_t v_isSharedCheck_3629_; 
lean_del_object(v___x_3563_);
lean_dec(v_fst_3561_);
lean_dec(v_fst_3557_);
lean_dec(v_fst_3556_);
lean_dec(v_fst_3555_);
lean_dec(v_snd_3264_);
lean_dec_ref(v_a_3239_);
v_a_3622_ = lean_ctor_get(v___x_3572_, 0);
v_isSharedCheck_3629_ = !lean_is_exclusive(v___x_3572_);
if (v_isSharedCheck_3629_ == 0)
{
v___x_3624_ = v___x_3572_;
v_isShared_3625_ = v_isSharedCheck_3629_;
goto v_resetjp_3623_;
}
else
{
lean_inc(v_a_3622_);
lean_dec(v___x_3572_);
v___x_3624_ = lean_box(0);
v_isShared_3625_ = v_isSharedCheck_3629_;
goto v_resetjp_3623_;
}
v_resetjp_3623_:
{
lean_object* v___x_3627_; 
if (v_isShared_3625_ == 0)
{
v___x_3627_ = v___x_3624_;
goto v_reusejp_3626_;
}
else
{
lean_object* v_reuseFailAlloc_3628_; 
v_reuseFailAlloc_3628_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3628_, 0, v_a_3622_);
v___x_3627_ = v_reuseFailAlloc_3628_;
goto v_reusejp_3626_;
}
v_reusejp_3626_:
{
return v___x_3627_;
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
lean_object* v_a_3635_; lean_object* v___x_3637_; uint8_t v_isShared_3638_; uint8_t v_isSharedCheck_3642_; 
lean_dec(v_snd_3264_);
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3635_ = lean_ctor_get(v___x_3314_, 0);
v_isSharedCheck_3642_ = !lean_is_exclusive(v___x_3314_);
if (v_isSharedCheck_3642_ == 0)
{
v___x_3637_ = v___x_3314_;
v_isShared_3638_ = v_isSharedCheck_3642_;
goto v_resetjp_3636_;
}
else
{
lean_inc(v_a_3635_);
lean_dec(v___x_3314_);
v___x_3637_ = lean_box(0);
v_isShared_3638_ = v_isSharedCheck_3642_;
goto v_resetjp_3636_;
}
v_resetjp_3636_:
{
lean_object* v___x_3640_; 
if (v_isShared_3638_ == 0)
{
v___x_3640_ = v___x_3637_;
goto v_reusejp_3639_;
}
else
{
lean_object* v_reuseFailAlloc_3641_; 
v_reuseFailAlloc_3641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3641_, 0, v_a_3635_);
v___x_3640_ = v_reuseFailAlloc_3641_;
goto v_reusejp_3639_;
}
v_reusejp_3639_:
{
return v___x_3640_;
}
}
}
}
else
{
lean_object* v_fst_3643_; lean_object* v_fst_3644_; lean_object* v_fst_3645_; lean_object* v_fst_3646_; lean_object* v___x_3647_; 
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
v_fst_3643_ = lean_ctor_get(v_a_3304_, 0);
lean_inc(v_fst_3643_);
lean_dec(v_a_3304_);
v_fst_3644_ = lean_ctor_get(v_snd_3305_, 0);
lean_inc(v_fst_3644_);
lean_dec(v_snd_3305_);
v_fst_3645_ = lean_ctor_get(v_snd_3307_, 0);
lean_inc(v_fst_3645_);
lean_dec(v_snd_3307_);
v_fst_3646_ = lean_ctor_get(v_snd_3308_, 0);
lean_inc(v_fst_3646_);
lean_dec(v_snd_3308_);
v___x_3647_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_fst_3644_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3647_) == 0)
{
lean_object* v_a_3648_; lean_object* v_fst_3649_; lean_object* v_snd_3650_; lean_object* v___x_3651_; 
v_a_3648_ = lean_ctor_get(v___x_3647_, 0);
lean_inc(v_a_3648_);
lean_dec_ref_known(v___x_3647_, 1);
v_fst_3649_ = lean_ctor_get(v_a_3648_, 0);
lean_inc_n(v_fst_3649_, 2);
v_snd_3650_ = lean_ctor_get(v_a_3648_, 1);
lean_inc(v_snd_3650_);
lean_dec(v_a_3648_);
lean_inc(v_fst_3643_);
v___x_3651_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3643_, v_fst_3649_, v_fst_3645_, v_snd_3650_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3651_) == 0)
{
lean_object* v_a_3652_; lean_object* v_fst_3653_; lean_object* v_snd_3654_; lean_object* v___x_3655_; 
v_a_3652_ = lean_ctor_get(v___x_3651_, 0);
lean_inc(v_a_3652_);
lean_dec_ref_known(v___x_3651_, 1);
v_fst_3653_ = lean_ctor_get(v_a_3652_, 0);
lean_inc(v_fst_3653_);
v_snd_3654_ = lean_ctor_get(v_a_3652_, 1);
lean_inc(v_snd_3654_);
lean_dec(v_a_3652_);
lean_inc(v_fst_3649_);
v___x_3655_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3643_, v_fst_3649_, v_fst_3646_, v_snd_3654_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3655_) == 0)
{
lean_object* v_a_3656_; lean_object* v_fst_3657_; lean_object* v_snd_3658_; lean_object* v___x_3659_; lean_object* v___x_3660_; 
v_a_3656_ = lean_ctor_get(v___x_3655_, 0);
lean_inc(v_a_3656_);
lean_dec_ref_known(v___x_3655_, 1);
v_fst_3657_ = lean_ctor_get(v_a_3656_, 0);
lean_inc(v_fst_3657_);
v_snd_3658_ = lean_ctor_get(v_a_3656_, 1);
lean_inc(v_snd_3658_);
lean_dec(v_a_3656_);
v___x_3659_ = lean_alloc_ctor(4, 3, 0);
lean_ctor_set(v___x_3659_, 0, v_fst_3653_);
lean_ctor_set(v___x_3659_, 1, v_fst_3657_);
lean_ctor_set(v___x_3659_, 2, v_snd_3264_);
v___x_3660_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_fst_3649_, v___x_3659_, v_snd_3658_);
return v___x_3660_;
}
else
{
lean_object* v_a_3661_; lean_object* v___x_3663_; uint8_t v_isShared_3664_; uint8_t v_isSharedCheck_3668_; 
lean_dec(v_fst_3653_);
lean_dec(v_fst_3649_);
lean_dec(v_snd_3264_);
v_a_3661_ = lean_ctor_get(v___x_3655_, 0);
v_isSharedCheck_3668_ = !lean_is_exclusive(v___x_3655_);
if (v_isSharedCheck_3668_ == 0)
{
v___x_3663_ = v___x_3655_;
v_isShared_3664_ = v_isSharedCheck_3668_;
goto v_resetjp_3662_;
}
else
{
lean_inc(v_a_3661_);
lean_dec(v___x_3655_);
v___x_3663_ = lean_box(0);
v_isShared_3664_ = v_isSharedCheck_3668_;
goto v_resetjp_3662_;
}
v_resetjp_3662_:
{
lean_object* v___x_3666_; 
if (v_isShared_3664_ == 0)
{
v___x_3666_ = v___x_3663_;
goto v_reusejp_3665_;
}
else
{
lean_object* v_reuseFailAlloc_3667_; 
v_reuseFailAlloc_3667_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3667_, 0, v_a_3661_);
v___x_3666_ = v_reuseFailAlloc_3667_;
goto v_reusejp_3665_;
}
v_reusejp_3665_:
{
return v___x_3666_;
}
}
}
}
else
{
lean_object* v_a_3669_; lean_object* v___x_3671_; uint8_t v_isShared_3672_; uint8_t v_isSharedCheck_3676_; 
lean_dec(v_fst_3649_);
lean_dec(v_fst_3646_);
lean_dec(v_fst_3643_);
lean_dec(v_snd_3264_);
v_a_3669_ = lean_ctor_get(v___x_3651_, 0);
v_isSharedCheck_3676_ = !lean_is_exclusive(v___x_3651_);
if (v_isSharedCheck_3676_ == 0)
{
v___x_3671_ = v___x_3651_;
v_isShared_3672_ = v_isSharedCheck_3676_;
goto v_resetjp_3670_;
}
else
{
lean_inc(v_a_3669_);
lean_dec(v___x_3651_);
v___x_3671_ = lean_box(0);
v_isShared_3672_ = v_isSharedCheck_3676_;
goto v_resetjp_3670_;
}
v_resetjp_3670_:
{
lean_object* v___x_3674_; 
if (v_isShared_3672_ == 0)
{
v___x_3674_ = v___x_3671_;
goto v_reusejp_3673_;
}
else
{
lean_object* v_reuseFailAlloc_3675_; 
v_reuseFailAlloc_3675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3675_, 0, v_a_3669_);
v___x_3674_ = v_reuseFailAlloc_3675_;
goto v_reusejp_3673_;
}
v_reusejp_3673_:
{
return v___x_3674_;
}
}
}
}
else
{
lean_object* v_a_3677_; lean_object* v___x_3679_; uint8_t v_isShared_3680_; uint8_t v_isSharedCheck_3684_; 
lean_dec(v_fst_3646_);
lean_dec(v_fst_3645_);
lean_dec(v_fst_3643_);
lean_dec(v_snd_3264_);
v_a_3677_ = lean_ctor_get(v___x_3647_, 0);
v_isSharedCheck_3684_ = !lean_is_exclusive(v___x_3647_);
if (v_isSharedCheck_3684_ == 0)
{
v___x_3679_ = v___x_3647_;
v_isShared_3680_ = v_isSharedCheck_3684_;
goto v_resetjp_3678_;
}
else
{
lean_inc(v_a_3677_);
lean_dec(v___x_3647_);
v___x_3679_ = lean_box(0);
v_isShared_3680_ = v_isSharedCheck_3684_;
goto v_resetjp_3678_;
}
v_resetjp_3678_:
{
lean_object* v___x_3682_; 
if (v_isShared_3680_ == 0)
{
v___x_3682_ = v___x_3679_;
goto v_reusejp_3681_;
}
else
{
lean_object* v_reuseFailAlloc_3683_; 
v_reuseFailAlloc_3683_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3683_, 0, v_a_3677_);
v___x_3682_ = v_reuseFailAlloc_3683_;
goto v_reusejp_3681_;
}
v_reusejp_3681_:
{
return v___x_3682_;
}
}
}
}
}
else
{
lean_object* v_a_3685_; lean_object* v___x_3687_; uint8_t v_isShared_3688_; uint8_t v_isSharedCheck_3692_; 
lean_dec(v_snd_3264_);
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3685_ = lean_ctor_get(v___x_3303_, 0);
v_isSharedCheck_3692_ = !lean_is_exclusive(v___x_3303_);
if (v_isSharedCheck_3692_ == 0)
{
v___x_3687_ = v___x_3303_;
v_isShared_3688_ = v_isSharedCheck_3692_;
goto v_resetjp_3686_;
}
else
{
lean_inc(v_a_3685_);
lean_dec(v___x_3303_);
v___x_3687_ = lean_box(0);
v_isShared_3688_ = v_isSharedCheck_3692_;
goto v_resetjp_3686_;
}
v_resetjp_3686_:
{
lean_object* v___x_3690_; 
if (v_isShared_3688_ == 0)
{
v___x_3690_ = v___x_3687_;
goto v_reusejp_3689_;
}
else
{
lean_object* v_reuseFailAlloc_3691_; 
v_reuseFailAlloc_3691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3691_, 0, v_a_3685_);
v___x_3690_ = v_reuseFailAlloc_3691_;
goto v_reusejp_3689_;
}
v_reusejp_3689_:
{
return v___x_3690_;
}
}
}
}
else
{
lean_object* v_fst_3693_; lean_object* v_fst_3694_; lean_object* v_fst_3695_; lean_object* v_fst_3696_; lean_object* v___x_3697_; 
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
v_fst_3693_ = lean_ctor_get(v_a_3293_, 0);
lean_inc(v_fst_3693_);
lean_dec(v_a_3293_);
v_fst_3694_ = lean_ctor_get(v_snd_3294_, 0);
lean_inc(v_fst_3694_);
lean_dec(v_snd_3294_);
v_fst_3695_ = lean_ctor_get(v_snd_3296_, 0);
lean_inc(v_fst_3695_);
lean_dec(v_snd_3296_);
v_fst_3696_ = lean_ctor_get(v_snd_3297_, 0);
lean_inc(v_fst_3696_);
lean_dec(v_snd_3297_);
v___x_3697_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_fst_3694_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3697_) == 0)
{
lean_object* v_a_3698_; lean_object* v_fst_3699_; lean_object* v_snd_3700_; lean_object* v___x_3701_; 
v_a_3698_ = lean_ctor_get(v___x_3697_, 0);
lean_inc(v_a_3698_);
lean_dec_ref_known(v___x_3697_, 1);
v_fst_3699_ = lean_ctor_get(v_a_3698_, 0);
lean_inc_n(v_fst_3699_, 2);
v_snd_3700_ = lean_ctor_get(v_a_3698_, 1);
lean_inc(v_snd_3700_);
lean_dec(v_a_3698_);
lean_inc(v_fst_3693_);
v___x_3701_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3693_, v_fst_3699_, v_fst_3695_, v_snd_3700_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3701_) == 0)
{
lean_object* v_a_3702_; lean_object* v_fst_3703_; lean_object* v_snd_3704_; lean_object* v___x_3705_; 
v_a_3702_ = lean_ctor_get(v___x_3701_, 0);
lean_inc(v_a_3702_);
lean_dec_ref_known(v___x_3701_, 1);
v_fst_3703_ = lean_ctor_get(v_a_3702_, 0);
lean_inc(v_fst_3703_);
v_snd_3704_ = lean_ctor_get(v_a_3702_, 1);
lean_inc(v_snd_3704_);
lean_dec(v_a_3702_);
lean_inc(v_fst_3699_);
v___x_3705_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3693_, v_fst_3699_, v_fst_3696_, v_snd_3704_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3705_) == 0)
{
lean_object* v_a_3706_; lean_object* v_fst_3707_; lean_object* v_snd_3708_; lean_object* v___x_3709_; lean_object* v___x_3710_; 
v_a_3706_ = lean_ctor_get(v___x_3705_, 0);
lean_inc(v_a_3706_);
lean_dec_ref_known(v___x_3705_, 1);
v_fst_3707_ = lean_ctor_get(v_a_3706_, 0);
lean_inc(v_fst_3707_);
v_snd_3708_ = lean_ctor_get(v_a_3706_, 1);
lean_inc(v_snd_3708_);
lean_dec(v_a_3706_);
v___x_3709_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3709_, 0, v_fst_3703_);
lean_ctor_set(v___x_3709_, 1, v_fst_3707_);
lean_ctor_set(v___x_3709_, 2, v_snd_3264_);
v___x_3710_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_fst_3699_, v___x_3709_, v_snd_3708_);
return v___x_3710_;
}
else
{
lean_object* v_a_3711_; lean_object* v___x_3713_; uint8_t v_isShared_3714_; uint8_t v_isSharedCheck_3718_; 
lean_dec(v_fst_3703_);
lean_dec(v_fst_3699_);
lean_dec(v_snd_3264_);
v_a_3711_ = lean_ctor_get(v___x_3705_, 0);
v_isSharedCheck_3718_ = !lean_is_exclusive(v___x_3705_);
if (v_isSharedCheck_3718_ == 0)
{
v___x_3713_ = v___x_3705_;
v_isShared_3714_ = v_isSharedCheck_3718_;
goto v_resetjp_3712_;
}
else
{
lean_inc(v_a_3711_);
lean_dec(v___x_3705_);
v___x_3713_ = lean_box(0);
v_isShared_3714_ = v_isSharedCheck_3718_;
goto v_resetjp_3712_;
}
v_resetjp_3712_:
{
lean_object* v___x_3716_; 
if (v_isShared_3714_ == 0)
{
v___x_3716_ = v___x_3713_;
goto v_reusejp_3715_;
}
else
{
lean_object* v_reuseFailAlloc_3717_; 
v_reuseFailAlloc_3717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3717_, 0, v_a_3711_);
v___x_3716_ = v_reuseFailAlloc_3717_;
goto v_reusejp_3715_;
}
v_reusejp_3715_:
{
return v___x_3716_;
}
}
}
}
else
{
lean_object* v_a_3719_; lean_object* v___x_3721_; uint8_t v_isShared_3722_; uint8_t v_isSharedCheck_3726_; 
lean_dec(v_fst_3699_);
lean_dec(v_fst_3696_);
lean_dec(v_fst_3693_);
lean_dec(v_snd_3264_);
v_a_3719_ = lean_ctor_get(v___x_3701_, 0);
v_isSharedCheck_3726_ = !lean_is_exclusive(v___x_3701_);
if (v_isSharedCheck_3726_ == 0)
{
v___x_3721_ = v___x_3701_;
v_isShared_3722_ = v_isSharedCheck_3726_;
goto v_resetjp_3720_;
}
else
{
lean_inc(v_a_3719_);
lean_dec(v___x_3701_);
v___x_3721_ = lean_box(0);
v_isShared_3722_ = v_isSharedCheck_3726_;
goto v_resetjp_3720_;
}
v_resetjp_3720_:
{
lean_object* v___x_3724_; 
if (v_isShared_3722_ == 0)
{
v___x_3724_ = v___x_3721_;
goto v_reusejp_3723_;
}
else
{
lean_object* v_reuseFailAlloc_3725_; 
v_reuseFailAlloc_3725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3725_, 0, v_a_3719_);
v___x_3724_ = v_reuseFailAlloc_3725_;
goto v_reusejp_3723_;
}
v_reusejp_3723_:
{
return v___x_3724_;
}
}
}
}
else
{
lean_object* v_a_3727_; lean_object* v___x_3729_; uint8_t v_isShared_3730_; uint8_t v_isSharedCheck_3734_; 
lean_dec(v_fst_3696_);
lean_dec(v_fst_3695_);
lean_dec(v_fst_3693_);
lean_dec(v_snd_3264_);
v_a_3727_ = lean_ctor_get(v___x_3697_, 0);
v_isSharedCheck_3734_ = !lean_is_exclusive(v___x_3697_);
if (v_isSharedCheck_3734_ == 0)
{
v___x_3729_ = v___x_3697_;
v_isShared_3730_ = v_isSharedCheck_3734_;
goto v_resetjp_3728_;
}
else
{
lean_inc(v_a_3727_);
lean_dec(v___x_3697_);
v___x_3729_ = lean_box(0);
v_isShared_3730_ = v_isSharedCheck_3734_;
goto v_resetjp_3728_;
}
v_resetjp_3728_:
{
lean_object* v___x_3732_; 
if (v_isShared_3730_ == 0)
{
v___x_3732_ = v___x_3729_;
goto v_reusejp_3731_;
}
else
{
lean_object* v_reuseFailAlloc_3733_; 
v_reuseFailAlloc_3733_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3733_, 0, v_a_3727_);
v___x_3732_ = v_reuseFailAlloc_3733_;
goto v_reusejp_3731_;
}
v_reusejp_3731_:
{
return v___x_3732_;
}
}
}
}
}
else
{
lean_object* v_a_3735_; lean_object* v___x_3737_; uint8_t v_isShared_3738_; uint8_t v_isSharedCheck_3742_; 
lean_dec(v_snd_3264_);
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3735_ = lean_ctor_get(v___x_3292_, 0);
v_isSharedCheck_3742_ = !lean_is_exclusive(v___x_3292_);
if (v_isSharedCheck_3742_ == 0)
{
v___x_3737_ = v___x_3292_;
v_isShared_3738_ = v_isSharedCheck_3742_;
goto v_resetjp_3736_;
}
else
{
lean_inc(v_a_3735_);
lean_dec(v___x_3292_);
v___x_3737_ = lean_box(0);
v_isShared_3738_ = v_isSharedCheck_3742_;
goto v_resetjp_3736_;
}
v_resetjp_3736_:
{
lean_object* v___x_3740_; 
if (v_isShared_3738_ == 0)
{
v___x_3740_ = v___x_3737_;
goto v_reusejp_3739_;
}
else
{
lean_object* v_reuseFailAlloc_3741_; 
v_reuseFailAlloc_3741_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3741_, 0, v_a_3735_);
v___x_3740_ = v_reuseFailAlloc_3741_;
goto v_reusejp_3739_;
}
v_reusejp_3739_:
{
return v___x_3740_;
}
}
}
}
else
{
lean_object* v___x_3743_; lean_object* v___x_3744_; lean_object* v___x_3746_; 
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
v___x_3743_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___closed__10));
v___x_3744_ = lean_box(0);
lean_inc(v_fst_3276_);
if (v_isShared_3281_ == 0)
{
lean_ctor_set_tag(v___x_3280_, 1);
lean_ctor_set(v___x_3280_, 1, v___x_3744_);
lean_ctor_set(v___x_3280_, 0, v_fst_3276_);
v___x_3746_ = v___x_3280_;
goto v_reusejp_3745_;
}
else
{
lean_object* v_reuseFailAlloc_3808_; 
v_reuseFailAlloc_3808_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3808_, 0, v_fst_3276_);
lean_ctor_set(v_reuseFailAlloc_3808_, 1, v___x_3744_);
v___x_3746_ = v_reuseFailAlloc_3808_;
goto v_reusejp_3745_;
}
v_reusejp_3745_:
{
lean_object* v___x_3747_; lean_object* v___x_3748_; lean_object* v___x_3749_; lean_object* v___x_3750_; 
v___x_3747_ = l_Lean_Expr_const___override(v___x_3743_, v___x_3746_);
lean_inc(v_fst_3277_);
v___x_3748_ = l_Lean_Expr_app___override(v___x_3747_, v_fst_3277_);
v___x_3749_ = lean_box(0);
v___x_3750_ = l_Lean_Meta_synthInstance_x3f(v___x_3748_, v___x_3749_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3750_) == 0)
{
lean_object* v_a_3751_; lean_object* v___x_3753_; uint8_t v_isShared_3754_; uint8_t v_isSharedCheck_3799_; 
v_a_3751_ = lean_ctor_get(v___x_3750_, 0);
v_isSharedCheck_3799_ = !lean_is_exclusive(v___x_3750_);
if (v_isSharedCheck_3799_ == 0)
{
v___x_3753_ = v___x_3750_;
v_isShared_3754_ = v_isSharedCheck_3799_;
goto v_resetjp_3752_;
}
else
{
lean_inc(v_a_3751_);
lean_dec(v___x_3750_);
v___x_3753_ = lean_box(0);
v_isShared_3754_ = v_isSharedCheck_3799_;
goto v_resetjp_3752_;
}
v_resetjp_3752_:
{
if (lean_obj_tag(v_a_3751_) == 0)
{
lean_object* v___x_3756_; 
lean_dec(v_fst_3282_);
lean_dec(v_fst_3278_);
lean_dec(v_fst_3277_);
lean_dec(v_fst_3276_);
lean_dec(v_snd_3264_);
if (v_isShared_3286_ == 0)
{
lean_ctor_set(v___x_3285_, 1, v_a_3239_);
lean_ctor_set(v___x_3285_, 0, v___x_3287_);
v___x_3756_ = v___x_3285_;
goto v_reusejp_3755_;
}
else
{
lean_object* v_reuseFailAlloc_3760_; 
v_reuseFailAlloc_3760_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3760_, 0, v___x_3287_);
lean_ctor_set(v_reuseFailAlloc_3760_, 1, v_a_3239_);
v___x_3756_ = v_reuseFailAlloc_3760_;
goto v_reusejp_3755_;
}
v_reusejp_3755_:
{
lean_object* v___x_3758_; 
if (v_isShared_3754_ == 0)
{
lean_ctor_set(v___x_3753_, 0, v___x_3756_);
v___x_3758_ = v___x_3753_;
goto v_reusejp_3757_;
}
else
{
lean_object* v_reuseFailAlloc_3759_; 
v_reuseFailAlloc_3759_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3759_, 0, v___x_3756_);
v___x_3758_ = v_reuseFailAlloc_3759_;
goto v_reusejp_3757_;
}
v_reusejp_3757_:
{
return v___x_3758_;
}
}
}
else
{
lean_object* v___x_3761_; 
lean_dec_ref_known(v_a_3751_, 1);
lean_del_object(v___x_3753_);
lean_del_object(v___x_3285_);
v___x_3761_ = lp_mathlib_Mathlib_Tactic_Order_addType___redArg(v_fst_3277_, v_a_3239_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3761_) == 0)
{
lean_object* v_a_3762_; lean_object* v_fst_3763_; lean_object* v_snd_3764_; lean_object* v___x_3765_; 
v_a_3762_ = lean_ctor_get(v___x_3761_, 0);
lean_inc(v_a_3762_);
lean_dec_ref_known(v___x_3761_, 1);
v_fst_3763_ = lean_ctor_get(v_a_3762_, 0);
lean_inc_n(v_fst_3763_, 2);
v_snd_3764_ = lean_ctor_get(v_a_3762_, 1);
lean_inc(v_snd_3764_);
lean_dec(v_a_3762_);
lean_inc(v_fst_3276_);
v___x_3765_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3276_, v_fst_3763_, v_fst_3278_, v_snd_3764_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3765_) == 0)
{
lean_object* v_a_3766_; lean_object* v_fst_3767_; lean_object* v_snd_3768_; lean_object* v___x_3769_; 
v_a_3766_ = lean_ctor_get(v___x_3765_, 0);
lean_inc(v_a_3766_);
lean_dec_ref_known(v___x_3765_, 1);
v_fst_3767_ = lean_ctor_get(v_a_3766_, 0);
lean_inc(v_fst_3767_);
v_snd_3768_ = lean_ctor_get(v_a_3766_, 1);
lean_inc(v_snd_3768_);
lean_dec(v_a_3766_);
lean_inc(v_fst_3763_);
v___x_3769_ = lp_mathlib_Mathlib_Tactic_Order_addAtom(v_fst_3276_, v_fst_3763_, v_fst_3282_, v_snd_3768_, v_a_3240_, v_a_3241_, v_a_3242_, v_a_3243_, v_a_3244_, v_a_3245_);
if (lean_obj_tag(v___x_3769_) == 0)
{
lean_object* v_a_3770_; lean_object* v_fst_3771_; lean_object* v_snd_3772_; lean_object* v___x_3773_; lean_object* v___x_3774_; 
v_a_3770_ = lean_ctor_get(v___x_3769_, 0);
lean_inc(v_a_3770_);
lean_dec_ref_known(v___x_3769_, 1);
v_fst_3771_ = lean_ctor_get(v_a_3770_, 0);
lean_inc(v_fst_3771_);
v_snd_3772_ = lean_ctor_get(v_a_3770_, 1);
lean_inc(v_snd_3772_);
lean_dec(v_a_3770_);
v___x_3773_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3773_, 0, v_fst_3767_);
lean_ctor_set(v___x_3773_, 1, v_fst_3771_);
lean_ctor_set(v___x_3773_, 2, v_snd_3264_);
v___x_3774_ = lp_mathlib_Mathlib_Tactic_Order_addFact___redArg(v_fst_3763_, v___x_3773_, v_snd_3772_);
return v___x_3774_;
}
else
{
lean_object* v_a_3775_; lean_object* v___x_3777_; uint8_t v_isShared_3778_; uint8_t v_isSharedCheck_3782_; 
lean_dec(v_fst_3767_);
lean_dec(v_fst_3763_);
lean_dec(v_snd_3264_);
v_a_3775_ = lean_ctor_get(v___x_3769_, 0);
v_isSharedCheck_3782_ = !lean_is_exclusive(v___x_3769_);
if (v_isSharedCheck_3782_ == 0)
{
v___x_3777_ = v___x_3769_;
v_isShared_3778_ = v_isSharedCheck_3782_;
goto v_resetjp_3776_;
}
else
{
lean_inc(v_a_3775_);
lean_dec(v___x_3769_);
v___x_3777_ = lean_box(0);
v_isShared_3778_ = v_isSharedCheck_3782_;
goto v_resetjp_3776_;
}
v_resetjp_3776_:
{
lean_object* v___x_3780_; 
if (v_isShared_3778_ == 0)
{
v___x_3780_ = v___x_3777_;
goto v_reusejp_3779_;
}
else
{
lean_object* v_reuseFailAlloc_3781_; 
v_reuseFailAlloc_3781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3781_, 0, v_a_3775_);
v___x_3780_ = v_reuseFailAlloc_3781_;
goto v_reusejp_3779_;
}
v_reusejp_3779_:
{
return v___x_3780_;
}
}
}
}
else
{
lean_object* v_a_3783_; lean_object* v___x_3785_; uint8_t v_isShared_3786_; uint8_t v_isSharedCheck_3790_; 
lean_dec(v_fst_3763_);
lean_dec(v_fst_3282_);
lean_dec(v_fst_3276_);
lean_dec(v_snd_3264_);
v_a_3783_ = lean_ctor_get(v___x_3765_, 0);
v_isSharedCheck_3790_ = !lean_is_exclusive(v___x_3765_);
if (v_isSharedCheck_3790_ == 0)
{
v___x_3785_ = v___x_3765_;
v_isShared_3786_ = v_isSharedCheck_3790_;
goto v_resetjp_3784_;
}
else
{
lean_inc(v_a_3783_);
lean_dec(v___x_3765_);
v___x_3785_ = lean_box(0);
v_isShared_3786_ = v_isSharedCheck_3790_;
goto v_resetjp_3784_;
}
v_resetjp_3784_:
{
lean_object* v___x_3788_; 
if (v_isShared_3786_ == 0)
{
v___x_3788_ = v___x_3785_;
goto v_reusejp_3787_;
}
else
{
lean_object* v_reuseFailAlloc_3789_; 
v_reuseFailAlloc_3789_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3789_, 0, v_a_3783_);
v___x_3788_ = v_reuseFailAlloc_3789_;
goto v_reusejp_3787_;
}
v_reusejp_3787_:
{
return v___x_3788_;
}
}
}
}
else
{
lean_object* v_a_3791_; lean_object* v___x_3793_; uint8_t v_isShared_3794_; uint8_t v_isSharedCheck_3798_; 
lean_dec(v_fst_3282_);
lean_dec(v_fst_3278_);
lean_dec(v_fst_3276_);
lean_dec(v_snd_3264_);
v_a_3791_ = lean_ctor_get(v___x_3761_, 0);
v_isSharedCheck_3798_ = !lean_is_exclusive(v___x_3761_);
if (v_isSharedCheck_3798_ == 0)
{
v___x_3793_ = v___x_3761_;
v_isShared_3794_ = v_isSharedCheck_3798_;
goto v_resetjp_3792_;
}
else
{
lean_inc(v_a_3791_);
lean_dec(v___x_3761_);
v___x_3793_ = lean_box(0);
v_isShared_3794_ = v_isSharedCheck_3798_;
goto v_resetjp_3792_;
}
v_resetjp_3792_:
{
lean_object* v___x_3796_; 
if (v_isShared_3794_ == 0)
{
v___x_3796_ = v___x_3793_;
goto v_reusejp_3795_;
}
else
{
lean_object* v_reuseFailAlloc_3797_; 
v_reuseFailAlloc_3797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3797_, 0, v_a_3791_);
v___x_3796_ = v_reuseFailAlloc_3797_;
goto v_reusejp_3795_;
}
v_reusejp_3795_:
{
return v___x_3796_;
}
}
}
}
}
}
else
{
lean_object* v_a_3800_; lean_object* v___x_3802_; uint8_t v_isShared_3803_; uint8_t v_isSharedCheck_3807_; 
lean_del_object(v___x_3285_);
lean_dec(v_fst_3282_);
lean_dec(v_fst_3278_);
lean_dec(v_fst_3277_);
lean_dec(v_fst_3276_);
lean_dec(v_snd_3264_);
lean_dec_ref(v_a_3239_);
v_a_3800_ = lean_ctor_get(v___x_3750_, 0);
v_isSharedCheck_3807_ = !lean_is_exclusive(v___x_3750_);
if (v_isSharedCheck_3807_ == 0)
{
v___x_3802_ = v___x_3750_;
v_isShared_3803_ = v_isSharedCheck_3807_;
goto v_resetjp_3801_;
}
else
{
lean_inc(v_a_3800_);
lean_dec(v___x_3750_);
v___x_3802_ = lean_box(0);
v_isShared_3803_ = v_isSharedCheck_3807_;
goto v_resetjp_3801_;
}
v_resetjp_3801_:
{
lean_object* v___x_3805_; 
if (v_isShared_3803_ == 0)
{
v___x_3805_ = v___x_3802_;
goto v_reusejp_3804_;
}
else
{
lean_object* v_reuseFailAlloc_3806_; 
v_reuseFailAlloc_3806_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3806_, 0, v_a_3800_);
v___x_3805_ = v_reuseFailAlloc_3806_;
goto v_reusejp_3804_;
}
v_reusejp_3804_:
{
return v___x_3805_;
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
lean_object* v_a_3812_; lean_object* v___x_3814_; uint8_t v_isShared_3815_; uint8_t v_isSharedCheck_3819_; 
lean_dec(v_snd_3264_);
lean_dec(v_fst_3263_);
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3812_ = lean_ctor_get(v___x_3271_, 0);
v_isSharedCheck_3819_ = !lean_is_exclusive(v___x_3271_);
if (v_isSharedCheck_3819_ == 0)
{
v___x_3814_ = v___x_3271_;
v_isShared_3815_ = v_isSharedCheck_3819_;
goto v_resetjp_3813_;
}
else
{
lean_inc(v_a_3812_);
lean_dec(v___x_3271_);
v___x_3814_ = lean_box(0);
v_isShared_3815_ = v_isSharedCheck_3819_;
goto v_resetjp_3813_;
}
v_resetjp_3813_:
{
lean_object* v___x_3817_; 
if (v_isShared_3815_ == 0)
{
v___x_3817_ = v___x_3814_;
goto v_reusejp_3816_;
}
else
{
lean_object* v_reuseFailAlloc_3818_; 
v_reuseFailAlloc_3818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3818_, 0, v_a_3812_);
v___x_3817_ = v_reuseFailAlloc_3818_;
goto v_reusejp_3816_;
}
v_reusejp_3816_:
{
return v___x_3817_;
}
}
}
}
else
{
lean_object* v_a_3820_; lean_object* v___x_3822_; uint8_t v_isShared_3823_; uint8_t v_isSharedCheck_3827_; 
lean_dec(v_a_3250_);
lean_dec_ref(v_a_3239_);
v_a_3820_ = lean_ctor_get(v___x_3260_, 0);
v_isSharedCheck_3827_ = !lean_is_exclusive(v___x_3260_);
if (v_isSharedCheck_3827_ == 0)
{
v___x_3822_ = v___x_3260_;
v_isShared_3823_ = v_isSharedCheck_3827_;
goto v_resetjp_3821_;
}
else
{
lean_inc(v_a_3820_);
lean_dec(v___x_3260_);
v___x_3822_ = lean_box(0);
v_isShared_3823_ = v_isSharedCheck_3827_;
goto v_resetjp_3821_;
}
v_resetjp_3821_:
{
lean_object* v___x_3825_; 
if (v_isShared_3823_ == 0)
{
v___x_3825_ = v___x_3822_;
goto v_reusejp_3824_;
}
else
{
lean_object* v_reuseFailAlloc_3826_; 
v_reuseFailAlloc_3826_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3826_, 0, v_a_3820_);
v___x_3825_ = v_reuseFailAlloc_3826_;
goto v_reusejp_3824_;
}
v_reusejp_3824_:
{
return v___x_3825_;
}
}
}
}
}
}
else
{
lean_object* v_a_3829_; lean_object* v___x_3831_; uint8_t v_isShared_3832_; uint8_t v_isSharedCheck_3836_; 
lean_dec_ref(v_a_3239_);
lean_dec_ref(v_expr_3238_);
v_a_3829_ = lean_ctor_get(v___x_3249_, 0);
v_isSharedCheck_3836_ = !lean_is_exclusive(v___x_3249_);
if (v_isSharedCheck_3836_ == 0)
{
v___x_3831_ = v___x_3249_;
v_isShared_3832_ = v_isSharedCheck_3836_;
goto v_resetjp_3830_;
}
else
{
lean_inc(v_a_3829_);
lean_dec(v___x_3249_);
v___x_3831_ = lean_box(0);
v_isShared_3832_ = v_isSharedCheck_3836_;
goto v_resetjp_3830_;
}
v_resetjp_3830_:
{
lean_object* v___x_3834_; 
if (v_isShared_3832_ == 0)
{
v___x_3834_ = v___x_3831_;
goto v_reusejp_3833_;
}
else
{
lean_object* v_reuseFailAlloc_3835_; 
v_reuseFailAlloc_3835_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3835_, 0, v_a_3829_);
v___x_3834_ = v_reuseFailAlloc_3835_;
goto v_reusejp_3833_;
}
v_reusejp_3833_:
{
return v___x_3834_;
}
}
}
}
else
{
lean_object* v_a_3837_; lean_object* v___x_3839_; uint8_t v_isShared_3840_; uint8_t v_isSharedCheck_3844_; 
lean_dec_ref(v_a_3239_);
lean_dec_ref(v_expr_3238_);
v_a_3837_ = lean_ctor_get(v___x_3247_, 0);
v_isSharedCheck_3844_ = !lean_is_exclusive(v___x_3247_);
if (v_isSharedCheck_3844_ == 0)
{
v___x_3839_ = v___x_3247_;
v_isShared_3840_ = v_isSharedCheck_3844_;
goto v_resetjp_3838_;
}
else
{
lean_inc(v_a_3837_);
lean_dec(v___x_3247_);
v___x_3839_ = lean_box(0);
v_isShared_3840_ = v_isSharedCheck_3844_;
goto v_resetjp_3838_;
}
v_resetjp_3838_:
{
lean_object* v___x_3842_; 
if (v_isShared_3840_ == 0)
{
v___x_3842_ = v___x_3839_;
goto v_reusejp_3841_;
}
else
{
lean_object* v_reuseFailAlloc_3843_; 
v_reuseFailAlloc_3843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3843_, 0, v_a_3837_);
v___x_3842_ = v_reuseFailAlloc_3843_;
goto v_reusejp_3841_;
}
v_reusejp_3841_:
{
return v___x_3842_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr___boxed(lean_object* v_expr_3845_, lean_object* v_a_3846_, lean_object* v_a_3847_, lean_object* v_a_3848_, lean_object* v_a_3849_, lean_object* v_a_3850_, lean_object* v_a_3851_, lean_object* v_a_3852_, lean_object* v_a_3853_){
_start:
{
lean_object* v_res_3854_; 
v_res_3854_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v_expr_3845_, v_a_3846_, v_a_3847_, v_a_3848_, v_a_3849_, v_a_3850_, v_a_3851_, v_a_3852_);
lean_dec(v_a_3852_);
lean_dec_ref(v_a_3851_);
lean_dec(v_a_3850_);
lean_dec_ref(v_a_3849_);
lean_dec(v_a_3848_);
lean_dec_ref(v_a_3847_);
return v_res_3854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_collectFactsImp_spec__0(lean_object* v_as_3855_, size_t v_sz_3856_, size_t v_i_3857_, lean_object* v_b_3858_, lean_object* v___y_3859_, lean_object* v___y_3860_, lean_object* v___y_3861_, lean_object* v___y_3862_, lean_object* v___y_3863_, lean_object* v___y_3864_, lean_object* v___y_3865_){
_start:
{
uint8_t v___x_3867_; 
v___x_3867_ = lean_usize_dec_lt(v_i_3857_, v_sz_3856_);
if (v___x_3867_ == 0)
{
lean_object* v___x_3868_; lean_object* v___x_3869_; 
v___x_3868_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3868_, 0, v_b_3858_);
lean_ctor_set(v___x_3868_, 1, v___y_3859_);
v___x_3869_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3869_, 0, v___x_3868_);
return v___x_3869_;
}
else
{
lean_object* v_a_3870_; lean_object* v___x_3871_; 
v_a_3870_ = lean_array_uget_borrowed(v_as_3855_, v_i_3857_);
lean_inc(v_a_3870_);
v___x_3871_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v_a_3870_, v___y_3859_, v___y_3860_, v___y_3861_, v___y_3862_, v___y_3863_, v___y_3864_, v___y_3865_);
if (lean_obj_tag(v___x_3871_) == 0)
{
lean_object* v_a_3872_; lean_object* v_snd_3873_; lean_object* v___x_3874_; size_t v___x_3875_; size_t v___x_3876_; 
v_a_3872_ = lean_ctor_get(v___x_3871_, 0);
lean_inc(v_a_3872_);
lean_dec_ref_known(v___x_3871_, 1);
v_snd_3873_ = lean_ctor_get(v_a_3872_, 1);
lean_inc(v_snd_3873_);
lean_dec(v_a_3872_);
v___x_3874_ = lean_box(0);
v___x_3875_ = ((size_t)1ULL);
v___x_3876_ = lean_usize_add(v_i_3857_, v___x_3875_);
v_i_3857_ = v___x_3876_;
v_b_3858_ = v___x_3874_;
v___y_3859_ = v_snd_3873_;
goto _start;
}
else
{
return v___x_3871_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_collectFactsImp_spec__0___boxed(lean_object* v_as_3878_, lean_object* v_sz_3879_, lean_object* v_i_3880_, lean_object* v_b_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_, lean_object* v___y_3884_, lean_object* v___y_3885_, lean_object* v___y_3886_, lean_object* v___y_3887_, lean_object* v___y_3888_, lean_object* v___y_3889_){
_start:
{
size_t v_sz_boxed_3890_; size_t v_i_boxed_3891_; lean_object* v_res_3892_; 
v_sz_boxed_3890_ = lean_unbox_usize(v_sz_3879_);
lean_dec(v_sz_3879_);
v_i_boxed_3891_ = lean_unbox_usize(v_i_3880_);
lean_dec(v_i_3880_);
v_res_3892_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_collectFactsImp_spec__0(v_as_3878_, v_sz_boxed_3890_, v_i_boxed_3891_, v_b_3881_, v___y_3882_, v___y_3883_, v___y_3884_, v___y_3885_, v___y_3886_, v___y_3887_, v___y_3888_);
lean_dec(v___y_3888_);
lean_dec_ref(v___y_3887_);
lean_dec(v___y_3886_);
lean_dec_ref(v___y_3885_);
lean_dec(v___y_3884_);
lean_dec_ref(v___y_3883_);
lean_dec_ref(v_as_3878_);
return v_res_3892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3_spec__4(lean_object* v_negGoal_3893_, lean_object* v_as_3894_, size_t v_sz_3895_, size_t v_i_3896_, lean_object* v_b_3897_, lean_object* v___y_3898_, lean_object* v___y_3899_, lean_object* v___y_3900_, lean_object* v___y_3901_, lean_object* v___y_3902_, lean_object* v___y_3903_, lean_object* v___y_3904_){
_start:
{
uint8_t v___x_3906_; 
v___x_3906_ = lean_usize_dec_lt(v_i_3896_, v_sz_3895_);
if (v___x_3906_ == 0)
{
lean_object* v___x_3907_; lean_object* v___x_3908_; 
v___x_3907_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3907_, 0, v_b_3897_);
lean_ctor_set(v___x_3907_, 1, v___y_3898_);
v___x_3908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3908_, 0, v___x_3907_);
return v___x_3908_;
}
else
{
lean_object* v_snd_3909_; lean_object* v___x_3911_; uint8_t v_isShared_3912_; uint8_t v_isSharedCheck_3940_; 
v_snd_3909_ = lean_ctor_get(v_b_3897_, 1);
v_isSharedCheck_3940_ = !lean_is_exclusive(v_b_3897_);
if (v_isSharedCheck_3940_ == 0)
{
lean_object* v_unused_3941_; 
v_unused_3941_ = lean_ctor_get(v_b_3897_, 0);
lean_dec(v_unused_3941_);
v___x_3911_ = v_b_3897_;
v_isShared_3912_ = v_isSharedCheck_3940_;
goto v_resetjp_3910_;
}
else
{
lean_inc(v_snd_3909_);
lean_dec(v_b_3897_);
v___x_3911_ = lean_box(0);
v_isShared_3912_ = v_isSharedCheck_3940_;
goto v_resetjp_3910_;
}
v_resetjp_3910_:
{
lean_object* v___x_3913_; lean_object* v_a_3915_; lean_object* v_snd_3916_; lean_object* v_a_3923_; 
v___x_3913_ = lean_box(0);
v_a_3923_ = lean_array_uget_borrowed(v_as_3894_, v_i_3896_);
if (lean_obj_tag(v_a_3923_) == 0)
{
v_a_3915_ = v_snd_3909_;
v_snd_3916_ = v___y_3898_;
goto v___jp_3914_;
}
else
{
lean_object* v_val_3924_; lean_object* v___x_3925_; uint8_t v___x_3926_; 
lean_dec(v_snd_3909_);
v_val_3924_ = lean_ctor_get(v_a_3923_, 0);
v___x_3925_ = lean_box(0);
v___x_3926_ = l_Lean_LocalDecl_isImplementationDetail(v_val_3924_);
if (v___x_3926_ == 0)
{
lean_object* v___x_3927_; uint8_t v___x_3928_; 
lean_inc(v_val_3924_);
v___x_3927_ = l_Lean_LocalDecl_toExpr(v_val_3924_);
v___x_3928_ = lean_expr_eqv(v___x_3927_, v_negGoal_3893_);
if (v___x_3928_ == 0)
{
lean_object* v___x_3929_; 
v___x_3929_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v___x_3927_, v___y_3898_, v___y_3899_, v___y_3900_, v___y_3901_, v___y_3902_, v___y_3903_, v___y_3904_);
if (lean_obj_tag(v___x_3929_) == 0)
{
lean_object* v_a_3930_; lean_object* v_snd_3931_; 
v_a_3930_ = lean_ctor_get(v___x_3929_, 0);
lean_inc(v_a_3930_);
lean_dec_ref_known(v___x_3929_, 1);
v_snd_3931_ = lean_ctor_get(v_a_3930_, 1);
lean_inc(v_snd_3931_);
lean_dec(v_a_3930_);
v_a_3915_ = v___x_3925_;
v_snd_3916_ = v_snd_3931_;
goto v___jp_3914_;
}
else
{
lean_object* v_a_3932_; lean_object* v___x_3934_; uint8_t v_isShared_3935_; uint8_t v_isSharedCheck_3939_; 
lean_del_object(v___x_3911_);
v_a_3932_ = lean_ctor_get(v___x_3929_, 0);
v_isSharedCheck_3939_ = !lean_is_exclusive(v___x_3929_);
if (v_isSharedCheck_3939_ == 0)
{
v___x_3934_ = v___x_3929_;
v_isShared_3935_ = v_isSharedCheck_3939_;
goto v_resetjp_3933_;
}
else
{
lean_inc(v_a_3932_);
lean_dec(v___x_3929_);
v___x_3934_ = lean_box(0);
v_isShared_3935_ = v_isSharedCheck_3939_;
goto v_resetjp_3933_;
}
v_resetjp_3933_:
{
lean_object* v___x_3937_; 
if (v_isShared_3935_ == 0)
{
v___x_3937_ = v___x_3934_;
goto v_reusejp_3936_;
}
else
{
lean_object* v_reuseFailAlloc_3938_; 
v_reuseFailAlloc_3938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3938_, 0, v_a_3932_);
v___x_3937_ = v_reuseFailAlloc_3938_;
goto v_reusejp_3936_;
}
v_reusejp_3936_:
{
return v___x_3937_;
}
}
}
}
else
{
lean_dec_ref(v___x_3927_);
v_a_3915_ = v___x_3925_;
v_snd_3916_ = v___y_3898_;
goto v___jp_3914_;
}
}
else
{
v_a_3915_ = v___x_3925_;
v_snd_3916_ = v___y_3898_;
goto v___jp_3914_;
}
}
v___jp_3914_:
{
lean_object* v___x_3918_; 
if (v_isShared_3912_ == 0)
{
lean_ctor_set(v___x_3911_, 1, v_a_3915_);
lean_ctor_set(v___x_3911_, 0, v___x_3913_);
v___x_3918_ = v___x_3911_;
goto v_reusejp_3917_;
}
else
{
lean_object* v_reuseFailAlloc_3922_; 
v_reuseFailAlloc_3922_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3922_, 0, v___x_3913_);
lean_ctor_set(v_reuseFailAlloc_3922_, 1, v_a_3915_);
v___x_3918_ = v_reuseFailAlloc_3922_;
goto v_reusejp_3917_;
}
v_reusejp_3917_:
{
size_t v___x_3919_; size_t v___x_3920_; 
v___x_3919_ = ((size_t)1ULL);
v___x_3920_ = lean_usize_add(v_i_3896_, v___x_3919_);
v_i_3896_ = v___x_3920_;
v_b_3897_ = v___x_3918_;
v___y_3898_ = v_snd_3916_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3_spec__4___boxed(lean_object* v_negGoal_3942_, lean_object* v_as_3943_, lean_object* v_sz_3944_, lean_object* v_i_3945_, lean_object* v_b_3946_, lean_object* v___y_3947_, lean_object* v___y_3948_, lean_object* v___y_3949_, lean_object* v___y_3950_, lean_object* v___y_3951_, lean_object* v___y_3952_, lean_object* v___y_3953_, lean_object* v___y_3954_){
_start:
{
size_t v_sz_boxed_3955_; size_t v_i_boxed_3956_; lean_object* v_res_3957_; 
v_sz_boxed_3955_ = lean_unbox_usize(v_sz_3944_);
lean_dec(v_sz_3944_);
v_i_boxed_3956_ = lean_unbox_usize(v_i_3945_);
lean_dec(v_i_3945_);
v_res_3957_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3_spec__4(v_negGoal_3942_, v_as_3943_, v_sz_boxed_3955_, v_i_boxed_3956_, v_b_3946_, v___y_3947_, v___y_3948_, v___y_3949_, v___y_3950_, v___y_3951_, v___y_3952_, v___y_3953_);
lean_dec(v___y_3953_);
lean_dec_ref(v___y_3952_);
lean_dec(v___y_3951_);
lean_dec_ref(v___y_3950_);
lean_dec(v___y_3949_);
lean_dec_ref(v___y_3948_);
lean_dec_ref(v_as_3943_);
lean_dec_ref(v_negGoal_3942_);
return v_res_3957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3(lean_object* v_negGoal_3958_, lean_object* v_as_3959_, size_t v_sz_3960_, size_t v_i_3961_, lean_object* v_b_3962_, lean_object* v___y_3963_, lean_object* v___y_3964_, lean_object* v___y_3965_, lean_object* v___y_3966_, lean_object* v___y_3967_, lean_object* v___y_3968_, lean_object* v___y_3969_){
_start:
{
uint8_t v___x_3971_; 
v___x_3971_ = lean_usize_dec_lt(v_i_3961_, v_sz_3960_);
if (v___x_3971_ == 0)
{
lean_object* v___x_3972_; lean_object* v___x_3973_; 
v___x_3972_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3972_, 0, v_b_3962_);
lean_ctor_set(v___x_3972_, 1, v___y_3963_);
v___x_3973_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3973_, 0, v___x_3972_);
return v___x_3973_;
}
else
{
lean_object* v_snd_3974_; lean_object* v___x_3976_; uint8_t v_isShared_3977_; uint8_t v_isSharedCheck_4005_; 
v_snd_3974_ = lean_ctor_get(v_b_3962_, 1);
v_isSharedCheck_4005_ = !lean_is_exclusive(v_b_3962_);
if (v_isSharedCheck_4005_ == 0)
{
lean_object* v_unused_4006_; 
v_unused_4006_ = lean_ctor_get(v_b_3962_, 0);
lean_dec(v_unused_4006_);
v___x_3976_ = v_b_3962_;
v_isShared_3977_ = v_isSharedCheck_4005_;
goto v_resetjp_3975_;
}
else
{
lean_inc(v_snd_3974_);
lean_dec(v_b_3962_);
v___x_3976_ = lean_box(0);
v_isShared_3977_ = v_isSharedCheck_4005_;
goto v_resetjp_3975_;
}
v_resetjp_3975_:
{
lean_object* v___x_3978_; lean_object* v_a_3980_; lean_object* v_snd_3981_; lean_object* v_a_3988_; 
v___x_3978_ = lean_box(0);
v_a_3988_ = lean_array_uget_borrowed(v_as_3959_, v_i_3961_);
if (lean_obj_tag(v_a_3988_) == 0)
{
v_a_3980_ = v_snd_3974_;
v_snd_3981_ = v___y_3963_;
goto v___jp_3979_;
}
else
{
lean_object* v_val_3989_; lean_object* v___x_3990_; uint8_t v___x_3991_; 
lean_dec(v_snd_3974_);
v_val_3989_ = lean_ctor_get(v_a_3988_, 0);
v___x_3990_ = lean_box(0);
v___x_3991_ = l_Lean_LocalDecl_isImplementationDetail(v_val_3989_);
if (v___x_3991_ == 0)
{
lean_object* v___x_3992_; uint8_t v___x_3993_; 
lean_inc(v_val_3989_);
v___x_3992_ = l_Lean_LocalDecl_toExpr(v_val_3989_);
v___x_3993_ = lean_expr_eqv(v___x_3992_, v_negGoal_3958_);
if (v___x_3993_ == 0)
{
lean_object* v___x_3994_; 
v___x_3994_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v___x_3992_, v___y_3963_, v___y_3964_, v___y_3965_, v___y_3966_, v___y_3967_, v___y_3968_, v___y_3969_);
if (lean_obj_tag(v___x_3994_) == 0)
{
lean_object* v_a_3995_; lean_object* v_snd_3996_; 
v_a_3995_ = lean_ctor_get(v___x_3994_, 0);
lean_inc(v_a_3995_);
lean_dec_ref_known(v___x_3994_, 1);
v_snd_3996_ = lean_ctor_get(v_a_3995_, 1);
lean_inc(v_snd_3996_);
lean_dec(v_a_3995_);
v_a_3980_ = v___x_3990_;
v_snd_3981_ = v_snd_3996_;
goto v___jp_3979_;
}
else
{
lean_object* v_a_3997_; lean_object* v___x_3999_; uint8_t v_isShared_4000_; uint8_t v_isSharedCheck_4004_; 
lean_del_object(v___x_3976_);
v_a_3997_ = lean_ctor_get(v___x_3994_, 0);
v_isSharedCheck_4004_ = !lean_is_exclusive(v___x_3994_);
if (v_isSharedCheck_4004_ == 0)
{
v___x_3999_ = v___x_3994_;
v_isShared_4000_ = v_isSharedCheck_4004_;
goto v_resetjp_3998_;
}
else
{
lean_inc(v_a_3997_);
lean_dec(v___x_3994_);
v___x_3999_ = lean_box(0);
v_isShared_4000_ = v_isSharedCheck_4004_;
goto v_resetjp_3998_;
}
v_resetjp_3998_:
{
lean_object* v___x_4002_; 
if (v_isShared_4000_ == 0)
{
v___x_4002_ = v___x_3999_;
goto v_reusejp_4001_;
}
else
{
lean_object* v_reuseFailAlloc_4003_; 
v_reuseFailAlloc_4003_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4003_, 0, v_a_3997_);
v___x_4002_ = v_reuseFailAlloc_4003_;
goto v_reusejp_4001_;
}
v_reusejp_4001_:
{
return v___x_4002_;
}
}
}
}
else
{
lean_dec_ref(v___x_3992_);
v_a_3980_ = v___x_3990_;
v_snd_3981_ = v___y_3963_;
goto v___jp_3979_;
}
}
else
{
v_a_3980_ = v___x_3990_;
v_snd_3981_ = v___y_3963_;
goto v___jp_3979_;
}
}
v___jp_3979_:
{
lean_object* v___x_3983_; 
if (v_isShared_3977_ == 0)
{
lean_ctor_set(v___x_3976_, 1, v_a_3980_);
lean_ctor_set(v___x_3976_, 0, v___x_3978_);
v___x_3983_ = v___x_3976_;
goto v_reusejp_3982_;
}
else
{
lean_object* v_reuseFailAlloc_3987_; 
v_reuseFailAlloc_3987_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3987_, 0, v___x_3978_);
lean_ctor_set(v_reuseFailAlloc_3987_, 1, v_a_3980_);
v___x_3983_ = v_reuseFailAlloc_3987_;
goto v_reusejp_3982_;
}
v_reusejp_3982_:
{
size_t v___x_3984_; size_t v___x_3985_; lean_object* v___x_3986_; 
v___x_3984_ = ((size_t)1ULL);
v___x_3985_ = lean_usize_add(v_i_3961_, v___x_3984_);
v___x_3986_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3_spec__4(v_negGoal_3958_, v_as_3959_, v_sz_3960_, v___x_3985_, v___x_3983_, v_snd_3981_, v___y_3964_, v___y_3965_, v___y_3966_, v___y_3967_, v___y_3968_, v___y_3969_);
return v___x_3986_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3___boxed(lean_object* v_negGoal_4007_, lean_object* v_as_4008_, lean_object* v_sz_4009_, lean_object* v_i_4010_, lean_object* v_b_4011_, lean_object* v___y_4012_, lean_object* v___y_4013_, lean_object* v___y_4014_, lean_object* v___y_4015_, lean_object* v___y_4016_, lean_object* v___y_4017_, lean_object* v___y_4018_, lean_object* v___y_4019_){
_start:
{
size_t v_sz_boxed_4020_; size_t v_i_boxed_4021_; lean_object* v_res_4022_; 
v_sz_boxed_4020_ = lean_unbox_usize(v_sz_4009_);
lean_dec(v_sz_4009_);
v_i_boxed_4021_ = lean_unbox_usize(v_i_4010_);
lean_dec(v_i_4010_);
v_res_4022_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3(v_negGoal_4007_, v_as_4008_, v_sz_boxed_4020_, v_i_boxed_4021_, v_b_4011_, v___y_4012_, v___y_4013_, v___y_4014_, v___y_4015_, v___y_4016_, v___y_4017_, v___y_4018_);
lean_dec(v___y_4018_);
lean_dec_ref(v___y_4017_);
lean_dec(v___y_4016_);
lean_dec_ref(v___y_4015_);
lean_dec(v___y_4014_);
lean_dec_ref(v___y_4013_);
lean_dec_ref(v_as_4008_);
lean_dec_ref(v_negGoal_4007_);
return v_res_4022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1(lean_object* v_init_4023_, lean_object* v_negGoal_4024_, lean_object* v_n_4025_, lean_object* v_b_4026_, lean_object* v___y_4027_, lean_object* v___y_4028_, lean_object* v___y_4029_, lean_object* v___y_4030_, lean_object* v___y_4031_, lean_object* v___y_4032_, lean_object* v___y_4033_){
_start:
{
if (lean_obj_tag(v_n_4025_) == 0)
{
lean_object* v_cs_4035_; lean_object* v___x_4036_; lean_object* v___x_4037_; size_t v_sz_4038_; size_t v___x_4039_; lean_object* v___x_4040_; 
v_cs_4035_ = lean_ctor_get(v_n_4025_, 0);
v___x_4036_ = lean_box(0);
v___x_4037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4037_, 0, v___x_4036_);
lean_ctor_set(v___x_4037_, 1, v_b_4026_);
v_sz_4038_ = lean_array_size(v_cs_4035_);
v___x_4039_ = ((size_t)0ULL);
v___x_4040_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__2(v_init_4023_, v_negGoal_4024_, v_cs_4035_, v_sz_4038_, v___x_4039_, v___x_4037_, v___y_4027_, v___y_4028_, v___y_4029_, v___y_4030_, v___y_4031_, v___y_4032_, v___y_4033_);
if (lean_obj_tag(v___x_4040_) == 0)
{
lean_object* v_a_4041_; lean_object* v___x_4043_; uint8_t v_isShared_4044_; uint8_t v_isSharedCheck_4075_; 
v_a_4041_ = lean_ctor_get(v___x_4040_, 0);
v_isSharedCheck_4075_ = !lean_is_exclusive(v___x_4040_);
if (v_isSharedCheck_4075_ == 0)
{
v___x_4043_ = v___x_4040_;
v_isShared_4044_ = v_isSharedCheck_4075_;
goto v_resetjp_4042_;
}
else
{
lean_inc(v_a_4041_);
lean_dec(v___x_4040_);
v___x_4043_ = lean_box(0);
v_isShared_4044_ = v_isSharedCheck_4075_;
goto v_resetjp_4042_;
}
v_resetjp_4042_:
{
lean_object* v_fst_4045_; lean_object* v_fst_4046_; 
v_fst_4045_ = lean_ctor_get(v_a_4041_, 0);
lean_inc(v_fst_4045_);
v_fst_4046_ = lean_ctor_get(v_fst_4045_, 0);
if (lean_obj_tag(v_fst_4046_) == 0)
{
lean_object* v_snd_4047_; lean_object* v_snd_4048_; lean_object* v___x_4050_; uint8_t v_isShared_4051_; uint8_t v_isSharedCheck_4059_; 
v_snd_4047_ = lean_ctor_get(v_a_4041_, 1);
lean_inc(v_snd_4047_);
lean_dec(v_a_4041_);
v_snd_4048_ = lean_ctor_get(v_fst_4045_, 1);
v_isSharedCheck_4059_ = !lean_is_exclusive(v_fst_4045_);
if (v_isSharedCheck_4059_ == 0)
{
lean_object* v_unused_4060_; 
v_unused_4060_ = lean_ctor_get(v_fst_4045_, 0);
lean_dec(v_unused_4060_);
v___x_4050_ = v_fst_4045_;
v_isShared_4051_ = v_isSharedCheck_4059_;
goto v_resetjp_4049_;
}
else
{
lean_inc(v_snd_4048_);
lean_dec(v_fst_4045_);
v___x_4050_ = lean_box(0);
v_isShared_4051_ = v_isSharedCheck_4059_;
goto v_resetjp_4049_;
}
v_resetjp_4049_:
{
lean_object* v___x_4052_; lean_object* v___x_4054_; 
v___x_4052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4052_, 0, v_snd_4048_);
if (v_isShared_4051_ == 0)
{
lean_ctor_set(v___x_4050_, 1, v_snd_4047_);
lean_ctor_set(v___x_4050_, 0, v___x_4052_);
v___x_4054_ = v___x_4050_;
goto v_reusejp_4053_;
}
else
{
lean_object* v_reuseFailAlloc_4058_; 
v_reuseFailAlloc_4058_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4058_, 0, v___x_4052_);
lean_ctor_set(v_reuseFailAlloc_4058_, 1, v_snd_4047_);
v___x_4054_ = v_reuseFailAlloc_4058_;
goto v_reusejp_4053_;
}
v_reusejp_4053_:
{
lean_object* v___x_4056_; 
if (v_isShared_4044_ == 0)
{
lean_ctor_set(v___x_4043_, 0, v___x_4054_);
v___x_4056_ = v___x_4043_;
goto v_reusejp_4055_;
}
else
{
lean_object* v_reuseFailAlloc_4057_; 
v_reuseFailAlloc_4057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4057_, 0, v___x_4054_);
v___x_4056_ = v_reuseFailAlloc_4057_;
goto v_reusejp_4055_;
}
v_reusejp_4055_:
{
return v___x_4056_;
}
}
}
}
else
{
lean_object* v___x_4062_; uint8_t v_isShared_4063_; uint8_t v_isSharedCheck_4072_; 
lean_inc_ref(v_fst_4046_);
v_isSharedCheck_4072_ = !lean_is_exclusive(v_fst_4045_);
if (v_isSharedCheck_4072_ == 0)
{
lean_object* v_unused_4073_; lean_object* v_unused_4074_; 
v_unused_4073_ = lean_ctor_get(v_fst_4045_, 1);
lean_dec(v_unused_4073_);
v_unused_4074_ = lean_ctor_get(v_fst_4045_, 0);
lean_dec(v_unused_4074_);
v___x_4062_ = v_fst_4045_;
v_isShared_4063_ = v_isSharedCheck_4072_;
goto v_resetjp_4061_;
}
else
{
lean_dec(v_fst_4045_);
v___x_4062_ = lean_box(0);
v_isShared_4063_ = v_isSharedCheck_4072_;
goto v_resetjp_4061_;
}
v_resetjp_4061_:
{
lean_object* v_snd_4064_; lean_object* v_val_4065_; lean_object* v___x_4067_; 
v_snd_4064_ = lean_ctor_get(v_a_4041_, 1);
lean_inc(v_snd_4064_);
lean_dec(v_a_4041_);
v_val_4065_ = lean_ctor_get(v_fst_4046_, 0);
lean_inc(v_val_4065_);
lean_dec_ref_known(v_fst_4046_, 1);
if (v_isShared_4063_ == 0)
{
lean_ctor_set(v___x_4062_, 1, v_snd_4064_);
lean_ctor_set(v___x_4062_, 0, v_val_4065_);
v___x_4067_ = v___x_4062_;
goto v_reusejp_4066_;
}
else
{
lean_object* v_reuseFailAlloc_4071_; 
v_reuseFailAlloc_4071_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4071_, 0, v_val_4065_);
lean_ctor_set(v_reuseFailAlloc_4071_, 1, v_snd_4064_);
v___x_4067_ = v_reuseFailAlloc_4071_;
goto v_reusejp_4066_;
}
v_reusejp_4066_:
{
lean_object* v___x_4069_; 
if (v_isShared_4044_ == 0)
{
lean_ctor_set(v___x_4043_, 0, v___x_4067_);
v___x_4069_ = v___x_4043_;
goto v_reusejp_4068_;
}
else
{
lean_object* v_reuseFailAlloc_4070_; 
v_reuseFailAlloc_4070_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4070_, 0, v___x_4067_);
v___x_4069_ = v_reuseFailAlloc_4070_;
goto v_reusejp_4068_;
}
v_reusejp_4068_:
{
return v___x_4069_;
}
}
}
}
}
}
else
{
lean_object* v_a_4076_; lean_object* v___x_4078_; uint8_t v_isShared_4079_; uint8_t v_isSharedCheck_4083_; 
v_a_4076_ = lean_ctor_get(v___x_4040_, 0);
v_isSharedCheck_4083_ = !lean_is_exclusive(v___x_4040_);
if (v_isSharedCheck_4083_ == 0)
{
v___x_4078_ = v___x_4040_;
v_isShared_4079_ = v_isSharedCheck_4083_;
goto v_resetjp_4077_;
}
else
{
lean_inc(v_a_4076_);
lean_dec(v___x_4040_);
v___x_4078_ = lean_box(0);
v_isShared_4079_ = v_isSharedCheck_4083_;
goto v_resetjp_4077_;
}
v_resetjp_4077_:
{
lean_object* v___x_4081_; 
if (v_isShared_4079_ == 0)
{
v___x_4081_ = v___x_4078_;
goto v_reusejp_4080_;
}
else
{
lean_object* v_reuseFailAlloc_4082_; 
v_reuseFailAlloc_4082_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4082_, 0, v_a_4076_);
v___x_4081_ = v_reuseFailAlloc_4082_;
goto v_reusejp_4080_;
}
v_reusejp_4080_:
{
return v___x_4081_;
}
}
}
}
else
{
lean_object* v_vs_4084_; lean_object* v___x_4085_; lean_object* v___x_4086_; size_t v_sz_4087_; size_t v___x_4088_; lean_object* v___x_4089_; 
v_vs_4084_ = lean_ctor_get(v_n_4025_, 0);
v___x_4085_ = lean_box(0);
v___x_4086_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4086_, 0, v___x_4085_);
lean_ctor_set(v___x_4086_, 1, v_b_4026_);
v_sz_4087_ = lean_array_size(v_vs_4084_);
v___x_4088_ = ((size_t)0ULL);
v___x_4089_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__3(v_negGoal_4024_, v_vs_4084_, v_sz_4087_, v___x_4088_, v___x_4086_, v___y_4027_, v___y_4028_, v___y_4029_, v___y_4030_, v___y_4031_, v___y_4032_, v___y_4033_);
if (lean_obj_tag(v___x_4089_) == 0)
{
lean_object* v_a_4090_; lean_object* v___x_4092_; uint8_t v_isShared_4093_; uint8_t v_isSharedCheck_4124_; 
v_a_4090_ = lean_ctor_get(v___x_4089_, 0);
v_isSharedCheck_4124_ = !lean_is_exclusive(v___x_4089_);
if (v_isSharedCheck_4124_ == 0)
{
v___x_4092_ = v___x_4089_;
v_isShared_4093_ = v_isSharedCheck_4124_;
goto v_resetjp_4091_;
}
else
{
lean_inc(v_a_4090_);
lean_dec(v___x_4089_);
v___x_4092_ = lean_box(0);
v_isShared_4093_ = v_isSharedCheck_4124_;
goto v_resetjp_4091_;
}
v_resetjp_4091_:
{
lean_object* v_fst_4094_; lean_object* v_fst_4095_; 
v_fst_4094_ = lean_ctor_get(v_a_4090_, 0);
lean_inc(v_fst_4094_);
v_fst_4095_ = lean_ctor_get(v_fst_4094_, 0);
if (lean_obj_tag(v_fst_4095_) == 0)
{
lean_object* v_snd_4096_; lean_object* v_snd_4097_; lean_object* v___x_4099_; uint8_t v_isShared_4100_; uint8_t v_isSharedCheck_4108_; 
v_snd_4096_ = lean_ctor_get(v_a_4090_, 1);
lean_inc(v_snd_4096_);
lean_dec(v_a_4090_);
v_snd_4097_ = lean_ctor_get(v_fst_4094_, 1);
v_isSharedCheck_4108_ = !lean_is_exclusive(v_fst_4094_);
if (v_isSharedCheck_4108_ == 0)
{
lean_object* v_unused_4109_; 
v_unused_4109_ = lean_ctor_get(v_fst_4094_, 0);
lean_dec(v_unused_4109_);
v___x_4099_ = v_fst_4094_;
v_isShared_4100_ = v_isSharedCheck_4108_;
goto v_resetjp_4098_;
}
else
{
lean_inc(v_snd_4097_);
lean_dec(v_fst_4094_);
v___x_4099_ = lean_box(0);
v_isShared_4100_ = v_isSharedCheck_4108_;
goto v_resetjp_4098_;
}
v_resetjp_4098_:
{
lean_object* v___x_4101_; lean_object* v___x_4103_; 
v___x_4101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4101_, 0, v_snd_4097_);
if (v_isShared_4100_ == 0)
{
lean_ctor_set(v___x_4099_, 1, v_snd_4096_);
lean_ctor_set(v___x_4099_, 0, v___x_4101_);
v___x_4103_ = v___x_4099_;
goto v_reusejp_4102_;
}
else
{
lean_object* v_reuseFailAlloc_4107_; 
v_reuseFailAlloc_4107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4107_, 0, v___x_4101_);
lean_ctor_set(v_reuseFailAlloc_4107_, 1, v_snd_4096_);
v___x_4103_ = v_reuseFailAlloc_4107_;
goto v_reusejp_4102_;
}
v_reusejp_4102_:
{
lean_object* v___x_4105_; 
if (v_isShared_4093_ == 0)
{
lean_ctor_set(v___x_4092_, 0, v___x_4103_);
v___x_4105_ = v___x_4092_;
goto v_reusejp_4104_;
}
else
{
lean_object* v_reuseFailAlloc_4106_; 
v_reuseFailAlloc_4106_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4106_, 0, v___x_4103_);
v___x_4105_ = v_reuseFailAlloc_4106_;
goto v_reusejp_4104_;
}
v_reusejp_4104_:
{
return v___x_4105_;
}
}
}
}
else
{
lean_object* v___x_4111_; uint8_t v_isShared_4112_; uint8_t v_isSharedCheck_4121_; 
lean_inc_ref(v_fst_4095_);
v_isSharedCheck_4121_ = !lean_is_exclusive(v_fst_4094_);
if (v_isSharedCheck_4121_ == 0)
{
lean_object* v_unused_4122_; lean_object* v_unused_4123_; 
v_unused_4122_ = lean_ctor_get(v_fst_4094_, 1);
lean_dec(v_unused_4122_);
v_unused_4123_ = lean_ctor_get(v_fst_4094_, 0);
lean_dec(v_unused_4123_);
v___x_4111_ = v_fst_4094_;
v_isShared_4112_ = v_isSharedCheck_4121_;
goto v_resetjp_4110_;
}
else
{
lean_dec(v_fst_4094_);
v___x_4111_ = lean_box(0);
v_isShared_4112_ = v_isSharedCheck_4121_;
goto v_resetjp_4110_;
}
v_resetjp_4110_:
{
lean_object* v_snd_4113_; lean_object* v_val_4114_; lean_object* v___x_4116_; 
v_snd_4113_ = lean_ctor_get(v_a_4090_, 1);
lean_inc(v_snd_4113_);
lean_dec(v_a_4090_);
v_val_4114_ = lean_ctor_get(v_fst_4095_, 0);
lean_inc(v_val_4114_);
lean_dec_ref_known(v_fst_4095_, 1);
if (v_isShared_4112_ == 0)
{
lean_ctor_set(v___x_4111_, 1, v_snd_4113_);
lean_ctor_set(v___x_4111_, 0, v_val_4114_);
v___x_4116_ = v___x_4111_;
goto v_reusejp_4115_;
}
else
{
lean_object* v_reuseFailAlloc_4120_; 
v_reuseFailAlloc_4120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4120_, 0, v_val_4114_);
lean_ctor_set(v_reuseFailAlloc_4120_, 1, v_snd_4113_);
v___x_4116_ = v_reuseFailAlloc_4120_;
goto v_reusejp_4115_;
}
v_reusejp_4115_:
{
lean_object* v___x_4118_; 
if (v_isShared_4093_ == 0)
{
lean_ctor_set(v___x_4092_, 0, v___x_4116_);
v___x_4118_ = v___x_4092_;
goto v_reusejp_4117_;
}
else
{
lean_object* v_reuseFailAlloc_4119_; 
v_reuseFailAlloc_4119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4119_, 0, v___x_4116_);
v___x_4118_ = v_reuseFailAlloc_4119_;
goto v_reusejp_4117_;
}
v_reusejp_4117_:
{
return v___x_4118_;
}
}
}
}
}
}
else
{
lean_object* v_a_4125_; lean_object* v___x_4127_; uint8_t v_isShared_4128_; uint8_t v_isSharedCheck_4132_; 
v_a_4125_ = lean_ctor_get(v___x_4089_, 0);
v_isSharedCheck_4132_ = !lean_is_exclusive(v___x_4089_);
if (v_isSharedCheck_4132_ == 0)
{
v___x_4127_ = v___x_4089_;
v_isShared_4128_ = v_isSharedCheck_4132_;
goto v_resetjp_4126_;
}
else
{
lean_inc(v_a_4125_);
lean_dec(v___x_4089_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__2(lean_object* v_init_4133_, lean_object* v_negGoal_4134_, lean_object* v_as_4135_, size_t v_sz_4136_, size_t v_i_4137_, lean_object* v_b_4138_, lean_object* v___y_4139_, lean_object* v___y_4140_, lean_object* v___y_4141_, lean_object* v___y_4142_, lean_object* v___y_4143_, lean_object* v___y_4144_, lean_object* v___y_4145_){
_start:
{
uint8_t v___x_4147_; 
v___x_4147_ = lean_usize_dec_lt(v_i_4137_, v_sz_4136_);
if (v___x_4147_ == 0)
{
lean_object* v___x_4148_; lean_object* v___x_4149_; 
v___x_4148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4148_, 0, v_b_4138_);
lean_ctor_set(v___x_4148_, 1, v___y_4139_);
v___x_4149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4149_, 0, v___x_4148_);
return v___x_4149_;
}
else
{
lean_object* v_snd_4150_; lean_object* v___x_4152_; uint8_t v_isShared_4153_; uint8_t v_isSharedCheck_4200_; 
v_snd_4150_ = lean_ctor_get(v_b_4138_, 1);
v_isSharedCheck_4200_ = !lean_is_exclusive(v_b_4138_);
if (v_isSharedCheck_4200_ == 0)
{
lean_object* v_unused_4201_; 
v_unused_4201_ = lean_ctor_get(v_b_4138_, 0);
lean_dec(v_unused_4201_);
v___x_4152_ = v_b_4138_;
v_isShared_4153_ = v_isSharedCheck_4200_;
goto v_resetjp_4151_;
}
else
{
lean_inc(v_snd_4150_);
lean_dec(v_b_4138_);
v___x_4152_ = lean_box(0);
v_isShared_4153_ = v_isSharedCheck_4200_;
goto v_resetjp_4151_;
}
v_resetjp_4151_:
{
lean_object* v_a_4154_; lean_object* v___x_4155_; 
v_a_4154_ = lean_array_uget_borrowed(v_as_4135_, v_i_4137_);
lean_inc(v_snd_4150_);
v___x_4155_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1(v_init_4133_, v_negGoal_4134_, v_a_4154_, v_snd_4150_, v___y_4139_, v___y_4140_, v___y_4141_, v___y_4142_, v___y_4143_, v___y_4144_, v___y_4145_);
if (lean_obj_tag(v___x_4155_) == 0)
{
lean_object* v_a_4156_; lean_object* v___x_4158_; uint8_t v_isShared_4159_; uint8_t v_isSharedCheck_4191_; 
v_a_4156_ = lean_ctor_get(v___x_4155_, 0);
v_isSharedCheck_4191_ = !lean_is_exclusive(v___x_4155_);
if (v_isSharedCheck_4191_ == 0)
{
v___x_4158_ = v___x_4155_;
v_isShared_4159_ = v_isSharedCheck_4191_;
goto v_resetjp_4157_;
}
else
{
lean_inc(v_a_4156_);
lean_dec(v___x_4155_);
v___x_4158_ = lean_box(0);
v_isShared_4159_ = v_isSharedCheck_4191_;
goto v_resetjp_4157_;
}
v_resetjp_4157_:
{
lean_object* v_fst_4160_; 
v_fst_4160_ = lean_ctor_get(v_a_4156_, 0);
lean_inc(v_fst_4160_);
if (lean_obj_tag(v_fst_4160_) == 0)
{
lean_object* v_snd_4161_; lean_object* v___x_4163_; uint8_t v_isShared_4164_; uint8_t v_isSharedCheck_4175_; 
v_snd_4161_ = lean_ctor_get(v_a_4156_, 1);
v_isSharedCheck_4175_ = !lean_is_exclusive(v_a_4156_);
if (v_isSharedCheck_4175_ == 0)
{
lean_object* v_unused_4176_; 
v_unused_4176_ = lean_ctor_get(v_a_4156_, 0);
lean_dec(v_unused_4176_);
v___x_4163_ = v_a_4156_;
v_isShared_4164_ = v_isSharedCheck_4175_;
goto v_resetjp_4162_;
}
else
{
lean_inc(v_snd_4161_);
lean_dec(v_a_4156_);
v___x_4163_ = lean_box(0);
v_isShared_4164_ = v_isSharedCheck_4175_;
goto v_resetjp_4162_;
}
v_resetjp_4162_:
{
lean_object* v___x_4165_; lean_object* v___x_4167_; 
v___x_4165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4165_, 0, v_fst_4160_);
if (v_isShared_4164_ == 0)
{
lean_ctor_set(v___x_4163_, 1, v_snd_4150_);
lean_ctor_set(v___x_4163_, 0, v___x_4165_);
v___x_4167_ = v___x_4163_;
goto v_reusejp_4166_;
}
else
{
lean_object* v_reuseFailAlloc_4174_; 
v_reuseFailAlloc_4174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4174_, 0, v___x_4165_);
lean_ctor_set(v_reuseFailAlloc_4174_, 1, v_snd_4150_);
v___x_4167_ = v_reuseFailAlloc_4174_;
goto v_reusejp_4166_;
}
v_reusejp_4166_:
{
lean_object* v___x_4169_; 
if (v_isShared_4153_ == 0)
{
lean_ctor_set(v___x_4152_, 1, v_snd_4161_);
lean_ctor_set(v___x_4152_, 0, v___x_4167_);
v___x_4169_ = v___x_4152_;
goto v_reusejp_4168_;
}
else
{
lean_object* v_reuseFailAlloc_4173_; 
v_reuseFailAlloc_4173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4173_, 0, v___x_4167_);
lean_ctor_set(v_reuseFailAlloc_4173_, 1, v_snd_4161_);
v___x_4169_ = v_reuseFailAlloc_4173_;
goto v_reusejp_4168_;
}
v_reusejp_4168_:
{
lean_object* v___x_4171_; 
if (v_isShared_4159_ == 0)
{
lean_ctor_set(v___x_4158_, 0, v___x_4169_);
v___x_4171_ = v___x_4158_;
goto v_reusejp_4170_;
}
else
{
lean_object* v_reuseFailAlloc_4172_; 
v_reuseFailAlloc_4172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4172_, 0, v___x_4169_);
v___x_4171_ = v_reuseFailAlloc_4172_;
goto v_reusejp_4170_;
}
v_reusejp_4170_:
{
return v___x_4171_;
}
}
}
}
}
else
{
lean_object* v_snd_4177_; lean_object* v___x_4179_; uint8_t v_isShared_4180_; uint8_t v_isSharedCheck_4189_; 
lean_del_object(v___x_4158_);
lean_del_object(v___x_4152_);
lean_dec(v_snd_4150_);
v_snd_4177_ = lean_ctor_get(v_a_4156_, 1);
v_isSharedCheck_4189_ = !lean_is_exclusive(v_a_4156_);
if (v_isSharedCheck_4189_ == 0)
{
lean_object* v_unused_4190_; 
v_unused_4190_ = lean_ctor_get(v_a_4156_, 0);
lean_dec(v_unused_4190_);
v___x_4179_ = v_a_4156_;
v_isShared_4180_ = v_isSharedCheck_4189_;
goto v_resetjp_4178_;
}
else
{
lean_inc(v_snd_4177_);
lean_dec(v_a_4156_);
v___x_4179_ = lean_box(0);
v_isShared_4180_ = v_isSharedCheck_4189_;
goto v_resetjp_4178_;
}
v_resetjp_4178_:
{
lean_object* v_a_4181_; lean_object* v___x_4182_; lean_object* v___x_4184_; 
v_a_4181_ = lean_ctor_get(v_fst_4160_, 0);
lean_inc(v_a_4181_);
lean_dec_ref_known(v_fst_4160_, 1);
v___x_4182_ = lean_box(0);
if (v_isShared_4180_ == 0)
{
lean_ctor_set(v___x_4179_, 1, v_a_4181_);
lean_ctor_set(v___x_4179_, 0, v___x_4182_);
v___x_4184_ = v___x_4179_;
goto v_reusejp_4183_;
}
else
{
lean_object* v_reuseFailAlloc_4188_; 
v_reuseFailAlloc_4188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4188_, 0, v___x_4182_);
lean_ctor_set(v_reuseFailAlloc_4188_, 1, v_a_4181_);
v___x_4184_ = v_reuseFailAlloc_4188_;
goto v_reusejp_4183_;
}
v_reusejp_4183_:
{
size_t v___x_4185_; size_t v___x_4186_; 
v___x_4185_ = ((size_t)1ULL);
v___x_4186_ = lean_usize_add(v_i_4137_, v___x_4185_);
v_i_4137_ = v___x_4186_;
v_b_4138_ = v___x_4184_;
v___y_4139_ = v_snd_4177_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_4192_; lean_object* v___x_4194_; uint8_t v_isShared_4195_; uint8_t v_isSharedCheck_4199_; 
lean_del_object(v___x_4152_);
lean_dec(v_snd_4150_);
v_a_4192_ = lean_ctor_get(v___x_4155_, 0);
v_isSharedCheck_4199_ = !lean_is_exclusive(v___x_4155_);
if (v_isSharedCheck_4199_ == 0)
{
v___x_4194_ = v___x_4155_;
v_isShared_4195_ = v_isSharedCheck_4199_;
goto v_resetjp_4193_;
}
else
{
lean_inc(v_a_4192_);
lean_dec(v___x_4155_);
v___x_4194_ = lean_box(0);
v_isShared_4195_ = v_isSharedCheck_4199_;
goto v_resetjp_4193_;
}
v_resetjp_4193_:
{
lean_object* v___x_4197_; 
if (v_isShared_4195_ == 0)
{
v___x_4197_ = v___x_4194_;
goto v_reusejp_4196_;
}
else
{
lean_object* v_reuseFailAlloc_4198_; 
v_reuseFailAlloc_4198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4198_, 0, v_a_4192_);
v___x_4197_ = v_reuseFailAlloc_4198_;
goto v_reusejp_4196_;
}
v_reusejp_4196_:
{
return v___x_4197_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__2___boxed(lean_object* v_init_4202_, lean_object* v_negGoal_4203_, lean_object* v_as_4204_, lean_object* v_sz_4205_, lean_object* v_i_4206_, lean_object* v_b_4207_, lean_object* v___y_4208_, lean_object* v___y_4209_, lean_object* v___y_4210_, lean_object* v___y_4211_, lean_object* v___y_4212_, lean_object* v___y_4213_, lean_object* v___y_4214_, lean_object* v___y_4215_){
_start:
{
size_t v_sz_boxed_4216_; size_t v_i_boxed_4217_; lean_object* v_res_4218_; 
v_sz_boxed_4216_ = lean_unbox_usize(v_sz_4205_);
lean_dec(v_sz_4205_);
v_i_boxed_4217_ = lean_unbox_usize(v_i_4206_);
lean_dec(v_i_4206_);
v_res_4218_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1_spec__2(v_init_4202_, v_negGoal_4203_, v_as_4204_, v_sz_boxed_4216_, v_i_boxed_4217_, v_b_4207_, v___y_4208_, v___y_4209_, v___y_4210_, v___y_4211_, v___y_4212_, v___y_4213_, v___y_4214_);
lean_dec(v___y_4214_);
lean_dec_ref(v___y_4213_);
lean_dec(v___y_4212_);
lean_dec_ref(v___y_4211_);
lean_dec(v___y_4210_);
lean_dec_ref(v___y_4209_);
lean_dec_ref(v_as_4204_);
lean_dec_ref(v_negGoal_4203_);
return v_res_4218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1___boxed(lean_object* v_init_4219_, lean_object* v_negGoal_4220_, lean_object* v_n_4221_, lean_object* v_b_4222_, lean_object* v___y_4223_, lean_object* v___y_4224_, lean_object* v___y_4225_, lean_object* v___y_4226_, lean_object* v___y_4227_, lean_object* v___y_4228_, lean_object* v___y_4229_, lean_object* v___y_4230_){
_start:
{
lean_object* v_res_4231_; 
v_res_4231_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1(v_init_4219_, v_negGoal_4220_, v_n_4221_, v_b_4222_, v___y_4223_, v___y_4224_, v___y_4225_, v___y_4226_, v___y_4227_, v___y_4228_, v___y_4229_);
lean_dec(v___y_4229_);
lean_dec_ref(v___y_4228_);
lean_dec(v___y_4227_);
lean_dec_ref(v___y_4226_);
lean_dec(v___y_4225_);
lean_dec_ref(v___y_4224_);
lean_dec_ref(v_n_4221_);
lean_dec_ref(v_negGoal_4220_);
return v_res_4231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2_spec__5(lean_object* v_negGoal_4232_, lean_object* v_as_4233_, size_t v_sz_4234_, size_t v_i_4235_, lean_object* v_b_4236_, lean_object* v___y_4237_, lean_object* v___y_4238_, lean_object* v___y_4239_, lean_object* v___y_4240_, lean_object* v___y_4241_, lean_object* v___y_4242_, lean_object* v___y_4243_){
_start:
{
uint8_t v___x_4245_; 
v___x_4245_ = lean_usize_dec_lt(v_i_4235_, v_sz_4234_);
if (v___x_4245_ == 0)
{
lean_object* v___x_4246_; lean_object* v___x_4247_; 
v___x_4246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4246_, 0, v_b_4236_);
lean_ctor_set(v___x_4246_, 1, v___y_4237_);
v___x_4247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4247_, 0, v___x_4246_);
return v___x_4247_;
}
else
{
lean_object* v_snd_4248_; lean_object* v___x_4250_; uint8_t v_isShared_4251_; uint8_t v_isSharedCheck_4279_; 
v_snd_4248_ = lean_ctor_get(v_b_4236_, 1);
v_isSharedCheck_4279_ = !lean_is_exclusive(v_b_4236_);
if (v_isSharedCheck_4279_ == 0)
{
lean_object* v_unused_4280_; 
v_unused_4280_ = lean_ctor_get(v_b_4236_, 0);
lean_dec(v_unused_4280_);
v___x_4250_ = v_b_4236_;
v_isShared_4251_ = v_isSharedCheck_4279_;
goto v_resetjp_4249_;
}
else
{
lean_inc(v_snd_4248_);
lean_dec(v_b_4236_);
v___x_4250_ = lean_box(0);
v_isShared_4251_ = v_isSharedCheck_4279_;
goto v_resetjp_4249_;
}
v_resetjp_4249_:
{
lean_object* v___x_4252_; lean_object* v_a_4254_; lean_object* v_snd_4255_; lean_object* v_a_4262_; 
v___x_4252_ = lean_box(0);
v_a_4262_ = lean_array_uget_borrowed(v_as_4233_, v_i_4235_);
if (lean_obj_tag(v_a_4262_) == 0)
{
v_a_4254_ = v_snd_4248_;
v_snd_4255_ = v___y_4237_;
goto v___jp_4253_;
}
else
{
lean_object* v_val_4263_; lean_object* v___x_4264_; uint8_t v___x_4265_; 
lean_dec(v_snd_4248_);
v_val_4263_ = lean_ctor_get(v_a_4262_, 0);
v___x_4264_ = lean_box(0);
v___x_4265_ = l_Lean_LocalDecl_isImplementationDetail(v_val_4263_);
if (v___x_4265_ == 0)
{
lean_object* v___x_4266_; uint8_t v___x_4267_; 
lean_inc(v_val_4263_);
v___x_4266_ = l_Lean_LocalDecl_toExpr(v_val_4263_);
v___x_4267_ = lean_expr_eqv(v___x_4266_, v_negGoal_4232_);
if (v___x_4267_ == 0)
{
lean_object* v___x_4268_; 
v___x_4268_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v___x_4266_, v___y_4237_, v___y_4238_, v___y_4239_, v___y_4240_, v___y_4241_, v___y_4242_, v___y_4243_);
if (lean_obj_tag(v___x_4268_) == 0)
{
lean_object* v_a_4269_; lean_object* v_snd_4270_; 
v_a_4269_ = lean_ctor_get(v___x_4268_, 0);
lean_inc(v_a_4269_);
lean_dec_ref_known(v___x_4268_, 1);
v_snd_4270_ = lean_ctor_get(v_a_4269_, 1);
lean_inc(v_snd_4270_);
lean_dec(v_a_4269_);
v_a_4254_ = v___x_4264_;
v_snd_4255_ = v_snd_4270_;
goto v___jp_4253_;
}
else
{
lean_object* v_a_4271_; lean_object* v___x_4273_; uint8_t v_isShared_4274_; uint8_t v_isSharedCheck_4278_; 
lean_del_object(v___x_4250_);
v_a_4271_ = lean_ctor_get(v___x_4268_, 0);
v_isSharedCheck_4278_ = !lean_is_exclusive(v___x_4268_);
if (v_isSharedCheck_4278_ == 0)
{
v___x_4273_ = v___x_4268_;
v_isShared_4274_ = v_isSharedCheck_4278_;
goto v_resetjp_4272_;
}
else
{
lean_inc(v_a_4271_);
lean_dec(v___x_4268_);
v___x_4273_ = lean_box(0);
v_isShared_4274_ = v_isSharedCheck_4278_;
goto v_resetjp_4272_;
}
v_resetjp_4272_:
{
lean_object* v___x_4276_; 
if (v_isShared_4274_ == 0)
{
v___x_4276_ = v___x_4273_;
goto v_reusejp_4275_;
}
else
{
lean_object* v_reuseFailAlloc_4277_; 
v_reuseFailAlloc_4277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4277_, 0, v_a_4271_);
v___x_4276_ = v_reuseFailAlloc_4277_;
goto v_reusejp_4275_;
}
v_reusejp_4275_:
{
return v___x_4276_;
}
}
}
}
else
{
lean_dec_ref(v___x_4266_);
v_a_4254_ = v___x_4264_;
v_snd_4255_ = v___y_4237_;
goto v___jp_4253_;
}
}
else
{
v_a_4254_ = v___x_4264_;
v_snd_4255_ = v___y_4237_;
goto v___jp_4253_;
}
}
v___jp_4253_:
{
lean_object* v___x_4257_; 
if (v_isShared_4251_ == 0)
{
lean_ctor_set(v___x_4250_, 1, v_a_4254_);
lean_ctor_set(v___x_4250_, 0, v___x_4252_);
v___x_4257_ = v___x_4250_;
goto v_reusejp_4256_;
}
else
{
lean_object* v_reuseFailAlloc_4261_; 
v_reuseFailAlloc_4261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4261_, 0, v___x_4252_);
lean_ctor_set(v_reuseFailAlloc_4261_, 1, v_a_4254_);
v___x_4257_ = v_reuseFailAlloc_4261_;
goto v_reusejp_4256_;
}
v_reusejp_4256_:
{
size_t v___x_4258_; size_t v___x_4259_; 
v___x_4258_ = ((size_t)1ULL);
v___x_4259_ = lean_usize_add(v_i_4235_, v___x_4258_);
v_i_4235_ = v___x_4259_;
v_b_4236_ = v___x_4257_;
v___y_4237_ = v_snd_4255_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2_spec__5___boxed(lean_object* v_negGoal_4281_, lean_object* v_as_4282_, lean_object* v_sz_4283_, lean_object* v_i_4284_, lean_object* v_b_4285_, lean_object* v___y_4286_, lean_object* v___y_4287_, lean_object* v___y_4288_, lean_object* v___y_4289_, lean_object* v___y_4290_, lean_object* v___y_4291_, lean_object* v___y_4292_, lean_object* v___y_4293_){
_start:
{
size_t v_sz_boxed_4294_; size_t v_i_boxed_4295_; lean_object* v_res_4296_; 
v_sz_boxed_4294_ = lean_unbox_usize(v_sz_4283_);
lean_dec(v_sz_4283_);
v_i_boxed_4295_ = lean_unbox_usize(v_i_4284_);
lean_dec(v_i_4284_);
v_res_4296_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2_spec__5(v_negGoal_4281_, v_as_4282_, v_sz_boxed_4294_, v_i_boxed_4295_, v_b_4285_, v___y_4286_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_, v___y_4291_, v___y_4292_);
lean_dec(v___y_4292_);
lean_dec_ref(v___y_4291_);
lean_dec(v___y_4290_);
lean_dec_ref(v___y_4289_);
lean_dec(v___y_4288_);
lean_dec_ref(v___y_4287_);
lean_dec_ref(v_as_4282_);
lean_dec_ref(v_negGoal_4281_);
return v_res_4296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2(lean_object* v_negGoal_4297_, lean_object* v_as_4298_, size_t v_sz_4299_, size_t v_i_4300_, lean_object* v_b_4301_, lean_object* v___y_4302_, lean_object* v___y_4303_, lean_object* v___y_4304_, lean_object* v___y_4305_, lean_object* v___y_4306_, lean_object* v___y_4307_, lean_object* v___y_4308_){
_start:
{
uint8_t v___x_4310_; 
v___x_4310_ = lean_usize_dec_lt(v_i_4300_, v_sz_4299_);
if (v___x_4310_ == 0)
{
lean_object* v___x_4311_; lean_object* v___x_4312_; 
v___x_4311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4311_, 0, v_b_4301_);
lean_ctor_set(v___x_4311_, 1, v___y_4302_);
v___x_4312_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4312_, 0, v___x_4311_);
return v___x_4312_;
}
else
{
lean_object* v_snd_4313_; lean_object* v___x_4315_; uint8_t v_isShared_4316_; uint8_t v_isSharedCheck_4344_; 
v_snd_4313_ = lean_ctor_get(v_b_4301_, 1);
v_isSharedCheck_4344_ = !lean_is_exclusive(v_b_4301_);
if (v_isSharedCheck_4344_ == 0)
{
lean_object* v_unused_4345_; 
v_unused_4345_ = lean_ctor_get(v_b_4301_, 0);
lean_dec(v_unused_4345_);
v___x_4315_ = v_b_4301_;
v_isShared_4316_ = v_isSharedCheck_4344_;
goto v_resetjp_4314_;
}
else
{
lean_inc(v_snd_4313_);
lean_dec(v_b_4301_);
v___x_4315_ = lean_box(0);
v_isShared_4316_ = v_isSharedCheck_4344_;
goto v_resetjp_4314_;
}
v_resetjp_4314_:
{
lean_object* v___x_4317_; lean_object* v_a_4319_; lean_object* v_snd_4320_; lean_object* v_a_4327_; 
v___x_4317_ = lean_box(0);
v_a_4327_ = lean_array_uget_borrowed(v_as_4298_, v_i_4300_);
if (lean_obj_tag(v_a_4327_) == 0)
{
v_a_4319_ = v_snd_4313_;
v_snd_4320_ = v___y_4302_;
goto v___jp_4318_;
}
else
{
lean_object* v_val_4328_; lean_object* v___x_4329_; uint8_t v___x_4330_; 
lean_dec(v_snd_4313_);
v_val_4328_ = lean_ctor_get(v_a_4327_, 0);
v___x_4329_ = lean_box(0);
v___x_4330_ = l_Lean_LocalDecl_isImplementationDetail(v_val_4328_);
if (v___x_4330_ == 0)
{
lean_object* v___x_4331_; uint8_t v___x_4332_; 
lean_inc(v_val_4328_);
v___x_4331_ = l_Lean_LocalDecl_toExpr(v_val_4328_);
v___x_4332_ = lean_expr_eqv(v___x_4331_, v_negGoal_4297_);
if (v___x_4332_ == 0)
{
lean_object* v___x_4333_; 
v___x_4333_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v___x_4331_, v___y_4302_, v___y_4303_, v___y_4304_, v___y_4305_, v___y_4306_, v___y_4307_, v___y_4308_);
if (lean_obj_tag(v___x_4333_) == 0)
{
lean_object* v_a_4334_; lean_object* v_snd_4335_; 
v_a_4334_ = lean_ctor_get(v___x_4333_, 0);
lean_inc(v_a_4334_);
lean_dec_ref_known(v___x_4333_, 1);
v_snd_4335_ = lean_ctor_get(v_a_4334_, 1);
lean_inc(v_snd_4335_);
lean_dec(v_a_4334_);
v_a_4319_ = v___x_4329_;
v_snd_4320_ = v_snd_4335_;
goto v___jp_4318_;
}
else
{
lean_object* v_a_4336_; lean_object* v___x_4338_; uint8_t v_isShared_4339_; uint8_t v_isSharedCheck_4343_; 
lean_del_object(v___x_4315_);
v_a_4336_ = lean_ctor_get(v___x_4333_, 0);
v_isSharedCheck_4343_ = !lean_is_exclusive(v___x_4333_);
if (v_isSharedCheck_4343_ == 0)
{
v___x_4338_ = v___x_4333_;
v_isShared_4339_ = v_isSharedCheck_4343_;
goto v_resetjp_4337_;
}
else
{
lean_inc(v_a_4336_);
lean_dec(v___x_4333_);
v___x_4338_ = lean_box(0);
v_isShared_4339_ = v_isSharedCheck_4343_;
goto v_resetjp_4337_;
}
v_resetjp_4337_:
{
lean_object* v___x_4341_; 
if (v_isShared_4339_ == 0)
{
v___x_4341_ = v___x_4338_;
goto v_reusejp_4340_;
}
else
{
lean_object* v_reuseFailAlloc_4342_; 
v_reuseFailAlloc_4342_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4342_, 0, v_a_4336_);
v___x_4341_ = v_reuseFailAlloc_4342_;
goto v_reusejp_4340_;
}
v_reusejp_4340_:
{
return v___x_4341_;
}
}
}
}
else
{
lean_dec_ref(v___x_4331_);
v_a_4319_ = v___x_4329_;
v_snd_4320_ = v___y_4302_;
goto v___jp_4318_;
}
}
else
{
v_a_4319_ = v___x_4329_;
v_snd_4320_ = v___y_4302_;
goto v___jp_4318_;
}
}
v___jp_4318_:
{
lean_object* v___x_4322_; 
if (v_isShared_4316_ == 0)
{
lean_ctor_set(v___x_4315_, 1, v_a_4319_);
lean_ctor_set(v___x_4315_, 0, v___x_4317_);
v___x_4322_ = v___x_4315_;
goto v_reusejp_4321_;
}
else
{
lean_object* v_reuseFailAlloc_4326_; 
v_reuseFailAlloc_4326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4326_, 0, v___x_4317_);
lean_ctor_set(v_reuseFailAlloc_4326_, 1, v_a_4319_);
v___x_4322_ = v_reuseFailAlloc_4326_;
goto v_reusejp_4321_;
}
v_reusejp_4321_:
{
size_t v___x_4323_; size_t v___x_4324_; lean_object* v___x_4325_; 
v___x_4323_ = ((size_t)1ULL);
v___x_4324_ = lean_usize_add(v_i_4300_, v___x_4323_);
v___x_4325_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2_spec__5(v_negGoal_4297_, v_as_4298_, v_sz_4299_, v___x_4324_, v___x_4322_, v_snd_4320_, v___y_4303_, v___y_4304_, v___y_4305_, v___y_4306_, v___y_4307_, v___y_4308_);
return v___x_4325_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2___boxed(lean_object* v_negGoal_4346_, lean_object* v_as_4347_, lean_object* v_sz_4348_, lean_object* v_i_4349_, lean_object* v_b_4350_, lean_object* v___y_4351_, lean_object* v___y_4352_, lean_object* v___y_4353_, lean_object* v___y_4354_, lean_object* v___y_4355_, lean_object* v___y_4356_, lean_object* v___y_4357_, lean_object* v___y_4358_){
_start:
{
size_t v_sz_boxed_4359_; size_t v_i_boxed_4360_; lean_object* v_res_4361_; 
v_sz_boxed_4359_ = lean_unbox_usize(v_sz_4348_);
lean_dec(v_sz_4348_);
v_i_boxed_4360_ = lean_unbox_usize(v_i_4349_);
lean_dec(v_i_4349_);
v_res_4361_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2(v_negGoal_4346_, v_as_4347_, v_sz_boxed_4359_, v_i_boxed_4360_, v_b_4350_, v___y_4351_, v___y_4352_, v___y_4353_, v___y_4354_, v___y_4355_, v___y_4356_, v___y_4357_);
lean_dec(v___y_4357_);
lean_dec_ref(v___y_4356_);
lean_dec(v___y_4355_);
lean_dec_ref(v___y_4354_);
lean_dec(v___y_4353_);
lean_dec_ref(v___y_4352_);
lean_dec_ref(v_as_4347_);
lean_dec_ref(v_negGoal_4346_);
return v_res_4361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1(lean_object* v_negGoal_4362_, lean_object* v_t_4363_, lean_object* v_init_4364_, lean_object* v___y_4365_, lean_object* v___y_4366_, lean_object* v___y_4367_, lean_object* v___y_4368_, lean_object* v___y_4369_, lean_object* v___y_4370_, lean_object* v___y_4371_){
_start:
{
lean_object* v_b_4374_; lean_object* v___y_4375_; lean_object* v_root_4378_; lean_object* v_tail_4379_; lean_object* v___x_4380_; 
v_root_4378_ = lean_ctor_get(v_t_4363_, 0);
v_tail_4379_ = lean_ctor_get(v_t_4363_, 1);
v___x_4380_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__1(v_init_4364_, v_negGoal_4362_, v_root_4378_, v_init_4364_, v___y_4365_, v___y_4366_, v___y_4367_, v___y_4368_, v___y_4369_, v___y_4370_, v___y_4371_);
if (lean_obj_tag(v___x_4380_) == 0)
{
lean_object* v_a_4381_; lean_object* v_fst_4382_; 
v_a_4381_ = lean_ctor_get(v___x_4380_, 0);
lean_inc(v_a_4381_);
lean_dec_ref_known(v___x_4380_, 1);
v_fst_4382_ = lean_ctor_get(v_a_4381_, 0);
lean_inc(v_fst_4382_);
if (lean_obj_tag(v_fst_4382_) == 0)
{
lean_object* v_snd_4383_; lean_object* v_a_4384_; 
v_snd_4383_ = lean_ctor_get(v_a_4381_, 1);
lean_inc(v_snd_4383_);
lean_dec(v_a_4381_);
v_a_4384_ = lean_ctor_get(v_fst_4382_, 0);
lean_inc(v_a_4384_);
lean_dec_ref_known(v_fst_4382_, 1);
v_b_4374_ = v_a_4384_;
v___y_4375_ = v_snd_4383_;
goto v___jp_4373_;
}
else
{
lean_object* v_snd_4385_; lean_object* v___x_4387_; uint8_t v_isShared_4388_; uint8_t v_isSharedCheck_4427_; 
v_snd_4385_ = lean_ctor_get(v_a_4381_, 1);
v_isSharedCheck_4427_ = !lean_is_exclusive(v_a_4381_);
if (v_isSharedCheck_4427_ == 0)
{
lean_object* v_unused_4428_; 
v_unused_4428_ = lean_ctor_get(v_a_4381_, 0);
lean_dec(v_unused_4428_);
v___x_4387_ = v_a_4381_;
v_isShared_4388_ = v_isSharedCheck_4427_;
goto v_resetjp_4386_;
}
else
{
lean_inc(v_snd_4385_);
lean_dec(v_a_4381_);
v___x_4387_ = lean_box(0);
v_isShared_4388_ = v_isSharedCheck_4427_;
goto v_resetjp_4386_;
}
v_resetjp_4386_:
{
lean_object* v_a_4389_; lean_object* v___x_4390_; lean_object* v___x_4392_; 
v_a_4389_ = lean_ctor_get(v_fst_4382_, 0);
lean_inc(v_a_4389_);
lean_dec_ref_known(v_fst_4382_, 1);
v___x_4390_ = lean_box(0);
if (v_isShared_4388_ == 0)
{
lean_ctor_set(v___x_4387_, 1, v_a_4389_);
lean_ctor_set(v___x_4387_, 0, v___x_4390_);
v___x_4392_ = v___x_4387_;
goto v_reusejp_4391_;
}
else
{
lean_object* v_reuseFailAlloc_4426_; 
v_reuseFailAlloc_4426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4426_, 0, v___x_4390_);
lean_ctor_set(v_reuseFailAlloc_4426_, 1, v_a_4389_);
v___x_4392_ = v_reuseFailAlloc_4426_;
goto v_reusejp_4391_;
}
v_reusejp_4391_:
{
size_t v_sz_4393_; size_t v___x_4394_; lean_object* v___x_4395_; 
v_sz_4393_ = lean_array_size(v_tail_4379_);
v___x_4394_ = ((size_t)0ULL);
v___x_4395_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1_spec__2(v_negGoal_4362_, v_tail_4379_, v_sz_4393_, v___x_4394_, v___x_4392_, v_snd_4385_, v___y_4366_, v___y_4367_, v___y_4368_, v___y_4369_, v___y_4370_, v___y_4371_);
if (lean_obj_tag(v___x_4395_) == 0)
{
lean_object* v_a_4396_; lean_object* v___x_4398_; uint8_t v_isShared_4399_; uint8_t v_isSharedCheck_4417_; 
v_a_4396_ = lean_ctor_get(v___x_4395_, 0);
v_isSharedCheck_4417_ = !lean_is_exclusive(v___x_4395_);
if (v_isSharedCheck_4417_ == 0)
{
v___x_4398_ = v___x_4395_;
v_isShared_4399_ = v_isSharedCheck_4417_;
goto v_resetjp_4397_;
}
else
{
lean_inc(v_a_4396_);
lean_dec(v___x_4395_);
v___x_4398_ = lean_box(0);
v_isShared_4399_ = v_isSharedCheck_4417_;
goto v_resetjp_4397_;
}
v_resetjp_4397_:
{
lean_object* v_fst_4400_; lean_object* v_fst_4401_; 
v_fst_4400_ = lean_ctor_get(v_a_4396_, 0);
lean_inc(v_fst_4400_);
v_fst_4401_ = lean_ctor_get(v_fst_4400_, 0);
if (lean_obj_tag(v_fst_4401_) == 0)
{
lean_object* v_snd_4402_; lean_object* v_snd_4403_; lean_object* v___x_4405_; uint8_t v_isShared_4406_; uint8_t v_isSharedCheck_4413_; 
v_snd_4402_ = lean_ctor_get(v_a_4396_, 1);
lean_inc(v_snd_4402_);
lean_dec(v_a_4396_);
v_snd_4403_ = lean_ctor_get(v_fst_4400_, 1);
v_isSharedCheck_4413_ = !lean_is_exclusive(v_fst_4400_);
if (v_isSharedCheck_4413_ == 0)
{
lean_object* v_unused_4414_; 
v_unused_4414_ = lean_ctor_get(v_fst_4400_, 0);
lean_dec(v_unused_4414_);
v___x_4405_ = v_fst_4400_;
v_isShared_4406_ = v_isSharedCheck_4413_;
goto v_resetjp_4404_;
}
else
{
lean_inc(v_snd_4403_);
lean_dec(v_fst_4400_);
v___x_4405_ = lean_box(0);
v_isShared_4406_ = v_isSharedCheck_4413_;
goto v_resetjp_4404_;
}
v_resetjp_4404_:
{
lean_object* v___x_4408_; 
if (v_isShared_4406_ == 0)
{
lean_ctor_set(v___x_4405_, 1, v_snd_4402_);
lean_ctor_set(v___x_4405_, 0, v_snd_4403_);
v___x_4408_ = v___x_4405_;
goto v_reusejp_4407_;
}
else
{
lean_object* v_reuseFailAlloc_4412_; 
v_reuseFailAlloc_4412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4412_, 0, v_snd_4403_);
lean_ctor_set(v_reuseFailAlloc_4412_, 1, v_snd_4402_);
v___x_4408_ = v_reuseFailAlloc_4412_;
goto v_reusejp_4407_;
}
v_reusejp_4407_:
{
lean_object* v___x_4410_; 
if (v_isShared_4399_ == 0)
{
lean_ctor_set(v___x_4398_, 0, v___x_4408_);
v___x_4410_ = v___x_4398_;
goto v_reusejp_4409_;
}
else
{
lean_object* v_reuseFailAlloc_4411_; 
v_reuseFailAlloc_4411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4411_, 0, v___x_4408_);
v___x_4410_ = v_reuseFailAlloc_4411_;
goto v_reusejp_4409_;
}
v_reusejp_4409_:
{
return v___x_4410_;
}
}
}
}
else
{
lean_object* v_snd_4415_; lean_object* v_val_4416_; 
lean_inc_ref(v_fst_4401_);
lean_dec(v_fst_4400_);
lean_del_object(v___x_4398_);
v_snd_4415_ = lean_ctor_get(v_a_4396_, 1);
lean_inc(v_snd_4415_);
lean_dec(v_a_4396_);
v_val_4416_ = lean_ctor_get(v_fst_4401_, 0);
lean_inc(v_val_4416_);
lean_dec_ref_known(v_fst_4401_, 1);
v_b_4374_ = v_val_4416_;
v___y_4375_ = v_snd_4415_;
goto v___jp_4373_;
}
}
}
else
{
lean_object* v_a_4418_; lean_object* v___x_4420_; uint8_t v_isShared_4421_; uint8_t v_isSharedCheck_4425_; 
v_a_4418_ = lean_ctor_get(v___x_4395_, 0);
v_isSharedCheck_4425_ = !lean_is_exclusive(v___x_4395_);
if (v_isSharedCheck_4425_ == 0)
{
v___x_4420_ = v___x_4395_;
v_isShared_4421_ = v_isSharedCheck_4425_;
goto v_resetjp_4419_;
}
else
{
lean_inc(v_a_4418_);
lean_dec(v___x_4395_);
v___x_4420_ = lean_box(0);
v_isShared_4421_ = v_isSharedCheck_4425_;
goto v_resetjp_4419_;
}
v_resetjp_4419_:
{
lean_object* v___x_4423_; 
if (v_isShared_4421_ == 0)
{
v___x_4423_ = v___x_4420_;
goto v_reusejp_4422_;
}
else
{
lean_object* v_reuseFailAlloc_4424_; 
v_reuseFailAlloc_4424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4424_, 0, v_a_4418_);
v___x_4423_ = v_reuseFailAlloc_4424_;
goto v_reusejp_4422_;
}
v_reusejp_4422_:
{
return v___x_4423_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_4429_; lean_object* v___x_4431_; uint8_t v_isShared_4432_; uint8_t v_isSharedCheck_4436_; 
v_a_4429_ = lean_ctor_get(v___x_4380_, 0);
v_isSharedCheck_4436_ = !lean_is_exclusive(v___x_4380_);
if (v_isSharedCheck_4436_ == 0)
{
v___x_4431_ = v___x_4380_;
v_isShared_4432_ = v_isSharedCheck_4436_;
goto v_resetjp_4430_;
}
else
{
lean_inc(v_a_4429_);
lean_dec(v___x_4380_);
v___x_4431_ = lean_box(0);
v_isShared_4432_ = v_isSharedCheck_4436_;
goto v_resetjp_4430_;
}
v_resetjp_4430_:
{
lean_object* v___x_4434_; 
if (v_isShared_4432_ == 0)
{
v___x_4434_ = v___x_4431_;
goto v_reusejp_4433_;
}
else
{
lean_object* v_reuseFailAlloc_4435_; 
v_reuseFailAlloc_4435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4435_, 0, v_a_4429_);
v___x_4434_ = v_reuseFailAlloc_4435_;
goto v_reusejp_4433_;
}
v_reusejp_4433_:
{
return v___x_4434_;
}
}
}
v___jp_4373_:
{
lean_object* v___x_4376_; lean_object* v___x_4377_; 
v___x_4376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4376_, 0, v_b_4374_);
lean_ctor_set(v___x_4376_, 1, v___y_4375_);
v___x_4377_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4377_, 0, v___x_4376_);
return v___x_4377_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1___boxed(lean_object* v_negGoal_4437_, lean_object* v_t_4438_, lean_object* v_init_4439_, lean_object* v___y_4440_, lean_object* v___y_4441_, lean_object* v___y_4442_, lean_object* v___y_4443_, lean_object* v___y_4444_, lean_object* v___y_4445_, lean_object* v___y_4446_, lean_object* v___y_4447_){
_start:
{
lean_object* v_res_4448_; 
v_res_4448_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1(v_negGoal_4437_, v_t_4438_, v_init_4439_, v___y_4440_, v___y_4441_, v___y_4442_, v___y_4443_, v___y_4444_, v___y_4445_, v___y_4446_);
lean_dec(v___y_4446_);
lean_dec_ref(v___y_4445_);
lean_dec(v___y_4444_);
lean_dec_ref(v___y_4443_);
lean_dec(v___y_4442_);
lean_dec_ref(v___y_4441_);
lean_dec_ref(v_t_4438_);
lean_dec_ref(v_negGoal_4437_);
return v_res_4448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFactsImp(uint8_t v_only_x3f_4449_, lean_object* v_hyps_4450_, lean_object* v_negGoal_4451_, lean_object* v_a_4452_, lean_object* v_a_4453_, lean_object* v_a_4454_, lean_object* v_a_4455_, lean_object* v_a_4456_, lean_object* v_a_4457_, lean_object* v_a_4458_){
_start:
{
lean_object* v_lctx_4460_; lean_object* v___x_4461_; size_t v_sz_4462_; size_t v___x_4463_; lean_object* v___x_4464_; 
v_lctx_4460_ = lean_ctor_get(v_a_4455_, 2);
v___x_4461_ = lean_box(0);
v_sz_4462_ = lean_array_size(v_hyps_4450_);
v___x_4463_ = ((size_t)0ULL);
v___x_4464_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_collectFactsImp_spec__0(v_hyps_4450_, v_sz_4462_, v___x_4463_, v___x_4461_, v_a_4452_, v_a_4453_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_, v_a_4458_);
if (lean_obj_tag(v___x_4464_) == 0)
{
lean_object* v_a_4465_; lean_object* v_snd_4466_; lean_object* v___x_4467_; 
v_a_4465_ = lean_ctor_get(v___x_4464_, 0);
lean_inc(v_a_4465_);
lean_dec_ref_known(v___x_4464_, 1);
v_snd_4466_ = lean_ctor_get(v_a_4465_, 1);
lean_inc(v_snd_4466_);
lean_dec(v_a_4465_);
lean_inc_ref(v_negGoal_4451_);
v___x_4467_ = lp_mathlib___private_Mathlib_Tactic_Order_CollectFacts_0__Mathlib_Tactic_Order_collectFactsImp_processExpr(v_negGoal_4451_, v_snd_4466_, v_a_4453_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_, v_a_4458_);
if (lean_obj_tag(v___x_4467_) == 0)
{
lean_object* v_a_4468_; lean_object* v___x_4470_; uint8_t v_isShared_4471_; uint8_t v_isSharedCheck_4504_; 
v_a_4468_ = lean_ctor_get(v___x_4467_, 0);
v_isSharedCheck_4504_ = !lean_is_exclusive(v___x_4467_);
if (v_isSharedCheck_4504_ == 0)
{
v___x_4470_ = v___x_4467_;
v_isShared_4471_ = v_isSharedCheck_4504_;
goto v_resetjp_4469_;
}
else
{
lean_inc(v_a_4468_);
lean_dec(v___x_4467_);
v___x_4470_ = lean_box(0);
v_isShared_4471_ = v_isSharedCheck_4504_;
goto v_resetjp_4469_;
}
v_resetjp_4469_:
{
if (v_only_x3f_4449_ == 0)
{
lean_object* v_snd_4472_; lean_object* v_decls_4473_; lean_object* v___x_4474_; 
lean_del_object(v___x_4470_);
v_snd_4472_ = lean_ctor_get(v_a_4468_, 1);
lean_inc(v_snd_4472_);
lean_dec(v_a_4468_);
v_decls_4473_ = lean_ctor_get(v_lctx_4460_, 1);
v___x_4474_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Order_collectFactsImp_spec__1(v_negGoal_4451_, v_decls_4473_, v___x_4461_, v_snd_4472_, v_a_4453_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_, v_a_4458_);
lean_dec_ref(v_negGoal_4451_);
if (lean_obj_tag(v___x_4474_) == 0)
{
lean_object* v_a_4475_; lean_object* v___x_4477_; uint8_t v_isShared_4478_; uint8_t v_isSharedCheck_4491_; 
v_a_4475_ = lean_ctor_get(v___x_4474_, 0);
v_isSharedCheck_4491_ = !lean_is_exclusive(v___x_4474_);
if (v_isSharedCheck_4491_ == 0)
{
v___x_4477_ = v___x_4474_;
v_isShared_4478_ = v_isSharedCheck_4491_;
goto v_resetjp_4476_;
}
else
{
lean_inc(v_a_4475_);
lean_dec(v___x_4474_);
v___x_4477_ = lean_box(0);
v_isShared_4478_ = v_isSharedCheck_4491_;
goto v_resetjp_4476_;
}
v_resetjp_4476_:
{
lean_object* v_snd_4479_; lean_object* v___x_4481_; uint8_t v_isShared_4482_; uint8_t v_isSharedCheck_4489_; 
v_snd_4479_ = lean_ctor_get(v_a_4475_, 1);
v_isSharedCheck_4489_ = !lean_is_exclusive(v_a_4475_);
if (v_isSharedCheck_4489_ == 0)
{
lean_object* v_unused_4490_; 
v_unused_4490_ = lean_ctor_get(v_a_4475_, 0);
lean_dec(v_unused_4490_);
v___x_4481_ = v_a_4475_;
v_isShared_4482_ = v_isSharedCheck_4489_;
goto v_resetjp_4480_;
}
else
{
lean_inc(v_snd_4479_);
lean_dec(v_a_4475_);
v___x_4481_ = lean_box(0);
v_isShared_4482_ = v_isSharedCheck_4489_;
goto v_resetjp_4480_;
}
v_resetjp_4480_:
{
lean_object* v___x_4484_; 
if (v_isShared_4482_ == 0)
{
lean_ctor_set(v___x_4481_, 0, v___x_4461_);
v___x_4484_ = v___x_4481_;
goto v_reusejp_4483_;
}
else
{
lean_object* v_reuseFailAlloc_4488_; 
v_reuseFailAlloc_4488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4488_, 0, v___x_4461_);
lean_ctor_set(v_reuseFailAlloc_4488_, 1, v_snd_4479_);
v___x_4484_ = v_reuseFailAlloc_4488_;
goto v_reusejp_4483_;
}
v_reusejp_4483_:
{
lean_object* v___x_4486_; 
if (v_isShared_4478_ == 0)
{
lean_ctor_set(v___x_4477_, 0, v___x_4484_);
v___x_4486_ = v___x_4477_;
goto v_reusejp_4485_;
}
else
{
lean_object* v_reuseFailAlloc_4487_; 
v_reuseFailAlloc_4487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4487_, 0, v___x_4484_);
v___x_4486_ = v_reuseFailAlloc_4487_;
goto v_reusejp_4485_;
}
v_reusejp_4485_:
{
return v___x_4486_;
}
}
}
}
}
else
{
return v___x_4474_;
}
}
else
{
lean_object* v_snd_4492_; lean_object* v___x_4494_; uint8_t v_isShared_4495_; uint8_t v_isSharedCheck_4502_; 
lean_dec_ref(v_negGoal_4451_);
v_snd_4492_ = lean_ctor_get(v_a_4468_, 1);
v_isSharedCheck_4502_ = !lean_is_exclusive(v_a_4468_);
if (v_isSharedCheck_4502_ == 0)
{
lean_object* v_unused_4503_; 
v_unused_4503_ = lean_ctor_get(v_a_4468_, 0);
lean_dec(v_unused_4503_);
v___x_4494_ = v_a_4468_;
v_isShared_4495_ = v_isSharedCheck_4502_;
goto v_resetjp_4493_;
}
else
{
lean_inc(v_snd_4492_);
lean_dec(v_a_4468_);
v___x_4494_ = lean_box(0);
v_isShared_4495_ = v_isSharedCheck_4502_;
goto v_resetjp_4493_;
}
v_resetjp_4493_:
{
lean_object* v___x_4497_; 
if (v_isShared_4495_ == 0)
{
lean_ctor_set(v___x_4494_, 0, v___x_4461_);
v___x_4497_ = v___x_4494_;
goto v_reusejp_4496_;
}
else
{
lean_object* v_reuseFailAlloc_4501_; 
v_reuseFailAlloc_4501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4501_, 0, v___x_4461_);
lean_ctor_set(v_reuseFailAlloc_4501_, 1, v_snd_4492_);
v___x_4497_ = v_reuseFailAlloc_4501_;
goto v_reusejp_4496_;
}
v_reusejp_4496_:
{
lean_object* v___x_4499_; 
if (v_isShared_4471_ == 0)
{
lean_ctor_set(v___x_4470_, 0, v___x_4497_);
v___x_4499_ = v___x_4470_;
goto v_reusejp_4498_;
}
else
{
lean_object* v_reuseFailAlloc_4500_; 
v_reuseFailAlloc_4500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4500_, 0, v___x_4497_);
v___x_4499_ = v_reuseFailAlloc_4500_;
goto v_reusejp_4498_;
}
v_reusejp_4498_:
{
return v___x_4499_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_negGoal_4451_);
return v___x_4467_;
}
}
else
{
lean_dec_ref(v_negGoal_4451_);
return v___x_4464_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFactsImp___boxed(lean_object* v_only_x3f_4505_, lean_object* v_hyps_4506_, lean_object* v_negGoal_4507_, lean_object* v_a_4508_, lean_object* v_a_4509_, lean_object* v_a_4510_, lean_object* v_a_4511_, lean_object* v_a_4512_, lean_object* v_a_4513_, lean_object* v_a_4514_, lean_object* v_a_4515_){
_start:
{
uint8_t v_only_x3f_boxed_4516_; lean_object* v_res_4517_; 
v_only_x3f_boxed_4516_ = lean_unbox(v_only_x3f_4505_);
v_res_4517_ = lp_mathlib_Mathlib_Tactic_Order_collectFactsImp(v_only_x3f_boxed_4516_, v_hyps_4506_, v_negGoal_4507_, v_a_4508_, v_a_4509_, v_a_4510_, v_a_4511_, v_a_4512_, v_a_4513_, v_a_4514_);
lean_dec(v_a_4514_);
lean_dec_ref(v_a_4513_);
lean_dec(v_a_4512_);
lean_dec_ref(v_a_4511_);
lean_dec(v_a_4510_);
lean_dec_ref(v_a_4509_);
lean_dec_ref(v_hyps_4506_);
return v_res_4517_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__0(void){
_start:
{
lean_object* v___x_4518_; lean_object* v___x_4519_; lean_object* v___x_4520_; 
v___x_4518_ = lean_box(0);
v___x_4519_ = lean_unsigned_to_nat(16u);
v___x_4520_ = lean_mk_array(v___x_4519_, v___x_4518_);
return v___x_4520_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__1(void){
_start:
{
lean_object* v___x_4521_; lean_object* v___x_4522_; lean_object* v___x_4523_; 
v___x_4521_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__0, &lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__0);
v___x_4522_ = lean_unsigned_to_nat(0u);
v___x_4523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4523_, 0, v___x_4522_);
lean_ctor_set(v___x_4523_, 1, v___x_4521_);
return v___x_4523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFacts(uint8_t v_only_x3f_4524_, lean_object* v_hyps_4525_, lean_object* v_negGoal_4526_, lean_object* v_a_4527_, lean_object* v_a_4528_, lean_object* v_a_4529_, lean_object* v_a_4530_, lean_object* v_a_4531_, lean_object* v_a_4532_){
_start:
{
lean_object* v___x_4534_; lean_object* v___x_4535_; 
v___x_4534_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__1, &lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Order_collectFacts___closed__1);
v___x_4535_ = lp_mathlib_Mathlib_Tactic_Order_collectFactsImp(v_only_x3f_4524_, v_hyps_4525_, v_negGoal_4526_, v___x_4534_, v_a_4527_, v_a_4528_, v_a_4529_, v_a_4530_, v_a_4531_, v_a_4532_);
if (lean_obj_tag(v___x_4535_) == 0)
{
lean_object* v_a_4536_; lean_object* v___x_4538_; uint8_t v_isShared_4539_; uint8_t v_isSharedCheck_4544_; 
v_a_4536_ = lean_ctor_get(v___x_4535_, 0);
v_isSharedCheck_4544_ = !lean_is_exclusive(v___x_4535_);
if (v_isSharedCheck_4544_ == 0)
{
v___x_4538_ = v___x_4535_;
v_isShared_4539_ = v_isSharedCheck_4544_;
goto v_resetjp_4537_;
}
else
{
lean_inc(v_a_4536_);
lean_dec(v___x_4535_);
v___x_4538_ = lean_box(0);
v_isShared_4539_ = v_isSharedCheck_4544_;
goto v_resetjp_4537_;
}
v_resetjp_4537_:
{
lean_object* v_snd_4540_; lean_object* v___x_4542_; 
v_snd_4540_ = lean_ctor_get(v_a_4536_, 1);
lean_inc(v_snd_4540_);
lean_dec(v_a_4536_);
if (v_isShared_4539_ == 0)
{
lean_ctor_set(v___x_4538_, 0, v_snd_4540_);
v___x_4542_ = v___x_4538_;
goto v_reusejp_4541_;
}
else
{
lean_object* v_reuseFailAlloc_4543_; 
v_reuseFailAlloc_4543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4543_, 0, v_snd_4540_);
v___x_4542_ = v_reuseFailAlloc_4543_;
goto v_reusejp_4541_;
}
v_reusejp_4541_:
{
return v___x_4542_;
}
}
}
else
{
lean_object* v_a_4545_; lean_object* v___x_4547_; uint8_t v_isShared_4548_; uint8_t v_isSharedCheck_4552_; 
v_a_4545_ = lean_ctor_get(v___x_4535_, 0);
v_isSharedCheck_4552_ = !lean_is_exclusive(v___x_4535_);
if (v_isSharedCheck_4552_ == 0)
{
v___x_4547_ = v___x_4535_;
v_isShared_4548_ = v_isSharedCheck_4552_;
goto v_resetjp_4546_;
}
else
{
lean_inc(v_a_4545_);
lean_dec(v___x_4535_);
v___x_4547_ = lean_box(0);
v_isShared_4548_ = v_isSharedCheck_4552_;
goto v_resetjp_4546_;
}
v_resetjp_4546_:
{
lean_object* v___x_4550_; 
if (v_isShared_4548_ == 0)
{
v___x_4550_ = v___x_4547_;
goto v_reusejp_4549_;
}
else
{
lean_object* v_reuseFailAlloc_4551_; 
v_reuseFailAlloc_4551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4551_, 0, v_a_4545_);
v___x_4550_ = v_reuseFailAlloc_4551_;
goto v_reusejp_4549_;
}
v_reusejp_4549_:
{
return v___x_4550_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFacts___boxed(lean_object* v_only_x3f_4553_, lean_object* v_hyps_4554_, lean_object* v_negGoal_4555_, lean_object* v_a_4556_, lean_object* v_a_4557_, lean_object* v_a_4558_, lean_object* v_a_4559_, lean_object* v_a_4560_, lean_object* v_a_4561_, lean_object* v_a_4562_){
_start:
{
uint8_t v_only_x3f_boxed_4563_; lean_object* v_res_4564_; 
v_only_x3f_boxed_4563_ = lean_unbox(v_only_x3f_4553_);
v_res_4564_ = lp_mathlib_Mathlib_Tactic_Order_collectFacts(v_only_x3f_boxed_4563_, v_hyps_4554_, v_negGoal_4555_, v_a_4556_, v_a_4557_, v_a_4558_, v_a_4559_, v_a_4560_, v_a_4561_);
lean_dec(v_a_4561_);
lean_dec_ref(v_a_4560_);
lean_dec(v_a_4559_);
lean_dec_ref(v_a_4558_);
lean_dec(v_a_4557_);
lean_dec_ref(v_a_4556_);
lean_dec_ref(v_hyps_4554_);
return v_res_4564_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default = _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact_default);
lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact = _init_lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Order_instInhabitedAtomicFact);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
}
#ifdef __cplusplus
}
#endif
