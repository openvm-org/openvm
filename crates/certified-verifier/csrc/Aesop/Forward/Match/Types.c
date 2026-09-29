// Lean compiler output
// Module: Aesop.Forward.Match.Types
// Imports: public import Init public meta import Init public import Aesop.Rule.Forward
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
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_Substitution_find_x3f(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lp_aesop_Aesop_instBEqSlotIndex_beq(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instOrdSlotIndex_ord(lean_object*, lean_object*);
uint8_t lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_compareArraySizeThenLex___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedSubstitution_default;
uint8_t lp_aesop_Aesop_ForwardRulePriority_compare(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_RuleName_compare(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofLevel(lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lp_aesop_Aesop_instHashableSlotIndex_hash(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedForwardRule_default;
lean_object* l_Array_filterMapM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_bracket(lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
static const lean_array_object lp_aesop_Aesop_instInhabitedMatch_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedMatch_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedMatch_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedMatch_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedMatch_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMatch_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedMatch;
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_Match_equiv_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_Match_equiv_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_equiv_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_equiv_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_equiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_equiv___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Match_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_equiv___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Match_instBEq = (const lean_object*)&lp_aesop_Aesop_Match_instBEq___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_Match_instHashable___lam__0(lean_object*, uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instHashable___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint64_t lp_aesop_Aesop_Match_instHashable___lam__1(lean_object*, uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instHashable___lam__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__1 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__2 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__3 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__4 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__5 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__6 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__0_value),((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__7 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__7_value),((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__2_value),((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__3_value),((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__4_value),((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__8 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Match_instHashable___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__8_value),((lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___closed__9 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___lam__3___closed__9_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_Match_instHashable___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Match_instHashable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_instHashable___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instHashable___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Match_instHashable = (const lean_object*)&lp_aesop_Aesop_Match_instHashable___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_instOrd___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instOrd___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_instOrd___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instOrd___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Match_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_instOrd___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Match_instOrd = (const lean_object*)&lp_aesop_Aesop_Match_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__0___boxed(lean_object*);
static const lean_string_object lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ↦ "};
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__1 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__2 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__3;
static const lean_string_object lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " | "};
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__4 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__5 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__6;
static const lean_string_object lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__7 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__7_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Match_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_instToMessageData___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Match_instToMessageData___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_instToMessageData___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___closed__1 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Match_instToMessageData___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_instToMessageData___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___closed__2 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Match_instToMessageData___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_instToMessageData___lam__3, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___closed__3 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Match_instToMessageData___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Match_instToMessageData___lam__4, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__0_value),((lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__1_value),((lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__2_value),((lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__3_value)} };
static const lean_object* lp_aesop_Aesop_Match_instToMessageData___closed__4 = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__4_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Match_instToMessageData = (const lean_object*)&lp_aesop_Aesop_Match_instToMessageData___closed__4_value;
static const lean_array_object lp_aesop_Aesop_instInhabitedCompleteMatch_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedCompleteMatch_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedCompleteMatch_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedCompleteMatch_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedCompleteMatch_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedCompleteMatch = (const lean_object*)&lp_aesop_Aesop_instInhabitedCompleteMatch_default___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqCompleteMatch_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqCompleteMatch_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqCompleteMatch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqCompleteMatch_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqCompleteMatch___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqCompleteMatch___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqCompleteMatch = (const lean_object*)&lp_aesop_Aesop_instBEqCompleteMatch___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, uint64_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint64_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0(lean_object*, lean_object*, size_t, size_t, uint64_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint64_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__1(lean_object*, size_t, size_t, uint64_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0;
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableCompleteMatch_hash(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableCompleteMatch_hash___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instHashableCompleteMatch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instHashableCompleteMatch_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instHashableCompleteMatch___closed__0 = (const lean_object*)&lp_aesop_Aesop_instHashableCompleteMatch___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instHashableCompleteMatch = (const lean_object*)&lp_aesop_Aesop_instHashableCompleteMatch___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instEmptyCollectionCompleteMatch = (const lean_object*)&lp_aesop_Aesop_instInhabitedCompleteMatch_default___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdCompleteMatch___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdCompleteMatch___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instOrdCompleteMatch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instOrdCompleteMatch___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instOrd___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_instOrdCompleteMatch___closed__0 = (const lean_object*)&lp_aesop_Aesop_instOrdCompleteMatch___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instOrdCompleteMatch = (const lean_object*)&lp_aesop_Aesop_instOrdCompleteMatch___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardRuleMatch_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardRuleMatch_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRuleMatch_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRuleMatch;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqForwardRuleMatch_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqForwardRuleMatch_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqForwardRuleMatch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqForwardRuleMatch_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqForwardRuleMatch___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqForwardRuleMatch___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqForwardRuleMatch = (const lean_object*)&lp_aesop_Aesop_instBEqForwardRuleMatch___closed__0_value;
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableForwardRuleMatch_hash(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableForwardRuleMatch_hash___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instHashableForwardRuleMatch___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instHashableForwardRuleMatch_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instHashableForwardRuleMatch___closed__0 = (const lean_object*)&lp_aesop_Aesop_instHashableForwardRuleMatch___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instHashableForwardRuleMatch = (const lean_object*)&lp_aesop_Aesop_instHashableForwardRuleMatch___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRuleMatch_ord___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_ord___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRuleMatch_ord___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRuleMatch_ord___lam__2___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_Match_instOrd___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_ord___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_ord___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRuleMatch_ord = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_ord___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRuleMatch_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_le___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedMatch_default___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_3_ = lean_unsigned_to_nat(0u);
v___x_4_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedMatch_default___closed__0));
v___x_5_ = lp_aesop_Aesop_instInhabitedSubstitution_default;
v___x_6_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_4_);
lean_ctor_set(v___x_6_, 2, v___x_3_);
lean_ctor_set(v___x_6_, 3, v___x_4_);
lean_ctor_set(v___x_6_, 4, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedMatch_default(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedMatch_default___closed__1, &lp_aesop_Aesop_instInhabitedMatch_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedMatch_default___closed__1);
return v___x_7_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedMatch(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_aesop_Aesop_instInhabitedMatch_default;
return v___x_8_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_Match_equiv_spec__0(lean_object* v_x_9_, lean_object* v_x_10_){
_start:
{
if (lean_obj_tag(v_x_9_) == 0)
{
if (lean_obj_tag(v_x_10_) == 0)
{
uint8_t v___x_11_; 
v___x_11_ = 1;
return v___x_11_;
}
else
{
uint8_t v___x_12_; 
v___x_12_ = 0;
return v___x_12_;
}
}
else
{
if (lean_obj_tag(v_x_10_) == 0)
{
uint8_t v___x_13_; 
v___x_13_ = 0;
return v___x_13_;
}
else
{
lean_object* v_val_14_; lean_object* v_val_15_; uint8_t v___x_16_; 
v_val_14_ = lean_ctor_get(v_x_9_, 0);
v_val_15_ = lean_ctor_get(v_x_10_, 0);
v___x_16_ = lean_expr_eqv(v_val_14_, v_val_15_);
return v___x_16_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_Match_equiv_spec__0___boxed(lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_aesop_Option_instBEq_beq___at___00Aesop_Match_equiv_spec__0(v_x_17_, v_x_18_);
lean_dec(v_x_18_);
lean_dec(v_x_17_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_equiv_spec__1(lean_object* v_m_u2081_21_, lean_object* v_m_u2082_22_, uint8_t v___y_23_, lean_object* v_as_24_, size_t v_i_25_, size_t v_stop_26_){
_start:
{
uint8_t v___x_27_; 
v___x_27_ = lean_usize_dec_eq(v_i_25_, v_stop_26_);
if (v___x_27_ == 0)
{
lean_object* v_subst_28_; lean_object* v_subst_29_; uint8_t v___x_30_; uint8_t v___y_32_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; uint8_t v___x_39_; 
v_subst_28_ = lean_ctor_get(v_m_u2081_21_, 0);
v_subst_29_ = lean_ctor_get(v_m_u2082_22_, 0);
v___x_30_ = 1;
v___x_36_ = lean_array_uget_borrowed(v_as_24_, v_i_25_);
v___x_37_ = lp_aesop_Aesop_Substitution_find_x3f(v___x_36_, v_subst_28_);
v___x_38_ = lp_aesop_Aesop_Substitution_find_x3f(v___x_36_, v_subst_29_);
v___x_39_ = lp_aesop_Option_instBEq_beq___at___00Aesop_Match_equiv_spec__0(v___x_37_, v___x_38_);
lean_dec(v___x_38_);
lean_dec(v___x_37_);
if (v___x_39_ == 0)
{
v___y_32_ = v___y_23_;
goto v___jp_31_;
}
else
{
v___y_32_ = v___x_27_;
goto v___jp_31_;
}
v___jp_31_:
{
if (v___y_32_ == 0)
{
size_t v___x_33_; size_t v___x_34_; 
v___x_33_ = ((size_t)1ULL);
v___x_34_ = lean_usize_add(v_i_25_, v___x_33_);
v_i_25_ = v___x_34_;
goto _start;
}
else
{
return v___x_30_;
}
}
}
else
{
uint8_t v___x_40_; 
v___x_40_ = 0;
return v___x_40_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_equiv_spec__1___boxed(lean_object* v_m_u2081_41_, lean_object* v_m_u2082_42_, lean_object* v___y_43_, lean_object* v_as_44_, lean_object* v_i_45_, lean_object* v_stop_46_){
_start:
{
uint8_t v___y_438__boxed_47_; size_t v_i_boxed_48_; size_t v_stop_boxed_49_; uint8_t v_res_50_; lean_object* v_r_51_; 
v___y_438__boxed_47_ = lean_unbox(v___y_43_);
v_i_boxed_48_ = lean_unbox_usize(v_i_45_);
lean_dec(v_i_45_);
v_stop_boxed_49_ = lean_unbox_usize(v_stop_46_);
lean_dec(v_stop_46_);
v_res_50_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_equiv_spec__1(v_m_u2081_41_, v_m_u2082_42_, v___y_438__boxed_47_, v_as_44_, v_i_boxed_48_, v_stop_boxed_49_);
lean_dec_ref(v_as_44_);
lean_dec_ref(v_m_u2082_42_);
lean_dec_ref(v_m_u2081_41_);
v_r_51_ = lean_box(v_res_50_);
return v_r_51_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_equiv(lean_object* v_m_u2081_52_, lean_object* v_m_u2082_53_){
_start:
{
lean_object* v_level_54_; lean_object* v_forwardDeps_55_; lean_object* v_conclusionDeps_56_; uint8_t v___y_58_; lean_object* v_level_66_; uint8_t v___x_67_; 
v_level_54_ = lean_ctor_get(v_m_u2081_52_, 2);
v_forwardDeps_55_ = lean_ctor_get(v_m_u2081_52_, 3);
v_conclusionDeps_56_ = lean_ctor_get(v_m_u2081_52_, 4);
v_level_66_ = lean_ctor_get(v_m_u2082_53_, 2);
v___x_67_ = lp_aesop_Aesop_instBEqSlotIndex_beq(v_level_54_, v_level_66_);
if (v___x_67_ == 0)
{
v___y_58_ = v___x_67_;
goto v___jp_57_;
}
else
{
lean_object* v___x_68_; lean_object* v___x_69_; uint8_t v___x_70_; 
v___x_68_ = lean_unsigned_to_nat(0u);
v___x_69_ = lean_array_get_size(v_forwardDeps_55_);
v___x_70_ = lean_nat_dec_lt(v___x_68_, v___x_69_);
if (v___x_70_ == 0)
{
v___y_58_ = v___x_67_;
goto v___jp_57_;
}
else
{
if (v___x_70_ == 0)
{
v___y_58_ = v___x_67_;
goto v___jp_57_;
}
else
{
size_t v___x_71_; size_t v___x_72_; uint8_t v___x_73_; 
v___x_71_ = ((size_t)0ULL);
v___x_72_ = lean_usize_of_nat(v___x_69_);
v___x_73_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_equiv_spec__1(v_m_u2081_52_, v_m_u2082_53_, v___x_67_, v_forwardDeps_55_, v___x_71_, v___x_72_);
if (v___x_73_ == 0)
{
v___y_58_ = v___x_67_;
goto v___jp_57_;
}
else
{
uint8_t v___x_74_; 
v___x_74_ = 0;
return v___x_74_;
}
}
}
}
v___jp_57_:
{
if (v___y_58_ == 0)
{
return v___y_58_;
}
else
{
lean_object* v___x_59_; lean_object* v___x_60_; uint8_t v___x_61_; 
v___x_59_ = lean_unsigned_to_nat(0u);
v___x_60_ = lean_array_get_size(v_conclusionDeps_56_);
v___x_61_ = lean_nat_dec_lt(v___x_59_, v___x_60_);
if (v___x_61_ == 0)
{
return v___y_58_;
}
else
{
if (v___x_61_ == 0)
{
return v___y_58_;
}
else
{
size_t v___x_62_; size_t v___x_63_; uint8_t v___x_64_; 
v___x_62_ = ((size_t)0ULL);
v___x_63_ = lean_usize_of_nat(v___x_60_);
v___x_64_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_equiv_spec__1(v_m_u2081_52_, v_m_u2082_53_, v___y_58_, v_conclusionDeps_56_, v___x_62_, v___x_63_);
if (v___x_64_ == 0)
{
return v___y_58_;
}
else
{
uint8_t v___x_65_; 
v___x_65_ = 0;
return v___x_65_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_equiv___boxed(lean_object* v_m_u2081_75_, lean_object* v_m_u2082_76_){
_start:
{
uint8_t v_res_77_; lean_object* v_r_78_; 
v_res_77_ = lp_aesop_Aesop_Match_equiv(v_m_u2081_75_, v_m_u2082_76_);
lean_dec_ref(v_m_u2082_76_);
lean_dec_ref(v_m_u2081_75_);
v_r_78_ = lean_box(v_res_77_);
return v_r_78_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Match_instHashable___lam__0(lean_object* v_subst_81_, uint64_t v_h_82_, lean_object* v_p_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_aesop_Aesop_Substitution_find_x3f(v_p_83_, v_subst_81_);
if (lean_obj_tag(v___x_84_) == 0)
{
uint64_t v___x_85_; uint64_t v___x_86_; 
v___x_85_ = 11ULL;
v___x_86_ = lean_uint64_mix_hash(v_h_82_, v___x_85_);
return v___x_86_;
}
else
{
lean_object* v_val_87_; uint64_t v___x_88_; uint64_t v___x_89_; uint64_t v___x_90_; uint64_t v___x_91_; 
v_val_87_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_val_87_);
lean_dec_ref_known(v___x_84_, 1);
v___x_88_ = l_Lean_Expr_hash(v_val_87_);
lean_dec(v_val_87_);
v___x_89_ = 13ULL;
v___x_90_ = lean_uint64_mix_hash(v___x_88_, v___x_89_);
v___x_91_ = lean_uint64_mix_hash(v_h_82_, v___x_90_);
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instHashable___lam__0___boxed(lean_object* v_subst_92_, lean_object* v_h_93_, lean_object* v_p_94_){
_start:
{
uint64_t v_h_boxed_95_; uint64_t v_res_96_; lean_object* v_r_97_; 
v_h_boxed_95_ = lean_unbox_uint64(v_h_93_);
lean_dec_ref(v_h_93_);
v_res_96_ = lp_aesop_Aesop_Match_instHashable___lam__0(v_subst_92_, v_h_boxed_95_, v_p_94_);
lean_dec(v_p_94_);
lean_dec_ref(v_subst_92_);
v_r_97_ = lean_box_uint64(v_res_96_);
return v_r_97_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Match_instHashable___lam__1(lean_object* v___f_98_, uint64_t v_x1_99_, lean_object* v_x2_100_){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; uint64_t v___x_103_; 
v___x_101_ = lean_box_uint64(v_x1_99_);
v___x_102_ = lean_apply_2(v___f_98_, v___x_101_, v_x2_100_);
v___x_103_ = lean_unbox_uint64(v___x_102_);
lean_dec_ref(v___x_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instHashable___lam__1___boxed(lean_object* v___f_104_, lean_object* v_x1_105_, lean_object* v_x2_106_){
_start:
{
uint64_t v_x1_189__boxed_107_; uint64_t v_res_108_; lean_object* v_r_109_; 
v_x1_189__boxed_107_ = lean_unbox_uint64(v_x1_105_);
lean_dec_ref(v_x1_105_);
v_res_108_ = lp_aesop_Aesop_Match_instHashable___lam__1(v___f_104_, v_x1_189__boxed_107_, v_x2_106_);
v_r_109_ = lean_box_uint64(v_res_108_);
return v_r_109_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_Match_instHashable___lam__3(lean_object* v_m_129_){
_start:
{
lean_object* v_subst_130_; lean_object* v_level_131_; lean_object* v_forwardDeps_132_; lean_object* v_conclusionDeps_133_; lean_object* v___f_134_; lean_object* v___f_135_; uint64_t v_h_136_; lean_object* v___x_137_; uint64_t v___y_139_; lean_object* v___x_154_; lean_object* v___x_155_; uint8_t v___x_156_; 
v_subst_130_ = lean_ctor_get(v_m_129_, 0);
lean_inc_ref(v_subst_130_);
v_level_131_ = lean_ctor_get(v_m_129_, 2);
lean_inc(v_level_131_);
v_forwardDeps_132_ = lean_ctor_get(v_m_129_, 3);
lean_inc_ref(v_forwardDeps_132_);
v_conclusionDeps_133_ = lean_ctor_get(v_m_129_, 4);
lean_inc_ref(v_conclusionDeps_133_);
lean_dec_ref(v_m_129_);
v___f_134_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Match_instHashable___lam__0___boxed), 3, 1);
lean_closure_set(v___f_134_, 0, v_subst_130_);
v___f_135_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Match_instHashable___lam__1___boxed), 3, 1);
lean_closure_set(v___f_135_, 0, v___f_134_);
v_h_136_ = lp_aesop_Aesop_instHashableSlotIndex_hash(v_level_131_);
lean_dec(v_level_131_);
v___x_137_ = lean_unsigned_to_nat(0u);
v___x_154_ = lean_array_get_size(v_forwardDeps_132_);
v___x_155_ = ((lean_object*)(lp_aesop_Aesop_Match_instHashable___lam__3___closed__9));
v___x_156_ = lean_nat_dec_lt(v___x_137_, v___x_154_);
if (v___x_156_ == 0)
{
lean_dec_ref(v_forwardDeps_132_);
v___y_139_ = v_h_136_;
goto v___jp_138_;
}
else
{
uint8_t v___x_157_; 
v___x_157_ = lean_nat_dec_le(v___x_154_, v___x_154_);
if (v___x_157_ == 0)
{
if (v___x_156_ == 0)
{
lean_dec_ref(v_forwardDeps_132_);
v___y_139_ = v_h_136_;
goto v___jp_138_;
}
else
{
size_t v___x_158_; size_t v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; uint64_t v___x_162_; 
v___x_158_ = ((size_t)0ULL);
v___x_159_ = lean_usize_of_nat(v___x_154_);
v___x_160_ = lean_box_uint64(v_h_136_);
lean_inc_ref(v___f_135_);
v___x_161_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_155_, v___f_135_, v_forwardDeps_132_, v___x_158_, v___x_159_, v___x_160_);
v___x_162_ = lean_unbox_uint64(v___x_161_);
lean_dec(v___x_161_);
v___y_139_ = v___x_162_;
goto v___jp_138_;
}
}
else
{
size_t v___x_163_; size_t v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint64_t v___x_167_; 
v___x_163_ = ((size_t)0ULL);
v___x_164_ = lean_usize_of_nat(v___x_154_);
v___x_165_ = lean_box_uint64(v_h_136_);
lean_inc_ref(v___f_135_);
v___x_166_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_155_, v___f_135_, v_forwardDeps_132_, v___x_163_, v___x_164_, v___x_165_);
v___x_167_ = lean_unbox_uint64(v___x_166_);
lean_dec(v___x_166_);
v___y_139_ = v___x_167_;
goto v___jp_138_;
}
}
v___jp_138_:
{
lean_object* v___x_140_; lean_object* v___x_141_; uint8_t v___x_142_; 
v___x_140_ = lean_array_get_size(v_conclusionDeps_133_);
v___x_141_ = ((lean_object*)(lp_aesop_Aesop_Match_instHashable___lam__3___closed__9));
v___x_142_ = lean_nat_dec_lt(v___x_137_, v___x_140_);
if (v___x_142_ == 0)
{
lean_dec_ref(v___f_135_);
lean_dec_ref(v_conclusionDeps_133_);
return v___y_139_;
}
else
{
uint8_t v___x_143_; 
v___x_143_ = lean_nat_dec_le(v___x_140_, v___x_140_);
if (v___x_143_ == 0)
{
if (v___x_142_ == 0)
{
lean_dec_ref(v___f_135_);
lean_dec_ref(v_conclusionDeps_133_);
return v___y_139_;
}
else
{
size_t v___x_144_; size_t v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; uint64_t v___x_148_; 
v___x_144_ = ((size_t)0ULL);
v___x_145_ = lean_usize_of_nat(v___x_140_);
v___x_146_ = lean_box_uint64(v___y_139_);
v___x_147_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_141_, v___f_135_, v_conclusionDeps_133_, v___x_144_, v___x_145_, v___x_146_);
v___x_148_ = lean_unbox_uint64(v___x_147_);
lean_dec(v___x_147_);
return v___x_148_;
}
}
else
{
size_t v___x_149_; size_t v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; uint64_t v___x_153_; 
v___x_149_ = ((size_t)0ULL);
v___x_150_ = lean_usize_of_nat(v___x_140_);
v___x_151_ = lean_box_uint64(v___y_139_);
v___x_152_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_141_, v___f_135_, v_conclusionDeps_133_, v___x_149_, v___x_150_, v___x_151_);
v___x_153_ = lean_unbox_uint64(v___x_152_);
lean_dec(v___x_152_);
return v___x_153_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instHashable___lam__3___boxed(lean_object* v_m_168_){
_start:
{
uint64_t v_res_169_; lean_object* v_r_170_; 
v_res_169_ = lp_aesop_Aesop_Match_instHashable___lam__3(v_m_168_);
v_r_170_ = lean_box_uint64(v_res_169_);
return v_r_170_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_instOrd___lam__0(uint8_t v___x_173_, lean_object* v_x_174_, lean_object* v_x_175_){
_start:
{
if (lean_obj_tag(v_x_174_) == 0)
{
if (lean_obj_tag(v_x_175_) == 0)
{
return v___x_173_;
}
else
{
uint8_t v___x_176_; 
v___x_176_ = 0;
return v___x_176_;
}
}
else
{
if (lean_obj_tag(v_x_175_) == 0)
{
uint8_t v___x_177_; 
v___x_177_ = 2;
return v___x_177_;
}
else
{
lean_object* v_val_178_; lean_object* v_val_179_; uint8_t v___x_180_; 
v_val_178_ = lean_ctor_get(v_x_174_, 0);
v_val_179_ = lean_ctor_get(v_x_175_, 0);
v___x_180_ = lp_aesop___private_Aesop_Forward_Substitution_0__Aesop_Substitution_cmpExprs(v_val_178_, v_val_179_);
return v___x_180_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instOrd___lam__0___boxed(lean_object* v___x_181_, lean_object* v_x_182_, lean_object* v_x_183_){
_start:
{
uint8_t v___x_135__boxed_184_; uint8_t v_res_185_; lean_object* v_r_186_; 
v___x_135__boxed_184_ = lean_unbox(v___x_181_);
v_res_185_ = lp_aesop_Aesop_Match_instOrd___lam__0(v___x_135__boxed_184_, v_x_182_, v_x_183_);
lean_dec(v_x_183_);
lean_dec(v_x_182_);
v_r_186_ = lean_box(v_res_185_);
return v_r_186_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_instOrd___lam__1(lean_object* v_m_u2081_187_, lean_object* v_m_u2082_188_){
_start:
{
lean_object* v_subst_189_; lean_object* v_level_190_; lean_object* v_subst_191_; lean_object* v_level_192_; uint8_t v___x_193_; 
v_subst_189_ = lean_ctor_get(v_m_u2081_187_, 0);
v_level_190_ = lean_ctor_get(v_m_u2081_187_, 2);
v_subst_191_ = lean_ctor_get(v_m_u2082_188_, 0);
v_level_192_ = lean_ctor_get(v_m_u2082_188_, 2);
v___x_193_ = lp_aesop_Aesop_instOrdSlotIndex_ord(v_level_190_, v_level_192_);
if (v___x_193_ == 1)
{
uint8_t v___x_194_; 
v___x_194_ = lp_aesop_Aesop_Match_equiv(v_m_u2081_187_, v_m_u2082_188_);
if (v___x_194_ == 0)
{
lean_object* v_premises_195_; lean_object* v_premises_196_; lean_object* v___x_197_; lean_object* v___f_198_; uint8_t v___x_199_; 
v_premises_195_ = lean_ctor_get(v_subst_189_, 0);
v_premises_196_ = lean_ctor_get(v_subst_191_, 0);
v___x_197_ = lean_box(v___x_193_);
v___f_198_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Match_instOrd___lam__0___boxed), 3, 1);
lean_closure_set(v___f_198_, 0, v___x_197_);
v___x_199_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v___f_198_, v_premises_195_, v_premises_196_);
return v___x_199_;
}
else
{
return v___x_193_;
}
}
else
{
return v___x_193_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instOrd___lam__1___boxed(lean_object* v_m_u2081_200_, lean_object* v_m_u2082_201_){
_start:
{
uint8_t v_res_202_; lean_object* v_r_203_; 
v_res_202_ = lp_aesop_Aesop_Match_instOrd___lam__1(v_m_u2081_200_, v_m_u2082_201_);
lean_dec_ref(v_m_u2082_201_);
lean_dec_ref(v_m_u2081_200_);
v_r_203_ = lean_box(v_res_202_);
return v_r_203_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__0(lean_object* v_x_206_){
_start:
{
lean_inc(v_x_206_);
return v_x_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__0___boxed(lean_object* v_x_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_aesop_Aesop_Match_instToMessageData___lam__0(v_x_207_);
lean_dec(v_x_207_);
return v_res_208_;
}
}
static lean_object* _init_lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_210_ = ((lean_object*)(lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__0));
v___x_211_ = l_Lean_stringToMessageData(v___x_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__1(lean_object* v_i_212_, lean_object* v_a_213_, lean_object* v_x_214_){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_215_ = l_Nat_reprFast(v_i_212_);
v___x_216_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_216_, 0, v___x_215_);
v___x_217_ = l_Lean_MessageData_ofFormat(v___x_216_);
v___x_218_ = lean_obj_once(&lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1, &lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1_once, _init_lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1);
v___x_219_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_219_, 0, v___x_217_);
lean_ctor_set(v___x_219_, 1, v___x_218_);
v___x_220_ = l_Lean_MessageData_ofExpr(v_a_213_);
v___x_221_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_221_, 0, v___x_219_);
lean_ctor_set(v___x_221_, 1, v___x_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__2(lean_object* v_x_222_){
_start:
{
lean_inc(v_x_222_);
return v_x_222_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__2___boxed(lean_object* v_x_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_aesop_Aesop_Match_instToMessageData___lam__2(v_x_223_);
lean_dec(v_x_223_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__3(lean_object* v_i_225_, lean_object* v_a_226_, lean_object* v_x_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_228_ = l_Nat_reprFast(v_i_225_);
v___x_229_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
v___x_230_ = l_Lean_MessageData_ofFormat(v___x_229_);
v___x_231_ = lean_obj_once(&lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1, &lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1_once, _init_lp_aesop_Aesop_Match_instToMessageData___lam__1___closed__1);
v___x_232_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_230_);
lean_ctor_set(v___x_232_, 1, v___x_231_);
v___x_233_ = l_Lean_MessageData_ofLevel(v_a_226_);
v___x_234_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_232_);
lean_ctor_set(v___x_234_, 1, v___x_233_);
return v___x_234_;
}
}
static lean_object* _init_lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__3(void){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_239_ = ((lean_object*)(lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__2));
v___x_240_ = l_Lean_MessageData_ofFormat(v___x_239_);
return v___x_240_;
}
}
static lean_object* _init_lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__6(void){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_244_ = ((lean_object*)(lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__5));
v___x_245_ = l_Lean_MessageData_ofFormat(v___x_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_instToMessageData___lam__4(lean_object* v___f_247_, lean_object* v___f_248_, lean_object* v___f_249_, lean_object* v___f_250_, lean_object* v_m_251_){
_start:
{
lean_object* v_subst_252_; lean_object* v_premises_253_; lean_object* v_levels_254_; lean_object* v___x_256_; uint8_t v_isShared_257_; uint8_t v_isSharedCheck_282_; 
v_subst_252_ = lean_ctor_get(v_m_251_, 0);
lean_inc_ref(v_subst_252_);
lean_dec_ref(v_m_251_);
v_premises_253_ = lean_ctor_get(v_subst_252_, 0);
v_levels_254_ = lean_ctor_get(v_subst_252_, 1);
v_isSharedCheck_282_ = !lean_is_exclusive(v_subst_252_);
if (v_isSharedCheck_282_ == 0)
{
v___x_256_ = v_subst_252_;
v_isShared_257_ = v_isSharedCheck_282_;
goto v_resetjp_255_;
}
else
{
lean_inc(v_levels_254_);
lean_inc(v_premises_253_);
lean_dec(v_subst_252_);
v___x_256_ = lean_box(0);
v_isShared_257_ = v_isSharedCheck_282_;
goto v_resetjp_255_;
}
v_resetjp_255_:
{
lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; size_t v_sz_262_; size_t v___x_263_; lean_object* v___x_264_; lean_object* v_ps_265_; lean_object* v___x_266_; lean_object* v___x_267_; size_t v_sz_268_; lean_object* v___x_269_; lean_object* v_ls_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_276_; 
v___x_258_ = lean_unsigned_to_nat(0u);
v___x_259_ = lean_array_get_size(v_premises_253_);
v___x_260_ = ((lean_object*)(lp_aesop_Aesop_Match_instHashable___lam__3___closed__9));
v___x_261_ = l_Array_filterMapM___redArg(v___x_260_, v___f_247_, v_premises_253_, v___x_258_, v___x_259_);
v_sz_262_ = lean_array_size(v___x_261_);
v___x_263_ = ((size_t)0ULL);
lean_inc(v___x_261_);
v___x_264_ = l___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_260_, v___x_261_, v___f_248_, v_sz_262_, v___x_263_, v___x_261_);
lean_dec(v___x_261_);
v_ps_265_ = lean_array_to_list(v___x_264_);
v___x_266_ = lean_array_get_size(v_levels_254_);
v___x_267_ = l_Array_filterMapM___redArg(v___x_260_, v___f_249_, v_levels_254_, v___x_258_, v___x_266_);
v_sz_268_ = lean_array_size(v___x_267_);
lean_inc(v___x_267_);
v___x_269_ = l___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_260_, v___x_267_, v___f_250_, v_sz_268_, v___x_263_, v___x_267_);
lean_dec(v___x_267_);
v_ls_270_ = lean_array_to_list(v___x_269_);
v___x_271_ = ((lean_object*)(lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__0));
v___x_272_ = lean_obj_once(&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__3, &lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__3_once, _init_lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__3);
v___x_273_ = l_Lean_MessageData_joinSep(v_ps_265_, v___x_272_);
v___x_274_ = lean_obj_once(&lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__6, &lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__6_once, _init_lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__6);
if (v_isShared_257_ == 0)
{
lean_ctor_set_tag(v___x_256_, 7);
lean_ctor_set(v___x_256_, 1, v___x_274_);
lean_ctor_set(v___x_256_, 0, v___x_273_);
v___x_276_ = v___x_256_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v___x_273_);
lean_ctor_set(v_reuseFailAlloc_281_, 1, v___x_274_);
v___x_276_ = v_reuseFailAlloc_281_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_277_ = l_Lean_MessageData_joinSep(v_ls_270_, v___x_272_);
v___x_278_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_278_, 0, v___x_276_);
lean_ctor_set(v___x_278_, 1, v___x_277_);
v___x_279_ = ((lean_object*)(lp_aesop_Aesop_Match_instToMessageData___lam__4___closed__7));
v___x_280_ = l_Lean_MessageData_bracket(v___x_271_, v___x_278_, v___x_279_);
return v___x_280_;
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___redArg(lean_object* v_xs_297_, lean_object* v_ys_298_, lean_object* v_x_299_){
_start:
{
lean_object* v_zero_300_; uint8_t v_isZero_301_; 
v_zero_300_ = lean_unsigned_to_nat(0u);
v_isZero_301_ = lean_nat_dec_eq(v_x_299_, v_zero_300_);
if (v_isZero_301_ == 1)
{
lean_dec(v_x_299_);
return v_isZero_301_;
}
else
{
lean_object* v_one_302_; lean_object* v_n_303_; lean_object* v___x_304_; lean_object* v___x_305_; uint8_t v___x_306_; 
v_one_302_ = lean_unsigned_to_nat(1u);
v_n_303_ = lean_nat_sub(v_x_299_, v_one_302_);
lean_dec(v_x_299_);
v___x_304_ = lean_array_fget_borrowed(v_xs_297_, v_n_303_);
v___x_305_ = lean_array_fget_borrowed(v_ys_298_, v_n_303_);
v___x_306_ = lp_aesop_Aesop_Match_equiv(v___x_304_, v___x_305_);
if (v___x_306_ == 0)
{
lean_dec(v_n_303_);
return v___x_306_;
}
else
{
v_x_299_ = v_n_303_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___redArg___boxed(lean_object* v_xs_308_, lean_object* v_ys_309_, lean_object* v_x_310_){
_start:
{
uint8_t v_res_311_; lean_object* v_r_312_; 
v_res_311_ = lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___redArg(v_xs_308_, v_ys_309_, v_x_310_);
lean_dec_ref(v_ys_309_);
lean_dec_ref(v_xs_308_);
v_r_312_ = lean_box(v_res_311_);
return v_r_312_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqCompleteMatch_beq(lean_object* v_x_313_, lean_object* v_x_314_){
_start:
{
lean_object* v___x_315_; lean_object* v___x_316_; uint8_t v___x_317_; 
v___x_315_ = lean_array_get_size(v_x_313_);
v___x_316_ = lean_array_get_size(v_x_314_);
v___x_317_ = lean_nat_dec_eq(v___x_315_, v___x_316_);
if (v___x_317_ == 0)
{
return v___x_317_;
}
else
{
uint8_t v___x_318_; 
v___x_318_ = lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___redArg(v_x_313_, v_x_314_, v___x_315_);
return v___x_318_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqCompleteMatch_beq___boxed(lean_object* v_x_319_, lean_object* v_x_320_){
_start:
{
uint8_t v_res_321_; lean_object* v_r_322_; 
v_res_321_ = lp_aesop_Aesop_instBEqCompleteMatch_beq(v_x_319_, v_x_320_);
lean_dec_ref(v_x_320_);
lean_dec_ref(v_x_319_);
v_r_322_ = lean_box(v_res_321_);
return v_r_322_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0(lean_object* v_xs_323_, lean_object* v_ys_324_, lean_object* v_hsz_325_, lean_object* v_x_326_, lean_object* v_x_327_){
_start:
{
uint8_t v___x_328_; 
v___x_328_ = lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___redArg(v_xs_323_, v_ys_324_, v_x_326_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0___boxed(lean_object* v_xs_329_, lean_object* v_ys_330_, lean_object* v_hsz_331_, lean_object* v_x_332_, lean_object* v_x_333_){
_start:
{
uint8_t v_res_334_; lean_object* v_r_335_; 
v_res_334_ = lp_aesop_Array_isEqvAux___at___00Aesop_instBEqCompleteMatch_beq_spec__0(v_xs_329_, v_ys_330_, v_hsz_331_, v_x_332_, v_x_333_);
lean_dec_ref(v_ys_330_);
lean_dec_ref(v_xs_329_);
v_r_335_ = lean_box(v_res_334_);
return v_r_335_;
}
}
LEAN_EXPORT uint64_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0_spec__0(lean_object* v_x2_338_, lean_object* v_as_339_, size_t v_i_340_, size_t v_stop_341_, uint64_t v_b_342_){
_start:
{
uint64_t v___y_344_; uint8_t v___x_348_; 
v___x_348_ = lean_usize_dec_eq(v_i_340_, v_stop_341_);
if (v___x_348_ == 0)
{
lean_object* v_subst_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v_subst_349_ = lean_ctor_get(v_x2_338_, 0);
v___x_350_ = lean_array_uget_borrowed(v_as_339_, v_i_340_);
v___x_351_ = lp_aesop_Aesop_Substitution_find_x3f(v___x_350_, v_subst_349_);
if (lean_obj_tag(v___x_351_) == 0)
{
uint64_t v___x_352_; uint64_t v___x_353_; 
v___x_352_ = 11ULL;
v___x_353_ = lean_uint64_mix_hash(v_b_342_, v___x_352_);
v___y_344_ = v___x_353_;
goto v___jp_343_;
}
else
{
lean_object* v_val_354_; uint64_t v___x_355_; uint64_t v___x_356_; uint64_t v___x_357_; uint64_t v___x_358_; 
v_val_354_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_val_354_);
lean_dec_ref_known(v___x_351_, 1);
v___x_355_ = l_Lean_Expr_hash(v_val_354_);
lean_dec(v_val_354_);
v___x_356_ = 13ULL;
v___x_357_ = lean_uint64_mix_hash(v___x_355_, v___x_356_);
v___x_358_ = lean_uint64_mix_hash(v_b_342_, v___x_357_);
v___y_344_ = v___x_358_;
goto v___jp_343_;
}
}
else
{
return v_b_342_;
}
v___jp_343_:
{
size_t v___x_345_; size_t v___x_346_; 
v___x_345_ = ((size_t)1ULL);
v___x_346_ = lean_usize_add(v_i_340_, v___x_345_);
v_i_340_ = v___x_346_;
v_b_342_ = v___y_344_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0_spec__0___boxed(lean_object* v_x2_359_, lean_object* v_as_360_, lean_object* v_i_361_, lean_object* v_stop_362_, lean_object* v_b_363_){
_start:
{
size_t v_i_boxed_364_; size_t v_stop_boxed_365_; uint64_t v_b_boxed_366_; uint64_t v_res_367_; lean_object* v_r_368_; 
v_i_boxed_364_ = lean_unbox_usize(v_i_361_);
lean_dec(v_i_361_);
v_stop_boxed_365_ = lean_unbox_usize(v_stop_362_);
lean_dec(v_stop_362_);
v_b_boxed_366_ = lean_unbox_uint64(v_b_363_);
lean_dec_ref(v_b_363_);
v_res_367_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0_spec__0(v_x2_359_, v_as_360_, v_i_boxed_364_, v_stop_boxed_365_, v_b_boxed_366_);
lean_dec_ref(v_as_360_);
lean_dec_ref(v_x2_359_);
v_r_368_ = lean_box_uint64(v_res_367_);
return v_r_368_;
}
}
LEAN_EXPORT uint64_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0(lean_object* v_x2_369_, lean_object* v_as_370_, size_t v_i_371_, size_t v_stop_372_, uint64_t v_b_373_){
_start:
{
uint64_t v___y_375_; uint8_t v___x_379_; 
v___x_379_ = lean_usize_dec_eq(v_i_371_, v_stop_372_);
if (v___x_379_ == 0)
{
lean_object* v_subst_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v_subst_380_ = lean_ctor_get(v_x2_369_, 0);
v___x_381_ = lean_array_uget_borrowed(v_as_370_, v_i_371_);
v___x_382_ = lp_aesop_Aesop_Substitution_find_x3f(v___x_381_, v_subst_380_);
if (lean_obj_tag(v___x_382_) == 0)
{
uint64_t v___x_383_; uint64_t v___x_384_; 
v___x_383_ = 11ULL;
v___x_384_ = lean_uint64_mix_hash(v_b_373_, v___x_383_);
v___y_375_ = v___x_384_;
goto v___jp_374_;
}
else
{
lean_object* v_val_385_; uint64_t v___x_386_; uint64_t v___x_387_; uint64_t v___x_388_; uint64_t v___x_389_; 
v_val_385_ = lean_ctor_get(v___x_382_, 0);
lean_inc(v_val_385_);
lean_dec_ref_known(v___x_382_, 1);
v___x_386_ = l_Lean_Expr_hash(v_val_385_);
lean_dec(v_val_385_);
v___x_387_ = 13ULL;
v___x_388_ = lean_uint64_mix_hash(v___x_386_, v___x_387_);
v___x_389_ = lean_uint64_mix_hash(v_b_373_, v___x_388_);
v___y_375_ = v___x_389_;
goto v___jp_374_;
}
}
else
{
return v_b_373_;
}
v___jp_374_:
{
size_t v___x_376_; size_t v___x_377_; uint64_t v___x_378_; 
v___x_376_ = ((size_t)1ULL);
v___x_377_ = lean_usize_add(v_i_371_, v___x_376_);
v___x_378_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0_spec__0(v_x2_369_, v_as_370_, v___x_377_, v_stop_372_, v___y_375_);
return v___x_378_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0___boxed(lean_object* v_x2_390_, lean_object* v_as_391_, lean_object* v_i_392_, lean_object* v_stop_393_, lean_object* v_b_394_){
_start:
{
size_t v_i_boxed_395_; size_t v_stop_boxed_396_; uint64_t v_b_boxed_397_; uint64_t v_res_398_; lean_object* v_r_399_; 
v_i_boxed_395_ = lean_unbox_usize(v_i_392_);
lean_dec(v_i_392_);
v_stop_boxed_396_ = lean_unbox_usize(v_stop_393_);
lean_dec(v_stop_393_);
v_b_boxed_397_ = lean_unbox_uint64(v_b_394_);
lean_dec_ref(v_b_394_);
v_res_398_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0(v_x2_390_, v_as_391_, v_i_boxed_395_, v_stop_boxed_396_, v_b_boxed_397_);
lean_dec_ref(v_as_391_);
lean_dec_ref(v_x2_390_);
v_r_399_ = lean_box_uint64(v_res_398_);
return v_r_399_;
}
}
LEAN_EXPORT uint64_t lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__1(lean_object* v_as_400_, size_t v_i_401_, size_t v_stop_402_, uint64_t v_b_403_){
_start:
{
uint64_t v___y_405_; uint8_t v___x_409_; 
v___x_409_ = lean_usize_dec_eq(v_i_401_, v_stop_402_);
if (v___x_409_ == 0)
{
lean_object* v___x_410_; lean_object* v_level_411_; lean_object* v_forwardDeps_412_; lean_object* v_conclusionDeps_413_; uint64_t v_h_414_; lean_object* v___x_415_; uint64_t v___y_417_; lean_object* v___x_431_; uint8_t v___x_432_; 
v___x_410_ = lean_array_uget_borrowed(v_as_400_, v_i_401_);
v_level_411_ = lean_ctor_get(v___x_410_, 2);
v_forwardDeps_412_ = lean_ctor_get(v___x_410_, 3);
v_conclusionDeps_413_ = lean_ctor_get(v___x_410_, 4);
v_h_414_ = lp_aesop_Aesop_instHashableSlotIndex_hash(v_level_411_);
v___x_415_ = lean_unsigned_to_nat(0u);
v___x_431_ = lean_array_get_size(v_forwardDeps_412_);
v___x_432_ = lean_nat_dec_lt(v___x_415_, v___x_431_);
if (v___x_432_ == 0)
{
v___y_417_ = v_h_414_;
goto v___jp_416_;
}
else
{
uint8_t v___x_433_; 
v___x_433_ = lean_nat_dec_le(v___x_431_, v___x_431_);
if (v___x_433_ == 0)
{
if (v___x_432_ == 0)
{
v___y_417_ = v_h_414_;
goto v___jp_416_;
}
else
{
size_t v___x_434_; size_t v___x_435_; uint64_t v___x_436_; 
v___x_434_ = ((size_t)0ULL);
v___x_435_ = lean_usize_of_nat(v___x_431_);
v___x_436_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0(v___x_410_, v_forwardDeps_412_, v___x_434_, v___x_435_, v_h_414_);
v___y_417_ = v___x_436_;
goto v___jp_416_;
}
}
else
{
size_t v___x_437_; size_t v___x_438_; uint64_t v___x_439_; 
v___x_437_ = ((size_t)0ULL);
v___x_438_ = lean_usize_of_nat(v___x_431_);
v___x_439_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0(v___x_410_, v_forwardDeps_412_, v___x_437_, v___x_438_, v_h_414_);
v___y_417_ = v___x_439_;
goto v___jp_416_;
}
}
v___jp_416_:
{
lean_object* v___x_418_; uint8_t v___x_419_; 
v___x_418_ = lean_array_get_size(v_conclusionDeps_413_);
v___x_419_ = lean_nat_dec_lt(v___x_415_, v___x_418_);
if (v___x_419_ == 0)
{
uint64_t v___x_420_; 
v___x_420_ = lean_uint64_mix_hash(v_b_403_, v___y_417_);
v___y_405_ = v___x_420_;
goto v___jp_404_;
}
else
{
uint8_t v___x_421_; 
v___x_421_ = lean_nat_dec_le(v___x_418_, v___x_418_);
if (v___x_421_ == 0)
{
if (v___x_419_ == 0)
{
uint64_t v___x_422_; 
v___x_422_ = lean_uint64_mix_hash(v_b_403_, v___y_417_);
v___y_405_ = v___x_422_;
goto v___jp_404_;
}
else
{
size_t v___x_423_; size_t v___x_424_; uint64_t v___x_425_; uint64_t v___x_426_; 
v___x_423_ = ((size_t)0ULL);
v___x_424_ = lean_usize_of_nat(v___x_418_);
v___x_425_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0(v___x_410_, v_conclusionDeps_413_, v___x_423_, v___x_424_, v___y_417_);
v___x_426_ = lean_uint64_mix_hash(v_b_403_, v___x_425_);
v___y_405_ = v___x_426_;
goto v___jp_404_;
}
}
else
{
size_t v___x_427_; size_t v___x_428_; uint64_t v___x_429_; uint64_t v___x_430_; 
v___x_427_ = ((size_t)0ULL);
v___x_428_ = lean_usize_of_nat(v___x_418_);
v___x_429_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__0(v___x_410_, v_conclusionDeps_413_, v___x_427_, v___x_428_, v___y_417_);
v___x_430_ = lean_uint64_mix_hash(v_b_403_, v___x_429_);
v___y_405_ = v___x_430_;
goto v___jp_404_;
}
}
}
}
else
{
return v_b_403_;
}
v___jp_404_:
{
size_t v___x_406_; size_t v___x_407_; 
v___x_406_ = ((size_t)1ULL);
v___x_407_ = lean_usize_add(v_i_401_, v___x_406_);
v_i_401_ = v___x_407_;
v_b_403_ = v___y_405_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__1___boxed(lean_object* v_as_440_, lean_object* v_i_441_, lean_object* v_stop_442_, lean_object* v_b_443_){
_start:
{
size_t v_i_boxed_444_; size_t v_stop_boxed_445_; uint64_t v_b_boxed_446_; uint64_t v_res_447_; lean_object* v_r_448_; 
v_i_boxed_444_ = lean_unbox_usize(v_i_441_);
lean_dec(v_i_441_);
v_stop_boxed_445_ = lean_unbox_usize(v_stop_442_);
lean_dec(v_stop_442_);
v_b_boxed_446_ = lean_unbox_uint64(v_b_443_);
lean_dec_ref(v_b_443_);
v_res_447_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__1(v_as_440_, v_i_boxed_444_, v_stop_boxed_445_, v_b_boxed_446_);
lean_dec_ref(v_as_440_);
v_r_448_ = lean_box_uint64(v_res_447_);
return v_r_448_;
}
}
static uint64_t _init_lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0(void){
_start:
{
uint64_t v___x_449_; uint64_t v___x_450_; uint64_t v___x_451_; 
v___x_449_ = 7ULL;
v___x_450_ = 0ULL;
v___x_451_ = lean_uint64_mix_hash(v___x_450_, v___x_449_);
return v___x_451_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableCompleteMatch_hash(lean_object* v_x_452_){
_start:
{
uint64_t v___x_453_; uint64_t v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; uint8_t v___x_457_; 
v___x_453_ = 0ULL;
v___x_454_ = 7ULL;
v___x_455_ = lean_unsigned_to_nat(0u);
v___x_456_ = lean_array_get_size(v_x_452_);
v___x_457_ = lean_nat_dec_lt(v___x_455_, v___x_456_);
if (v___x_457_ == 0)
{
uint64_t v___x_458_; 
v___x_458_ = lean_uint64_once(&lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0, &lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0_once, _init_lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0);
return v___x_458_;
}
else
{
uint8_t v___x_459_; 
v___x_459_ = lean_nat_dec_le(v___x_456_, v___x_456_);
if (v___x_459_ == 0)
{
if (v___x_457_ == 0)
{
uint64_t v___x_460_; 
v___x_460_ = lean_uint64_once(&lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0, &lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0_once, _init_lp_aesop_Aesop_instHashableCompleteMatch_hash___closed__0);
return v___x_460_;
}
else
{
size_t v___x_461_; size_t v___x_462_; uint64_t v___x_463_; uint64_t v___x_464_; 
v___x_461_ = ((size_t)0ULL);
v___x_462_ = lean_usize_of_nat(v___x_456_);
v___x_463_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__1(v_x_452_, v___x_461_, v___x_462_, v___x_454_);
v___x_464_ = lean_uint64_mix_hash(v___x_453_, v___x_463_);
return v___x_464_;
}
}
else
{
size_t v___x_465_; size_t v___x_466_; uint64_t v___x_467_; uint64_t v___x_468_; 
v___x_465_ = ((size_t)0ULL);
v___x_466_ = lean_usize_of_nat(v___x_456_);
v___x_467_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_instHashableCompleteMatch_hash_spec__1(v_x_452_, v___x_465_, v___x_466_, v___x_454_);
v___x_468_ = lean_uint64_mix_hash(v___x_453_, v___x_467_);
return v___x_468_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableCompleteMatch_hash___boxed(lean_object* v_x_469_){
_start:
{
uint64_t v_res_470_; lean_object* v_r_471_; 
v_res_470_ = lp_aesop_Aesop_instHashableCompleteMatch_hash(v_x_469_);
lean_dec_ref(v_x_469_);
v_r_471_ = lean_box_uint64(v_res_470_);
return v_r_471_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdCompleteMatch___lam__2(lean_object* v___f_475_, lean_object* v_m_u2081_476_, lean_object* v_m_u2082_477_){
_start:
{
uint8_t v___x_478_; 
v___x_478_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v___f_475_, v_m_u2081_476_, v_m_u2082_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdCompleteMatch___lam__2___boxed(lean_object* v___f_479_, lean_object* v_m_u2081_480_, lean_object* v_m_u2082_481_){
_start:
{
uint8_t v_res_482_; lean_object* v_r_483_; 
v_res_482_ = lp_aesop_Aesop_instOrdCompleteMatch___lam__2(v___f_479_, v_m_u2081_480_, v_m_u2082_481_);
lean_dec_ref(v_m_u2082_481_);
lean_dec_ref(v_m_u2081_480_);
v_r_483_ = lean_box(v_res_482_);
return v_r_483_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRuleMatch_default___closed__0(void){
_start:
{
lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; 
v___x_487_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedCompleteMatch_default));
v___x_488_ = lp_aesop_Aesop_instInhabitedForwardRule_default;
v___x_489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_489_, 0, v___x_488_);
lean_ctor_set(v___x_489_, 1, v___x_487_);
return v___x_489_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRuleMatch_default(void){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardRuleMatch_default___closed__0, &lp_aesop_Aesop_instInhabitedForwardRuleMatch_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedForwardRuleMatch_default___closed__0);
return v___x_490_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRuleMatch(void){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lp_aesop_Aesop_instInhabitedForwardRuleMatch_default;
return v___x_491_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqForwardRuleMatch_beq(lean_object* v_x_492_, lean_object* v_x_493_){
_start:
{
lean_object* v_rule_494_; lean_object* v_match_495_; lean_object* v_rule_496_; lean_object* v_match_497_; uint8_t v___y_499_; lean_object* v_name_501_; lean_object* v_name_502_; lean_object* v_name_503_; uint8_t v_builder_504_; uint8_t v_phase_505_; uint8_t v_scope_506_; uint64_t v_hash_507_; lean_object* v_name_508_; uint8_t v_builder_509_; uint8_t v_phase_510_; uint8_t v_scope_511_; uint64_t v_hash_512_; uint8_t v___y_514_; uint8_t v___x_518_; 
v_rule_494_ = lean_ctor_get(v_x_492_, 0);
v_match_495_ = lean_ctor_get(v_x_492_, 1);
v_rule_496_ = lean_ctor_get(v_x_493_, 0);
v_match_497_ = lean_ctor_get(v_x_493_, 1);
v_name_501_ = lean_ctor_get(v_rule_494_, 1);
v_name_502_ = lean_ctor_get(v_rule_496_, 1);
v_name_503_ = lean_ctor_get(v_name_501_, 0);
v_builder_504_ = lean_ctor_get_uint8(v_name_501_, sizeof(void*)*1 + 8);
v_phase_505_ = lean_ctor_get_uint8(v_name_501_, sizeof(void*)*1 + 9);
v_scope_506_ = lean_ctor_get_uint8(v_name_501_, sizeof(void*)*1 + 10);
v_hash_507_ = lean_ctor_get_uint64(v_name_501_, sizeof(void*)*1);
v_name_508_ = lean_ctor_get(v_name_502_, 0);
v_builder_509_ = lean_ctor_get_uint8(v_name_502_, sizeof(void*)*1 + 8);
v_phase_510_ = lean_ctor_get_uint8(v_name_502_, sizeof(void*)*1 + 9);
v_scope_511_ = lean_ctor_get_uint8(v_name_502_, sizeof(void*)*1 + 10);
v_hash_512_ = lean_ctor_get_uint64(v_name_502_, sizeof(void*)*1);
v___x_518_ = lean_uint64_dec_eq(v_hash_507_, v_hash_512_);
if (v___x_518_ == 0)
{
v___y_514_ = v___x_518_;
goto v___jp_513_;
}
else
{
uint8_t v___x_519_; 
v___x_519_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_504_, v_builder_509_);
v___y_514_ = v___x_519_;
goto v___jp_513_;
}
v___jp_498_:
{
if (v___y_499_ == 0)
{
return v___y_499_;
}
else
{
uint8_t v___x_500_; 
v___x_500_ = lp_aesop_Aesop_instBEqCompleteMatch_beq(v_match_495_, v_match_497_);
return v___x_500_;
}
}
v___jp_513_:
{
if (v___y_514_ == 0)
{
return v___y_514_;
}
else
{
uint8_t v___x_515_; 
v___x_515_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_505_, v_phase_510_);
if (v___x_515_ == 0)
{
v___y_499_ = v___x_515_;
goto v___jp_498_;
}
else
{
uint8_t v___x_516_; 
v___x_516_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_506_, v_scope_511_);
if (v___x_516_ == 0)
{
v___y_499_ = v___x_516_;
goto v___jp_498_;
}
else
{
uint8_t v___x_517_; 
v___x_517_ = lean_name_eq(v_name_503_, v_name_508_);
v___y_499_ = v___x_517_;
goto v___jp_498_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqForwardRuleMatch_beq___boxed(lean_object* v_x_520_, lean_object* v_x_521_){
_start:
{
uint8_t v_res_522_; lean_object* v_r_523_; 
v_res_522_ = lp_aesop_Aesop_instBEqForwardRuleMatch_beq(v_x_520_, v_x_521_);
lean_dec_ref(v_x_521_);
lean_dec_ref(v_x_520_);
v_r_523_ = lean_box(v_res_522_);
return v_r_523_;
}
}
LEAN_EXPORT uint64_t lp_aesop_Aesop_instHashableForwardRuleMatch_hash(lean_object* v_x_526_){
_start:
{
lean_object* v_rule_527_; lean_object* v_name_528_; lean_object* v_match_529_; uint64_t v_hash_530_; uint64_t v___x_531_; uint64_t v___x_532_; uint64_t v___x_533_; uint64_t v___x_534_; 
v_rule_527_ = lean_ctor_get(v_x_526_, 0);
v_name_528_ = lean_ctor_get(v_rule_527_, 1);
v_match_529_ = lean_ctor_get(v_x_526_, 1);
v_hash_530_ = lean_ctor_get_uint64(v_name_528_, sizeof(void*)*1);
v___x_531_ = 0ULL;
v___x_532_ = lean_uint64_mix_hash(v___x_531_, v_hash_530_);
v___x_533_ = lp_aesop_Aesop_instHashableCompleteMatch_hash(v_match_529_);
v___x_534_ = lean_uint64_mix_hash(v___x_532_, v___x_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instHashableForwardRuleMatch_hash___boxed(lean_object* v_x_535_){
_start:
{
uint64_t v_res_536_; lean_object* v_r_537_; 
v_res_536_ = lp_aesop_Aesop_instHashableForwardRuleMatch_hash(v_x_535_);
lean_dec_ref(v_x_535_);
v_r_537_ = lean_box_uint64(v_res_536_);
return v_r_537_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRuleMatch_ord___lam__2(lean_object* v___f_540_, lean_object* v_m_u2081_541_, lean_object* v_m_u2082_542_){
_start:
{
lean_object* v_rule_543_; lean_object* v_match_544_; lean_object* v_rule_545_; lean_object* v_match_546_; uint8_t v___y_548_; lean_object* v_name_550_; lean_object* v_prio_551_; lean_object* v_name_552_; lean_object* v_prio_553_; uint8_t v___x_554_; 
v_rule_543_ = lean_ctor_get(v_m_u2081_541_, 0);
v_match_544_ = lean_ctor_get(v_m_u2081_541_, 1);
v_rule_545_ = lean_ctor_get(v_m_u2082_542_, 0);
v_match_546_ = lean_ctor_get(v_m_u2082_542_, 1);
v_name_550_ = lean_ctor_get(v_rule_543_, 1);
v_prio_551_ = lean_ctor_get(v_rule_543_, 3);
v_name_552_ = lean_ctor_get(v_rule_545_, 1);
v_prio_553_ = lean_ctor_get(v_rule_545_, 3);
v___x_554_ = lp_aesop_Aesop_ForwardRulePriority_compare(v_prio_551_, v_prio_553_);
if (v___x_554_ == 1)
{
uint8_t v___x_555_; 
v___x_555_ = lp_aesop_Aesop_RuleName_compare(v_name_550_, v_name_552_);
v___y_548_ = v___x_555_;
goto v___jp_547_;
}
else
{
v___y_548_ = v___x_554_;
goto v___jp_547_;
}
v___jp_547_:
{
if (v___y_548_ == 1)
{
uint8_t v___x_549_; 
v___x_549_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v___f_540_, v_match_544_, v_match_546_);
return v___x_549_;
}
else
{
lean_dec_ref(v___f_540_);
return v___y_548_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_ord___lam__2___boxed(lean_object* v___f_556_, lean_object* v_m_u2081_557_, lean_object* v_m_u2082_558_){
_start:
{
uint8_t v_res_559_; lean_object* v_r_560_; 
v_res_559_ = lp_aesop_Aesop_ForwardRuleMatch_ord___lam__2(v___f_556_, v_m_u2081_557_, v_m_u2082_558_);
lean_dec_ref(v_m_u2082_558_);
lean_dec_ref(v_m_u2081_557_);
v_r_560_ = lean_box(v_res_559_);
return v_r_560_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRuleMatch_le(lean_object* v_m_u2081_564_, lean_object* v_m_u2082_565_){
_start:
{
uint8_t v___y_567_; lean_object* v_rule_570_; lean_object* v_rule_571_; lean_object* v_match_572_; lean_object* v_match_573_; lean_object* v_name_574_; lean_object* v_prio_575_; lean_object* v_name_576_; lean_object* v_prio_577_; lean_object* v___f_578_; uint8_t v___y_580_; uint8_t v___x_582_; 
v_rule_570_ = lean_ctor_get(v_m_u2081_564_, 0);
v_rule_571_ = lean_ctor_get(v_m_u2082_565_, 0);
v_match_572_ = lean_ctor_get(v_m_u2081_564_, 1);
v_match_573_ = lean_ctor_get(v_m_u2082_565_, 1);
v_name_574_ = lean_ctor_get(v_rule_570_, 1);
v_prio_575_ = lean_ctor_get(v_rule_570_, 3);
v_name_576_ = lean_ctor_get(v_rule_571_, 1);
v_prio_577_ = lean_ctor_get(v_rule_571_, 3);
v___f_578_ = ((lean_object*)(lp_aesop_Aesop_Match_instOrd___closed__0));
v___x_582_ = lp_aesop_Aesop_ForwardRulePriority_compare(v_prio_575_, v_prio_577_);
if (v___x_582_ == 1)
{
uint8_t v___x_583_; 
v___x_583_ = lp_aesop_Aesop_RuleName_compare(v_name_574_, v_name_576_);
v___y_580_ = v___x_583_;
goto v___jp_579_;
}
else
{
v___y_580_ = v___x_582_;
goto v___jp_579_;
}
v___jp_566_:
{
if (v___y_567_ == 2)
{
uint8_t v___x_568_; 
v___x_568_ = 0;
return v___x_568_;
}
else
{
uint8_t v___x_569_; 
v___x_569_ = 1;
return v___x_569_;
}
}
v___jp_579_:
{
if (v___y_580_ == 1)
{
uint8_t v___x_581_; 
v___x_581_ = lp_aesop_Aesop_compareArraySizeThenLex___redArg(v___f_578_, v_match_572_, v_match_573_);
v___y_567_ = v___x_581_;
goto v___jp_566_;
}
else
{
v___y_567_ = v___y_580_;
goto v___jp_566_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_le___boxed(lean_object* v_m_u2081_584_, lean_object* v_m_u2082_585_){
_start:
{
uint8_t v_res_586_; lean_object* v_r_587_; 
v_res_586_ = lp_aesop_Aesop_ForwardRuleMatch_le(v_m_u2081_584_, v_m_u2082_585_);
lean_dec_ref(v_m_u2082_585_);
lean_dec_ref(v_m_u2081_584_);
v_r_587_ = lean_box(v_res_586_);
return v_r_587_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule_Forward(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Forward_Match_Types(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedMatch_default = _init_lp_aesop_Aesop_instInhabitedMatch_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedMatch_default);
lp_aesop_Aesop_instInhabitedMatch = _init_lp_aesop_Aesop_instInhabitedMatch();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedMatch);
lp_aesop_Aesop_instInhabitedForwardRuleMatch_default = _init_lp_aesop_Aesop_instInhabitedForwardRuleMatch_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRuleMatch_default);
lp_aesop_Aesop_instInhabitedForwardRuleMatch = _init_lp_aesop_Aesop_instInhabitedForwardRuleMatch();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRuleMatch);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Forward_Match_Types(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Rule_Forward(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_Match_Types(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Match_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Forward_Match_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Forward_Match_Types(builtin);
}
#ifdef __cplusplus
}
#endif
