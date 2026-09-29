// Lean compiler output
// Module: Mathlib.Lean.Meta.RefinedDiscrTree.Basic
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_instToFormatFormat___lam__0___boxed(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Std_Format_joinSep___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_String_quote(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_AssocList_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_List_format___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqLiteral_beq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
lean_object* lean_expr_dbg_to_string(lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint64_t l_Lean_Literal_hash(lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_mkLevelParam(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_paren(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_star_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_star_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_labelledStar_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_labelledStar_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_opaque_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_opaque_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_const_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_const_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_fvar_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_fvar_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_bvar_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_bvar_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lit_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lit_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_sort_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_sort_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lam_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lam_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_forall_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_forall_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_proj_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_proj_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey_default;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey;
LEAN_EXPORT uint8_t lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey___closed__0_value;
LEAN_EXPORT uint64_t lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_hash(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_hash___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_instHashableKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instHashableKey___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instHashableKey___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instHashableKey = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instHashableKey___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "◾"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__4_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__6_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__7 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__7_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__8 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__8_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__8_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__9 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__9_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⟨#"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__10 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__11 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__11_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Sort"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__12 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__13 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__13_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "λ"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__14 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__14_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__15 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__15_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∀"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__16 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__16_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__17 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__17_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__18 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__18_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__19 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__19_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatKey___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatKey___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatKey___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatKey = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatKey___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "illegal discrimination tree entry: "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_parenthesize(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_parenthesize___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__0;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__2;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 3, .m_data = "λ, "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__4;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " → "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_arity(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_arity___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_star_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_star_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_expr_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_expr_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = ".star"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = ".expr "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatStackEntry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatStackEntry___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatStackEntry___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatStackEntry = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatStackEntry___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry;
static const lean_array_object lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__0 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__0_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__1 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__1_value;
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__2 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__2_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__2_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__3 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__3_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__4 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__4_value;
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__5 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__5_value;
static const lean_string_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__6 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__6_value;
static lean_once_cell_t lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__7;
static lean_once_cell_t lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__5_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__9 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__9_value;
static const lean_ctor_object lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__6_value)}};
static const lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__10 = (const lean_object*)&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3_spec__6_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "todo: "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "stack: "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "results: "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__4_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__3(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatLazyEntry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatLazyEntry___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatLazyEntry___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatLazyEntry = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormatLazyEntry___closed__0_value;
static const lean_array_object lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__1;
static const lean_array_object lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__4___boxed(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " =>"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__1_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Std_instToFormatFormat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "<empty node>"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__5_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__12_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__11 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__11_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__10_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__9 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__9_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__8_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__7 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__7_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__6_value),((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__7_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__13 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__13_value),((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__8_value),((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__9_value),((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__10_value),((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__11_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__14_value),((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__15 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__15_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "* =>"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__16_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__17 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__17_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__18_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "entries: "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__19 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__19_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__19_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__20 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__20_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__1_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__21 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__21_value;
static const lean_array_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__22 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__22_value;
static const lean_closure_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__4___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__23 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__23_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "pending entries: "};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__24 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__24_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__24_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__25 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__25_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Discrimination tree flowchart:\n"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "<empty discrimination tree>"};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorIdx(lean_object* v_x_1_){
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
case 9:
{
lean_object* v___x_11_; 
v___x_11_ = lean_unsigned_to_nat(9u);
return v___x_11_;
}
default: 
{
lean_object* v___x_12_; 
v___x_12_ = lean_unsigned_to_nat(10u);
return v___x_12_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorIdx___boxed(lean_object* v_x_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorIdx(v_x_13_);
lean_dec(v_x_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(lean_object* v_t_15_, lean_object* v_k_16_){
_start:
{
switch(lean_obj_tag(v_t_15_))
{
case 1:
{
lean_object* v_id_17_; lean_object* v___x_18_; 
v_id_17_ = lean_ctor_get(v_t_15_, 0);
lean_inc(v_id_17_);
lean_dec_ref_known(v_t_15_, 1);
v___x_18_ = lean_apply_1(v_k_16_, v_id_17_);
return v___x_18_;
}
case 3:
{
lean_object* v_declName_19_; lean_object* v_nargs_20_; lean_object* v___x_21_; 
v_declName_19_ = lean_ctor_get(v_t_15_, 0);
lean_inc(v_declName_19_);
v_nargs_20_ = lean_ctor_get(v_t_15_, 1);
lean_inc(v_nargs_20_);
lean_dec_ref_known(v_t_15_, 2);
v___x_21_ = lean_apply_2(v_k_16_, v_declName_19_, v_nargs_20_);
return v___x_21_;
}
case 4:
{
lean_object* v_fvarId_22_; lean_object* v_nargs_23_; lean_object* v___x_24_; 
v_fvarId_22_ = lean_ctor_get(v_t_15_, 0);
lean_inc(v_fvarId_22_);
v_nargs_23_ = lean_ctor_get(v_t_15_, 1);
lean_inc(v_nargs_23_);
lean_dec_ref_known(v_t_15_, 2);
v___x_24_ = lean_apply_2(v_k_16_, v_fvarId_22_, v_nargs_23_);
return v___x_24_;
}
case 5:
{
lean_object* v_deBruijnIndex_25_; lean_object* v_nargs_26_; lean_object* v___x_27_; 
v_deBruijnIndex_25_ = lean_ctor_get(v_t_15_, 0);
lean_inc(v_deBruijnIndex_25_);
v_nargs_26_ = lean_ctor_get(v_t_15_, 1);
lean_inc(v_nargs_26_);
lean_dec_ref_known(v_t_15_, 2);
v___x_27_ = lean_apply_2(v_k_16_, v_deBruijnIndex_25_, v_nargs_26_);
return v___x_27_;
}
case 6:
{
lean_object* v_v_28_; lean_object* v___x_29_; 
v_v_28_ = lean_ctor_get(v_t_15_, 0);
lean_inc_ref(v_v_28_);
lean_dec_ref_known(v_t_15_, 1);
v___x_29_ = lean_apply_1(v_k_16_, v_v_28_);
return v___x_29_;
}
case 10:
{
lean_object* v_typeName_30_; lean_object* v_idx_31_; lean_object* v_nargs_32_; lean_object* v___x_33_; 
v_typeName_30_ = lean_ctor_get(v_t_15_, 0);
lean_inc(v_typeName_30_);
v_idx_31_ = lean_ctor_get(v_t_15_, 1);
lean_inc(v_idx_31_);
v_nargs_32_ = lean_ctor_get(v_t_15_, 2);
lean_inc(v_nargs_32_);
lean_dec_ref_known(v_t_15_, 3);
v___x_33_ = lean_apply_3(v_k_16_, v_typeName_30_, v_idx_31_, v_nargs_32_);
return v___x_33_;
}
default: 
{
lean_dec(v_t_15_);
return v_k_16_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim(lean_object* v_motive_34_, lean_object* v_ctorIdx_35_, lean_object* v_t_36_, lean_object* v_h_37_, lean_object* v_k_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_36_, v_k_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___boxed(lean_object* v_motive_40_, lean_object* v_ctorIdx_41_, lean_object* v_t_42_, lean_object* v_h_43_, lean_object* v_k_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim(v_motive_40_, v_ctorIdx_41_, v_t_42_, v_h_43_, v_k_44_);
lean_dec(v_ctorIdx_41_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_star_elim___redArg(lean_object* v_t_46_, lean_object* v_star_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_46_, v_star_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_star_elim(lean_object* v_motive_49_, lean_object* v_t_50_, lean_object* v_h_51_, lean_object* v_star_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_50_, v_star_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_labelledStar_elim___redArg(lean_object* v_t_54_, lean_object* v_labelledStar_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_54_, v_labelledStar_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_labelledStar_elim(lean_object* v_motive_57_, lean_object* v_t_58_, lean_object* v_h_59_, lean_object* v_labelledStar_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_58_, v_labelledStar_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_opaque_elim___redArg(lean_object* v_t_62_, lean_object* v_opaque_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_62_, v_opaque_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_opaque_elim(lean_object* v_motive_65_, lean_object* v_t_66_, lean_object* v_h_67_, lean_object* v_opaque_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_66_, v_opaque_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_const_elim___redArg(lean_object* v_t_70_, lean_object* v_const_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_70_, v_const_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_const_elim(lean_object* v_motive_73_, lean_object* v_t_74_, lean_object* v_h_75_, lean_object* v_const_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_74_, v_const_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_fvar_elim___redArg(lean_object* v_t_78_, lean_object* v_fvar_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_78_, v_fvar_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_fvar_elim(lean_object* v_motive_81_, lean_object* v_t_82_, lean_object* v_h_83_, lean_object* v_fvar_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_82_, v_fvar_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_bvar_elim___redArg(lean_object* v_t_86_, lean_object* v_bvar_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_86_, v_bvar_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_bvar_elim(lean_object* v_motive_89_, lean_object* v_t_90_, lean_object* v_h_91_, lean_object* v_bvar_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_90_, v_bvar_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lit_elim___redArg(lean_object* v_t_94_, lean_object* v_lit_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_94_, v_lit_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lit_elim(lean_object* v_motive_97_, lean_object* v_t_98_, lean_object* v_h_99_, lean_object* v_lit_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_98_, v_lit_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_sort_elim___redArg(lean_object* v_t_102_, lean_object* v_sort_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_102_, v_sort_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_sort_elim(lean_object* v_motive_105_, lean_object* v_t_106_, lean_object* v_h_107_, lean_object* v_sort_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_106_, v_sort_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lam_elim___redArg(lean_object* v_t_110_, lean_object* v_lam_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_110_, v_lam_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_lam_elim(lean_object* v_motive_113_, lean_object* v_t_114_, lean_object* v_h_115_, lean_object* v_lam_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_114_, v_lam_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_forall_elim___redArg(lean_object* v_t_118_, lean_object* v_forall_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_118_, v_forall_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_forall_elim(lean_object* v_motive_121_, lean_object* v_t_122_, lean_object* v_h_123_, lean_object* v_forall_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_122_, v_forall_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_proj_elim___redArg(lean_object* v_t_126_, lean_object* v_proj_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_126_, v_proj_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_proj_elim(lean_object* v_motive_129_, lean_object* v_t_130_, lean_object* v_h_131_, lean_object* v_proj_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorElim___redArg(v_t_130_, v_proj_132_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey_default(void){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lean_box(0);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey(void){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lean_box(0);
return v___x_135_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey_beq(lean_object* v_x_136_, lean_object* v_x_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_138_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorIdx(v_x_136_);
v___x_139_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_ctorIdx(v_x_137_);
v___x_140_ = lean_nat_dec_eq(v___x_138_, v___x_139_);
lean_dec(v___x_139_);
lean_dec(v___x_138_);
if (v___x_140_ == 0)
{
return v___x_140_;
}
else
{
switch(lean_obj_tag(v_x_136_))
{
case 1:
{
lean_object* v_id_141_; lean_object* v_id_142_; uint8_t v___x_143_; 
v_id_141_ = lean_ctor_get(v_x_136_, 0);
v_id_142_ = lean_ctor_get(v_x_137_, 0);
v___x_143_ = lean_nat_dec_eq(v_id_141_, v_id_142_);
return v___x_143_;
}
case 3:
{
lean_object* v_declName_144_; lean_object* v_nargs_145_; lean_object* v_declName_146_; lean_object* v_nargs_147_; uint8_t v___x_148_; 
v_declName_144_ = lean_ctor_get(v_x_136_, 0);
v_nargs_145_ = lean_ctor_get(v_x_136_, 1);
v_declName_146_ = lean_ctor_get(v_x_137_, 0);
v_nargs_147_ = lean_ctor_get(v_x_137_, 1);
v___x_148_ = lean_name_eq(v_declName_144_, v_declName_146_);
if (v___x_148_ == 0)
{
return v___x_148_;
}
else
{
uint8_t v___x_149_; 
v___x_149_ = lean_nat_dec_eq(v_nargs_145_, v_nargs_147_);
return v___x_149_;
}
}
case 4:
{
lean_object* v_fvarId_150_; lean_object* v_nargs_151_; lean_object* v_fvarId_152_; lean_object* v_nargs_153_; uint8_t v___x_154_; 
v_fvarId_150_ = lean_ctor_get(v_x_136_, 0);
v_nargs_151_ = lean_ctor_get(v_x_136_, 1);
v_fvarId_152_ = lean_ctor_get(v_x_137_, 0);
v_nargs_153_ = lean_ctor_get(v_x_137_, 1);
v___x_154_ = l_Lean_instBEqFVarId_beq(v_fvarId_150_, v_fvarId_152_);
if (v___x_154_ == 0)
{
return v___x_154_;
}
else
{
uint8_t v___x_155_; 
v___x_155_ = lean_nat_dec_eq(v_nargs_151_, v_nargs_153_);
return v___x_155_;
}
}
case 5:
{
lean_object* v_deBruijnIndex_156_; lean_object* v_nargs_157_; lean_object* v_deBruijnIndex_158_; lean_object* v_nargs_159_; uint8_t v___x_160_; 
v_deBruijnIndex_156_ = lean_ctor_get(v_x_136_, 0);
v_nargs_157_ = lean_ctor_get(v_x_136_, 1);
v_deBruijnIndex_158_ = lean_ctor_get(v_x_137_, 0);
v_nargs_159_ = lean_ctor_get(v_x_137_, 1);
v___x_160_ = lean_nat_dec_eq(v_deBruijnIndex_156_, v_deBruijnIndex_158_);
if (v___x_160_ == 0)
{
return v___x_160_;
}
else
{
uint8_t v___x_161_; 
v___x_161_ = lean_nat_dec_eq(v_nargs_157_, v_nargs_159_);
return v___x_161_;
}
}
case 6:
{
lean_object* v_v_162_; lean_object* v_v_163_; uint8_t v___x_164_; 
v_v_162_ = lean_ctor_get(v_x_136_, 0);
v_v_163_ = lean_ctor_get(v_x_137_, 0);
v___x_164_ = l_Lean_instBEqLiteral_beq(v_v_162_, v_v_163_);
return v___x_164_;
}
case 10:
{
lean_object* v_typeName_165_; lean_object* v_idx_166_; lean_object* v_nargs_167_; lean_object* v_typeName_168_; lean_object* v_idx_169_; lean_object* v_nargs_170_; uint8_t v___x_171_; 
v_typeName_165_ = lean_ctor_get(v_x_136_, 0);
v_idx_166_ = lean_ctor_get(v_x_136_, 1);
v_nargs_167_ = lean_ctor_get(v_x_136_, 2);
v_typeName_168_ = lean_ctor_get(v_x_137_, 0);
v_idx_169_ = lean_ctor_get(v_x_137_, 1);
v_nargs_170_ = lean_ctor_get(v_x_137_, 2);
v___x_171_ = lean_name_eq(v_typeName_165_, v_typeName_168_);
if (v___x_171_ == 0)
{
return v___x_171_;
}
else
{
uint8_t v___x_172_; 
v___x_172_ = lean_nat_dec_eq(v_idx_166_, v_idx_169_);
if (v___x_172_ == 0)
{
return v___x_172_;
}
else
{
uint8_t v___x_173_; 
v___x_173_ = lean_nat_dec_eq(v_nargs_167_, v_nargs_170_);
return v___x_173_;
}
}
}
default: 
{
return v___x_140_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey_beq___boxed(lean_object* v_x_174_, lean_object* v_x_175_){
_start:
{
uint8_t v_res_176_; lean_object* v_r_177_; 
v_res_176_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey_beq(v_x_174_, v_x_175_);
lean_dec(v_x_175_);
lean_dec(v_x_174_);
v_r_177_ = lean_box(v_res_176_);
return v_r_177_;
}
}
LEAN_EXPORT uint64_t lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_hash(lean_object* v_x_180_){
_start:
{
switch(lean_obj_tag(v_x_180_))
{
case 0:
{
uint64_t v___x_181_; 
v___x_181_ = 0ULL;
return v___x_181_;
}
case 1:
{
lean_object* v_id_182_; uint64_t v___x_183_; uint64_t v___x_184_; uint64_t v___x_185_; 
v_id_182_ = lean_ctor_get(v_x_180_, 0);
v___x_183_ = 5ULL;
v___x_184_ = lean_uint64_of_nat(v_id_182_);
v___x_185_ = lean_uint64_mix_hash(v___x_183_, v___x_184_);
return v___x_185_;
}
case 2:
{
uint64_t v___x_186_; 
v___x_186_ = 1ULL;
return v___x_186_;
}
case 3:
{
lean_object* v_declName_187_; 
v_declName_187_ = lean_ctor_get(v_x_180_, 0);
if (lean_obj_tag(v_declName_187_) == 0)
{
uint64_t v___x_188_; 
v___x_188_ = 1723ULL;
return v___x_188_;
}
else
{
uint64_t v_hash_189_; 
v_hash_189_ = lean_ctor_get_uint64(v_declName_187_, sizeof(void*)*2);
return v_hash_189_;
}
}
case 4:
{
lean_object* v_fvarId_190_; lean_object* v_nargs_191_; uint64_t v___x_192_; uint64_t v___x_193_; uint64_t v___x_194_; uint64_t v___x_195_; uint64_t v___x_196_; 
v_fvarId_190_ = lean_ctor_get(v_x_180_, 0);
v_nargs_191_ = lean_ctor_get(v_x_180_, 1);
v___x_192_ = 6ULL;
v___x_193_ = l_Lean_instHashableFVarId_hash(v_fvarId_190_);
v___x_194_ = lean_uint64_of_nat(v_nargs_191_);
v___x_195_ = lean_uint64_mix_hash(v___x_193_, v___x_194_);
v___x_196_ = lean_uint64_mix_hash(v___x_192_, v___x_195_);
return v___x_196_;
}
case 5:
{
lean_object* v_deBruijnIndex_197_; lean_object* v_nargs_198_; uint64_t v___x_199_; uint64_t v___x_200_; uint64_t v___x_201_; uint64_t v___x_202_; uint64_t v___x_203_; 
v_deBruijnIndex_197_ = lean_ctor_get(v_x_180_, 0);
v_nargs_198_ = lean_ctor_get(v_x_180_, 1);
v___x_199_ = 7ULL;
v___x_200_ = lean_uint64_of_nat(v_deBruijnIndex_197_);
v___x_201_ = lean_uint64_of_nat(v_nargs_198_);
v___x_202_ = lean_uint64_mix_hash(v___x_200_, v___x_201_);
v___x_203_ = lean_uint64_mix_hash(v___x_199_, v___x_202_);
return v___x_203_;
}
case 6:
{
lean_object* v_v_204_; uint64_t v___x_205_; uint64_t v___x_206_; uint64_t v___x_207_; 
v_v_204_ = lean_ctor_get(v_x_180_, 0);
v___x_205_ = 8ULL;
v___x_206_ = l_Lean_Literal_hash(v_v_204_);
v___x_207_ = lean_uint64_mix_hash(v___x_205_, v___x_206_);
return v___x_207_;
}
case 7:
{
uint64_t v___x_208_; 
v___x_208_ = 2ULL;
return v___x_208_;
}
case 8:
{
uint64_t v___x_209_; 
v___x_209_ = 3ULL;
return v___x_209_;
}
case 9:
{
uint64_t v___x_210_; 
v___x_210_ = 4ULL;
return v___x_210_;
}
default: 
{
lean_object* v_typeName_211_; lean_object* v_idx_212_; lean_object* v_nargs_213_; uint64_t v___x_214_; uint64_t v___y_216_; 
v_typeName_211_ = lean_ctor_get(v_x_180_, 0);
v_idx_212_ = lean_ctor_get(v_x_180_, 1);
v_nargs_213_ = lean_ctor_get(v_x_180_, 2);
v___x_214_ = lean_uint64_of_nat(v_nargs_213_);
if (lean_obj_tag(v_typeName_211_) == 0)
{
uint64_t v___x_220_; 
v___x_220_ = 1723ULL;
v___y_216_ = v___x_220_;
goto v___jp_215_;
}
else
{
uint64_t v_hash_221_; 
v_hash_221_ = lean_ctor_get_uint64(v_typeName_211_, sizeof(void*)*2);
v___y_216_ = v_hash_221_;
goto v___jp_215_;
}
v___jp_215_:
{
uint64_t v___x_217_; uint64_t v___x_218_; uint64_t v___x_219_; 
v___x_217_ = lean_uint64_of_nat(v_idx_212_);
v___x_218_ = lean_uint64_mix_hash(v___y_216_, v___x_217_);
v___x_219_ = lean_uint64_mix_hash(v___x_214_, v___x_218_);
return v___x_219_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_hash___boxed(lean_object* v_x_222_){
_start:
{
uint64_t v_res_223_; lean_object* v_r_224_; 
v_res_223_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_hash(v_x_222_);
lean_dec(v_x_222_);
v_r_224_ = lean_box_uint64(v_res_223_);
return v_r_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(lean_object* v_x_257_){
_start:
{
switch(lean_obj_tag(v_x_257_))
{
case 0:
{
lean_object* v___x_258_; 
v___x_258_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__1));
return v___x_258_;
}
case 1:
{
lean_object* v_id_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_269_; 
v_id_259_ = lean_ctor_get(v_x_257_, 0);
v_isSharedCheck_269_ = !lean_is_exclusive(v_x_257_);
if (v_isSharedCheck_269_ == 0)
{
v___x_261_ = v_x_257_;
v_isShared_262_ = v_isSharedCheck_269_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_id_259_);
lean_dec(v_x_257_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_269_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_266_; 
v___x_263_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__1));
v___x_264_ = l_Nat_reprFast(v_id_259_);
if (v_isShared_262_ == 0)
{
lean_ctor_set_tag(v___x_261_, 3);
lean_ctor_set(v___x_261_, 0, v___x_264_);
v___x_266_ = v___x_261_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v___x_264_);
v___x_266_ = v_reuseFailAlloc_268_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
lean_object* v___x_267_; 
v___x_267_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_263_);
lean_ctor_set(v___x_267_, 1, v___x_266_);
return v___x_267_;
}
}
}
case 2:
{
lean_object* v___x_270_; 
v___x_270_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__3));
return v___x_270_;
}
case 5:
{
lean_object* v_deBruijnIndex_271_; lean_object* v_nargs_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_289_; 
v_deBruijnIndex_271_ = lean_ctor_get(v_x_257_, 0);
v_nargs_272_ = lean_ctor_get(v_x_257_, 1);
v_isSharedCheck_289_ = !lean_is_exclusive(v_x_257_);
if (v_isSharedCheck_289_ == 0)
{
v___x_274_ = v_x_257_;
v_isShared_275_ = v_isSharedCheck_289_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_nargs_272_);
lean_inc(v_deBruijnIndex_271_);
lean_dec(v_x_257_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_289_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_280_; 
v___x_276_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__11));
v___x_277_ = l_Nat_reprFast(v_deBruijnIndex_271_);
v___x_278_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_278_, 0, v___x_277_);
if (v_isShared_275_ == 0)
{
lean_ctor_set(v___x_274_, 1, v___x_278_);
lean_ctor_set(v___x_274_, 0, v___x_276_);
v___x_280_ = v___x_274_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_288_; 
v_reuseFailAlloc_288_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_288_, 0, v___x_276_);
lean_ctor_set(v_reuseFailAlloc_288_, 1, v___x_278_);
v___x_280_ = v_reuseFailAlloc_288_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_281_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__7));
v___x_282_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_282_, 0, v___x_280_);
lean_ctor_set(v___x_282_, 1, v___x_281_);
v___x_283_ = l_Nat_reprFast(v_nargs_272_);
v___x_284_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
v___x_285_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_285_, 0, v___x_282_);
lean_ctor_set(v___x_285_, 1, v___x_284_);
v___x_286_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__9));
v___x_287_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_285_);
lean_ctor_set(v___x_287_, 1, v___x_286_);
return v___x_287_;
}
}
}
case 6:
{
lean_object* v_v_290_; 
v_v_290_ = lean_ctor_get(v_x_257_, 0);
lean_inc_ref(v_v_290_);
lean_dec_ref_known(v_x_257_, 1);
if (lean_obj_tag(v_v_290_) == 0)
{
lean_object* v_val_291_; lean_object* v___x_293_; uint8_t v_isShared_294_; uint8_t v_isSharedCheck_299_; 
v_val_291_ = lean_ctor_get(v_v_290_, 0);
v_isSharedCheck_299_ = !lean_is_exclusive(v_v_290_);
if (v_isSharedCheck_299_ == 0)
{
v___x_293_ = v_v_290_;
v_isShared_294_ = v_isSharedCheck_299_;
goto v_resetjp_292_;
}
else
{
lean_inc(v_val_291_);
lean_dec(v_v_290_);
v___x_293_ = lean_box(0);
v_isShared_294_ = v_isSharedCheck_299_;
goto v_resetjp_292_;
}
v_resetjp_292_:
{
lean_object* v___x_295_; lean_object* v___x_297_; 
v___x_295_ = l_Nat_reprFast(v_val_291_);
if (v_isShared_294_ == 0)
{
lean_ctor_set_tag(v___x_293_, 3);
lean_ctor_set(v___x_293_, 0, v___x_295_);
v___x_297_ = v___x_293_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v___x_295_);
v___x_297_ = v_reuseFailAlloc_298_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
return v___x_297_;
}
}
}
else
{
lean_object* v_val_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_308_; 
v_val_300_ = lean_ctor_get(v_v_290_, 0);
v_isSharedCheck_308_ = !lean_is_exclusive(v_v_290_);
if (v_isSharedCheck_308_ == 0)
{
v___x_302_ = v_v_290_;
v_isShared_303_ = v_isSharedCheck_308_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_val_300_);
lean_dec(v_v_290_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_308_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v___x_304_; lean_object* v___x_306_; 
v___x_304_ = l_String_quote(v_val_300_);
if (v_isShared_303_ == 0)
{
lean_ctor_set_tag(v___x_302_, 3);
lean_ctor_set(v___x_302_, 0, v___x_304_);
v___x_306_ = v___x_302_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_307_; 
v_reuseFailAlloc_307_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_307_, 0, v___x_304_);
v___x_306_ = v_reuseFailAlloc_307_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
return v___x_306_;
}
}
}
}
case 7:
{
lean_object* v___x_309_; 
v___x_309_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__13));
return v___x_309_;
}
case 8:
{
lean_object* v___x_310_; 
v___x_310_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__15));
return v___x_310_;
}
case 9:
{
lean_object* v___x_311_; 
v___x_311_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__17));
return v___x_311_;
}
case 10:
{
lean_object* v_typeName_312_; lean_object* v_idx_313_; lean_object* v_nargs_314_; lean_object* v___x_315_; uint8_t v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v_typeName_312_ = lean_ctor_get(v_x_257_, 0);
lean_inc(v_typeName_312_);
v_idx_313_ = lean_ctor_get(v_x_257_, 1);
lean_inc(v_idx_313_);
v_nargs_314_ = lean_ctor_get(v_x_257_, 2);
lean_inc(v_nargs_314_);
lean_dec_ref_known(v_x_257_, 3);
v___x_315_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__5));
v___x_316_ = 1;
v___x_317_ = l_Lean_Name_toString(v_typeName_312_, v___x_316_);
v___x_318_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_318_, 0, v___x_317_);
v___x_319_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_319_, 0, v___x_315_);
lean_ctor_set(v___x_319_, 1, v___x_318_);
v___x_320_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__19));
v___x_321_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_321_, 0, v___x_319_);
lean_ctor_set(v___x_321_, 1, v___x_320_);
v___x_322_ = l_Nat_reprFast(v_idx_313_);
v___x_323_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_323_, 0, v___x_322_);
v___x_324_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_321_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
v___x_325_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__7));
v___x_326_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_326_, 0, v___x_324_);
lean_ctor_set(v___x_326_, 1, v___x_325_);
v___x_327_ = l_Nat_reprFast(v_nargs_314_);
v___x_328_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_328_, 0, v___x_327_);
v___x_329_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_329_, 0, v___x_326_);
lean_ctor_set(v___x_329_, 1, v___x_328_);
v___x_330_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__9));
v___x_331_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_331_, 0, v___x_329_);
lean_ctor_set(v___x_331_, 1, v___x_330_);
return v___x_331_;
}
default: 
{
lean_object* v_declName_332_; lean_object* v_nargs_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_351_; 
v_declName_332_ = lean_ctor_get(v_x_257_, 0);
v_nargs_333_ = lean_ctor_get(v_x_257_, 1);
v_isSharedCheck_351_ = !lean_is_exclusive(v_x_257_);
if (v_isSharedCheck_351_ == 0)
{
v___x_335_ = v_x_257_;
v_isShared_336_ = v_isSharedCheck_351_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_nargs_333_);
lean_inc(v_declName_332_);
lean_dec(v_x_257_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_351_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_337_; uint8_t v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_342_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__5));
v___x_338_ = 1;
v___x_339_ = l_Lean_Name_toString(v_declName_332_, v___x_338_);
v___x_340_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_340_, 0, v___x_339_);
if (v_isShared_336_ == 0)
{
lean_ctor_set_tag(v___x_335_, 5);
lean_ctor_set(v___x_335_, 1, v___x_340_);
lean_ctor_set(v___x_335_, 0, v___x_337_);
v___x_342_ = v___x_335_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_350_; 
v_reuseFailAlloc_350_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_350_, 0, v___x_337_);
lean_ctor_set(v_reuseFailAlloc_350_, 1, v___x_340_);
v___x_342_ = v_reuseFailAlloc_350_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_343_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__7));
v___x_344_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_342_);
lean_ctor_set(v___x_344_, 1, v___x_343_);
v___x_345_ = l_Nat_reprFast(v_nargs_333_);
v___x_346_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_346_, 0, v___x_345_);
v___x_347_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_344_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__9));
v___x_349_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_349_, 0, v___x_347_);
lean_ctor_set(v___x_349_, 1, v___x_348_);
return v___x_349_;
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__0(void){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_354_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1(void){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_355_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__0);
v___x_356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
return v___x_356_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2(void){
_start:
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_357_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1);
v___x_358_ = lean_unsigned_to_nat(0u);
v___x_359_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_359_, 0, v___x_358_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
lean_ctor_set(v___x_359_, 2, v___x_358_);
lean_ctor_set(v___x_359_, 3, v___x_358_);
lean_ctor_set(v___x_359_, 4, v___x_357_);
lean_ctor_set(v___x_359_, 5, v___x_357_);
lean_ctor_set(v___x_359_, 6, v___x_357_);
lean_ctor_set(v___x_359_, 7, v___x_357_);
lean_ctor_set(v___x_359_, 8, v___x_357_);
lean_ctor_set(v___x_359_, 9, v___x_357_);
return v___x_359_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__3(void){
_start:
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
v___x_360_ = lean_unsigned_to_nat(32u);
v___x_361_ = lean_mk_empty_array_with_capacity(v___x_360_);
v___x_362_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_362_, 0, v___x_361_);
return v___x_362_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__4(void){
_start:
{
size_t v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_363_ = ((size_t)5ULL);
v___x_364_ = lean_unsigned_to_nat(0u);
v___x_365_ = lean_unsigned_to_nat(32u);
v___x_366_ = lean_mk_empty_array_with_capacity(v___x_365_);
v___x_367_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__3);
v___x_368_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_368_, 0, v___x_367_);
lean_ctor_set(v___x_368_, 1, v___x_366_);
lean_ctor_set(v___x_368_, 2, v___x_364_);
lean_ctor_set(v___x_368_, 3, v___x_364_);
lean_ctor_set_usize(v___x_368_, 4, v___x_363_);
return v___x_368_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5(void){
_start:
{
lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_369_ = lean_box(1);
v___x_370_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__4);
v___x_371_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__1);
v___x_372_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v___x_370_);
lean_ctor_set(v___x_372_, 2, v___x_369_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2(lean_object* v_msgData_373_, lean_object* v___y_374_, lean_object* v___y_375_){
_start:
{
lean_object* v___x_377_; lean_object* v_env_378_; lean_object* v_options_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_377_ = lean_st_ref_get(v___y_375_);
v_env_378_ = lean_ctor_get(v___x_377_, 0);
lean_inc_ref(v_env_378_);
lean_dec(v___x_377_);
v_options_379_ = lean_ctor_get(v___y_374_, 2);
v___x_380_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2);
v___x_381_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5);
lean_inc_ref(v_options_379_);
v___x_382_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_382_, 0, v_env_378_);
lean_ctor_set(v___x_382_, 1, v___x_380_);
lean_ctor_set(v___x_382_, 2, v___x_381_);
lean_ctor_set(v___x_382_, 3, v_options_379_);
v___x_383_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_383_, 0, v___x_382_);
lean_ctor_set(v___x_383_, 1, v_msgData_373_);
v___x_384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_384_, 0, v___x_383_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___boxed(lean_object* v_msgData_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2(v_msgData_385_, v___y_386_, v___y_387_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg(lean_object* v_msg_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v_ref_394_; lean_object* v___x_395_; lean_object* v_a_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_404_; 
v_ref_394_ = lean_ctor_get(v___y_391_, 5);
v___x_395_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2(v_msg_390_, v___y_391_, v___y_392_);
v_a_396_ = lean_ctor_get(v___x_395_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_395_);
if (v_isSharedCheck_404_ == 0)
{
v___x_398_ = v___x_395_;
v_isShared_399_ = v_isSharedCheck_404_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_a_396_);
lean_dec(v___x_395_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_404_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
lean_object* v___x_400_; lean_object* v___x_402_; 
lean_inc(v_ref_394_);
v___x_400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_400_, 0, v_ref_394_);
lean_ctor_set(v___x_400_, 1, v_a_396_);
if (v_isShared_399_ == 0)
{
lean_ctor_set_tag(v___x_398_, 1);
lean_ctor_set(v___x_398_, 0, v___x_400_);
v___x_402_ = v___x_398_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v___x_400_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg___boxed(lean_object* v_msg_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg(v_msg_405_, v___y_406_, v___y_407_);
lean_dec(v___y_407_);
lean_dec_ref(v___y_406_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__1(lean_object* v_a_410_, lean_object* v_a_411_){
_start:
{
if (lean_obj_tag(v_a_410_) == 0)
{
lean_object* v___x_412_; 
v___x_412_ = l_List_reverse___redArg(v_a_411_);
return v___x_412_;
}
else
{
lean_object* v_head_413_; lean_object* v_tail_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_423_; 
v_head_413_ = lean_ctor_get(v_a_410_, 0);
v_tail_414_ = lean_ctor_get(v_a_410_, 1);
v_isSharedCheck_423_ = !lean_is_exclusive(v_a_410_);
if (v_isSharedCheck_423_ == 0)
{
v___x_416_ = v_a_410_;
v_isShared_417_ = v_isSharedCheck_423_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_tail_414_);
lean_inc(v_head_413_);
lean_dec(v_a_410_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_423_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_418_; lean_object* v___x_420_; 
v___x_418_ = l_Lean_MessageData_ofFormat(v_head_413_);
if (v_isShared_417_ == 0)
{
lean_ctor_set(v___x_416_, 1, v_a_411_);
lean_ctor_set(v___x_416_, 0, v___x_418_);
v___x_420_ = v___x_416_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v___x_418_);
lean_ctor_set(v_reuseFailAlloc_422_, 1, v_a_411_);
v___x_420_ = v_reuseFailAlloc_422_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
v_a_410_ = v_tail_414_;
v_a_411_ = v___x_420_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__0(size_t v_sz_424_, size_t v_i_425_, lean_object* v_bs_426_){
_start:
{
uint8_t v___x_427_; 
v___x_427_ = lean_usize_dec_lt(v_i_425_, v_sz_424_);
if (v___x_427_ == 0)
{
return v_bs_426_;
}
else
{
lean_object* v_v_428_; lean_object* v___x_429_; lean_object* v_bs_x27_430_; lean_object* v___x_431_; size_t v___x_432_; size_t v___x_433_; lean_object* v___x_434_; 
v_v_428_ = lean_array_uget(v_bs_426_, v_i_425_);
v___x_429_ = lean_unsigned_to_nat(0u);
v_bs_x27_430_ = lean_array_uset(v_bs_426_, v_i_425_, v___x_429_);
v___x_431_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_v_428_);
v___x_432_ = ((size_t)1ULL);
v___x_433_ = lean_usize_add(v_i_425_, v___x_432_);
v___x_434_ = lean_array_uset(v_bs_x27_430_, v_i_425_, v___x_431_);
v_i_425_ = v___x_433_;
v_bs_426_ = v___x_434_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__0___boxed(lean_object* v_sz_436_, lean_object* v_i_437_, lean_object* v_bs_438_){
_start:
{
size_t v_sz_boxed_439_; size_t v_i_boxed_440_; lean_object* v_res_441_; 
v_sz_boxed_439_ = lean_unbox_usize(v_sz_436_);
lean_dec(v_sz_436_);
v_i_boxed_440_ = lean_unbox_usize(v_i_437_);
lean_dec(v_i_437_);
v_res_441_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__0(v_sz_boxed_439_, v_i_boxed_440_, v_bs_438_);
return v_res_441_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1(void){
_start:
{
lean_object* v___x_443_; lean_object* v___x_444_; 
v___x_443_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__0));
v___x_444_ = l_Lean_stringToMessageData(v___x_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next(lean_object* v_keys_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lean_st_ref_get(v_a_446_);
if (lean_obj_tag(v___x_450_) == 1)
{
lean_object* v_head_451_; lean_object* v_tail_452_; lean_object* v___x_453_; lean_object* v___x_454_; 
lean_dec_ref(v_keys_445_);
v_head_451_ = lean_ctor_get(v___x_450_, 0);
lean_inc(v_head_451_);
v_tail_452_ = lean_ctor_get(v___x_450_, 1);
lean_inc(v_tail_452_);
lean_dec_ref_known(v___x_450_, 2);
v___x_453_ = lean_st_ref_set(v_a_446_, v_tail_452_);
v___x_454_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_454_, 0, v_head_451_);
return v___x_454_;
}
else
{
lean_object* v___x_455_; size_t v_sz_456_; size_t v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; 
lean_dec(v___x_450_);
v___x_455_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1, &lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1);
v_sz_456_ = lean_array_size(v_keys_445_);
v___x_457_ = ((size_t)0ULL);
v___x_458_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__0(v_sz_456_, v___x_457_, v_keys_445_);
v___x_459_ = lean_array_to_list(v___x_458_);
v___x_460_ = lean_box(0);
v___x_461_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__1(v___x_459_, v___x_460_);
v___x_462_ = l_Lean_MessageData_ofList(v___x_461_);
v___x_463_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_463_, 0, v___x_455_);
lean_ctor_set(v___x_463_, 1, v___x_462_);
v___x_464_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg(v___x_463_, v_a_447_, v_a_448_);
return v___x_464_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___boxed(lean_object* v_keys_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_, lean_object* v_a_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next(v_keys_465_, v_a_466_, v_a_467_, v_a_468_);
lean_dec(v_a_468_);
lean_dec_ref(v_a_467_);
lean_dec(v_a_466_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2(lean_object* v_00_u03b1_471_, lean_object* v_msg_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
lean_object* v___x_477_; 
v___x_477_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg(v_msg_472_, v___y_474_, v___y_475_);
return v___x_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___boxed(lean_object* v_00_u03b1_478_, lean_object* v_msg_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2(v_00_u03b1_478_, v_msg_479_, v___y_480_, v___y_481_, v___y_482_);
lean_dec(v___y_482_);
lean_dec_ref(v___y_481_);
lean_dec(v___y_480_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_parenthesize(lean_object* v_msg_485_, uint8_t v_paren_486_){
_start:
{
if (v_paren_486_ == 0)
{
lean_object* v___x_487_; 
v___x_487_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v___x_487_, 0, v_msg_485_);
return v___x_487_;
}
else
{
lean_object* v___x_488_; 
v___x_488_ = l_Lean_MessageData_paren(v_msg_485_);
return v___x_488_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_parenthesize___boxed(lean_object* v_msg_489_, lean_object* v_paren_490_){
_start:
{
uint8_t v_paren_boxed_491_; lean_object* v_res_492_; 
v_paren_boxed_491_ = lean_unbox(v_paren_490_);
v_res_492_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_parenthesize(v_msg_489_, v_paren_boxed_491_);
return v_res_492_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1(void){
_start:
{
lean_object* v___x_494_; lean_object* v___x_495_; 
v___x_494_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__0));
v___x_495_ = l_Lean_stringToMessageData(v___x_494_);
return v___x_495_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__3(void){
_start:
{
lean_object* v___x_497_; lean_object* v___x_498_; 
v___x_497_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__2));
v___x_498_ = l_Lean_stringToMessageData(v___x_497_);
return v___x_498_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__5(void){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_500_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__4));
v___x_501_ = l_Lean_stringToMessageData(v___x_500_);
return v___x_501_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__7(void){
_start:
{
lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_503_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__6));
v___x_504_ = l_Lean_stringToMessageData(v___x_503_);
return v___x_504_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__9(void){
_start:
{
lean_object* v___x_506_; lean_object* v___x_507_; 
v___x_506_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__8));
v___x_507_ = l_Lean_stringToMessageData(v___x_506_);
return v___x_507_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__11(void){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_509_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__10));
v___x_510_ = l_Lean_stringToMessageData(v___x_509_);
return v___x_510_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__13(void){
_start:
{
lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_512_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__12));
v___x_513_ = l_Lean_stringToMessageData(v___x_512_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg(lean_object* v_msg_514_, lean_object* v_declHint_515_, lean_object* v___y_516_){
_start:
{
lean_object* v___x_518_; lean_object* v_env_519_; uint8_t v___x_520_; 
v___x_518_ = lean_st_ref_get(v___y_516_);
v_env_519_ = lean_ctor_get(v___x_518_, 0);
lean_inc_ref(v_env_519_);
lean_dec(v___x_518_);
v___x_520_ = l_Lean_Name_isAnonymous(v_declHint_515_);
if (v___x_520_ == 0)
{
uint8_t v_isExporting_521_; 
v_isExporting_521_ = lean_ctor_get_uint8(v_env_519_, sizeof(void*)*8);
if (v_isExporting_521_ == 0)
{
lean_object* v___x_522_; 
lean_dec_ref(v_env_519_);
lean_dec(v_declHint_515_);
v___x_522_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_522_, 0, v_msg_514_);
return v___x_522_;
}
else
{
lean_object* v___x_523_; uint8_t v___x_524_; 
lean_inc_ref(v_env_519_);
v___x_523_ = l_Lean_Environment_setExporting(v_env_519_, v___x_520_);
lean_inc(v_declHint_515_);
lean_inc_ref(v___x_523_);
v___x_524_ = l_Lean_Environment_contains(v___x_523_, v_declHint_515_, v_isExporting_521_);
if (v___x_524_ == 0)
{
lean_object* v___x_525_; 
lean_dec_ref(v___x_523_);
lean_dec_ref(v_env_519_);
lean_dec(v_declHint_515_);
v___x_525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_525_, 0, v_msg_514_);
return v___x_525_;
}
else
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v_c_531_; lean_object* v___x_532_; 
v___x_526_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2);
v___x_527_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__5);
v___x_528_ = l_Lean_Options_empty;
v___x_529_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_529_, 0, v___x_523_);
lean_ctor_set(v___x_529_, 1, v___x_526_);
lean_ctor_set(v___x_529_, 2, v___x_527_);
lean_ctor_set(v___x_529_, 3, v___x_528_);
lean_inc(v_declHint_515_);
v___x_530_ = l_Lean_MessageData_ofConstName(v_declHint_515_, v___x_520_);
v_c_531_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_531_, 0, v___x_529_);
lean_ctor_set(v_c_531_, 1, v___x_530_);
v___x_532_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_519_, v_declHint_515_);
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
lean_dec_ref(v_env_519_);
lean_dec(v_declHint_515_);
v___x_533_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1);
v___x_534_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_533_);
lean_ctor_set(v___x_534_, 1, v_c_531_);
v___x_535_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__3);
v___x_536_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_536_, 0, v___x_534_);
lean_ctor_set(v___x_536_, 1, v___x_535_);
v___x_537_ = l_Lean_MessageData_note(v___x_536_);
v___x_538_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_538_, 0, v_msg_514_);
lean_ctor_set(v___x_538_, 1, v___x_537_);
v___x_539_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_539_, 0, v___x_538_);
return v___x_539_;
}
else
{
lean_object* v_val_540_; lean_object* v___x_542_; uint8_t v_isShared_543_; uint8_t v_isSharedCheck_575_; 
v_val_540_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_575_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_575_ == 0)
{
v___x_542_ = v___x_532_;
v_isShared_543_ = v_isSharedCheck_575_;
goto v_resetjp_541_;
}
else
{
lean_inc(v_val_540_);
lean_dec(v___x_532_);
v___x_542_ = lean_box(0);
v_isShared_543_ = v_isSharedCheck_575_;
goto v_resetjp_541_;
}
v_resetjp_541_:
{
lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v_mod_547_; uint8_t v___x_548_; 
v___x_544_ = lean_box(0);
v___x_545_ = l_Lean_Environment_header(v_env_519_);
lean_dec_ref(v_env_519_);
v___x_546_ = l_Lean_EnvironmentHeader_moduleNames(v___x_545_);
v_mod_547_ = lean_array_get(v___x_544_, v___x_546_, v_val_540_);
lean_dec(v_val_540_);
lean_dec_ref(v___x_546_);
v___x_548_ = l_Lean_isPrivateName(v_declHint_515_);
lean_dec(v_declHint_515_);
if (v___x_548_ == 0)
{
lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_560_; 
v___x_549_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__5);
v___x_550_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_550_, 0, v___x_549_);
lean_ctor_set(v___x_550_, 1, v_c_531_);
v___x_551_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__7);
v___x_552_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_552_, 0, v___x_550_);
lean_ctor_set(v___x_552_, 1, v___x_551_);
v___x_553_ = l_Lean_MessageData_ofName(v_mod_547_);
v___x_554_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_554_, 0, v___x_552_);
lean_ctor_set(v___x_554_, 1, v___x_553_);
v___x_555_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__9);
v___x_556_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_556_, 0, v___x_554_);
lean_ctor_set(v___x_556_, 1, v___x_555_);
v___x_557_ = l_Lean_MessageData_note(v___x_556_);
v___x_558_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_558_, 0, v_msg_514_);
lean_ctor_set(v___x_558_, 1, v___x_557_);
if (v_isShared_543_ == 0)
{
lean_ctor_set_tag(v___x_542_, 0);
lean_ctor_set(v___x_542_, 0, v___x_558_);
v___x_560_ = v___x_542_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v___x_558_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
else
{
lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_573_; 
v___x_562_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__1);
v___x_563_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_563_, 0, v___x_562_);
lean_ctor_set(v___x_563_, 1, v_c_531_);
v___x_564_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__11);
v___x_565_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_565_, 0, v___x_563_);
lean_ctor_set(v___x_565_, 1, v___x_564_);
v___x_566_ = l_Lean_MessageData_ofName(v_mod_547_);
v___x_567_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_567_, 0, v___x_565_);
lean_ctor_set(v___x_567_, 1, v___x_566_);
v___x_568_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___closed__13);
v___x_569_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_569_, 0, v___x_567_);
lean_ctor_set(v___x_569_, 1, v___x_568_);
v___x_570_ = l_Lean_MessageData_note(v___x_569_);
v___x_571_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_571_, 0, v_msg_514_);
lean_ctor_set(v___x_571_, 1, v___x_570_);
if (v_isShared_543_ == 0)
{
lean_ctor_set_tag(v___x_542_, 0);
lean_ctor_set(v___x_542_, 0, v___x_571_);
v___x_573_ = v___x_542_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v___x_571_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
return v___x_573_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_576_; 
lean_dec_ref(v_env_519_);
lean_dec(v_declHint_515_);
v___x_576_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_576_, 0, v_msg_514_);
return v___x_576_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg___boxed(lean_object* v_msg_577_, lean_object* v_declHint_578_, lean_object* v___y_579_, lean_object* v___y_580_){
_start:
{
lean_object* v_res_581_; 
v_res_581_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg(v_msg_577_, v_declHint_578_, v___y_579_);
lean_dec(v___y_579_);
return v_res_581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7(lean_object* v_msg_582_, lean_object* v_declHint_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_){
_start:
{
lean_object* v___x_588_; lean_object* v_a_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_598_; 
v___x_588_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg(v_msg_582_, v_declHint_583_, v___y_586_);
v_a_589_ = lean_ctor_get(v___x_588_, 0);
v_isSharedCheck_598_ = !lean_is_exclusive(v___x_588_);
if (v_isSharedCheck_598_ == 0)
{
v___x_591_ = v___x_588_;
v_isShared_592_ = v_isSharedCheck_598_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_a_589_);
lean_dec(v___x_588_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_598_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_596_; 
v___x_593_ = l_Lean_unknownIdentifierMessageTag;
v___x_594_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_594_, 0, v___x_593_);
lean_ctor_set(v___x_594_, 1, v_a_589_);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_594_);
v___x_596_ = v___x_591_;
goto v_reusejp_595_;
}
else
{
lean_object* v_reuseFailAlloc_597_; 
v_reuseFailAlloc_597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_597_, 0, v___x_594_);
v___x_596_ = v_reuseFailAlloc_597_;
goto v_reusejp_595_;
}
v_reusejp_595_:
{
return v___x_596_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7___boxed(lean_object* v_msg_599_, lean_object* v_declHint_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7(v_msg_599_, v_declHint_600_, v___y_601_, v___y_602_, v___y_603_);
lean_dec(v___y_603_);
lean_dec_ref(v___y_602_);
lean_dec(v___y_601_);
return v_res_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___redArg(lean_object* v_ref_606_, lean_object* v_msg_607_, lean_object* v___y_608_, lean_object* v___y_609_){
_start:
{
lean_object* v_fileName_611_; lean_object* v_fileMap_612_; lean_object* v_options_613_; lean_object* v_currRecDepth_614_; lean_object* v_maxRecDepth_615_; lean_object* v_ref_616_; lean_object* v_currNamespace_617_; lean_object* v_openDecls_618_; lean_object* v_initHeartbeats_619_; lean_object* v_maxHeartbeats_620_; lean_object* v_quotContext_621_; lean_object* v_currMacroScope_622_; uint8_t v_diag_623_; lean_object* v_cancelTk_x3f_624_; uint8_t v_suppressElabErrors_625_; lean_object* v_inheritedTraceOptions_626_; lean_object* v_ref_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v_fileName_611_ = lean_ctor_get(v___y_608_, 0);
v_fileMap_612_ = lean_ctor_get(v___y_608_, 1);
v_options_613_ = lean_ctor_get(v___y_608_, 2);
v_currRecDepth_614_ = lean_ctor_get(v___y_608_, 3);
v_maxRecDepth_615_ = lean_ctor_get(v___y_608_, 4);
v_ref_616_ = lean_ctor_get(v___y_608_, 5);
v_currNamespace_617_ = lean_ctor_get(v___y_608_, 6);
v_openDecls_618_ = lean_ctor_get(v___y_608_, 7);
v_initHeartbeats_619_ = lean_ctor_get(v___y_608_, 8);
v_maxHeartbeats_620_ = lean_ctor_get(v___y_608_, 9);
v_quotContext_621_ = lean_ctor_get(v___y_608_, 10);
v_currMacroScope_622_ = lean_ctor_get(v___y_608_, 11);
v_diag_623_ = lean_ctor_get_uint8(v___y_608_, sizeof(void*)*14);
v_cancelTk_x3f_624_ = lean_ctor_get(v___y_608_, 12);
v_suppressElabErrors_625_ = lean_ctor_get_uint8(v___y_608_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_626_ = lean_ctor_get(v___y_608_, 13);
v_ref_627_ = l_Lean_replaceRef(v_ref_606_, v_ref_616_);
lean_inc_ref(v_inheritedTraceOptions_626_);
lean_inc(v_cancelTk_x3f_624_);
lean_inc(v_currMacroScope_622_);
lean_inc(v_quotContext_621_);
lean_inc(v_maxHeartbeats_620_);
lean_inc(v_initHeartbeats_619_);
lean_inc(v_openDecls_618_);
lean_inc(v_currNamespace_617_);
lean_inc(v_maxRecDepth_615_);
lean_inc(v_currRecDepth_614_);
lean_inc_ref(v_options_613_);
lean_inc_ref(v_fileMap_612_);
lean_inc_ref(v_fileName_611_);
v___x_628_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_628_, 0, v_fileName_611_);
lean_ctor_set(v___x_628_, 1, v_fileMap_612_);
lean_ctor_set(v___x_628_, 2, v_options_613_);
lean_ctor_set(v___x_628_, 3, v_currRecDepth_614_);
lean_ctor_set(v___x_628_, 4, v_maxRecDepth_615_);
lean_ctor_set(v___x_628_, 5, v_ref_627_);
lean_ctor_set(v___x_628_, 6, v_currNamespace_617_);
lean_ctor_set(v___x_628_, 7, v_openDecls_618_);
lean_ctor_set(v___x_628_, 8, v_initHeartbeats_619_);
lean_ctor_set(v___x_628_, 9, v_maxHeartbeats_620_);
lean_ctor_set(v___x_628_, 10, v_quotContext_621_);
lean_ctor_set(v___x_628_, 11, v_currMacroScope_622_);
lean_ctor_set(v___x_628_, 12, v_cancelTk_x3f_624_);
lean_ctor_set(v___x_628_, 13, v_inheritedTraceOptions_626_);
lean_ctor_set_uint8(v___x_628_, sizeof(void*)*14, v_diag_623_);
lean_ctor_set_uint8(v___x_628_, sizeof(void*)*14 + 1, v_suppressElabErrors_625_);
v___x_629_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2___redArg(v_msg_607_, v___x_628_, v___y_609_);
lean_dec_ref_known(v___x_628_, 14);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___redArg___boxed(lean_object* v_ref_630_, lean_object* v_msg_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___redArg(v_ref_630_, v_msg_631_, v___y_632_, v___y_633_);
lean_dec(v___y_633_);
lean_dec_ref(v___y_632_);
lean_dec(v_ref_630_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___redArg(lean_object* v_ref_636_, lean_object* v_msg_637_, lean_object* v_declHint_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_){
_start:
{
lean_object* v___x_643_; lean_object* v_a_644_; lean_object* v___x_645_; 
v___x_643_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7(v_msg_637_, v_declHint_638_, v___y_639_, v___y_640_, v___y_641_);
v_a_644_ = lean_ctor_get(v___x_643_, 0);
lean_inc(v_a_644_);
lean_dec_ref(v___x_643_);
v___x_645_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___redArg(v_ref_636_, v_a_644_, v___y_640_, v___y_641_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___redArg___boxed(lean_object* v_ref_646_, lean_object* v_msg_647_, lean_object* v_declHint_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___redArg(v_ref_646_, v_msg_647_, v_declHint_648_, v___y_649_, v___y_650_, v___y_651_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec(v_ref_646_);
return v_res_653_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__1(void){
_start:
{
lean_object* v___x_655_; lean_object* v___x_656_; 
v___x_655_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__0));
v___x_656_ = l_Lean_stringToMessageData(v___x_655_);
return v___x_656_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__3(void){
_start:
{
lean_object* v___x_658_; lean_object* v___x_659_; 
v___x_658_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__2));
v___x_659_ = l_Lean_stringToMessageData(v___x_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg(lean_object* v_ref_660_, lean_object* v_constName_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
lean_object* v___x_666_; uint8_t v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v___x_666_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__1);
v___x_667_ = 0;
lean_inc(v_constName_661_);
v___x_668_ = l_Lean_MessageData_ofConstName(v_constName_661_, v___x_667_);
v___x_669_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_669_, 0, v___x_666_);
lean_ctor_set(v___x_669_, 1, v___x_668_);
v___x_670_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___closed__3);
v___x_671_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_671_, 0, v___x_669_);
lean_ctor_set(v___x_671_, 1, v___x_670_);
v___x_672_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___redArg(v_ref_660_, v___x_671_, v_constName_661_, v___y_662_, v___y_663_, v___y_664_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_ref_673_, lean_object* v_constName_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg(v_ref_673_, v_constName_674_, v___y_675_, v___y_676_, v___y_677_);
lean_dec(v___y_677_);
lean_dec_ref(v___y_676_);
lean_dec(v___y_675_);
lean_dec(v_ref_673_);
return v_res_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___redArg(lean_object* v_constName_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_){
_start:
{
lean_object* v_ref_685_; lean_object* v___x_686_; 
v_ref_685_ = lean_ctor_get(v___y_682_, 5);
v___x_686_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg(v_ref_685_, v_constName_680_, v___y_681_, v___y_682_, v___y_683_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___redArg___boxed(lean_object* v_constName_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_){
_start:
{
lean_object* v_res_692_; 
v_res_692_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___redArg(v_constName_687_, v___y_688_, v___y_689_, v___y_690_);
lean_dec(v___y_690_);
lean_dec_ref(v___y_689_);
lean_dec(v___y_688_);
return v_res_692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2(lean_object* v_constName_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_){
_start:
{
lean_object* v___x_698_; lean_object* v_env_699_; uint8_t v___x_700_; lean_object* v___x_701_; 
v___x_698_ = lean_st_ref_get(v___y_696_);
v_env_699_ = lean_ctor_get(v___x_698_, 0);
lean_inc_ref(v_env_699_);
lean_dec(v___x_698_);
v___x_700_ = 0;
lean_inc(v_constName_693_);
v___x_701_ = l_Lean_Environment_findConstVal_x3f(v_env_699_, v_constName_693_, v___x_700_);
if (lean_obj_tag(v___x_701_) == 0)
{
lean_object* v___x_702_; 
v___x_702_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___redArg(v_constName_693_, v___y_694_, v___y_695_, v___y_696_);
return v___x_702_;
}
else
{
lean_object* v_val_703_; lean_object* v___x_705_; uint8_t v_isShared_706_; uint8_t v_isSharedCheck_710_; 
lean_dec(v_constName_693_);
v_val_703_ = lean_ctor_get(v___x_701_, 0);
v_isSharedCheck_710_ = !lean_is_exclusive(v___x_701_);
if (v_isSharedCheck_710_ == 0)
{
v___x_705_ = v___x_701_;
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
else
{
lean_inc(v_val_703_);
lean_dec(v___x_701_);
v___x_705_ = lean_box(0);
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
v_resetjp_704_:
{
lean_object* v___x_708_; 
if (v_isShared_706_ == 0)
{
lean_ctor_set_tag(v___x_705_, 0);
v___x_708_ = v___x_705_;
goto v_reusejp_707_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_val_703_);
v___x_708_ = v_reuseFailAlloc_709_;
goto v_reusejp_707_;
}
v_reusejp_707_:
{
return v___x_708_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2___boxed(lean_object* v_constName_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2(v_constName_711_, v___y_712_, v___y_713_, v___y_714_);
lean_dec(v___y_714_);
lean_dec_ref(v___y_713_);
lean_dec(v___y_712_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__3(lean_object* v_a_717_, lean_object* v_a_718_){
_start:
{
if (lean_obj_tag(v_a_717_) == 0)
{
lean_object* v___x_719_; 
v___x_719_ = l_List_reverse___redArg(v_a_718_);
return v___x_719_;
}
else
{
lean_object* v_head_720_; lean_object* v_tail_721_; lean_object* v___x_723_; uint8_t v_isShared_724_; uint8_t v_isSharedCheck_730_; 
v_head_720_ = lean_ctor_get(v_a_717_, 0);
v_tail_721_ = lean_ctor_get(v_a_717_, 1);
v_isSharedCheck_730_ = !lean_is_exclusive(v_a_717_);
if (v_isSharedCheck_730_ == 0)
{
v___x_723_ = v_a_717_;
v_isShared_724_ = v_isSharedCheck_730_;
goto v_resetjp_722_;
}
else
{
lean_inc(v_tail_721_);
lean_inc(v_head_720_);
lean_dec(v_a_717_);
v___x_723_ = lean_box(0);
v_isShared_724_ = v_isSharedCheck_730_;
goto v_resetjp_722_;
}
v_resetjp_722_:
{
lean_object* v___x_725_; lean_object* v___x_727_; 
v___x_725_ = l_Lean_mkLevelParam(v_head_720_);
if (v_isShared_724_ == 0)
{
lean_ctor_set(v___x_723_, 1, v_a_718_);
lean_ctor_set(v___x_723_, 0, v___x_725_);
v___x_727_ = v___x_723_;
goto v_reusejp_726_;
}
else
{
lean_object* v_reuseFailAlloc_729_; 
v_reuseFailAlloc_729_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_729_, 0, v___x_725_);
lean_ctor_set(v_reuseFailAlloc_729_, 1, v_a_718_);
v___x_727_ = v_reuseFailAlloc_729_;
goto v_reusejp_726_;
}
v_reusejp_726_:
{
v_a_717_ = v_tail_721_;
v_a_718_ = v___x_727_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2(lean_object* v_constName_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_){
_start:
{
lean_object* v___x_736_; 
lean_inc(v_constName_731_);
v___x_736_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2(v_constName_731_, v___y_732_, v___y_733_, v___y_734_);
if (lean_obj_tag(v___x_736_) == 0)
{
lean_object* v_a_737_; lean_object* v___x_739_; uint8_t v_isShared_740_; uint8_t v_isSharedCheck_748_; 
v_a_737_ = lean_ctor_get(v___x_736_, 0);
v_isSharedCheck_748_ = !lean_is_exclusive(v___x_736_);
if (v_isSharedCheck_748_ == 0)
{
v___x_739_ = v___x_736_;
v_isShared_740_ = v_isSharedCheck_748_;
goto v_resetjp_738_;
}
else
{
lean_inc(v_a_737_);
lean_dec(v___x_736_);
v___x_739_ = lean_box(0);
v_isShared_740_ = v_isSharedCheck_748_;
goto v_resetjp_738_;
}
v_resetjp_738_:
{
lean_object* v_levelParams_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_746_; 
v_levelParams_741_ = lean_ctor_get(v_a_737_, 1);
lean_inc(v_levelParams_741_);
lean_dec(v_a_737_);
v___x_742_ = lean_box(0);
v___x_743_ = lp_mathlib_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__3(v_levelParams_741_, v___x_742_);
v___x_744_ = l_Lean_mkConst(v_constName_731_, v___x_743_);
if (v_isShared_740_ == 0)
{
lean_ctor_set(v___x_739_, 0, v___x_744_);
v___x_746_ = v___x_739_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v___x_744_);
v___x_746_ = v_reuseFailAlloc_747_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
return v___x_746_;
}
}
}
else
{
lean_object* v_a_749_; lean_object* v___x_751_; uint8_t v_isShared_752_; uint8_t v_isSharedCheck_756_; 
lean_dec(v_constName_731_);
v_a_749_ = lean_ctor_get(v___x_736_, 0);
v_isSharedCheck_756_ = !lean_is_exclusive(v___x_736_);
if (v_isSharedCheck_756_ == 0)
{
v___x_751_ = v___x_736_;
v_isShared_752_ = v_isSharedCheck_756_;
goto v_resetjp_750_;
}
else
{
lean_inc(v_a_749_);
lean_dec(v___x_736_);
v___x_751_ = lean_box(0);
v_isShared_752_ = v_isSharedCheck_756_;
goto v_resetjp_750_;
}
v_resetjp_750_:
{
lean_object* v___x_754_; 
if (v_isShared_752_ == 0)
{
v___x_754_ = v___x_751_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v_a_749_);
v___x_754_ = v_reuseFailAlloc_755_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
return v___x_754_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2___boxed(lean_object* v_constName_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_){
_start:
{
lean_object* v_res_762_; 
v_res_762_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2(v_constName_757_, v___y_758_, v___y_759_, v___y_760_);
lean_dec(v___y_760_);
lean_dec_ref(v___y_759_);
lean_dec(v___y_758_);
return v_res_762_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_763_ = lean_box(1);
v___x_764_ = l_Lean_MessageData_ofFormat(v___x_763_);
return v___x_764_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__1(void){
_start:
{
lean_object* v___x_766_; lean_object* v_r_767_; 
v___x_766_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__0));
v_r_767_ = l_Lean_stringToMessageData(v___x_766_);
return v_r_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp(lean_object* v_keys_768_, lean_object* v_f_769_, lean_object* v_nargs_770_, uint8_t v_paren_771_, lean_object* v_a_772_, lean_object* v_a_773_, lean_object* v_a_774_){
_start:
{
lean_object* v___x_776_; uint8_t v___x_777_; 
v___x_776_ = lean_unsigned_to_nat(0u);
v___x_777_ = lean_nat_dec_eq(v_nargs_770_, v___x_776_);
if (v___x_777_ == 0)
{
lean_object* v_r_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; 
v_r_778_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__1, &lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__1_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___closed__1);
v___x_779_ = lean_unsigned_to_nat(1u);
v___x_780_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_780_, 0, v___x_776_);
lean_ctor_set(v___x_780_, 1, v_nargs_770_);
lean_ctor_set(v___x_780_, 2, v___x_779_);
v___x_781_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg(v_keys_768_, v___x_780_, v_r_778_, v___x_776_, v_a_772_, v_a_773_, v_a_774_);
lean_dec_ref_known(v___x_780_, 3);
if (lean_obj_tag(v___x_781_) == 0)
{
lean_object* v_a_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_797_; 
v_a_782_ = lean_ctor_get(v___x_781_, 0);
v_isSharedCheck_797_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_797_ == 0)
{
v___x_784_ = v___x_781_;
v_isShared_785_ = v_isSharedCheck_797_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_a_782_);
lean_dec(v___x_781_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_797_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; 
v___x_786_ = lean_unsigned_to_nat(2u);
v___x_787_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_787_, 0, v___x_786_);
lean_ctor_set(v___x_787_, 1, v_a_782_);
v___x_788_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_788_, 0, v_f_769_);
lean_ctor_set(v___x_788_, 1, v___x_787_);
if (v_paren_771_ == 0)
{
lean_object* v___x_789_; lean_object* v___x_791_; 
v___x_789_ = lean_alloc_ctor(6, 1, 0);
lean_ctor_set(v___x_789_, 0, v___x_788_);
if (v_isShared_785_ == 0)
{
lean_ctor_set(v___x_784_, 0, v___x_789_);
v___x_791_ = v___x_784_;
goto v_reusejp_790_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v___x_789_);
v___x_791_ = v_reuseFailAlloc_792_;
goto v_reusejp_790_;
}
v_reusejp_790_:
{
return v___x_791_;
}
}
else
{
lean_object* v___x_793_; lean_object* v___x_795_; 
v___x_793_ = l_Lean_MessageData_paren(v___x_788_);
if (v_isShared_785_ == 0)
{
lean_ctor_set(v___x_784_, 0, v___x_793_);
v___x_795_ = v___x_784_;
goto v_reusejp_794_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v___x_793_);
v___x_795_ = v_reuseFailAlloc_796_;
goto v_reusejp_794_;
}
v_reusejp_794_:
{
return v___x_795_;
}
}
}
}
else
{
lean_dec_ref(v_f_769_);
return v___x_781_;
}
}
else
{
lean_object* v___x_798_; 
lean_dec(v_nargs_770_);
lean_dec_ref(v_keys_768_);
v___x_798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_798_, 0, v_f_769_);
return v___x_798_;
}
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__0(void){
_start:
{
lean_object* v___x_799_; lean_object* v___x_800_; 
v___x_799_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__18));
v___x_800_ = l_Lean_stringToMessageData(v___x_799_);
return v___x_800_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__2(void){
_start:
{
lean_object* v___x_802_; lean_object* v___x_803_; 
v___x_802_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__1));
v___x_803_ = l_Lean_stringToMessageData(v___x_802_);
return v___x_803_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__4(void){
_start:
{
lean_object* v___x_805_; lean_object* v___x_806_; 
v___x_805_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__3));
v___x_806_ = l_Lean_stringToMessageData(v___x_805_);
return v___x_806_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__6(void){
_start:
{
lean_object* v___x_808_; lean_object* v___x_809_; 
v___x_808_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__5));
v___x_809_ = l_Lean_stringToMessageData(v___x_808_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(lean_object* v_keys_810_, uint8_t v_paren_811_, lean_object* v_a_812_, lean_object* v_a_813_, lean_object* v_a_814_){
_start:
{
lean_object* v___x_816_; 
lean_inc_ref(v_keys_810_);
v___x_816_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next(v_keys_810_, v_a_812_, v_a_813_, v_a_814_);
if (lean_obj_tag(v___x_816_) == 0)
{
lean_object* v_a_817_; lean_object* v___x_819_; uint8_t v_isShared_820_; uint8_t v_isSharedCheck_909_; 
v_a_817_ = lean_ctor_get(v___x_816_, 0);
v_isSharedCheck_909_ = !lean_is_exclusive(v___x_816_);
if (v_isSharedCheck_909_ == 0)
{
v___x_819_ = v___x_816_;
v_isShared_820_ = v_isSharedCheck_909_;
goto v_resetjp_818_;
}
else
{
lean_inc(v_a_817_);
lean_dec(v___x_816_);
v___x_819_ = lean_box(0);
v_isShared_820_ = v_isSharedCheck_909_;
goto v_resetjp_818_;
}
v_resetjp_818_:
{
switch(lean_obj_tag(v_a_817_))
{
case 3:
{
lean_object* v_declName_821_; lean_object* v_nargs_822_; lean_object* v___x_823_; 
lean_del_object(v___x_819_);
v_declName_821_ = lean_ctor_get(v_a_817_, 0);
lean_inc(v_declName_821_);
v_nargs_822_ = lean_ctor_get(v_a_817_, 1);
lean_inc(v_nargs_822_);
lean_dec_ref_known(v_a_817_, 2);
v___x_823_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2(v_declName_821_, v_a_812_, v_a_813_, v_a_814_);
if (lean_obj_tag(v___x_823_) == 0)
{
lean_object* v_a_824_; lean_object* v___x_825_; lean_object* v___x_826_; 
v_a_824_ = lean_ctor_get(v___x_823_, 0);
lean_inc(v_a_824_);
lean_dec_ref_known(v___x_823_, 1);
v___x_825_ = l_Lean_MessageData_ofExpr(v_a_824_);
v___x_826_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp(v_keys_810_, v___x_825_, v_nargs_822_, v_paren_811_, v_a_812_, v_a_813_, v_a_814_);
return v___x_826_;
}
else
{
lean_object* v_a_827_; lean_object* v___x_829_; uint8_t v_isShared_830_; uint8_t v_isSharedCheck_834_; 
lean_dec(v_nargs_822_);
lean_dec_ref(v_keys_810_);
v_a_827_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_834_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_834_ == 0)
{
v___x_829_ = v___x_823_;
v_isShared_830_ = v_isSharedCheck_834_;
goto v_resetjp_828_;
}
else
{
lean_inc(v_a_827_);
lean_dec(v___x_823_);
v___x_829_ = lean_box(0);
v_isShared_830_ = v_isSharedCheck_834_;
goto v_resetjp_828_;
}
v_resetjp_828_:
{
lean_object* v___x_832_; 
if (v_isShared_830_ == 0)
{
v___x_832_ = v___x_829_;
goto v_reusejp_831_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v_a_827_);
v___x_832_ = v_reuseFailAlloc_833_;
goto v_reusejp_831_;
}
v_reusejp_831_:
{
return v___x_832_;
}
}
}
}
case 4:
{
lean_object* v_fvarId_835_; lean_object* v_nargs_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; 
lean_del_object(v___x_819_);
v_fvarId_835_ = lean_ctor_get(v_a_817_, 0);
lean_inc(v_fvarId_835_);
v_nargs_836_ = lean_ctor_get(v_a_817_, 1);
lean_inc(v_nargs_836_);
lean_dec_ref_known(v_a_817_, 2);
v___x_837_ = l_Lean_mkFVar(v_fvarId_835_);
v___x_838_ = l_Lean_MessageData_ofExpr(v___x_837_);
v___x_839_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp(v_keys_810_, v___x_838_, v_nargs_836_, v_paren_811_, v_a_812_, v_a_813_, v_a_814_);
return v___x_839_;
}
case 10:
{
lean_object* v_idx_840_; lean_object* v_nargs_841_; uint8_t v___x_842_; lean_object* v___x_843_; 
lean_del_object(v___x_819_);
v_idx_840_ = lean_ctor_get(v_a_817_, 1);
lean_inc(v_idx_840_);
v_nargs_841_ = lean_ctor_get(v_a_817_, 2);
lean_inc(v_nargs_841_);
lean_dec_ref_known(v_a_817_, 3);
v___x_842_ = 1;
lean_inc_ref(v_keys_810_);
v___x_843_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(v_keys_810_, v___x_842_, v_a_812_, v_a_813_, v_a_814_);
if (lean_obj_tag(v___x_843_) == 0)
{
lean_object* v_a_844_; lean_object* v___x_846_; uint8_t v_isShared_847_; uint8_t v_isSharedCheck_859_; 
v_a_844_ = lean_ctor_get(v___x_843_, 0);
v_isSharedCheck_859_ = !lean_is_exclusive(v___x_843_);
if (v_isSharedCheck_859_ == 0)
{
v___x_846_ = v___x_843_;
v_isShared_847_ = v_isSharedCheck_859_;
goto v_resetjp_845_;
}
else
{
lean_inc(v_a_844_);
lean_dec(v___x_843_);
v___x_846_ = lean_box(0);
v_isShared_847_ = v_isSharedCheck_859_;
goto v_resetjp_845_;
}
v_resetjp_845_:
{
lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_854_; 
v___x_848_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__0, &lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__0_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__0);
v___x_849_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_849_, 0, v_a_844_);
lean_ctor_set(v___x_849_, 1, v___x_848_);
v___x_850_ = lean_unsigned_to_nat(1u);
v___x_851_ = lean_nat_add(v_idx_840_, v___x_850_);
lean_dec(v_idx_840_);
v___x_852_ = l_Nat_reprFast(v___x_851_);
if (v_isShared_847_ == 0)
{
lean_ctor_set_tag(v___x_846_, 3);
lean_ctor_set(v___x_846_, 0, v___x_852_);
v___x_854_ = v___x_846_;
goto v_reusejp_853_;
}
else
{
lean_object* v_reuseFailAlloc_858_; 
v_reuseFailAlloc_858_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_858_, 0, v___x_852_);
v___x_854_ = v_reuseFailAlloc_858_;
goto v_reusejp_853_;
}
v_reusejp_853_:
{
lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_855_ = l_Lean_MessageData_ofFormat(v___x_854_);
v___x_856_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_856_, 0, v___x_849_);
lean_ctor_set(v___x_856_, 1, v___x_855_);
v___x_857_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp(v_keys_810_, v___x_856_, v_nargs_841_, v_paren_811_, v_a_812_, v_a_813_, v_a_814_);
return v___x_857_;
}
}
}
else
{
lean_dec(v_nargs_841_);
lean_dec(v_idx_840_);
lean_dec_ref(v_keys_810_);
return v___x_843_;
}
}
case 5:
{
lean_object* v_deBruijnIndex_860_; lean_object* v_nargs_861_; lean_object* v___x_863_; uint8_t v_isShared_864_; uint8_t v_isSharedCheck_873_; 
lean_del_object(v___x_819_);
v_deBruijnIndex_860_ = lean_ctor_get(v_a_817_, 0);
v_nargs_861_ = lean_ctor_get(v_a_817_, 1);
v_isSharedCheck_873_ = !lean_is_exclusive(v_a_817_);
if (v_isSharedCheck_873_ == 0)
{
v___x_863_ = v_a_817_;
v_isShared_864_ = v_isSharedCheck_873_;
goto v_resetjp_862_;
}
else
{
lean_inc(v_nargs_861_);
lean_inc(v_deBruijnIndex_860_);
lean_dec(v_a_817_);
v___x_863_ = lean_box(0);
v_isShared_864_ = v_isSharedCheck_873_;
goto v_resetjp_862_;
}
v_resetjp_862_:
{
lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_870_; 
v___x_865_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__2, &lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__2_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__2);
v___x_866_ = l_Nat_reprFast(v_deBruijnIndex_860_);
v___x_867_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_867_, 0, v___x_866_);
v___x_868_ = l_Lean_MessageData_ofFormat(v___x_867_);
if (v_isShared_864_ == 0)
{
lean_ctor_set_tag(v___x_863_, 7);
lean_ctor_set(v___x_863_, 1, v___x_868_);
lean_ctor_set(v___x_863_, 0, v___x_865_);
v___x_870_ = v___x_863_;
goto v_reusejp_869_;
}
else
{
lean_object* v_reuseFailAlloc_872_; 
v_reuseFailAlloc_872_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_872_, 0, v___x_865_);
lean_ctor_set(v_reuseFailAlloc_872_, 1, v___x_868_);
v___x_870_ = v_reuseFailAlloc_872_;
goto v_reusejp_869_;
}
v_reusejp_869_:
{
lean_object* v___x_871_; 
v___x_871_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp(v_keys_810_, v___x_870_, v_nargs_861_, v_paren_811_, v_a_812_, v_a_813_, v_a_814_);
return v___x_871_;
}
}
}
case 8:
{
uint8_t v___x_874_; lean_object* v___x_875_; 
lean_del_object(v___x_819_);
v___x_874_ = 0;
v___x_875_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(v_keys_810_, v___x_874_, v_a_812_, v_a_813_, v_a_814_);
if (lean_obj_tag(v___x_875_) == 0)
{
lean_object* v_a_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_886_; 
v_a_876_ = lean_ctor_get(v___x_875_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v___x_875_);
if (v_isSharedCheck_886_ == 0)
{
v___x_878_ = v___x_875_;
v_isShared_879_ = v_isSharedCheck_886_;
goto v_resetjp_877_;
}
else
{
lean_inc(v_a_876_);
lean_dec(v___x_875_);
v___x_878_ = lean_box(0);
v_isShared_879_ = v_isSharedCheck_886_;
goto v_resetjp_877_;
}
v_resetjp_877_:
{
lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_884_; 
v___x_880_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__4, &lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__4_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__4);
v___x_881_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_881_, 0, v___x_880_);
lean_ctor_set(v___x_881_, 1, v_a_876_);
v___x_882_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_parenthesize(v___x_881_, v_paren_811_);
if (v_isShared_879_ == 0)
{
lean_ctor_set(v___x_878_, 0, v___x_882_);
v___x_884_ = v___x_878_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v___x_882_);
v___x_884_ = v_reuseFailAlloc_885_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
return v___x_884_;
}
}
}
else
{
return v___x_875_;
}
}
case 9:
{
uint8_t v___x_887_; lean_object* v___x_888_; 
lean_del_object(v___x_819_);
v___x_887_ = 1;
lean_inc_ref(v_keys_810_);
v___x_888_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(v_keys_810_, v___x_887_, v_a_812_, v_a_813_, v_a_814_);
if (lean_obj_tag(v___x_888_) == 0)
{
lean_object* v_a_889_; uint8_t v___x_890_; lean_object* v___x_891_; 
v_a_889_ = lean_ctor_get(v___x_888_, 0);
lean_inc(v_a_889_);
lean_dec_ref_known(v___x_888_, 1);
v___x_890_ = 0;
v___x_891_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(v_keys_810_, v___x_890_, v_a_812_, v_a_813_, v_a_814_);
if (lean_obj_tag(v___x_891_) == 0)
{
lean_object* v_a_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_903_; 
v_a_892_ = lean_ctor_get(v___x_891_, 0);
v_isSharedCheck_903_ = !lean_is_exclusive(v___x_891_);
if (v_isSharedCheck_903_ == 0)
{
v___x_894_ = v___x_891_;
v_isShared_895_ = v_isSharedCheck_903_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_a_892_);
lean_dec(v___x_891_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_903_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_901_; 
v___x_896_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__6, &lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__6_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___closed__6);
v___x_897_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_897_, 0, v_a_889_);
lean_ctor_set(v___x_897_, 1, v___x_896_);
v___x_898_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_898_, 0, v___x_897_);
lean_ctor_set(v___x_898_, 1, v_a_892_);
v___x_899_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_parenthesize(v___x_898_, v_paren_811_);
if (v_isShared_895_ == 0)
{
lean_ctor_set(v___x_894_, 0, v___x_899_);
v___x_901_ = v___x_894_;
goto v_reusejp_900_;
}
else
{
lean_object* v_reuseFailAlloc_902_; 
v_reuseFailAlloc_902_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_902_, 0, v___x_899_);
v___x_901_ = v_reuseFailAlloc_902_;
goto v_reusejp_900_;
}
v_reusejp_900_:
{
return v___x_901_;
}
}
}
else
{
lean_dec(v_a_889_);
return v___x_891_;
}
}
else
{
lean_dec_ref(v_keys_810_);
return v___x_888_;
}
}
default: 
{
lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_907_; 
lean_dec_ref(v_keys_810_);
v___x_904_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_a_817_);
v___x_905_ = l_Lean_MessageData_ofFormat(v___x_904_);
if (v_isShared_820_ == 0)
{
lean_ctor_set(v___x_819_, 0, v___x_905_);
v___x_907_ = v___x_819_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v___x_905_);
v___x_907_ = v_reuseFailAlloc_908_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
return v___x_907_;
}
}
}
}
}
else
{
lean_object* v_a_910_; lean_object* v___x_912_; uint8_t v_isShared_913_; uint8_t v_isSharedCheck_917_; 
lean_dec_ref(v_keys_810_);
v_a_910_ = lean_ctor_get(v___x_816_, 0);
v_isSharedCheck_917_ = !lean_is_exclusive(v___x_816_);
if (v_isSharedCheck_917_ == 0)
{
v___x_912_ = v___x_816_;
v_isShared_913_ = v_isSharedCheck_917_;
goto v_resetjp_911_;
}
else
{
lean_inc(v_a_910_);
lean_dec(v___x_816_);
v___x_912_ = lean_box(0);
v_isShared_913_ = v_isSharedCheck_917_;
goto v_resetjp_911_;
}
v_resetjp_911_:
{
lean_object* v___x_915_; 
if (v_isShared_913_ == 0)
{
v___x_915_ = v___x_912_;
goto v_reusejp_914_;
}
else
{
lean_object* v_reuseFailAlloc_916_; 
v_reuseFailAlloc_916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_916_, 0, v_a_910_);
v___x_915_ = v_reuseFailAlloc_916_;
goto v_reusejp_914_;
}
v_reusejp_914_:
{
return v___x_915_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg(lean_object* v_keys_918_, lean_object* v_range_919_, lean_object* v_b_920_, lean_object* v_i_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_){
_start:
{
lean_object* v_stop_926_; lean_object* v_step_927_; uint8_t v___x_928_; 
v_stop_926_ = lean_ctor_get(v_range_919_, 1);
v_step_927_ = lean_ctor_get(v_range_919_, 2);
v___x_928_ = lean_nat_dec_lt(v_i_921_, v_stop_926_);
if (v___x_928_ == 0)
{
lean_object* v___x_929_; 
lean_dec(v_i_921_);
lean_dec_ref(v_keys_918_);
v___x_929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_929_, 0, v_b_920_);
return v___x_929_;
}
else
{
lean_object* v___x_930_; 
lean_inc_ref(v_keys_918_);
v___x_930_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(v_keys_918_, v___x_928_, v___y_922_, v___y_923_, v___y_924_);
if (lean_obj_tag(v___x_930_) == 0)
{
lean_object* v_a_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; 
v_a_931_ = lean_ctor_get(v___x_930_, 0);
lean_inc(v_a_931_);
lean_dec_ref_known(v___x_930_, 1);
v___x_932_ = lean_obj_once(&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___closed__0, &lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___closed__0);
v___x_933_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_933_, 0, v_b_920_);
lean_ctor_set(v___x_933_, 1, v___x_932_);
v___x_934_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_934_, 0, v___x_933_);
lean_ctor_set(v___x_934_, 1, v_a_931_);
v___x_935_ = lean_nat_add(v_i_921_, v_step_927_);
lean_dec(v_i_921_);
v_b_920_ = v___x_934_;
v_i_921_ = v___x_935_;
goto _start;
}
else
{
lean_dec(v_i_921_);
lean_dec_ref(v_b_920_);
lean_dec_ref(v_keys_918_);
return v___x_930_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg___boxed(lean_object* v_keys_937_, lean_object* v_range_938_, lean_object* v_b_939_, lean_object* v_i_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_){
_start:
{
lean_object* v_res_945_; 
v_res_945_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg(v_keys_937_, v_range_938_, v_b_939_, v_i_940_, v___y_941_, v___y_942_, v___y_943_);
lean_dec(v___y_943_);
lean_dec_ref(v___y_942_);
lean_dec(v___y_941_);
lean_dec_ref(v_range_938_);
return v_res_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp___boxed(lean_object* v_keys_946_, lean_object* v_f_947_, lean_object* v_nargs_948_, lean_object* v_paren_949_, lean_object* v_a_950_, lean_object* v_a_951_, lean_object* v_a_952_, lean_object* v_a_953_){
_start:
{
uint8_t v_paren_boxed_954_; lean_object* v_res_955_; 
v_paren_boxed_954_ = lean_unbox(v_paren_949_);
v_res_955_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp(v_keys_946_, v_f_947_, v_nargs_948_, v_paren_boxed_954_, v_a_950_, v_a_951_, v_a_952_);
lean_dec(v_a_952_);
lean_dec_ref(v_a_951_);
lean_dec(v_a_950_);
return v_res_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go___boxed(lean_object* v_keys_956_, lean_object* v_paren_957_, lean_object* v_a_958_, lean_object* v_a_959_, lean_object* v_a_960_, lean_object* v_a_961_){
_start:
{
uint8_t v_paren_boxed_962_; lean_object* v_res_963_; 
v_paren_boxed_962_ = lean_unbox(v_paren_957_);
v_res_963_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(v_keys_956_, v_paren_boxed_962_, v_a_958_, v_a_959_, v_a_960_);
lean_dec(v_a_960_);
lean_dec_ref(v_a_959_);
lean_dec(v_a_958_);
return v_res_963_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0(lean_object* v_keys_964_, lean_object* v_range_965_, lean_object* v_b_966_, lean_object* v_i_967_, lean_object* v_hs_968_, lean_object* v_hl_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_){
_start:
{
lean_object* v___x_974_; 
v___x_974_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___redArg(v_keys_964_, v_range_965_, v_b_966_, v_i_967_, v___y_970_, v___y_971_, v___y_972_);
return v___x_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0___boxed(lean_object* v_keys_975_, lean_object* v_range_976_, lean_object* v_b_977_, lean_object* v_i_978_, lean_object* v_hs_979_, lean_object* v_hl_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
lean_object* v_res_985_; 
v_res_985_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_mkApp_spec__0(v_keys_975_, v_range_976_, v_b_977_, v_i_978_, v_hs_979_, v_hl_980_, v___y_981_, v___y_982_, v___y_983_);
lean_dec(v___y_983_);
lean_dec_ref(v___y_982_);
lean_dec(v___y_981_);
lean_dec_ref(v_range_976_);
return v_res_985_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3(lean_object* v_00_u03b1_986_, lean_object* v_constName_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_){
_start:
{
lean_object* v___x_992_; 
v___x_992_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___redArg(v_constName_987_, v___y_988_, v___y_989_, v___y_990_);
return v___x_992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3___boxed(lean_object* v_00_u03b1_993_, lean_object* v_constName_994_, lean_object* v___y_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_){
_start:
{
lean_object* v_res_999_; 
v_res_999_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3(v_00_u03b1_993_, v_constName_994_, v___y_995_, v___y_996_, v___y_997_);
lean_dec(v___y_997_);
lean_dec_ref(v___y_996_);
lean_dec(v___y_995_);
return v_res_999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4(lean_object* v_00_u03b1_1000_, lean_object* v_ref_1001_, lean_object* v_constName_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___redArg(v_ref_1001_, v_constName_1002_, v___y_1003_, v___y_1004_, v___y_1005_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4___boxed(lean_object* v_00_u03b1_1008_, lean_object* v_ref_1009_, lean_object* v_constName_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_){
_start:
{
lean_object* v_res_1015_; 
v_res_1015_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4(v_00_u03b1_1008_, v_ref_1009_, v_constName_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
lean_dec(v___y_1011_);
lean_dec(v_ref_1009_);
return v_res_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6(lean_object* v_00_u03b1_1016_, lean_object* v_ref_1017_, lean_object* v_msg_1018_, lean_object* v_declHint_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_){
_start:
{
lean_object* v___x_1024_; 
v___x_1024_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___redArg(v_ref_1017_, v_msg_1018_, v_declHint_1019_, v___y_1020_, v___y_1021_, v___y_1022_);
return v___x_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6___boxed(lean_object* v_00_u03b1_1025_, lean_object* v_ref_1026_, lean_object* v_msg_1027_, lean_object* v_declHint_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_){
_start:
{
lean_object* v_res_1033_; 
v_res_1033_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6(v_00_u03b1_1025_, v_ref_1026_, v_msg_1027_, v_declHint_1028_, v___y_1029_, v___y_1030_, v___y_1031_);
lean_dec(v___y_1031_);
lean_dec_ref(v___y_1030_);
lean_dec(v___y_1029_);
lean_dec(v_ref_1026_);
return v_res_1033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8(lean_object* v_msg_1034_, lean_object* v_declHint_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_){
_start:
{
lean_object* v___x_1040_; 
v___x_1040_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___redArg(v_msg_1034_, v_declHint_1035_, v___y_1038_);
return v___x_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8___boxed(lean_object* v_msg_1041_, lean_object* v_declHint_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v_res_1047_; 
v_res_1047_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__7_spec__8(v_msg_1041_, v_declHint_1042_, v___y_1043_, v___y_1044_, v___y_1045_);
lean_dec(v___y_1045_);
lean_dec_ref(v___y_1044_);
lean_dec(v___y_1043_);
return v_res_1047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(lean_object* v_00_u03b1_1048_, lean_object* v_ref_1049_, lean_object* v_msg_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_){
_start:
{
lean_object* v___x_1055_; 
v___x_1055_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___redArg(v_ref_1049_, v_msg_1050_, v___y_1052_, v___y_1053_);
return v___x_1055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8___boxed(lean_object* v_00_u03b1_1056_, lean_object* v_ref_1057_, lean_object* v_msg_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v_res_1063_; 
v_res_1063_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_go_spec__2_spec__2_spec__3_spec__4_spec__6_spec__8(v_00_u03b1_1056_, v_ref_1057_, v_msg_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
lean_dec(v___y_1061_);
lean_dec_ref(v___y_1060_);
lean_dec(v___y_1059_);
lean_dec(v_ref_1057_);
return v_res_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__0(lean_object* v_a_1064_, lean_object* v_a_1065_){
_start:
{
if (lean_obj_tag(v_a_1064_) == 0)
{
lean_object* v___x_1066_; 
v___x_1066_ = l_List_reverse___redArg(v_a_1065_);
return v___x_1066_;
}
else
{
lean_object* v_head_1067_; lean_object* v_tail_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1077_; 
v_head_1067_ = lean_ctor_get(v_a_1064_, 0);
v_tail_1068_ = lean_ctor_get(v_a_1064_, 1);
v_isSharedCheck_1077_ = !lean_is_exclusive(v_a_1064_);
if (v_isSharedCheck_1077_ == 0)
{
v___x_1070_ = v_a_1064_;
v_isShared_1071_ = v_isSharedCheck_1077_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_tail_1068_);
lean_inc(v_head_1067_);
lean_dec(v_a_1064_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1077_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v___x_1072_; lean_object* v___x_1074_; 
v___x_1072_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_head_1067_);
if (v_isShared_1071_ == 0)
{
lean_ctor_set(v___x_1070_, 1, v_a_1065_);
lean_ctor_set(v___x_1070_, 0, v___x_1072_);
v___x_1074_ = v___x_1070_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1076_; 
v_reuseFailAlloc_1076_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1076_, 0, v___x_1072_);
lean_ctor_set(v_reuseFailAlloc_1076_, 1, v_a_1065_);
v___x_1074_ = v_reuseFailAlloc_1076_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
v_a_1064_ = v_tail_1068_;
v_a_1065_ = v___x_1074_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___redArg(lean_object* v_msg_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v_ref_1082_; lean_object* v___x_1083_; lean_object* v_a_1084_; lean_object* v___x_1086_; uint8_t v_isShared_1087_; uint8_t v_isSharedCheck_1092_; 
v_ref_1082_ = lean_ctor_get(v___y_1079_, 5);
v___x_1083_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2(v_msg_1078_, v___y_1079_, v___y_1080_);
v_a_1084_ = lean_ctor_get(v___x_1083_, 0);
v_isSharedCheck_1092_ = !lean_is_exclusive(v___x_1083_);
if (v_isSharedCheck_1092_ == 0)
{
v___x_1086_ = v___x_1083_;
v_isShared_1087_ = v_isSharedCheck_1092_;
goto v_resetjp_1085_;
}
else
{
lean_inc(v_a_1084_);
lean_dec(v___x_1083_);
v___x_1086_ = lean_box(0);
v_isShared_1087_ = v_isSharedCheck_1092_;
goto v_resetjp_1085_;
}
v_resetjp_1085_:
{
lean_object* v___x_1088_; lean_object* v___x_1090_; 
lean_inc(v_ref_1082_);
v___x_1088_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1088_, 0, v_ref_1082_);
lean_ctor_set(v___x_1088_, 1, v_a_1084_);
if (v_isShared_1087_ == 0)
{
lean_ctor_set_tag(v___x_1086_, 1);
lean_ctor_set(v___x_1086_, 0, v___x_1088_);
v___x_1090_ = v___x_1086_;
goto v_reusejp_1089_;
}
else
{
lean_object* v_reuseFailAlloc_1091_; 
v_reuseFailAlloc_1091_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1091_, 0, v___x_1088_);
v___x_1090_ = v_reuseFailAlloc_1091_;
goto v_reusejp_1089_;
}
v_reusejp_1089_:
{
return v___x_1090_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___redArg___boxed(lean_object* v_msg_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
lean_object* v_res_1097_; 
v_res_1097_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___redArg(v_msg_1093_, v___y_1094_, v___y_1095_);
lean_dec(v___y_1095_);
lean_dec_ref(v___y_1094_);
return v_res_1097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern(lean_object* v_keys_1098_, lean_object* v_a_1099_, lean_object* v_a_1100_){
_start:
{
lean_object* v___x_1102_; lean_object* v___x_1103_; uint8_t v___x_1104_; lean_object* v___x_1105_; 
lean_inc_ref(v_keys_1098_);
v___x_1102_ = lean_array_to_list(v_keys_1098_);
v___x_1103_ = lean_st_mk_ref(v___x_1102_);
v___x_1104_ = 0;
v___x_1105_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_go(v_keys_1098_, v___x_1104_, v___x_1103_, v_a_1099_, v_a_1100_);
if (lean_obj_tag(v___x_1105_) == 0)
{
lean_object* v_a_1106_; lean_object* v___x_1108_; uint8_t v_isShared_1109_; uint8_t v_isSharedCheck_1130_; 
v_a_1106_ = lean_ctor_get(v___x_1105_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1105_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1108_ = v___x_1105_;
v_isShared_1109_ = v_isSharedCheck_1130_;
goto v_resetjp_1107_;
}
else
{
lean_inc(v_a_1106_);
lean_dec(v___x_1105_);
v___x_1108_ = lean_box(0);
v_isShared_1109_ = v_isSharedCheck_1130_;
goto v_resetjp_1107_;
}
v_resetjp_1107_:
{
lean_object* v___x_1110_; uint8_t v___x_1111_; 
v___x_1110_ = lean_st_ref_get(v___x_1103_);
lean_dec(v___x_1103_);
v___x_1111_ = l_List_isEmpty___redArg(v___x_1110_);
if (v___x_1111_ == 0)
{
lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v_a_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1126_; 
lean_del_object(v___x_1108_);
lean_dec(v_a_1106_);
v___x_1112_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1, &lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern_next___closed__1);
v___x_1113_ = lean_box(0);
v___x_1114_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__0(v___x_1110_, v___x_1113_);
v___x_1115_ = lp_mathlib_List_mapTR_loop___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__1(v___x_1114_, v___x_1113_);
v___x_1116_ = l_Lean_MessageData_ofList(v___x_1115_);
v___x_1117_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1117_, 0, v___x_1112_);
lean_ctor_set(v___x_1117_, 1, v___x_1116_);
v___x_1118_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___redArg(v___x_1117_, v_a_1099_, v_a_1100_);
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1121_ = v___x_1118_;
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_a_1119_);
lean_dec(v___x_1118_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1126_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
lean_object* v___x_1124_; 
if (v_isShared_1122_ == 0)
{
v___x_1124_ = v___x_1121_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1125_; 
v_reuseFailAlloc_1125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1125_, 0, v_a_1119_);
v___x_1124_ = v_reuseFailAlloc_1125_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
return v___x_1124_;
}
}
}
else
{
lean_object* v___x_1128_; 
lean_dec(v___x_1110_);
if (v_isShared_1109_ == 0)
{
v___x_1128_ = v___x_1108_;
goto v_reusejp_1127_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v_a_1106_);
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
lean_dec(v___x_1103_);
return v___x_1105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern___boxed(lean_object* v_keys_1131_, lean_object* v_a_1132_, lean_object* v_a_1133_, lean_object* v_a_1134_){
_start:
{
lean_object* v_res_1135_; 
v_res_1135_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_keysAsPattern(v_keys_1131_, v_a_1132_, v_a_1133_);
lean_dec(v_a_1133_);
lean_dec_ref(v_a_1132_);
return v_res_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1(lean_object* v_00_u03b1_1136_, lean_object* v_msg_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_){
_start:
{
lean_object* v___x_1141_; 
v___x_1141_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___redArg(v_msg_1137_, v___y_1138_, v___y_1139_);
return v___x_1141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1___boxed(lean_object* v_00_u03b1_1142_, lean_object* v_msg_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v_res_1147_; 
v_res_1147_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_spec__1(v_00_u03b1_1142_, v_msg_1143_, v___y_1144_, v___y_1145_);
lean_dec(v___y_1145_);
lean_dec_ref(v___y_1144_);
return v_res_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_arity(lean_object* v_x_1148_){
_start:
{
switch(lean_obj_tag(v_x_1148_))
{
case 3:
{
lean_object* v_nargs_1149_; 
v_nargs_1149_ = lean_ctor_get(v_x_1148_, 1);
lean_inc(v_nargs_1149_);
return v_nargs_1149_;
}
case 4:
{
lean_object* v_nargs_1150_; 
v_nargs_1150_ = lean_ctor_get(v_x_1148_, 1);
lean_inc(v_nargs_1150_);
return v_nargs_1150_;
}
case 5:
{
lean_object* v_nargs_1151_; 
v_nargs_1151_ = lean_ctor_get(v_x_1148_, 1);
lean_inc(v_nargs_1151_);
return v_nargs_1151_;
}
case 8:
{
lean_object* v___x_1152_; 
v___x_1152_ = lean_unsigned_to_nat(1u);
return v___x_1152_;
}
case 9:
{
lean_object* v___x_1153_; 
v___x_1153_ = lean_unsigned_to_nat(2u);
return v___x_1153_;
}
case 10:
{
lean_object* v_nargs_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; 
v_nargs_1154_ = lean_ctor_get(v_x_1148_, 2);
v___x_1155_ = lean_unsigned_to_nat(1u);
v___x_1156_ = lean_nat_add(v_nargs_1154_, v___x_1155_);
return v___x_1156_;
}
default: 
{
lean_object* v___x_1157_; 
v___x_1157_ = lean_unsigned_to_nat(0u);
return v___x_1157_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_arity___boxed(lean_object* v_x_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_arity(v_x_1158_);
lean_dec(v_x_1158_);
return v_res_1159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(lean_object* v_expr_1160_, lean_object* v_bvars_1161_, lean_object* v_a_1162_){
_start:
{
lean_object* v_lctx_1164_; lean_object* v_localInstances_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; 
v_lctx_1164_ = lean_ctor_get(v_a_1162_, 2);
v_localInstances_1165_ = lean_ctor_get(v_a_1162_, 3);
v___x_1166_ = l_Lean_Meta_Context_config(v_a_1162_);
lean_inc_ref(v_localInstances_1165_);
lean_inc_ref(v_lctx_1164_);
v___x_1167_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1167_, 0, v_expr_1160_);
lean_ctor_set(v___x_1167_, 1, v_bvars_1161_);
lean_ctor_set(v___x_1167_, 2, v_lctx_1164_);
lean_ctor_set(v___x_1167_, 3, v_localInstances_1165_);
lean_ctor_set(v___x_1167_, 4, v___x_1166_);
v___x_1168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1168_, 0, v___x_1167_);
return v___x_1168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg___boxed(lean_object* v_expr_1169_, lean_object* v_bvars_1170_, lean_object* v_a_1171_, lean_object* v_a_1172_){
_start:
{
lean_object* v_res_1173_; 
v_res_1173_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v_expr_1169_, v_bvars_1170_, v_a_1171_);
lean_dec_ref(v_a_1171_);
return v_res_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo(lean_object* v_expr_1174_, lean_object* v_bvars_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_){
_start:
{
lean_object* v___x_1181_; 
v___x_1181_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___redArg(v_expr_1174_, v_bvars_1175_, v_a_1176_);
return v___x_1181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo___boxed(lean_object* v_expr_1182_, lean_object* v_bvars_1183_, lean_object* v_a_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_, lean_object* v_a_1188_){
_start:
{
lean_object* v_res_1189_; 
v_res_1189_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkExprInfo(v_expr_1182_, v_bvars_1183_, v_a_1184_, v_a_1185_, v_a_1186_, v_a_1187_);
lean_dec(v_a_1187_);
lean_dec_ref(v_a_1186_);
lean_dec(v_a_1185_);
lean_dec_ref(v_a_1184_);
return v_res_1189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorIdx(lean_object* v_x_1190_){
_start:
{
if (lean_obj_tag(v_x_1190_) == 0)
{
lean_object* v___x_1191_; 
v___x_1191_ = lean_unsigned_to_nat(0u);
return v___x_1191_;
}
else
{
lean_object* v___x_1192_; 
v___x_1192_ = lean_unsigned_to_nat(1u);
return v___x_1192_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorIdx___boxed(lean_object* v_x_1193_){
_start:
{
lean_object* v_res_1194_; 
v_res_1194_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorIdx(v_x_1193_);
lean_dec(v_x_1193_);
return v_res_1194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___redArg(lean_object* v_t_1195_, lean_object* v_k_1196_){
_start:
{
if (lean_obj_tag(v_t_1195_) == 0)
{
return v_k_1196_;
}
else
{
lean_object* v_info_1197_; lean_object* v___x_1198_; 
v_info_1197_ = lean_ctor_get(v_t_1195_, 0);
lean_inc_ref(v_info_1197_);
lean_dec_ref_known(v_t_1195_, 1);
v___x_1198_ = lean_apply_1(v_k_1196_, v_info_1197_);
return v___x_1198_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim(lean_object* v_motive_1199_, lean_object* v_ctorIdx_1200_, lean_object* v_t_1201_, lean_object* v_h_1202_, lean_object* v_k_1203_){
_start:
{
lean_object* v___x_1204_; 
v___x_1204_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___redArg(v_t_1201_, v_k_1203_);
return v___x_1204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___boxed(lean_object* v_motive_1205_, lean_object* v_ctorIdx_1206_, lean_object* v_t_1207_, lean_object* v_h_1208_, lean_object* v_k_1209_){
_start:
{
lean_object* v_res_1210_; 
v_res_1210_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim(v_motive_1205_, v_ctorIdx_1206_, v_t_1207_, v_h_1208_, v_k_1209_);
lean_dec(v_ctorIdx_1206_);
return v_res_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_star_elim___redArg(lean_object* v_t_1211_, lean_object* v_star_1212_){
_start:
{
lean_object* v___x_1213_; 
v___x_1213_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___redArg(v_t_1211_, v_star_1212_);
return v___x_1213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_star_elim(lean_object* v_motive_1214_, lean_object* v_t_1215_, lean_object* v_h_1216_, lean_object* v_star_1217_){
_start:
{
lean_object* v___x_1218_; 
v___x_1218_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___redArg(v_t_1215_, v_star_1217_);
return v___x_1218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_expr_elim___redArg(lean_object* v_t_1219_, lean_object* v_expr_1220_){
_start:
{
lean_object* v___x_1221_; 
v___x_1221_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___redArg(v_t_1219_, v_expr_1220_);
return v___x_1221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_expr_elim(lean_object* v_motive_1222_, lean_object* v_t_1223_, lean_object* v_h_1224_, lean_object* v_expr_1225_){
_start:
{
lean_object* v___x_1226_; 
v___x_1226_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_ctorElim___redArg(v_t_1223_, v_expr_1225_);
return v___x_1226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format(lean_object* v_x_1233_){
_start:
{
if (lean_obj_tag(v_x_1233_) == 0)
{
lean_object* v___x_1234_; 
v___x_1234_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__1));
return v___x_1234_;
}
else
{
lean_object* v_info_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1246_; 
v_info_1235_ = lean_ctor_get(v_x_1233_, 0);
v_isSharedCheck_1246_ = !lean_is_exclusive(v_x_1233_);
if (v_isSharedCheck_1246_ == 0)
{
v___x_1237_ = v_x_1233_;
v_isShared_1238_ = v_isSharedCheck_1246_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_info_1235_);
lean_dec(v_x_1233_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1246_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v_expr_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1243_; 
v_expr_1239_ = lean_ctor_get(v_info_1235_, 0);
lean_inc_ref(v_expr_1239_);
lean_dec_ref(v_info_1235_);
v___x_1240_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format___closed__3));
v___x_1241_ = lean_expr_dbg_to_string(v_expr_1239_);
lean_dec_ref(v_expr_1239_);
if (v_isShared_1238_ == 0)
{
lean_ctor_set_tag(v___x_1237_, 3);
lean_ctor_set(v___x_1237_, 0, v___x_1241_);
v___x_1243_ = v___x_1237_;
goto v_reusejp_1242_;
}
else
{
lean_object* v_reuseFailAlloc_1245_; 
v_reuseFailAlloc_1245_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1245_, 0, v___x_1241_);
v___x_1243_ = v_reuseFailAlloc_1245_;
goto v_reusejp_1242_;
}
v_reusejp_1242_:
{
lean_object* v___x_1244_; 
v___x_1244_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1244_, 0, v___x_1240_);
lean_ctor_set(v___x_1244_, 1, v___x_1243_);
return v___x_1244_;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default___closed__0(void){
_start:
{
lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; 
v___x_1249_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_Meta_RefinedDiscrTree_keysAsPattern_next_spec__2_spec__2___closed__2);
v___x_1250_ = lean_box(0);
v___x_1251_ = lean_box(0);
v___x_1252_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1252_, 0, v___x_1251_);
lean_ctor_set(v___x_1252_, 1, v___x_1250_);
lean_ctor_set(v___x_1252_, 2, v___x_1249_);
lean_ctor_set(v___x_1252_, 3, v___x_1251_);
lean_ctor_set(v___x_1252_, 4, v___x_1250_);
return v___x_1252_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default(void){
_start:
{
lean_object* v___x_1253_; 
v___x_1253_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default___closed__0, &lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default___closed__0_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default___closed__0);
return v___x_1253_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry(void){
_start:
{
lean_object* v___x_1254_; 
v___x_1254_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default;
return v___x_1254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(uint8_t v_labelledStars_1259_, lean_object* v_a_1260_){
_start:
{
lean_object* v___x_1262_; lean_object* v_mctx_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1276_; 
v___x_1262_ = lean_st_ref_get(v_a_1260_);
v_mctx_1263_ = lean_ctor_get(v___x_1262_, 0);
v_isSharedCheck_1276_ = !lean_is_exclusive(v___x_1262_);
if (v_isSharedCheck_1276_ == 0)
{
lean_object* v_unused_1277_; lean_object* v_unused_1278_; lean_object* v_unused_1279_; lean_object* v_unused_1280_; 
v_unused_1277_ = lean_ctor_get(v___x_1262_, 4);
lean_dec(v_unused_1277_);
v_unused_1278_ = lean_ctor_get(v___x_1262_, 3);
lean_dec(v_unused_1278_);
v_unused_1279_ = lean_ctor_get(v___x_1262_, 2);
lean_dec(v_unused_1279_);
v_unused_1280_ = lean_ctor_get(v___x_1262_, 1);
lean_dec(v_unused_1280_);
v___x_1265_ = v___x_1262_;
v_isShared_1266_ = v_isSharedCheck_1276_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_mctx_1263_);
lean_dec(v___x_1262_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1276_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___y_1270_; 
v___x_1267_ = lean_box(0);
v___x_1268_ = lean_box(0);
if (v_labelledStars_1259_ == 0)
{
v___y_1270_ = v___x_1267_;
goto v___jp_1269_;
}
else
{
lean_object* v___x_1275_; 
v___x_1275_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___closed__1));
v___y_1270_ = v___x_1275_;
goto v___jp_1269_;
}
v___jp_1269_:
{
lean_object* v___x_1272_; 
lean_inc(v___y_1270_);
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 4, v___x_1268_);
lean_ctor_set(v___x_1265_, 3, v___y_1270_);
lean_ctor_set(v___x_1265_, 2, v_mctx_1263_);
lean_ctor_set(v___x_1265_, 1, v___x_1268_);
lean_ctor_set(v___x_1265_, 0, v___x_1267_);
v___x_1272_ = v___x_1265_;
goto v_reusejp_1271_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v___x_1267_);
lean_ctor_set(v_reuseFailAlloc_1274_, 1, v___x_1268_);
lean_ctor_set(v_reuseFailAlloc_1274_, 2, v_mctx_1263_);
lean_ctor_set(v_reuseFailAlloc_1274_, 3, v___y_1270_);
lean_ctor_set(v_reuseFailAlloc_1274_, 4, v___x_1268_);
v___x_1272_ = v_reuseFailAlloc_1274_;
goto v_reusejp_1271_;
}
v_reusejp_1271_:
{
lean_object* v___x_1273_; 
v___x_1273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1273_, 0, v___x_1272_);
return v___x_1273_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg___boxed(lean_object* v_labelledStars_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_){
_start:
{
uint8_t v_labelledStars_boxed_1284_; lean_object* v_res_1285_; 
v_labelledStars_boxed_1284_ = lean_unbox(v_labelledStars_1281_);
v_res_1285_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(v_labelledStars_boxed_1284_, v_a_1282_);
lean_dec(v_a_1282_);
return v_res_1285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry(uint8_t v_labelledStars_1286_, lean_object* v_a_1287_, lean_object* v_a_1288_, lean_object* v_a_1289_, lean_object* v_a_1290_){
_start:
{
lean_object* v___x_1292_; 
v___x_1292_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___redArg(v_labelledStars_1286_, v_a_1288_);
return v___x_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry___boxed(lean_object* v_labelledStars_1293_, lean_object* v_a_1294_, lean_object* v_a_1295_, lean_object* v_a_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_){
_start:
{
uint8_t v_labelledStars_boxed_1299_; lean_object* v_res_1300_; 
v_labelledStars_boxed_1299_ = lean_unbox(v_labelledStars_1293_);
v_res_1300_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_mkInitLazyEntry(v_labelledStars_boxed_1299_, v_a_1294_, v_a_1295_, v_a_1296_, v_a_1297_);
lean_dec(v_a_1297_);
lean_dec_ref(v_a_1296_);
lean_dec(v_a_1295_);
lean_dec_ref(v_a_1294_);
return v_res_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__2_spec__3(lean_object* v_x_1301_, lean_object* v_x_1302_, lean_object* v_x_1303_){
_start:
{
if (lean_obj_tag(v_x_1303_) == 0)
{
lean_dec(v_x_1301_);
return v_x_1302_;
}
else
{
lean_object* v_head_1304_; lean_object* v_tail_1305_; lean_object* v___x_1307_; uint8_t v_isShared_1308_; uint8_t v_isSharedCheck_1315_; 
v_head_1304_ = lean_ctor_get(v_x_1303_, 0);
v_tail_1305_ = lean_ctor_get(v_x_1303_, 1);
v_isSharedCheck_1315_ = !lean_is_exclusive(v_x_1303_);
if (v_isSharedCheck_1315_ == 0)
{
v___x_1307_ = v_x_1303_;
v_isShared_1308_ = v_isSharedCheck_1315_;
goto v_resetjp_1306_;
}
else
{
lean_inc(v_tail_1305_);
lean_inc(v_head_1304_);
lean_dec(v_x_1303_);
v___x_1307_ = lean_box(0);
v_isShared_1308_ = v_isSharedCheck_1315_;
goto v_resetjp_1306_;
}
v_resetjp_1306_:
{
lean_object* v___x_1310_; 
lean_inc(v_x_1301_);
if (v_isShared_1308_ == 0)
{
lean_ctor_set_tag(v___x_1307_, 5);
lean_ctor_set(v___x_1307_, 1, v_x_1301_);
lean_ctor_set(v___x_1307_, 0, v_x_1302_);
v___x_1310_ = v___x_1307_;
goto v_reusejp_1309_;
}
else
{
lean_object* v_reuseFailAlloc_1314_; 
v_reuseFailAlloc_1314_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1314_, 0, v_x_1302_);
lean_ctor_set(v_reuseFailAlloc_1314_, 1, v_x_1301_);
v___x_1310_ = v_reuseFailAlloc_1314_;
goto v_reusejp_1309_;
}
v_reusejp_1309_:
{
lean_object* v___x_1311_; lean_object* v___x_1312_; 
v___x_1311_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format(v_head_1304_);
v___x_1312_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1312_, 0, v___x_1310_);
lean_ctor_set(v___x_1312_, 1, v___x_1311_);
v_x_1302_ = v___x_1312_;
v_x_1303_ = v_tail_1305_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__2(lean_object* v_x_1316_, lean_object* v_x_1317_){
_start:
{
if (lean_obj_tag(v_x_1316_) == 0)
{
lean_object* v___x_1318_; 
lean_dec(v_x_1317_);
v___x_1318_ = lean_box(0);
return v___x_1318_;
}
else
{
lean_object* v_tail_1319_; 
v_tail_1319_ = lean_ctor_get(v_x_1316_, 1);
if (lean_obj_tag(v_tail_1319_) == 0)
{
lean_object* v_head_1320_; lean_object* v___x_1321_; 
lean_dec(v_x_1317_);
v_head_1320_ = lean_ctor_get(v_x_1316_, 0);
lean_inc(v_head_1320_);
lean_dec_ref_known(v_x_1316_, 2);
v___x_1321_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format(v_head_1320_);
return v___x_1321_;
}
else
{
lean_object* v_head_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; 
lean_inc(v_tail_1319_);
v_head_1322_ = lean_ctor_get(v_x_1316_, 0);
lean_inc(v_head_1322_);
lean_dec_ref_known(v_x_1316_, 2);
v___x_1323_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_StackEntry_format(v_head_1322_);
v___x_1324_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__2_spec__3(v_x_1317_, v___x_1323_, v_tail_1319_);
return v___x_1324_;
}
}
}
}
static lean_object* _init_lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__7(void){
_start:
{
lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1336_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__5));
v___x_1337_ = lean_string_length(v___x_1336_);
return v___x_1337_;
}
}
static lean_object* _init_lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8(void){
_start:
{
lean_object* v___x_1338_; lean_object* v___x_1339_; 
v___x_1338_ = lean_obj_once(&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__7, &lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__7_once, _init_lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__7);
v___x_1339_ = lean_nat_to_int(v___x_1338_);
return v___x_1339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1(lean_object* v_x_1344_){
_start:
{
if (lean_obj_tag(v_x_1344_) == 0)
{
lean_object* v___x_1345_; 
v___x_1345_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__1));
return v___x_1345_;
}
else
{
lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; uint8_t v___x_1354_; lean_object* v___x_1355_; 
v___x_1346_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__4));
v___x_1347_ = lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__2(v_x_1344_, v___x_1346_);
v___x_1348_ = lean_obj_once(&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8, &lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8_once, _init_lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8);
v___x_1349_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__9));
v___x_1350_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1350_, 0, v___x_1349_);
lean_ctor_set(v___x_1350_, 1, v___x_1347_);
v___x_1351_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__10));
v___x_1352_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1352_, 0, v___x_1350_);
lean_ctor_set(v___x_1352_, 1, v___x_1351_);
v___x_1353_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1353_, 0, v___x_1348_);
lean_ctor_set(v___x_1353_, 1, v___x_1352_);
v___x_1354_ = 0;
v___x_1355_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1355_, 0, v___x_1353_);
lean_ctor_set_uint8(v___x_1355_, sizeof(void*)*1, v___x_1354_);
return v___x_1355_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3_spec__6_spec__8(lean_object* v_x_1356_, lean_object* v_x_1357_, lean_object* v_x_1358_){
_start:
{
if (lean_obj_tag(v_x_1358_) == 0)
{
lean_dec(v_x_1356_);
return v_x_1357_;
}
else
{
lean_object* v_head_1359_; lean_object* v_tail_1360_; lean_object* v___x_1362_; uint8_t v_isShared_1363_; uint8_t v_isSharedCheck_1370_; 
v_head_1359_ = lean_ctor_get(v_x_1358_, 0);
v_tail_1360_ = lean_ctor_get(v_x_1358_, 1);
v_isSharedCheck_1370_ = !lean_is_exclusive(v_x_1358_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1362_ = v_x_1358_;
v_isShared_1363_ = v_isSharedCheck_1370_;
goto v_resetjp_1361_;
}
else
{
lean_inc(v_tail_1360_);
lean_inc(v_head_1359_);
lean_dec(v_x_1358_);
v___x_1362_ = lean_box(0);
v_isShared_1363_ = v_isSharedCheck_1370_;
goto v_resetjp_1361_;
}
v_resetjp_1361_:
{
lean_object* v___x_1365_; 
lean_inc(v_x_1356_);
if (v_isShared_1363_ == 0)
{
lean_ctor_set_tag(v___x_1362_, 5);
lean_ctor_set(v___x_1362_, 1, v_x_1356_);
lean_ctor_set(v___x_1362_, 0, v_x_1357_);
v___x_1365_ = v___x_1362_;
goto v_reusejp_1364_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_x_1357_);
lean_ctor_set(v_reuseFailAlloc_1369_, 1, v_x_1356_);
v___x_1365_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1364_;
}
v_reusejp_1364_:
{
lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1366_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_head_1359_);
v___x_1367_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1367_, 0, v___x_1365_);
lean_ctor_set(v___x_1367_, 1, v___x_1366_);
v_x_1357_ = v___x_1367_;
v_x_1358_ = v_tail_1360_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3_spec__6(lean_object* v_x_1371_, lean_object* v_x_1372_){
_start:
{
if (lean_obj_tag(v_x_1371_) == 0)
{
lean_object* v___x_1373_; 
lean_dec(v_x_1372_);
v___x_1373_ = lean_box(0);
return v___x_1373_;
}
else
{
lean_object* v_tail_1374_; 
v_tail_1374_ = lean_ctor_get(v_x_1371_, 1);
if (lean_obj_tag(v_tail_1374_) == 0)
{
lean_object* v_head_1375_; lean_object* v___x_1376_; 
lean_dec(v_x_1372_);
v_head_1375_ = lean_ctor_get(v_x_1371_, 0);
lean_inc(v_head_1375_);
lean_dec_ref_known(v_x_1371_, 2);
v___x_1376_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_head_1375_);
return v___x_1376_;
}
else
{
lean_object* v_head_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; 
lean_inc(v_tail_1374_);
v_head_1377_ = lean_ctor_get(v_x_1371_, 0);
lean_inc(v_head_1377_);
lean_dec_ref_known(v_x_1371_, 2);
v___x_1378_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_head_1377_);
v___x_1379_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3_spec__6_spec__8(v_x_1372_, v___x_1378_, v_tail_1374_);
return v___x_1379_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3(lean_object* v_x_1380_){
_start:
{
if (lean_obj_tag(v_x_1380_) == 0)
{
lean_object* v___x_1381_; 
v___x_1381_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__1));
return v___x_1381_;
}
else
{
lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; uint8_t v___x_1390_; lean_object* v___x_1391_; 
v___x_1382_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__4));
v___x_1383_ = lp_mathlib_Std_Format_joinSep___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3_spec__6(v_x_1380_, v___x_1382_);
v___x_1384_ = lean_obj_once(&lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8, &lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8_once, _init_lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__8);
v___x_1385_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__9));
v___x_1386_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1386_, 0, v___x_1385_);
lean_ctor_set(v___x_1386_, 1, v___x_1383_);
v___x_1387_ = ((lean_object*)(lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1___closed__10));
v___x_1388_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1388_, 0, v___x_1386_);
lean_ctor_set(v___x_1388_, 1, v___x_1387_);
v___x_1389_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1389_, 0, v___x_1384_);
lean_ctor_set(v___x_1389_, 1, v___x_1388_);
v___x_1390_ = 0;
v___x_1391_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1391_, 0, v___x_1389_);
lean_ctor_set_uint8(v___x_1391_, sizeof(void*)*1, v___x_1390_);
return v___x_1391_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__0_spec__0(lean_object* v_x_1392_, lean_object* v_x_1393_, lean_object* v_x_1394_){
_start:
{
if (lean_obj_tag(v_x_1394_) == 0)
{
lean_dec(v_x_1392_);
return v_x_1393_;
}
else
{
lean_object* v_head_1395_; lean_object* v_tail_1396_; lean_object* v___x_1398_; uint8_t v_isShared_1399_; uint8_t v_isSharedCheck_1405_; 
v_head_1395_ = lean_ctor_get(v_x_1394_, 0);
v_tail_1396_ = lean_ctor_get(v_x_1394_, 1);
v_isSharedCheck_1405_ = !lean_is_exclusive(v_x_1394_);
if (v_isSharedCheck_1405_ == 0)
{
v___x_1398_ = v_x_1394_;
v_isShared_1399_ = v_isSharedCheck_1405_;
goto v_resetjp_1397_;
}
else
{
lean_inc(v_tail_1396_);
lean_inc(v_head_1395_);
lean_dec(v_x_1394_);
v___x_1398_ = lean_box(0);
v_isShared_1399_ = v_isSharedCheck_1405_;
goto v_resetjp_1397_;
}
v_resetjp_1397_:
{
lean_object* v___x_1401_; 
lean_inc(v_x_1392_);
if (v_isShared_1399_ == 0)
{
lean_ctor_set_tag(v___x_1398_, 5);
lean_ctor_set(v___x_1398_, 1, v_x_1392_);
lean_ctor_set(v___x_1398_, 0, v_x_1393_);
v___x_1401_ = v___x_1398_;
goto v_reusejp_1400_;
}
else
{
lean_object* v_reuseFailAlloc_1404_; 
v_reuseFailAlloc_1404_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1404_, 0, v_x_1393_);
lean_ctor_set(v_reuseFailAlloc_1404_, 1, v_x_1392_);
v___x_1401_ = v_reuseFailAlloc_1404_;
goto v_reusejp_1400_;
}
v_reusejp_1400_:
{
lean_object* v___x_1402_; 
v___x_1402_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1402_, 0, v___x_1401_);
lean_ctor_set(v___x_1402_, 1, v_head_1395_);
v_x_1393_ = v___x_1402_;
v_x_1394_ = v_tail_1396_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_Format_joinSep___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__0(lean_object* v_x_1406_, lean_object* v_x_1407_){
_start:
{
if (lean_obj_tag(v_x_1406_) == 0)
{
lean_object* v___x_1408_; 
lean_dec(v_x_1407_);
v___x_1408_ = lean_box(0);
return v___x_1408_;
}
else
{
lean_object* v_tail_1409_; 
v_tail_1409_ = lean_ctor_get(v_x_1406_, 1);
if (lean_obj_tag(v_tail_1409_) == 0)
{
lean_object* v_head_1410_; 
lean_dec(v_x_1407_);
v_head_1410_ = lean_ctor_get(v_x_1406_, 0);
lean_inc(v_head_1410_);
lean_dec_ref_known(v_x_1406_, 2);
return v_head_1410_;
}
else
{
lean_object* v_head_1411_; lean_object* v___x_1412_; 
lean_inc(v_tail_1409_);
v_head_1411_ = lean_ctor_get(v_x_1406_, 0);
lean_inc(v_head_1411_);
lean_dec_ref_known(v_x_1406_, 2);
v___x_1412_ = lp_mathlib_List_foldl___at___00Std_Format_joinSep___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__0_spec__0(v_x_1407_, v_head_1411_, v_tail_1409_);
return v___x_1412_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__2(lean_object* v_x_1413_, lean_object* v_x_1414_){
_start:
{
if (lean_obj_tag(v_x_1413_) == 0)
{
if (lean_obj_tag(v_x_1414_) == 0)
{
uint8_t v___x_1415_; 
v___x_1415_ = 1;
return v___x_1415_;
}
else
{
uint8_t v___x_1416_; 
v___x_1416_ = 0;
return v___x_1416_;
}
}
else
{
if (lean_obj_tag(v_x_1414_) == 0)
{
uint8_t v___x_1417_; 
v___x_1417_ = 0;
return v___x_1417_;
}
else
{
lean_object* v_head_1418_; lean_object* v_tail_1419_; lean_object* v_head_1420_; lean_object* v_tail_1421_; uint8_t v___x_1422_; 
v_head_1418_ = lean_ctor_get(v_x_1413_, 0);
v_tail_1419_ = lean_ctor_get(v_x_1413_, 1);
v_head_1420_ = lean_ctor_get(v_x_1414_, 0);
v_tail_1421_ = lean_ctor_get(v_x_1414_, 1);
v___x_1422_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_instBEqKey_beq(v_head_1418_, v_head_1420_);
if (v___x_1422_ == 0)
{
return v___x_1422_;
}
else
{
v_x_1413_ = v_tail_1419_;
v_x_1414_ = v_tail_1421_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__2___boxed(lean_object* v_x_1424_, lean_object* v_x_1425_){
_start:
{
uint8_t v_res_1426_; lean_object* v_r_1427_; 
v_res_1426_ = lp_mathlib_List_beq___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__2(v_x_1424_, v_x_1425_);
lean_dec(v_x_1425_);
lean_dec(v_x_1424_);
v_r_1427_ = lean_box(v_res_1426_);
return v_r_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format(lean_object* v_entry_1437_){
_start:
{
lean_object* v_parts_1439_; lean_object* v_previous_1443_; lean_object* v_stack_1444_; lean_object* v_computedKeys_1445_; lean_object* v_parts_1447_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v_parts_1466_; lean_object* v___x_1467_; uint8_t v___x_1468_; 
v_previous_1443_ = lean_ctor_get(v_entry_1437_, 0);
lean_inc(v_previous_1443_);
v_stack_1444_ = lean_ctor_get(v_entry_1437_, 1);
lean_inc(v_stack_1444_);
v_computedKeys_1445_ = lean_ctor_get(v_entry_1437_, 4);
lean_inc(v_computedKeys_1445_);
lean_dec_ref(v_entry_1437_);
v___x_1461_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__3));
v___x_1462_ = lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1(v_stack_1444_);
v___x_1463_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1463_, 0, v___x_1461_);
lean_ctor_set(v___x_1463_, 1, v___x_1462_);
v___x_1464_ = lean_unsigned_to_nat(1u);
v___x_1465_ = lean_mk_empty_array_with_capacity(v___x_1464_);
v_parts_1466_ = lean_array_push(v___x_1465_, v___x_1463_);
v___x_1467_ = lean_box(0);
v___x_1468_ = lp_mathlib_List_beq___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__2(v_computedKeys_1445_, v___x_1467_);
if (v___x_1468_ == 0)
{
lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v_parts_1472_; 
v___x_1469_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__5));
v___x_1470_ = lp_mathlib_List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__3(v_computedKeys_1445_);
v___x_1471_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1471_, 0, v___x_1469_);
lean_ctor_set(v___x_1471_, 1, v___x_1470_);
v_parts_1472_ = lean_array_push(v_parts_1466_, v___x_1471_);
v_parts_1447_ = v_parts_1472_;
goto v___jp_1446_;
}
else
{
lean_dec(v_computedKeys_1445_);
v_parts_1447_ = v_parts_1466_;
goto v___jp_1446_;
}
v___jp_1438_:
{
lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; 
v___x_1440_ = lean_array_to_list(v_parts_1439_);
v___x_1441_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__7));
v___x_1442_ = lp_mathlib_Std_Format_joinSep___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__0(v___x_1440_, v___x_1441_);
return v___x_1442_;
}
v___jp_1446_:
{
if (lean_obj_tag(v_previous_1443_) == 1)
{
lean_object* v_val_1448_; lean_object* v___x_1450_; uint8_t v_isShared_1451_; uint8_t v_isSharedCheck_1460_; 
v_val_1448_ = lean_ctor_get(v_previous_1443_, 0);
v_isSharedCheck_1460_ = !lean_is_exclusive(v_previous_1443_);
if (v_isSharedCheck_1460_ == 0)
{
v___x_1450_ = v_previous_1443_;
v_isShared_1451_ = v_isSharedCheck_1460_;
goto v_resetjp_1449_;
}
else
{
lean_inc(v_val_1448_);
lean_dec(v_previous_1443_);
v___x_1450_ = lean_box(0);
v_isShared_1451_ = v_isSharedCheck_1460_;
goto v_resetjp_1449_;
}
v_resetjp_1449_:
{
lean_object* v_expr_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1456_; 
v_expr_1452_ = lean_ctor_get(v_val_1448_, 0);
lean_inc_ref(v_expr_1452_);
lean_dec(v_val_1448_);
v___x_1453_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_LazyEntry_format___closed__1));
v___x_1454_ = lean_expr_dbg_to_string(v_expr_1452_);
lean_dec_ref(v_expr_1452_);
if (v_isShared_1451_ == 0)
{
lean_ctor_set_tag(v___x_1450_, 3);
lean_ctor_set(v___x_1450_, 0, v___x_1454_);
v___x_1456_ = v___x_1450_;
goto v_reusejp_1455_;
}
else
{
lean_object* v_reuseFailAlloc_1459_; 
v_reuseFailAlloc_1459_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1459_, 0, v___x_1454_);
v___x_1456_ = v_reuseFailAlloc_1459_;
goto v_reusejp_1455_;
}
v_reusejp_1455_:
{
lean_object* v___x_1457_; lean_object* v_parts_1458_; 
v___x_1457_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1457_, 0, v___x_1453_);
lean_ctor_set(v___x_1457_, 1, v___x_1456_);
v_parts_1458_ = lean_array_push(v_parts_1447_, v___x_1457_);
v_parts_1439_ = v_parts_1458_;
goto v___jp_1438_;
}
}
}
else
{
lean_dec(v_previous_1443_);
v_parts_1439_ = v_parts_1447_;
goto v___jp_1438_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00List_format___at___00Lean_Meta_RefinedDiscrTree_LazyEntry_format_spec__1_spec__3(lean_object* v_a_1473_){
_start:
{
lean_object* v___x_1474_; 
v___x_1474_ = lean_nat_to_int(v_a_1473_);
return v___x_1474_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__1(void){
_start:
{
lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; 
v___x_1479_ = lean_box(0);
v___x_1480_ = lean_unsigned_to_nat(16u);
v___x_1481_ = lean_mk_array(v___x_1480_, v___x_1479_);
return v___x_1481_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__2(void){
_start:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; 
v___x_1482_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__1, &lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__1_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__1);
v___x_1483_ = lean_unsigned_to_nat(0u);
v___x_1484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1484_, 0, v___x_1483_);
lean_ctor_set(v___x_1484_, 1, v___x_1482_);
return v___x_1484_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__3(void){
_start:
{
lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; 
v___x_1485_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__2, &lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__2_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__2);
v___x_1486_ = lean_box(0);
v___x_1487_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__0));
v___x_1488_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1488_, 0, v___x_1487_);
lean_ctor_set(v___x_1488_, 1, v___x_1486_);
lean_ctor_set(v___x_1488_, 2, v___x_1485_);
lean_ctor_set(v___x_1488_, 3, v___x_1485_);
lean_ctor_set(v___x_1488_, 4, v___x_1487_);
return v___x_1488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie(lean_object* v_00_u03b1_1489_){
_start:
{
lean_object* v___x_1490_; 
v___x_1490_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__3, &lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__3_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie___closed__3);
return v___x_1490_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__0(void){
_start:
{
lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; 
v___x_1491_ = lean_box(0);
v___x_1492_ = lean_unsigned_to_nat(16u);
v___x_1493_ = lean_mk_array(v___x_1492_, v___x_1491_);
return v___x_1493_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__1(void){
_start:
{
lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; 
v___x_1494_ = lean_obj_once(&lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__0, &lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__0_once, _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__0);
v___x_1495_ = lean_unsigned_to_nat(0u);
v___x_1496_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1496_, 0, v___x_1495_);
lean_ctor_set(v___x_1496_, 1, v___x_1494_);
return v___x_1496_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__3(void){
_start:
{
lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; 
v___x_1499_ = ((lean_object*)(lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__2));
v___x_1500_ = lean_obj_once(&lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__1, &lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__1_once, _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__1);
v___x_1501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1501_, 0, v___x_1500_);
lean_ctor_set(v___x_1501_, 1, v___x_1499_);
return v___x_1501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default(lean_object* v_00_u03b1_1502_){
_start:
{
lean_object* v___x_1503_; 
v___x_1503_ = lean_obj_once(&lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__3, &lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__3_once, _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default___closed__3);
return v___x_1503_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree___closed__0(void){
_start:
{
lean_object* v___x_1504_; 
v___x_1504_ = lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree_default(lean_box(0));
return v___x_1504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree(lean_object* v_a_1505_){
_start:
{
lean_object* v___x_1506_; 
v___x_1506_ = lean_obj_once(&lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree___closed__0, &lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree___closed__0_once, _init_lp_mathlib_Lean_Meta_instInhabitedRefinedDiscrTree___closed__0);
return v___x_1506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__2(lean_object* v___x_1507_, lean_object* v___f_1508_, lean_object* v_acc_1509_, lean_object* v_l_1510_){
_start:
{
lean_object* v___x_1511_; 
v___x_1511_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v___x_1507_, v___f_1508_, v_acc_1509_, v_l_1510_);
return v___x_1511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__3(lean_object* v___x_1512_, lean_object* v___f_1513_, lean_object* v_acc_1514_, lean_object* v_l_1515_){
_start:
{
lean_object* v___x_1516_; 
v___x_1516_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v___x_1512_, v___f_1513_, v_acc_1514_, v_l_1515_);
return v___x_1516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__4(lean_object* v_x_1517_){
_start:
{
lean_object* v_snd_1518_; 
v_snd_1518_ = lean_ctor_get(v_x_1517_, 1);
lean_inc(v_snd_1518_);
return v_snd_1518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__4___boxed(lean_object* v_x_1519_){
_start:
{
lean_object* v_res_1520_; 
v_res_1520_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__4(v_x_1519_);
lean_dec_ref(v_x_1519_);
return v_res_1520_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1521_; lean_object* v___x_1522_; 
v___x_1521_ = lean_unsigned_to_nat(2u);
v___x_1522_ = lean_nat_to_int(v___x_1521_);
return v___x_1522_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__0(void){
_start:
{
lean_object* v___x_1526_; 
v___x_1526_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedTrie(lean_box(0));
return v___x_1526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__0(lean_object* v_inst_1527_, lean_object* v_tree_1528_, lean_object* v_x1_1529_, lean_object* v_x2_1530_, lean_object* v_x3_1531_){
_start:
{
lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; 
v___x_1532_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0, &lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0);
v___x_1533_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format___closed__1));
v___x_1534_ = l_Nat_reprFast(v_x2_1530_);
v___x_1535_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1535_, 0, v___x_1534_);
v___x_1536_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1536_, 0, v___x_1533_);
lean_ctor_set(v___x_1536_, 1, v___x_1535_);
v___x_1537_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__2));
v___x_1538_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1538_, 0, v___x_1536_);
lean_ctor_set(v___x_1538_, 1, v___x_1537_);
v___x_1539_ = lean_box(1);
v___x_1540_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1540_, 0, v___x_1538_);
lean_ctor_set(v___x_1540_, 1, v___x_1539_);
v___x_1541_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(v_inst_1527_, v_tree_1528_, v_x3_1531_);
v___x_1542_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1542_, 0, v___x_1540_);
lean_ctor_set(v___x_1542_, 1, v___x_1541_);
v___x_1543_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1543_, 0, v___x_1532_);
lean_ctor_set(v___x_1543_, 1, v___x_1542_);
v___x_1544_ = lean_array_push(v_x1_1529_, v___x_1543_);
return v___x_1544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__0___boxed(lean_object* v_inst_1545_, lean_object* v_tree_1546_, lean_object* v_x1_1547_, lean_object* v_x2_1548_, lean_object* v_x3_1549_){
_start:
{
lean_object* v_res_1550_; 
v_res_1550_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__0(v_inst_1545_, v_tree_1546_, v_x1_1547_, v_x2_1548_, v_x3_1549_);
lean_dec(v_x3_1549_);
return v_res_1550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___boxed(lean_object* v_inst_1551_, lean_object* v_tree_1552_, lean_object* v_x1_1553_, lean_object* v_x2_1554_, lean_object* v_x3_1555_){
_start:
{
lean_object* v_res_1556_; 
v_res_1556_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1(v_inst_1551_, v_tree_1552_, v_x1_1553_, v_x2_1554_, v_x3_1555_);
lean_dec(v_x3_1555_);
return v_res_1556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(lean_object* v_inst_1600_, lean_object* v_tree_1601_, lean_object* v_trie_1602_){
_start:
{
lean_object* v_tries_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v_values_1606_; lean_object* v_star_1607_; lean_object* v_labelledStars_1608_; lean_object* v_children_1609_; lean_object* v_pending_1610_; lean_object* v___f_1611_; lean_object* v___f_1612_; lean_object* v___f_1613_; lean_object* v___y_1615_; lean_object* v___y_1624_; lean_object* v_lines_1639_; lean_object* v_lines_1654_; lean_object* v_lines_1663_; lean_object* v___x_1674_; lean_object* v_lines_1675_; lean_object* v___x_1676_; uint8_t v___x_1677_; 
v_tries_1603_ = lean_ctor_get(v_tree_1601_, 1);
v___x_1604_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__0, &lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__0_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__0);
v___x_1605_ = lean_array_get_borrowed(v___x_1604_, v_tries_1603_, v_trie_1602_);
v_values_1606_ = lean_ctor_get(v___x_1605_, 0);
v_star_1607_ = lean_ctor_get(v___x_1605_, 1);
v_labelledStars_1608_ = lean_ctor_get(v___x_1605_, 2);
lean_inc_ref(v_labelledStars_1608_);
v_children_1609_ = lean_ctor_get(v___x_1605_, 3);
lean_inc_ref(v_children_1609_);
v_pending_1610_ = lean_ctor_get(v___x_1605_, 4);
lean_inc_ref_n(v_tree_1601_, 2);
lean_inc_ref_n(v_inst_1600_, 2);
v___f_1611_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__0___boxed), 5, 2);
lean_closure_set(v___f_1611_, 0, v_inst_1600_);
lean_closure_set(v___f_1611_, 1, v_tree_1601_);
v___f_1612_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___boxed), 5, 2);
lean_closure_set(v___f_1612_, 0, v_inst_1600_);
lean_closure_set(v___f_1612_, 1, v_tree_1601_);
v___f_1613_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__1));
v___x_1674_ = lean_unsigned_to_nat(0u);
v_lines_1675_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__22));
v___x_1676_ = lean_array_get_size(v_pending_1610_);
v___x_1677_ = lean_nat_dec_eq(v___x_1676_, v___x_1674_);
if (v___x_1677_ == 0)
{
lean_object* v___f_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; size_t v_sz_1681_; size_t v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; lean_object* v_lines_1689_; 
v___f_1678_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__23));
v___x_1679_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__25));
v___x_1680_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__15));
v_sz_1681_ = lean_array_size(v_pending_1610_);
v___x_1682_ = ((size_t)0ULL);
lean_inc_ref(v_pending_1610_);
v___x_1683_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_1680_, v___f_1678_, v_sz_1681_, v___x_1682_, v_pending_1610_);
v___x_1684_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__21));
v___x_1685_ = lean_array_to_list(v___x_1683_);
lean_inc_ref(v_inst_1600_);
v___x_1686_ = l_List_format___redArg(v_inst_1600_, v___x_1685_);
v___x_1687_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1687_, 0, v___x_1684_);
lean_ctor_set(v___x_1687_, 1, v___x_1686_);
v___x_1688_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1688_, 0, v___x_1679_);
lean_ctor_set(v___x_1688_, 1, v___x_1687_);
v_lines_1689_ = lean_array_push(v_lines_1675_, v___x_1688_);
v_lines_1663_ = v_lines_1689_;
goto v___jp_1662_;
}
else
{
v_lines_1663_ = v_lines_1675_;
goto v___jp_1662_;
}
v___jp_1614_:
{
lean_object* v___x_1616_; lean_object* v___x_1617_; uint8_t v___x_1618_; 
v___x_1616_ = lean_array_get_size(v___y_1615_);
v___x_1617_ = lean_unsigned_to_nat(0u);
v___x_1618_ = lean_nat_dec_eq(v___x_1616_, v___x_1617_);
if (v___x_1618_ == 0)
{
lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; 
v___x_1619_ = lean_array_to_list(v___y_1615_);
v___x_1620_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__3));
v___x_1621_ = l_Std_Format_joinSep___redArg(v___f_1613_, v___x_1619_, v___x_1620_);
return v___x_1621_;
}
else
{
lean_object* v___x_1622_; 
lean_dec_ref(v___y_1615_);
v___x_1622_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__5));
return v___x_1622_;
}
}
v___jp_1623_:
{
lean_object* v___x_1625_; lean_object* v_buckets_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; uint8_t v___x_1629_; 
v___x_1625_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__15));
v_buckets_1626_ = lean_ctor_get(v_children_1609_, 1);
lean_inc_ref(v_buckets_1626_);
lean_dec_ref(v_children_1609_);
v___x_1627_ = lean_unsigned_to_nat(0u);
v___x_1628_ = lean_array_get_size(v_buckets_1626_);
v___x_1629_ = lean_nat_dec_lt(v___x_1627_, v___x_1628_);
if (v___x_1629_ == 0)
{
lean_dec_ref(v_buckets_1626_);
lean_dec_ref(v___f_1612_);
v___y_1615_ = v___y_1624_;
goto v___jp_1614_;
}
else
{
lean_object* v___f_1630_; uint8_t v___x_1631_; 
v___f_1630_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1630_, 0, v___x_1625_);
lean_closure_set(v___f_1630_, 1, v___f_1612_);
v___x_1631_ = lean_nat_dec_le(v___x_1628_, v___x_1628_);
if (v___x_1631_ == 0)
{
if (v___x_1629_ == 0)
{
lean_dec_ref(v___f_1630_);
lean_dec_ref(v_buckets_1626_);
v___y_1615_ = v___y_1624_;
goto v___jp_1614_;
}
else
{
size_t v___x_1632_; size_t v___x_1633_; lean_object* v___x_1634_; 
v___x_1632_ = ((size_t)0ULL);
v___x_1633_ = lean_usize_of_nat(v___x_1628_);
v___x_1634_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1625_, v___f_1630_, v_buckets_1626_, v___x_1632_, v___x_1633_, v___y_1624_);
v___y_1615_ = v___x_1634_;
goto v___jp_1614_;
}
}
else
{
size_t v___x_1635_; size_t v___x_1636_; lean_object* v___x_1637_; 
v___x_1635_ = ((size_t)0ULL);
v___x_1636_ = lean_usize_of_nat(v___x_1628_);
v___x_1637_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1625_, v___f_1630_, v_buckets_1626_, v___x_1635_, v___x_1636_, v___y_1624_);
v___y_1615_ = v___x_1637_;
goto v___jp_1614_;
}
}
}
v___jp_1638_:
{
lean_object* v___x_1640_; lean_object* v_buckets_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; uint8_t v___x_1644_; 
v___x_1640_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__15));
v_buckets_1641_ = lean_ctor_get(v_labelledStars_1608_, 1);
lean_inc_ref(v_buckets_1641_);
lean_dec_ref(v_labelledStars_1608_);
v___x_1642_ = lean_unsigned_to_nat(0u);
v___x_1643_ = lean_array_get_size(v_buckets_1641_);
v___x_1644_ = lean_nat_dec_lt(v___x_1642_, v___x_1643_);
if (v___x_1644_ == 0)
{
lean_dec_ref(v_buckets_1641_);
lean_dec_ref(v___f_1611_);
v___y_1624_ = v_lines_1639_;
goto v___jp_1623_;
}
else
{
lean_object* v___f_1645_; uint8_t v___x_1646_; 
v___f_1645_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1645_, 0, v___x_1640_);
lean_closure_set(v___f_1645_, 1, v___f_1611_);
v___x_1646_ = lean_nat_dec_le(v___x_1643_, v___x_1643_);
if (v___x_1646_ == 0)
{
if (v___x_1644_ == 0)
{
lean_dec_ref(v___f_1645_);
lean_dec_ref(v_buckets_1641_);
v___y_1624_ = v_lines_1639_;
goto v___jp_1623_;
}
else
{
size_t v___x_1647_; size_t v___x_1648_; lean_object* v___x_1649_; 
v___x_1647_ = ((size_t)0ULL);
v___x_1648_ = lean_usize_of_nat(v___x_1643_);
v___x_1649_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1640_, v___f_1645_, v_buckets_1641_, v___x_1647_, v___x_1648_, v_lines_1639_);
v___y_1624_ = v___x_1649_;
goto v___jp_1623_;
}
}
else
{
size_t v___x_1650_; size_t v___x_1651_; lean_object* v___x_1652_; 
v___x_1650_ = ((size_t)0ULL);
v___x_1651_ = lean_usize_of_nat(v___x_1643_);
v___x_1652_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1640_, v___f_1645_, v_buckets_1641_, v___x_1650_, v___x_1651_, v_lines_1639_);
v___y_1624_ = v___x_1652_;
goto v___jp_1623_;
}
}
}
v___jp_1653_:
{
if (lean_obj_tag(v_star_1607_) == 1)
{
lean_object* v_val_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v_lines_1661_; 
v_val_1655_ = lean_ctor_get(v_star_1607_, 0);
lean_inc(v_val_1655_);
v___x_1656_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0, &lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0);
v___x_1657_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__18));
v___x_1658_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(v_inst_1600_, v_tree_1601_, v_val_1655_);
lean_dec(v_val_1655_);
v___x_1659_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1659_, 0, v___x_1657_);
lean_ctor_set(v___x_1659_, 1, v___x_1658_);
v___x_1660_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1660_, 0, v___x_1656_);
lean_ctor_set(v___x_1660_, 1, v___x_1659_);
v_lines_1661_ = lean_array_push(v_lines_1654_, v___x_1660_);
v_lines_1639_ = v_lines_1661_;
goto v___jp_1638_;
}
else
{
lean_dec_ref(v_tree_1601_);
lean_dec_ref(v_inst_1600_);
v_lines_1639_ = v_lines_1654_;
goto v___jp_1638_;
}
}
v___jp_1662_:
{
lean_object* v___x_1664_; lean_object* v___x_1665_; uint8_t v___x_1666_; 
v___x_1664_ = lean_array_get_size(v_values_1606_);
v___x_1665_ = lean_unsigned_to_nat(0u);
v___x_1666_ = lean_nat_dec_eq(v___x_1664_, v___x_1665_);
if (v___x_1666_ == 0)
{
lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v_lines_1673_; 
v___x_1667_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__20));
v___x_1668_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__21));
lean_inc_ref(v_values_1606_);
v___x_1669_ = lean_array_to_list(v_values_1606_);
lean_inc_ref(v_inst_1600_);
v___x_1670_ = l_List_format___redArg(v_inst_1600_, v___x_1669_);
v___x_1671_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1671_, 0, v___x_1668_);
lean_ctor_set(v___x_1671_, 1, v___x_1670_);
v___x_1672_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1672_, 0, v___x_1667_);
lean_ctor_set(v___x_1672_, 1, v___x_1671_);
v_lines_1673_ = lean_array_push(v_lines_1663_, v___x_1672_);
v_lines_1654_ = v_lines_1673_;
goto v___jp_1653_;
}
else
{
v_lines_1654_ = v_lines_1663_;
goto v___jp_1653_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1(lean_object* v_inst_1690_, lean_object* v_tree_1691_, lean_object* v_x1_1692_, lean_object* v_x2_1693_, lean_object* v_x3_1694_){
_start:
{
lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; 
v___x_1695_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0, &lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0);
v___x_1696_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_x2_1693_);
v___x_1697_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__2));
v___x_1698_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1698_, 0, v___x_1696_);
lean_ctor_set(v___x_1698_, 1, v___x_1697_);
v___x_1699_ = lean_box(1);
v___x_1700_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1700_, 0, v___x_1698_);
lean_ctor_set(v___x_1700_, 1, v___x_1699_);
v___x_1701_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(v_inst_1690_, v_tree_1691_, v_x3_1694_);
v___x_1702_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1702_, 0, v___x_1700_);
lean_ctor_set(v___x_1702_, 1, v___x_1701_);
v___x_1703_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1703_, 0, v___x_1695_);
lean_ctor_set(v___x_1703_, 1, v___x_1702_);
v___x_1704_ = lean_array_push(v_x1_1692_, v___x_1703_);
return v___x_1704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___boxed(lean_object* v_inst_1705_, lean_object* v_tree_1706_, lean_object* v_trie_1707_){
_start:
{
lean_object* v_res_1708_; 
v_res_1708_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(v_inst_1705_, v_tree_1706_, v_trie_1707_);
lean_dec(v_trie_1707_);
return v_res_1708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go(lean_object* v_00_u03b1_1709_, lean_object* v_inst_1710_, lean_object* v_tree_1711_, lean_object* v_trie_1712_){
_start:
{
lean_object* v___x_1713_; 
v___x_1713_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(v_inst_1710_, v_tree_1711_, v_trie_1712_);
return v___x_1713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___boxed(lean_object* v_00_u03b1_1714_, lean_object* v_inst_1715_, lean_object* v_tree_1716_, lean_object* v_trie_1717_){
_start:
{
lean_object* v_res_1718_; 
v_res_1718_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go(v_00_u03b1_1714_, v_inst_1715_, v_tree_1716_, v_trie_1717_);
lean_dec(v_trie_1717_);
return v_res_1718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__0(lean_object* v_inst_1719_, lean_object* v_tree_1720_, lean_object* v_x1_1721_, lean_object* v_x2_1722_, lean_object* v_x3_1723_){
_start:
{
lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; 
v___x_1724_ = lean_obj_once(&lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0, &lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0_once, _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__0);
v___x_1725_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_Key_format(v_x2_1722_);
v___x_1726_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___lam__1___closed__2));
v___x_1727_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1727_, 0, v___x_1725_);
lean_ctor_set(v___x_1727_, 1, v___x_1726_);
v___x_1728_ = lean_box(1);
v___x_1729_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1729_, 0, v___x_1727_);
lean_ctor_set(v___x_1729_, 1, v___x_1728_);
v___x_1730_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg(v_inst_1719_, v_tree_1720_, v_x3_1723_);
v___x_1731_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1731_, 0, v___x_1729_);
lean_ctor_set(v___x_1731_, 1, v___x_1730_);
v___x_1732_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1732_, 0, v___x_1724_);
lean_ctor_set(v___x_1732_, 1, v___x_1731_);
v___x_1733_ = lean_array_push(v_x1_1721_, v___x_1732_);
return v___x_1733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__0___boxed(lean_object* v_inst_1734_, lean_object* v_tree_1735_, lean_object* v_x1_1736_, lean_object* v_x2_1737_, lean_object* v_x3_1738_){
_start:
{
lean_object* v_res_1739_; 
v_res_1739_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__0(v_inst_1734_, v_tree_1735_, v_x1_1736_, v_x2_1737_, v_x3_1738_);
lean_dec(v_x3_1738_);
return v_res_1739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__1(lean_object* v___x_1740_, lean_object* v___f_1741_, lean_object* v_acc_1742_, lean_object* v_l_1743_){
_start:
{
lean_object* v___x_1744_; 
v___x_1744_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v___x_1740_, v___f_1741_, v_acc_1742_, v_l_1743_);
return v___x_1744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg(lean_object* v_inst_1751_, lean_object* v_tree_1752_){
_start:
{
lean_object* v_root_1753_; lean_object* v___x_1754_; lean_object* v_buckets_1755_; lean_object* v___x_1757_; uint8_t v_isShared_1758_; uint8_t v_isSharedCheck_1786_; 
v_root_1753_ = lean_ctor_get(v_tree_1752_, 0);
lean_inc_ref(v_root_1753_);
v___x_1754_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__15));
v_buckets_1755_ = lean_ctor_get(v_root_1753_, 1);
v_isSharedCheck_1786_ = !lean_is_exclusive(v_root_1753_);
if (v_isSharedCheck_1786_ == 0)
{
lean_object* v_unused_1787_; 
v_unused_1787_ = lean_ctor_get(v_root_1753_, 0);
lean_dec(v_unused_1787_);
v___x_1757_ = v_root_1753_;
v_isShared_1758_ = v_isSharedCheck_1786_;
goto v_resetjp_1756_;
}
else
{
lean_inc(v_buckets_1755_);
lean_dec(v_root_1753_);
v___x_1757_ = lean_box(0);
v_isShared_1758_ = v_isSharedCheck_1786_;
goto v_resetjp_1756_;
}
v_resetjp_1756_:
{
lean_object* v___f_1759_; lean_object* v___y_1761_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; uint8_t v___x_1776_; 
v___f_1759_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__1));
v___x_1773_ = lean_unsigned_to_nat(0u);
v___x_1774_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__22));
v___x_1775_ = lean_array_get_size(v_buckets_1755_);
v___x_1776_ = lean_nat_dec_lt(v___x_1773_, v___x_1775_);
if (v___x_1776_ == 0)
{
lean_dec_ref(v_buckets_1755_);
lean_dec_ref(v_tree_1752_);
lean_dec_ref(v_inst_1751_);
v___y_1761_ = v___x_1774_;
goto v___jp_1760_;
}
else
{
lean_object* v___f_1777_; lean_object* v___f_1778_; uint8_t v___x_1779_; 
v___f_1777_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__0___boxed), 5, 2);
lean_closure_set(v___f_1777_, 0, v_inst_1751_);
lean_closure_set(v___f_1777_, 1, v_tree_1752_);
v___f_1778_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1778_, 0, v___x_1754_);
lean_closure_set(v___f_1778_, 1, v___f_1777_);
v___x_1779_ = lean_nat_dec_le(v___x_1775_, v___x_1775_);
if (v___x_1779_ == 0)
{
if (v___x_1776_ == 0)
{
lean_dec_ref(v___f_1778_);
lean_dec_ref(v_buckets_1755_);
v___y_1761_ = v___x_1774_;
goto v___jp_1760_;
}
else
{
size_t v___x_1780_; size_t v___x_1781_; lean_object* v___x_1782_; 
v___x_1780_ = ((size_t)0ULL);
v___x_1781_ = lean_usize_of_nat(v___x_1775_);
v___x_1782_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1754_, v___f_1778_, v_buckets_1755_, v___x_1780_, v___x_1781_, v___x_1774_);
v___y_1761_ = v___x_1782_;
goto v___jp_1760_;
}
}
else
{
size_t v___x_1783_; size_t v___x_1784_; lean_object* v___x_1785_; 
v___x_1783_ = ((size_t)0ULL);
v___x_1784_ = lean_usize_of_nat(v___x_1775_);
v___x_1785_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1754_, v___f_1778_, v_buckets_1755_, v___x_1783_, v___x_1784_, v___x_1774_);
v___y_1761_ = v___x_1785_;
goto v___jp_1760_;
}
}
v___jp_1760_:
{
lean_object* v___x_1762_; lean_object* v___x_1763_; uint8_t v___x_1764_; 
v___x_1762_ = lean_array_get_size(v___y_1761_);
v___x_1763_ = lean_unsigned_to_nat(0u);
v___x_1764_ = lean_nat_dec_eq(v___x_1762_, v___x_1763_);
if (v___x_1764_ == 0)
{
lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1770_; 
v___x_1765_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__1));
v___x_1766_ = lean_array_to_list(v___y_1761_);
v___x_1767_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format_go___redArg___closed__3));
v___x_1768_ = l_Std_Format_joinSep___redArg(v___f_1759_, v___x_1766_, v___x_1767_);
if (v_isShared_1758_ == 0)
{
lean_ctor_set_tag(v___x_1757_, 5);
lean_ctor_set(v___x_1757_, 1, v___x_1768_);
lean_ctor_set(v___x_1757_, 0, v___x_1765_);
v___x_1770_ = v___x_1757_;
goto v_reusejp_1769_;
}
else
{
lean_object* v_reuseFailAlloc_1771_; 
v_reuseFailAlloc_1771_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1771_, 0, v___x_1765_);
lean_ctor_set(v_reuseFailAlloc_1771_, 1, v___x_1768_);
v___x_1770_ = v_reuseFailAlloc_1771_;
goto v_reusejp_1769_;
}
v_reusejp_1769_:
{
return v___x_1770_;
}
}
else
{
lean_object* v___x_1772_; 
lean_dec_ref(v___y_1761_);
lean_del_object(v___x_1757_);
v___x_1772_ = ((lean_object*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg___closed__3));
return v___x_1772_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_format(lean_object* v_00_u03b1_1788_, lean_object* v_inst_1789_, lean_object* v_tree_1790_){
_start:
{
lean_object* v___x_1791_; 
v___x_1791_ = lp_mathlib_Lean_Meta_RefinedDiscrTree_format___redArg(v_inst_1789_, v_tree_1790_);
return v___x_1791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormat___redArg(lean_object* v_inst_1792_){
_start:
{
lean_object* v___x_1793_; 
v___x_1793_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format), 3, 2);
lean_closure_set(v___x_1793_, 0, lean_box(0));
lean_closure_set(v___x_1793_, 1, v_inst_1792_);
return v___x_1793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_RefinedDiscrTree_instToFormat(lean_object* v_00_u03b1_1794_, lean_object* v_inst_1795_){
_start:
{
lean_object* v___x_1796_; 
v___x_1796_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_RefinedDiscrTree_format), 3, 2);
lean_closure_set(v___x_1796_, 0, lean_box(0));
lean_closure_set(v___x_1796_, 1, v_inst_1795_);
return v___x_1796_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin) {
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
lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey_default = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey_default();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey_default);
lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedKey);
lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry_default);
lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry = _init_lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry();
lean_mark_persistent(lp_mathlib_Lean_Meta_RefinedDiscrTree_instInhabitedLazyEntry);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta_RefinedDiscrTree_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
