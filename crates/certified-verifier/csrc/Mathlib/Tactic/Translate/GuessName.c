// Lean compiler output
// Module: Mathlib.Tactic.Translate.GuessName
// Imports: public import Init public meta import Init public meta import Std.Data.TreeMap.Basic public meta import Mathlib.Data.String.Defs public import Mathlib.Init
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
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_string_utf8_set(lean_object*, lean_object*, uint32_t);
lean_object* l_Char_utf8Size(uint32_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_le(uint32_t, uint32_t);
uint32_t lean_uint32_add(uint32_t, uint32_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_string_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
lean_object* lean_string_utf8_set(lean_object*, lean_object*, uint32_t);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
uint8_t lean_string_utf8_at_end(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
uint8_t lean_string_compare(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_get_x21Internal___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_registerEnvExtension___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_String_mapTokens(uint32_t, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_EnvExtension_modifyState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_get_x3f(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "WF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Coe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "TC"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "T"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HTCT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__26;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_String_splitCase(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_String_decapitalizeSeq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeLike___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeFirstLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeFirstLike___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_mapAux___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_applyNameDict(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_applyNameDict___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_GuessName_fixAbbreviationAux_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_GuessName_fixAbbreviationAux_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviation(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviation___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___lam__0(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__9___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "` should be capitalized"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_box(0);
v___x_2_ = lean_unsigned_to_nat(16u);
v___x_3_ = lean_mk_array(v___x_2_, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__1(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__0, &lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__0);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__2(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__1, &lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__1);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default(void){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__2, &lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default___closed__2);
return v___x_9_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default;
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0___redArg(lean_object* v_k_11_, lean_object* v_v_12_, lean_object* v_t_13_){
_start:
{
if (lean_obj_tag(v_t_13_) == 0)
{
lean_object* v_size_14_; lean_object* v_k_15_; lean_object* v_v_16_; lean_object* v_l_17_; lean_object* v_r_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_298_; 
v_size_14_ = lean_ctor_get(v_t_13_, 0);
v_k_15_ = lean_ctor_get(v_t_13_, 1);
v_v_16_ = lean_ctor_get(v_t_13_, 2);
v_l_17_ = lean_ctor_get(v_t_13_, 3);
v_r_18_ = lean_ctor_get(v_t_13_, 4);
v_isSharedCheck_298_ = !lean_is_exclusive(v_t_13_);
if (v_isSharedCheck_298_ == 0)
{
v___x_20_ = v_t_13_;
v_isShared_21_ = v_isSharedCheck_298_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_r_18_);
lean_inc(v_l_17_);
lean_inc(v_v_16_);
lean_inc(v_k_15_);
lean_inc(v_size_14_);
lean_dec(v_t_13_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_298_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
uint8_t v___x_22_; 
v___x_22_ = lean_string_compare(v_k_11_, v_k_15_);
switch(v___x_22_)
{
case 0:
{
lean_object* v_impl_23_; lean_object* v___x_24_; 
lean_dec(v_size_14_);
v_impl_23_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0___redArg(v_k_11_, v_v_12_, v_l_17_);
v___x_24_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_18_) == 0)
{
lean_object* v_size_25_; lean_object* v_size_26_; lean_object* v_k_27_; lean_object* v_v_28_; lean_object* v_l_29_; lean_object* v_r_30_; lean_object* v___x_31_; lean_object* v___x_32_; uint8_t v___x_33_; 
v_size_25_ = lean_ctor_get(v_r_18_, 0);
v_size_26_ = lean_ctor_get(v_impl_23_, 0);
lean_inc(v_size_26_);
v_k_27_ = lean_ctor_get(v_impl_23_, 1);
lean_inc(v_k_27_);
v_v_28_ = lean_ctor_get(v_impl_23_, 2);
lean_inc(v_v_28_);
v_l_29_ = lean_ctor_get(v_impl_23_, 3);
lean_inc(v_l_29_);
v_r_30_ = lean_ctor_get(v_impl_23_, 4);
lean_inc(v_r_30_);
v___x_31_ = lean_unsigned_to_nat(3u);
v___x_32_ = lean_nat_mul(v___x_31_, v_size_25_);
v___x_33_ = lean_nat_dec_lt(v___x_32_, v_size_26_);
lean_dec(v___x_32_);
if (v___x_33_ == 0)
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_37_; 
lean_dec(v_r_30_);
lean_dec(v_l_29_);
lean_dec(v_v_28_);
lean_dec(v_k_27_);
v___x_34_ = lean_nat_add(v___x_24_, v_size_26_);
lean_dec(v_size_26_);
v___x_35_ = lean_nat_add(v___x_34_, v_size_25_);
lean_dec(v___x_34_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 3, v_impl_23_);
lean_ctor_set(v___x_20_, 0, v___x_35_);
v___x_37_ = v___x_20_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
lean_ctor_set(v_reuseFailAlloc_38_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_38_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_38_, 3, v_impl_23_);
lean_ctor_set(v_reuseFailAlloc_38_, 4, v_r_18_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
else
{
lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_104_; 
v_isSharedCheck_104_ = !lean_is_exclusive(v_impl_23_);
if (v_isSharedCheck_104_ == 0)
{
lean_object* v_unused_105_; lean_object* v_unused_106_; lean_object* v_unused_107_; lean_object* v_unused_108_; lean_object* v_unused_109_; 
v_unused_105_ = lean_ctor_get(v_impl_23_, 4);
lean_dec(v_unused_105_);
v_unused_106_ = lean_ctor_get(v_impl_23_, 3);
lean_dec(v_unused_106_);
v_unused_107_ = lean_ctor_get(v_impl_23_, 2);
lean_dec(v_unused_107_);
v_unused_108_ = lean_ctor_get(v_impl_23_, 1);
lean_dec(v_unused_108_);
v_unused_109_ = lean_ctor_get(v_impl_23_, 0);
lean_dec(v_unused_109_);
v___x_40_ = v_impl_23_;
v_isShared_41_ = v_isSharedCheck_104_;
goto v_resetjp_39_;
}
else
{
lean_dec(v_impl_23_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_104_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v_size_42_; lean_object* v_size_43_; lean_object* v_k_44_; lean_object* v_v_45_; lean_object* v_l_46_; lean_object* v_r_47_; lean_object* v___x_48_; lean_object* v___x_49_; uint8_t v___x_50_; 
v_size_42_ = lean_ctor_get(v_l_29_, 0);
v_size_43_ = lean_ctor_get(v_r_30_, 0);
v_k_44_ = lean_ctor_get(v_r_30_, 1);
v_v_45_ = lean_ctor_get(v_r_30_, 2);
v_l_46_ = lean_ctor_get(v_r_30_, 3);
v_r_47_ = lean_ctor_get(v_r_30_, 4);
v___x_48_ = lean_unsigned_to_nat(2u);
v___x_49_ = lean_nat_mul(v___x_48_, v_size_42_);
v___x_50_ = lean_nat_dec_lt(v_size_43_, v___x_49_);
lean_dec(v___x_49_);
if (v___x_50_ == 0)
{
lean_object* v___x_52_; uint8_t v_isShared_53_; uint8_t v_isSharedCheck_79_; 
lean_inc(v_r_47_);
lean_inc(v_l_46_);
lean_inc(v_v_45_);
lean_inc(v_k_44_);
v_isSharedCheck_79_ = !lean_is_exclusive(v_r_30_);
if (v_isSharedCheck_79_ == 0)
{
lean_object* v_unused_80_; lean_object* v_unused_81_; lean_object* v_unused_82_; lean_object* v_unused_83_; lean_object* v_unused_84_; 
v_unused_80_ = lean_ctor_get(v_r_30_, 4);
lean_dec(v_unused_80_);
v_unused_81_ = lean_ctor_get(v_r_30_, 3);
lean_dec(v_unused_81_);
v_unused_82_ = lean_ctor_get(v_r_30_, 2);
lean_dec(v_unused_82_);
v_unused_83_ = lean_ctor_get(v_r_30_, 1);
lean_dec(v_unused_83_);
v_unused_84_ = lean_ctor_get(v_r_30_, 0);
lean_dec(v_unused_84_);
v___x_52_ = v_r_30_;
v_isShared_53_ = v_isSharedCheck_79_;
goto v_resetjp_51_;
}
else
{
lean_dec(v_r_30_);
v___x_52_ = lean_box(0);
v_isShared_53_ = v_isSharedCheck_79_;
goto v_resetjp_51_;
}
v_resetjp_51_:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___y_57_; lean_object* v___y_58_; lean_object* v___y_59_; lean_object* v___x_67_; lean_object* v___y_69_; 
v___x_54_ = lean_nat_add(v___x_24_, v_size_26_);
lean_dec(v_size_26_);
v___x_55_ = lean_nat_add(v___x_54_, v_size_25_);
lean_dec(v___x_54_);
v___x_67_ = lean_nat_add(v___x_24_, v_size_42_);
if (lean_obj_tag(v_l_46_) == 0)
{
lean_object* v_size_77_; 
v_size_77_ = lean_ctor_get(v_l_46_, 0);
lean_inc(v_size_77_);
v___y_69_ = v_size_77_;
goto v___jp_68_;
}
else
{
lean_object* v___x_78_; 
v___x_78_ = lean_unsigned_to_nat(0u);
v___y_69_ = v___x_78_;
goto v___jp_68_;
}
v___jp_56_:
{
lean_object* v___x_60_; lean_object* v___x_62_; 
v___x_60_ = lean_nat_add(v___y_58_, v___y_59_);
lean_dec(v___y_59_);
lean_dec(v___y_58_);
if (v_isShared_53_ == 0)
{
lean_ctor_set(v___x_52_, 4, v_r_18_);
lean_ctor_set(v___x_52_, 3, v_r_47_);
lean_ctor_set(v___x_52_, 2, v_v_16_);
lean_ctor_set(v___x_52_, 1, v_k_15_);
lean_ctor_set(v___x_52_, 0, v___x_60_);
v___x_62_ = v___x_52_;
goto v_reusejp_61_;
}
else
{
lean_object* v_reuseFailAlloc_66_; 
v_reuseFailAlloc_66_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_66_, 0, v___x_60_);
lean_ctor_set(v_reuseFailAlloc_66_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_66_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_66_, 3, v_r_47_);
lean_ctor_set(v_reuseFailAlloc_66_, 4, v_r_18_);
v___x_62_ = v_reuseFailAlloc_66_;
goto v_reusejp_61_;
}
v_reusejp_61_:
{
lean_object* v___x_64_; 
if (v_isShared_41_ == 0)
{
lean_ctor_set(v___x_40_, 4, v___x_62_);
lean_ctor_set(v___x_40_, 3, v___y_57_);
lean_ctor_set(v___x_40_, 2, v_v_45_);
lean_ctor_set(v___x_40_, 1, v_k_44_);
lean_ctor_set(v___x_40_, 0, v___x_55_);
v___x_64_ = v___x_40_;
goto v_reusejp_63_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v___x_55_);
lean_ctor_set(v_reuseFailAlloc_65_, 1, v_k_44_);
lean_ctor_set(v_reuseFailAlloc_65_, 2, v_v_45_);
lean_ctor_set(v_reuseFailAlloc_65_, 3, v___y_57_);
lean_ctor_set(v_reuseFailAlloc_65_, 4, v___x_62_);
v___x_64_ = v_reuseFailAlloc_65_;
goto v_reusejp_63_;
}
v_reusejp_63_:
{
return v___x_64_;
}
}
}
v___jp_68_:
{
lean_object* v___x_70_; lean_object* v___x_72_; 
v___x_70_ = lean_nat_add(v___x_67_, v___y_69_);
lean_dec(v___y_69_);
lean_dec(v___x_67_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v_l_46_);
lean_ctor_set(v___x_20_, 3, v_l_29_);
lean_ctor_set(v___x_20_, 2, v_v_28_);
lean_ctor_set(v___x_20_, 1, v_k_27_);
lean_ctor_set(v___x_20_, 0, v___x_70_);
v___x_72_ = v___x_20_;
goto v_reusejp_71_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v___x_70_);
lean_ctor_set(v_reuseFailAlloc_76_, 1, v_k_27_);
lean_ctor_set(v_reuseFailAlloc_76_, 2, v_v_28_);
lean_ctor_set(v_reuseFailAlloc_76_, 3, v_l_29_);
lean_ctor_set(v_reuseFailAlloc_76_, 4, v_l_46_);
v___x_72_ = v_reuseFailAlloc_76_;
goto v_reusejp_71_;
}
v_reusejp_71_:
{
lean_object* v___x_73_; 
v___x_73_ = lean_nat_add(v___x_24_, v_size_25_);
if (lean_obj_tag(v_r_47_) == 0)
{
lean_object* v_size_74_; 
v_size_74_ = lean_ctor_get(v_r_47_, 0);
lean_inc(v_size_74_);
v___y_57_ = v___x_72_;
v___y_58_ = v___x_73_;
v___y_59_ = v_size_74_;
goto v___jp_56_;
}
else
{
lean_object* v___x_75_; 
v___x_75_ = lean_unsigned_to_nat(0u);
v___y_57_ = v___x_72_;
v___y_58_ = v___x_73_;
v___y_59_ = v___x_75_;
goto v___jp_56_;
}
}
}
}
}
else
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_90_; 
lean_del_object(v___x_20_);
v___x_85_ = lean_nat_add(v___x_24_, v_size_26_);
lean_dec(v_size_26_);
v___x_86_ = lean_nat_add(v___x_85_, v_size_25_);
lean_dec(v___x_85_);
v___x_87_ = lean_nat_add(v___x_24_, v_size_25_);
v___x_88_ = lean_nat_add(v___x_87_, v_size_43_);
lean_dec(v___x_87_);
lean_inc_ref(v_r_18_);
if (v_isShared_41_ == 0)
{
lean_ctor_set(v___x_40_, 4, v_r_18_);
lean_ctor_set(v___x_40_, 3, v_r_30_);
lean_ctor_set(v___x_40_, 2, v_v_16_);
lean_ctor_set(v___x_40_, 1, v_k_15_);
lean_ctor_set(v___x_40_, 0, v___x_88_);
v___x_90_ = v___x_40_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_103_; 
v_reuseFailAlloc_103_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_103_, 0, v___x_88_);
lean_ctor_set(v_reuseFailAlloc_103_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_103_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_103_, 3, v_r_30_);
lean_ctor_set(v_reuseFailAlloc_103_, 4, v_r_18_);
v___x_90_ = v_reuseFailAlloc_103_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_97_; 
v_isSharedCheck_97_ = !lean_is_exclusive(v_r_18_);
if (v_isSharedCheck_97_ == 0)
{
lean_object* v_unused_98_; lean_object* v_unused_99_; lean_object* v_unused_100_; lean_object* v_unused_101_; lean_object* v_unused_102_; 
v_unused_98_ = lean_ctor_get(v_r_18_, 4);
lean_dec(v_unused_98_);
v_unused_99_ = lean_ctor_get(v_r_18_, 3);
lean_dec(v_unused_99_);
v_unused_100_ = lean_ctor_get(v_r_18_, 2);
lean_dec(v_unused_100_);
v_unused_101_ = lean_ctor_get(v_r_18_, 1);
lean_dec(v_unused_101_);
v_unused_102_ = lean_ctor_get(v_r_18_, 0);
lean_dec(v_unused_102_);
v___x_92_ = v_r_18_;
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
else
{
lean_dec(v_r_18_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_95_; 
if (v_isShared_93_ == 0)
{
lean_ctor_set(v___x_92_, 4, v___x_90_);
lean_ctor_set(v___x_92_, 3, v_l_29_);
lean_ctor_set(v___x_92_, 2, v_v_28_);
lean_ctor_set(v___x_92_, 1, v_k_27_);
lean_ctor_set(v___x_92_, 0, v___x_86_);
v___x_95_ = v___x_92_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v___x_86_);
lean_ctor_set(v_reuseFailAlloc_96_, 1, v_k_27_);
lean_ctor_set(v_reuseFailAlloc_96_, 2, v_v_28_);
lean_ctor_set(v_reuseFailAlloc_96_, 3, v_l_29_);
lean_ctor_set(v_reuseFailAlloc_96_, 4, v___x_90_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_110_; 
v_l_110_ = lean_ctor_get(v_impl_23_, 3);
lean_inc(v_l_110_);
if (lean_obj_tag(v_l_110_) == 0)
{
lean_object* v_r_111_; lean_object* v_k_112_; lean_object* v_v_113_; lean_object* v___x_115_; uint8_t v_isShared_116_; uint8_t v_isSharedCheck_124_; 
v_r_111_ = lean_ctor_get(v_impl_23_, 4);
v_k_112_ = lean_ctor_get(v_impl_23_, 1);
v_v_113_ = lean_ctor_get(v_impl_23_, 2);
v_isSharedCheck_124_ = !lean_is_exclusive(v_impl_23_);
if (v_isSharedCheck_124_ == 0)
{
lean_object* v_unused_125_; lean_object* v_unused_126_; 
v_unused_125_ = lean_ctor_get(v_impl_23_, 3);
lean_dec(v_unused_125_);
v_unused_126_ = lean_ctor_get(v_impl_23_, 0);
lean_dec(v_unused_126_);
v___x_115_ = v_impl_23_;
v_isShared_116_ = v_isSharedCheck_124_;
goto v_resetjp_114_;
}
else
{
lean_inc(v_r_111_);
lean_inc(v_v_113_);
lean_inc(v_k_112_);
lean_dec(v_impl_23_);
v___x_115_ = lean_box(0);
v_isShared_116_ = v_isSharedCheck_124_;
goto v_resetjp_114_;
}
v_resetjp_114_:
{
lean_object* v___x_117_; lean_object* v___x_119_; 
v___x_117_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_111_);
if (v_isShared_116_ == 0)
{
lean_ctor_set(v___x_115_, 3, v_r_111_);
lean_ctor_set(v___x_115_, 2, v_v_16_);
lean_ctor_set(v___x_115_, 1, v_k_15_);
lean_ctor_set(v___x_115_, 0, v___x_24_);
v___x_119_ = v___x_115_;
goto v_reusejp_118_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v___x_24_);
lean_ctor_set(v_reuseFailAlloc_123_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_123_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_123_, 3, v_r_111_);
lean_ctor_set(v_reuseFailAlloc_123_, 4, v_r_111_);
v___x_119_ = v_reuseFailAlloc_123_;
goto v_reusejp_118_;
}
v_reusejp_118_:
{
lean_object* v___x_121_; 
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v___x_119_);
lean_ctor_set(v___x_20_, 3, v_l_110_);
lean_ctor_set(v___x_20_, 2, v_v_113_);
lean_ctor_set(v___x_20_, 1, v_k_112_);
lean_ctor_set(v___x_20_, 0, v___x_117_);
v___x_121_ = v___x_20_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_122_; 
v_reuseFailAlloc_122_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_122_, 0, v___x_117_);
lean_ctor_set(v_reuseFailAlloc_122_, 1, v_k_112_);
lean_ctor_set(v_reuseFailAlloc_122_, 2, v_v_113_);
lean_ctor_set(v_reuseFailAlloc_122_, 3, v_l_110_);
lean_ctor_set(v_reuseFailAlloc_122_, 4, v___x_119_);
v___x_121_ = v_reuseFailAlloc_122_;
goto v_reusejp_120_;
}
v_reusejp_120_:
{
return v___x_121_;
}
}
}
}
else
{
lean_object* v_r_127_; 
v_r_127_ = lean_ctor_get(v_impl_23_, 4);
lean_inc(v_r_127_);
if (lean_obj_tag(v_r_127_) == 0)
{
lean_object* v_k_128_; lean_object* v_v_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_152_; 
v_k_128_ = lean_ctor_get(v_impl_23_, 1);
v_v_129_ = lean_ctor_get(v_impl_23_, 2);
v_isSharedCheck_152_ = !lean_is_exclusive(v_impl_23_);
if (v_isSharedCheck_152_ == 0)
{
lean_object* v_unused_153_; lean_object* v_unused_154_; lean_object* v_unused_155_; 
v_unused_153_ = lean_ctor_get(v_impl_23_, 4);
lean_dec(v_unused_153_);
v_unused_154_ = lean_ctor_get(v_impl_23_, 3);
lean_dec(v_unused_154_);
v_unused_155_ = lean_ctor_get(v_impl_23_, 0);
lean_dec(v_unused_155_);
v___x_131_ = v_impl_23_;
v_isShared_132_ = v_isSharedCheck_152_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_v_129_);
lean_inc(v_k_128_);
lean_dec(v_impl_23_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_152_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
lean_object* v_k_133_; lean_object* v_v_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_148_; 
v_k_133_ = lean_ctor_get(v_r_127_, 1);
v_v_134_ = lean_ctor_get(v_r_127_, 2);
v_isSharedCheck_148_ = !lean_is_exclusive(v_r_127_);
if (v_isSharedCheck_148_ == 0)
{
lean_object* v_unused_149_; lean_object* v_unused_150_; lean_object* v_unused_151_; 
v_unused_149_ = lean_ctor_get(v_r_127_, 4);
lean_dec(v_unused_149_);
v_unused_150_ = lean_ctor_get(v_r_127_, 3);
lean_dec(v_unused_150_);
v_unused_151_ = lean_ctor_get(v_r_127_, 0);
lean_dec(v_unused_151_);
v___x_136_ = v_r_127_;
v_isShared_137_ = v_isSharedCheck_148_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_v_134_);
lean_inc(v_k_133_);
lean_dec(v_r_127_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_148_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v___x_138_; lean_object* v___x_140_; 
v___x_138_ = lean_unsigned_to_nat(3u);
if (v_isShared_137_ == 0)
{
lean_ctor_set(v___x_136_, 4, v_l_110_);
lean_ctor_set(v___x_136_, 3, v_l_110_);
lean_ctor_set(v___x_136_, 2, v_v_129_);
lean_ctor_set(v___x_136_, 1, v_k_128_);
lean_ctor_set(v___x_136_, 0, v___x_24_);
v___x_140_ = v___x_136_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_147_; 
v_reuseFailAlloc_147_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_147_, 0, v___x_24_);
lean_ctor_set(v_reuseFailAlloc_147_, 1, v_k_128_);
lean_ctor_set(v_reuseFailAlloc_147_, 2, v_v_129_);
lean_ctor_set(v_reuseFailAlloc_147_, 3, v_l_110_);
lean_ctor_set(v_reuseFailAlloc_147_, 4, v_l_110_);
v___x_140_ = v_reuseFailAlloc_147_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
lean_object* v___x_142_; 
if (v_isShared_132_ == 0)
{
lean_ctor_set(v___x_131_, 4, v_l_110_);
lean_ctor_set(v___x_131_, 2, v_v_16_);
lean_ctor_set(v___x_131_, 1, v_k_15_);
lean_ctor_set(v___x_131_, 0, v___x_24_);
v___x_142_ = v___x_131_;
goto v_reusejp_141_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v___x_24_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_146_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_146_, 3, v_l_110_);
lean_ctor_set(v_reuseFailAlloc_146_, 4, v_l_110_);
v___x_142_ = v_reuseFailAlloc_146_;
goto v_reusejp_141_;
}
v_reusejp_141_:
{
lean_object* v___x_144_; 
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v___x_142_);
lean_ctor_set(v___x_20_, 3, v___x_140_);
lean_ctor_set(v___x_20_, 2, v_v_134_);
lean_ctor_set(v___x_20_, 1, v_k_133_);
lean_ctor_set(v___x_20_, 0, v___x_138_);
v___x_144_ = v___x_20_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v___x_138_);
lean_ctor_set(v_reuseFailAlloc_145_, 1, v_k_133_);
lean_ctor_set(v_reuseFailAlloc_145_, 2, v_v_134_);
lean_ctor_set(v_reuseFailAlloc_145_, 3, v___x_140_);
lean_ctor_set(v_reuseFailAlloc_145_, 4, v___x_142_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
}
}
}
else
{
lean_object* v___x_156_; lean_object* v___x_158_; 
v___x_156_ = lean_unsigned_to_nat(2u);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v_r_127_);
lean_ctor_set(v___x_20_, 3, v_impl_23_);
lean_ctor_set(v___x_20_, 0, v___x_156_);
v___x_158_ = v___x_20_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v___x_156_);
lean_ctor_set(v_reuseFailAlloc_159_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_159_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_159_, 3, v_impl_23_);
lean_ctor_set(v_reuseFailAlloc_159_, 4, v_r_127_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
}
}
}
case 1:
{
lean_object* v___x_161_; 
lean_dec(v_v_16_);
lean_dec(v_k_15_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 2, v_v_12_);
lean_ctor_set(v___x_20_, 1, v_k_11_);
v___x_161_ = v___x_20_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_size_14_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v_k_11_);
lean_ctor_set(v_reuseFailAlloc_162_, 2, v_v_12_);
lean_ctor_set(v_reuseFailAlloc_162_, 3, v_l_17_);
lean_ctor_set(v_reuseFailAlloc_162_, 4, v_r_18_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
default: 
{
lean_object* v_impl_163_; lean_object* v___x_164_; 
lean_dec(v_size_14_);
v_impl_163_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0___redArg(v_k_11_, v_v_12_, v_r_18_);
v___x_164_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_17_) == 0)
{
lean_object* v_size_165_; lean_object* v_size_166_; lean_object* v_k_167_; lean_object* v_v_168_; lean_object* v_l_169_; lean_object* v_r_170_; lean_object* v___x_171_; lean_object* v___x_172_; uint8_t v___x_173_; 
v_size_165_ = lean_ctor_get(v_l_17_, 0);
v_size_166_ = lean_ctor_get(v_impl_163_, 0);
lean_inc(v_size_166_);
v_k_167_ = lean_ctor_get(v_impl_163_, 1);
lean_inc(v_k_167_);
v_v_168_ = lean_ctor_get(v_impl_163_, 2);
lean_inc(v_v_168_);
v_l_169_ = lean_ctor_get(v_impl_163_, 3);
lean_inc(v_l_169_);
v_r_170_ = lean_ctor_get(v_impl_163_, 4);
lean_inc(v_r_170_);
v___x_171_ = lean_unsigned_to_nat(3u);
v___x_172_ = lean_nat_mul(v___x_171_, v_size_165_);
v___x_173_ = lean_nat_dec_lt(v___x_172_, v_size_166_);
lean_dec(v___x_172_);
if (v___x_173_ == 0)
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_177_; 
lean_dec(v_r_170_);
lean_dec(v_l_169_);
lean_dec(v_v_168_);
lean_dec(v_k_167_);
v___x_174_ = lean_nat_add(v___x_164_, v_size_165_);
v___x_175_ = lean_nat_add(v___x_174_, v_size_166_);
lean_dec(v_size_166_);
lean_dec(v___x_174_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v_impl_163_);
lean_ctor_set(v___x_20_, 0, v___x_175_);
v___x_177_ = v___x_20_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v___x_175_);
lean_ctor_set(v_reuseFailAlloc_178_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_178_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_178_, 3, v_l_17_);
lean_ctor_set(v_reuseFailAlloc_178_, 4, v_impl_163_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
else
{
lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_242_; 
v_isSharedCheck_242_ = !lean_is_exclusive(v_impl_163_);
if (v_isSharedCheck_242_ == 0)
{
lean_object* v_unused_243_; lean_object* v_unused_244_; lean_object* v_unused_245_; lean_object* v_unused_246_; lean_object* v_unused_247_; 
v_unused_243_ = lean_ctor_get(v_impl_163_, 4);
lean_dec(v_unused_243_);
v_unused_244_ = lean_ctor_get(v_impl_163_, 3);
lean_dec(v_unused_244_);
v_unused_245_ = lean_ctor_get(v_impl_163_, 2);
lean_dec(v_unused_245_);
v_unused_246_ = lean_ctor_get(v_impl_163_, 1);
lean_dec(v_unused_246_);
v_unused_247_ = lean_ctor_get(v_impl_163_, 0);
lean_dec(v_unused_247_);
v___x_180_ = v_impl_163_;
v_isShared_181_ = v_isSharedCheck_242_;
goto v_resetjp_179_;
}
else
{
lean_dec(v_impl_163_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_242_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v_size_182_; lean_object* v_k_183_; lean_object* v_v_184_; lean_object* v_l_185_; lean_object* v_r_186_; lean_object* v_size_187_; lean_object* v___x_188_; lean_object* v___x_189_; uint8_t v___x_190_; 
v_size_182_ = lean_ctor_get(v_l_169_, 0);
v_k_183_ = lean_ctor_get(v_l_169_, 1);
v_v_184_ = lean_ctor_get(v_l_169_, 2);
v_l_185_ = lean_ctor_get(v_l_169_, 3);
v_r_186_ = lean_ctor_get(v_l_169_, 4);
v_size_187_ = lean_ctor_get(v_r_170_, 0);
v___x_188_ = lean_unsigned_to_nat(2u);
v___x_189_ = lean_nat_mul(v___x_188_, v_size_187_);
v___x_190_ = lean_nat_dec_lt(v_size_182_, v___x_189_);
lean_dec(v___x_189_);
if (v___x_190_ == 0)
{
lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_218_; 
lean_inc(v_r_186_);
lean_inc(v_l_185_);
lean_inc(v_v_184_);
lean_inc(v_k_183_);
v_isSharedCheck_218_ = !lean_is_exclusive(v_l_169_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; lean_object* v_unused_220_; lean_object* v_unused_221_; lean_object* v_unused_222_; lean_object* v_unused_223_; 
v_unused_219_ = lean_ctor_get(v_l_169_, 4);
lean_dec(v_unused_219_);
v_unused_220_ = lean_ctor_get(v_l_169_, 3);
lean_dec(v_unused_220_);
v_unused_221_ = lean_ctor_get(v_l_169_, 2);
lean_dec(v_unused_221_);
v_unused_222_ = lean_ctor_get(v_l_169_, 1);
lean_dec(v_unused_222_);
v_unused_223_ = lean_ctor_get(v_l_169_, 0);
lean_dec(v_unused_223_);
v___x_192_ = v_l_169_;
v_isShared_193_ = v_isSharedCheck_218_;
goto v_resetjp_191_;
}
else
{
lean_dec(v_l_169_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_218_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___y_197_; lean_object* v___y_198_; lean_object* v___y_199_; lean_object* v___y_208_; 
v___x_194_ = lean_nat_add(v___x_164_, v_size_165_);
v___x_195_ = lean_nat_add(v___x_194_, v_size_166_);
lean_dec(v_size_166_);
if (lean_obj_tag(v_l_185_) == 0)
{
lean_object* v_size_216_; 
v_size_216_ = lean_ctor_get(v_l_185_, 0);
lean_inc(v_size_216_);
v___y_208_ = v_size_216_;
goto v___jp_207_;
}
else
{
lean_object* v___x_217_; 
v___x_217_ = lean_unsigned_to_nat(0u);
v___y_208_ = v___x_217_;
goto v___jp_207_;
}
v___jp_196_:
{
lean_object* v___x_200_; lean_object* v___x_202_; 
v___x_200_ = lean_nat_add(v___y_197_, v___y_199_);
lean_dec(v___y_199_);
lean_dec(v___y_197_);
if (v_isShared_193_ == 0)
{
lean_ctor_set(v___x_192_, 4, v_r_170_);
lean_ctor_set(v___x_192_, 3, v_r_186_);
lean_ctor_set(v___x_192_, 2, v_v_168_);
lean_ctor_set(v___x_192_, 1, v_k_167_);
lean_ctor_set(v___x_192_, 0, v___x_200_);
v___x_202_ = v___x_192_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v___x_200_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v_k_167_);
lean_ctor_set(v_reuseFailAlloc_206_, 2, v_v_168_);
lean_ctor_set(v_reuseFailAlloc_206_, 3, v_r_186_);
lean_ctor_set(v_reuseFailAlloc_206_, 4, v_r_170_);
v___x_202_ = v_reuseFailAlloc_206_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
lean_object* v___x_204_; 
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 4, v___x_202_);
lean_ctor_set(v___x_180_, 3, v___y_198_);
lean_ctor_set(v___x_180_, 2, v_v_184_);
lean_ctor_set(v___x_180_, 1, v_k_183_);
lean_ctor_set(v___x_180_, 0, v___x_195_);
v___x_204_ = v___x_180_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_205_; 
v_reuseFailAlloc_205_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_205_, 0, v___x_195_);
lean_ctor_set(v_reuseFailAlloc_205_, 1, v_k_183_);
lean_ctor_set(v_reuseFailAlloc_205_, 2, v_v_184_);
lean_ctor_set(v_reuseFailAlloc_205_, 3, v___y_198_);
lean_ctor_set(v_reuseFailAlloc_205_, 4, v___x_202_);
v___x_204_ = v_reuseFailAlloc_205_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
return v___x_204_;
}
}
}
v___jp_207_:
{
lean_object* v___x_209_; lean_object* v___x_211_; 
v___x_209_ = lean_nat_add(v___x_194_, v___y_208_);
lean_dec(v___y_208_);
lean_dec(v___x_194_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v_l_185_);
lean_ctor_set(v___x_20_, 0, v___x_209_);
v___x_211_ = v___x_20_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v___x_209_);
lean_ctor_set(v_reuseFailAlloc_215_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_215_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_215_, 3, v_l_17_);
lean_ctor_set(v_reuseFailAlloc_215_, 4, v_l_185_);
v___x_211_ = v_reuseFailAlloc_215_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_212_; 
v___x_212_ = lean_nat_add(v___x_164_, v_size_187_);
if (lean_obj_tag(v_r_186_) == 0)
{
lean_object* v_size_213_; 
v_size_213_ = lean_ctor_get(v_r_186_, 0);
lean_inc(v_size_213_);
v___y_197_ = v___x_212_;
v___y_198_ = v___x_211_;
v___y_199_ = v_size_213_;
goto v___jp_196_;
}
else
{
lean_object* v___x_214_; 
v___x_214_ = lean_unsigned_to_nat(0u);
v___y_197_ = v___x_212_;
v___y_198_ = v___x_211_;
v___y_199_ = v___x_214_;
goto v___jp_196_;
}
}
}
}
}
else
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_228_; 
lean_del_object(v___x_20_);
v___x_224_ = lean_nat_add(v___x_164_, v_size_165_);
v___x_225_ = lean_nat_add(v___x_224_, v_size_166_);
lean_dec(v_size_166_);
v___x_226_ = lean_nat_add(v___x_224_, v_size_182_);
lean_dec(v___x_224_);
lean_inc_ref(v_l_17_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 4, v_l_169_);
lean_ctor_set(v___x_180_, 3, v_l_17_);
lean_ctor_set(v___x_180_, 2, v_v_16_);
lean_ctor_set(v___x_180_, 1, v_k_15_);
lean_ctor_set(v___x_180_, 0, v___x_226_);
v___x_228_ = v___x_180_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v___x_226_);
lean_ctor_set(v_reuseFailAlloc_241_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_241_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_241_, 3, v_l_17_);
lean_ctor_set(v_reuseFailAlloc_241_, 4, v_l_169_);
v___x_228_ = v_reuseFailAlloc_241_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
v_isSharedCheck_235_ = !lean_is_exclusive(v_l_17_);
if (v_isSharedCheck_235_ == 0)
{
lean_object* v_unused_236_; lean_object* v_unused_237_; lean_object* v_unused_238_; lean_object* v_unused_239_; lean_object* v_unused_240_; 
v_unused_236_ = lean_ctor_get(v_l_17_, 4);
lean_dec(v_unused_236_);
v_unused_237_ = lean_ctor_get(v_l_17_, 3);
lean_dec(v_unused_237_);
v_unused_238_ = lean_ctor_get(v_l_17_, 2);
lean_dec(v_unused_238_);
v_unused_239_ = lean_ctor_get(v_l_17_, 1);
lean_dec(v_unused_239_);
v_unused_240_ = lean_ctor_get(v_l_17_, 0);
lean_dec(v_unused_240_);
v___x_230_ = v_l_17_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_dec(v_l_17_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 4, v_r_170_);
lean_ctor_set(v___x_230_, 3, v___x_228_);
lean_ctor_set(v___x_230_, 2, v_v_168_);
lean_ctor_set(v___x_230_, 1, v_k_167_);
lean_ctor_set(v___x_230_, 0, v___x_225_);
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v___x_225_);
lean_ctor_set(v_reuseFailAlloc_234_, 1, v_k_167_);
lean_ctor_set(v_reuseFailAlloc_234_, 2, v_v_168_);
lean_ctor_set(v_reuseFailAlloc_234_, 3, v___x_228_);
lean_ctor_set(v_reuseFailAlloc_234_, 4, v_r_170_);
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
}
}
}
else
{
lean_object* v_l_248_; 
v_l_248_ = lean_ctor_get(v_impl_163_, 3);
lean_inc(v_l_248_);
if (lean_obj_tag(v_l_248_) == 0)
{
lean_object* v_r_249_; lean_object* v_k_250_; lean_object* v_v_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_274_; 
v_r_249_ = lean_ctor_get(v_impl_163_, 4);
v_k_250_ = lean_ctor_get(v_impl_163_, 1);
v_v_251_ = lean_ctor_get(v_impl_163_, 2);
v_isSharedCheck_274_ = !lean_is_exclusive(v_impl_163_);
if (v_isSharedCheck_274_ == 0)
{
lean_object* v_unused_275_; lean_object* v_unused_276_; 
v_unused_275_ = lean_ctor_get(v_impl_163_, 3);
lean_dec(v_unused_275_);
v_unused_276_ = lean_ctor_get(v_impl_163_, 0);
lean_dec(v_unused_276_);
v___x_253_ = v_impl_163_;
v_isShared_254_ = v_isSharedCheck_274_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_r_249_);
lean_inc(v_v_251_);
lean_inc(v_k_250_);
lean_dec(v_impl_163_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_274_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v_k_255_; lean_object* v_v_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_270_; 
v_k_255_ = lean_ctor_get(v_l_248_, 1);
v_v_256_ = lean_ctor_get(v_l_248_, 2);
v_isSharedCheck_270_ = !lean_is_exclusive(v_l_248_);
if (v_isSharedCheck_270_ == 0)
{
lean_object* v_unused_271_; lean_object* v_unused_272_; lean_object* v_unused_273_; 
v_unused_271_ = lean_ctor_get(v_l_248_, 4);
lean_dec(v_unused_271_);
v_unused_272_ = lean_ctor_get(v_l_248_, 3);
lean_dec(v_unused_272_);
v_unused_273_ = lean_ctor_get(v_l_248_, 0);
lean_dec(v_unused_273_);
v___x_258_ = v_l_248_;
v_isShared_259_ = v_isSharedCheck_270_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_v_256_);
lean_inc(v_k_255_);
lean_dec(v_l_248_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_270_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___x_260_; lean_object* v___x_262_; 
v___x_260_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_249_, 2);
if (v_isShared_259_ == 0)
{
lean_ctor_set(v___x_258_, 4, v_r_249_);
lean_ctor_set(v___x_258_, 3, v_r_249_);
lean_ctor_set(v___x_258_, 2, v_v_16_);
lean_ctor_set(v___x_258_, 1, v_k_15_);
lean_ctor_set(v___x_258_, 0, v___x_164_);
v___x_262_ = v___x_258_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_269_; 
v_reuseFailAlloc_269_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_269_, 0, v___x_164_);
lean_ctor_set(v_reuseFailAlloc_269_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_269_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_269_, 3, v_r_249_);
lean_ctor_set(v_reuseFailAlloc_269_, 4, v_r_249_);
v___x_262_ = v_reuseFailAlloc_269_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
lean_object* v___x_264_; 
lean_inc(v_r_249_);
if (v_isShared_254_ == 0)
{
lean_ctor_set(v___x_253_, 3, v_r_249_);
lean_ctor_set(v___x_253_, 0, v___x_164_);
v___x_264_ = v___x_253_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v___x_164_);
lean_ctor_set(v_reuseFailAlloc_268_, 1, v_k_250_);
lean_ctor_set(v_reuseFailAlloc_268_, 2, v_v_251_);
lean_ctor_set(v_reuseFailAlloc_268_, 3, v_r_249_);
lean_ctor_set(v_reuseFailAlloc_268_, 4, v_r_249_);
v___x_264_ = v_reuseFailAlloc_268_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
lean_object* v___x_266_; 
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v___x_264_);
lean_ctor_set(v___x_20_, 3, v___x_262_);
lean_ctor_set(v___x_20_, 2, v_v_256_);
lean_ctor_set(v___x_20_, 1, v_k_255_);
lean_ctor_set(v___x_20_, 0, v___x_260_);
v___x_266_ = v___x_20_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v___x_260_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v_k_255_);
lean_ctor_set(v_reuseFailAlloc_267_, 2, v_v_256_);
lean_ctor_set(v_reuseFailAlloc_267_, 3, v___x_262_);
lean_ctor_set(v_reuseFailAlloc_267_, 4, v___x_264_);
v___x_266_ = v_reuseFailAlloc_267_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
return v___x_266_;
}
}
}
}
}
}
else
{
lean_object* v_r_277_; 
v_r_277_ = lean_ctor_get(v_impl_163_, 4);
lean_inc(v_r_277_);
if (lean_obj_tag(v_r_277_) == 0)
{
lean_object* v_k_278_; lean_object* v_v_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_290_; 
v_k_278_ = lean_ctor_get(v_impl_163_, 1);
v_v_279_ = lean_ctor_get(v_impl_163_, 2);
v_isSharedCheck_290_ = !lean_is_exclusive(v_impl_163_);
if (v_isSharedCheck_290_ == 0)
{
lean_object* v_unused_291_; lean_object* v_unused_292_; lean_object* v_unused_293_; 
v_unused_291_ = lean_ctor_get(v_impl_163_, 4);
lean_dec(v_unused_291_);
v_unused_292_ = lean_ctor_get(v_impl_163_, 3);
lean_dec(v_unused_292_);
v_unused_293_ = lean_ctor_get(v_impl_163_, 0);
lean_dec(v_unused_293_);
v___x_281_ = v_impl_163_;
v_isShared_282_ = v_isSharedCheck_290_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_v_279_);
lean_inc(v_k_278_);
lean_dec(v_impl_163_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_290_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_283_; lean_object* v___x_285_; 
v___x_283_ = lean_unsigned_to_nat(3u);
if (v_isShared_282_ == 0)
{
lean_ctor_set(v___x_281_, 4, v_l_248_);
lean_ctor_set(v___x_281_, 2, v_v_16_);
lean_ctor_set(v___x_281_, 1, v_k_15_);
lean_ctor_set(v___x_281_, 0, v___x_164_);
v___x_285_ = v___x_281_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v___x_164_);
lean_ctor_set(v_reuseFailAlloc_289_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_289_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_289_, 3, v_l_248_);
lean_ctor_set(v_reuseFailAlloc_289_, 4, v_l_248_);
v___x_285_ = v_reuseFailAlloc_289_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
lean_object* v___x_287_; 
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v_r_277_);
lean_ctor_set(v___x_20_, 3, v___x_285_);
lean_ctor_set(v___x_20_, 2, v_v_279_);
lean_ctor_set(v___x_20_, 1, v_k_278_);
lean_ctor_set(v___x_20_, 0, v___x_283_);
v___x_287_ = v___x_20_;
goto v_reusejp_286_;
}
else
{
lean_object* v_reuseFailAlloc_288_; 
v_reuseFailAlloc_288_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_288_, 0, v___x_283_);
lean_ctor_set(v_reuseFailAlloc_288_, 1, v_k_278_);
lean_ctor_set(v_reuseFailAlloc_288_, 2, v_v_279_);
lean_ctor_set(v_reuseFailAlloc_288_, 3, v___x_285_);
lean_ctor_set(v_reuseFailAlloc_288_, 4, v_r_277_);
v___x_287_ = v_reuseFailAlloc_288_;
goto v_reusejp_286_;
}
v_reusejp_286_:
{
return v___x_287_;
}
}
}
}
else
{
lean_object* v___x_294_; lean_object* v___x_296_; 
v___x_294_ = lean_unsigned_to_nat(2u);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 4, v_impl_163_);
lean_ctor_set(v___x_20_, 3, v_r_277_);
lean_ctor_set(v___x_20_, 0, v___x_294_);
v___x_296_ = v___x_20_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_297_; 
v_reuseFailAlloc_297_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_297_, 0, v___x_294_);
lean_ctor_set(v_reuseFailAlloc_297_, 1, v_k_15_);
lean_ctor_set(v_reuseFailAlloc_297_, 2, v_v_16_);
lean_ctor_set(v_reuseFailAlloc_297_, 3, v_r_277_);
lean_ctor_set(v_reuseFailAlloc_297_, 4, v_impl_163_);
v___x_296_ = v_reuseFailAlloc_297_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
return v___x_296_;
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
lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_299_ = lean_unsigned_to_nat(1u);
v___x_300_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
lean_ctor_set(v___x_300_, 1, v_k_11_);
lean_ctor_set(v___x_300_, 2, v_v_12_);
lean_ctor_set(v___x_300_, 3, v_t_13_);
lean_ctor_set(v___x_300_, 4, v_t_13_);
return v___x_300_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___redArg(lean_object* v_as_x27_301_, lean_object* v_b_302_){
_start:
{
if (lean_obj_tag(v_as_x27_301_) == 0)
{
return v_b_302_;
}
else
{
lean_object* v_head_303_; lean_object* v_tail_304_; lean_object* v_fst_305_; lean_object* v_snd_306_; lean_object* v_r_307_; 
v_head_303_ = lean_ctor_get(v_as_x27_301_, 0);
v_tail_304_ = lean_ctor_get(v_as_x27_301_, 1);
v_fst_305_ = lean_ctor_get(v_head_303_, 0);
v_snd_306_ = lean_ctor_get(v_head_303_, 1);
lean_inc(v_snd_306_);
lean_inc(v_fst_305_);
v_r_307_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0___redArg(v_fst_305_, v_snd_306_, v_b_302_);
v_as_x27_301_ = v_tail_304_;
v_b_302_ = v_r_307_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___redArg___boxed(lean_object* v_as_x27_309_, lean_object* v_b_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___redArg(v_as_x27_309_, v_b_310_);
lean_dec(v_as_x27_309_);
return v_res_311_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__26(void){
_start:
{
lean_object* v_r_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v_r_370_ = lean_box(1);
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__25));
v___x_372_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___redArg(v___x_371_, v_r_370_);
return v___x_372_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames(void){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__26, &lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__26);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0(lean_object* v_00_u03b2_374_, lean_object* v_k_375_, lean_object* v_v_376_, lean_object* v_t_377_, lean_object* v_hl_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__0___redArg(v_k_375_, v_v_376_, v_t_377_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1(lean_object* v_as_380_, lean_object* v_as_x27_381_, lean_object* v_b_382_, lean_object* v_a_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___redArg(v_as_x27_381_, v_b_382_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1___boxed(lean_object* v_as_385_, lean_object* v_as_x27_386_, lean_object* v_b_387_, lean_object* v_a_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_GuessName_endCapitalNames_spec__1(v_as_385_, v_as_x27_386_, v_b_387_, v_a_388_);
lean_dec(v_as_x27_386_);
lean_dec(v_as_385_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___redArg(lean_object* v_x_390_, lean_object* v_s_391_){
_start:
{
lean_object* v___x_392_; lean_object* v___x_393_; uint8_t v___x_394_; 
v___x_392_ = lean_string_utf8_byte_size(v_s_391_);
v___x_393_ = lean_string_utf8_byte_size(v_x_390_);
v___x_394_ = lean_nat_dec_le(v___x_393_, v___x_392_);
if (v___x_394_ == 0)
{
lean_object* v___x_395_; 
lean_dec_ref(v_s_391_);
v___x_395_ = lean_box(0);
return v___x_395_;
}
else
{
lean_object* v___x_396_; uint8_t v___x_397_; 
v___x_396_ = lean_unsigned_to_nat(0u);
v___x_397_ = lean_string_memcmp(v_s_391_, v_x_390_, v___x_396_, v___x_396_, v___x_393_);
if (v___x_397_ == 0)
{
lean_object* v___x_398_; 
lean_dec_ref(v_s_391_);
v___x_398_ = lean_box(0);
return v___x_398_;
}
else
{
lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
lean_inc_ref(v_s_391_);
v___x_399_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_399_, 0, v_s_391_);
lean_ctor_set(v___x_399_, 1, v___x_396_);
lean_ctor_set(v___x_399_, 2, v___x_392_);
v___x_400_ = l_String_Slice_pos_x21(v___x_399_, v___x_393_);
lean_dec_ref_known(v___x_399_, 3);
v___x_401_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_401_, 0, v_s_391_);
lean_ctor_set(v___x_401_, 1, v___x_400_);
lean_ctor_set(v___x_401_, 2, v___x_392_);
v___x_402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_402_, 0, v___x_401_);
return v___x_402_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___redArg___boxed(lean_object* v_x_403_, lean_object* v_s_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___redArg(v_x_403_, v_s_404_);
lean_dec_ref(v_x_403_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0(lean_object* v_x_406_, lean_object* v_s_407_, lean_object* v_pat_408_){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___redArg(v_x_406_, v_s_407_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___boxed(lean_object* v_x_410_, lean_object* v_s_411_, lean_object* v_pat_412_){
_start:
{
lean_object* v_res_413_; 
v_res_413_ = lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0(v_x_410_, v_s_411_, v_pat_412_);
lean_dec_ref(v_pat_412_);
lean_dec_ref(v_x_410_);
return v_res_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___redArg(lean_object* v_t_414_, lean_object* v_k_415_){
_start:
{
if (lean_obj_tag(v_t_414_) == 0)
{
lean_object* v_k_416_; lean_object* v_v_417_; lean_object* v_l_418_; lean_object* v_r_419_; uint8_t v___x_420_; 
v_k_416_ = lean_ctor_get(v_t_414_, 1);
v_v_417_ = lean_ctor_get(v_t_414_, 2);
v_l_418_ = lean_ctor_get(v_t_414_, 3);
v_r_419_ = lean_ctor_get(v_t_414_, 4);
v___x_420_ = lean_string_compare(v_k_415_, v_k_416_);
switch(v___x_420_)
{
case 0:
{
v_t_414_ = v_l_418_;
goto _start;
}
case 1:
{
lean_object* v___x_422_; 
lean_inc(v_v_417_);
v___x_422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_422_, 0, v_v_417_);
return v___x_422_;
}
default: 
{
v_t_414_ = v_r_419_;
goto _start;
}
}
}
else
{
lean_object* v___x_424_; 
v___x_424_ = lean_box(0);
return v___x_424_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___redArg___boxed(lean_object* v_t_425_, lean_object* v_k_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___redArg(v_t_425_, v_k_426_);
lean_dec_ref(v_k_426_);
lean_dec(v_t_425_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__2(lean_object* v_s_428_, lean_object* v_i_u2081_429_, lean_object* v_x_430_){
_start:
{
if (lean_obj_tag(v_x_430_) == 0)
{
lean_object* v___x_431_; 
v___x_431_ = lean_box(0);
return v___x_431_;
}
else
{
lean_object* v_head_432_; lean_object* v_tail_433_; lean_object* v___x_435_; uint8_t v_isShared_436_; uint8_t v_isSharedCheck_453_; 
v_head_432_ = lean_ctor_get(v_x_430_, 0);
v_tail_433_ = lean_ctor_get(v_x_430_, 1);
v_isSharedCheck_453_ = !lean_is_exclusive(v_x_430_);
if (v_isSharedCheck_453_ == 0)
{
v___x_435_ = v_x_430_;
v_isShared_436_ = v_isSharedCheck_453_;
goto v_resetjp_434_;
}
else
{
lean_inc(v_tail_433_);
lean_inc(v_head_432_);
lean_dec(v_x_430_);
v___x_435_ = lean_box(0);
v_isShared_436_ = v_isSharedCheck_453_;
goto v_resetjp_434_;
}
v_resetjp_434_:
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; 
v___x_437_ = lean_string_utf8_byte_size(v_s_428_);
v___x_438_ = lean_string_utf8_extract(v_s_428_, v_i_u2081_429_, v___x_437_);
v___x_439_ = lp_mathlib_String_dropPrefix_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__0___redArg(v_head_432_, v___x_438_);
if (lean_obj_tag(v___x_439_) == 0)
{
lean_del_object(v___x_435_);
lean_dec(v_head_432_);
v_x_430_ = v_tail_433_;
goto _start;
}
else
{
lean_object* v_val_441_; lean_object* v___x_443_; uint8_t v_isShared_444_; uint8_t v_isSharedCheck_452_; 
lean_dec(v_tail_433_);
v_val_441_ = lean_ctor_get(v___x_439_, 0);
v_isSharedCheck_452_ = !lean_is_exclusive(v___x_439_);
if (v_isSharedCheck_452_ == 0)
{
v___x_443_ = v___x_439_;
v_isShared_444_ = v_isSharedCheck_452_;
goto v_resetjp_442_;
}
else
{
lean_inc(v_val_441_);
lean_dec(v___x_439_);
v___x_443_ = lean_box(0);
v_isShared_444_ = v_isSharedCheck_452_;
goto v_resetjp_442_;
}
v_resetjp_442_:
{
lean_object* v___x_445_; lean_object* v___x_447_; 
v___x_445_ = l_String_Slice_toString(v_val_441_);
lean_dec(v_val_441_);
if (v_isShared_436_ == 0)
{
lean_ctor_set_tag(v___x_435_, 0);
lean_ctor_set(v___x_435_, 1, v___x_445_);
v___x_447_ = v___x_435_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_head_432_);
lean_ctor_set(v_reuseFailAlloc_451_, 1, v___x_445_);
v___x_447_ = v_reuseFailAlloc_451_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
lean_object* v___x_449_; 
if (v_isShared_444_ == 0)
{
lean_ctor_set(v___x_443_, 0, v___x_447_);
v___x_449_ = v___x_443_;
goto v_reusejp_448_;
}
else
{
lean_object* v_reuseFailAlloc_450_; 
v_reuseFailAlloc_450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_450_, 0, v___x_447_);
v___x_449_ = v_reuseFailAlloc_450_;
goto v_reusejp_448_;
}
v_reusejp_448_:
{
return v___x_449_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_findSome_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__2___boxed(lean_object* v_s_454_, lean_object* v_i_u2081_455_, lean_object* v_x_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_List_findSome_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__2(v_s_454_, v_i_u2081_455_, v_x_456_);
lean_dec(v_i_u2081_455_);
lean_dec_ref(v_s_454_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_String_splitCase(lean_object* v_s_458_, lean_object* v_i_u2080_459_, lean_object* v_r_460_){
_start:
{
lean_object* v_i_u2081_461_; uint8_t v___y_470_; uint8_t v___x_478_; 
v_i_u2081_461_ = lean_string_utf8_next(v_s_458_, v_i_u2080_459_);
v___x_478_ = lean_string_utf8_at_end(v_s_458_, v_i_u2081_461_);
if (v___x_478_ == 0)
{
uint32_t v___x_479_; uint32_t v___x_480_; uint8_t v___x_481_; 
v___x_479_ = lean_string_utf8_get(v_s_458_, v_i_u2080_459_);
lean_dec(v_i_u2080_459_);
v___x_480_ = 95;
v___x_481_ = lean_uint32_dec_eq(v___x_479_, v___x_480_);
if (v___x_481_ == 0)
{
uint32_t v___x_482_; uint8_t v___x_483_; 
v___x_482_ = lean_string_utf8_get(v_s_458_, v_i_u2081_461_);
v___x_483_ = lean_uint32_dec_eq(v___x_482_, v___x_480_);
if (v___x_483_ == 0)
{
uint32_t v___x_490_; uint8_t v___x_491_; 
v___x_490_ = 65;
v___x_491_ = lean_uint32_dec_le(v___x_490_, v___x_482_);
if (v___x_491_ == 0)
{
v_i_u2080_459_ = v_i_u2081_461_;
goto _start;
}
else
{
uint32_t v___x_493_; uint8_t v___x_494_; 
v___x_493_ = 90;
v___x_494_ = lean_uint32_dec_le(v___x_482_, v___x_493_);
if (v___x_494_ == 0)
{
v_i_u2080_459_ = v_i_u2081_461_;
goto _start;
}
else
{
lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_496_ = lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames;
v___x_497_ = lean_unsigned_to_nat(0u);
v___x_498_ = lean_string_utf8_extract(v_s_458_, v___x_497_, v_i_u2081_461_);
v___x_499_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___redArg(v___x_496_, v___x_498_);
if (lean_obj_tag(v___x_499_) == 1)
{
lean_object* v_val_500_; lean_object* v___x_501_; 
v_val_500_ = lean_ctor_get(v___x_499_, 0);
lean_inc(v_val_500_);
lean_dec_ref_known(v___x_499_, 1);
v___x_501_ = lp_mathlib_List_findSome_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__2(v_s_458_, v_i_u2081_461_, v_val_500_);
if (lean_obj_tag(v___x_501_) == 1)
{
lean_object* v_val_502_; lean_object* v_fst_503_; lean_object* v_snd_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_513_; 
lean_dec(v_i_u2081_461_);
lean_dec_ref(v_s_458_);
v_val_502_ = lean_ctor_get(v___x_501_, 0);
lean_inc(v_val_502_);
lean_dec_ref_known(v___x_501_, 1);
v_fst_503_ = lean_ctor_get(v_val_502_, 0);
v_snd_504_ = lean_ctor_get(v_val_502_, 1);
v_isSharedCheck_513_ = !lean_is_exclusive(v_val_502_);
if (v_isSharedCheck_513_ == 0)
{
v___x_506_ = v_val_502_;
v_isShared_507_ = v_isSharedCheck_513_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_snd_504_);
lean_inc(v_fst_503_);
lean_dec(v_val_502_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_513_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v___x_508_; lean_object* v___x_510_; 
v___x_508_ = lean_string_append(v___x_498_, v_fst_503_);
lean_dec(v_fst_503_);
if (v_isShared_507_ == 0)
{
lean_ctor_set_tag(v___x_506_, 1);
lean_ctor_set(v___x_506_, 1, v_r_460_);
lean_ctor_set(v___x_506_, 0, v___x_508_);
v___x_510_ = v___x_506_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_512_; 
v_reuseFailAlloc_512_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_512_, 0, v___x_508_);
lean_ctor_set(v_reuseFailAlloc_512_, 1, v_r_460_);
v___x_510_ = v_reuseFailAlloc_512_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
v_s_458_ = v_snd_504_;
v_i_u2080_459_ = v___x_497_;
v_r_460_ = v___x_510_;
goto _start;
}
}
}
else
{
lean_dec(v___x_501_);
lean_dec_ref(v___x_498_);
goto v___jp_484_;
}
}
else
{
lean_dec(v___x_499_);
lean_dec_ref(v___x_498_);
goto v___jp_484_;
}
}
}
}
else
{
goto v___jp_462_;
}
v___jp_484_:
{
uint32_t v___x_485_; uint8_t v___x_486_; 
v___x_485_ = 65;
v___x_486_ = lean_uint32_dec_le(v___x_485_, v___x_479_);
if (v___x_486_ == 0)
{
v___y_470_ = v___x_483_;
goto v___jp_469_;
}
else
{
uint32_t v___x_487_; uint8_t v___x_488_; 
v___x_487_ = 90;
v___x_488_ = lean_uint32_dec_le(v___x_479_, v___x_487_);
if (v___x_488_ == 0)
{
v___y_470_ = v___x_483_;
goto v___jp_469_;
}
else
{
v_i_u2080_459_ = v_i_u2081_461_;
goto _start;
}
}
}
}
else
{
goto v___jp_462_;
}
}
else
{
lean_object* v_r_514_; lean_object* v___x_515_; 
lean_dec(v_i_u2081_461_);
lean_dec(v_i_u2080_459_);
v_r_514_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_r_514_, 0, v_s_458_);
lean_ctor_set(v_r_514_, 1, v_r_460_);
v___x_515_ = l_List_reverse___redArg(v_r_514_);
return v___x_515_;
}
v___jp_462_:
{
lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_463_ = lean_string_utf8_byte_size(v_s_458_);
v___x_464_ = lean_string_utf8_extract(v_s_458_, v_i_u2081_461_, v___x_463_);
v___x_465_ = lean_unsigned_to_nat(0u);
v___x_466_ = lean_string_utf8_extract(v_s_458_, v___x_465_, v_i_u2081_461_);
lean_dec(v_i_u2081_461_);
lean_dec_ref(v_s_458_);
v___x_467_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_467_, 0, v___x_466_);
lean_ctor_set(v___x_467_, 1, v_r_460_);
v_s_458_ = v___x_464_;
v_i_u2080_459_ = v___x_465_;
v_r_460_ = v___x_467_;
goto _start;
}
v___jp_469_:
{
if (v___y_470_ == 0)
{
lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_471_ = lean_string_utf8_byte_size(v_s_458_);
v___x_472_ = lean_string_utf8_extract(v_s_458_, v_i_u2081_461_, v___x_471_);
v___x_473_ = lean_unsigned_to_nat(0u);
v___x_474_ = lean_string_utf8_extract(v_s_458_, v___x_473_, v_i_u2081_461_);
lean_dec(v_i_u2081_461_);
lean_dec_ref(v_s_458_);
v___x_475_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
lean_ctor_set(v___x_475_, 1, v_r_460_);
v_s_458_ = v___x_472_;
v_i_u2080_459_ = v___x_473_;
v_r_460_ = v___x_475_;
goto _start;
}
else
{
v_i_u2080_459_ = v_i_u2081_461_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1(lean_object* v_00_u03b4_516_, lean_object* v_t_517_, lean_object* v_k_518_){
_start:
{
lean_object* v___x_519_; 
v___x_519_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___redArg(v_t_517_, v_k_518_);
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1___boxed(lean_object* v_00_u03b4_520_, lean_object* v_t_521_, lean_object* v_k_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Mathlib_Tactic_GuessName_String_splitCase_spec__1(v_00_u03b4_520_, v_t_521_, v_k_522_);
lean_dec_ref(v_k_522_);
lean_dec(v_t_521_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_String_decapitalizeSeq(lean_object* v_s_524_, lean_object* v_i_525_){
_start:
{
uint32_t v___y_527_; uint8_t v___y_532_; uint8_t v___x_540_; 
v___x_540_ = lean_string_utf8_at_end(v_s_524_, v_i_525_);
if (v___x_540_ == 0)
{
uint32_t v___x_541_; uint32_t v___x_542_; uint8_t v___x_543_; 
v___x_541_ = lean_string_utf8_get(v_s_524_, v_i_525_);
v___x_542_ = 65;
v___x_543_ = lean_uint32_dec_le(v___x_542_, v___x_541_);
if (v___x_543_ == 0)
{
lean_dec(v_i_525_);
return v_s_524_;
}
else
{
uint32_t v___x_544_; uint8_t v___x_545_; 
v___x_544_ = 90;
v___x_545_ = lean_uint32_dec_le(v___x_541_, v___x_544_);
if (v___x_545_ == 0)
{
lean_dec(v_i_525_);
return v_s_524_;
}
else
{
v___y_532_ = v___x_540_;
goto v___jp_531_;
}
}
}
else
{
v___y_532_ = v___x_540_;
goto v___jp_531_;
}
v___jp_526_:
{
lean_object* v___x_528_; lean_object* v___x_529_; 
lean_inc_ref(v_s_524_);
v___x_528_ = lean_string_utf8_set(v_s_524_, v_i_525_, v___y_527_);
v___x_529_ = lean_string_utf8_next(v_s_524_, v_i_525_);
lean_dec(v_i_525_);
lean_dec_ref(v_s_524_);
v_s_524_ = v___x_528_;
v_i_525_ = v___x_529_;
goto _start;
}
v___jp_531_:
{
if (v___y_532_ == 0)
{
uint32_t v___x_533_; uint32_t v___x_534_; uint8_t v___x_535_; 
v___x_533_ = lean_string_utf8_get(v_s_524_, v_i_525_);
v___x_534_ = 65;
v___x_535_ = lean_uint32_dec_le(v___x_534_, v___x_533_);
if (v___x_535_ == 0)
{
v___y_527_ = v___x_533_;
goto v___jp_526_;
}
else
{
uint32_t v___x_536_; uint8_t v___x_537_; 
v___x_536_ = 90;
v___x_537_ = lean_uint32_dec_le(v___x_533_, v___x_536_);
if (v___x_537_ == 0)
{
v___y_527_ = v___x_533_;
goto v___jp_526_;
}
else
{
uint32_t v___x_538_; uint32_t v___x_539_; 
v___x_538_ = 32;
v___x_539_ = lean_uint32_add(v___x_533_, v___x_538_);
v___y_527_ = v___x_539_;
goto v___jp_526_;
}
}
}
else
{
lean_dec(v_i_525_);
return v_s_524_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeLike(lean_object* v_r_546_, lean_object* v_s_547_){
_start:
{
lean_object* v___x_548_; uint32_t v___x_549_; uint32_t v___x_550_; uint8_t v___x_551_; 
v___x_548_ = lean_unsigned_to_nat(0u);
v___x_549_ = lean_string_utf8_get(v_r_546_, v___x_548_);
v___x_550_ = 65;
v___x_551_ = lean_uint32_dec_le(v___x_550_, v___x_549_);
if (v___x_551_ == 0)
{
lean_object* v___x_552_; 
v___x_552_ = lp_mathlib_Mathlib_Tactic_GuessName_String_decapitalizeSeq(v_s_547_, v___x_548_);
return v___x_552_;
}
else
{
uint32_t v___x_553_; uint8_t v___x_554_; 
v___x_553_ = 90;
v___x_554_ = lean_uint32_dec_le(v___x_549_, v___x_553_);
if (v___x_554_ == 0)
{
lean_object* v___x_555_; 
v___x_555_ = lp_mathlib_Mathlib_Tactic_GuessName_String_decapitalizeSeq(v_s_547_, v___x_548_);
return v___x_555_;
}
else
{
return v_s_547_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeLike___boxed(lean_object* v_r_556_, lean_object* v_s_557_){
_start:
{
lean_object* v_res_558_; 
v_res_558_ = lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeLike(v_r_556_, v_s_557_);
lean_dec_ref(v_r_556_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeFirstLike(lean_object* v_s_559_, lean_object* v_x_560_){
_start:
{
if (lean_obj_tag(v_x_560_) == 0)
{
return v_x_560_;
}
else
{
lean_object* v_head_561_; lean_object* v_tail_562_; lean_object* v___x_564_; uint8_t v_isShared_565_; uint8_t v_isSharedCheck_570_; 
v_head_561_ = lean_ctor_get(v_x_560_, 0);
v_tail_562_ = lean_ctor_get(v_x_560_, 1);
v_isSharedCheck_570_ = !lean_is_exclusive(v_x_560_);
if (v_isSharedCheck_570_ == 0)
{
v___x_564_ = v_x_560_;
v_isShared_565_ = v_isSharedCheck_570_;
goto v_resetjp_563_;
}
else
{
lean_inc(v_tail_562_);
lean_inc(v_head_561_);
lean_dec(v_x_560_);
v___x_564_ = lean_box(0);
v_isShared_565_ = v_isSharedCheck_570_;
goto v_resetjp_563_;
}
v_resetjp_563_:
{
lean_object* v___x_566_; lean_object* v___x_568_; 
v___x_566_ = lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeLike(v_s_559_, v_head_561_);
if (v_isShared_565_ == 0)
{
lean_ctor_set(v___x_564_, 0, v___x_566_);
v___x_568_ = v___x_564_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v___x_566_);
lean_ctor_set(v_reuseFailAlloc_569_, 1, v_tail_562_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeFirstLike___boxed(lean_object* v_s_571_, lean_object* v_x_572_){
_start:
{
lean_object* v_res_573_; 
v_res_573_ = lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeFirstLike(v_s_571_, v_x_572_);
lean_dec_ref(v_s_571_);
return v_res_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___redArg(lean_object* v_a_574_, lean_object* v_x_575_){
_start:
{
if (lean_obj_tag(v_x_575_) == 0)
{
lean_object* v___x_576_; 
v___x_576_ = lean_box(0);
return v___x_576_;
}
else
{
lean_object* v_key_577_; lean_object* v_value_578_; lean_object* v_tail_579_; uint8_t v___x_580_; 
v_key_577_ = lean_ctor_get(v_x_575_, 0);
v_value_578_ = lean_ctor_get(v_x_575_, 1);
v_tail_579_ = lean_ctor_get(v_x_575_, 2);
v___x_580_ = lean_string_dec_eq(v_key_577_, v_a_574_);
if (v___x_580_ == 0)
{
v_x_575_ = v_tail_579_;
goto _start;
}
else
{
lean_object* v___x_582_; 
lean_inc(v_value_578_);
v___x_582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_582_, 0, v_value_578_);
return v___x_582_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___redArg___boxed(lean_object* v_a_583_, lean_object* v_x_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___redArg(v_a_583_, v_x_584_);
lean_dec(v_x_584_);
lean_dec_ref(v_a_583_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg(lean_object* v_m_586_, lean_object* v_a_587_){
_start:
{
lean_object* v_buckets_588_; lean_object* v___x_589_; uint64_t v___x_590_; uint64_t v___x_591_; uint64_t v___x_592_; uint64_t v_fold_593_; uint64_t v___x_594_; uint64_t v___x_595_; uint64_t v___x_596_; size_t v___x_597_; size_t v___x_598_; size_t v___x_599_; size_t v___x_600_; size_t v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
v_buckets_588_ = lean_ctor_get(v_m_586_, 1);
v___x_589_ = lean_array_get_size(v_buckets_588_);
v___x_590_ = lean_string_hash(v_a_587_);
v___x_591_ = 32ULL;
v___x_592_ = lean_uint64_shift_right(v___x_590_, v___x_591_);
v_fold_593_ = lean_uint64_xor(v___x_590_, v___x_592_);
v___x_594_ = 16ULL;
v___x_595_ = lean_uint64_shift_right(v_fold_593_, v___x_594_);
v___x_596_ = lean_uint64_xor(v_fold_593_, v___x_595_);
v___x_597_ = lean_uint64_to_usize(v___x_596_);
v___x_598_ = lean_usize_of_nat(v___x_589_);
v___x_599_ = ((size_t)1ULL);
v___x_600_ = lean_usize_sub(v___x_598_, v___x_599_);
v___x_601_ = lean_usize_land(v___x_597_, v___x_600_);
v___x_602_ = lean_array_uget_borrowed(v_buckets_588_, v___x_601_);
v___x_603_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___redArg(v_a_587_, v___x_602_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg___boxed(lean_object* v_m_604_, lean_object* v_a_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg(v_m_604_, v_a_605_);
lean_dec_ref(v_a_605_);
lean_dec_ref(v_m_604_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_mapAux___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__0(lean_object* v_s_607_, lean_object* v_p_608_){
_start:
{
uint32_t v___y_610_; lean_object* v___x_615_; uint8_t v___x_616_; 
v___x_615_ = lean_string_utf8_byte_size(v_s_607_);
v___x_616_ = lean_nat_dec_eq(v_p_608_, v___x_615_);
if (v___x_616_ == 0)
{
uint32_t v___x_617_; uint32_t v___x_618_; uint8_t v___x_619_; 
v___x_617_ = lean_string_utf8_get_fast(v_s_607_, v_p_608_);
v___x_618_ = 65;
v___x_619_ = lean_uint32_dec_le(v___x_618_, v___x_617_);
if (v___x_619_ == 0)
{
v___y_610_ = v___x_617_;
goto v___jp_609_;
}
else
{
uint32_t v___x_620_; uint8_t v___x_621_; 
v___x_620_ = 90;
v___x_621_ = lean_uint32_dec_le(v___x_617_, v___x_620_);
if (v___x_621_ == 0)
{
v___y_610_ = v___x_617_;
goto v___jp_609_;
}
else
{
uint32_t v___x_622_; uint32_t v___x_623_; 
v___x_622_ = 32;
v___x_623_ = lean_uint32_add(v___x_617_, v___x_622_);
v___y_610_ = v___x_623_;
goto v___jp_609_;
}
}
}
else
{
lean_dec(v_p_608_);
return v_s_607_;
}
v___jp_609_:
{
lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; 
lean_inc(v_p_608_);
v___x_611_ = lean_string_utf8_set(v_s_607_, v_p_608_, v___y_610_);
v___x_612_ = l_Char_utf8Size(v___y_610_);
v___x_613_ = lean_nat_add(v_p_608_, v___x_612_);
lean_dec(v___x_612_);
lean_dec(v_p_608_);
v_s_607_ = v___x_611_;
v_p_608_ = v___x_613_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_applyNameDict(lean_object* v_g_624_, lean_object* v_x_625_){
_start:
{
if (lean_obj_tag(v_x_625_) == 0)
{
return v_x_625_;
}
else
{
lean_object* v_head_626_; lean_object* v_tail_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_645_; 
v_head_626_ = lean_ctor_get(v_x_625_, 0);
v_tail_627_ = lean_ctor_get(v_x_625_, 1);
v_isSharedCheck_645_ = !lean_is_exclusive(v_x_625_);
if (v_isSharedCheck_645_ == 0)
{
v___x_629_ = v_x_625_;
v_isShared_630_ = v_isSharedCheck_645_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_tail_627_);
lean_inc(v_head_626_);
lean_dec(v_x_625_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_645_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v___y_632_; lean_object* v_nameDict_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; 
v_nameDict_635_ = lean_ctor_get(v_g_624_, 0);
v___x_636_ = lean_unsigned_to_nat(0u);
lean_inc(v_head_626_);
v___x_637_ = lp_mathlib_String_mapAux___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__0(v_head_626_, v___x_636_);
v___x_638_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg(v_nameDict_635_, v___x_637_);
lean_dec_ref(v___x_637_);
if (lean_obj_tag(v___x_638_) == 0)
{
lean_object* v___x_639_; lean_object* v___x_641_; 
v___x_639_ = lean_box(0);
if (v_isShared_630_ == 0)
{
lean_ctor_set(v___x_629_, 1, v___x_639_);
v___x_641_ = v___x_629_;
goto v_reusejp_640_;
}
else
{
lean_object* v_reuseFailAlloc_642_; 
v_reuseFailAlloc_642_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_642_, 0, v_head_626_);
lean_ctor_set(v_reuseFailAlloc_642_, 1, v___x_639_);
v___x_641_ = v_reuseFailAlloc_642_;
goto v_reusejp_640_;
}
v_reusejp_640_:
{
v___y_632_ = v___x_641_;
goto v___jp_631_;
}
}
else
{
lean_object* v_val_643_; lean_object* v___x_644_; 
lean_del_object(v___x_629_);
v_val_643_ = lean_ctor_get(v___x_638_, 0);
lean_inc(v_val_643_);
lean_dec_ref_known(v___x_638_, 1);
v___x_644_ = lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeFirstLike(v_head_626_, v_val_643_);
lean_dec(v_head_626_);
v___y_632_ = v___x_644_;
goto v___jp_631_;
}
v___jp_631_:
{
lean_object* v___x_633_; lean_object* v___x_634_; 
v___x_633_ = lp_mathlib_Mathlib_Tactic_GuessName_applyNameDict(v_g_624_, v_tail_627_);
v___x_634_ = l_List_appendTR___redArg(v___y_632_, v___x_633_);
return v___x_634_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_applyNameDict___boxed(lean_object* v_g_646_, lean_object* v_x_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib_Mathlib_Tactic_GuessName_applyNameDict(v_g_646_, v_x_647_);
lean_dec_ref(v_g_646_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1(lean_object* v_00_u03b2_649_, lean_object* v_m_650_, lean_object* v_a_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg(v_m_650_, v_a_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___boxed(lean_object* v_00_u03b2_653_, lean_object* v_m_654_, lean_object* v_a_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1(v_00_u03b2_653_, v_m_654_, v_a_655_);
lean_dec_ref(v_a_655_);
lean_dec_ref(v_m_654_);
return v_res_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1(lean_object* v_00_u03b2_657_, lean_object* v_a_658_, lean_object* v_x_659_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___redArg(v_a_658_, v_x_659_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1___boxed(lean_object* v_00_u03b2_661_, lean_object* v_a_662_, lean_object* v_x_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1_spec__1(v_00_u03b2_661_, v_a_662_, v_x_663_);
lean_dec(v_x_663_);
lean_dec_ref(v_a_662_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_GuessName_fixAbbreviationAux_spec__0(lean_object* v_x_665_, lean_object* v_x_666_){
_start:
{
if (lean_obj_tag(v_x_666_) == 0)
{
return v_x_665_;
}
else
{
lean_object* v_head_667_; lean_object* v_tail_668_; lean_object* v___x_669_; 
v_head_667_ = lean_ctor_get(v_x_666_, 0);
v_tail_668_ = lean_ctor_get(v_x_666_, 1);
v___x_669_ = lean_string_append(v_x_665_, v_head_667_);
v_x_665_ = v___x_669_;
v_x_666_ = v_tail_668_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Tactic_GuessName_fixAbbreviationAux_spec__0___boxed(lean_object* v_x_671_, lean_object* v_x_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_mathlib_List_foldl___at___00Mathlib_Tactic_GuessName_fixAbbreviationAux_spec__0(v_x_671_, v_x_672_);
lean_dec(v_x_672_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux(lean_object* v_g_675_, lean_object* v_x_676_, lean_object* v_x_677_){
_start:
{
if (lean_obj_tag(v_x_676_) == 0)
{
if (lean_obj_tag(v_x_677_) == 0)
{
lean_object* v___x_678_; 
v___x_678_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__1));
return v___x_678_;
}
else
{
lean_object* v_head_679_; lean_object* v_tail_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
v_head_679_ = lean_ctor_get(v_x_677_, 0);
lean_inc(v_head_679_);
v_tail_680_ = lean_ctor_get(v_x_677_, 1);
lean_inc(v_tail_680_);
lean_dec_ref_known(v_x_677_, 2);
v___x_681_ = lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux(v_g_675_, v_tail_680_, v_x_676_);
v___x_682_ = lean_string_append(v_head_679_, v___x_681_);
lean_dec_ref(v___x_681_);
return v___x_682_;
}
}
else
{
lean_object* v_head_683_; lean_object* v_tail_684_; lean_object* v___x_686_; uint8_t v_isShared_687_; uint8_t v_isSharedCheck_722_; 
v_head_683_ = lean_ctor_get(v_x_676_, 0);
v_tail_684_ = lean_ctor_get(v_x_676_, 1);
v_isSharedCheck_722_ = !lean_is_exclusive(v_x_676_);
if (v_isSharedCheck_722_ == 0)
{
v___x_686_ = v_x_676_;
v_isShared_687_ = v_isSharedCheck_722_;
goto v_resetjp_685_;
}
else
{
lean_inc(v_tail_684_);
lean_inc(v_head_683_);
lean_dec(v_x_676_);
v___x_686_ = lean_box(0);
v_isShared_687_ = v_isSharedCheck_722_;
goto v_resetjp_685_;
}
v_resetjp_685_:
{
lean_object* v___x_688_; lean_object* v___x_690_; 
v___x_688_ = lean_box(0);
lean_inc(v_head_683_);
if (v_isShared_687_ == 0)
{
lean_ctor_set(v___x_686_, 1, v___x_688_);
v___x_690_ = v___x_686_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_721_; 
v_reuseFailAlloc_721_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_721_, 0, v_head_683_);
lean_ctor_set(v_reuseFailAlloc_721_, 1, v___x_688_);
v___x_690_ = v_reuseFailAlloc_721_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
lean_object* v_s_691_; lean_object* v___x_692_; lean_object* v_t_693_; uint8_t v___y_705_; lean_object* v___x_713_; uint8_t v___x_714_; 
v_s_691_ = l_List_appendTR___redArg(v_x_677_, v___x_690_);
v___x_692_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames___closed__1));
v_t_693_ = lp_mathlib_List_foldl___at___00Mathlib_Tactic_GuessName_fixAbbreviationAux_spec__0(v___x_692_, v_s_691_);
v___x_713_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux___closed__0));
v___x_714_ = lean_string_dec_eq(v_head_683_, v___x_713_);
lean_dec(v_head_683_);
if (v___x_714_ == 0)
{
v___y_705_ = v___x_714_;
goto v___jp_704_;
}
else
{
lean_object* v___x_715_; uint32_t v___x_716_; uint32_t v___x_717_; uint8_t v___x_718_; 
v___x_715_ = lean_unsigned_to_nat(0u);
v___x_716_ = lean_string_utf8_get(v_t_693_, v___x_715_);
v___x_717_ = 65;
v___x_718_ = lean_uint32_dec_le(v___x_717_, v___x_716_);
if (v___x_718_ == 0)
{
goto v___jp_694_;
}
else
{
uint32_t v___x_719_; uint8_t v___x_720_; 
v___x_719_ = 90;
v___x_720_ = lean_uint32_dec_le(v___x_716_, v___x_719_);
if (v___x_720_ == 0)
{
goto v___jp_694_;
}
else
{
v___y_705_ = v___x_714_;
goto v___jp_704_;
}
}
}
v___jp_694_:
{
lean_object* v_abbreviationDict_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; 
v_abbreviationDict_695_ = lean_ctor_get(v_g_675_, 1);
v___x_696_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_t_693_);
v___x_697_ = lp_mathlib_Mathlib_Tactic_GuessName_String_decapitalizeSeq(v_t_693_, v___x_696_);
v___x_698_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Mathlib_Tactic_GuessName_applyNameDict_spec__1___redArg(v_abbreviationDict_695_, v___x_697_);
lean_dec_ref(v___x_697_);
if (lean_obj_tag(v___x_698_) == 0)
{
lean_dec_ref(v_t_693_);
v_x_676_ = v_tail_684_;
v_x_677_ = v_s_691_;
goto _start;
}
else
{
lean_object* v_val_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; 
lean_dec(v_s_691_);
v_val_700_ = lean_ctor_get(v___x_698_, 0);
lean_inc(v_val_700_);
lean_dec_ref_known(v___x_698_, 1);
v___x_701_ = lp_mathlib_Mathlib_Tactic_GuessName_decapitalizeLike(v_t_693_, v_val_700_);
lean_dec_ref(v_t_693_);
v___x_702_ = lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux(v_g_675_, v_tail_684_, v___x_688_);
v___x_703_ = lean_string_append(v___x_701_, v___x_702_);
lean_dec_ref(v___x_702_);
return v___x_703_;
}
}
v___jp_704_:
{
if (v___y_705_ == 0)
{
goto v___jp_694_;
}
else
{
lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; 
lean_dec_ref(v_t_693_);
v___x_706_ = lean_unsigned_to_nat(0u);
v___x_707_ = l_List_get_x21Internal___redArg(v___x_692_, v_s_691_, v___x_706_);
v___x_708_ = lean_unsigned_to_nat(1u);
v___x_709_ = l_List_drop___redArg(v___x_708_, v_s_691_);
lean_dec(v_s_691_);
v___x_710_ = l_List_appendTR___redArg(v___x_709_, v_tail_684_);
v___x_711_ = lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux(v_g_675_, v___x_710_, v___x_688_);
v___x_712_ = lean_string_append(v___x_707_, v___x_711_);
lean_dec_ref(v___x_711_);
return v___x_712_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux___boxed(lean_object* v_g_723_, lean_object* v_x_724_, lean_object* v_x_725_){
_start:
{
lean_object* v_res_726_; 
v_res_726_ = lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux(v_g_723_, v_x_724_, v_x_725_);
lean_dec_ref(v_g_723_);
return v_res_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__3_splitter___redArg(lean_object* v_x_727_, lean_object* v_x_728_, lean_object* v_h__1_729_, lean_object* v_h__2_730_, lean_object* v_h__3_731_){
_start:
{
if (lean_obj_tag(v_x_727_) == 0)
{
lean_dec(v_h__3_731_);
if (lean_obj_tag(v_x_728_) == 0)
{
lean_object* v___x_732_; lean_object* v___x_733_; 
lean_dec(v_h__2_730_);
v___x_732_ = lean_box(0);
v___x_733_ = lean_apply_1(v_h__1_729_, v___x_732_);
return v___x_733_;
}
else
{
lean_object* v_head_734_; lean_object* v_tail_735_; lean_object* v___x_736_; 
lean_dec(v_h__1_729_);
v_head_734_ = lean_ctor_get(v_x_728_, 0);
lean_inc(v_head_734_);
v_tail_735_ = lean_ctor_get(v_x_728_, 1);
lean_inc(v_tail_735_);
lean_dec_ref_known(v_x_728_, 2);
v___x_736_ = lean_apply_2(v_h__2_730_, v_head_734_, v_tail_735_);
return v___x_736_;
}
}
else
{
lean_object* v_head_737_; lean_object* v_tail_738_; lean_object* v___x_739_; 
lean_dec(v_h__2_730_);
lean_dec(v_h__1_729_);
v_head_737_ = lean_ctor_get(v_x_727_, 0);
lean_inc(v_head_737_);
v_tail_738_ = lean_ctor_get(v_x_727_, 1);
lean_inc(v_tail_738_);
lean_dec_ref_known(v_x_727_, 2);
v___x_739_ = lean_apply_3(v_h__3_731_, v_head_737_, v_tail_738_, v_x_728_);
return v___x_739_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__3_splitter(lean_object* v_motive_740_, lean_object* v_x_741_, lean_object* v_x_742_, lean_object* v_h__1_743_, lean_object* v_h__2_744_, lean_object* v_h__3_745_){
_start:
{
if (lean_obj_tag(v_x_741_) == 0)
{
lean_dec(v_h__3_745_);
if (lean_obj_tag(v_x_742_) == 0)
{
lean_object* v___x_746_; lean_object* v___x_747_; 
lean_dec(v_h__2_744_);
v___x_746_ = lean_box(0);
v___x_747_ = lean_apply_1(v_h__1_743_, v___x_746_);
return v___x_747_;
}
else
{
lean_object* v_head_748_; lean_object* v_tail_749_; lean_object* v___x_750_; 
lean_dec(v_h__1_743_);
v_head_748_ = lean_ctor_get(v_x_742_, 0);
lean_inc(v_head_748_);
v_tail_749_ = lean_ctor_get(v_x_742_, 1);
lean_inc(v_tail_749_);
lean_dec_ref_known(v_x_742_, 2);
v___x_750_ = lean_apply_2(v_h__2_744_, v_head_748_, v_tail_749_);
return v___x_750_;
}
}
else
{
lean_object* v_head_751_; lean_object* v_tail_752_; lean_object* v___x_753_; 
lean_dec(v_h__2_744_);
lean_dec(v_h__1_743_);
v_head_751_ = lean_ctor_get(v_x_741_, 0);
lean_inc(v_head_751_);
v_tail_752_ = lean_ctor_get(v_x_741_, 1);
lean_inc(v_tail_752_);
lean_dec_ref_known(v_x_741_, 2);
v___x_753_ = lean_apply_3(v_h__3_745_, v_head_751_, v_tail_752_, v_x_742_);
return v___x_753_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__1_splitter___redArg(lean_object* v_x_754_, lean_object* v_h__1_755_, lean_object* v_h__2_756_){
_start:
{
if (lean_obj_tag(v_x_754_) == 0)
{
lean_object* v___x_757_; lean_object* v___x_758_; 
lean_dec(v_h__1_755_);
v___x_757_ = lean_box(0);
v___x_758_ = lean_apply_1(v_h__2_756_, v___x_757_);
return v___x_758_;
}
else
{
lean_object* v_val_759_; lean_object* v___x_760_; 
lean_dec(v_h__2_756_);
v_val_759_ = lean_ctor_get(v_x_754_, 0);
lean_inc(v_val_759_);
lean_dec_ref_known(v_x_754_, 1);
v___x_760_ = lean_apply_1(v_h__1_755_, v_val_759_);
return v___x_760_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Translate_GuessName_0__Mathlib_Tactic_GuessName_fixAbbreviationAux_match__1_splitter(lean_object* v_motive_761_, lean_object* v_x_762_, lean_object* v_h__1_763_, lean_object* v_h__2_764_){
_start:
{
if (lean_obj_tag(v_x_762_) == 0)
{
lean_object* v___x_765_; lean_object* v___x_766_; 
lean_dec(v_h__1_763_);
v___x_765_ = lean_box(0);
v___x_766_ = lean_apply_1(v_h__2_764_, v___x_765_);
return v___x_766_;
}
else
{
lean_object* v_val_767_; lean_object* v___x_768_; 
lean_dec(v_h__2_764_);
v_val_767_ = lean_ctor_get(v_x_762_, 0);
lean_inc(v_val_767_);
lean_dec_ref_known(v_x_762_, 1);
v___x_768_ = lean_apply_1(v_h__1_763_, v_val_767_);
return v___x_768_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviation(lean_object* v_g_769_, lean_object* v_l_770_){
_start:
{
lean_object* v___x_771_; lean_object* v___x_772_; 
v___x_771_ = lean_box(0);
v___x_772_ = lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviationAux(v_g_769_, v_l_770_, v___x_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviation___boxed(lean_object* v_g_773_, lean_object* v_l_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviation(v_g_773_, v_l_774_);
lean_dec_ref(v_g_773_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName___lam__0(lean_object* v_g_776_, lean_object* v_s_777_){
_start:
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; 
v___x_778_ = lean_unsigned_to_nat(0u);
v___x_779_ = lean_box(0);
v___x_780_ = lp_mathlib_Mathlib_Tactic_GuessName_String_splitCase(v_s_777_, v___x_778_, v___x_779_);
v___x_781_ = lp_mathlib_Mathlib_Tactic_GuessName_applyNameDict(v_g_776_, v___x_780_);
v___x_782_ = lp_mathlib_Mathlib_Tactic_GuessName_fixAbbreviation(v_g_776_, v___x_781_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName___lam__0___boxed(lean_object* v_g_783_, lean_object* v_s_784_){
_start:
{
lean_object* v_res_785_; 
v_res_785_ = lp_mathlib_Mathlib_Tactic_GuessName_guessName___lam__0(v_g_783_, v_s_784_);
lean_dec_ref(v_g_783_);
return v_res_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName(lean_object* v_g_786_, lean_object* v_a_787_){
_start:
{
lean_object* v___f_788_; uint32_t v___x_789_; lean_object* v___x_790_; 
v___f_788_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_GuessName_guessName___lam__0___boxed), 2, 1);
lean_closure_set(v___f_788_, 0, v_g_786_);
v___x_789_ = 39;
v___x_790_ = lp_mathlib_String_mapTokens(v___x_789_, v___f_788_, v_a_787_);
return v___x_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_guessName___boxed(lean_object* v_g_791_, lean_object* v_a_792_){
_start:
{
lean_object* v_res_793_; 
v_res_793_ = lp_mathlib_Mathlib_Tactic_GuessName_guessName(v_g_791_, v_a_792_);
lean_dec_ref(v_a_792_);
return v_res_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___lam__0(lean_object* v_data_794_){
_start:
{
lean_object* v___x_796_; 
v___x_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_796_, 0, v_data_794_);
return v___x_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___lam__0___boxed(lean_object* v_data_797_, lean_object* v___y_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___lam__0(v_data_797_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt(lean_object* v_data_800_){
_start:
{
lean_object* v___f_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; 
v___f_802_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___lam__0___boxed), 2, 1);
lean_closure_set(v___f_802_, 0, v_data_800_);
v___x_803_ = lean_box(0);
v___x_804_ = lean_box(2);
v___x_805_ = l_Lean_registerEnvExtension___redArg(v___f_802_, v___x_803_, v___x_804_);
return v___x_805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt___boxed(lean_object* v_data_806_, lean_object* v_a_807_){
_start:
{
lean_object* v_res_808_; 
v_res_808_ = lp_mathlib_Mathlib_Tactic_GuessName_registerGuessNameExt(v_data_806_);
return v_res_808_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___redArg(lean_object* v_a_809_, lean_object* v_x_810_){
_start:
{
if (lean_obj_tag(v_x_810_) == 0)
{
uint8_t v___x_811_; 
v___x_811_ = 0;
return v___x_811_;
}
else
{
lean_object* v_key_812_; lean_object* v_tail_813_; uint8_t v___x_814_; 
v_key_812_ = lean_ctor_get(v_x_810_, 0);
v_tail_813_ = lean_ctor_get(v_x_810_, 2);
v___x_814_ = lean_string_dec_eq(v_key_812_, v_a_809_);
if (v___x_814_ == 0)
{
v_x_810_ = v_tail_813_;
goto _start;
}
else
{
return v___x_814_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___redArg___boxed(lean_object* v_a_816_, lean_object* v_x_817_){
_start:
{
uint8_t v_res_818_; lean_object* v_r_819_; 
v_res_818_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___redArg(v_a_816_, v_x_817_);
lean_dec(v_x_817_);
lean_dec_ref(v_a_816_);
v_r_819_ = lean_box(v_res_818_);
return v_r_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__2___redArg(lean_object* v_a_820_, lean_object* v_b_821_, lean_object* v_x_822_){
_start:
{
if (lean_obj_tag(v_x_822_) == 0)
{
lean_dec(v_b_821_);
lean_dec_ref(v_a_820_);
return v_x_822_;
}
else
{
lean_object* v_key_823_; lean_object* v_value_824_; lean_object* v_tail_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_837_; 
v_key_823_ = lean_ctor_get(v_x_822_, 0);
v_value_824_ = lean_ctor_get(v_x_822_, 1);
v_tail_825_ = lean_ctor_get(v_x_822_, 2);
v_isSharedCheck_837_ = !lean_is_exclusive(v_x_822_);
if (v_isSharedCheck_837_ == 0)
{
v___x_827_ = v_x_822_;
v_isShared_828_ = v_isSharedCheck_837_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_tail_825_);
lean_inc(v_value_824_);
lean_inc(v_key_823_);
lean_dec(v_x_822_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_837_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
uint8_t v___x_829_; 
v___x_829_ = lean_string_dec_eq(v_key_823_, v_a_820_);
if (v___x_829_ == 0)
{
lean_object* v___x_830_; lean_object* v___x_832_; 
v___x_830_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__2___redArg(v_a_820_, v_b_821_, v_tail_825_);
if (v_isShared_828_ == 0)
{
lean_ctor_set(v___x_827_, 2, v___x_830_);
v___x_832_ = v___x_827_;
goto v_reusejp_831_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v_key_823_);
lean_ctor_set(v_reuseFailAlloc_833_, 1, v_value_824_);
lean_ctor_set(v_reuseFailAlloc_833_, 2, v___x_830_);
v___x_832_ = v_reuseFailAlloc_833_;
goto v_reusejp_831_;
}
v_reusejp_831_:
{
return v___x_832_;
}
}
else
{
lean_object* v___x_835_; 
lean_dec(v_value_824_);
lean_dec(v_key_823_);
if (v_isShared_828_ == 0)
{
lean_ctor_set(v___x_827_, 1, v_b_821_);
lean_ctor_set(v___x_827_, 0, v_a_820_);
v___x_835_ = v___x_827_;
goto v_reusejp_834_;
}
else
{
lean_object* v_reuseFailAlloc_836_; 
v_reuseFailAlloc_836_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_836_, 0, v_a_820_);
lean_ctor_set(v_reuseFailAlloc_836_, 1, v_b_821_);
lean_ctor_set(v_reuseFailAlloc_836_, 2, v_tail_825_);
v___x_835_ = v_reuseFailAlloc_836_;
goto v_reusejp_834_;
}
v_reusejp_834_:
{
return v___x_835_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_838_, lean_object* v_x_839_){
_start:
{
if (lean_obj_tag(v_x_839_) == 0)
{
return v_x_838_;
}
else
{
lean_object* v_key_840_; lean_object* v_value_841_; lean_object* v_tail_842_; lean_object* v___x_844_; uint8_t v_isShared_845_; uint8_t v_isSharedCheck_865_; 
v_key_840_ = lean_ctor_get(v_x_839_, 0);
v_value_841_ = lean_ctor_get(v_x_839_, 1);
v_tail_842_ = lean_ctor_get(v_x_839_, 2);
v_isSharedCheck_865_ = !lean_is_exclusive(v_x_839_);
if (v_isSharedCheck_865_ == 0)
{
v___x_844_ = v_x_839_;
v_isShared_845_ = v_isSharedCheck_865_;
goto v_resetjp_843_;
}
else
{
lean_inc(v_tail_842_);
lean_inc(v_value_841_);
lean_inc(v_key_840_);
lean_dec(v_x_839_);
v___x_844_ = lean_box(0);
v_isShared_845_ = v_isSharedCheck_865_;
goto v_resetjp_843_;
}
v_resetjp_843_:
{
lean_object* v___x_846_; uint64_t v___x_847_; uint64_t v___x_848_; uint64_t v___x_849_; uint64_t v_fold_850_; uint64_t v___x_851_; uint64_t v___x_852_; uint64_t v___x_853_; size_t v___x_854_; size_t v___x_855_; size_t v___x_856_; size_t v___x_857_; size_t v___x_858_; lean_object* v___x_859_; lean_object* v___x_861_; 
v___x_846_ = lean_array_get_size(v_x_838_);
v___x_847_ = lean_string_hash(v_key_840_);
v___x_848_ = 32ULL;
v___x_849_ = lean_uint64_shift_right(v___x_847_, v___x_848_);
v_fold_850_ = lean_uint64_xor(v___x_847_, v___x_849_);
v___x_851_ = 16ULL;
v___x_852_ = lean_uint64_shift_right(v_fold_850_, v___x_851_);
v___x_853_ = lean_uint64_xor(v_fold_850_, v___x_852_);
v___x_854_ = lean_uint64_to_usize(v___x_853_);
v___x_855_ = lean_usize_of_nat(v___x_846_);
v___x_856_ = ((size_t)1ULL);
v___x_857_ = lean_usize_sub(v___x_855_, v___x_856_);
v___x_858_ = lean_usize_land(v___x_854_, v___x_857_);
v___x_859_ = lean_array_uget_borrowed(v_x_838_, v___x_858_);
lean_inc(v___x_859_);
if (v_isShared_845_ == 0)
{
lean_ctor_set(v___x_844_, 2, v___x_859_);
v___x_861_ = v___x_844_;
goto v_reusejp_860_;
}
else
{
lean_object* v_reuseFailAlloc_864_; 
v_reuseFailAlloc_864_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_864_, 0, v_key_840_);
lean_ctor_set(v_reuseFailAlloc_864_, 1, v_value_841_);
lean_ctor_set(v_reuseFailAlloc_864_, 2, v___x_859_);
v___x_861_ = v_reuseFailAlloc_864_;
goto v_reusejp_860_;
}
v_reusejp_860_:
{
lean_object* v___x_862_; 
v___x_862_ = lean_array_uset(v_x_838_, v___x_858_, v___x_861_);
v_x_838_ = v___x_862_;
v_x_839_ = v_tail_842_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2___redArg(lean_object* v_i_866_, lean_object* v_source_867_, lean_object* v_target_868_){
_start:
{
lean_object* v___x_869_; uint8_t v___x_870_; 
v___x_869_ = lean_array_get_size(v_source_867_);
v___x_870_ = lean_nat_dec_lt(v_i_866_, v___x_869_);
if (v___x_870_ == 0)
{
lean_dec_ref(v_source_867_);
lean_dec(v_i_866_);
return v_target_868_;
}
else
{
lean_object* v_es_871_; lean_object* v___x_872_; lean_object* v_source_873_; lean_object* v_target_874_; lean_object* v___x_875_; lean_object* v___x_876_; 
v_es_871_ = lean_array_fget(v_source_867_, v_i_866_);
v___x_872_ = lean_box(0);
v_source_873_ = lean_array_fset(v_source_867_, v_i_866_, v___x_872_);
v_target_874_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2_spec__4___redArg(v_target_868_, v_es_871_);
v___x_875_ = lean_unsigned_to_nat(1u);
v___x_876_ = lean_nat_add(v_i_866_, v___x_875_);
lean_dec(v_i_866_);
v_i_866_ = v___x_876_;
v_source_867_ = v_source_873_;
v_target_868_ = v_target_874_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1___redArg(lean_object* v_data_878_){
_start:
{
lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v_nbuckets_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; 
v___x_879_ = lean_array_get_size(v_data_878_);
v___x_880_ = lean_unsigned_to_nat(2u);
v_nbuckets_881_ = lean_nat_mul(v___x_879_, v___x_880_);
v___x_882_ = lean_unsigned_to_nat(0u);
v___x_883_ = lean_box(0);
v___x_884_ = lean_mk_array(v_nbuckets_881_, v___x_883_);
v___x_885_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2___redArg(v___x_882_, v_data_878_, v___x_884_);
return v___x_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0___redArg(lean_object* v_m_886_, lean_object* v_a_887_, lean_object* v_b_888_){
_start:
{
lean_object* v_size_889_; lean_object* v_buckets_890_; lean_object* v___x_892_; uint8_t v_isShared_893_; uint8_t v_isSharedCheck_933_; 
v_size_889_ = lean_ctor_get(v_m_886_, 0);
v_buckets_890_ = lean_ctor_get(v_m_886_, 1);
v_isSharedCheck_933_ = !lean_is_exclusive(v_m_886_);
if (v_isSharedCheck_933_ == 0)
{
v___x_892_ = v_m_886_;
v_isShared_893_ = v_isSharedCheck_933_;
goto v_resetjp_891_;
}
else
{
lean_inc(v_buckets_890_);
lean_inc(v_size_889_);
lean_dec(v_m_886_);
v___x_892_ = lean_box(0);
v_isShared_893_ = v_isSharedCheck_933_;
goto v_resetjp_891_;
}
v_resetjp_891_:
{
lean_object* v___x_894_; uint64_t v___x_895_; uint64_t v___x_896_; uint64_t v___x_897_; uint64_t v_fold_898_; uint64_t v___x_899_; uint64_t v___x_900_; uint64_t v___x_901_; size_t v___x_902_; size_t v___x_903_; size_t v___x_904_; size_t v___x_905_; size_t v___x_906_; lean_object* v_bkt_907_; uint8_t v___x_908_; 
v___x_894_ = lean_array_get_size(v_buckets_890_);
v___x_895_ = lean_string_hash(v_a_887_);
v___x_896_ = 32ULL;
v___x_897_ = lean_uint64_shift_right(v___x_895_, v___x_896_);
v_fold_898_ = lean_uint64_xor(v___x_895_, v___x_897_);
v___x_899_ = 16ULL;
v___x_900_ = lean_uint64_shift_right(v_fold_898_, v___x_899_);
v___x_901_ = lean_uint64_xor(v_fold_898_, v___x_900_);
v___x_902_ = lean_uint64_to_usize(v___x_901_);
v___x_903_ = lean_usize_of_nat(v___x_894_);
v___x_904_ = ((size_t)1ULL);
v___x_905_ = lean_usize_sub(v___x_903_, v___x_904_);
v___x_906_ = lean_usize_land(v___x_902_, v___x_905_);
v_bkt_907_ = lean_array_uget_borrowed(v_buckets_890_, v___x_906_);
v___x_908_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___redArg(v_a_887_, v_bkt_907_);
if (v___x_908_ == 0)
{
lean_object* v___x_909_; lean_object* v_size_x27_910_; lean_object* v___x_911_; lean_object* v_buckets_x27_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; uint8_t v___x_918_; 
v___x_909_ = lean_unsigned_to_nat(1u);
v_size_x27_910_ = lean_nat_add(v_size_889_, v___x_909_);
lean_dec(v_size_889_);
lean_inc(v_bkt_907_);
v___x_911_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_911_, 0, v_a_887_);
lean_ctor_set(v___x_911_, 1, v_b_888_);
lean_ctor_set(v___x_911_, 2, v_bkt_907_);
v_buckets_x27_912_ = lean_array_uset(v_buckets_890_, v___x_906_, v___x_911_);
v___x_913_ = lean_unsigned_to_nat(4u);
v___x_914_ = lean_nat_mul(v_size_x27_910_, v___x_913_);
v___x_915_ = lean_unsigned_to_nat(3u);
v___x_916_ = lean_nat_div(v___x_914_, v___x_915_);
lean_dec(v___x_914_);
v___x_917_ = lean_array_get_size(v_buckets_x27_912_);
v___x_918_ = lean_nat_dec_le(v___x_916_, v___x_917_);
lean_dec(v___x_916_);
if (v___x_918_ == 0)
{
lean_object* v_val_919_; lean_object* v___x_921_; 
v_val_919_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1___redArg(v_buckets_x27_912_);
if (v_isShared_893_ == 0)
{
lean_ctor_set(v___x_892_, 1, v_val_919_);
lean_ctor_set(v___x_892_, 0, v_size_x27_910_);
v___x_921_ = v___x_892_;
goto v_reusejp_920_;
}
else
{
lean_object* v_reuseFailAlloc_922_; 
v_reuseFailAlloc_922_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_922_, 0, v_size_x27_910_);
lean_ctor_set(v_reuseFailAlloc_922_, 1, v_val_919_);
v___x_921_ = v_reuseFailAlloc_922_;
goto v_reusejp_920_;
}
v_reusejp_920_:
{
return v___x_921_;
}
}
else
{
lean_object* v___x_924_; 
if (v_isShared_893_ == 0)
{
lean_ctor_set(v___x_892_, 1, v_buckets_x27_912_);
lean_ctor_set(v___x_892_, 0, v_size_x27_910_);
v___x_924_ = v___x_892_;
goto v_reusejp_923_;
}
else
{
lean_object* v_reuseFailAlloc_925_; 
v_reuseFailAlloc_925_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_925_, 0, v_size_x27_910_);
lean_ctor_set(v_reuseFailAlloc_925_, 1, v_buckets_x27_912_);
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
lean_object* v___x_926_; lean_object* v_buckets_x27_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_931_; 
lean_inc(v_bkt_907_);
v___x_926_ = lean_box(0);
v_buckets_x27_927_ = lean_array_uset(v_buckets_890_, v___x_906_, v___x_926_);
v___x_928_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__2___redArg(v_a_887_, v_b_888_, v_bkt_907_);
v___x_929_ = lean_array_uset(v_buckets_x27_927_, v___x_906_, v___x_928_);
if (v_isShared_893_ == 0)
{
lean_ctor_set(v___x_892_, 1, v___x_929_);
v___x_931_ = v___x_892_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_932_; 
v_reuseFailAlloc_932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_932_, 0, v_size_889_);
lean_ctor_set(v_reuseFailAlloc_932_, 1, v___x_929_);
v___x_931_ = v_reuseFailAlloc_932_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
return v___x_931_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___lam__0(lean_object* v_src_934_, lean_object* v_tgt_935_, lean_object* v_data_936_){
_start:
{
lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_950_; lean_object* v___x_951_; 
v___x_937_ = lean_unsigned_to_nat(0u);
v___x_938_ = lp_mathlib_Mathlib_Tactic_GuessName_String_decapitalizeSeq(v_src_934_, v___x_937_);
v___x_950_ = lean_box(0);
lean_inc_ref(v___x_938_);
v___x_951_ = lp_mathlib_Mathlib_Tactic_GuessName_String_splitCase(v___x_938_, v___x_937_, v___x_950_);
if (lean_obj_tag(v___x_951_) == 1)
{
lean_object* v_tail_952_; 
v_tail_952_ = lean_ctor_get(v___x_951_, 1);
lean_inc(v_tail_952_);
lean_dec_ref_known(v___x_951_, 2);
if (lean_obj_tag(v_tail_952_) == 0)
{
lean_object* v_nameDict_953_; lean_object* v_abbreviationDict_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_963_; 
v_nameDict_953_ = lean_ctor_get(v_data_936_, 0);
v_abbreviationDict_954_ = lean_ctor_get(v_data_936_, 1);
v_isSharedCheck_963_ = !lean_is_exclusive(v_data_936_);
if (v_isSharedCheck_963_ == 0)
{
v___x_956_ = v_data_936_;
v_isShared_957_ = v_isSharedCheck_963_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_abbreviationDict_954_);
lean_inc(v_nameDict_953_);
lean_dec(v_data_936_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_963_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_961_; 
v___x_958_ = lp_mathlib_Mathlib_Tactic_GuessName_String_splitCase(v_tgt_935_, v___x_937_, v___x_950_);
v___x_959_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0___redArg(v_nameDict_953_, v___x_938_, v___x_958_);
if (v_isShared_957_ == 0)
{
lean_ctor_set(v___x_956_, 0, v___x_959_);
v___x_961_ = v___x_956_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v___x_959_);
lean_ctor_set(v_reuseFailAlloc_962_, 1, v_abbreviationDict_954_);
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
lean_dec(v_tail_952_);
goto v___jp_939_;
}
}
else
{
lean_dec(v___x_951_);
goto v___jp_939_;
}
v___jp_939_:
{
lean_object* v_nameDict_940_; lean_object* v_abbreviationDict_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_949_; 
v_nameDict_940_ = lean_ctor_get(v_data_936_, 0);
v_abbreviationDict_941_ = lean_ctor_get(v_data_936_, 1);
v_isSharedCheck_949_ = !lean_is_exclusive(v_data_936_);
if (v_isSharedCheck_949_ == 0)
{
v___x_943_ = v_data_936_;
v_isShared_944_ = v_isSharedCheck_949_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_abbreviationDict_941_);
lean_inc(v_nameDict_940_);
lean_dec(v_data_936_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_949_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_945_; lean_object* v___x_947_; 
v___x_945_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0___redArg(v_abbreviationDict_941_, v___x_938_, v_tgt_935_);
if (v_isShared_944_ == 0)
{
lean_ctor_set(v___x_943_, 1, v___x_945_);
v___x_947_ = v___x_943_;
goto v_reusejp_946_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v_nameDict_940_);
lean_ctor_set(v_reuseFailAlloc_948_, 1, v___x_945_);
v___x_947_ = v_reuseFailAlloc_948_;
goto v_reusejp_946_;
}
v_reusejp_946_:
{
return v___x_947_;
}
}
}
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0(void){
_start:
{
lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_964_ = lean_box(1);
v___x_965_ = l_Lean_MessageData_ofFormat(v___x_964_);
return v___x_965_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__3(void){
_start:
{
lean_object* v___x_969_; lean_object* v___x_970_; 
v___x_969_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__2));
v___x_970_ = l_Lean_MessageData_ofFormat(v___x_969_);
return v___x_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10(lean_object* v_x_971_, lean_object* v_x_972_){
_start:
{
if (lean_obj_tag(v_x_972_) == 0)
{
return v_x_971_;
}
else
{
lean_object* v_head_973_; lean_object* v_tail_974_; lean_object* v___x_976_; uint8_t v_isShared_977_; uint8_t v_isSharedCheck_996_; 
v_head_973_ = lean_ctor_get(v_x_972_, 0);
v_tail_974_ = lean_ctor_get(v_x_972_, 1);
v_isSharedCheck_996_ = !lean_is_exclusive(v_x_972_);
if (v_isSharedCheck_996_ == 0)
{
v___x_976_ = v_x_972_;
v_isShared_977_ = v_isSharedCheck_996_;
goto v_resetjp_975_;
}
else
{
lean_inc(v_tail_974_);
lean_inc(v_head_973_);
lean_dec(v_x_972_);
v___x_976_ = lean_box(0);
v_isShared_977_ = v_isSharedCheck_996_;
goto v_resetjp_975_;
}
v_resetjp_975_:
{
lean_object* v_before_978_; lean_object* v___x_980_; uint8_t v_isShared_981_; uint8_t v_isSharedCheck_994_; 
v_before_978_ = lean_ctor_get(v_head_973_, 0);
v_isSharedCheck_994_ = !lean_is_exclusive(v_head_973_);
if (v_isSharedCheck_994_ == 0)
{
lean_object* v_unused_995_; 
v_unused_995_ = lean_ctor_get(v_head_973_, 1);
lean_dec(v_unused_995_);
v___x_980_ = v_head_973_;
v_isShared_981_ = v_isSharedCheck_994_;
goto v_resetjp_979_;
}
else
{
lean_inc(v_before_978_);
lean_dec(v_head_973_);
v___x_980_ = lean_box(0);
v_isShared_981_ = v_isSharedCheck_994_;
goto v_resetjp_979_;
}
v_resetjp_979_:
{
lean_object* v___x_982_; lean_object* v___x_984_; 
v___x_982_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0);
if (v_isShared_981_ == 0)
{
lean_ctor_set_tag(v___x_980_, 7);
lean_ctor_set(v___x_980_, 1, v___x_982_);
lean_ctor_set(v___x_980_, 0, v_x_971_);
v___x_984_ = v___x_980_;
goto v_reusejp_983_;
}
else
{
lean_object* v_reuseFailAlloc_993_; 
v_reuseFailAlloc_993_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_993_, 0, v_x_971_);
lean_ctor_set(v_reuseFailAlloc_993_, 1, v___x_982_);
v___x_984_ = v_reuseFailAlloc_993_;
goto v_reusejp_983_;
}
v_reusejp_983_:
{
lean_object* v___x_985_; lean_object* v___x_987_; 
v___x_985_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__3);
if (v_isShared_977_ == 0)
{
lean_ctor_set_tag(v___x_976_, 7);
lean_ctor_set(v___x_976_, 1, v___x_985_);
lean_ctor_set(v___x_976_, 0, v___x_984_);
v___x_987_ = v___x_976_;
goto v_reusejp_986_;
}
else
{
lean_object* v_reuseFailAlloc_992_; 
v_reuseFailAlloc_992_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_992_, 0, v___x_984_);
lean_ctor_set(v_reuseFailAlloc_992_, 1, v___x_985_);
v___x_987_ = v_reuseFailAlloc_992_;
goto v_reusejp_986_;
}
v_reusejp_986_:
{
lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; 
v___x_988_ = l_Lean_MessageData_ofSyntax(v_before_978_);
v___x_989_ = l_Lean_indentD(v___x_988_);
v___x_990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_990_, 0, v___x_987_);
lean_ctor_set(v___x_990_, 1, v___x_989_);
v_x_971_ = v___x_990_;
v_x_972_ = v_tail_974_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__9(lean_object* v_opts_997_, lean_object* v_opt_998_){
_start:
{
lean_object* v_name_999_; lean_object* v_defValue_1000_; lean_object* v_map_1001_; lean_object* v___x_1002_; 
v_name_999_ = lean_ctor_get(v_opt_998_, 0);
v_defValue_1000_ = lean_ctor_get(v_opt_998_, 1);
v_map_1001_ = lean_ctor_get(v_opts_997_, 0);
v___x_1002_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1001_, v_name_999_);
if (lean_obj_tag(v___x_1002_) == 0)
{
uint8_t v___x_1003_; 
v___x_1003_ = lean_unbox(v_defValue_1000_);
return v___x_1003_;
}
else
{
lean_object* v_val_1004_; 
v_val_1004_ = lean_ctor_get(v___x_1002_, 0);
lean_inc(v_val_1004_);
lean_dec_ref_known(v___x_1002_, 1);
if (lean_obj_tag(v_val_1004_) == 1)
{
uint8_t v_v_1005_; 
v_v_1005_ = lean_ctor_get_uint8(v_val_1004_, 0);
lean_dec_ref_known(v_val_1004_, 0);
return v_v_1005_;
}
else
{
uint8_t v___x_1006_; 
lean_dec(v_val_1004_);
v___x_1006_ = lean_unbox(v_defValue_1000_);
return v___x_1006_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__9___boxed(lean_object* v_opts_1007_, lean_object* v_opt_1008_){
_start:
{
uint8_t v_res_1009_; lean_object* v_r_1010_; 
v_res_1009_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__9(v_opts_1007_, v_opt_1008_);
lean_dec_ref(v_opt_1008_);
lean_dec_ref(v_opts_1007_);
v_r_1010_ = lean_box(v_res_1009_);
return v_r_1010_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__2(void){
_start:
{
lean_object* v___x_1014_; lean_object* v___x_1015_; 
v___x_1014_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__1));
v___x_1015_ = l_Lean_MessageData_ofFormat(v___x_1014_);
return v___x_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg(lean_object* v_msgData_1016_, lean_object* v_macroStack_1017_, lean_object* v___y_1018_){
_start:
{
lean_object* v___x_1020_; lean_object* v_scopes_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v_opts_1024_; lean_object* v___x_1025_; uint8_t v___x_1026_; 
v___x_1020_ = lean_st_ref_get(v___y_1018_);
v_scopes_1021_ = lean_ctor_get(v___x_1020_, 2);
lean_inc(v_scopes_1021_);
lean_dec(v___x_1020_);
v___x_1022_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1023_ = l_List_head_x21___redArg(v___x_1022_, v_scopes_1021_);
lean_dec(v_scopes_1021_);
v_opts_1024_ = lean_ctor_get(v___x_1023_, 1);
lean_inc_ref(v_opts_1024_);
lean_dec(v___x_1023_);
v___x_1025_ = l_Lean_Elab_pp_macroStack;
v___x_1026_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__9(v_opts_1024_, v___x_1025_);
lean_dec_ref(v_opts_1024_);
if (v___x_1026_ == 0)
{
lean_object* v___x_1027_; 
lean_dec(v_macroStack_1017_);
v___x_1027_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1027_, 0, v_msgData_1016_);
return v___x_1027_;
}
else
{
if (lean_obj_tag(v_macroStack_1017_) == 0)
{
lean_object* v___x_1028_; 
v___x_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1028_, 0, v_msgData_1016_);
return v___x_1028_;
}
else
{
lean_object* v_head_1029_; lean_object* v_after_1030_; lean_object* v___x_1032_; uint8_t v_isShared_1033_; uint8_t v_isSharedCheck_1045_; 
v_head_1029_ = lean_ctor_get(v_macroStack_1017_, 0);
lean_inc(v_head_1029_);
v_after_1030_ = lean_ctor_get(v_head_1029_, 1);
v_isSharedCheck_1045_ = !lean_is_exclusive(v_head_1029_);
if (v_isSharedCheck_1045_ == 0)
{
lean_object* v_unused_1046_; 
v_unused_1046_ = lean_ctor_get(v_head_1029_, 0);
lean_dec(v_unused_1046_);
v___x_1032_ = v_head_1029_;
v_isShared_1033_ = v_isSharedCheck_1045_;
goto v_resetjp_1031_;
}
else
{
lean_inc(v_after_1030_);
lean_dec(v_head_1029_);
v___x_1032_ = lean_box(0);
v_isShared_1033_ = v_isSharedCheck_1045_;
goto v_resetjp_1031_;
}
v_resetjp_1031_:
{
lean_object* v___x_1034_; lean_object* v___x_1036_; 
v___x_1034_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10___closed__0);
if (v_isShared_1033_ == 0)
{
lean_ctor_set_tag(v___x_1032_, 7);
lean_ctor_set(v___x_1032_, 1, v___x_1034_);
lean_ctor_set(v___x_1032_, 0, v_msgData_1016_);
v___x_1036_ = v___x_1032_;
goto v_reusejp_1035_;
}
else
{
lean_object* v_reuseFailAlloc_1044_; 
v_reuseFailAlloc_1044_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1044_, 0, v_msgData_1016_);
lean_ctor_set(v_reuseFailAlloc_1044_, 1, v___x_1034_);
v___x_1036_ = v_reuseFailAlloc_1044_;
goto v_reusejp_1035_;
}
v_reusejp_1035_:
{
lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v_msgData_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; 
v___x_1037_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___closed__2);
v___x_1038_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1036_);
lean_ctor_set(v___x_1038_, 1, v___x_1037_);
v___x_1039_ = l_Lean_MessageData_ofSyntax(v_after_1030_);
v___x_1040_ = l_Lean_indentD(v___x_1039_);
v_msgData_1041_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1041_, 0, v___x_1038_);
lean_ctor_set(v_msgData_1041_, 1, v___x_1040_);
v___x_1042_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7_spec__10(v_msgData_1041_, v_macroStack_1017_);
v___x_1043_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1043_, 0, v___x_1042_);
return v___x_1043_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg___boxed(lean_object* v_msgData_1047_, lean_object* v_macroStack_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
lean_object* v_res_1051_; 
v_res_1051_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg(v_msgData_1047_, v_macroStack_1048_, v___y_1049_);
lean_dec(v___y_1049_);
return v_res_1051_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_1052_; 
v___x_1052_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1052_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_1053_; lean_object* v___x_1054_; 
v___x_1053_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__0);
v___x_1054_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1054_, 0, v___x_1053_);
return v___x_1054_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___x_1055_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1);
v___x_1056_ = lean_unsigned_to_nat(0u);
v___x_1057_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1057_, 0, v___x_1056_);
lean_ctor_set(v___x_1057_, 1, v___x_1056_);
lean_ctor_set(v___x_1057_, 2, v___x_1056_);
lean_ctor_set(v___x_1057_, 3, v___x_1056_);
lean_ctor_set(v___x_1057_, 4, v___x_1055_);
lean_ctor_set(v___x_1057_, 5, v___x_1055_);
lean_ctor_set(v___x_1057_, 6, v___x_1055_);
lean_ctor_set(v___x_1057_, 7, v___x_1055_);
lean_ctor_set(v___x_1057_, 8, v___x_1055_);
lean_ctor_set(v___x_1057_, 9, v___x_1055_);
return v___x_1057_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; 
v___x_1058_ = lean_unsigned_to_nat(32u);
v___x_1059_ = lean_mk_empty_array_with_capacity(v___x_1058_);
v___x_1060_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1060_, 0, v___x_1059_);
return v___x_1060_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__4(void){
_start:
{
size_t v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; 
v___x_1061_ = ((size_t)5ULL);
v___x_1062_ = lean_unsigned_to_nat(0u);
v___x_1063_ = lean_unsigned_to_nat(32u);
v___x_1064_ = lean_mk_empty_array_with_capacity(v___x_1063_);
v___x_1065_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__3);
v___x_1066_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1066_, 0, v___x_1065_);
lean_ctor_set(v___x_1066_, 1, v___x_1064_);
lean_ctor_set(v___x_1066_, 2, v___x_1062_);
lean_ctor_set(v___x_1066_, 3, v___x_1062_);
lean_ctor_set_usize(v___x_1066_, 4, v___x_1061_);
return v___x_1066_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; 
v___x_1067_ = lean_box(1);
v___x_1068_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__4);
v___x_1069_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__1);
v___x_1070_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1070_, 0, v___x_1069_);
lean_ctor_set(v___x_1070_, 1, v___x_1068_);
lean_ctor_set(v___x_1070_, 2, v___x_1067_);
return v___x_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg(lean_object* v_msgData_1071_, lean_object* v___y_1072_){
_start:
{
lean_object* v___x_1074_; lean_object* v_env_1075_; lean_object* v___x_1076_; lean_object* v_scopes_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v_opts_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; 
v___x_1074_ = lean_st_ref_get(v___y_1072_);
v_env_1075_ = lean_ctor_get(v___x_1074_, 0);
lean_inc_ref(v_env_1075_);
lean_dec(v___x_1074_);
v___x_1076_ = lean_st_ref_get(v___y_1072_);
v_scopes_1077_ = lean_ctor_get(v___x_1076_, 2);
lean_inc(v_scopes_1077_);
lean_dec(v___x_1076_);
v___x_1078_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1079_ = l_List_head_x21___redArg(v___x_1078_, v_scopes_1077_);
lean_dec(v_scopes_1077_);
v_opts_1080_ = lean_ctor_get(v___x_1079_, 1);
lean_inc_ref(v_opts_1080_);
lean_dec(v___x_1079_);
v___x_1081_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__2);
v___x_1082_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___closed__5);
v___x_1083_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1083_, 0, v_env_1075_);
lean_ctor_set(v___x_1083_, 1, v___x_1081_);
lean_ctor_set(v___x_1083_, 2, v___x_1082_);
lean_ctor_set(v___x_1083_, 3, v_opts_1080_);
v___x_1084_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1084_, 0, v___x_1083_);
lean_ctor_set(v___x_1084_, 1, v_msgData_1071_);
v___x_1085_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1085_, 0, v___x_1084_);
return v___x_1085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg___boxed(lean_object* v_msgData_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_){
_start:
{
lean_object* v_res_1089_; 
v_res_1089_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg(v_msgData_1086_, v___y_1087_);
lean_dec(v___y_1087_);
return v_res_1089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___redArg(lean_object* v_msg_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v___x_1094_; 
v___x_1094_ = l_Lean_Elab_Command_getRef___redArg(v___y_1091_);
if (lean_obj_tag(v___x_1094_) == 0)
{
lean_object* v_a_1095_; lean_object* v_macroStack_1096_; lean_object* v___x_1097_; lean_object* v_a_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v_a_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1109_; 
v_a_1095_ = lean_ctor_get(v___x_1094_, 0);
lean_inc(v_a_1095_);
lean_dec_ref_known(v___x_1094_, 1);
v_macroStack_1096_ = lean_ctor_get(v___y_1091_, 4);
v___x_1097_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg(v_msg_1090_, v___y_1092_);
v_a_1098_ = lean_ctor_get(v___x_1097_, 0);
lean_inc(v_a_1098_);
lean_dec_ref(v___x_1097_);
v___x_1099_ = l_Lean_Elab_getBetterRef(v_a_1095_, v_macroStack_1096_);
lean_dec(v_a_1095_);
lean_inc(v_macroStack_1096_);
v___x_1100_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg(v_a_1098_, v_macroStack_1096_, v___y_1092_);
v_a_1101_ = lean_ctor_get(v___x_1100_, 0);
v_isSharedCheck_1109_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1109_ == 0)
{
v___x_1103_ = v___x_1100_;
v_isShared_1104_ = v_isSharedCheck_1109_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_a_1101_);
lean_dec(v___x_1100_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1109_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v___x_1105_; lean_object* v___x_1107_; 
v___x_1105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1105_, 0, v___x_1099_);
lean_ctor_set(v___x_1105_, 1, v_a_1101_);
if (v_isShared_1104_ == 0)
{
lean_ctor_set_tag(v___x_1103_, 1);
lean_ctor_set(v___x_1103_, 0, v___x_1105_);
v___x_1107_ = v___x_1103_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v___x_1105_);
v___x_1107_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
return v___x_1107_;
}
}
}
else
{
lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1117_; 
lean_dec_ref(v_msg_1090_);
v_a_1110_ = lean_ctor_get(v___x_1094_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1112_ = v___x_1094_;
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1094_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1115_; 
if (v_isShared_1113_ == 0)
{
v___x_1115_ = v___x_1112_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_a_1110_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___redArg___boxed(lean_object* v_msg_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___redArg(v_msg_1118_, v___y_1119_, v___y_1120_);
lean_dec(v___y_1120_);
lean_dec_ref(v___y_1119_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg(lean_object* v_ref_1123_, lean_object* v_msg_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_){
_start:
{
lean_object* v___x_1128_; 
v___x_1128_ = l_Lean_Elab_Command_getRef___redArg(v___y_1125_);
if (lean_obj_tag(v___x_1128_) == 0)
{
lean_object* v_a_1129_; lean_object* v_fileName_1130_; lean_object* v_fileMap_1131_; lean_object* v_currRecDepth_1132_; lean_object* v_cmdPos_1133_; lean_object* v_macroStack_1134_; lean_object* v_quotContext_x3f_1135_; lean_object* v_currMacroScope_1136_; lean_object* v_snap_x3f_1137_; lean_object* v_cancelTk_x3f_1138_; uint8_t v_suppressElabErrors_1139_; lean_object* v_ref_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; 
v_a_1129_ = lean_ctor_get(v___x_1128_, 0);
lean_inc(v_a_1129_);
lean_dec_ref_known(v___x_1128_, 1);
v_fileName_1130_ = lean_ctor_get(v___y_1125_, 0);
v_fileMap_1131_ = lean_ctor_get(v___y_1125_, 1);
v_currRecDepth_1132_ = lean_ctor_get(v___y_1125_, 2);
v_cmdPos_1133_ = lean_ctor_get(v___y_1125_, 3);
v_macroStack_1134_ = lean_ctor_get(v___y_1125_, 4);
v_quotContext_x3f_1135_ = lean_ctor_get(v___y_1125_, 5);
v_currMacroScope_1136_ = lean_ctor_get(v___y_1125_, 6);
v_snap_x3f_1137_ = lean_ctor_get(v___y_1125_, 8);
v_cancelTk_x3f_1138_ = lean_ctor_get(v___y_1125_, 9);
v_suppressElabErrors_1139_ = lean_ctor_get_uint8(v___y_1125_, sizeof(void*)*10);
v_ref_1140_ = l_Lean_replaceRef(v_ref_1123_, v_a_1129_);
lean_dec(v_a_1129_);
lean_inc(v_cancelTk_x3f_1138_);
lean_inc(v_snap_x3f_1137_);
lean_inc(v_currMacroScope_1136_);
lean_inc(v_quotContext_x3f_1135_);
lean_inc(v_macroStack_1134_);
lean_inc(v_cmdPos_1133_);
lean_inc(v_currRecDepth_1132_);
lean_inc_ref(v_fileMap_1131_);
lean_inc_ref(v_fileName_1130_);
v___x_1141_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_1141_, 0, v_fileName_1130_);
lean_ctor_set(v___x_1141_, 1, v_fileMap_1131_);
lean_ctor_set(v___x_1141_, 2, v_currRecDepth_1132_);
lean_ctor_set(v___x_1141_, 3, v_cmdPos_1133_);
lean_ctor_set(v___x_1141_, 4, v_macroStack_1134_);
lean_ctor_set(v___x_1141_, 5, v_quotContext_x3f_1135_);
lean_ctor_set(v___x_1141_, 6, v_currMacroScope_1136_);
lean_ctor_set(v___x_1141_, 7, v_ref_1140_);
lean_ctor_set(v___x_1141_, 8, v_snap_x3f_1137_);
lean_ctor_set(v___x_1141_, 9, v_cancelTk_x3f_1138_);
lean_ctor_set_uint8(v___x_1141_, sizeof(void*)*10, v_suppressElabErrors_1139_);
v___x_1142_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___redArg(v_msg_1124_, v___x_1141_, v___y_1126_);
lean_dec_ref_known(v___x_1141_, 10);
return v___x_1142_;
}
else
{
lean_object* v_a_1143_; lean_object* v___x_1145_; uint8_t v_isShared_1146_; uint8_t v_isSharedCheck_1150_; 
lean_dec_ref(v_msg_1124_);
v_a_1143_ = lean_ctor_get(v___x_1128_, 0);
v_isSharedCheck_1150_ = !lean_is_exclusive(v___x_1128_);
if (v_isSharedCheck_1150_ == 0)
{
v___x_1145_ = v___x_1128_;
v_isShared_1146_ = v_isSharedCheck_1150_;
goto v_resetjp_1144_;
}
else
{
lean_inc(v_a_1143_);
lean_dec(v___x_1128_);
v___x_1145_ = lean_box(0);
v_isShared_1146_ = v_isSharedCheck_1150_;
goto v_resetjp_1144_;
}
v_resetjp_1144_:
{
lean_object* v___x_1148_; 
if (v_isShared_1146_ == 0)
{
v___x_1148_ = v___x_1145_;
goto v_reusejp_1147_;
}
else
{
lean_object* v_reuseFailAlloc_1149_; 
v_reuseFailAlloc_1149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1149_, 0, v_a_1143_);
v___x_1148_ = v_reuseFailAlloc_1149_;
goto v_reusejp_1147_;
}
v_reusejp_1147_:
{
return v___x_1148_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg___boxed(lean_object* v_ref_1151_, lean_object* v_msg_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_){
_start:
{
lean_object* v_res_1156_; 
v_res_1156_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg(v_ref_1151_, v_msg_1152_, v___y_1153_, v___y_1154_);
lean_dec(v___y_1154_);
lean_dec_ref(v___y_1153_);
lean_dec(v_ref_1151_);
return v_res_1156_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1(void){
_start:
{
lean_object* v___x_1158_; lean_object* v___x_1159_; 
v___x_1158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__0));
v___x_1159_ = l_Lean_stringToMessageData(v___x_1158_);
return v___x_1159_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3(void){
_start:
{
lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1161_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__2));
v___x_1162_ = l_Lean_stringToMessageData(v___x_1161_);
return v___x_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(lean_object* v_ext_1163_, lean_object* v_srcId_1164_, lean_object* v_tgtId_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_){
_start:
{
lean_object* v___x_1169_; uint8_t v___x_1170_; lean_object* v_src_1171_; lean_object* v___x_1172_; lean_object* v_tgt_1173_; lean_object* v___f_1174_; lean_object* v___y_1176_; lean_object* v___y_1204_; lean_object* v___y_1205_; lean_object* v___y_1213_; lean_object* v___y_1214_; uint32_t v___y_1215_; lean_object* v___y_1221_; lean_object* v___y_1222_; uint32_t v___y_1238_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; 
v___x_1169_ = l_Lean_TSyntax_getId(v_srcId_1164_);
v___x_1170_ = 1;
v_src_1171_ = l_Lean_Name_toString(v___x_1169_, v___x_1170_);
v___x_1172_ = l_Lean_TSyntax_getId(v_tgtId_1165_);
v_tgt_1173_ = l_Lean_Name_toString(v___x_1172_, v___x_1170_);
lean_inc_ref(v_tgt_1173_);
lean_inc_ref_n(v_src_1171_, 2);
v___f_1174_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___lam__0), 3, 2);
lean_closure_set(v___f_1174_, 0, v_src_1171_);
lean_closure_set(v___f_1174_, 1, v_tgt_1173_);
v___x_1243_ = lean_unsigned_to_nat(0u);
v___x_1244_ = lean_string_utf8_byte_size(v_src_1171_);
v___x_1245_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1245_, 0, v_src_1171_);
lean_ctor_set(v___x_1245_, 1, v___x_1243_);
lean_ctor_set(v___x_1245_, 2, v___x_1244_);
v___x_1246_ = l_String_Slice_Pos_get_x3f(v___x_1245_, v___x_1243_);
lean_dec_ref_known(v___x_1245_, 3);
if (lean_obj_tag(v___x_1246_) == 0)
{
uint32_t v___x_1247_; 
v___x_1247_ = 65;
v___y_1238_ = v___x_1247_;
goto v___jp_1237_;
}
else
{
lean_object* v_val_1248_; uint32_t v___x_1249_; 
v_val_1248_ = lean_ctor_get(v___x_1246_, 0);
lean_inc(v_val_1248_);
lean_dec_ref_known(v___x_1246_, 1);
v___x_1249_ = lean_unbox_uint32(v_val_1248_);
lean_dec(v_val_1248_);
v___y_1238_ = v___x_1249_;
goto v___jp_1237_;
}
v___jp_1175_:
{
lean_object* v___x_1177_; lean_object* v_env_1178_; lean_object* v_messages_1179_; lean_object* v_scopes_1180_; lean_object* v_usedQuotCtxts_1181_; lean_object* v_nextMacroScope_1182_; lean_object* v_maxRecDepth_1183_; lean_object* v_ngen_1184_; lean_object* v_auxDeclNGen_1185_; lean_object* v_infoState_1186_; lean_object* v_traceState_1187_; lean_object* v_snapshotTasks_1188_; lean_object* v_prevLinterStates_1189_; lean_object* v___x_1191_; uint8_t v_isShared_1192_; uint8_t v_isSharedCheck_1202_; 
v___x_1177_ = lean_st_ref_take(v___y_1176_);
v_env_1178_ = lean_ctor_get(v___x_1177_, 0);
v_messages_1179_ = lean_ctor_get(v___x_1177_, 1);
v_scopes_1180_ = lean_ctor_get(v___x_1177_, 2);
v_usedQuotCtxts_1181_ = lean_ctor_get(v___x_1177_, 3);
v_nextMacroScope_1182_ = lean_ctor_get(v___x_1177_, 4);
v_maxRecDepth_1183_ = lean_ctor_get(v___x_1177_, 5);
v_ngen_1184_ = lean_ctor_get(v___x_1177_, 6);
v_auxDeclNGen_1185_ = lean_ctor_get(v___x_1177_, 7);
v_infoState_1186_ = lean_ctor_get(v___x_1177_, 8);
v_traceState_1187_ = lean_ctor_get(v___x_1177_, 9);
v_snapshotTasks_1188_ = lean_ctor_get(v___x_1177_, 10);
v_prevLinterStates_1189_ = lean_ctor_get(v___x_1177_, 11);
v_isSharedCheck_1202_ = !lean_is_exclusive(v___x_1177_);
if (v_isSharedCheck_1202_ == 0)
{
v___x_1191_ = v___x_1177_;
v_isShared_1192_ = v_isSharedCheck_1202_;
goto v_resetjp_1190_;
}
else
{
lean_inc(v_prevLinterStates_1189_);
lean_inc(v_snapshotTasks_1188_);
lean_inc(v_traceState_1187_);
lean_inc(v_infoState_1186_);
lean_inc(v_auxDeclNGen_1185_);
lean_inc(v_ngen_1184_);
lean_inc(v_maxRecDepth_1183_);
lean_inc(v_nextMacroScope_1182_);
lean_inc(v_usedQuotCtxts_1181_);
lean_inc(v_scopes_1180_);
lean_inc(v_messages_1179_);
lean_inc(v_env_1178_);
lean_dec(v___x_1177_);
v___x_1191_ = lean_box(0);
v_isShared_1192_ = v_isSharedCheck_1202_;
goto v_resetjp_1190_;
}
v_resetjp_1190_:
{
lean_object* v_asyncMode_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1197_; 
v_asyncMode_1193_ = lean_ctor_get(v_ext_1163_, 2);
lean_inc(v_asyncMode_1193_);
v___x_1194_ = lean_box(0);
v___x_1195_ = l_Lean_EnvExtension_modifyState___redArg(v_ext_1163_, v_env_1178_, v___f_1174_, v_asyncMode_1193_, v___x_1194_);
lean_dec(v_asyncMode_1193_);
if (v_isShared_1192_ == 0)
{
lean_ctor_set(v___x_1191_, 0, v___x_1195_);
v___x_1197_ = v___x_1191_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1201_; 
v_reuseFailAlloc_1201_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1201_, 0, v___x_1195_);
lean_ctor_set(v_reuseFailAlloc_1201_, 1, v_messages_1179_);
lean_ctor_set(v_reuseFailAlloc_1201_, 2, v_scopes_1180_);
lean_ctor_set(v_reuseFailAlloc_1201_, 3, v_usedQuotCtxts_1181_);
lean_ctor_set(v_reuseFailAlloc_1201_, 4, v_nextMacroScope_1182_);
lean_ctor_set(v_reuseFailAlloc_1201_, 5, v_maxRecDepth_1183_);
lean_ctor_set(v_reuseFailAlloc_1201_, 6, v_ngen_1184_);
lean_ctor_set(v_reuseFailAlloc_1201_, 7, v_auxDeclNGen_1185_);
lean_ctor_set(v_reuseFailAlloc_1201_, 8, v_infoState_1186_);
lean_ctor_set(v_reuseFailAlloc_1201_, 9, v_traceState_1187_);
lean_ctor_set(v_reuseFailAlloc_1201_, 10, v_snapshotTasks_1188_);
lean_ctor_set(v_reuseFailAlloc_1201_, 11, v_prevLinterStates_1189_);
v___x_1197_ = v_reuseFailAlloc_1201_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; 
v___x_1198_ = lean_st_ref_set(v___y_1176_, v___x_1197_);
v___x_1199_ = lean_box(0);
v___x_1200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1200_, 0, v___x_1199_);
return v___x_1200_;
}
}
}
v___jp_1203_:
{
lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; 
v___x_1206_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1, &lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1);
v___x_1207_ = l_Lean_stringToMessageData(v_tgt_1173_);
v___x_1208_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1208_, 0, v___x_1206_);
lean_ctor_set(v___x_1208_, 1, v___x_1207_);
v___x_1209_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3, &lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3);
v___x_1210_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1210_, 0, v___x_1208_);
lean_ctor_set(v___x_1210_, 1, v___x_1209_);
v___x_1211_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg(v_tgtId_1165_, v___x_1210_, v___y_1204_, v___y_1205_);
return v___x_1211_;
}
v___jp_1212_:
{
uint32_t v___x_1216_; uint8_t v___x_1217_; 
v___x_1216_ = 65;
v___x_1217_ = lean_uint32_dec_le(v___x_1216_, v___y_1215_);
if (v___x_1217_ == 0)
{
lean_dec_ref(v___f_1174_);
lean_dec_ref(v_ext_1163_);
v___y_1204_ = v___y_1213_;
v___y_1205_ = v___y_1214_;
goto v___jp_1203_;
}
else
{
uint32_t v___x_1218_; uint8_t v___x_1219_; 
v___x_1218_ = 90;
v___x_1219_ = lean_uint32_dec_le(v___y_1215_, v___x_1218_);
if (v___x_1219_ == 0)
{
lean_dec_ref(v___f_1174_);
lean_dec_ref(v_ext_1163_);
v___y_1204_ = v___y_1213_;
v___y_1205_ = v___y_1214_;
goto v___jp_1203_;
}
else
{
lean_dec_ref(v_tgt_1173_);
v___y_1176_ = v___y_1214_;
goto v___jp_1175_;
}
}
}
v___jp_1220_:
{
lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; 
v___x_1223_ = lean_unsigned_to_nat(0u);
v___x_1224_ = lean_string_utf8_byte_size(v_tgt_1173_);
lean_inc_ref(v_tgt_1173_);
v___x_1225_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1225_, 0, v_tgt_1173_);
lean_ctor_set(v___x_1225_, 1, v___x_1223_);
lean_ctor_set(v___x_1225_, 2, v___x_1224_);
v___x_1226_ = l_String_Slice_Pos_get_x3f(v___x_1225_, v___x_1223_);
lean_dec_ref_known(v___x_1225_, 3);
if (lean_obj_tag(v___x_1226_) == 0)
{
uint32_t v___x_1227_; 
v___x_1227_ = 65;
v___y_1213_ = v___y_1221_;
v___y_1214_ = v___y_1222_;
v___y_1215_ = v___x_1227_;
goto v___jp_1212_;
}
else
{
lean_object* v_val_1228_; uint32_t v___x_1229_; 
v_val_1228_ = lean_ctor_get(v___x_1226_, 0);
lean_inc(v_val_1228_);
lean_dec_ref_known(v___x_1226_, 1);
v___x_1229_ = lean_unbox_uint32(v_val_1228_);
lean_dec(v_val_1228_);
v___y_1213_ = v___y_1221_;
v___y_1214_ = v___y_1222_;
v___y_1215_ = v___x_1229_;
goto v___jp_1212_;
}
}
v___jp_1230_:
{
lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; 
v___x_1231_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1, &lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__1);
v___x_1232_ = l_Lean_stringToMessageData(v_src_1171_);
v___x_1233_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1233_, 0, v___x_1231_);
lean_ctor_set(v___x_1233_, 1, v___x_1232_);
v___x_1234_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3, &lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___closed__3);
v___x_1235_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1235_, 0, v___x_1233_);
lean_ctor_set(v___x_1235_, 1, v___x_1234_);
v___x_1236_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg(v_srcId_1164_, v___x_1235_, v_a_1166_, v_a_1167_);
return v___x_1236_;
}
v___jp_1237_:
{
uint32_t v___x_1239_; uint8_t v___x_1240_; 
v___x_1239_ = 65;
v___x_1240_ = lean_uint32_dec_le(v___x_1239_, v___y_1238_);
if (v___x_1240_ == 0)
{
lean_dec_ref(v___f_1174_);
lean_dec_ref(v_tgt_1173_);
lean_dec_ref(v_ext_1163_);
goto v___jp_1230_;
}
else
{
uint32_t v___x_1241_; uint8_t v___x_1242_; 
v___x_1241_ = 90;
v___x_1242_ = lean_uint32_dec_le(v___y_1238_, v___x_1241_);
if (v___x_1242_ == 0)
{
lean_dec_ref(v___f_1174_);
lean_dec_ref(v_tgt_1173_);
lean_dec_ref(v_ext_1163_);
goto v___jp_1230_;
}
else
{
lean_dec_ref(v_src_1171_);
v___y_1221_ = v_a_1166_;
v___y_1222_ = v_a_1167_;
goto v___jp_1220_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation___boxed(lean_object* v_ext_1250_, lean_object* v_srcId_1251_, lean_object* v_tgtId_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_, lean_object* v_a_1255_){
_start:
{
lean_object* v_res_1256_; 
v_res_1256_ = lp_mathlib_Mathlib_Tactic_GuessName_GuessNameExt_addTranslation(v_ext_1250_, v_srcId_1251_, v_tgtId_1252_, v_a_1253_, v_a_1254_);
lean_dec(v_a_1254_);
lean_dec_ref(v_a_1253_);
lean_dec(v_tgtId_1252_);
lean_dec(v_srcId_1251_);
return v_res_1256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0(lean_object* v_00_u03b2_1257_, lean_object* v_m_1258_, lean_object* v_a_1259_, lean_object* v_b_1260_){
_start:
{
lean_object* v___x_1261_; 
v___x_1261_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0___redArg(v_m_1258_, v_a_1259_, v_b_1260_);
return v___x_1261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1(lean_object* v_00_u03b1_1262_, lean_object* v_ref_1263_, lean_object* v_msg_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_){
_start:
{
lean_object* v___x_1268_; 
v___x_1268_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___redArg(v_ref_1263_, v_msg_1264_, v___y_1265_, v___y_1266_);
return v___x_1268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1___boxed(lean_object* v_00_u03b1_1269_, lean_object* v_ref_1270_, lean_object* v_msg_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_){
_start:
{
lean_object* v_res_1275_; 
v_res_1275_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1(v_00_u03b1_1269_, v_ref_1270_, v_msg_1271_, v___y_1272_, v___y_1273_);
lean_dec(v___y_1273_);
lean_dec_ref(v___y_1272_);
lean_dec(v_ref_1270_);
return v_res_1275_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0(lean_object* v_00_u03b2_1276_, lean_object* v_a_1277_, lean_object* v_x_1278_){
_start:
{
uint8_t v___x_1279_; 
v___x_1279_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___redArg(v_a_1277_, v_x_1278_);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1280_, lean_object* v_a_1281_, lean_object* v_x_1282_){
_start:
{
uint8_t v_res_1283_; lean_object* v_r_1284_; 
v_res_1283_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__0(v_00_u03b2_1280_, v_a_1281_, v_x_1282_);
lean_dec(v_x_1282_);
lean_dec_ref(v_a_1281_);
v_r_1284_ = lean_box(v_res_1283_);
return v_r_1284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1(lean_object* v_00_u03b2_1285_, lean_object* v_data_1286_){
_start:
{
lean_object* v___x_1287_; 
v___x_1287_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1___redArg(v_data_1286_);
return v___x_1287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__2(lean_object* v_00_u03b2_1288_, lean_object* v_a_1289_, lean_object* v_b_1290_, lean_object* v_x_1291_){
_start:
{
lean_object* v___x_1292_; 
v___x_1292_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__2___redArg(v_a_1289_, v_b_1290_, v_x_1291_);
return v___x_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6(lean_object* v_msgData_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_){
_start:
{
lean_object* v___x_1297_; 
v___x_1297_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___redArg(v_msgData_1293_, v___y_1295_);
return v___x_1297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6___boxed(lean_object* v_msgData_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_){
_start:
{
lean_object* v_res_1302_; 
v_res_1302_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__6(v_msgData_1298_, v___y_1299_, v___y_1300_);
lean_dec(v___y_1300_);
lean_dec_ref(v___y_1299_);
return v_res_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4(lean_object* v_00_u03b1_1303_, lean_object* v_msg_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_){
_start:
{
lean_object* v___x_1308_; 
v___x_1308_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___redArg(v_msg_1304_, v___y_1305_, v___y_1306_);
return v___x_1308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4___boxed(lean_object* v_00_u03b1_1309_, lean_object* v_msg_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_){
_start:
{
lean_object* v_res_1314_; 
v_res_1314_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4(v_00_u03b1_1309_, v_msg_1310_, v___y_1311_, v___y_1312_);
lean_dec(v___y_1312_);
lean_dec_ref(v___y_1311_);
return v_res_1314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_1315_, lean_object* v_i_1316_, lean_object* v_source_1317_, lean_object* v_target_1318_){
_start:
{
lean_object* v___x_1319_; 
v___x_1319_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2___redArg(v_i_1316_, v_source_1317_, v_target_1318_);
return v___x_1319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7(lean_object* v_msgData_1320_, lean_object* v_macroStack_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_){
_start:
{
lean_object* v___x_1325_; 
v___x_1325_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___redArg(v_msgData_1320_, v_macroStack_1321_, v___y_1323_);
return v___x_1325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7___boxed(lean_object* v_msgData_1326_, lean_object* v_macroStack_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_){
_start:
{
lean_object* v_res_1331_; 
v_res_1331_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__1_spec__4_spec__7(v_msgData_1326_, v_macroStack_1327_, v___y_1328_, v___y_1329_);
lean_dec(v___y_1329_);
lean_dec_ref(v___y_1328_);
return v_res_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1332_, lean_object* v_x_1333_, lean_object* v_x_1334_){
_start:
{
lean_object* v___x_1335_; 
v___x_1335_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_GuessName_GuessNameExt_addTranslation_spec__0_spec__1_spec__2_spec__4___redArg(v_x_1333_, v_x_1334_);
return v___x_1335_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_GuessName(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_TreeMap_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_String_Defs(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Translate_GuessName(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_TreeMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_String_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default = _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData_default);
lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData = _init_lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_GuessName_instInhabitedGuessNameData);
lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames = _init_lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_GuessName_endCapitalNames);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Std_Data_TreeMap_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_String_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Translate_GuessName(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_TreeMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_String_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_GuessName(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Translate_GuessName(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Translate_GuessName(builtin);
}
#ifdef __cplusplus
}
#endif
